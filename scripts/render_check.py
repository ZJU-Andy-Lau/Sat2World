"""Sat2World 精简版跨视角渲染检查脚本。

仅保留：
1) 最新 RPC2Pinhole 相机拟合逻辑；
2) 跨视角渲染测试主链路；
3) 清晰的过程日志与核心指标输出。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataset, rpc_scene_collate_fn
from render.rpc_gaussian_renderer import RPCGaussianRenderer, RPCGaussianRendererCfg, VirtualPinholeCamera
from render.rpc2pinhole_camera_fit import RPC2PinholeFitCfg, fit_view_pinhole_from_rpc2, format_rpc2_fit_diag


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


class ProgressLogger:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.step = 0

    def log(self, msg: str) -> None:
        self.step += 1
        now = time.perf_counter()
        print(f"[render_check][step={self.step:02d}] {msg} | step={now - self.t_last:.2f}s total={now - self.t0:.2f}s", flush=True)
        self.t_last = now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World simplified cross-view render checker")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--view-i", type=int, default=0, help="source view")
    p.add_argument("--view-j", type=int, default=1, help="target view")

    # rpc2pinhole 拟合参数
    p.add_argument("--rpc2-downsample", type=int, default=8)
    p.add_argument("--rpc2-num-heights", type=int, default=100)
    p.add_argument("--rpc2-height-low", type=float, default=-20.0)
    p.add_argument("--rpc2-height-high", type=float, default=80.0)
    p.add_argument("--rpc2-focal", type=float, default=1e5)
    p.add_argument("--rpc2-disable-ransac", action="store_true")
    p.add_argument("--rpc2-seed", type=int, default=42)

    # 核心阈值
    p.add_argument("--fit-p95-threshold", type=float, default=3.0)
    p.add_argument("--fit-max-threshold", type=float, default=10.0)
    p.add_argument("--reproj-p95-threshold", type=float, default=3.0)
    p.add_argument("--render-time-threshold-sec", type=float, default=2.0)
    p.add_argument("--rgb-l1-threshold", type=float, default=0.35)
    p.add_argument("--alpha-max-min-threshold", type=float, default=1e-3)
    p.add_argument("--alpha-coverage-min-threshold", type=float, default=1e-4)

    p.add_argument("--save-images", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_val_batch(cfg: dict[str, Any], scene_index: int) -> dict[str, Any]:
    ds = build_dataset(mode="val", **cfg.get("data", {}).get("val", {}))
    if len(ds) == 0:
        raise RuntimeError("val dataset empty")
    if not (0 <= scene_index < len(ds)):
        raise ValueError(f"scene_index out of range: {scene_index}/{len(ds)}")
    return rpc_scene_collate_fn([ds[scene_index]])


def to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from engine.distributed import move_batch_to_device

    return move_batch_to_device(batch, device)


def make_renderer(cfg: dict[str, Any]) -> RPCGaussianRenderer:
    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)

    # 仅保留跨视角渲染测试必要配置
    rcfg.source_stride = 1
    rcfg.confidence_threshold = 0.0
    rcfg.topk_per_target = None
    rcfg.exclude_self_source = False
    rcfg.val_num_target_views = max(int(getattr(rcfg, "val_num_target_views", 1)), 1)
    rcfg.use_all_targets_in_val = True
    rcfg.render_downsample_factor_val = 1

    from geometry import RPCGeometryOps

    return RPCGaussianRenderer(RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32), rcfg)


def _make_rpc2_cfg(args: argparse.Namespace) -> RPC2PinholeFitCfg:
    return RPC2PinholeFitCfg(
        downsample=int(args.rpc2_downsample),
        num_heights=int(args.rpc2_num_heights),
        height_low=float(args.rpc2_height_low),
        height_high=float(args.rpc2_height_high),
        focal=float(args.rpc2_focal),
        use_ransac=not bool(args.rpc2_disable_ransac),
        seed=int(args.rpc2_seed),
    )


def install_rpc2_camera_fitter(renderer: RPCGaussianRenderer, fit_cfg: RPC2PinholeFitCfg) -> None:
    cache: dict[tuple[int, int, int, int], VirtualPinholeCamera] = {}

    def _fit_virtual_camera_override(self: RPCGaussianRenderer, batch: dict[str, Any], bi: int, tv: int, image_hw: tuple[int, int]) -> VirtualPinholeCamera:
        sid = int(batch["scene_id"][bi].item()) if torch.is_tensor(batch.get("scene_id", None)) else int(bi)
        h, w = int(image_hw[0]), int(image_hw[1])
        key = (sid, int(tv), h, w)
        if key not in cache:
            cache[key] = fit_view_pinhole_from_rpc2(batch=batch, bi=int(bi), tv=int(tv), image_hw=(h, w), cfg=fit_cfg)
        return cache[key]

    def _fit_virtual_camera_for_target_override(
        self: RPCGaussianRenderer,
        batch: dict[str, Any],
        batch_index: int,
        target_view_index: int,
        image_hw: tuple[int, int] | None = None,
    ) -> VirtualPinholeCamera:
        if image_hw is None:
            image_hw = (int(batch["images"].shape[-2]), int(batch["images"].shape[-1]))
        return self._fit_virtual_camera(batch, int(batch_index), int(target_view_index), image_hw)

    renderer._fit_virtual_camera = types.MethodType(_fit_virtual_camera_override, renderer)  # type: ignore[method-assign]
    renderer.fit_virtual_camera_for_target = types.MethodType(_fit_virtual_camera_for_target_override, renderer)  # type: ignore[method-assign]


def _safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def build_dense_anchor_outputs(batch: dict[str, Any], vi: int, sh_dim: int = 48) -> dict[str, Any]:
    """使用 source view_i 的全像素锚点构造可渲染高斯输出。"""
    b, v, _, h, w = batch["height_gt"].shape
    dev = batch["images"].device

    centers_rpc = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)
    centers_point = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)

    rpc_i = batch["rpc_gt"][0][vi]
    scene_center = batch["scene_xy_center"][0].to(dtype=torch.double, device=rpc_i.device)
    scene_scale = torch.ones_like(scene_center)

    lines = torch.arange(h, device=rpc_i.device, dtype=torch.double).view(-1, 1).expand(h, w).reshape(-1)
    samps = torch.arange(w, device=rpc_i.device, dtype=torch.double).view(1, -1).expand(h, w).reshape(-1)
    hs = batch["height_gt"][0, vi, 0].to(dtype=torch.double, device=rpc_i.device).reshape(-1)

    x, y = rpc_i.RPC_LINESAMP2XY(
        line_in=lines,
        samp_in=samps,
        h_in=hs,
        output_type="tensor",
        xy_center=scene_center,
        xy_scale=scene_scale,
    )

    centers_rpc[0, vi, 0] = x.view(h, w).to(torch.float32, non_blocking=True)
    centers_rpc[0, vi, 1] = y.view(h, w).to(torch.float32, non_blocking=True)
    centers_rpc[0, vi, 2] = hs.view(h, w).to(torch.float32, non_blocking=True)
    centers_point.copy_(centers_rpc)

    opacity = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    opacity[0, vi] = 1.0

    scale = torch.full((b, v, 3, h, w), 0.5, device=dev, dtype=torch.float32)

    rotation = torch.zeros((b, v, 4, h, w), device=dev, dtype=torch.float32)
    rotation[:, :, 0] = 1.0

    sh = torch.zeros((b, v, sh_dim, h, w), device=dev, dtype=torch.float32)
    sh[0, vi, 0:3] = _safe_logit(batch["images"][0, vi].clamp(0.0, 1.0))

    conf = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    conf[0, vi] = 1.0

    return {
        "gaussian_centers_rpc": centers_rpc,
        "gaussian_centers_point": centers_point,
        "gaussian_opacity": opacity,
        "gaussian_scale": scale,
        "gaussian_rotation": rotation,
        "gaussian_sh": sh,
        "gaussian_confidence_rpc": conf,
        "gaussian_confidence_point": conf,
        "rpc_corrected": batch["rpc_gt"],
    }


def project_source_to_target_rpc(batch: dict[str, Any], vi: int, vj: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 RPC 建立 source(view_i) -> target(view_j) 的几何期望投影。"""
    h, w = batch["height_gt"].shape[-2:]
    rpc_i = batch["rpc_gt"][0][vi]
    rpc_j = batch["rpc_gt"][0][vj]
    dev = rpc_i.device

    scene_center_i = batch["scene_xy_center"][0].to(dtype=torch.double, device=dev)
    scene_scale_i = torch.ones_like(scene_center_i)
    scene_center_j = batch["scene_xy_center"][0].to(dtype=torch.double, device=rpc_j.device)
    scene_scale_j = torch.ones_like(scene_center_j)

    lines = torch.arange(h, device=dev, dtype=torch.double).view(-1, 1).expand(h, w).reshape(-1)
    samps = torch.arange(w, device=dev, dtype=torch.double).view(1, -1).expand(h, w).reshape(-1)
    hs = batch["height_gt"][0, vi, 0].to(dtype=torch.double, device=dev).reshape(-1)

    x, y = rpc_i.RPC_LINESAMP2XY(
        line_in=lines,
        samp_in=samps,
        h_in=hs,
        output_type="tensor",
        xy_center=scene_center_i,
        xy_scale=scene_scale_i,
    )
    line_j, samp_j = rpc_j.RPC_XY2LINESAMP(
        x_in=x.to(device=rpc_j.device),
        y_in=y.to(device=rpc_j.device),
        h_in=hs.to(device=rpc_j.device),
        output_type="tensor",
        xy_center=scene_center_j,
        xy_scale=scene_scale_j,
    )
    xyz = torch.stack([x, y, hs], dim=-1).detach().cpu().numpy().astype(np.float64)
    return (
        line_j.detach().cpu().numpy().astype(np.float64),
        samp_j.detach().cpu().numpy().astype(np.float64),
        xyz,
    )


def project_with_camera_np(cam: VirtualPinholeCamera, xyz_world: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_hw
    n = xyz_world.shape[0]
    xyz_h = np.concatenate([xyz_world.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    K = cam.K.detach().cpu().numpy().astype(np.float64)
    w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)
    cam_xyz = (w2c[:3, :] @ xyz_h.T).T
    z = cam_xyz[:, 2]
    proj = (K @ cam_xyz.T).T
    valid = np.isfinite(z) & (z > 1e-8) & np.isfinite(proj).all(axis=1)
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    uv[valid] = proj[valid, :2] / proj[valid, 2:3]
    in_frame = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, z, in_frame


def reproj_metrics(gt_line: np.ndarray, gt_samp: np.ndarray, pd_uv: np.ndarray) -> dict[str, float]:
    gt_uv = np.stack([gt_samp, gt_line], axis=-1)
    valid = np.isfinite(gt_uv).all(axis=1) & np.isfinite(pd_uv).all(axis=1)
    if not valid.any():
        return {"count": 0, "p50": float("inf"), "p95": float("inf"), "max": float("inf")}
    err = np.linalg.norm(pd_uv[valid] - gt_uv[valid], axis=1)
    return {
        "count": int(err.shape[0]),
        "p50": float(np.quantile(err, 0.50)),
        "p95": float(np.quantile(err, 0.95)),
        "max": float(np.max(err)),
    }


def save_chw_rgb(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def save_chw_gray(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().squeeze(0).clamp(0.0, 1.0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_cfg(args.config)
    prog = ProgressLogger()

    prog.log("build batch")
    batch = to_device_batch(build_val_batch(cfg, args.scene_index), device)
    b, v, _, h, w = batch["images"].shape
    if b != 1:
        raise RuntimeError("render_check currently supports batch_size=1")
    if not (0 <= args.view_i < v and 0 <= args.view_j < v):
        raise ValueError(f"invalid view ids: i={args.view_i}, j={args.view_j}, V={v}")
    if args.view_i == args.view_j:
        raise ValueError("view_i and view_j must be different for cross-view render check")

    prog.log("build renderer")
    renderer = make_renderer(cfg)

    prog.log("install rpc2 camera fitter")
    rpc2_cfg = _make_rpc2_cfg(args)
    install_rpc2_camera_fitter(renderer, rpc2_cfg)

    prog.log("fit target virtual camera (rpc2pinhole)")
    cam = renderer.fit_virtual_camera_for_target(batch, 0, args.view_j)
    print("\n-- rpc2pinhole fit diagnostics --")
    print(format_rpc2_fit_diag(cam))

    checks: list[CheckResult] = [
        CheckResult("fit_p95_px", cam.fit_p95, args.fit_p95_threshold, cam.fit_p95 <= args.fit_p95_threshold),
        CheckResult("fit_max_px", cam.fit_max, args.fit_max_threshold, cam.fit_max <= args.fit_max_threshold),
    ]

    prog.log("prepare source-view dense gaussian outputs")
    outputs = build_dense_anchor_outputs(batch, args.view_i)

    prog.log("render target view")
    t0 = time.perf_counter()
    render_out = renderer.render_paths(outputs, batch, mode="val", global_step=0, epoch=0)
    dt = time.perf_counter() - t0
    checks.append(CheckResult("render_time_sec", dt, args.render_time_threshold_sec, dt <= args.render_time_threshold_sec))

    rpc_path = render_out["rpc"]
    tv = rpc_path["target_view_indices"].detach().cpu()
    match = (tv == int(args.view_j)).nonzero(as_tuple=False).view(-1)
    if match.numel() == 0:
        raise RuntimeError(f"target view_j={args.view_j} not rendered")
    ridx = int(match[0].item())

    rendered_rgb = rpc_path["rendered_rgb"][ridx]
    rendered_alpha = rpc_path["rendered_alpha"][ridx]
    target_rgb = rpc_path["target_rgb"][ridx]

    prog.log("compute cross-view projection metrics")
    gt_line, gt_samp, xyz_world = project_source_to_target_rpc(batch, args.view_i, args.view_j)
    pd_uv, z_cam, in_frame = project_with_camera_np(cam, xyz_world, image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])))
    reproj = reproj_metrics(gt_line, gt_samp, pd_uv)

    rgb_l1 = float((rendered_rgb - target_rgb).abs().mean().item())
    alpha_max = float(rendered_alpha.max().item())
    alpha_cov = float((rendered_alpha > 1e-4).to(torch.float32).mean().item())
    pos_depth_ratio = float(np.mean(z_cam > 1e-8))
    in_frame_ratio = float(np.mean(in_frame))

    checks.extend(
        [
            CheckResult("cross_reproj_p95_px", reproj["p95"], args.reproj_p95_threshold, reproj["p95"] <= args.reproj_p95_threshold),
            CheckResult("rgb_l1", rgb_l1, args.rgb_l1_threshold, rgb_l1 <= args.rgb_l1_threshold),
            CheckResult("alpha_max", alpha_max, args.alpha_max_min_threshold, alpha_max >= args.alpha_max_min_threshold),
            CheckResult("alpha_coverage", alpha_cov, args.alpha_coverage_min_threshold, alpha_cov >= args.alpha_coverage_min_threshold),
        ]
    )

    work_dir = Path(cfg.get("system", {}).get("work_dir", "work_dirs/sat2world_default"))
    out_dir = work_dir / "render_check" / f"scene_{int(batch['scene_id'][0].item())}" / f"vi_{args.view_i}_vj_{args.view_j}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_images:
        save_chw_rgb(out_dir / "rendered_rgb.png", rendered_rgb)
        save_chw_rgb(out_dir / "target_rgb.png", target_rgb)
        save_chw_gray(out_dir / "rendered_alpha.png", rendered_alpha)

    report = {
        "scene_id": int(batch["scene_id"][0].item()),
        "view_i": int(args.view_i),
        "view_j": int(args.view_j),
        "fit": {
            "p50": float(cam.fit_p50),
            "p95": float(cam.fit_p95),
            "max": float(cam.fit_max),
            "diagnostics": cam.diagnostics,
        },
        "render": {
            "time_sec": float(dt),
            "rgb_l1": rgb_l1,
            "alpha_max": alpha_max,
            "alpha_coverage": alpha_cov,
        },
        "cross_view_projection": {
            "reproj": reproj,
            "positive_depth_ratio": pos_depth_ratio,
            "in_frame_ratio": in_frame_ratio,
        },
        "checks": [
            {
                "name": c.name,
                "value": float(c.value),
                "threshold": float(c.threshold),
                "passed": bool(c.passed),
                "note": c.note,
            }
            for c in checks
        ],
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nscene_index={args.scene_index} view_i={args.view_i} view_j={args.view_j}")
    print(f"fit: p50={cam.fit_p50:.4f}px p95={cam.fit_p95:.4f}px max={cam.fit_max:.4f}px")
    print(
        f"cross-proj: p50={reproj['p50']:.4f}px p95={reproj['p95']:.4f}px max={reproj['max']:.4f}px "
        f"pos_depth_ratio={pos_depth_ratio:.4f} in_frame_ratio={in_frame_ratio:.4f}"
    )
    print(f"render: time={dt:.4f}s rgb_l1={rgb_l1:.6f} alpha_max={alpha_max:.6f} alpha_cov={alpha_cov:.6f}")

    n_pass = sum(1 for c in checks if c.passed)
    print(f"checks: {n_pass}/{len(checks)} passed")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        note = f" ({c.note})" if c.note else ""
        print(f"[{status}] {c.name}: value={c.value:.6f} thr={c.threshold:.6f}{note}")

    print(f"report saved: {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()

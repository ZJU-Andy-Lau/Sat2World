"""单视图 RPC->pinhole->gsplat 渲染探针脚本。

目标：
1) 从 val 数据集中读取单个 scene，仅使用一个 view。
2) 拟合该 view 的虚拟针孔相机。
3) 在该 view 覆盖区域的局部米制坐标中随机生成少量高斯。
4) 直接调用 gsplat 渲染（最小链路），并进行多约定探针。
5) 在关键节点输出诊断指标，定位渲染失败来源。
6) 输出图片和 JSON 到 work_dir。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataset, rpc_scene_collate_fn
from render.rpc_gaussian_renderer import RPCGaussianRenderer, RPCGaussianRendererCfg, VirtualPinholeCamera


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World single-view render probe")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--view-k", type=int, default=0)
    p.add_argument("--num-gaussians", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--xy-span-ratio", type=float, default=0.6, help="xy 采样范围比例，基于 scene_xy_scale")
    p.add_argument("--z-quantile-low", type=float, default=0.10)
    p.add_argument("--z-quantile-high", type=float, default=0.90)
    p.add_argument("--opacity", type=float, default=0.95)
    p.add_argument("--scale-min", type=float, default=0.3)
    p.add_argument("--scale-max", type=float, default=1.2)
    p.add_argument("--alpha-max-threshold", type=float, default=1e-3)
    p.add_argument("--alpha-coverage-threshold", type=float, default=1e-4)
    p.add_argument("--fit-p95-threshold", type=float, default=1.0)
    p.add_argument("--reproj-p95-threshold", type=float, default=1.0)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--dump-json", action="store_true")
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
    sample = ds[scene_index]
    return rpc_scene_collate_fn([sample])


def to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from engine.distributed import move_batch_to_device

    return move_batch_to_device(batch, device)


def make_renderer(cfg: dict[str, Any]) -> RPCGaussianRenderer:
    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)

    # 保持与 render_check 一致的高精度拟合设置
    rcfg.fit_grid_nx = max(int(getattr(rcfg, "fit_grid_nx", 24)), 24)
    rcfg.fit_grid_ny = max(int(getattr(rcfg, "fit_grid_ny", 24)), 24)
    rcfg.fit_grid_nz = max(int(getattr(rcfg, "fit_grid_nz", 7)), 7)
    rcfg.render_downsample_factor_val = 1

    from geometry import RPCGeometryOps

    return RPCGaussianRenderer(RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32), rcfg)


def _save_chw_rgb(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def _save_chw_gray(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().squeeze(0).clamp(0.0, 1.0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def _sample_gaussians(
    batch: dict[str, Any],
    view_k: int,
    n: int,
    seed: int,
    xy_span_ratio: float,
    z_q_low: float,
    z_q_high: float,
    opacity: float,
    scale_min: float,
    scale_max: float,
) -> dict[str, torch.Tensor]:
    dev = batch["images"].device
    scene_scale = batch["scene_xy_scale"][0].to(torch.float32)
    sy = float(scene_scale[0].abs().clamp_min(1e-6).item())
    sx = float(scene_scale[1].abs().clamp_min(1e-6).item())

    hgt = batch["height_gt"][0, view_k, 0].to(torch.float32)
    m = batch.get("height_valid_mask", None)
    if m is not None:
        valid = m[0, view_k, 0] > 0.5
        hs = hgt[valid] if bool(valid.any()) else hgt.reshape(-1)
    else:
        hs = hgt.reshape(-1)

    z_low = float(torch.quantile(hs, float(z_q_low)).item())
    z_high = float(torch.quantile(hs, float(z_q_high)).item())
    if not np.isfinite(z_low) or not np.isfinite(z_high) or z_high <= z_low:
        z_ref = float(batch["height_ref"][0, view_k].item())
        z_low, z_high = z_ref - 10.0, z_ref + 10.0

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    xr = (torch.rand((n,), generator=g) * 2.0 - 1.0) * (sx * float(xy_span_ratio))
    yr = (torch.rand((n,), generator=g) * 2.0 - 1.0) * (sy * float(xy_span_ratio))
    zr = torch.rand((n,), generator=g) * (z_high - z_low) + z_low
    centers = torch.stack([xr, yr, zr], dim=-1).to(device=dev, dtype=torch.float32)

    s = torch.rand((n, 3), generator=g) * (float(scale_max) - float(scale_min)) + float(scale_min)
    scale = s.to(device=dev, dtype=torch.float32)

    rotation = torch.zeros((n, 4), device=dev, dtype=torch.float32)
    rotation[:, 0] = 1.0

    op = torch.full((n, 1), float(opacity), device=dev, dtype=torch.float32)

    # 稳定可见的颜色（避免太暗）
    palette = torch.tensor(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.2, 1.0],
            [1.0, 1.0, 0.2],
            [1.0, 0.2, 1.0],
            [0.2, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=dev,
    )
    color_idx = torch.arange(n, device=dev) % palette.shape[0]
    rgb = palette[color_idx]

    return {
        "centers": centers,
        "scale": scale,
        "rotation": rotation,
        "opacity": op,
        "rgb": rgb,
        "z_low": torch.tensor(z_low),
        "z_high": torch.tensor(z_high),
        "sx": torch.tensor(sx),
        "sy": torch.tensor(sy),
    }


def _rpc_project_to_view(batch: dict[str, Any], view_k: int, xyz: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rpc = batch["rpc_gt"][0][view_k]
    scene_center = batch["scene_xy_center"][0].to(dtype=torch.double, device=rpc.device)
    scene_scale = torch.ones_like(scene_center)

    x = xyz[:, 0].to(dtype=torch.double, device=rpc.device)
    y = xyz[:, 1].to(dtype=torch.double, device=rpc.device)
    z = xyz[:, 2].to(dtype=torch.double, device=rpc.device)
    line, samp = rpc.RPC_XY2LINESAMP(
        x_in=x,
        y_in=y,
        h_in=z,
        output_type="tensor",
        xy_center=scene_center,
        xy_scale=scene_scale,
    )
    return (
        line.detach().cpu().numpy().astype(np.float64),
        samp.detach().cpu().numpy().astype(np.float64),
        z.detach().cpu().numpy().astype(np.float64),
    )


def _pinhole_project(cam: VirtualPinholeCamera, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    K = cam.K.detach().cpu().numpy().astype(np.float64)
    w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)
    cam_xyz = (w2c[:3, :] @ xyz_h.T).T
    z = cam_xyz[:, 2]
    proj = (K @ cam_xyz.T).T
    valid = np.isfinite(z) & (z > 1e-8) & np.isfinite(proj).all(axis=1)
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    uv[valid] = proj[valid, :2] / proj[valid, 2:3]
    return uv[:, 1], uv[:, 0], z  # line, samp, depth


def _reproj_metrics(gt_line: np.ndarray, gt_samp: np.ndarray, pd_line: np.ndarray, pd_samp: np.ndarray, h: int, w: int) -> dict[str, float]:
    gt_uv = np.stack([gt_samp, gt_line], axis=-1)
    pd_uv = np.stack([pd_samp, pd_line], axis=-1)
    valid = np.isfinite(gt_uv).all(axis=1) & np.isfinite(pd_uv).all(axis=1)

    if not valid.any():
        return {
            "reproj_p50_px": float("inf"),
            "reproj_p95_px": float("inf"),
            "reproj_max_px": float("inf"),
            "in_frame_ratio": 0.0,
        }

    err = np.linalg.norm(pd_uv[valid] - gt_uv[valid], axis=1)
    in_frame = (
        (pd_uv[valid, 0] >= 0)
        & (pd_uv[valid, 0] < w)
        & (pd_uv[valid, 1] >= 0)
        & (pd_uv[valid, 1] < h)
    )
    return {
        "reproj_p50_px": float(np.quantile(err, 0.50)),
        "reproj_p95_px": float(np.quantile(err, 0.95)),
        "reproj_max_px": float(err.max()),
        "in_frame_ratio": float(in_frame.mean()),
    }


def _camera_matrix_diagnostics(cam: VirtualPinholeCamera, xyz: np.ndarray) -> dict[str, float]:
    n = xyz.shape[0]
    xyz_h = np.concatenate([xyz.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)
    R = w2c[:3, :3]
    cam_xyz = (w2c[:3, :] @ xyz_h.T).T
    z = cam_xyz[:, 2]
    return {
        "det_R": float(np.linalg.det(R)),
        "orthogonality_fro": float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro")),
        "positive_depth_ratio": float((z > 1e-6).mean()),
    }


def _render_variant(
    renderer: RPCGaussianRenderer,
    centers: torch.Tensor,
    opacity: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    rgb: torch.Tensor,
    cam: VirtualPinholeCamera,
    image_hw: tuple[int, int],
    w2c_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_v = VirtualPinholeCamera(
        K=cam.K,
        w2c=cam.w2c if w2c_override is None else w2c_override,
        fit_p50=cam.fit_p50,
        fit_p95=cam.fit_p95,
        fit_max=cam.fit_max,
    )
    return renderer._render_cuda(
        centers=centers,
        opacity=opacity,
        scale=scale,
        rotation=rotation,
        rgb=rgb,
        cam=cam_v,
        image_hw=image_hw,
    )


def _alpha_stats(alpha: torch.Tensor) -> dict[str, float]:
    ra = alpha.detach().cpu().squeeze(0).numpy().astype(np.float32)
    alpha_max = float(np.maximum(ra, 0.0).max())
    alpha_thr = max(0.01, alpha_max * 0.1)
    alpha_cov = float((ra > alpha_thr).mean())
    return {
        "alpha_max": alpha_max,
        "alpha_thr": float(alpha_thr),
        "alpha_coverage": alpha_cov,
    }


def _draw_points_overlay(base_chw: torch.Tensor, line: np.ndarray, samp: np.ndarray, out_path: Path, color: tuple[int, int, int]) -> None:
    arr = (base_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    h, w = arr.shape[0], arr.shape[1]
    for l, s in zip(line, samp):
        if np.isfinite(l) and np.isfinite(s):
            x = float(s)
            y = float(l)
            if 0 <= x < w and 0 <= y < h:
                r = 2
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_cfg(args.config)

    batch = to_device_batch(build_val_batch(cfg, args.scene_index), device)
    b, v, _, h, w = batch["images"].shape
    if b != 1:
        raise RuntimeError("single-view probe supports batch_size=1 only")
    if not (0 <= args.view_k < v):
        raise ValueError(f"invalid view-k={args.view_k}, V={v}")

    renderer = make_renderer(cfg)
    cam = renderer.fit_virtual_camera_for_target(batch, 0, args.view_k)

    # 1) 随机高斯
    gs = _sample_gaussians(
        batch=batch,
        view_k=args.view_k,
        n=max(int(args.num_gaussians), 1),
        seed=int(args.seed),
        xy_span_ratio=float(args.xy_span_ratio),
        z_q_low=float(args.z_quantile_low),
        z_q_high=float(args.z_quantile_high),
        opacity=float(args.opacity),
        scale_min=float(args.scale_min),
        scale_max=float(args.scale_max),
    )

    centers = gs["centers"]
    scale = gs["scale"]
    rotation = gs["rotation"]
    opacity = gs["opacity"]
    rgb = gs["rgb"]

    # 2) RPC 几何投影（基准）
    rpc_line, rpc_samp, _rpc_h = _rpc_project_to_view(batch, args.view_k, centers)
    in_frame_rpc = (
        np.isfinite(rpc_line)
        & np.isfinite(rpc_samp)
        & (rpc_line >= 0)
        & (rpc_line < h)
        & (rpc_samp >= 0)
        & (rpc_samp < w)
    )

    # 3) Pinhole 投影与重投影误差
    pd_line, pd_samp, pd_depth = _pinhole_project(cam, centers.detach().cpu().numpy())
    reproj = _reproj_metrics(rpc_line, rpc_samp, pd_line, pd_samp, h=h, w=w)
    cam_diag = _camera_matrix_diagnostics(cam, centers.detach().cpu().numpy())

    # 4) gsplat 约定探针
    dev = centers.device
    I4 = torch.eye(4, dtype=torch.float32, device=dev)
    Fz = I4.clone()
    Fz[2, 2] = -1.0
    Fy = I4.clone()
    Fy[1, 1] = -1.0
    Fzy = Fy @ Fz

    w2c_curr = cam.w2c.to(dtype=torch.float32, device=dev)
    variants: dict[str, torch.Tensor] = {
        "current": w2c_curr,
        "cam_z_flip": Fz @ w2c_curr,
        "cam_y_flip": Fy @ w2c_curr,
        "cam_zy_flip": Fzy @ w2c_curr,
        "w2c_transposed": w2c_curr.transpose(0, 1).contiguous(),
    }

    render_by_variant: dict[str, dict[str, Any]] = {}
    for name, w2c_var in variants.items():
        rr, ra, rd = _render_variant(
            renderer=renderer,
            centers=centers,
            opacity=opacity,
            scale=scale,
            rotation=rotation,
            rgb=rgb,
            cam=cam,
            image_hw=(h, w),
            w2c_override=w2c_var,
        )
        st = _alpha_stats(ra)
        render_by_variant[name] = {
            "rendered_rgb": rr,
            "rendered_alpha": ra,
            "rendered_depth": rd,
            "alpha_max": st["alpha_max"],
            "alpha_thr": st["alpha_thr"],
            "alpha_coverage": st["alpha_coverage"],
        }

    best_variant = max(render_by_variant.items(), key=lambda kv: kv[1]["alpha_max"])[0]
    current_alpha_max = float(render_by_variant["current"]["alpha_max"])
    best_alpha_max = float(render_by_variant[best_variant]["alpha_max"])
    conv_mismatch_suspected = (
        current_alpha_max < float(args.alpha_max_threshold)
        and best_alpha_max >= float(args.alpha_max_threshold)
        and best_variant != "current"
    )

    # 5) 关键检查
    checks: list[CheckResult] = [
        CheckResult("fit_p95_px", float(cam.fit_p95), float(args.fit_p95_threshold), float(cam.fit_p95) <= float(args.fit_p95_threshold)),
        CheckResult("rpc_in_frame_ratio", float(in_frame_rpc.mean()), 0.2, float(in_frame_rpc.mean()) >= 0.2),
        CheckResult("reproj_p95_px", float(reproj["reproj_p95_px"]), float(args.reproj_p95_threshold), float(reproj["reproj_p95_px"]) <= float(args.reproj_p95_threshold)),
        CheckResult("current_alpha_max", current_alpha_max, float(args.alpha_max_threshold), current_alpha_max >= float(args.alpha_max_threshold)),
        CheckResult(
            "current_alpha_coverage",
            float(render_by_variant["current"]["alpha_coverage"]),
            float(args.alpha_coverage_threshold),
            float(render_by_variant["current"]["alpha_coverage"]) >= float(args.alpha_coverage_threshold),
        ),
        CheckResult(
            "gsplat_convention_mismatch_suspected",
            1.0 if conv_mismatch_suspected else 0.0,
            0.5,
            not conv_mismatch_suspected,
            note=(
                f"best_variant={best_variant}, current_alpha_max={current_alpha_max:.6f}, "
                f"best_alpha_max={best_alpha_max:.6f}"
            ),
        ),
    ]

    scene_id = int(batch["scene_id"][0].item())
    work_dir = Path(cfg.get("system", {}).get("work_dir", "work_dirs/sat2world_default"))
    out_dir = work_dir / "render_single_view_probe" / f"scene_{scene_id}" / f"view_{int(args.view_k)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_images:
        target_rgb = batch["images"][0, args.view_k]
        _save_chw_rgb(out_dir / "target_rgb.png", target_rgb)

        rr_curr = render_by_variant["current"]["rendered_rgb"]
        ra_curr = render_by_variant["current"]["rendered_alpha"]
        _save_chw_rgb(out_dir / "render_current_rgb.png", rr_curr)
        _save_chw_gray(out_dir / "render_current_alpha.png", ra_curr)

        rr_best = render_by_variant[best_variant]["rendered_rgb"]
        ra_best = render_by_variant[best_variant]["rendered_alpha"]
        _save_chw_rgb(out_dir / "render_best_variant_rgb.png", rr_best)
        _save_chw_gray(out_dir / "render_best_variant_alpha.png", ra_best)

        _draw_points_overlay(target_rgb, rpc_line, rpc_samp, out_dir / "rpc_projection_overlay.png", color=(255, 255, 0))
        _draw_points_overlay(target_rgb, pd_line, pd_samp, out_dir / "pinhole_projection_overlay.png", color=(255, 0, 255))

    report = {
        "scene_index": int(args.scene_index),
        "scene_id": int(scene_id),
        "view_k": int(args.view_k),
        "image_hw": [int(h), int(w)],
        "fit": {
            "p50": float(cam.fit_p50),
            "p95": float(cam.fit_p95),
            "max": float(cam.fit_max),
            "K": cam.K.detach().cpu().numpy().tolist(),
            "w2c": cam.w2c.detach().cpu().numpy().tolist(),
        },
        "sampled_gaussians": {
            "num": int(centers.shape[0]),
            "xy_span_ratio": float(args.xy_span_ratio),
            "sx": float(gs["sx"].item()),
            "sy": float(gs["sy"].item()),
            "z_low": float(gs["z_low"].item()),
            "z_high": float(gs["z_high"].item()),
            "xyz_min": centers.detach().cpu().numpy().min(axis=0).tolist(),
            "xyz_max": centers.detach().cpu().numpy().max(axis=0).tolist(),
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
        },
        "probe_rpc_projection": {
            "in_frame_ratio": float(in_frame_rpc.mean()),
            "finite_ratio": float((np.isfinite(rpc_line) & np.isfinite(rpc_samp)).mean()),
        },
        "probe_pinhole_reprojection": reproj,
        "probe_camera_matrix": cam_diag,
        "probe_render_variants": {
            k: {
                "alpha_max": float(vv["alpha_max"]),
                "alpha_thr": float(vv["alpha_thr"]),
                "alpha_coverage": float(vv["alpha_coverage"]),
            }
            for k, vv in render_by_variant.items()
        },
        "best_variant": best_variant,
        "convention_mismatch_suspected": bool(conv_mismatch_suspected),
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

    if args.dump_json:
        with open(out_dir / "probe_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n================ Single-View Render Probe Summary ================")
    print(f"scene_index={args.scene_index} scene_id={scene_id} view_k={args.view_k}")
    print(f"image_hw={h}x{w}")
    print(f"fit: p50={cam.fit_p50:.4f}px p95={cam.fit_p95:.4f}px max={cam.fit_max:.4f}px")
    print(
        f"gaussians: N={centers.shape[0]}, xyz_min={report['sampled_gaussians']['xyz_min']}, "
        f"xyz_max={report['sampled_gaussians']['xyz_max']}"
    )

    all_pass = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        extra = f" ({c.note})" if c.note else ""
        print(f"[{status}] {c.name}: value={c.value:.6f}, threshold={c.threshold:.6f}{extra}")
        all_pass = all_pass and c.passed

    print("\n-- variant alpha probe --")
    for name, vv in report["probe_render_variants"].items():
        print(f"  {name:16s}: alpha_max={vv['alpha_max']:.6f}, alpha_cov={vv['alpha_coverage']:.6f}")
    print(
        f"  best_variant={best_variant}, current_alpha_max={current_alpha_max:.6f}, "
        f"best_alpha_max={best_alpha_max:.6f}"
    )

    print("---------------------------------------------------------------")
    print("RESULT:", "PASS" if all_pass else "FAIL")
    print(f"output_dir: {str(out_dir)}")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

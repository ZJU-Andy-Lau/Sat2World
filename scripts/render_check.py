"""Sat2World 新渲染链路检查脚本（RPC->虚拟pinhole->CUDA栅格）。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataset, rpc_scene_collate_fn
from render.rpc_gaussian_renderer import RPCGaussianRenderer, RPCGaussianRendererCfg


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

    def log(self, message: str) -> None:
        self.step += 1
        now = time.perf_counter()
        print(
            f"[render_check][step={self.step:02d}] {message} | step={now - self.t_last:.2f}s total={now - self.t0:.2f}s",
            flush=True,
        )
        self.t_last = now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World virtual-pinhole render checker")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--view-i", type=int, default=0)
    p.add_argument("--view-j", type=int, default=1)
    p.add_argument("--fit-p95-threshold", type=float, default=1.0)
    p.add_argument("--fit-max-threshold", type=float, default=3.0)
    p.add_argument("--render-time-threshold-sec", type=float, default=2.0)
    p.add_argument("--center-centroid-threshold-px", type=float, default=2.0)
    p.add_argument("--scale-radius-expected-px", type=float, default=1.0)
    p.add_argument("--scale-radius-threshold-px", type=float, default=0.75)
    p.add_argument("--visibility-l1-threshold", type=float, default=0.20)
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
    # 检查时采用较密拟合
    rcfg.fit_grid_nx = max(int(getattr(rcfg, "fit_grid_nx", 24)), 24)
    rcfg.fit_grid_ny = max(int(getattr(rcfg, "fit_grid_ny", 24)), 24)
    rcfg.fit_grid_nz = max(int(getattr(rcfg, "fit_grid_nz", 7)), 7)
    # correctness 检查时尽量保留全部高斯，避免被筛选影响评估
    rcfg.source_stride = 1
    rcfg.confidence_threshold = 0.0
    rcfg.topk_per_target = None
    rcfg.exclude_self_source = False
    rcfg.val_num_target_views = max(int(getattr(rcfg, "val_num_target_views", 1)), 1)
    rcfg.use_all_targets_in_val = True
    rcfg.render_downsample_factor_val = 1
    from geometry import RPCGeometryOps

    return RPCGaussianRenderer(RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32), rcfg)


def _safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def build_dense_anchor_outputs(batch: dict[str, Any], vi: int, sh_dim: int = 48) -> dict[str, Any]:
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
    rgb_i = batch["images"][0, vi].clamp(0.0, 1.0)
    sh[0, vi, 0:3] = _safe_logit(rgb_i)

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


def _save_chw_rgb(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def _save_chw_gray(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().squeeze(0).clamp(0.0, 1.0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def _project_anchor_to_view_j(
    batch: dict[str, Any],
    vi: int,
    vj: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    rgb = batch["images"][0, vi].permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy().astype(np.float32)

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
    return (
        line_j.detach().cpu().numpy().astype(np.float32),
        samp_j.detach().cpu().numpy().astype(np.float32),
        x.detach().cpu().numpy().astype(np.float32),
        y.detach().cpu().numpy().astype(np.float32),
        hs.detach().cpu().numpy().astype(np.float32),
        rgb,
    )


def _build_expected_maps_from_projection(
    line_j: np.ndarray,
    samp_j: np.ndarray,
    depth_like: np.ndarray,
    rgb: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l_int = np.rint(line_j).astype(np.int64)
    s_int = np.rint(samp_j).astype(np.int64)
    valid = (l_int >= 0) & (l_int < h) & (s_int >= 0) & (s_int < w) & np.isfinite(depth_like)
    expected_rgb = np.zeros((h, w, 3), dtype=np.float32)
    expected_depth = np.full((h, w), np.inf, dtype=np.float32)
    expected_occ = np.zeros((h, w), dtype=np.float32)

    idxs = np.nonzero(valid)[0]
    for idx in idxs:
        li = l_int[idx]
        si = s_int[idx]
        d = depth_like[idx]
        if d < expected_depth[li, si]:
            expected_depth[li, si] = d
            expected_rgb[li, si] = rgb[idx]
        expected_occ[li, si] += 1.0
    return expected_rgb, expected_depth, expected_occ


def _compute_correctness_metrics(
    rendered_rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
    expected_rgb: np.ndarray,
    expected_occ: np.ndarray,
    expected_line: np.ndarray,
    expected_samp: np.ndarray,
    scale_expected_px: float,
) -> dict[str, float]:
    h, w = rendered_rgb.shape[-2:]
    rr = rendered_rgb.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    ra = rendered_alpha.detach().cpu().squeeze(0).numpy().astype(np.float32)

    valid_occ = expected_occ > 0
    if valid_occ.any():
        color_l1 = np.abs(rr[valid_occ] - expected_rgb[valid_occ]).mean()
    else:
        color_l1 = float("inf")

    alpha_max = float(np.maximum(ra, 0.0).max())
    alpha_thr = max(0.01, alpha_max * 0.1)
    alpha_cov = float((ra > alpha_thr).mean())

    # 中心正确性：比较投影点质心与渲染 alpha 质心
    l_int = np.rint(expected_line).astype(np.int64)
    s_int = np.rint(expected_samp).astype(np.int64)
    valid = (l_int >= 0) & (l_int < h) & (s_int >= 0) & (s_int < w)
    if valid.any():
        gt_cy = float(expected_line[valid].mean())
        gt_cx = float(expected_samp[valid].mean())
    else:
        gt_cy, gt_cx = 0.0, 0.0

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    a_sum = float(np.maximum(ra, 0.0).sum())
    if a_sum > 1e-8:
        rd_cy = float((yy * ra).sum() / a_sum)
        rd_cx = float((xx * ra).sum() / a_sum)
    else:
        rd_cy, rd_cx = float("inf"), float("inf")
    center_centroid_err = float(np.sqrt((gt_cy - rd_cy) ** 2 + (gt_cx - rd_cx) ** 2)) if np.isfinite(rd_cy) else float("inf")

    # 尺度正确性：把 alpha mask 视作所有基元叠加后的覆盖，折算“每个高斯平均等效半径”
    valid_num = int(valid.sum())
    area = float((ra > alpha_thr).sum())
    if valid_num > 0:
        area_per = area / float(valid_num)
        radius_obs = float(np.sqrt(max(area_per, 0.0) / np.pi))
    else:
        radius_obs = 0.0
    radius_bias = abs(radius_obs - float(scale_expected_px))

    # 可见性（深度排序）近似正确性：在期望有投影的像素处，颜色接近比例
    if valid_occ.any():
        pix_l1 = np.abs(rr[valid_occ] - expected_rgb[valid_occ]).mean(axis=1)
        visibility_ok_ratio = float((pix_l1 < 0.10).mean())
    else:
        visibility_ok_ratio = 0.0

    return {
        "center_centroid_err_px": center_centroid_err,
        "scale_radius_obs_px": radius_obs,
        "scale_radius_bias_px": radius_bias,
        "visibility_color_l1": float(color_l1),
        "visibility_ok_ratio": visibility_ok_ratio,
        "alpha_max": alpha_max,
        "alpha_thr": float(alpha_thr),
        "alpha_coverage": alpha_cov,
    }


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
        raise ValueError("view_i and view_j must be different for cross-view render correctness check")

    prog.log("build renderer")
    renderer = make_renderer(cfg)

    prog.log("fit virtual camera")
    cam = renderer.fit_virtual_camera_for_target(batch, 0, args.view_j)

    checks: list[CheckResult] = [
        CheckResult("fit_p95_px", cam.fit_p95, args.fit_p95_threshold, cam.fit_p95 <= args.fit_p95_threshold),
        CheckResult("fit_max_px", cam.fit_max, args.fit_max_threshold, cam.fit_max <= args.fit_max_threshold),
    ]

    prog.log("prepare synthetic outputs")
    outputs = build_dense_anchor_outputs(batch, args.view_i)

    prog.log("render paths")
    t0 = time.perf_counter()
    render_out = renderer.render_paths(outputs, batch, mode="val", global_step=0, epoch=0)
    dt = time.perf_counter() - t0
    checks.append(CheckResult("render_time_sec", dt, args.render_time_threshold_sec, dt <= args.render_time_threshold_sec))

    rpc_path = render_out["rpc"]
    point_path = render_out["point"]
    if int(rpc_path.get("num_targets", 0)) <= 0 or int(point_path.get("num_targets", 0)) <= 0:
        checks.append(CheckResult("num_targets", 0.0, 1.0, False, "render output has no targets"))

    # 在 rpc path 中选择目标 view_j 对应的渲染结果
    tv = rpc_path["target_view_indices"].detach().cpu()
    match = (tv == int(args.view_j)).nonzero(as_tuple=False).view(-1)
    if match.numel() == 0:
        checks.append(CheckResult("target_view_present", 0.0, 1.0, False, f"view_j={args.view_j} not rendered"))
    else:
        ridx = int(match[0].item())
        rendered_rgb = rpc_path["rendered_rgb"][ridx]
        rendered_alpha = rpc_path["rendered_alpha"][ridx]
        target_rgb = rpc_path["target_rgb"][ridx]

        # 构建期望投影（用于中心/尺度/可见性检查）
        line_j, samp_j, x_obj, y_obj, h_obj, src_rgb = _project_anchor_to_view_j(batch, args.view_i, args.view_j)
        # 可见性期望图应使用 target 相机深度，而非直接使用高程值。
        xyz_h = np.stack([x_obj, y_obj, h_obj, np.ones_like(h_obj)], axis=-1)  # [N,4]
        cam_np = cam.w2c.detach().cpu().numpy().astype(np.float64)
        cam_xyz = (cam_np @ xyz_h.T).T
        depth_like = cam_xyz[:, 2].astype(np.float32)
        exp_rgb, _exp_depth, exp_occ = _build_expected_maps_from_projection(
            line_j,
            samp_j,
            depth_like,
            src_rgb,
            int(rendered_rgb.shape[-2]),
            int(rendered_rgb.shape[-1]),
        )
        metrics = _compute_correctness_metrics(
            rendered_rgb,
            rendered_alpha,
            exp_rgb,
            exp_occ,
            line_j,
            samp_j,
            scale_expected_px=float(args.scale_radius_expected_px),
        )
        checks.extend(
            [
                CheckResult(
                    "alpha_max",
                    metrics["alpha_max"],
                    args.alpha_max_min_threshold,
                    metrics["alpha_max"] >= args.alpha_max_min_threshold,
                    note=f"alpha_thr={metrics['alpha_thr']:.6f}",
                ),
                CheckResult(
                    "alpha_coverage",
                    metrics["alpha_coverage"],
                    args.alpha_coverage_min_threshold,
                    metrics["alpha_coverage"] >= args.alpha_coverage_min_threshold,
                ),
            ]
        )

        checks.extend(
            [
                CheckResult(
                    "center_centroid_err_px",
                    metrics["center_centroid_err_px"],
                    args.center_centroid_threshold_px,
                    metrics["center_centroid_err_px"] <= args.center_centroid_threshold_px,
                ),
                CheckResult(
                    "scale_radius_bias_px",
                    metrics["scale_radius_bias_px"],
                    args.scale_radius_threshold_px,
                    metrics["scale_radius_bias_px"] <= args.scale_radius_threshold_px,
                    note=f"obs={metrics['scale_radius_obs_px']:.4f}px expected={args.scale_radius_expected_px:.4f}px",
                ),
                CheckResult(
                    "visibility_color_l1",
                    metrics["visibility_color_l1"],
                    args.visibility_l1_threshold,
                    metrics["visibility_color_l1"] <= args.visibility_l1_threshold,
                    note=f"ok_ratio={metrics['visibility_ok_ratio']:.4f}, alpha_max={metrics['alpha_max']:.6f}",
                ),
            ]
        )

        if args.save_images:
            work_dir = Path(cfg.get("system", {}).get("work_dir", "work_dirs/sat2world_default"))
            out_dir = work_dir / "render_check" / f"scene_{int(batch['scene_id'][0].item())}" / f"vi_{args.view_i}_vj_{args.view_j}"
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_chw_rgb(out_dir / "rendered_rgb.png", rendered_rgb)
            _save_chw_rgb(out_dir / "target_rgb.png", target_rgb)
            _save_chw_gray(out_dir / "rendered_alpha.png", rendered_alpha)
            _save_chw_rgb(out_dir / "expected_rgb_nearest_depth.png", torch.from_numpy(exp_rgb).permute(2, 0, 1))
            abs_err = (rendered_rgb.detach().cpu() - torch.from_numpy(exp_rgb).permute(2, 0, 1)).abs().clamp(0.0, 1.0)
            _save_chw_rgb(out_dir / "abs_error_render_vs_expected.png", abs_err)
            occ_norm = torch.from_numpy(exp_occ / max(float(exp_occ.max()), 1.0)).unsqueeze(0)
            _save_chw_gray(out_dir / "expected_occupancy.png", occ_norm)

    print("\n================ Render Check Summary ================")
    print(f"scene_index={args.scene_index} view_i={args.view_i} view_j={args.view_j}")
    print(f"image_hw={h}x{w}")
    print(f"fit: p50={cam.fit_p50:.4f}px p95={cam.fit_p95:.4f}px max={cam.fit_max:.4f}px")
    print(f"render_time={dt:.4f}s")

    all_pass = True
    for r in checks:
        status = "PASS" if r.passed else "FAIL"
        extra = f" ({r.note})" if r.note else ""
        print(f"[{status}] {r.name}: value={r.value:.6f}, threshold={r.threshold:.6f}{extra}")
        all_pass = all_pass and r.passed

    print("------------------------------------------------------")
    print("RESULT:", "PASS" if all_pass else "FAIL")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

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
from render import (
    RPCGaussianRenderer,
    RPCGaussianRendererCfg,
    RPCGaussianRendererDGR,
    RPCGaussianRendererDGRCfg,
    VirtualPinholeCamera,
)


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
    p.add_argument("--cam-det-min-threshold", type=float, default=0.5)
    p.add_argument("--cam-orthogonality-max-threshold", type=float, default=1e-2)
    p.add_argument("--cam-source-positive-depth-ratio-min-threshold", type=float, default=0.5)
    p.add_argument("--cam-reproj-gap-min-threshold-px", type=float, default=1e-3)
    p.add_argument("--convention-probe-alpha-max-threshold", type=float, default=1e-3)
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--renderer-backend", type=str, default="", help="override renderer backend: gsplat|dgr")
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


def make_renderer(cfg: dict[str, Any], backend_override: str = "") -> tuple[RPCGaussianRenderer, str]:
    renderer_cfg = cfg.get("renderer", {})
    backend = str(
        backend_override or renderer_cfg.get("backend", renderer_cfg.get("render_backend", "gsplat"))
    ).lower()

    from geometry import RPCGeometryOps

    if backend in {"dgr", "diff_gaussian_rasterization", "diff-gaussian"}:
        rcfg = RPCGaussianRendererDGRCfg()
        for k, v in renderer_cfg.items():
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
        return RPCGaussianRendererDGR(RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32), rcfg), "dgr"

    rcfg = RPCGaussianRendererCfg()
    for k, v in renderer_cfg.items():
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
    return RPCGaussianRenderer(RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32), rcfg), "gsplat"


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


def _camera_handedness_diagnostics(
    cam: Any,
    xyz_local: np.ndarray,
    gt_line: np.ndarray,
    gt_samp: np.ndarray,
) -> dict[str, float]:
    """诊断虚拟相机是否存在“手性错误但重投影可通过”的问题。"""
    if xyz_local.ndim != 2 or xyz_local.shape[1] != 3:
        raise ValueError(f"xyz_local must be [N,3], got {xyz_local.shape}")
    n = xyz_local.shape[0]
    xyz_h = np.concatenate([xyz_local.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    K = cam.K.detach().cpu().numpy().astype(np.float64)
    w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    det_r = float(np.linalg.det(R))
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3, dtype=np.float64), ord="fro"))

    cam_xyz = (w2c[:3, :] @ xyz_h.T).T
    z = cam_xyz[:, 2]
    pos_ratio = float((z > 1e-6).mean())

    # 与当前代码中的符号翻转保持一致：R,t 同时乘 -1。
    R_neg = -R
    t_neg = -t
    w2c_neg = np.eye(4, dtype=np.float64)
    w2c_neg[:3, :3] = R_neg
    w2c_neg[:3, 3] = t_neg
    cam_xyz_neg = (w2c_neg[:3, :] @ xyz_h.T).T
    z_neg = cam_xyz_neg[:, 2]
    pos_ratio_neg = float((z_neg > 1e-6).mean())

    proj = (K @ cam_xyz.T).T
    proj_neg = (K @ cam_xyz_neg.T).T
    valid = np.isfinite(proj[:, 2]) & (np.abs(proj[:, 2]) > 1e-8)
    valid_neg = np.isfinite(proj_neg[:, 2]) & (np.abs(proj_neg[:, 2]) > 1e-8)
    uv = np.zeros((n, 2), dtype=np.float64)
    uv_neg = np.zeros((n, 2), dtype=np.float64)
    uv[valid] = proj[valid, :2] / proj[valid, 2:3]
    uv_neg[valid_neg] = proj_neg[valid_neg, :2] / proj_neg[valid_neg, 2:3]

    gt_uv = np.stack([gt_samp.astype(np.float64), gt_line.astype(np.float64)], axis=-1)
    both = valid & valid_neg & np.isfinite(gt_uv).all(axis=1)
    if both.any():
        err = np.linalg.norm(uv[both] - gt_uv[both], axis=1)
        err_neg = np.linalg.norm(uv_neg[both] - gt_uv[both], axis=1)
        reproj_p95 = float(np.quantile(err, 0.95))
        reproj_p95_neg = float(np.quantile(err_neg, 0.95))
        reproj_gap = abs(reproj_p95 - reproj_p95_neg)
    else:
        reproj_p95 = float("inf")
        reproj_p95_neg = float("inf")
        reproj_gap = float("inf")

    return {
        "cam_det_r": det_r,
        "cam_orthogonality_err": orth_err,
        "cam_source_positive_depth_ratio": pos_ratio,
        "cam_source_positive_depth_ratio_if_negated": pos_ratio_neg,
        "cam_reproj_p95_current_px": reproj_p95,
        "cam_reproj_p95_negated_px": reproj_p95_neg,
        "cam_reproj_p95_gap_px": reproj_gap,
    }


def _run_backend_convention_probe(
    renderer: RPCGaussianRenderer,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    vi: int,
    tv: int,
    cam: VirtualPinholeCamera,
) -> dict[str, Any]:
    """用多种相机矩阵约定探测“拟合链路 vs 渲染后端链路”的不一致。"""
    target_rgb = batch["images"][0, tv]
    h_out, w_out = int(target_rgb.shape[-2]), int(target_rgb.shape[-1])
    dev = target_rgb.device

    centers = outputs["gaussian_centers_rpc"][0].permute(0, 2, 3, 1).reshape(-1, 3)
    opacity = outputs["gaussian_opacity"][0].permute(0, 2, 3, 1).reshape(-1, 1)
    conf = outputs["gaussian_confidence_rpc"][0].permute(0, 2, 3, 1).reshape(-1, 1)
    scale = outputs["gaussian_scale"][0].permute(0, 2, 3, 1).reshape(-1, 3)
    rotation = outputs["gaussian_rotation"][0].permute(0, 2, 3, 1).reshape(-1, 4)
    sh = outputs["gaussian_sh"][0].permute(0, 2, 3, 1).reshape(-1, outputs["gaussian_sh"].shape[2])
    rgb = torch.sigmoid(sh[:, :3])

    op_eff = (opacity * conf).clamp(0.0, 1.0).squeeze(1)
    keep = op_eff > 1.0e-6
    if not bool(keep.any()):
        return {
            "num_probe_gaussians": 0,
            "alpha_max_by_variant": {},
            "best_variant": "none",
            "best_alpha_max": 0.0,
            "current_alpha_max": 0.0,
            "suspect_convention_mismatch": False,
        }

    centers = centers[keep].to(device=dev, dtype=torch.float32)
    scale = scale[keep].to(device=dev, dtype=torch.float32)
    rotation = rotation[keep].to(device=dev, dtype=torch.float32)
    rgb = rgb[keep].to(device=dev, dtype=torch.float32)
    op_eff = op_eff[keep].unsqueeze(1).to(device=dev, dtype=torch.float32)

    I4 = torch.eye(4, dtype=torch.float32, device=dev)
    Fz = I4.clone()
    Fz[2, 2] = -1.0
    Fy = I4.clone()
    Fy[1, 1] = -1.0
    Fzy = Fy @ Fz

    variants: dict[str, torch.Tensor] = {
        "current": cam.w2c.to(device=dev, dtype=torch.float32),
        "cam_z_flip": Fz @ cam.w2c.to(device=dev, dtype=torch.float32),
        "cam_y_flip": Fy @ cam.w2c.to(device=dev, dtype=torch.float32),
        "cam_zy_flip": Fzy @ cam.w2c.to(device=dev, dtype=torch.float32),
        "w2c_transposed": cam.w2c.to(device=dev, dtype=torch.float32).transpose(0, 1).contiguous(),
    }

    alpha_max_by_variant: dict[str, float] = {}
    for name, w2c_var in variants.items():
        cam_var = VirtualPinholeCamera(
            K=cam.K.to(device=dev, dtype=torch.float32),
            w2c=w2c_var,
            fit_p50=cam.fit_p50,
            fit_p95=cam.fit_p95,
            fit_max=cam.fit_max,
        )
        _, ra_var, _ = renderer._render_cuda(
            centers=centers,
            opacity=op_eff,
            scale=scale,
            rotation=rotation,
            rgb=rgb,
            cam=cam_var,
            image_hw=(h_out, w_out),
        )
        alpha_max_by_variant[name] = float(torch.maximum(ra_var, torch.zeros_like(ra_var)).max().item())

    best_variant = max(alpha_max_by_variant.items(), key=lambda kv: kv[1])[0]
    best_alpha = float(alpha_max_by_variant[best_variant])
    cur_alpha = float(alpha_max_by_variant.get("current", 0.0))
    return {
        "num_probe_gaussians": int(keep.sum().item()),
        "alpha_max_by_variant": alpha_max_by_variant,
        "best_variant": best_variant,
        "best_alpha_max": best_alpha,
        "current_alpha_max": cur_alpha,
    }


def _run_camera_variant_projection_probe(
    cam: VirtualPinholeCamera,
    xyz_world: np.ndarray,
    gt_line: np.ndarray,
    gt_samp: np.ndarray,
    image_hw: tuple[int, int],
) -> dict[str, Any]:
    """不依赖栅格化，仅用投影误差评估相机矩阵约定。"""
    h, w = image_hw
    xyz_h = np.concatenate([xyz_world.astype(np.float64), np.ones((xyz_world.shape[0], 1), dtype=np.float64)], axis=1)
    K = cam.K.detach().cpu().numpy().astype(np.float64)
    w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)

    I4 = np.eye(4, dtype=np.float64)
    Fz = I4.copy()
    Fz[2, 2] = -1.0
    Fy = I4.copy()
    Fy[1, 1] = -1.0
    Fzy = Fy @ Fz
    variants: dict[str, np.ndarray] = {
        "current": w2c,
        "cam_z_flip": Fz @ w2c,
        "cam_y_flip": Fy @ w2c,
        "cam_zy_flip": Fzy @ w2c,
        "w2c_transposed": w2c.T.copy(),
    }

    gt_uv = np.stack([gt_samp.astype(np.float64), gt_line.astype(np.float64)], axis=-1)
    metrics: dict[str, dict[str, float]] = {}
    for name, w2c_var in variants.items():
        cam_xyz = (w2c_var[:3, :] @ xyz_h.T).T
        z = cam_xyz[:, 2]
        proj = (K @ cam_xyz.T).T
        valid = np.isfinite(z) & (z > 1e-8) & np.isfinite(proj).all(axis=1) & np.isfinite(gt_uv).all(axis=1)
        if valid.any():
            uv = proj[valid, :2] / proj[valid, 2:3]
            err = np.linalg.norm(uv - gt_uv[valid], axis=1)
            in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            center_err = float(np.linalg.norm(uv.mean(axis=0) - gt_uv[valid].mean(axis=0)))
            metrics[name] = {
                "reproj_p50_px": float(np.quantile(err, 0.50)),
                "reproj_p95_px": float(np.quantile(err, 0.95)),
                "reproj_max_px": float(err.max()),
                "positive_depth_ratio": float(valid.mean()),
                "in_frame_ratio": float(in_frame.mean()),
                "center_centroid_err_px": center_err,
            }
        else:
            metrics[name] = {
                "reproj_p50_px": float("inf"),
                "reproj_p95_px": float("inf"),
                "reproj_max_px": float("inf"),
                "positive_depth_ratio": 0.0,
                "in_frame_ratio": 0.0,
                "center_centroid_err_px": float("inf"),
            }

    best_variant = min(metrics.items(), key=lambda kv: kv[1]["reproj_p95_px"])[0]
    return {
        "variants": metrics,
        "best_variant_by_reproj_p95": best_variant,
        "best_reproj_p95_px": float(metrics[best_variant]["reproj_p95_px"]),
        "current_reproj_p95_px": float(metrics["current"]["reproj_p95_px"]),
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
    renderer, renderer_backend = make_renderer(cfg, backend_override=args.renderer_backend)

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
        cam_diag = _camera_handedness_diagnostics(
            cam=cam,
            xyz_local=np.stack([x_obj, y_obj, h_obj], axis=-1),
            gt_line=line_j,
            gt_samp=samp_j,
        )
        conv_probe = _run_backend_convention_probe(
            renderer=renderer,
            outputs=outputs,
            batch=batch,
            vi=args.view_i,
            tv=args.view_j,
            cam=cam,
        )
        proj_probe = _run_camera_variant_projection_probe(
            cam=cam,
            xyz_world=np.stack([x_obj, y_obj, h_obj], axis=-1),
            gt_line=line_j,
            gt_samp=samp_j,
            image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
        )
        conv_suspect = (
            conv_probe["current_alpha_max"] < args.convention_probe_alpha_max_threshold
            and conv_probe["best_alpha_max"] >= args.convention_probe_alpha_max_threshold
            and conv_probe["best_variant"] != "current"
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
                CheckResult(
                    "cam_det_r",
                    cam_diag["cam_det_r"],
                    args.cam_det_min_threshold,
                    cam_diag["cam_det_r"] >= args.cam_det_min_threshold,
                    note=f"det should be close to +1; negated_depth_ratio={cam_diag['cam_source_positive_depth_ratio_if_negated']:.4f}",
                ),
                CheckResult(
                    "cam_orthogonality_err",
                    cam_diag["cam_orthogonality_err"],
                    args.cam_orthogonality_max_threshold,
                    cam_diag["cam_orthogonality_err"] <= args.cam_orthogonality_max_threshold,
                ),
                CheckResult(
                    "cam_source_positive_depth_ratio",
                    cam_diag["cam_source_positive_depth_ratio"],
                    args.cam_source_positive_depth_ratio_min_threshold,
                    cam_diag["cam_source_positive_depth_ratio"] >= args.cam_source_positive_depth_ratio_min_threshold,
                    note=f"if negated={cam_diag['cam_source_positive_depth_ratio_if_negated']:.4f}",
                ),
                CheckResult(
                    "cam_reproj_gap_current_vs_negated_p95_px",
                    cam_diag["cam_reproj_p95_gap_px"],
                    args.cam_reproj_gap_min_threshold_px,
                    cam_diag["cam_reproj_p95_gap_px"] >= args.cam_reproj_gap_min_threshold_px,
                    note=(
                        f"curr_p95={cam_diag['cam_reproj_p95_current_px']:.6f}, "
                        f"neg_p95={cam_diag['cam_reproj_p95_negated_px']:.6f}; "
                        "gap too small indicates sign ambiguity"
                    ),
                ),
                CheckResult(
                    "backend_convention_mismatch_suspected",
                    1.0 if conv_suspect else 0.0,
                    0.5,
                    not conv_suspect,
                    note=(
                        f"probeN={conv_probe['num_probe_gaussians']}, "
                        f"current_alpha_max={conv_probe['current_alpha_max']:.6f}, "
                        f"best_variant={conv_probe['best_variant']}, "
                        f"best_alpha_max={conv_probe['best_alpha_max']:.6f}, "
                        f"threshold={args.convention_probe_alpha_max_threshold:.6f}"
                    ),
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
    print(f"renderer_backend={renderer_backend}")
    print(f"fit: p50={cam.fit_p50:.4f}px p95={cam.fit_p95:.4f}px max={cam.fit_max:.4f}px")
    print(f"render_time={dt:.4f}s")

    all_pass = True
    for r in checks:
        status = "PASS" if r.passed else "FAIL"
        extra = f" ({r.note})" if r.note else ""
        print(f"[{status}] {r.name}: value={r.value:.6f}, threshold={r.threshold:.6f}{extra}")
        all_pass = all_pass and r.passed

    if "conv_probe" in locals():
        print("\n-- backend convention probe (alpha-based) --")
        for k, v in conv_probe["alpha_max_by_variant"].items():
            print(f"  {k:16s}: alpha_max={v:.6f}")
        print(
            f"  best_variant={conv_probe['best_variant']}, "
            f"best_alpha_max={conv_probe['best_alpha_max']:.6f}, "
            f"current_alpha_max={conv_probe['current_alpha_max']:.6f}"
        )

    if "proj_probe" in locals():
        print("\n-- camera variant projection probe (geometry-based) --")
        for name, m in proj_probe["variants"].items():
            print(
                f"  {name:16s}: "
                f"reproj_p50={m['reproj_p50_px']:.6f}px, "
                f"reproj_p95={m['reproj_p95_px']:.6f}px, "
                f"reproj_max={m['reproj_max_px']:.6f}px, "
                f"center_err={m['center_centroid_err_px']:.6f}px, "
                f"pos_depth_ratio={m['positive_depth_ratio']:.4f}, "
                f"in_frame_ratio={m['in_frame_ratio']:.4f}"
            )
        print(
            f"  best_variant_by_reproj_p95={proj_probe['best_variant_by_reproj_p95']}, "
            f"best_reproj_p95={proj_probe['best_reproj_p95_px']:.6f}px, "
            f"current_reproj_p95={proj_probe['current_reproj_p95_px']:.6f}px"
        )

    print("------------------------------------------------------")
    print("RESULT:", "PASS" if all_pass else "FAIL")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

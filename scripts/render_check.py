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
from render.rpc_gaussian_renderer import RPCGaussianRenderer, RPCGaussianRendererCfg, VirtualPinholeCamera


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
    p.add_argument("--convention-probe-alpha-max-threshold", type=float, default=1e-3)
    p.add_argument("--enable-cover-camera-probe", action="store_true")
    p.add_argument("--cover-camera-margin-ratio", type=float, default=0.05)
    p.add_argument("--cover-camera-distance-factor", type=float, default=3.0)
    p.add_argument("--cover-camera-max-retries", type=int, default=8)
    p.add_argument("--enable-cover-to-fit-sweep", action="store_true")
    p.add_argument("--sweep-num-cameras", type=int, default=15)
    p.add_argument("--sweep-min-scene-dist-ratio", type=float, default=0.15)
    p.add_argument("--root-cause-alpha-ratio-threshold", type=float, default=10.0)
    p.add_argument("--root-cause-reproj-threshold-px", type=float, default=2.0)
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


def _look_at_w2c_np(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    r = np.cross(f, up_hint)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, f)
    R = np.stack([r, u, f], axis=0)
    t = -R @ eye
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def _fit_intrinsics_cover_points_np(points_xyz: np.ndarray, w2c: np.ndarray, width: int, height: int, margin_ratio: float) -> np.ndarray:
    cxy = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    xyz_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (w2c[:3, :] @ xyz_h.T).T
    z = cam[:, 2]
    if not np.all(z > 1e-6):
        raise RuntimeError("cover camera invalid: some points are behind camera")

    xz = np.abs(cam[:, 0] / z)
    yz = np.abs(cam[:, 1] / z)
    usable_w = max(float(width) * (1.0 - 2.0 * margin_ratio), 1.0)
    usable_h = max(float(height) * (1.0 - 2.0 * margin_ratio), 1.0)
    fx_max = (usable_w / 2.0) / max(float(np.max(xz)), 1e-8)
    fy_max = (usable_h / 2.0) / max(float(np.max(yz)), 1e-8)
    fx = 0.98 * fx_max
    fy = 0.98 * fy_max
    K = np.array(
        [
            [fx, 0.0, cxy[0]],
            [0.0, fy, cxy[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def _build_cover_camera_from_points(
    xyz_world: np.ndarray,
    image_hw: tuple[int, int],
    device: torch.device,
    margin_ratio: float,
    distance_factor: float,
    max_retries: int,
) -> VirtualPinholeCamera:
    h, w = image_hw
    xyz = xyz_world.astype(np.float64)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    extent = xyz_max - xyz_min
    diag = float(np.linalg.norm(extent) + 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    last_err: Exception | None = None
    for i in range(max(int(max_retries), 1)):
        d = max(diag * float(distance_factor) * (1.6**i), 1.0)
        eye = center + np.array([0.0, 0.0, d], dtype=np.float64)
        w2c = _look_at_w2c_np(eye=eye, target=center, up_hint=up)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
        cam = (w2c[:3, :] @ xyz_h.T).T
        if not np.all(cam[:, 2] > 1e-6):
            last_err = RuntimeError("some points behind camera after look-at")
            continue
        try:
            K = _fit_intrinsics_cover_points_np(xyz, w2c, w, h, float(margin_ratio))
            return VirtualPinholeCamera(
                K=torch.from_numpy(K).to(device=device, dtype=torch.float32),
                w2c=torch.from_numpy(w2c).to(device=device, dtype=torch.float32),
                fit_p50=float("nan"),
                fit_p95=float("nan"),
                fit_max=float("nan"),
            )
        except Exception as e:  # pragma: no cover - diagnostics fallback path
            last_err = e
            continue

    raise RuntimeError(f"failed to build cover camera after {max_retries} retries: {last_err}")


def _camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return (-R.T @ t).reshape(3)


def _build_cover_camera_from_eye(
    *,
    eye_world: np.ndarray,
    target_world: np.ndarray,
    points_xyz: np.ndarray,
    image_hw: tuple[int, int],
    device: torch.device,
    margin_ratio: float,
) -> VirtualPinholeCamera:
    h, w = image_hw
    w2c = _look_at_w2c_np(eye=eye_world, target=target_world, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    K = _fit_intrinsics_cover_points_np(points_xyz, w2c, w, h, margin_ratio)
    return VirtualPinholeCamera(
        K=torch.from_numpy(K).to(device=device, dtype=torch.float32),
        w2c=torch.from_numpy(w2c).to(device=device, dtype=torch.float32),
        fit_p50=float("nan"),
        fit_p95=float("nan"),
        fit_max=float("nan"),
    )


def _extract_probe_gaussians(outputs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    centers = outputs["gaussian_centers_rpc"][0].permute(0, 2, 3, 1).reshape(-1, 3)
    opacity = outputs["gaussian_opacity"][0].permute(0, 2, 3, 1).reshape(-1, 1)
    conf = outputs["gaussian_confidence_rpc"][0].permute(0, 2, 3, 1).reshape(-1, 1)
    scale = outputs["gaussian_scale"][0].permute(0, 2, 3, 1).reshape(-1, 3)
    rotation = outputs["gaussian_rotation"][0].permute(0, 2, 3, 1).reshape(-1, 4)
    sh = outputs["gaussian_sh"][0].permute(0, 2, 3, 1).reshape(-1, outputs["gaussian_sh"].shape[2])
    rgb = torch.sigmoid(sh[:, :3])
    op_eff = (opacity * conf).clamp(0.0, 1.0).squeeze(1)
    keep = op_eff > 1.0e-6
    return {
        "centers": centers[keep].to(device=device, dtype=torch.float32),
        "scale": scale[keep].to(device=device, dtype=torch.float32),
        "rotation": rotation[keep].to(device=device, dtype=torch.float32),
        "rgb": rgb[keep].to(device=device, dtype=torch.float32),
        "opacity": op_eff[keep].unsqueeze(1).to(device=device, dtype=torch.float32),
    }


def _camera_intrinsics_diagnostics(cam: VirtualPinholeCamera, image_hw: tuple[int, int]) -> dict[str, float]:
    h, w = image_hw
    K = cam.K.detach().cpu().numpy().astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    skew = float(K[0, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fov_x = float(2.0 * np.arctan2(float(w), max(2.0 * abs(fx), 1e-12)) * 180.0 / np.pi)
    fov_y = float(2.0 * np.arctan2(float(h), max(2.0 * abs(fy), 1e-12)) * 180.0 / np.pi)
    A = K[:2, :2]
    cond = float(np.linalg.cond(A)) if np.isfinite(A).all() else float("inf")
    return {
        "fx": fx,
        "fy": fy,
        "skew": skew,
        "cx": cx,
        "cy": cy,
        "fx_over_w": fx / max(float(w), 1.0),
        "fy_over_h": fy / max(float(h), 1.0),
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "k2x2_cond": cond,
        "fx_positive": 1.0 if fx > 0 else 0.0,
        "fy_positive": 1.0 if fy > 0 else 0.0,
        "principal_in_frame": 1.0 if (0.0 <= cx < float(w) and 0.0 <= cy < float(h)) else 0.0,
    }


def _project_with_camera_np(cam: VirtualPinholeCamera, xyz_world: np.ndarray, image_hw: tuple[int, int]) -> dict[str, float]:
    h, w = image_hw
    n = xyz_world.shape[0]
    xyz_h = np.concatenate([xyz_world.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    K = cam.K.detach().cpu().numpy().astype(np.float64)
    w2c = cam.w2c.detach().cpu().numpy().astype(np.float64)
    cam_xyz = (w2c[:3, :] @ xyz_h.T).T
    z = cam_xyz[:, 2]
    proj = (K @ cam_xyz.T).T
    valid = np.isfinite(z) & (z > 1e-8) & np.isfinite(proj).all(axis=1)
    if not valid.any():
        return {
            "positive_depth_ratio": 0.0,
            "in_frame_ratio": 0.0,
            "z_p01": float("inf"),
            "z_p50": float("inf"),
            "z_p99": float("inf"),
            "uv_bbox_w": 0.0,
            "uv_bbox_h": 0.0,
        }
    uv = proj[valid, :2] / proj[valid, 2:3]
    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    z_valid = z[valid]
    return {
        "positive_depth_ratio": float(valid.mean()),
        "in_frame_ratio": float(in_frame.mean()),
        "z_p01": float(np.quantile(z_valid, 0.01)),
        "z_p50": float(np.quantile(z_valid, 0.50)),
        "z_p99": float(np.quantile(z_valid, 0.99)),
        "uv_bbox_w": float(np.max(uv[:, 0]) - np.min(uv[:, 0])) if uv.shape[0] > 0 else 0.0,
        "uv_bbox_h": float(np.max(uv[:, 1]) - np.min(uv[:, 1])) if uv.shape[0] > 0 else 0.0,
    }


def _k_ablation_render_probe(
    renderer: RPCGaussianRenderer,
    cam: VirtualPinholeCamera,
    gauss: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
) -> dict[str, dict[str, float]]:
    K0 = cam.K.detach().clone()
    h, w = image_hw
    K_center = K0.clone()
    K_center[0, 2] = float(w) * 0.5
    K_center[1, 2] = float(h) * 0.5
    variants: dict[str, torch.Tensor] = {
        "current": K0,
        "abs_focal": torch.stack(
            [
                torch.stack([K0[0, 0].abs(), K0[0, 1], K0[0, 2]]),
                torch.stack([K0[1, 0], K0[1, 1].abs(), K0[1, 2]]),
                K0[2],
            ]
        ),
        "zero_skew": torch.stack(
            [
                torch.stack([K0[0, 0], torch.zeros_like(K0[0, 1]), K0[0, 2]]),
                K0[1],
                K0[2],
            ]
        ),
        "center_principal": K_center,
    }
    out: dict[str, dict[str, float]] = {}
    for name, K_var in variants.items():
        cam_v = VirtualPinholeCamera(K=K_var, w2c=cam.w2c, fit_p50=cam.fit_p50, fit_p95=cam.fit_p95, fit_max=cam.fit_max)
        _, ra, _ = renderer._render_cuda(
            centers=gauss["centers"],
            opacity=gauss["opacity"],
            scale=gauss["scale"],
            rotation=gauss["rotation"],
            rgb=gauss["rgb"],
            cam=cam_v,
            image_hw=image_hw,
        )
        a = ra.detach().cpu().numpy().squeeze(0).astype(np.float32)
        amax = float(np.maximum(a, 0.0).max())
        athr = max(0.01, amax * 0.1)
        acov = float((a > athr).mean())
        out[name] = {
            "alpha_max": amax,
            "alpha_cov": acov,
            "alpha_thr": float(athr),
        }
    return out


def _camera_handedness_diagnostics(
    cam: Any,
    xyz_local: np.ndarray,
    gt_line: np.ndarray,
    gt_samp: np.ndarray,
) -> dict[str, float]:
    """诊断虚拟相机旋转矩阵与深度方向是否合理。"""
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

    proj = (K @ cam_xyz.T).T
    valid = np.isfinite(proj[:, 2]) & (np.abs(proj[:, 2]) > 1e-8) & np.isfinite(gt_line) & np.isfinite(gt_samp)
    if valid.any():
        uv = proj[valid, :2] / proj[valid, 2:3]
        gt_uv = np.stack([gt_samp.astype(np.float64), gt_line.astype(np.float64)], axis=-1)[valid]
        err = np.linalg.norm(uv - gt_uv, axis=1)
        reproj_p95 = float(np.quantile(err, 0.95))
    else:
        reproj_p95 = float("inf")

    return {
        "cam_det_r": det_r,
        "cam_orthogonality_err": orth_err,
        "cam_source_positive_depth_ratio": pos_ratio,
        "cam_reproj_p95_current_px": reproj_p95,
    }


def _run_gsplat_convention_probe(
    renderer: RPCGaussianRenderer,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    vi: int,
    tv: int,
    cam: VirtualPinholeCamera,
) -> dict[str, Any]:
    """用多种相机矩阵约定探测“拟合链路 vs gsplat 渲染链路”的不一致。"""
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
        cam_diag = _camera_handedness_diagnostics(
            cam=cam,
            xyz_local=np.stack([x_obj, y_obj, h_obj], axis=-1),
            gt_line=line_j,
            gt_samp=samp_j,
        )
        conv_probe = _run_gsplat_convention_probe(
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
        fit_intr_diag = _camera_intrinsics_diagnostics(cam, (int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])))
        cover_metrics: dict[str, float] | None = None
        cover_diag: dict[str, float] | None = None
        fit_render_domain_diag: dict[str, float] | None = None
        cover_render_domain_diag: dict[str, float] | None = None
        k_ablation_diag: dict[str, dict[str, float]] | None = None
        cover_rr: torch.Tensor | None = None
        cover_ra: torch.Tensor | None = None
        gauss = _extract_probe_gaussians(outputs, device=target_rgb.device)
        if gauss["centers"].shape[0] > 0:
            fit_render_domain_diag = _project_with_camera_np(
                cam,
                gauss["centers"].detach().cpu().numpy().astype(np.float64),
                (int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
            )
            k_ablation_diag = _k_ablation_render_probe(
                renderer=renderer,
                cam=cam,
                gauss=gauss,
                image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
            )
        if args.enable_cover_camera_probe and gauss["centers"].shape[0] > 0:
                cover_cam = _build_cover_camera_from_points(
                    xyz_world=gauss["centers"].detach().cpu().numpy().astype(np.float64),
                    image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
                    device=target_rgb.device,
                    margin_ratio=float(args.cover_camera_margin_ratio),
                    distance_factor=float(args.cover_camera_distance_factor),
                    max_retries=int(args.cover_camera_max_retries),
                )
                cover_rr, cover_ra, _ = renderer._render_cuda(
                    centers=gauss["centers"],
                    opacity=gauss["opacity"],
                    scale=gauss["scale"],
                    rotation=gauss["rotation"],
                    rgb=gauss["rgb"],
                    cam=cover_cam,
                    image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
                )
                # 覆盖相机不是目标视图几何，不计算中心/可见性误差，仅评估渲染有效性。
                cover_alpha = cover_ra.detach().cpu().numpy().squeeze(0).astype(np.float32)
                cover_alpha_max = float(np.maximum(cover_alpha, 0.0).max())
                cover_alpha_thr = max(0.01, cover_alpha_max * 0.1)
                cover_alpha_cov = float((cover_alpha > cover_alpha_thr).mean())
                cover_metrics = {
                    "alpha_max": cover_alpha_max,
                    "alpha_thr": float(cover_alpha_thr),
                    "alpha_coverage": cover_alpha_cov,
                }
                cover_diag = {
                    "alpha_max_ratio_cover_vs_fit": float(cover_alpha_max / max(float(metrics["alpha_max"]), 1e-12)),
                    "alpha_cov_ratio_cover_vs_fit": float(cover_alpha_cov / max(float(metrics["alpha_coverage"]), 1e-12)),
                }
                cover_render_domain_diag = _project_with_camera_np(
                    cover_cam,
                    gauss["centers"].detach().cpu().numpy().astype(np.float64),
                    (int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
                )
                checks.extend(
                    [
                        CheckResult(
                            "cover_cam_alpha_max",
                            cover_metrics["alpha_max"],
                            args.alpha_max_min_threshold,
                            cover_metrics["alpha_max"] >= args.alpha_max_min_threshold,
                            note="synthetic coverage camera",
                        ),
                        CheckResult(
                            "cover_cam_alpha_coverage",
                            cover_metrics["alpha_coverage"],
                            args.alpha_coverage_min_threshold,
                            cover_metrics["alpha_coverage"] >= args.alpha_coverage_min_threshold,
                            note="synthetic coverage camera",
                        ),
                    ]
                )
        sweep_records: list[dict[str, Any]] = []
        if args.enable_cover_to_fit_sweep and gauss["centers"].shape[0] > 0:
            xyz_np = gauss["centers"].detach().cpu().numpy().astype(np.float64)
            scene_center = xyz_np.mean(axis=0)
            fit_w2c_np = cam.w2c.detach().cpu().numpy().astype(np.float64)
            fit_center = _camera_center_from_w2c(fit_w2c_np)
            vec = fit_center - scene_center
            vec_norm = float(np.linalg.norm(vec) + 1e-12)
            min_dist = max(float(np.linalg.norm(xyz_np.max(axis=0) - xyz_np.min(axis=0))) * float(args.sweep_min_scene_dist_ratio), 1.0)
            for si in range(max(int(args.sweep_num_cameras), 2)):
                lam = float(si) / float(max(int(args.sweep_num_cameras) - 1, 1))
                eye = (1.0 - lam) * scene_center + lam * fit_center
                cur_dist = float(np.linalg.norm(eye - scene_center))
                if cur_dist < min_dist:
                    eye = scene_center + vec / max(vec_norm, 1e-12) * min_dist
                    cur_dist = min_dist
                cam_s = _build_cover_camera_from_eye(
                    eye_world=eye,
                    target_world=scene_center,
                    points_xyz=xyz_np,
                    image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
                    device=target_rgb.device,
                    margin_ratio=float(args.cover_camera_margin_ratio),
                )
                rr_s, ra_s, _ = renderer._render_cuda(
                    centers=gauss["centers"],
                    opacity=gauss["opacity"],
                    scale=gauss["scale"],
                    rotation=gauss["rotation"],
                    rgb=gauss["rgb"],
                    cam=cam_s,
                    image_hw=(int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
                )
                a = ra_s.detach().cpu().numpy().squeeze(0).astype(np.float32)
                amax = float(np.maximum(a, 0.0).max())
                athr = max(0.01, amax * 0.1)
                acov = float((a > athr).mean())
                proj_s = _project_with_camera_np(
                    cam_s,
                    xyz_np,
                    (int(rendered_rgb.shape[-2]), int(rendered_rgb.shape[-1])),
                )
                rec = {
                    "index": int(si),
                    "lambda": lam,
                    "distance_to_scene_center": cur_dist,
                    "distance_to_fit_center": float(np.linalg.norm(eye - fit_center)),
                    "alpha_max": amax,
                    "alpha_cov": acov,
                    "alpha_thr": float(athr),
                    "proj": proj_s,
                    "K": cam_s.K.detach().cpu().numpy().astype(np.float64).tolist(),
                    "w2c": cam_s.w2c.detach().cpu().numpy().astype(np.float64).tolist(),
                }
                sweep_records.append(rec)
                if args.save_images:
                    work_dir = Path(cfg.get("system", {}).get("work_dir", "work_dirs/sat2world_default"))
                    out_dir = work_dir / "render_check" / f"scene_{int(batch['scene_id'][0].item())}" / f"vi_{args.view_i}_vj_{args.view_j}" / "sweep_cover_to_fit"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    _save_chw_rgb(out_dir / f"sweep_{si:03d}_rgb.png", rr_s)
                    _save_chw_gray(out_dir / f"sweep_{si:03d}_alpha.png", ra_s)
        root_cause_note = "insufficient evidence"
        if cover_metrics is not None and fit_render_domain_diag is not None:
            alpha_ratio = float(cover_metrics["alpha_max"] / max(float(metrics["alpha_max"]), 1e-12))
            reproj_bad = float(metrics["center_centroid_err_px"]) > float(args.root_cause_reproj_threshold_px)
            k_sus = (
                fit_intr_diag["fx_positive"] < 0.5
                or fit_intr_diag["fy_positive"] < 0.5
                or fit_intr_diag["principal_in_frame"] < 0.5
                or abs(fit_intr_diag["skew"]) > 1e-3
            )
            if alpha_ratio >= float(args.root_cause_alpha_ratio_threshold):
                if k_sus:
                    root_cause_note = "fit camera intrinsics likely non-canonical (fx/fy sign/skew/principal)"
                elif reproj_bad:
                    root_cause_note = "fit camera geometry good on fit-grid but unstable on render domain"
                else:
                    root_cause_note = "fit camera likely convention mismatch with rasterizer despite good reprojection"
        checks.append(
            CheckResult(
                "root_cause_fit_camera_suspected",
                1.0 if root_cause_note != "insufficient evidence" else 0.0,
                0.5,
                True,
                note=root_cause_note,
            )
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
                    note="det should be close to +1",
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
                    note=f"reproj_p95={cam_diag['cam_reproj_p95_current_px']:.6f}",
                ),
                CheckResult(
                    "gsplat_convention_mismatch_suspected",
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
            if cover_rr is not None and cover_ra is not None:
                _save_chw_rgb(out_dir / "cover_cam_rendered_rgb.png", cover_rr)
                _save_chw_gray(out_dir / "cover_cam_rendered_alpha.png", cover_ra)
            if "sweep_records" in locals() and len(sweep_records) > 0:
                import json

                with open(out_dir / "sweep_cover_to_fit_report.json", "w", encoding="utf-8") as f:
                    json.dump(sweep_records, f, ensure_ascii=False, indent=2)

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

    if "conv_probe" in locals():
        print("\n-- gsplat convention probe (alpha-based) --")
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
    if "cover_metrics" in locals() and cover_metrics is not None:
        print("\n-- cover camera probe (renderability-based) --")
        print(
            f"  alpha_max={cover_metrics['alpha_max']:.6f}, "
            f"alpha_cov={cover_metrics['alpha_coverage']:.6f}, "
            f"alpha_thr={cover_metrics['alpha_thr']:.6f}"
        )
        if cover_diag is not None:
            print(
                f"  ratio_cover_vs_fit: alpha_max={cover_diag['alpha_max_ratio_cover_vs_fit']:.3e}, "
                f"alpha_cov={cover_diag['alpha_cov_ratio_cover_vs_fit']:.3e}"
            )
    if "fit_intr_diag" in locals():
        print("\n-- fit camera intrinsics diagnostics --")
        print(
            f"  fx={fit_intr_diag['fx']:.6f}, fy={fit_intr_diag['fy']:.6f}, "
            f"skew={fit_intr_diag['skew']:.6f}, cx={fit_intr_diag['cx']:.6f}, cy={fit_intr_diag['cy']:.6f}"
        )
        print(
            f"  fx/w={fit_intr_diag['fx_over_w']:.6f}, fy/h={fit_intr_diag['fy_over_h']:.6f}, "
            f"fov_x={fit_intr_diag['fov_x_deg']:.3f}deg, fov_y={fit_intr_diag['fov_y_deg']:.3f}deg, "
            f"Kcond={fit_intr_diag['k2x2_cond']:.3e}, principal_in_frame={fit_intr_diag['principal_in_frame']:.1f}"
        )
    if "fit_render_domain_diag" in locals() and fit_render_domain_diag is not None:
        print("\n-- fit camera render-domain projection diagnostics --")
        print(
            f"  pos_depth_ratio={fit_render_domain_diag['positive_depth_ratio']:.4f}, "
            f"in_frame_ratio={fit_render_domain_diag['in_frame_ratio']:.4f}, "
            f"z(p01,p50,p99)=({fit_render_domain_diag['z_p01']:.4f}, {fit_render_domain_diag['z_p50']:.4f}, {fit_render_domain_diag['z_p99']:.4f}), "
            f"uv_bbox=({fit_render_domain_diag['uv_bbox_w']:.2f}, {fit_render_domain_diag['uv_bbox_h']:.2f})"
        )
    if "cover_render_domain_diag" in locals() and cover_render_domain_diag is not None:
        print("\n-- cover camera render-domain projection diagnostics --")
        print(
            f"  pos_depth_ratio={cover_render_domain_diag['positive_depth_ratio']:.4f}, "
            f"in_frame_ratio={cover_render_domain_diag['in_frame_ratio']:.4f}, "
            f"z(p01,p50,p99)=({cover_render_domain_diag['z_p01']:.4f}, {cover_render_domain_diag['z_p50']:.4f}, {cover_render_domain_diag['z_p99']:.4f}), "
            f"uv_bbox=({cover_render_domain_diag['uv_bbox_w']:.2f}, {cover_render_domain_diag['uv_bbox_h']:.2f})"
        )
    if "k_ablation_diag" in locals() and k_ablation_diag is not None:
        print("\n-- fit camera K-ablation render probe --")
        for name, d in k_ablation_diag.items():
            print(
                f"  {name:16s}: alpha_max={d['alpha_max']:.6f}, "
                f"alpha_cov={d['alpha_cov']:.6f}, alpha_thr={d['alpha_thr']:.6f}"
            )
    if "sweep_records" in locals() and len(sweep_records) > 0:
        print("\n-- cover->fit sweep probe --")
        first_fail = None
        for r in sweep_records:
            if first_fail is None and (r["alpha_max"] < args.alpha_max_min_threshold or r["alpha_cov"] < args.alpha_coverage_min_threshold):
                first_fail = r
            print(
                f"  idx={r['index']:02d} lam={r['lambda']:.3f} "
                f"d_scene={r['distance_to_scene_center']:.2f} d_fit={r['distance_to_fit_center']:.2f} "
                f"alpha_max={r['alpha_max']:.6f} alpha_cov={r['alpha_cov']:.6f} "
                f"in_frame={r['proj']['in_frame_ratio']:.4f} z50={r['proj']['z_p50']:.3f}"
            )
        if first_fail is None:
            print("  first_fail: none")
        else:
            print(
                "  first_fail: "
                f"idx={first_fail['index']} lam={first_fail['lambda']:.3f} "
                f"alpha_max={first_fail['alpha_max']:.6f} alpha_cov={first_fail['alpha_cov']:.6f}"
            )

    print("------------------------------------------------------")
    print("RESULT:", "PASS" if all_pass else "FAIL")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""RPC -> pinhole 相机拟合"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .rpc_gaussian_renderer import VirtualPinholeCamera


@dataclass
class RPC2PinholeFitCfg:
    downsample: int = 8
    num_heights: int = 100
    height_low: float = -20.0
    height_high: float = 80.0
    focal: float = 1.0e5
    use_ransac: bool = True
    seed: int = 42


def _build_image_grid_with_heights(
    h: int,
    w: int,
    cfg: RPC2PinholeFitCfg,
    height_off: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ny = h // max(int(cfg.downsample), 1)
    nx = w // max(int(cfg.downsample), 1)
    if ny <= 0 or nx <= 0:
        raise ValueError(f"invalid downsample={cfg.downsample} for image size {h}x{w}")
    lines_1d = torch.arange(ny, device=device, dtype=torch.double) * float(cfg.downsample) + (float(cfg.downsample) * 0.5 - 0.5)
    samps_1d = torch.arange(nx, device=device, dtype=torch.double) * float(cfg.downsample) + (float(cfg.downsample) * 0.5 - 0.5)
    line_2d, samp_2d = torch.meshgrid(lines_1d, samps_1d, indexing="ij")
    dh = torch.linspace(cfg.height_low, cfg.height_high, steps=int(cfg.num_heights) + 1, device=device, dtype=torch.double)[1:]
    h_1d = float(height_off) + dh
    line_3d = line_2d.unsqueeze(-1).expand(ny, nx, int(cfg.num_heights)).contiguous()
    samp_3d = samp_2d.unsqueeze(-1).expand(ny, nx, int(cfg.num_heights)).contiguous()
    h_3d = h_1d.view(1, 1, int(cfg.num_heights)).expand(ny, nx, int(cfg.num_heights)).contiguous()
    return line_3d, samp_3d, h_3d


def _rpc_linesamp_to_world(
    rpc: Any,
    line_3d: torch.Tensor,
    samp_3d: torch.Tensor,
    h_3d: torch.Tensor,
    scene_center_yx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = line_3d.shape
    l = line_3d.reshape(-1).to(dtype=torch.double, device=rpc.device)
    s = samp_3d.reshape(-1).to(dtype=torch.double, device=rpc.device)
    hh = h_3d.reshape(-1).to(dtype=torch.double, device=rpc.device)
    center = scene_center_yx.to(dtype=torch.double, device=rpc.device)
    scale = torch.ones_like(center)
    x, y = rpc.RPC_LINESAMP2XY(line_in=l, samp_in=s, h_in=hh, output_type="tensor", xy_center=center, xy_scale=scale)
    return x.view(*shape), y.view(*shape), hh.view(*shape)



def _build_correspondence(
    line_3d: torch.Tensor,
    samp_3d: torch.Tensor,
    world1_x: torch.Tensor,
    world1_y: torch.Tensor,
    world1_h: torch.Tensor,
    image_h: int,
    image_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    in_frame = (
        torch.isfinite(line_3d)
        & torch.isfinite(samp_3d)
        & (line_3d >= 0.0)
        & (line_3d < float(image_h))
        & (samp_3d >= 0.0)
        & (samp_3d < float(image_w))
    )
    finite_world = torch.isfinite(world1_x) & torch.isfinite(world1_y) & torch.isfinite(world1_h)
    valid = in_frame & finite_world
    obj = torch.stack([world1_x[valid], world1_y[valid], world1_h[valid]], dim=-1)
    img = torch.stack([samp_3d[valid], line_3d[valid]], dim=-1)
    return obj, img



def _project_pinhole(K: np.ndarray, R: np.ndarray, t: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = obj.shape[0]
    xyz_h = np.concatenate([obj.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    w2c_3x4 = np.concatenate([R, t], axis=1)
    cam = (w2c_3x4 @ xyz_h.T).T
    z = cam[:, 2]
    uvw = (K @ cam.T).T
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1.0e-12) & np.isfinite(uvw).all(axis=1)
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, z


def _reproj_metrics(gt_uv: np.ndarray, pd_uv: np.ndarray, z: np.ndarray | None = None) -> dict[str, float]:
    m = np.isfinite(gt_uv).all(axis=1) & np.isfinite(pd_uv).all(axis=1)
    if z is not None:
        m = m & np.isfinite(z)
    if not m.any():
        return {"count": 0, "p50": float("inf"), "p95": float("inf"), "max": float("inf")}
    e = np.linalg.norm(pd_uv[m] - gt_uv[m], axis=1)
    return {"count": int(e.shape[0]), "p50": float(np.quantile(e, 0.5)), "p95": float(np.quantile(e, 0.95)), "max": float(np.max(e))}


def _fit_pnp(obj_pts: torch.Tensor, img_pts: torch.Tensor, image_h: int, image_w: int, cfg: RPC2PinholeFitCfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], dict[str, float], np.ndarray]:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV(cv2) is required for RPC2Pinhole fitting") from e

    obj = obj_pts.detach().cpu().numpy().astype(np.float64)
    img = img_pts.detach().cpu().numpy().astype(np.float64)
    if obj.shape[0] < 6:
        raise RuntimeError(f"PnP requires >=6 correspondences, got {obj.shape[0]}")

    cx = (float(image_w) - 1.0) * 0.5
    cy = (float(image_h) - 1.0) * 0.5
    K = np.array([[float(cfg.focal), 0.0, cx], [0.0, float(cfg.focal), cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    cv2.setRNGSeed(int(cfg.seed))

    if bool(cfg.use_ransac):
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=4.0,
            confidence=0.999,
            iterationsCount=200,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(objectPoints=obj, imagePoints=img, cameraMatrix=K, distCoeffs=dist, flags=cv2.SOLVEPNP_EPNP)
        inliers = np.arange(obj.shape[0], dtype=np.int32).reshape(-1, 1) if ok else None

    if not ok or inliers is None or inliers.size == 0:
        raise RuntimeError("solvePnP/solvePnPRansac failed")

    inlier_mask = np.zeros((obj.shape[0],), dtype=bool)
    inlier_mask[inliers.reshape(-1).astype(np.int64)] = True

    R0, _ = cv2.Rodrigues(rvec)
    uv0, z0 = _project_pinhole(K, R0, tvec.reshape(3, 1), obj)
    init_m = _reproj_metrics(img, uv0, z0)

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=obj[inlier_mask],
            imagePoints=img[inlier_mask],
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec,
            tvec=tvec,
        )
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1).astype(np.float64)
    uv, z = _project_pinhole(K, R, t, obj)
    ref_m = _reproj_metrics(img, uv, z)
    return K, R, t, init_m, ref_m, inlier_mask


def fit_view_pinhole_from_rpc(
    batch: dict[str, Any],
    bi: int,
    tv: int,
    image_hw: tuple[int, int],
    cfg: RPC2PinholeFitCfg | None = None,
) -> VirtualPinholeCamera:
    """对单 view 按 rpc2pinhole 方案拟合 pinhole 相机。"""
    cfg = cfg or RPC2PinholeFitCfg()
    h, w = image_hw
    rpc = batch["rpc_gt"][bi][tv]
    scene_center = batch["scene_xy_center"][bi].to(torch.double)
    dev = rpc.device

    height_off = float(rpc.HEIGHT_OFF.item())
    line_3d, samp_3d, h_3d = _build_image_grid_with_heights(h=h, w=w, cfg=cfg, height_off=height_off, device=dev)
    world1_x, world1_y, world1_h = _rpc_linesamp_to_world(
        rpc=rpc,
        line_3d=line_3d,
        samp_3d=samp_3d,
        h_3d=h_3d,
        scene_center_yx=scene_center,
    )
    obj, img = _build_correspondence(
        line_3d=line_3d,
        samp_3d=samp_3d,
        world1_x=world1_x,
        world1_y=world1_y,
        world1_h=world1_h,
        image_h=h,
        image_w=w,
    )

    K, R, t, init_m, ref_m, inlier_mask = _fit_pnp(obj, img, image_h=h, image_w=w, cfg=cfg)

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t
    diag = {
        "method": "rpc2pinhole",
        "correspondence": int(obj.shape[0]),
        "inlier_count": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "init_metrics": init_m,
        "refined_metrics": ref_m,
        "fit_cfg": {
            "downsample": int(cfg.downsample),
            "num_heights": int(cfg.num_heights),
            "height_low": float(cfg.height_low),
            "height_high": float(cfg.height_high),
            "focal": float(cfg.focal),
            "use_ransac": bool(cfg.use_ransac),
            "seed": int(cfg.seed),
        },
    }
    return VirtualPinholeCamera(
        K=torch.from_numpy(K).to(dtype=torch.float32, device=dev),
        w2c=torch.from_numpy(w2c).to(dtype=torch.float32, device=dev),
        fit_p50=float(ref_m["p50"]),
        fit_p95=float(ref_m["p95"]),
        fit_max=float(ref_m["max"]),
        diagnostics=diag,
    )


def format_rpc2_fit_diag(cam: VirtualPinholeCamera) -> str:
    d = cam.diagnostics or {}
    return json.dumps(d, ensure_ascii=False, indent=2)


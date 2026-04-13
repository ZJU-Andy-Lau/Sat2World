"""RPC 拟合针孔相机可行性实验脚本（严格按实验步骤实现）。

实验流程：
1) 读取 config，加载 val 数据集中的一个 scene 与一个 view。
2) 构建像方下采样网格，并为每个网格点分配多高度。
3) 通过 RPC 反投影得到第一部分物方网格（使用 scene_center，scale=1）。
4) 根据第一部分网格确定 xy 范围，构建第二部分物方网格并投回像方，筛选有效 3D-2D 对应。
5) 固定 K（skew=0, cx/cy=center, f=1e5），通过 PnP 估计外参。
6) 随机 100 点误差测试：RPC->3D->拟合针孔回投影。
7) 全像素+h_gt 生成高斯并通过 gsplat 渲染，输出指标与图像。

注意：本脚本不复用工程中已有相机拟合代码。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataset, rpc_scene_collate_fn
from engine.distributed import move_batch_to_device


@dataclass
class PnPResult:
    K: np.ndarray  # [3,3]
    R: np.ndarray  # [3,3]
    t: np.ndarray  # [3,1]
    w2c: np.ndarray  # [4,4]
    inlier_mask: np.ndarray  # [N]
    init_metrics: dict[str, float]
    refined_metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World RPC->Pinhole feasibility test")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--view-k", type=int, default=0)
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--num-heights", type=int, default=100)
    p.add_argument("--height-low", type=float, default=-20.0)
    p.add_argument("--height-high", type=float, default=80.0)
    p.add_argument("--random-test-points", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_val_batch(cfg: dict[str, Any], scene_index: int, device: torch.device) -> dict[str, Any]:
    ds = build_dataset(mode="val", **cfg.get("data", {}).get("val", {}))
    if len(ds) == 0:
        raise RuntimeError("val dataset empty")
    if not (0 <= scene_index < len(ds)):
        raise ValueError(f"scene_index out of range: {scene_index}/{len(ds)}")
    sample = ds[scene_index]
    batch = rpc_scene_collate_fn([sample])
    batch = move_batch_to_device(batch, device)
    return batch


def _stats_t(name: str, t: torch.Tensor) -> str:
    td = t.detach().to(torch.float64)
    fin = torch.isfinite(td)
    ratio = float(fin.to(torch.float32).mean().item())
    if bool(fin.any()):
        vals = td[fin]
        return (
            f"{name}: shape={tuple(t.shape)} finite_ratio={ratio:.6f} "
            f"min={float(vals.min().item()):.6f} max={float(vals.max().item()):.6f} "
            f"mean={float(vals.mean().item()):.6f} std={float(vals.std(unbiased=False).item()):.6f}"
        )
    return f"{name}: shape={tuple(t.shape)} finite_ratio={ratio:.6f} (all non-finite)"


def _stats_np(name: str, a: np.ndarray) -> str:
    fin = np.isfinite(a)
    ratio = float(fin.mean())
    if fin.any():
        v = a[fin]
        return (
            f"{name}: shape={a.shape} finite_ratio={ratio:.6f} "
            f"min={float(np.min(v)):.6f} max={float(np.max(v)):.6f} "
            f"mean={float(np.mean(v)):.6f} std={float(np.std(v)):.6f}"
        )
    return f"{name}: shape={a.shape} finite_ratio={ratio:.6f} (all non-finite)"


def print_view_info(batch: dict[str, Any], view_k: int) -> None:
    b, v, _, h, w = batch["images"].shape
    if b != 1:
        raise RuntimeError("Only batch_size=1 is supported")
    if not (0 <= view_k < v):
        raise ValueError(f"invalid view_k={view_k}, total views={v}")

    rpc = batch["rpc_gt"][0][view_k]
    scene_id = int(batch["scene_id"][0].item())
    view_id = int(batch["view_ids"][0, view_k].item())
    img_path = str(batch["image_paths"][0][view_k])

    print("\n========== [Step2] View 信息 ==========")
    print(f"scene_id={scene_id}, view_k={view_k}, view_id={view_id}")
    print(f"image_path={img_path}")
    print(f"image_shape={tuple(batch['images'][0, view_k].shape)}, height_shape={tuple(batch['height_gt'][0, view_k].shape)}")
    print(_stats_t("image", batch["images"][0, view_k]))
    print(_stats_t("height_gt", batch["height_gt"][0, view_k, 0]))

    if "height_valid_mask" in batch:
        vm = batch["height_valid_mask"][0, view_k, 0]
        print(_stats_t("height_valid_mask", vm))
        valid_ratio = float((vm > 0.5).to(torch.float32).mean().item())
        print(f"height_valid_ratio(>0.5)={valid_ratio:.6f}")

    scene_center = batch["scene_xy_center"][0].detach().cpu().numpy()
    scene_scale = batch["scene_xy_scale"][0].detach().cpu().numpy()
    print(f"scene_xy_center(y,x)={scene_center.tolist()}")
    print(f"scene_xy_scale(y,x)={scene_scale.tolist()}")

    print("RPC core params:")
    print(f"  LINE_OFF={float(rpc.LINE_OFF.item()):.6f}, SAMP_OFF={float(rpc.SAMP_OFF.item()):.6f}")
    print(f"  LINE_SCALE={float(rpc.LINE_SCALE.item()):.6f}, SAMP_SCALE={float(rpc.SAMP_SCALE.item()):.6f}")
    print(f"  HEIGHT_OFF={float(rpc.HEIGHT_OFF.item()):.6f}, HEIGHT_SCALE={float(rpc.HEIGHT_SCALE.item()):.6f}")


def build_image_grid_with_heights(
    h: int,
    w: int,
    downsample: int,
    num_heights: int,
    height_off: float,
    height_low: float,
    height_high: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if downsample <= 0:
        raise ValueError("downsample must be > 0")
    ny = h // downsample
    nx = w // downsample
    if ny <= 0 or nx <= 0:
        raise ValueError(f"invalid downsample={downsample} for image size {h}x{w}")

    # 网格中心采样（像素坐标 line/samp）
    lines_1d = torch.arange(ny, device=device, dtype=torch.double) * float(downsample) + (float(downsample) * 0.5 - 0.5)
    samps_1d = torch.arange(nx, device=device, dtype=torch.double) * float(downsample) + (float(downsample) * 0.5 - 0.5)
    line_2d, samp_2d = torch.meshgrid(lines_1d, samps_1d, indexing="ij")

    # 生成 (-20,80] 风格高度采样：使用 num_heights+1 等分后去掉首点
    dh = torch.linspace(height_low, height_high, steps=num_heights + 1, device=device, dtype=torch.double)[1:]
    h_1d = float(height_off) + dh

    line_3d = line_2d.unsqueeze(-1).expand(ny, nx, num_heights).contiguous()
    samp_3d = samp_2d.unsqueeze(-1).expand(ny, nx, num_heights).contiguous()
    h_3d = h_1d.view(1, 1, num_heights).expand(ny, nx, num_heights).contiguous()

    return line_3d, samp_3d, h_3d


def rpc_linesamp_to_world(
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

    x, y = rpc.RPC_LINESAMP2XY(
        line_in=l,
        samp_in=s,
        h_in=hh,
        output_type="tensor",
        xy_center=center,
        xy_scale=scale,
    )

    return x.view(*shape), y.view(*shape), hh.view(*shape)


def rpc_world_to_linesamp(
    rpc: Any,
    x_3d: torch.Tensor,
    y_3d: torch.Tensor,
    h_3d: torch.Tensor,
    scene_center_yx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x_3d.shape
    x = x_3d.reshape(-1).to(dtype=torch.double, device=rpc.device)
    y = y_3d.reshape(-1).to(dtype=torch.double, device=rpc.device)
    hh = h_3d.reshape(-1).to(dtype=torch.double, device=rpc.device)

    center = scene_center_yx.to(dtype=torch.double, device=rpc.device)
    scale = torch.ones_like(center)

    line, samp = rpc.RPC_XY2LINESAMP(
        x_in=x,
        y_in=y,
        h_in=hh,
        output_type="tensor",
        xy_center=center,
        xy_scale=scale,
    )

    return line.view(*shape), samp.view(*shape), hh.view(*shape)


def build_second_world_grid_and_correspondence(
    rpc: Any,
    world1_x: torch.Tensor,
    world1_y: torch.Tensor,
    h_3d: torch.Tensor,
    scene_center_yx: torch.Tensor,
    image_h: int,
    image_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ny, nx, nz = world1_x.shape

    finite_xy = torch.isfinite(world1_x) & torch.isfinite(world1_y)
    if not bool(finite_xy.any()):
        raise RuntimeError("world1_x/world1_y have no finite values")

    x_min = float(world1_x[finite_xy].min().item())
    x_max = float(world1_x[finite_xy].max().item())
    y_min = float(world1_y[finite_xy].min().item())
    y_max = float(world1_y[finite_xy].max().item())

    print("\n========== [Step5] 第二部分物方网格构建 ==========")
    print(f"world1 xy range: x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")

    x_1d = torch.linspace(x_min, x_max, steps=nx, device=world1_x.device, dtype=torch.double)
    y_1d = torch.linspace(y_min, y_max, steps=ny, device=world1_y.device, dtype=torch.double)
    y_2d, x_2d = torch.meshgrid(y_1d, x_1d, indexing="ij")

    world2_x = x_2d.unsqueeze(-1).expand(ny, nx, nz).contiguous()
    world2_y = y_2d.unsqueeze(-1).expand(ny, nx, nz).contiguous()
    world2_h = h_3d.clone()

    line2, samp2, _ = rpc_world_to_linesamp(
        rpc=rpc,
        x_3d=world2_x,
        y_3d=world2_y,
        h_3d=world2_h,
        scene_center_yx=scene_center_yx,
    )

    in_frame = (
        torch.isfinite(line2)
        & torch.isfinite(samp2)
        & (line2 >= 0.0)
        & (line2 < float(image_h))
        & (samp2 >= 0.0)
        & (samp2 < float(image_w))
    )

    # 3D-2D 对应关系（OpenCV: image_points=(x=samp, y=line)）
    obj_pts = torch.stack([world2_x[in_frame], world2_y[in_frame], world2_h[in_frame]], dim=-1)
    img_pts = torch.stack([samp2[in_frame], line2[in_frame]], dim=-1)

    print(_stats_t("world2_x", world2_x))
    print(_stats_t("world2_y", world2_y))
    print(_stats_t("world2_h", world2_h))
    print(_stats_t("proj_line2", line2))
    print(_stats_t("proj_samp2", samp2))
    print(f"in_frame_count={int(in_frame.sum().item())}, total={in_frame.numel()}, ratio={float(in_frame.to(torch.float32).mean().item()):.6f}")
    print(f"correspondence_count={obj_pts.shape[0]}")

    return world2_x, world2_y, world2_h, obj_pts, img_pts


def build_first_correspondence_from_image_grid(
    line_3d: torch.Tensor,
    samp_3d: torch.Tensor,
    world1_x: torch.Tensor,
    world1_y: torch.Tensor,
    world1_h: torch.Tensor,
    image_h: int,
    image_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由像方均匀网格(含多高程)直接构建第一部分 3D-2D 对应。"""
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

    obj_pts = torch.stack([world1_x[valid], world1_y[valid], world1_h[valid]], dim=-1)
    img_pts = torch.stack([samp_3d[valid], line_3d[valid]], dim=-1)  # OpenCV: (x=samp, y=line)
    return obj_pts, img_pts


def merge_correspondences(
    obj_a: torch.Tensor,
    img_a: torch.Tensor,
    obj_b: torch.Tensor,
    img_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """合并两部分对应关系，并进行基础有效性筛选。"""
    obj = torch.cat([obj_a, obj_b], dim=0)
    img = torch.cat([img_a, img_b], dim=0)
    valid = torch.isfinite(obj).all(dim=-1) & torch.isfinite(img).all(dim=-1)
    return obj[valid], img[valid]


def _project_pinhole(K: np.ndarray, R: np.ndarray, t: np.ndarray, obj_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # obj_pts: [N,3]
    n = obj_pts.shape[0]
    xyz_h = np.concatenate([obj_pts.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    w2c_3x4 = np.concatenate([R, t], axis=1)
    cam = (w2c_3x4 @ xyz_h.T).T
    z = cam[:, 2]
    uvw = (K @ cam.T).T
    uv = np.full((n, 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1.0e-12) & np.isfinite(uvw).all(axis=1)
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, z


def _reproj_metrics_np(gt_uv: np.ndarray, pd_uv: np.ndarray, depth: np.ndarray | None = None) -> dict[str, float]:
    m = np.isfinite(gt_uv).all(axis=1) & np.isfinite(pd_uv).all(axis=1)
    if depth is not None:
        m = m & np.isfinite(depth)
    if not m.any():
        return {
            "count": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "p50": float("inf"),
            "p90": float("inf"),
            "p95": float("inf"),
            "max": float("inf"),
        }
    e = np.linalg.norm(pd_uv[m] - gt_uv[m], axis=1)
    return {
        "count": int(e.shape[0]),
        "mean": float(np.mean(e)),
        "std": float(np.std(e)),
        "p50": float(np.quantile(e, 0.50)),
        "p90": float(np.quantile(e, 0.90)),
        "p95": float(np.quantile(e, 0.95)),
        "max": float(np.max(e)),
    }


def fit_pnp_extrinsic_fixed_K(
    obj_pts: torch.Tensor,
    img_pts: torch.Tensor,
    image_h: int,
    image_w: int,
    *,
    f: float = 1.0e5,
    use_ransac: bool = True,
    seed: int = 42,
) -> PnPResult:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("需要 OpenCV(cv2) 才能进行 PnP 估计") from e

    obj = obj_pts.detach().cpu().numpy().astype(np.float64)
    img = img_pts.detach().cpu().numpy().astype(np.float64)
    n = obj.shape[0]
    if n < 6:
        raise RuntimeError(f"PnP requires at least 6 correspondences, got {n}")

    cx = (float(image_w) - 1.0) * 0.5
    cy = (float(image_h) - 1.0) * 0.5
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    print("\n========== [Step6] PnP 外参估计（固定内参） ==========")
    print(f"K=\n{K}")
    print(f"num_correspondences={n}, image_hw=({image_h},{image_w})")

    cv2.setRNGSeed(int(seed))

    if use_ransac:
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
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_EPNP,
        )
        inliers = np.arange(n, dtype=np.int32).reshape(-1, 1) if ok else None

    if not ok or inliers is None or inliers.size == 0:
        raise RuntimeError("solvePnPRansac failed to find a valid solution")

    inlier_mask = np.zeros((n,), dtype=bool)
    inlier_ids = inliers.reshape(-1).astype(np.int64)
    inlier_mask[inlier_ids] = True

    R0, _ = cv2.Rodrigues(rvec)
    uv0, z0 = _project_pinhole(K, R0, tvec.reshape(3, 1), obj)
    init_metrics = _reproj_metrics_np(gt_uv=img, pd_uv=uv0, depth=z0)

    print(f"RANSAC inliers: {int(inlier_mask.sum())}/{n} ({float(inlier_mask.mean()):.6f})")
    print(f"Initial reproj metrics: {json.dumps(init_metrics, ensure_ascii=False, indent=2)}")

    # LM 精化（仅 inlier）
    obj_in = obj[inlier_mask]
    img_in = img[inlier_mask]
    rvec_ref = rvec.copy()
    tvec_ref = tvec.copy()

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
            objectPoints=obj_in,
            imagePoints=img_in,
            cameraMatrix=K,
            distCoeffs=dist,
            rvec=rvec_ref,
            tvec=tvec_ref,
        )

    R, _ = cv2.Rodrigues(rvec_ref)
    t = tvec_ref.reshape(3, 1).astype(np.float64)

    uv_ref, z_ref = _project_pinhole(K, R, t, obj)
    refined_metrics = _reproj_metrics_np(gt_uv=img, pd_uv=uv_ref, depth=z_ref)

    z_pos_ratio = float(np.mean(z_ref > 1.0e-8))
    c_world = -R.T @ t

    print(f"Refined reproj metrics: {json.dumps(refined_metrics, ensure_ascii=False, indent=2)}")
    print(f"positive_depth_ratio={z_pos_ratio:.6f}")
    print(f"R=\n{R}")
    print(f"t=\n{t}")
    print(f"camera_center_world=\n{c_world}")

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3:] = t

    return PnPResult(
        K=K,
        R=R,
        t=t,
        w2c=w2c,
        inlier_mask=inlier_mask,
        init_metrics=init_metrics,
        refined_metrics=refined_metrics,
    )


def random_100_reprojection_test(
    rpc: Any,
    scene_center_yx: torch.Tensor,
    pnp: PnPResult,
    image_h: int,
    image_w: int,
    height_off: float,
    height_low: float,
    height_high: float,
    n_points: int,
    seed: int,
) -> dict[str, float]:
    print("\n========== [Step7] 随机100点重投影误差测试 ==========")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    lines = torch.randint(low=0, high=image_h, size=(n_points,), generator=g, dtype=torch.long).to(torch.double)
    samps = torch.randint(low=0, high=image_w, size=(n_points,), generator=g, dtype=torch.long).to(torch.double)
    hs = float(height_off) + (torch.rand((n_points,), generator=g, dtype=torch.float64) * (height_high - height_low) + height_low)

    x, y, h = rpc_linesamp_to_world(
        rpc=rpc,
        line_3d=lines.view(-1, 1, 1),
        samp_3d=samps.view(-1, 1, 1),
        h_3d=hs.view(-1, 1, 1),
        scene_center_yx=scene_center_yx,
    )

    obj = torch.stack([x.view(-1), y.view(-1), h.view(-1)], dim=-1).detach().cpu().numpy().astype(np.float64)
    gt_uv = np.stack([samps.detach().cpu().numpy(), lines.detach().cpu().numpy()], axis=-1)

    pd_uv, z = _project_pinhole(pnp.K, pnp.R, pnp.t, obj)
    metrics = _reproj_metrics_np(gt_uv=gt_uv, pd_uv=pd_uv, depth=z)

    print(_stats_np("random_obj_x", obj[:, 0]))
    print(_stats_np("random_obj_y", obj[:, 1]))
    print(_stats_np("random_obj_h", obj[:, 2]))
    print(_stats_np("random_pred_u", pd_uv[:, 0]))
    print(_stats_np("random_pred_v", pd_uv[:, 1]))
    print(_stats_np("random_depth", z))
    print(f"random reproj metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")

    if metrics["count"] > 0:
        e = np.linalg.norm(pd_uv - gt_uv, axis=1)
        order = np.argsort(-e)
        topk = order[: min(10, order.shape[0])]
        print("Top worst samples (idx, line_gt, samp_gt, h, line_pd, samp_pd, err):")
        for idx in topk.tolist():
            print(
                f"  idx={idx:03d} line_gt={gt_uv[idx,1]:.3f} samp_gt={gt_uv[idx,0]:.3f} "
                f"h={obj[idx,2]:.3f} line_pd={pd_uv[idx,1]:.3f} samp_pd={pd_uv[idx,0]:.3f} err={e[idx]:.6f}"
            )

    return metrics


def save_rgb_tensor(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img_u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def save_gray_tensor(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().squeeze(0).numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = max(float(arr.max()), 1.0e-12)
    arr01 = np.clip(arr / vmax, 0.0, 1.0)
    img_u8 = np.clip(np.round(arr01 * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def gsplat_render_full_view(
    batch: dict[str, Any],
    view_k: int,
    rpc: Any,
    scene_center_yx: torch.Tensor,
    pnp: PnPResult,
    out_dir: Path,
) -> dict[str, float]:
    print("\n========== [Step8] 全像素高斯构建 + gsplat渲染 ==========")

    try:
        from gsplat import rasterization  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("未安装 gsplat，无法执行步骤8") from e

    device = batch["images"].device
    target_rgb = batch["images"][0, view_k].to(torch.float32)
    h_gt = batch["height_gt"][0, view_k, 0].to(torch.float64)
    _, h, w = target_rgb.shape

    t0 = time.perf_counter()

    # 全像素像方点
    lines = torch.arange(h, device=device, dtype=torch.double).view(-1, 1).expand(h, w).reshape(-1)
    samps = torch.arange(w, device=device, dtype=torch.double).view(1, -1).expand(h, w).reshape(-1)
    hs = h_gt.reshape(-1).to(torch.double)

    x, y, hh = rpc_linesamp_to_world(
        rpc=rpc,
        line_3d=lines.view(-1, 1, 1),
        samp_3d=samps.view(-1, 1, 1),
        h_3d=hs.view(-1, 1, 1),
        scene_center_yx=scene_center_yx,
    )

    means = torch.stack([x.view(-1), y.view(-1), hh.view(-1)], dim=-1).to(torch.float32)

    n = means.shape[0]
    scales = torch.full((n, 3), 0.5, dtype=torch.float32, device=device)
    rots = torch.zeros((n, 4), dtype=torch.float32, device=device)
    rots[:, 0] = 1.0
    opacities = torch.ones((n,), dtype=torch.float32, device=device)
    colors = target_rgb.permute(1, 2, 0).reshape(-1, 3).to(torch.float32)

    K_t = torch.from_numpy(pnp.K).to(dtype=torch.float32, device=device).unsqueeze(0)
    w2c_t = torch.from_numpy(pnp.w2c).to(dtype=torch.float32, device=device).unsqueeze(0)
    bg = torch.zeros((1, 3), dtype=torch.float32, device=device)

    print(_stats_t("means_x", means[:, 0]))
    print(_stats_t("means_y", means[:, 1]))
    print(_stats_t("means_h", means[:, 2]))
    print(_stats_t("gaussian_colors", colors))
    print(f"num_gaussians={n}")

    rendering, alpha, _aux = rasterization(
        means,
        rots,
        scales,
        opacities,
        colors,
        w2c_t,
        K_t,
        w,
        h,
        sh_degree=None,
        render_mode="RGB+D",
        packed=False,
        near_plane=1e-6,
        backgrounds=bg,
        rasterize_mode="classic",
    )

    render_rgb = rendering[0, ..., :3].permute(2, 0, 1).contiguous()
    render_depth = rendering[0, ..., 3:4].permute(2, 0, 1).contiguous()
    render_alpha = alpha[0].permute(2, 0, 1).contiguous() if alpha.ndim == 4 else alpha[0].unsqueeze(0)

    elapsed = time.perf_counter() - t0

    # 评估指标
    diff = (render_rgb - target_rgb).to(torch.float64)
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1.0e-12)))
    alpha_cov = float((render_alpha > 1.0e-4).to(torch.float32).mean().item())
    alpha_mean = float(render_alpha.mean().item())
    alpha_max = float(render_alpha.max().item())

    out_dir.mkdir(parents=True, exist_ok=True)
    save_rgb_tensor(out_dir / "target_rgb.png", target_rgb)
    save_rgb_tensor(out_dir / "render_rgb.png", render_rgb)
    save_gray_tensor(out_dir / "render_alpha.png", render_alpha)
    save_gray_tensor(out_dir / "render_depth_vis.png", render_depth)
    np.save(out_dir / "render_depth.npy", render_depth.detach().cpu().numpy())

    metrics = {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "alpha_cov@1e-4": alpha_cov,
        "alpha_mean": alpha_mean,
        "alpha_max": alpha_max,
        "render_time_sec": float(elapsed),
        "num_gaussians": int(n),
    }

    print(_stats_t("render_rgb", render_rgb))
    print(_stats_t("render_alpha", render_alpha))
    print(_stats_t("render_depth", render_depth))
    print(f"render metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")
    print(f"images saved to: {out_dir}")

    return metrics


def run_internal_audit(
    args: argparse.Namespace,
    batch: dict[str, Any],
    view_k: int,
    line_3d: torch.Tensor,
    samp_3d: torch.Tensor,
    h_3d: torch.Tensor,
    world1_x: torch.Tensor,
    world1_y: torch.Tensor,
    obj_pts_part1: torch.Tensor,
    img_pts_part1: torch.Tensor,
    obj_pts_part2: torch.Tensor,
    img_pts_part2: torch.Tensor,
    obj_pts: torch.Tensor,
    img_pts: torch.Tensor,
    pnp: PnPResult,
    random_metrics: dict[str, float],
    render_metrics: dict[str, float],
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    _, _, _, ih, iw = batch["images"].shape
    ny, nx, nz = line_3d.shape

    checks.append((
        "Step3 grid size",
        (ny == ih // args.downsample and nx == iw // args.downsample and nz == args.num_heights),
        f"grid=({ny},{nx},{nz}), expected=({ih // args.downsample},{iw // args.downsample},{args.num_heights})",
    ))

    checks.append((
        "Step4 world finite ratio",
        bool(torch.isfinite(world1_x).any() and torch.isfinite(world1_y).any()),
        f"world1 finite ratio x={float(torch.isfinite(world1_x).to(torch.float32).mean().item()):.6f}, "
        f"y={float(torch.isfinite(world1_y).to(torch.float32).mean().item()):.6f}",
    ))

    checks.append((
        "Step4 part1 correspondences",
        obj_pts_part1.shape[0] >= 6 and img_pts_part1.shape[0] == obj_pts_part1.shape[0],
        f"part1 obj={obj_pts_part1.shape}, img={img_pts_part1.shape}",
    ))

    checks.append((
        "Step5 part2 correspondences",
        obj_pts_part2.shape[0] >= 6 and img_pts_part2.shape[0] == obj_pts_part2.shape[0],
        f"part2 obj={obj_pts_part2.shape}, img={img_pts_part2.shape}",
    ))

    checks.append((
        "Step5 correspondences",
        obj_pts.shape[0] >= 6 and img_pts.shape[0] == obj_pts.shape[0],
        f"obj_pts={obj_pts.shape}, img_pts={img_pts.shape}",
    ))

    checks.append((
        "Step6 pnp inlier",
        bool(np.sum(pnp.inlier_mask) >= 6),
        f"inliers={int(np.sum(pnp.inlier_mask))}, total={int(pnp.inlier_mask.shape[0])}",
    ))

    checks.append((
        "Step7 random reproj count",
        int(random_metrics.get("count", 0)) > 0,
        json.dumps(random_metrics, ensure_ascii=False),
    ))

    checks.append((
        "Step8 render finite",
        np.isfinite(render_metrics["mse"]) and np.isfinite(render_metrics["psnr"]),
        json.dumps(render_metrics, ensure_ascii=False),
    ))

    return checks


def main() -> None:
    args = parse_args()
    torch.set_printoptions(precision=6, sci_mode=False)
    np.set_printoptions(precision=6, suppress=True)

    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    print("========== RPC2Pinhole Test: Start ==========")
    print(f"args={args}")
    print(f"device={device}")

    batch = build_val_batch(cfg, args.scene_index, device)

    b, v, _, h, w = batch["images"].shape
    if b != 1:
        raise RuntimeError(f"only batch_size=1 supported, got {b}")
    if not (0 <= args.view_k < v):
        raise ValueError(f"invalid view_k={args.view_k}, V={v}")

    rpc = batch["rpc_gt"][0][args.view_k]
    scene_center_yx = batch["scene_xy_center"][0].to(torch.double)
    height_off = float(rpc.HEIGHT_OFF.item())

    print_view_info(batch, args.view_k)

    # Step3: 像方下采样网格 + 高度层
    print("\n========== [Step3] 像方网格构建 ==========")
    line_3d, samp_3d, h_3d = build_image_grid_with_heights(
        h=h,
        w=w,
        downsample=args.downsample,
        num_heights=args.num_heights,
        height_off=height_off,
        height_low=args.height_low,
        height_high=args.height_high,
        device=device,
    )
    print(_stats_t("line_3d", line_3d))
    print(_stats_t("samp_3d", samp_3d))
    print(_stats_t("h_3d", h_3d))
    print(f"grid_shape(line/samp/h)={tuple(line_3d.shape)}")

    # Step4: 第一部分物方网格
    print("\n========== [Step4] RPC反投影到物方网格（中心化+scale=1） ==========")
    world1_x, world1_y, world1_h = rpc_linesamp_to_world(
        rpc=rpc,
        line_3d=line_3d,
        samp_3d=samp_3d,
        h_3d=h_3d,
        scene_center_yx=scene_center_yx,
    )
    print(_stats_t("world1_x", world1_x))
    print(_stats_t("world1_y", world1_y))
    print(_stats_t("world1_h", world1_h))

    # Step5: 第二部分物方网格 + 3D-2D 对应
    obj_pts_part1, img_pts_part1 = build_first_correspondence_from_image_grid(
        line_3d=line_3d,
        samp_3d=samp_3d,
        world1_x=world1_x,
        world1_y=world1_y,
        world1_h=world1_h,
        image_h=h,
        image_w=w,
    )
    print("\n========== [Step4+] 第一部分对应关系统计（像方均匀网格 -> 物方） ==========")
    print(f"part1 correspondence_count={obj_pts_part1.shape[0]}")
    print(_stats_t("part1_obj_x", obj_pts_part1[:, 0] if obj_pts_part1.numel() > 0 else torch.empty((0,), device=device)))
    print(_stats_t("part1_obj_y", obj_pts_part1[:, 1] if obj_pts_part1.numel() > 0 else torch.empty((0,), device=device)))
    print(_stats_t("part1_obj_h", obj_pts_part1[:, 2] if obj_pts_part1.numel() > 0 else torch.empty((0,), device=device)))

    _world2_x, _world2_y, _world2_h, obj_pts_part2, img_pts_part2 = build_second_world_grid_and_correspondence(
        rpc=rpc,
        world1_x=world1_x,
        world1_y=world1_y,
        h_3d=h_3d,
        scene_center_yx=scene_center_yx,
        image_h=h,
        image_w=w,
    )
    obj_pts, img_pts = merge_correspondences(
        obj_a=obj_pts_part1,
        img_a=img_pts_part1,
        obj_b=obj_pts_part2,
        img_b=img_pts_part2,
    )
    print("\n========== [Step5+] 合并后对应关系统计（用于PnP） ==========")
    print(f"merged correspondence_count={obj_pts.shape[0]} (part1={obj_pts_part1.shape[0]}, part2={obj_pts_part2.shape[0]})")
    print(_stats_t("merged_obj_x", obj_pts[:, 0]))
    print(_stats_t("merged_obj_y", obj_pts[:, 1]))
    print(_stats_t("merged_obj_h", obj_pts[:, 2]))

    # Step6: PnP
    pnp = fit_pnp_extrinsic_fixed_K(
        obj_pts=obj_pts,
        img_pts=img_pts,
        image_h=h,
        image_w=w,
        f=1.0e5,
        use_ransac=True,
        seed=args.seed,
    )

    # Step7: 随机100点误差测试
    random_metrics = random_100_reprojection_test(
        rpc=rpc,
        scene_center_yx=scene_center_yx,
        pnp=pnp,
        image_h=h,
        image_w=w,
        height_off=height_off,
        height_low=args.height_low,
        height_high=args.height_high,
        n_points=args.random_test_points,
        seed=args.seed + 2026,
    )

    # Step8: 全像素点云+gsplat渲染
    work_dir = Path(args.work_dir) if args.work_dir else Path(cfg.get("system", {}).get("work_dir", "work_dirs"))
    out_dir = work_dir / "rpc2pinhole_test" / f"scene_{int(batch['scene_id'][0].item())}" / f"view_{int(batch['view_ids'][0, args.view_k].item())}"

    render_metrics = gsplat_render_full_view(
        batch=batch,
        view_k=args.view_k,
        rpc=rpc,
        scene_center_yx=scene_center_yx,
        pnp=pnp,
        out_dir=out_dir,
    )

    summary = {
        "scene_id": int(batch["scene_id"][0].item()),
        "view_k": int(args.view_k),
        "view_id": int(batch["view_ids"][0, args.view_k].item()),
        "image_hw": [int(h), int(w)],
        "grid_shape": [int(x) for x in line_3d.shape],
        "pnp_init_metrics": pnp.init_metrics,
        "pnp_refined_metrics": pnp.refined_metrics,
        "pnp_inliers": int(np.sum(pnp.inlier_mask)),
        "pnp_total_correspondences": int(pnp.inlier_mask.shape[0]),
        "random_test_metrics": random_metrics,
        "render_metrics": render_metrics,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 运行后自审（脚本内部）
    checks = run_internal_audit(
        args=args,
        batch=batch,
        view_k=args.view_k,
        line_3d=line_3d,
        samp_3d=samp_3d,
        h_3d=h_3d,
        world1_x=world1_x,
        world1_y=world1_y,
        obj_pts_part1=obj_pts_part1,
        img_pts_part1=img_pts_part1,
        obj_pts_part2=obj_pts_part2,
        img_pts_part2=img_pts_part2,
        obj_pts=obj_pts,
        img_pts=img_pts,
        pnp=pnp,
        random_metrics=random_metrics,
        render_metrics=render_metrics,
    )

    print("\n========== [Internal Audit] 方案落实检查 ==========")
    num_pass = 0
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        num_pass += int(ok)
    print(f"Internal audit: {num_pass}/{len(checks)} passed")

    print("\n========== RPC2Pinhole Test: Done ==========")
    print(f"summary saved to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

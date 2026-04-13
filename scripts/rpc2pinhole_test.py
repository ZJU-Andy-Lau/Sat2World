#!/usr/bin/env python3
"""RPC -> Pinhole feasibility experiment script.

严格按照用户实验流程：
1) 读取 config 与 val scene/view；
2) 构建像方 64x64x100 网格（默认 8x 下采样 + 100 高度层）；
3) RPC 反投影到物方（scene_center 中心化，scale=1）；
4) 固定 skew=0、cx/cy 为中心，拟合 pinhole 其余参数；
5) 随机100点闭环重投影测试；
6) 全像素+h_gt 生成点云高斯并通过 gsplat 渲染；
7) 输出详细日志与图像到 work_dir。
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.spatial.transform
import torch
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataset


@dataclass
class FitResult:
    K: np.ndarray
    w2c: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    rmse: float
    p50: float
    p95: float
    p99: float
    pmax: float
    positive_depth_ratio: float
    diagnostics: dict[str, Any]


class FitLogger:
    def __init__(self, print_every: int = 20) -> None:
        self.eval_count = 0
        self.print_every = max(int(print_every), 1)
        self.history: list[dict[str, float]] = []

    def add(self, metrics: dict[str, float]) -> None:
        self.eval_count += 1
        rec = {"eval": float(self.eval_count)}
        rec.update(metrics)
        self.history.append(rec)
        if self.eval_count == 1 or self.eval_count % self.print_every == 0:
            print(
                "[fit][eval={:04d}] rmse={:.6f} p50={:.6f} p95={:.6f} p99={:.6f} max={:.6f} pos_depth={:.4f}".format(
                    self.eval_count,
                    metrics["rmse"],
                    metrics["p50"],
                    metrics["p95"],
                    metrics["p99"],
                    metrics["pmax"],
                    metrics["pos_ratio"],
                ),
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RPC->Pinhole feasibility test")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--scene-id", type=int, default=None)
    p.add_argument("--view-index", type=int, default=0)
    p.add_argument("--grid-downsample", type=int, default=8)
    p.add_argument("--num-heights", type=int, default=100)
    p.add_argument("--height-low", type=float, default=-20.0)
    p.add_argument("--height-high", type=float, default=80.0)
    p.add_argument("--num-random-test-points", type=int, default=100)
    p.add_argument("--fit-max-nfev", type=int, default=300)
    p.add_argument("--fit-print-every", type=int, default=20)
    p.add_argument("--f-fixed", type=float, default=1.0e5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--work-dir", type=str, default="")
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick_val_sample(cfg: dict[str, Any], scene_index: int, scene_id: int | None) -> tuple[dict[str, Any], int]:
    val_cfg = cfg.get("data", {}).get("val", {})
    if not isinstance(val_cfg, dict) or len(val_cfg) == 0:
        raise RuntimeError("config 缺少 data.val")

    ds = build_dataset(mode="val", **val_cfg)
    if len(ds) == 0:
        raise RuntimeError("val dataset 为空")

    idx = int(scene_index)
    if scene_id is not None:
        idx = -1
        for i, rec in enumerate(getattr(ds, "scenes", [])):
            if int(getattr(rec, "scene_id", -1)) == int(scene_id):
                idx = i
                break
        if idx < 0:
            raise RuntimeError(f"val 中找不到 scene_id={scene_id}")

    if not (0 <= idx < len(ds)):
        raise ValueError(f"scene_index 超范围: {idx}/{len(ds)}")

    sample = ds[idx]
    return sample, idx


def make_downsampled_pixel_grid(h: int, w: int, downsample: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if downsample <= 0:
        raise ValueError("downsample 必须 > 0")
    gh = max(h // downsample, 1)
    gw = max(w // downsample, 1)

    # 均匀覆盖全图，中心对齐采样
    ys = (torch.arange(gh, dtype=torch.float64, device=device) + 0.5) * (float(h) / float(gh)) - 0.5
    xs = (torch.arange(gw, dtype=torch.float64, device=device) + 0.5) * (float(w) / float(gw)) - 0.5
    ys = ys.clamp(0.0, float(h - 1))
    xs = xs.clamp(0.0, float(w - 1))
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")  # line(v), samp(u)
    return vv, uu


def build_image_space_3d_grid(
    h: int,
    w: int,
    downsample: int,
    height_offset: float,
    low: float,
    high: float,
    num_heights: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vv, uu = make_downsampled_pixel_grid(h, w, downsample, device=device)
    hs = height_offset + torch.linspace(low, high, steps=num_heights, dtype=torch.float64, device=device)
    gv, gu, gh = torch.meshgrid(vv[:, 0], uu[0], hs, indexing="ij")
    return gv, gu, gh


def build_object_space_3d_grid_from_xy_range(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_h: int,
    grid_w: int,
    height_offset: float,
    low: float,
    high: float,
    num_heights: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(float(x_min), float(x_max), steps=max(int(grid_w), 1), dtype=torch.float64, device=device)
    ys = torch.linspace(float(y_min), float(y_max), steps=max(int(grid_h), 1), dtype=torch.float64, device=device)
    hs = height_offset + torch.linspace(float(low), float(high), steps=max(int(num_heights), 1), dtype=torch.float64, device=device)
    gy, gx, gh = torch.meshgrid(ys, xs, hs, indexing="ij")
    return gx, gy, gh


def print_tensor_stats(name: str, t: torch.Tensor) -> None:
    t_cpu = t.detach().cpu()
    print(
        f"{name}: shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}, min={float(t_cpu.min()):.6f}, "
        f"max={float(t_cpu.max()):.6f}, mean={float(t_cpu.mean()):.6f}, std={float(t_cpu.std()):.6f}",
        flush=True,
    )


def normalize_points_2d(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = uv.mean(axis=0)
    d = np.linalg.norm(uv - mu[None, :], axis=1)
    s = np.sqrt(2.0) / max(float(d.mean()), 1e-12)
    T = np.eye(3, dtype=np.float64)
    T[0, 0] = s
    T[1, 1] = s
    T[0, 2] = -s * mu[0]
    T[1, 2] = -s * mu[1]
    uv_h = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=np.float64)], axis=1)
    uv_n = (T @ uv_h.T).T[:, :2]
    return uv_n, T


def normalize_points_3d(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = xyz.mean(axis=0)
    d = np.linalg.norm(xyz - mu[None, :], axis=1)
    s = np.sqrt(3.0) / max(float(d.mean()), 1e-12)
    U = np.eye(4, dtype=np.float64)
    U[0, 0] = s
    U[1, 1] = s
    U[2, 2] = s
    U[0, 3] = -s * mu[0]
    U[1, 3] = -s * mu[1]
    U[2, 3] = -s * mu[2]
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    xyz_n = (U @ xyz_h.T).T[:, :3]
    return xyz_n, U


def dlt_init_projection(xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    xyz_n, U = normalize_points_3d(xyz)
    uv_n, T = normalize_points_2d(uv)
    n = xyz.shape[0]
    X = np.concatenate([xyz_n, np.ones((n, 1), dtype=np.float64)], axis=1)
    u = uv_n[:, 0:1]
    v = uv_n[:, 1:2]
    O = np.zeros_like(X)
    A1 = np.concatenate([X, O, -u * X], axis=1)
    A2 = np.concatenate([O, X, -v * X], axis=1)
    A = np.concatenate([A1, A2], axis=0)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    Pn = vh[-1].reshape(3, 4)
    P = np.linalg.inv(T) @ Pn @ U
    return P / max(np.linalg.norm(P), 1e-12)


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = P[:, :3]
    K, R = scipy.linalg.rq(M)
    diag = np.diag(K).copy()
    sign = np.where(diag >= 0.0, 1.0, -1.0)
    D = np.diag(sign)
    K = K @ D
    R = D @ R
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1.0
        R[2, :] *= -1.0
    # 保符号归一化，避免 K[2,2] < 0 时被 max(..., eps) 夹到 +eps 导致 K 数值爆炸。
    k22 = float(K[2, 2])
    k22_den = math.copysign(max(abs(k22), 1e-12), k22 if k22 != 0.0 else 1.0)
    K = K / k22_den
    _, _, vh = np.linalg.svd(P)
    C_h = vh[-1]
    # SVD 零空间向量只在整体符号上确定（v 与 -v 等价）。
    # 反齐次化时必须保留分母符号，避免 w<0 被错误夹成 +eps 导致相机中心数值爆炸。
    w = float(C_h[3])
    w_den = math.copysign(max(abs(w), 1e-12), w if w != 0.0 else 1.0)
    C = C_h[:3] / w_den
    return K, R, C


def project_points(
    xyz: np.ndarray,
    rvec: np.ndarray,
    C: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    R = scipy.spatial.transform.Rotation.from_rotvec(rvec).as_matrix()
    t = -R @ C
    cam = (R @ xyz.T).T + t[None, :]
    z = cam[:, 2]
    z_safe = np.maximum(z, z_eps)
    u = fx * (cam[:, 0] / z_safe) + cx
    v = fy * (cam[:, 1] / z_safe) + cy
    return np.stack([u, v], axis=1), z


def reproj_metrics(err_xy: np.ndarray) -> dict[str, float]:
    e = np.linalg.norm(err_xy, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "p50": float(np.quantile(e, 0.5)),
        "p95": float(np.quantile(e, 0.95)),
        "p99": float(np.quantile(e, 0.99)),
        "pmax": float(np.max(e)),
    }


def fit_extrinsics_fixed_f(
    xyz: np.ndarray,
    uv: np.ndarray,
    image_hw: tuple[int, int],
    f_fixed: float,
    max_nfev: int,
    print_every: int,
) -> FitResult:
    h, w = image_hw
    cx_fix = 0.5 * float(w - 1)
    cy_fix = 0.5 * float(h - 1)

    if xyz.shape[0] < 20:
        raise ValueError(f"有效对应点过少: {xyz.shape[0]}")

    P0 = dlt_init_projection(xyz, uv)
    K0, R0, C0 = decompose_projection(P0)
    rvec0 = scipy.spatial.transform.Rotation.from_matrix(R0).as_rotvec()

    x0 = np.concatenate([rvec0, C0], axis=0)

    z_eps = 1e-6
    lam_depth = 1.0
    k_depth = 20.0
    min_depth = 1e-2
    lam_center_prior = 5.0e-3
    c_scale = max(float(np.std(xyz[:, 0])), float(np.std(xyz[:, 1])), float(np.std(xyz[:, 2])), 1.0)
    logger = FitLogger(print_every=print_every)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rvec = theta[0:3]
        C = theta[3:6]
        return rvec, C

    def residual(theta: np.ndarray) -> np.ndarray:
        rvec, C = unpack(theta)
        pred_uv, z = project_points(xyz, rvec, C, float(f_fixed), float(f_fixed), cx_fix, cy_fix, z_eps=z_eps)
        data_res = (pred_uv - uv).reshape(-1)

        # 深度屏障，抑制 z<=0
        depth_soft = np.logaddexp(0.0, k_depth * (min_depth - z)) / max(k_depth, 1e-6)
        depth_res = np.sqrt(lam_depth) * np.sqrt(np.maximum(depth_soft, 0.0))

        center_prior_res = np.sqrt(lam_center_prior) * ((C - C0) / c_scale)
        all_res = np.concatenate([data_res, depth_res, center_prior_res], axis=0)

        metrics = reproj_metrics(pred_uv - uv)
        metrics["pos_ratio"] = float(np.mean(z > z_eps))
        logger.add(metrics)

        if not np.isfinite(all_res).all():
            all_res = np.nan_to_num(all_res, nan=1e6, posinf=1e6, neginf=-1e6)
        return all_res

    c_win = max(2.0e5, 10.0 * c_scale)
    lb = np.array(
        [-2 * np.pi, -2 * np.pi, -2 * np.pi, C0[0] - c_win, C0[1] - c_win, C0[2] - c_win],
        dtype=np.float64,
    )
    ub = np.array(
        [2 * np.pi, 2 * np.pi, 2 * np.pi, C0[0] + c_win, C0[1] + c_win, C0[2] + c_win],
        dtype=np.float64,
    )

    t0 = time.perf_counter()
    opt = scipy.optimize.least_squares(
        residual,
        x0=np.clip(x0, lb + 1e-9, ub - 1e-9),
        bounds=(lb, ub),
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        x_scale="jac",
    )
    fit_elapsed = time.perf_counter() - t0

    rvec, C = unpack(opt.x)
    pred_uv, z = project_points(xyz, rvec, C, float(f_fixed), float(f_fixed), cx_fix, cy_fix, z_eps=z_eps)
    met = reproj_metrics(pred_uv - uv)
    pos_ratio = float(np.mean(z > z_eps))

    R = scipy.spatial.transform.Rotation.from_rotvec(rvec).as_matrix()
    t = -R @ C
    K = np.array([[float(f_fixed), 0.0, cx_fix], [0.0, float(f_fixed), cy_fix], [0.0, 0.0, 1.0]], dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    print("\n========== 拟合结果 ==========")
    print(f"optimizer.success={bool(opt.success)}")
    print(f"optimizer.message={opt.message}")
    print(f"nfev={int(opt.nfev)} elapsed={fit_elapsed:.3f}s")
    print(f"f_fixed={float(f_fixed):.6f} (fx=fy), cx={cx_fix:.6f}, cy={cy_fix:.6f}, skew=0")
    print(
        "reproj: rmse={:.6f}, p50={:.6f}, p95={:.6f}, p99={:.6f}, max={:.6f}, pos_depth_ratio={:.4f}".format(
            met["rmse"], met["p50"], met["p95"], met["p99"], met["pmax"], pos_ratio
        )
    )

    diag: dict[str, Any] = {
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
        "nfev": int(opt.nfev),
        "fit_elapsed_sec": float(fit_elapsed),
        "f_fixed": float(f_fixed),
        "center_prior_lambda": float(lam_center_prior),
        "history": logger.history,
    }

    return FitResult(
        K=K,
        w2c=w2c,
        fx=float(f_fixed),
        fy=float(f_fixed),
        cx=float(cx_fix),
        cy=float(cy_fix),
        rmse=met["rmse"],
        p50=met["p50"],
        p95=met["p95"],
        p99=met["p99"],
        pmax=met["pmax"],
        positive_depth_ratio=pos_ratio,
        diagnostics=diag,
    )


def random_reprojection_test(
    rpc_obj: Any,
    scene_center: torch.Tensor,
    h: int,
    w: int,
    fit_res: FitResult,
    num_points: int,
    height_offset: float,
    low: float,
    high: float,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    us = rng.uniform(0.0, float(w - 1), size=(num_points,)).astype(np.float64)
    vs = rng.uniform(0.0, float(h - 1), size=(num_points,)).astype(np.float64)
    hs = (height_offset + rng.uniform(low, high, size=(num_points,))).astype(np.float64)

    dev = rpc_obj.device
    scene_center_d = scene_center.to(dtype=torch.double, device=dev)
    scene_scale = torch.ones_like(scene_center_d)

    x, y = rpc_obj.RPC_LINESAMP2XY(
        line_in=torch.from_numpy(vs).to(dev, dtype=torch.double),
        samp_in=torch.from_numpy(us).to(dev, dtype=torch.double),
        h_in=torch.from_numpy(hs).to(dev, dtype=torch.double),
        output_type="tensor",
        xy_center=scene_center_d,
        xy_scale=scene_scale,
    )

    xyz = np.stack(
        [
            x.detach().cpu().numpy().astype(np.float64),
            y.detach().cpu().numpy().astype(np.float64),
            hs,
        ],
        axis=1,
    )

    r = scipy.spatial.transform.Rotation.from_matrix(fit_res.w2c[:3, :3])
    rvec = r.as_rotvec()
    C = -fit_res.w2c[:3, :3].T @ fit_res.w2c[:3, 3]

    uv_pred, z = project_points(
        xyz,
        rvec=rvec,
        C=C,
        fx=fit_res.fx,
        fy=fit_res.fy,
        cx=fit_res.cx,
        cy=fit_res.cy,
        z_eps=1e-6,
    )
    uv_gt = np.stack([us, vs], axis=1)

    err = np.linalg.norm(uv_pred - uv_gt, axis=1)
    stat = {
        "mean": float(np.mean(err)),
        "median": float(np.quantile(err, 0.5)),
        "p90": float(np.quantile(err, 0.9)),
        "p95": float(np.quantile(err, 0.95)),
        "p99": float(np.quantile(err, 0.99)),
        "max": float(np.max(err)),
        "pos_depth_ratio": float(np.mean(z > 1e-6)),
    }

    print("\n========== 随机100点测试 ==========")
    print(
        "err(px): mean={mean:.6f}, median={median:.6f}, p90={p90:.6f}, p95={p95:.6f}, p99={p99:.6f}, max={max:.6f}, pos_depth={pos_depth_ratio:.4f}".format(
            **stat
        )
    )
    print("前10个点误差(px):", np.array2string(err[:10], precision=6, separator=", "))

    return stat


def save_rgb(path: Path, chw: np.ndarray) -> None:
    arr = np.clip(chw.transpose(1, 2, 0), 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


def save_gray(path: Path, hw: np.ndarray) -> None:
    arr = np.clip(hw, 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path)


def render_with_gsplat(
    fit_res: FitResult,
    xyz_world: torch.Tensor,
    rgb: torch.Tensor,
    image_hw: tuple[int, int],
    device: torch.device,
) -> dict[str, Any]:
    try:
        from gsplat import rasterization  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("未安装gsplat，无法执行第7步渲染实验") from e

    h, w = image_hw
    means = xyz_world.to(device=device, dtype=torch.float32)
    n = means.shape[0]
    scales = torch.full((n, 3), 0.5, device=device, dtype=torch.float32)
    rots = torch.zeros((n, 4), device=device, dtype=torch.float32)
    rots[:, 0] = 1.0
    opacities = torch.ones((n,), device=device, dtype=torch.float32)
    colors = rgb.to(device=device, dtype=torch.float32)

    viewmats = torch.from_numpy(fit_res.w2c).to(device=device, dtype=torch.float32).unsqueeze(0)
    Ks = torch.from_numpy(fit_res.K).to(device=device, dtype=torch.float32).unsqueeze(0)
    bg = torch.zeros((1, 3), device=device, dtype=torch.float32)

    t0 = time.perf_counter()
    rendering, alpha, _ = rasterization(
        means,
        rots,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        w,
        h,
        sh_degree=None,
        render_mode="RGB+D",
        packed=False,
        near_plane=1e-6,
        backgrounds=bg,
        rasterize_mode="classic",
    )
    t_render = time.perf_counter() - t0

    rgb_out = rendering[0, ..., :3].permute(2, 0, 1).contiguous()
    alpha_out = alpha[0].permute(2, 0, 1).contiguous() if alpha.ndim == 4 else alpha[0].unsqueeze(0)

    return {
        "render_rgb": rgb_out,
        "render_alpha": alpha_out,
        "render_sec": float(t_render),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_cfg(args.config)
    sample, scene_index_used = pick_val_sample(cfg, scene_index=args.scene_index, scene_id=args.scene_id)

    v_total = int(sample["images"].shape[0])
    if not (0 <= args.view_index < v_total):
        raise ValueError(f"view_index 超范围: {args.view_index}/{v_total}")

    vi = int(args.view_index)
    img = sample["images"][vi]
    hgt = sample["height_gt"][vi, 0]
    rpc = sample["rpc_gt"][vi]

    h, w = int(img.shape[-2]), int(img.shape[-1])
    scene_center = sample["scene_xy_center"].to(torch.double)
    scene_scale = torch.ones_like(scene_center)

    work_dir_cfg = cfg.get("system", {}).get("work_dir", "work_dirs/sat2world_default")
    work_dir = Path(args.work_dir if args.work_dir else work_dir_cfg)
    out_dir = work_dir / "rpc2pinhole_test" / f"scene_{int(sample['scene_id'])}" / f"view_{int(sample['view_ids'][vi])}"
    out_dir.mkdir(parents=True, exist_ok=True)

    height_offset = float(rpc.HEIGHT_OFF.detach().cpu().item() if torch.is_tensor(rpc.HEIGHT_OFF) else rpc.HEIGHT_OFF)

    print("========== 视图信息 ==========")
    print(f"config={args.config}")
    print(f"scene_index_used={scene_index_used}, scene_id={int(sample['scene_id'])}")
    print(f"view_index={vi}, view_id={int(sample['view_ids'][vi])}")
    print(f"image_hw=({h}, {w})")
    print(f"height_offset(HEIGHT_OFF)={height_offset:.6f}")
    print(f"scene_center(y,x)={scene_center.detach().cpu().numpy().tolist()}")
    print(f"scene_scale_fixed={scene_scale.detach().cpu().numpy().tolist()}")
    print(f"image_path={sample['image_paths'][vi]}")

    # Step 3: 像方 64x64x100（默认）网格
    dev_rpc = rpc.device
    grid_v, grid_u, grid_h = build_image_space_3d_grid(
        h=h,
        w=w,
        downsample=int(args.grid_downsample),
        height_offset=height_offset,
        low=float(args.height_low),
        high=float(args.height_high),
        num_heights=int(args.num_heights),
        device=dev_rpc,
    )
    print("\n========== 像方网格信息 ==========")
    print_tensor_stats("grid_v(line)", grid_v)
    print_tensor_stats("grid_u(samp)", grid_u)
    print_tensor_stats("grid_h", grid_h)

    # Step 4: RPC 投影到物方
    v_flat = grid_v.reshape(-1)
    u_flat = grid_u.reshape(-1)
    h_flat = grid_h.reshape(-1)
    x_flat, y_flat = rpc.RPC_LINESAMP2XY(
        line_in=v_flat,
        samp_in=u_flat,
        h_in=h_flat,
        output_type="tensor",
        xy_center=scene_center.to(device=dev_rpc, dtype=torch.double),
        xy_scale=scene_scale.to(device=dev_rpc, dtype=torch.double),
    )
    xyz_flat = torch.stack([x_flat, y_flat, h_flat], dim=-1)
    finite = torch.isfinite(xyz_flat).all(dim=-1)
    if not bool(finite.any()):
        raise RuntimeError("物方网格全为无效点")

    xyz_a = xyz_flat[finite].detach().cpu().numpy().astype(np.float64)
    uv_a = torch.stack([u_flat, v_flat], dim=-1)[finite].detach().cpu().numpy().astype(np.float64)

    # Part-B: 基于 Part-A 的物方范围构建均匀网格，再通过 RPC 正投影到像方，筛选图像内点
    grid_h, grid_w = int(grid_v.shape[0]), int(grid_v.shape[1])
    x_min, x_max = float(xyz_a[:, 0].min()), float(xyz_a[:, 0].max())
    y_min, y_max = float(xyz_a[:, 1].min()), float(xyz_a[:, 1].max())
    x_b, y_b, h_b = build_object_space_3d_grid_from_xy_range(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        grid_h=grid_h,
        grid_w=grid_w,
        height_offset=height_offset,
        low=float(args.height_low),
        high=float(args.height_high),
        num_heights=int(args.num_heights),
        device=dev_rpc,
    )
    x_bf = x_b.reshape(-1)
    y_bf = y_b.reshape(-1)
    h_bf = h_b.reshape(-1)
    line_b, samp_b = rpc.RPC_XY2LINESAMP(
        x_in=x_bf,
        y_in=y_bf,
        h_in=h_bf,
        output_type="tensor",
        xy_center=scene_center.to(device=dev_rpc, dtype=torch.double),
        xy_scale=scene_scale.to(device=dev_rpc, dtype=torch.double),
    )
    uv_b_all = torch.stack([samp_b, line_b], dim=-1)
    xyz_b_all = torch.stack([x_bf, y_bf, h_bf], dim=-1)
    valid_b = (
        torch.isfinite(uv_b_all).all(dim=-1)
        & torch.isfinite(xyz_b_all).all(dim=-1)
        & (uv_b_all[:, 0] >= 0.0)
        & (uv_b_all[:, 0] <= float(w - 1))
        & (uv_b_all[:, 1] >= 0.0)
        & (uv_b_all[:, 1] <= float(h - 1))
    )
    xyz_b = xyz_b_all[valid_b].detach().cpu().numpy().astype(np.float64)
    uv_b = uv_b_all[valid_b].detach().cpu().numpy().astype(np.float64)
    xyz = np.concatenate([xyz_a, xyz_b], axis=0)
    uv = np.concatenate([uv_a, uv_b], axis=0)

    print("\n========== 物方网格信息 ==========")
    print(
        f"partA={xyz_a.shape[0]}, partB_raw={xyz_b_all.shape[0]}, partB_valid={xyz_b.shape[0]}, total={xyz.shape[0]}, partA_valid_ratio={float(finite.float().mean().item()):.6f}"
    )
    print(
        "x[min,max]=[{:.6f},{:.6f}] y[min,max]=[{:.6f},{:.6f}] h[min,max]=[{:.6f},{:.6f}]".format(
            float(xyz[:, 0].min()),
            float(xyz[:, 0].max()),
            float(xyz[:, 1].min()),
            float(xyz[:, 1].max()),
            float(xyz[:, 2].min()),
            float(xyz[:, 2].max()),
        )
    )

    # Step 5: 拟合 pinhole（独立实现）
    fit_res = fit_extrinsics_fixed_f(
        xyz=xyz,
        uv=uv,
        image_hw=(h, w),
        f_fixed=float(args.f_fixed),
        max_nfev=int(args.fit_max_nfev),
        print_every=int(args.fit_print_every),
    )

    # Step 6: 随机100点测试
    random_stats = random_reprojection_test(
        rpc_obj=rpc,
        scene_center=scene_center,
        h=h,
        w=w,
        fit_res=fit_res,
        num_points=int(args.num_random_test_points),
        height_offset=height_offset,
        low=float(args.height_low),
        high=float(args.height_high),
        seed=int(args.seed) + 123,
    )

    # Step 7: 全像素点云 + gsplat 渲染
    print("\n========== 全像素点云与渲染 ==========")
    line_map = torch.arange(h, dtype=torch.double, device=dev_rpc).view(-1, 1).expand(h, w).reshape(-1)
    samp_map = torch.arange(w, dtype=torch.double, device=dev_rpc).view(1, -1).expand(h, w).reshape(-1)
    h_map = hgt.to(dtype=torch.double, device=dev_rpc).reshape(-1)

    x_map, y_map = rpc.RPC_LINESAMP2XY(
        line_in=line_map,
        samp_in=samp_map,
        h_in=h_map,
        output_type="tensor",
        xy_center=scene_center.to(device=dev_rpc, dtype=torch.double),
        xy_scale=scene_scale.to(device=dev_rpc, dtype=torch.double),
    )

    xyz_world = torch.stack([x_map, y_map, h_map], dim=-1)
    rgb = img.to(torch.float32).permute(1, 2, 0).reshape(-1, 3)
    valid = torch.isfinite(xyz_world).all(dim=-1)
    xyz_world = xyz_world[valid]
    rgb = rgb[valid]
    print(f"point_cloud_size={int(xyz_world.shape[0])}")

    render_dev = torch.device(args.device)
    render_out = render_with_gsplat(
        fit_res=fit_res,
        xyz_world=xyz_world,
        rgb=rgb,
        image_hw=(h, w),
        device=render_dev,
    )

    render_rgb = render_out["render_rgb"].detach().cpu().numpy().astype(np.float32)
    render_alpha = render_out["render_alpha"].detach().cpu().numpy().astype(np.float32)
    gt_rgb = img.detach().cpu().numpy().astype(np.float32)

    diff = render_rgb - gt_rgb
    l1 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))
    alpha_cov = float((render_alpha > 0.05).mean())

    print(
        "render metrics: L1={:.6f}, MSE={:.6f}, PSNR={:.4f}dB, alpha_cov(>0.05)={:.6f}, render_sec={:.4f}".format(
            l1, mse, psnr, alpha_cov, float(render_out["render_sec"])
        )
    )

    save_rgb(out_dir / "gt_rgb.png", gt_rgb)
    save_rgb(out_dir / "render_rgb.png", render_rgb)
    save_gray(out_dir / "render_alpha.png", np.clip(render_alpha[0], 0.0, 1.0))

    summary = {
        "scene_id": int(sample["scene_id"]),
        "view_id": int(sample["view_ids"][vi]),
        "image_hw": [h, w],
        "fit": {
            "fx": fit_res.fx,
            "fy": fit_res.fy,
            "cx": fit_res.cx,
            "cy": fit_res.cy,
            "rmse": fit_res.rmse,
            "p50": fit_res.p50,
            "p95": fit_res.p95,
            "p99": fit_res.p99,
            "max": fit_res.pmax,
            "positive_depth_ratio": fit_res.positive_depth_ratio,
        },
        "random_test": random_stats,
        "render": {
            "l1": l1,
            "mse": mse,
            "psnr": psnr,
            "alpha_cov": alpha_cov,
            "render_sec": float(render_out["render_sec"]),
        },
    }

    with (out_dir / "summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    with (out_dir / "fit_diagnostics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(fit_res.diagnostics, f, sort_keys=False, allow_unicode=True)

    print(f"\n结果已输出到: {out_dir}")


if __name__ == "__main__":
    main()

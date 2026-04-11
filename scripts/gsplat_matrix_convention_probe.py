"""Synthetic gsplat matrix-convention probe.

This script builds a fully synthetic scene and camera, then sweeps multiple
viewmat/K variants to determine which convention best matches geometric
expectation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


@dataclass
class VariantResult:
    name: str
    score: float
    alpha_max: float
    alpha_coverage: float
    positive_depth_ratio: float
    in_frame_ratio: float
    reproj_p95_px: float
    occ_iou: float
    centroid_err_px: float
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("gsplat matrix convention probe")
    p.add_argument("--output-dir", type=str, default="work_dirs/gsplat_matrix_probe")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-gaussians", type=int, default=600)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--margin-ratio", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-all-images", action="store_true")
    return p.parse_args()


def look_at_w2c(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
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


def scene_corners(bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    xs = bounds["x"]
    ys = bounds["y"]
    zs = bounds["h"]
    pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float64)


def project_points_np(xyz: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (w2c[:3, :] @ xyz_h.T).T
    z = cam[:, 2]
    proj = (K @ cam.T).T
    uv = np.full((xyz.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1e-8) & np.isfinite(proj).all(axis=1)
    uv[valid] = proj[valid, :2] / proj[valid, 2:3]
    return uv, z, valid


def fit_intrinsics_cover_scene(corners: np.ndarray, w2c: np.ndarray, width: int, height: int, margin_ratio: float) -> np.ndarray:
    cxy = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    xyz_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (w2c[:3, :] @ xyz_h.T).T
    z = cam[:, 2]
    if not np.all(z > 1e-6):
        raise RuntimeError("camera pose invalid: not all scene corners are in front of camera")

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


def sample_gaussians(bounds: dict[str, tuple[float, float]], n: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(bounds["x"][0], bounds["x"][1], size=(n,))
    y = rng.uniform(bounds["y"][0], bounds["y"][1], size=(n,))
    z = rng.uniform(bounds["h"][0], bounds["h"][1], size=(n,))
    means = np.stack([x, y, z], axis=1).astype(np.float32)

    scales = rng.uniform(0.8, 4.0, size=(n, 3)).astype(np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    opacities = rng.uniform(0.75, 1.0, size=(n,)).astype(np.float32)

    palette = np.asarray(
        [
            [1.0, 0.15, 0.15],
            [0.15, 1.0, 0.15],
            [0.15, 0.15, 1.0],
            [1.0, 1.0, 0.2],
            [1.0, 0.2, 1.0],
            [0.2, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    colors = palette[np.arange(n) % len(palette)]

    return {
        "means": torch.from_numpy(means).to(device=device),
        "scales": torch.from_numpy(scales).to(device=device),
        "quats": torch.from_numpy(quats).to(device=device),
        "opacities": torch.from_numpy(opacities).to(device=device),
        "colors": torch.from_numpy(colors).to(device=device),
    }


def expected_occ_map(uv_ref: np.ndarray, depth_ref: np.ndarray, scales: np.ndarray, width: int, height: int, fx: float) -> np.ndarray:
    occ = np.zeros((height, width), dtype=np.float32)
    for i in range(uv_ref.shape[0]):
        if not np.isfinite(uv_ref[i]).all() or depth_ref[i] <= 1e-8:
            continue
        u, v = float(uv_ref[i, 0]), float(uv_ref[i, 1])
        if u < -10 or u >= width + 10 or v < -10 or v >= height + 10:
            continue
        rad = float(np.clip((fx * float(np.max(scales[i]))) / float(depth_ref[i]) * 0.8, 1.0, 16.0))
        xmin = max(int(np.floor(u - rad)), 0)
        xmax = min(int(np.ceil(u + rad)) + 1, width)
        ymin = max(int(np.floor(v - rad)), 0)
        ymax = min(int(np.ceil(v + rad)) + 1, height)
        if xmin >= xmax or ymin >= ymax:
            continue
        ys = np.arange(ymin, ymax, dtype=np.float32)[:, None]
        xs = np.arange(xmin, xmax, dtype=np.float32)[None, :]
        d2 = (xs - u) ** 2 + (ys - v) ** 2
        g = np.exp(-0.5 * d2 / max(rad * rad * 0.35, 1e-6)).astype(np.float32)
        occ[ymin:ymax, xmin:xmax] = np.maximum(occ[ymin:ymax, xmin:xmax], g)
    return occ


def render_with_variant(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from gsplat import rasterization

    viewmats = torch.from_numpy(w2c.astype(np.float32)).unsqueeze(0).to(device=means.device)
    Ks = torch.from_numpy(K.astype(np.float32)).unsqueeze(0).to(device=means.device)
    bg = torch.zeros((1, 3), dtype=torch.float32, device=means.device)

    rendering, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        sh_degree=None,
        render_mode="RGB+D",
        packed=False,
        near_plane=1e-6,
        backgrounds=bg,
        rasterize_mode="classic",
    )
    rgb = rendering[0, ..., :3].permute(2, 0, 1).contiguous()
    depth = rendering[0, ..., 3:4].permute(2, 0, 1).contiguous()
    alpha = alpha[0].permute(2, 0, 1).contiguous() if alpha.ndim == 4 else alpha[0].unsqueeze(0)
    return rgb, alpha, depth


def alpha_stats(alpha_chw: torch.Tensor) -> tuple[float, float, np.ndarray]:
    a = alpha_chw.detach().cpu().numpy().squeeze(0).astype(np.float32)
    a = np.maximum(a, 0.0)
    amax = float(np.max(a))
    thr = max(0.02, amax * 0.15)
    cov = float((a > thr).mean())
    return amax, cov, a


def occ_metrics(alpha_hw: np.ndarray, occ_ref: np.ndarray) -> tuple[float, float]:
    ta = max(0.02, float(alpha_hw.max()) * 0.15)
    pa = alpha_hw > ta
    tr = max(0.15, float(occ_ref.max()) * 0.25)
    pr = occ_ref > tr
    inter = float(np.logical_and(pa, pr).sum())
    union = float(np.logical_or(pa, pr).sum())
    iou = inter / max(union, 1.0)

    ya, xa = np.nonzero(pa)
    yr, xr = np.nonzero(pr)
    if len(xa) == 0 or len(xr) == 0:
        centroid_err = float("inf")
    else:
        ca = np.array([xa.mean(), ya.mean()], dtype=np.float64)
        cr = np.array([xr.mean(), yr.mean()], dtype=np.float64)
        centroid_err = float(np.linalg.norm(ca - cr))
    return iou, centroid_err


def save_rgb(path: Path, chw: torch.Tensor) -> None:
    arr = chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)


def save_gray(path: Path, hw: np.ndarray, vmin: float = 0.0, vmax: float | None = None) -> None:
    m = hw.astype(np.float32)
    if vmax is None:
        vmax = float(np.max(m))
    vmax = max(vmax, vmin + 1e-8)
    m = np.clip((m - vmin) / (vmax - vmin), 0.0, 1.0)
    Image.fromarray((m * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    width = int(args.width)
    height = int(args.height)

    bounds = {
        "x": (-1000.0, 1000.0),
        "y": (-1000.0, 1000.0),
        "h": (-100.0, 100.0),
    }

    eye = np.array([0.0, 0.0, 2400.0], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    w2c_gt = look_at_w2c(eye, target, up)
    corners = scene_corners(bounds)
    K_gt = fit_intrinsics_cover_scene(corners, w2c_gt, width, height, float(args.margin_ratio))

    gauss = sample_gaussians(bounds, int(args.num_gaussians), int(args.seed), device)

    means_np = gauss["means"].detach().cpu().numpy().astype(np.float64)
    scales_np = gauss["scales"].detach().cpu().numpy().astype(np.float64)

    uv_ref, z_ref, valid_ref = project_points_np(means_np, K_gt, w2c_gt)
    in_frame_ref = valid_ref & (uv_ref[:, 0] >= 0) & (uv_ref[:, 0] < width) & (uv_ref[:, 1] >= 0) & (uv_ref[:, 1] < height)

    occ_ref = expected_occ_map(uv_ref, z_ref, scales_np, width, height, fx=float(K_gt[0, 0]))

    I4 = np.eye(4, dtype=np.float64)
    Fz = I4.copy()
    Fz[2, 2] = -1.0
    Fy = I4.copy()
    Fy[1, 1] = -1.0

    w2c_variants: dict[str, np.ndarray] = {
        "w2c_gt": w2c_gt,
        "inv_w2c": np.linalg.inv(w2c_gt),
        "w2c_T": w2c_gt.T.copy(),
        "Fz_w2c": Fz @ w2c_gt,
        "Fy_w2c": Fy @ w2c_gt,
        "FyFz_w2c": Fy @ Fz @ w2c_gt,
        "w2c_Fz": w2c_gt @ Fz,
    }

    K_norm = K_gt.copy()
    K_norm[0, :] = K_norm[0, :] / float(width)
    K_norm[1, :] = K_norm[1, :] / float(height)

    K_flip = K_gt.copy()
    K_flip[0, 2] = float(width - 1) - K_flip[0, 2]
    K_flip[1, 2] = float(height - 1) - K_flip[1, 2]

    K_swap = K_gt.copy()
    K_swap[0, 0], K_swap[1, 1] = K_swap[1, 1], K_swap[0, 0]

    K_variants: dict[str, np.ndarray] = {
        "K_pixel": K_gt,
        "K_normalized": K_norm,
        "K_T": K_gt.T.copy(),
        "K_flip_cxcy": K_flip,
        "K_swap_fxfy": K_swap,
    }

    results: list[VariantResult] = []
    leaderboard_rows: list[dict[str, Any]] = []

    for wname, wmat in w2c_variants.items():
        for kname, Kmat in K_variants.items():
            variant = f"{wname}__{kname}"
            uv, z, valid = project_points_np(means_np, Kmat, wmat)
            reproj = np.linalg.norm(uv - uv_ref, axis=1)
            mask = valid & valid_ref & np.isfinite(reproj)
            reproj_p95 = float(np.quantile(reproj[mask], 0.95)) if mask.any() else float("inf")
            in_frame = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)

            rgb, alpha, depth = render_with_variant(
                means=gauss["means"],
                quats=gauss["quats"],
                scales=gauss["scales"],
                opacities=gauss["opacities"],
                colors=gauss["colors"],
                w2c=wmat,
                K=Kmat,
                width=width,
                height=height,
            )

            amax, acov, alpha_hw = alpha_stats(alpha)
            occ_iou, centroid_err = occ_metrics(alpha_hw, occ_ref)
            z_valid = np.isfinite(z)
            pos_depth_ratio = float((z[z_valid] > 1e-8).mean()) if z_valid.any() else 0.0
            in_frame_ratio = float(in_frame.mean()) if in_frame.size > 0 else 0.0

            reproj_norm = min(reproj_p95 / max(float(width), float(height)), 5.0)
            centroid_norm = min(centroid_err / max(float(width), float(height)), 5.0) if np.isfinite(centroid_err) else 5.0
            score = (
                2.0 * (1.0 - min(reproj_norm, 1.0))
                + 1.6 * acov
                + 1.8 * occ_iou
                + 0.6 * in_frame_ratio
                - 1.2 * centroid_norm
            )

            note = "ok" if (amax > 1e-3 and acov > 1e-4 and np.isfinite(reproj_p95)) else "weak/invalid"
            results.append(
                VariantResult(
                    name=variant,
                    score=float(score),
                    alpha_max=float(amax),
                    alpha_coverage=float(acov),
                    positive_depth_ratio=float(pos_depth_ratio),
                    in_frame_ratio=float(in_frame_ratio),
                    reproj_p95_px=float(reproj_p95),
                    occ_iou=float(occ_iou),
                    centroid_err_px=float(centroid_err),
                    note=note,
                )
            )

            leaderboard_rows.append(
                {
                    "variant": variant,
                    "score": float(score),
                    "alpha_max": float(amax),
                    "alpha_coverage": float(acov),
                    "positive_depth_ratio": float(pos_depth_ratio),
                    "in_frame_ratio": float(in_frame_ratio),
                    "reproj_p95_px": float(reproj_p95),
                    "occ_iou": float(occ_iou),
                    "centroid_err_px": float(centroid_err),
                    "note": note,
                }
            )

            if args.save_all_images:
                vdir = out_dir / variant
                vdir.mkdir(parents=True, exist_ok=True)
                save_rgb(vdir / "rgb.png", rgb)
                save_gray(vdir / "alpha.png", alpha_hw, vmin=0.0, vmax=max(1e-6, float(alpha_hw.max())))
                save_gray(vdir / "depth.png", depth.detach().cpu().numpy().squeeze(0), vmin=0.0)

    results_sorted = sorted(results, key=lambda x: x.score, reverse=True)
    best = results_sorted[0]

    with open(out_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(leaderboard_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(leaderboard_rows, key=lambda r: r["score"], reverse=True))

    summary = {
        "scene_bounds_m": bounds,
        "image_size": [height, width],
        "camera_gt": {
            "eye": eye.tolist(),
            "target": target.tolist(),
            "K": K_gt.tolist(),
            "w2c": w2c_gt.tolist(),
        },
        "gaussians": {
            "num": int(args.num_gaussians),
            "seed": int(args.seed),
            "in_frame_ratio_gt": float(in_frame_ref.mean()),
        },
        "best_variant": {
            "name": best.name,
            "score": best.score,
            "alpha_max": best.alpha_max,
            "alpha_coverage": best.alpha_coverage,
            "reproj_p95_px": best.reproj_p95_px,
            "occ_iou": best.occ_iou,
            "centroid_err_px": best.centroid_err_px,
        },
        "top10": [r.__dict__ for r in results_sorted[:10]],
        "all_variants": [r.__dict__ for r in results_sorted],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    save_gray(out_dir / "expected_occ_ref.png", occ_ref, vmin=0.0, vmax=max(1e-6, float(occ_ref.max())))
    img_ref = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img_ref)
    for i in range(uv_ref.shape[0]):
        if not in_frame_ref[i]:
            continue
        u, v = float(uv_ref[i, 0]), float(uv_ref[i, 1])
        draw.ellipse((u - 1.5, v - 1.5, u + 1.5, v + 1.5), fill=(255, 255, 0))
    img_ref.save(out_dir / "oracle_projection_points.png")

    print("================ gsplat matrix convention probe ================")
    print(f"output_dir: {out_dir}")
    print(f"num_variants: {len(results_sorted)}")
    print("best_variant:", best.name)
    print(
        "best_metrics:",
        f"score={best.score:.6f}",
        f"alpha_max={best.alpha_max:.6f}",
        f"alpha_cov={best.alpha_coverage:.6f}",
        f"reproj_p95={best.reproj_p95_px:.6f}px",
        f"occ_iou={best.occ_iou:.6f}",
        f"centroid_err={best.centroid_err_px:.6f}px",
    )


if __name__ == "__main__":
    main()

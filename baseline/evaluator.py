from __future__ import annotations

import numpy as np
import torch

from loss.common import apply_affine_to_points, make_uniform_grid_points

from .types import Track
from .utils import stats


def compute_affine_grid_error_samples(affine_pred_corr: torch.Tensor, affine_gt_forward: torch.Tensor, image_hw: tuple[int, int], ref_view_idx: int):
    b, v = affine_pred_corr.shape[:2]
    h, w = image_hw
    grid = make_uniform_grid_points(h, w, 16, 16, affine_pred_corr.device, affine_pred_corr.dtype)
    n = int(grid.shape[0])
    g_true = grid.view(1, 1, n, 2).expand(b, v, n, 2)
    g_obs = apply_affine_to_points(g_true, affine_gt_forward)
    g_rec = apply_affine_to_points(g_obs, affine_pred_corr)
    err = torch.linalg.norm(g_rec - g_true, dim=-1)
    mask = torch.ones((b, v, n), dtype=torch.bool, device=err.device)
    mask[:, ref_view_idx] = False
    return err[mask]


def evaluate_affine(aff_est_corr: np.ndarray, aff_gt_corr: np.ndarray, aff_gt_fwd: np.ndarray, image_hw: tuple[int, int], ref_view_idx: int) -> dict:
    diff = aff_est_corr - aff_gt_corr
    per_view = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    m = np.ones((per_view.shape[0],), dtype=bool)
    m[ref_view_idx] = False
    grid_err = compute_affine_grid_error_samples(
        torch.from_numpy(aff_est_corr[None]).double(),
        torch.from_numpy(aff_gt_fwd[None]).double(),
        image_hw=image_hw,
        ref_view_idx=ref_view_idx,
    ).detach().cpu().numpy()
    return {
        "param_error_stats": stats(per_view[m]),
        "grid_error_stats": stats(grid_err),
    }


def _bilinear_sample(height: np.ndarray, line: float, samp: float) -> float | None:
    h, w = height.shape
    if line < 0 or samp < 0 or line > h - 1 or samp > w - 1:
        return None
    y0 = int(np.floor(line)); x0 = int(np.floor(samp))
    y1 = min(y0 + 1, h - 1); x1 = min(x0 + 1, w - 1)
    wy = line - y0; wx = samp - x0
    v = (1 - wy) * (1 - wx) * height[y0, x0] + (1 - wy) * wx * height[y0, x1] + wy * (1 - wx) * height[y1, x0] + wy * wx * height[y1, x1]
    return float(v)


def evaluate_track_heights(points_xyz: np.ndarray, tracks: list[Track], rpc_gt_views, heights: list[np.ndarray], xy_center, xy_scale) -> dict:
    est_h = []
    gt_h = []
    valid = 0
    for ti, tr in enumerate(tracks):
        xyz = points_xyz[ti]
        per = []
        for o in tr.observations:
            rpc = rpc_gt_views[o.view_idx]
            line, samp = rpc.RPC_XY2LINESAMP(
                x_in=torch.tensor([xyz[0]], dtype=torch.double, device=rpc.device),
                y_in=torch.tensor([xyz[1]], dtype=torch.double, device=rpc.device),
                h_in=torch.tensor([xyz[2]], dtype=torch.double, device=rpc.device),
                xy_center=xy_center,
                xy_scale=xy_scale,
            )
            l = float(line.item()); s = float(samp.item())
            v = _bilinear_sample(heights[o.view_idx], l, s)
            if v is not None:
                per.append(v)
        if len(per) == 0:
            continue
        valid += 1
        est_h.append(float(xyz[2]))
        gt_h.append(float(np.mean(per)))

    if len(est_h) == 0:
        return {"valid_tracks": 0, "valid_track_ratio": 0.0}
    e = np.asarray(est_h) - np.asarray(gt_h)
    ae = np.abs(e)
    return {
        "valid_tracks": int(valid),
        "valid_track_ratio": float(valid / max(len(tracks), 1)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "median": float(np.median(ae)),
        "p95": float(np.quantile(ae, 0.95)),
        "max": float(ae.max()),
    }

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from .triangulation import _project_rpc
from .types import BAConfig, BASolution, Track
from .utils import stats


def _pack(aff: np.ndarray, pts: np.ndarray, ref: int) -> np.ndarray:
    v = aff.shape[0]
    vec = []
    for i in range(v):
        if i == ref:
            continue
        vec.append(aff[i].reshape(-1))
    vec.append(pts.reshape(-1))
    return np.concatenate(vec).astype(np.float64)


def _unpack(x: np.ndarray, v: int, t: int, ref: int) -> tuple[np.ndarray, np.ndarray]:
    aff = np.tile(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64), (v, 1, 1))
    pos = 0
    for i in range(v):
        if i == ref:
            continue
        aff[i] = x[pos:pos + 6].reshape(2, 3)
        pos += 6
    pts = x[pos:pos + 3 * t].reshape(t, 3)
    return aff, pts


def _residuals(x, tracks, rpc_views, v, t, ref, xy_center, xy_scale, cfg: BAConfig):
    aff, pts = _unpack(x, v=v, t=t, ref=ref)
    r = []
    for ti, tr in enumerate(tracks):
        xyz = pts[ti]
        for o in tr.observations:
            l_pred, s_pred = _project_rpc(rpc_views[o.view_idx], xyz, aff[o.view_idx], xy_center, xy_scale)
            r.extend([l_pred - o.line, s_pred - o.samp])
    if cfg.affine_prior_weight > 0:
        w = float(cfg.affine_prior_weight)
        for i in range(v):
            if i == ref:
                continue
            r.extend((w * (aff[i] - np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))).reshape(-1).tolist())
    return np.asarray(r, dtype=np.float64)


def _compute_obs_res(tracks, rpc_views, aff, pts, xy_center, xy_scale):
    vals = []
    for ti, tr in enumerate(tracks):
        xyz = pts[ti]
        for o in tr.observations:
            l_pred, s_pred = _project_rpc(rpc_views[o.view_idx], xyz, aff[o.view_idx], xy_center, xy_scale)
            vals.append(np.hypot(l_pred - o.line, s_pred - o.samp))
    return np.asarray(vals, dtype=np.float64)


def solve_free_network_ba(
    tracks: list[Track],
    rpc_views,
    points_init: np.ndarray,
    ref_view_idx: int,
    xy_center,
    xy_scale,
    cfg: BAConfig,
) -> BASolution:
    v = len(rpc_views)
    t = len(tracks)
    aff0 = np.tile(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64), (v, 1, 1))

    before = _compute_obs_res(tracks, rpc_views, aff0, points_init, xy_center, xy_scale)
    x0 = _pack(aff0, points_init, ref=ref_view_idx)
    res = least_squares(
        _residuals,
        x0=x0,
        args=(tracks, rpc_views, v, t, ref_view_idx, xy_center, xy_scale, cfg),
        method="trf",
        loss=cfg.robust_loss,
        f_scale=cfg.huber_delta,
        max_nfev=cfg.max_iterations,
    )
    aff, pts = _unpack(res.x, v=v, t=t, ref=ref_view_idx)

    obs_err = _compute_obs_res(tracks, rpc_views, aff, pts, xy_center, xy_scale)
    inlier = obs_err <= cfg.outlier_threshold_px
    kept_ratio = float(inlier.mean()) if inlier.size > 0 else 0.0

    valid_track = np.ones((t,), dtype=bool)
    return BASolution(
        affines_correction=aff,
        points_xyz=pts,
        valid_track_mask=valid_track,
        reproj_before=stats(before),
        reproj_after=stats(obs_err),
        num_iterations=int(res.nfev),
        kept_observation_ratio=kept_ratio,
    )

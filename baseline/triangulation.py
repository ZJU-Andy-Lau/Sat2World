from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import least_squares

from .types import Track


def _project_rpc(rpc, xyz: np.ndarray, affine_corr: np.ndarray, xy_center, xy_scale) -> tuple[float, float]:
    x, y, h = xyz.tolist()
    line, samp = rpc.RPC_XY2LINESAMP(
        x_in=torch.tensor([x], dtype=torch.double, device=rpc.device),
        y_in=torch.tensor([y], dtype=torch.double, device=rpc.device),
        h_in=torch.tensor([h], dtype=torch.double, device=rpc.device),
        xy_center=xy_center,
        xy_scale=xy_scale,
    )
    line = float(line.detach().cpu().item())
    samp = float(samp.detach().cpu().item())
    p = np.array([line, samp], dtype=np.float64)
    p_corr = affine_corr[:, :2] @ p + affine_corr[:, 2]
    return float(p_corr[0]), float(p_corr[1])


def init_track_points(tracks: list[Track], rpc_views, affines_corr: np.ndarray, xy_center, xy_scale) -> np.ndarray:
    if len(tracks) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    h0 = np.mean([
        float(rpc.HEIGHT_OFF.detach().cpu().item() if torch.is_tensor(rpc.HEIGHT_OFF) else rpc.HEIGHT_OFF)
        for rpc in rpc_views
    ])
    points = np.zeros((len(tracks), 3), dtype=np.float64)
    for ti, tr in enumerate(tracks):
        obs0 = tr.observations[0]
        rpc0 = rpc_views[obs0.view_idx]
        x0, y0 = rpc0.RPC_LINESAMP2XY(
            line_in=torch.tensor([obs0.line], dtype=torch.double, device=rpc0.device),
            samp_in=torch.tensor([obs0.samp], dtype=torch.double, device=rpc0.device),
            h_in=torch.tensor([h0], dtype=torch.double, device=rpc0.device),
            xy_center=xy_center,
            xy_scale=xy_scale,
        )
        x_init, y_init = float(x0.item()), float(y0.item())

        def fun(xyz):
            r = []
            for o in tr.observations:
                l_pred, s_pred = _project_rpc(rpc_views[o.view_idx], xyz, affines_corr[o.view_idx], xy_center, xy_scale)
                r.extend([l_pred - o.line, s_pred - o.samp])
            return np.asarray(r, dtype=np.float64)

        res = least_squares(fun, x0=np.array([x_init, y_init, h0], dtype=np.float64), method="trf", max_nfev=40)
        points[ti] = res.x
    return points

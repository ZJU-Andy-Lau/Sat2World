from __future__ import annotations

import numpy as np
import torch

from baseline.free_ba import solve_free_network_ba
from baseline.triangulation import init_track_points
from baseline.types import BAConfig, Track, TrackObservation


class MockRPC:
    def __init__(self, a: np.ndarray):
        self.a = a.astype(np.float64)
        self.a_inv = np.linalg.inv(self.a)
        self.device = torch.device("cpu")
        self.HEIGHT_OFF = torch.tensor(10.0, dtype=torch.double)

    def RPC_XY2LINESAMP(self, x_in, y_in, h_in, output_type='tensor', xy_center=None, xy_scale=None):
        del h_in, output_type, xy_center, xy_scale
        x = float(torch.as_tensor(x_in).reshape(-1)[0].item())
        y = float(torch.as_tensor(y_in).reshape(-1)[0].item())
        p = self.a @ np.array([x, y, 1.0])
        return torch.tensor([p[0]], dtype=torch.double), torch.tensor([p[1]], dtype=torch.double)

    def RPC_LINESAMP2XY(self, line_in, samp_in, h_in, output_type='tensor', xy_center=None, xy_scale=None):
        del h_in, output_type, xy_center, xy_scale
        l = float(torch.as_tensor(line_in).reshape(-1)[0].item())
        s = float(torch.as_tensor(samp_in).reshape(-1)[0].item())
        p = self.a_inv @ np.array([l, s, 1.0])
        return torch.tensor([p[0]], dtype=torch.double), torch.tensor([p[1]], dtype=torch.double)


def test_free_ba_smoke_recovers_affine():
    rpc0 = MockRPC(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
    rpc1 = MockRPC(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
    true_corr_1 = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, 3.0]], dtype=np.float64)

    pts = np.array([[10, 20, 10], [30, 15, 10], [45, 60, 10], [70, 50, 10]], dtype=np.float64)
    tracks = []
    for i, p in enumerate(pts):
        l0, s0 = rpc0.RPC_XY2LINESAMP(torch.tensor([p[0]]), torch.tensor([p[1]]), torch.tensor([p[2]]))
        l1, s1 = rpc1.RPC_XY2LINESAMP(torch.tensor([p[0]]), torch.tensor([p[1]]), torch.tensor([p[2]]))
        obs1 = true_corr_1[:, :2] @ np.array([float(l1), float(s1)]) + true_corr_1[:, 2]
        tr = Track(
            track_id=i,
            observations=[
                TrackObservation(view_idx=0, line=float(l0), samp=float(s0)),
                TrackObservation(view_idx=1, line=float(obs1[0]), samp=float(obs1[1])),
            ],
        )
        tracks.append(tr)

    pts_init = init_track_points(tracks, [rpc0, rpc1], np.array([[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]], dtype=np.float64), None, None)
    sol = solve_free_network_ba(
        tracks=tracks,
        rpc_views=[rpc0, rpc1],
        points_init=pts_init,
        ref_view_idx=0,
        xy_center=None,
        xy_scale=None,
        cfg=BAConfig(max_iterations=80, huber_delta=1.0),
    )
    assert np.allclose(sol.affines_correction[0], np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64), atol=1e-6)
    assert np.allclose(sol.affines_correction[1][:, 2], true_corr_1[:, 2], atol=0.5)

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import BaselineResult
from .utils import save_json


def save_baseline_outputs(save_dir: str | Path, result: BaselineResult, extra: dict | None = None) -> None:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    save_json(
        p / "estimated_affines.json",
        {
            "view_ids": result.view_ids,
            "affine_gt_forward": result.affine_gt_forward.tolist(),
            "affine_gt_correction": result.affine_gt_correction.tolist(),
            "affine_est_correction": result.affine_est_correction.tolist(),
        },
    )
    tracks_payload = []
    for ti, tr in enumerate(result.tracks):
        tracks_payload.append(
            {
                "track_id": tr.track_id,
                "height": float(result.points_xyz[ti, 2]),
                "xyz": result.points_xyz[ti].tolist(),
                "observations": [o.__dict__ for o in tr.observations],
            }
        )
    save_json(p / "tracks_with_height.json", {"tracks": tracks_payload})
    payload = {"scene_id": result.scene_id, "metrics": result.metrics}
    if extra is not None:
        payload["extra"] = extra
    save_json(p / "baseline_report.json", payload)
    np.savez(p / "tracks_with_height.npz", points_xyz=result.points_xyz)

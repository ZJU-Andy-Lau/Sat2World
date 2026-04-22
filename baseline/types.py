from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PairMatch:
    view_i: int
    view_j: int
    matches: np.ndarray  # [N,4] (line_i, samp_i, line_j, samp_j)
    scores: np.ndarray | None = None
    raw_count: int = 0


@dataclass
class TrackObservation:
    view_idx: int
    line: float
    samp: float
    score: float = 1.0


@dataclass
class Track:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)


@dataclass
class BAConfig:
    robust_loss: str = "huber"
    huber_delta: float = 2.0
    max_iterations: int = 100
    affine_prior_weight: float = 0.0
    outlier_threshold_px: float = 6.0
    min_track_length: int = 2


@dataclass
class BASolution:
    affines_correction: np.ndarray  # [V,2,3], observed->true
    points_xyz: np.ndarray  # [T,3], (x,y,h)
    valid_track_mask: np.ndarray
    reproj_before: dict[str, float]
    reproj_after: dict[str, float]
    num_iterations: int
    kept_observation_ratio: float


@dataclass
class BaselineResult:
    scene_id: int
    view_ids: list[int]
    affine_gt_forward: np.ndarray
    affine_gt_correction: np.ndarray
    affine_est_correction: np.ndarray
    tracks: list[Track]
    points_xyz: np.ndarray
    metrics: dict[str, Any]

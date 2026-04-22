from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from dataset.io import read_height_tif, read_image_tif, read_rpc_file
from dataset.perturbation import PerturbationConfig, build_synthetic_rpc_inputs, identity_affine_2x3

from .evaluator import evaluate_affine, evaluate_track_heights
from .free_ba import solve_free_network_ba
from .matcher import build_matcher
from .overlap import estimate_candidate_pairs
from .pairwise_matching import run_pairwise_matching
from .track_builder import build_tracks
from .triangulation import init_track_points
from .types import BAConfig, BaselineResult


@dataclass
class PipelineConfig:
    matcher: str = "sift"
    matcher_weights: str | None = None
    device: str = "cpu"
    max_pairs: int = 0
    min_track_length: int = 2
    max_matches_per_pair: int = 2000
    robust_loss: str = "huber"
    huber_delta: float = 2.0
    max_iterations: int = 100
    outlier_threshold_px: float = 6.0


def run_free_ba_pipeline(
    scene_id: int,
    selected_views,
    crop_windows,
    scene_xy_center,
    scene_xy_scale,
    apply_random_init_error: bool,
    perturb_cfg: PerturbationConfig,
    seed: int,
    cfg: PipelineConfig,
):
    images = []
    heights = []
    rpc_gt = []
    for v, win in zip(selected_views, crop_windows):
        img = read_image_tif(v.image_path, window=win).numpy()
        hgt, _ = read_height_tif(v.height_path, window=win)
        rpc = read_rpc_file(v.rpc_path)
        rpc.LINE_OFF = rpc.LINE_OFF - float(win[0])
        rpc.SAMP_OFF = rpc.SAMP_OFF - float(win[1])
        images.append(img)
        heights.append(hgt.numpy()[0])
        rpc_gt.append(rpc)

    rng = np.random.default_rng(seed)
    if apply_random_init_error:
        rpc_init, aff_gt_fwd, aff_gt_corr = build_synthetic_rpc_inputs(
            rpc_gt_views=rpc_gt,
            ref_view_idx=0,
            rng=rng,
            perturb_cfg=perturb_cfg,
            dtype=np.float32,
            device=None,
        )
        aff_gt_fwd_np = aff_gt_fwd.detach().cpu().numpy().astype(np.float64)
        aff_gt_corr_np = aff_gt_corr.detach().cpu().numpy().astype(np.float64)
    else:
        rpc_init = [copy.deepcopy(r) for r in rpc_gt]
        eye = identity_affine_2x3(dtype=torch.float32)
        eye_np = eye.detach().cpu().numpy()
        aff_gt_fwd_np = np.stack([eye_np for _ in rpc_gt], axis=0).astype(np.float64)
        aff_gt_corr_np = np.stack([eye_np for _ in rpc_gt], axis=0).astype(np.float64)

    matcher = build_matcher(cfg.matcher, weights_path=cfg.matcher_weights, device=cfg.device)
    pairs = estimate_candidate_pairs(rpc_init, [img.shape[-2:] for img in images], scene_xy_center, scene_xy_scale)
    if cfg.max_pairs > 0:
        pairs = pairs[: cfg.max_pairs]
    pair_matches = run_pairwise_matching(images, matcher, pairs, max_matches_per_pair=cfg.max_matches_per_pair)
    tracks = build_tracks(pair_matches, min_track_length=cfg.min_track_length)

    aff_init = np.tile(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64), (len(rpc_init), 1, 1))
    points_init = init_track_points(tracks, rpc_init, aff_init, scene_xy_center, scene_xy_scale)
    ba_cfg = BAConfig(
        robust_loss=cfg.robust_loss,
        huber_delta=cfg.huber_delta,
        max_iterations=cfg.max_iterations,
        outlier_threshold_px=cfg.outlier_threshold_px,
        min_track_length=cfg.min_track_length,
    )
    sol = solve_free_network_ba(
        tracks=tracks,
        rpc_views=rpc_init,
        points_init=points_init,
        ref_view_idx=0,
        xy_center=scene_xy_center,
        xy_scale=scene_xy_scale,
        cfg=ba_cfg,
    )

    affine_metrics = evaluate_affine(sol.affines_correction, aff_gt_corr_np, aff_gt_fwd_np, images[0].shape[-2:], ref_view_idx=0)
    height_metrics = evaluate_track_heights(sol.points_xyz, tracks, rpc_gt, heights, scene_xy_center, scene_xy_scale)

    metrics = {
        **affine_metrics,
        "height_metrics": height_metrics,
        "pair_stats": {
            "num_pairs": len(pairs),
            "raw_matches": int(sum(p.raw_count for p in pair_matches)),
            "filtered_matches": int(sum(p.matches.shape[0] for p in pair_matches)),
        },
        "tracks": {
            "num_tracks": len(tracks),
            "avg_obs_per_track": float(np.mean([len(t.observations) for t in tracks])) if tracks else 0.0,
        },
        "ba": {
            "reproj_before": sol.reproj_before,
            "reproj_after": sol.reproj_after,
            "num_iterations": sol.num_iterations,
            "kept_observation_ratio": sol.kept_observation_ratio,
        },
    }

    return BaselineResult(
        scene_id=scene_id,
        view_ids=[v.view_id for v in selected_views],
        affine_gt_forward=aff_gt_fwd_np,
        affine_gt_correction=aff_gt_corr_np,
        affine_est_correction=sol.affines_correction,
        tracks=tracks,
        points_xyz=sol.points_xyz,
        metrics=metrics,
    ), {
        "config": asdict(cfg),
        "num_pair_matches": len(pair_matches),
    }

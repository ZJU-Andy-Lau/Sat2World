from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.perturbation import PerturbationConfig
from scripts.inference import build_inference_batch, load_scene, parse_view_indices, select_views

from baseline.io import save_baseline_outputs
from baseline.pipeline import PipelineConfig, run_free_ba_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run baseline free-network BA")
    p.add_argument("--scene-dir", type=str, required=True)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--view-num", type=int, default=4)
    p.add_argument("--view-idxs", type=str, default="")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--apply-random-init-error", action="store_true")
    p.add_argument("--tx-range", type=float, nargs=2, default=[-20.0, 20.0])
    p.add_argument("--ty-range", type=float, nargs=2, default=[-20.0, 20.0])
    p.add_argument("--scale-range", type=float, nargs=2, default=[-1e-4, 1e-4])
    p.add_argument("--shear-range", type=float, nargs=2, default=[-1e-4, 1e-4])

    p.add_argument("--matcher", type=str, default="sift", choices=["sift", "loftr"])
    p.add_argument("--matcher-weights", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--min-track-length", type=int, default=2)
    p.add_argument("--max-matches-per-pair", type=int, default=2000)

    p.add_argument("--robust-loss", type=str, default="huber", choices=["linear", "huber", "cauchy", "soft_l1"])
    p.add_argument("--huber-delta", type=float, default=2.0)
    p.add_argument("--max-iterations", type=int, default=120)
    p.add_argument("--outlier-threshold-px", type=float, default=6.0)
    p.add_argument("--save-debug-visualization", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene = load_scene(args.scene_dir)
    selected = select_views(scene, args.view_num, parse_view_indices(args.view_idxs), args.seed)

    pert_cfg = PerturbationConfig(
        tx_range=(float(args.tx_range[0]), float(args.tx_range[1])),
        ty_range=(float(args.ty_range[0]), float(args.ty_range[1])),
        scale_range=(float(args.scale_range[0]), float(args.scale_range[1])),
        shear_range=(float(args.shear_range[0]), float(args.shear_range[1])),
    )

    batch, diag = build_inference_batch(
        selected_views=selected,
        scene_id=scene.scene_id,
        crop_size=args.crop_size,
        seed=args.seed,
        perturb_cfg=pert_cfg,
        apply_random_init_error=False,
    )

    cfg = PipelineConfig(
        matcher=args.matcher,
        matcher_weights=args.matcher_weights,
        device=args.device,
        max_pairs=args.max_pairs,
        min_track_length=args.min_track_length,
        max_matches_per_pair=args.max_matches_per_pair,
        robust_loss=args.robust_loss,
        huber_delta=args.huber_delta,
        max_iterations=args.max_iterations,
        outlier_threshold_px=args.outlier_threshold_px,
    )

    result, extra = run_free_ba_pipeline(
        scene_id=scene.scene_id,
        selected_views=selected,
        crop_windows=diag["crop_windows"],
        scene_xy_center=batch["scene_xy_center"][0],
        scene_xy_scale=batch["scene_xy_scale"][0],
        apply_random_init_error=args.apply_random_init_error,
        perturb_cfg=pert_cfg,
        seed=args.seed,
        cfg=cfg,
    )

    out_dir = Path(args.save_dir)
    save_baseline_outputs(out_dir, result, extra=extra)
    print(f"[baseline] done. outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

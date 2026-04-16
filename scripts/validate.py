"""Sat2World 独立验证脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World Validate")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--work-dir", type=str, default="")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--local-rank", type=int, default=0)
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(cfg: dict[str, Any]) -> Sat2World:
    from model import Sat2World, Sat2WorldCfg

    m = cfg.get("model", {})
    scfg = Sat2WorldCfg()
    scfg.backbone.dino_weight_path = str(m.get("dino_weight_path", scfg.backbone.dino_weight_path))
    scfg.encoder.dim = int(m.get("embed_dim", scfg.encoder.dim))
    scfg.encoder.num_heads = int(m.get("encoder_num_heads", scfg.encoder.num_heads))
    scfg.encoder.num_layers = int(m.get("encoder_depth", scfg.encoder.num_layers))
    scfg.encoder.ffn_ratio = float(m.get("encoder_ffn_ratio", scfg.encoder.ffn_ratio))
    scfg.encoder.dropout = float(m.get("encoder_dropout", scfg.encoder.dropout))
    scfg.encoder.num_scene_tokens = int(m.get("encoder_num_scene_tokens", scfg.encoder.num_scene_tokens))
    scfg.encoder.aa_order = tuple(m.get("encoder_aa_order", list(scfg.encoder.aa_order)))
    scfg.encoder.aa_block_size = int(m.get("encoder_aa_block_size", scfg.encoder.aa_block_size))
    scfg.encoder.qkv_bias = bool(m.get("encoder_qkv_bias", scfg.encoder.qkv_bias))
    scfg.encoder.proj_bias = bool(m.get("encoder_proj_bias", scfg.encoder.proj_bias))
    scfg.encoder.ffn_bias = bool(m.get("encoder_ffn_bias", scfg.encoder.ffn_bias))
    scfg.encoder.qk_norm = bool(m.get("encoder_qk_norm", scfg.encoder.qk_norm))
    scfg.encoder.fused_attn = bool(m.get("encoder_fused_attn", scfg.encoder.fused_attn))
    scfg.encoder.rope_freq = float(m.get("encoder_rope_freq", scfg.encoder.rope_freq))
    scfg.encoder.drop_path_rate = float(m.get("encoder_drop_path_rate", scfg.encoder.drop_path_rate))
    scfg.encoder.init_values = float(m.get("encoder_layerscale_init_values", scfg.encoder.init_values))
    scfg.intermediate_layer_idx = tuple(int(x) for x in m.get("intermediate_layer_idx", list(scfg.intermediate_layer_idx)))
    scfg.dense_pos_embed = bool(m.get("dense_pos_embed", scfg.dense_pos_embed))
    scfg.dense_down_ratio = int(m.get("dense_down_ratio", scfg.dense_down_ratio))
    scfg.dense_frames_chunk_size = int(m.get("dense_frames_chunk_size", scfg.dense_frames_chunk_size))
    scfg.task_adapter_depth = int(m.get("task_adapter_depth", scfg.task_adapter_depth))
    scfg.affine_head.diag_scale = float(m.get("affine_diag_scale", scfg.affine_head.diag_scale))
    scfg.affine_head.offdiag_scale = float(m.get("affine_offdiag_scale", scfg.affine_head.offdiag_scale))
    scfg.affine_head.trans_scale = float(m.get("affine_trans_scale", scfg.affine_head.trans_scale))
    scfg.height_bins = int(m.get("height_bins", scfg.height_bins))
    point_bins_legacy = int(m.get("point_bins", scfg.point_bins_xy))
    scfg.point_bins_xy = int(m.get("point_bins_xy", point_bins_legacy))
    scfg.point_bins_h = int(m.get("point_bins_h", point_bins_legacy))
    scfg.sh_dim = int(m.get("sh_dim", scfg.sh_dim))
    return Sat2World(scfg)


def build_objective(cfg: dict[str, Any], geometry_ops: Any) -> RPCAnySplatTrainingObjective:
    from loss.affine_loss import AffineGridLossCfg, AffinePairwiseGeometryLossCfg
    from loss.normal_loss import PointNormalLossCfg
    from loss.total_loss import LossWeightScheduler, RPCAnySplatTrainingObjective

    lcfg = cfg.get("loss", {})
    pair_cfg = AffinePairwiseGeometryLossCfg(
        anchors_per_pair=int(lcfg.get("anchors_per_pair", 256)),
        max_pairs=lcfg.get("max_pairs", None),
        sample_from_valid_only=bool(lcfg.get("sample_from_valid_only", True)),
    )
    grid_cfg = AffineGridLossCfg(
        grid_h=int(lcfg.get("affine_grid_h", 16)),
        grid_w=int(lcfg.get("affine_grid_w", 16)),
    )
    scheduler = LossWeightScheduler(
        warmup_steps_geom_only=int(lcfg.get("warmup_steps_geom_only", 1000)),
        render_ramp_steps=int(lcfg.get("render_ramp_steps", 2000)),
    )
    normal_cfg = PointNormalLossCfg(
        w_cos=float(lcfg.get("normal_w_cos", 1.0)),
        w_l1=float(lcfg.get("normal_w_l1", 0.5)),
        eps=float(lcfg.get("normal_eps", 1e-6)),
        sign_invariant=bool(lcfg.get("normal_sign_invariant", True)),
        detach_gt=bool(lcfg.get("normal_detach_gt", True)),
    )
    return RPCAnySplatTrainingObjective(
        geometry_ops=geometry_ops,
        affine_grid_cfg=grid_cfg,
        affine_pair_cfg=pair_cfg,
        normal_cfg=normal_cfg,
        height_beta=float(lcfg.get("height_beta", 1.0)),
        point_beta=float(lcfg.get("point_beta", 1.0)),
        scale_min=float(lcfg.get("scale_min", 1e-4)),
        scale_max=float(lcfg.get("scale_max", 0.5)),
        scheduler=scheduler,
    )


def main() -> None:
    from dataset import build_dataset, rpc_scene_collate_fn
    from engine import (
        TensorBoardMonitor,
        Trainer,
        configure_cuda_runtime,
        destroy_distributed,
        init_distributed,
        is_main_process,
        resume_from_checkpoint,
        seed_everything,
        wrap_ddp,
    )
    from render import RPCGaussianRenderer, RPCGaussianRendererCfg

    args = parse_args()
    cfg = load_cfg(args.config)

    system_cfg = cfg.get("system", {})
    if args.work_dir:
        system_cfg["work_dir"] = args.work_dir
    if args.seed >= 0:
        system_cfg["seed"] = args.seed

    work_dir = Path(system_cfg.get("work_dir", "work_dirs/default"))
    work_dir.mkdir(parents=True, exist_ok=True)

    dist_state = init_distributed(backend=str(system_cfg.get("ddp_backend", "nccl")))
    device = dist_state["device"]
    seed_everything(int(system_cfg.get("seed", 42)))
    configure_cuda_runtime(
        {
            "cudnn_benchmark": bool(system_cfg.get("cudnn_benchmark", True)),
            "allow_tf32": bool(system_cfg.get("allow_tf32", True)),
        }
    )

    data_cfg = cfg.get("data", {})
    loader_cfg = data_cfg.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", 4))
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = loader_cfg.get("prefetch_factor", None)
    if num_workers <= 0:
        # DataLoader 约束：num_workers==0 时 persistent_workers 必须为 False。
        persistent_workers = False
        prefetch_factor = None
    val_dataset = build_dataset(mode="val", **data_cfg.get("val", {}))
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist_state["distributed"] else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(loader_cfg.get("val_batch_size", 1)),
        num_workers=num_workers,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=rpc_scene_collate_fn,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    model = build_model(cfg).to(device)
    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)
    renderer = RPCGaussianRenderer(model.rpc_ops, rcfg)
    objective = build_objective(cfg, model.rpc_ops)
    model = wrap_ddp(model, device)

    monitor = None
    if is_main_process() and bool(cfg.get("logging", {}).get("tensorboard", {}).get("enable", True)):
        tb_cfg = cfg.get("logging", {}).get("tensorboard", {})
        monitor = TensorBoardMonitor(log_dir=str(tb_cfg.get("log_dir", work_dir / "tb_val")), is_enabled=True)

    _ = resume_from_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location="cpu",
        model_strict=bool(cfg.get("system", {}).get("checkpoint_model_strict", False)),
        load_model_only=True,
        restore_rng=False,
    )

    trainer_cfg = dict(cfg.get("train", {}))
    trainer_cfg["enable_render_train"] = False
    trainer_cfg["enable_render_val"] = bool(cfg.get("train", {}).get("enable_render_val", True))

    trainer = Trainer(
        cfg=trainer_cfg,
        model=model,
        renderer=renderer,
        objective=objective,
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4),
        scheduler=None,
        train_loader=val_loader,
        val_loader=val_loader,
        device=device,
        monitor=monitor,
        scaler=None,
        distributed_state=dist_state,
        work_dir=str(work_dir),
        resume_state=None,
    )
    trainer.validate()

    if monitor is not None:
        monitor.close()
    destroy_distributed()


if __name__ == "__main__":
    main()

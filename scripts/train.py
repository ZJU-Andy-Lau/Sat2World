"""Sat2World 训练入口。

核心约定提示：
- 磁盘 RPC 为 GT 已平差 RPC；
- dataset 中 rpc_init 由 GT RPC 注入 forward 扰动（true->observed）得到；
- 模型预测 affine_pred 为 correction（observed->true）；
- 渲染使用 outputs['rpc_corrected']，双路径中心分别来自 rpc+height 与 point_latlon_norm+height_abs。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World Train")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--work-dir", type=str, default="")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--sanity-only", action="store_true")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--local-rank", type=int, default=0)
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    system = cfg.setdefault("system", {})
    if args.work_dir:
        system["work_dir"] = args.work_dir
    if args.seed >= 0:
        system["seed"] = int(args.seed)
    if args.eval_only:
        system["eval_only"] = True
    if args.resume:
        system["resume_path"] = args.resume
    if args.checkpoint:
        system["checkpoint_path"] = args.checkpoint
    return cfg


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

    scfg.point_bins_latlon = int(m.get("point_bins_latlon", scfg.point_bins_latlon))
    scfg.geometry_feature_dim = int(m.get("geometry_feature_dim", scfg.geometry_feature_dim))
    scfg.sh_dim = int(m.get("sh_dim", scfg.sh_dim))
    scfg.height_anchor_scale = float(m.get("height_anchor_scale", scfg.height_anchor_scale))
    scfg.height_local_scale = float(m.get("height_local_scale", scfg.height_local_scale))
    scfg.height_z_max = float(m.get("height_z_max", scfg.height_z_max))
    scfg.point_bin_size_latlon = float(m.get("point_bin_size_latlon", scfg.point_bin_size_latlon))
    scfg.point_fine_range_latlon = float(m.get("point_fine_range_latlon", scfg.point_fine_range_latlon))
    if "center_downsample_stage_steps" in m:
        scfg.center_downsample_stage_steps = tuple(int(x) for x in m.get("center_downsample_stage_steps", scfg.center_downsample_stage_steps))
    if "center_downsample_factors" in m:
        scfg.center_downsample_factors = tuple(int(x) for x in m.get("center_downsample_factors", scfg.center_downsample_factors))
    scfg.enable_gaussian_branch = bool(m.get("enable_gaussian_branch", scfg.enable_gaussian_branch))
    scfg.nce_layer_index = int(m.get("nce_layer_index", scfg.nce_layer_index))
    scfg.nce_projector_dim = int(m.get("nce_projector_dim", scfg.nce_projector_dim))
    scfg.nce_projector_hidden_dim = int(m.get("nce_projector_hidden_dim", scfg.nce_projector_hidden_dim))
    scfg.detail_patch_size = int(m.get("detail_patch_size", scfg.detail_patch_size))
    scfg.detail_token_dim = int(m.get("detail_token_dim", scfg.detail_token_dim))
    scfg.patch_match_dim = int(m.get("patch_match_dim", scfg.patch_match_dim))
    scfg.patch_match_layers = int(m.get("patch_match_layers", scfg.patch_match_layers))
    scfg.early_global_match.match_dim = int(m.get("early_match_dim", scfg.early_global_match.match_dim))
    scfg.early_global_match.residual_hidden_dim = int(m.get("early_match_residual_hidden_dim", scfg.early_global_match.residual_hidden_dim))
    scfg.early_global_match.residual_scale = float(m.get("early_match_residual_scale", scfg.early_global_match.residual_scale))
    scfg.early_global_match.enable_residual = bool(m.get("early_match_enable_residual", scfg.early_global_match.enable_residual))
    scfg.early_projection.hidden_dim = int(m.get("early_projection_hidden_dim", scfg.early_projection.hidden_dim))
    scfg.early_height.hidden_dim = int(m.get("early_height_hidden_dim", scfg.early_height.hidden_dim))
    scfg.early_height.height_scale = float(m.get("early_height_scale", scfg.early_height.height_scale))
    return Sat2World(scfg)


def build_renderer(cfg: dict[str, Any], geometry_ops: Any) -> RPCGaussianRenderer:
    from render import RPCGaussianRenderer, RPCGaussianRendererCfg

    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)
    return RPCGaussianRenderer(geometry_ops, rcfg)


def build_objective(cfg: dict[str, Any], geometry_ops: Any, patch_matcher: torch.nn.Module) -> RPCAnySplatTrainingObjective:
    from loss.affine_loss import AffineGridLossCfg, AffinePairwiseGeometryLossCfg
    from loss.feature_nce_loss import FeatureInfoNCELossCfg
    from loss.normal_loss import PointNormalLossCfg
    from loss.patch_match_loss import PatchInternalMatchLossCfg
    from loss.point_pair_loss import PointPairwiseLossCfg
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
        stage1_steps=int(lcfg.get("stage1_steps", 5000)),
        stage2_steps=int(lcfg.get("stage2_steps", 20000)),
        abs_keep_steps=int(lcfg.get("abs_keep_steps", lcfg.get("height_abs_keep_steps", 5000))),
        base_weights={
            "lambda_affine_grid": float(lcfg.get("lambda_affine_grid", 1.0)),
            "lambda_affine_pair": float(lcfg.get("lambda_affine_pair", 1.0)),
            "lambda_affine_reg": float(lcfg.get("lambda_affine_reg", 0.1)),
            "lambda_height": float(lcfg.get("lambda_height", 1.0)),
            "lambda_height_anchor": float(lcfg.get("lambda_height_anchor", 0.5)),
            "lambda_point": float(lcfg.get("lambda_point", 1.0)),
            "lambda_height_meter_aux": float(lcfg.get("lambda_height_meter_aux", 1.0e-3)),
            "lambda_height_anchor_meter_aux": float(lcfg.get("lambda_height_anchor_meter_aux", 1.0e-3)),
            "lambda_point_reproj": float(lcfg.get("lambda_point_reproj", 0.2)),
            "lambda_height_reproj": float(lcfg.get("lambda_height_reproj", 0.2)),
            "lambda_point_pair": float(lcfg.get("lambda_point_pair", 0.2)),
            "lambda_normal_height": float(lcfg.get("lambda_normal_height", 0.2)),
            "lambda_feature_nce": float(lcfg.get("lambda_feature_nce", 0.1)),
            "lambda_patch_match": float(lcfg.get("lambda_patch_match", 0.5)),
            "lambda_center": float(lcfg.get("lambda_center", 0.2)),
            "lambda_opacity_reg": float(lcfg.get("lambda_opacity_reg", 0.01)),
            "lambda_scale_reg": float(lcfg.get("lambda_scale_reg", 0.01)),
            "lambda_render_rpc": float(lcfg.get("lambda_render_rpc", 1.0)),
            "lambda_render_point": float(lcfg.get("lambda_render_point", 1.0)),
            "render_rgb_l1": float(lcfg.get("render_rgb_l1", 1.0)),
            "render_rgb_ssim": float(lcfg.get("render_rgb_ssim", 0.2)),
            "render_height": float(lcfg.get("render_height", 0.5)),
            "render_alpha": float(lcfg.get("render_alpha", 0.01)),
        },
        ramp_mode=str(lcfg.get("ramp_mode", "linear")),
    )
    feature_nce_cfg = FeatureInfoNCELossCfg(
        temperature=float(lcfg.get("feature_nce_temperature", 0.1)),
        max_pairs=int(lcfg.get("feature_nce_max_pairs", 4096)),
        match_max_pair=int(lcfg.get("feature_nce_match_max_pair", lcfg.get("match_max_pair", 12))),
    )
    normal_cfg = PointNormalLossCfg(
        w_cos=float(lcfg.get("normal_w_cos", 1.0)),
        w_l1=float(lcfg.get("normal_w_l1", 0.5)),
        eps=float(lcfg.get("normal_eps", 1e-6)),
        sign_invariant=bool(lcfg.get("normal_sign_invariant", True)),
        detach_gt=bool(lcfg.get("normal_detach_gt", True)),
    )
    point_pair_cfg = PointPairwiseLossCfg(
        grid_h=int(lcfg.get("point_pair_grid_h", 64)),
        grid_w=int(lcfg.get("point_pair_grid_w", 64)),
    )
    patch_match_cfg = PatchInternalMatchLossCfg(
        patch_size=int(cfg.get("model", {}).get("detail_patch_size", 16)),
        subpix_weight=float(lcfg.get("patch_match_subpix_weight", 0.25)),
        max_pairs=int(lcfg.get("patch_match_max_pairs", 4096)),
        match_max_pair=int(lcfg.get("patch_match_match_max_pair", lcfg.get("match_max_pair", 12))),
    )
    return RPCAnySplatTrainingObjective(
        geometry_ops=geometry_ops,
        affine_grid_cfg=grid_cfg,
        affine_pair_cfg=pair_cfg,
        point_pair_cfg=point_pair_cfg,
        feature_nce_cfg=feature_nce_cfg,
        patch_match_cfg=patch_match_cfg,
        patch_matcher=patch_matcher,
        normal_cfg=normal_cfg,
        height_beta=float(lcfg.get("height_beta", 1.0)),
        point_beta=float(lcfg.get("point_beta", 1.0)),
        height_z_beta_meter=(None if "height_z_beta_meter" not in lcfg else float(lcfg.get("height_z_beta_meter"))),
        height_anchor_z_beta_meter=float(lcfg.get("height_anchor_z_beta_meter", 5.0)),
        scale_min=float(lcfg.get("scale_min", 1e-4)),
        scale_max=float(lcfg.get("scale_max", 0.5)),
        scheduler=scheduler,
    )


def build_optimizer_and_scheduler(cfg: dict[str, Any], model: torch.nn.Module):
    ocfg = cfg.get("optim", {})
    lr = float(ocfg.get("base_lr", 1e-4))
    wd = float(ocfg.get("weight_decay", 0.01))
    betas = tuple(ocfg.get("betas", [0.9, 0.999]))
    eps = float(ocfg.get("eps", 1e-8))
    backbone_mult = float(ocfg.get("backbone_lr_mult", 1.0))

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group_lr = lr * (backbone_mult if "backbone" in name else 1.0)
        no_decay = name.endswith("bias") or ("norm" in name.lower()) or ("bn" in name.lower())
        item = {"params": [p], "lr": group_lr, "weight_decay": 0.0 if no_decay else wd}
        (no_decay_params if no_decay else decay_params).append(item)

    optimizer = torch.optim.AdamW(decay_params + no_decay_params, lr=lr, betas=betas, eps=eps)

    warmup_steps = int(ocfg.get("warmup_steps", 1000))
    max_steps = int(ocfg.get("max_steps", 100000))
    min_lr_ratio = float(ocfg.get("min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def build_dataloaders(cfg: dict[str, Any], distributed: bool):
    from dataset import build_dataset, rpc_scene_collate_fn

    data_cfg = cfg.get("data", {})
    loader_cfg = data_cfg.get("loader", {})
    t0 = time.perf_counter()
    t_last = t0

    def _log(stage: str) -> None:
        nonlocal t_last
        now = time.perf_counter()
        print(
            f"[startup][dataloader] {stage} | step={now - t_last:.2f}s total={now - t0:.2f}s",
            flush=True,
        )
        t_last = now

    train_dataset = build_dataset(mode="train", **data_cfg.get("train", {}))
    _log("train_dataset_built")
    val_dataset = build_dataset(mode="val", **data_cfg.get("val", {}))
    _log("val_dataset_built")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    _log("samplers_built")

    num_workers = int(loader_cfg.get("num_workers", 4))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    persistent_workers = bool(loader_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = loader_cfg.get("prefetch_factor", None)
    if num_workers <= 0:
        # DataLoader 约束：num_workers==0 时 persistent_workers 必须为 False。
        persistent_workers = False
        prefetch_factor = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        num_workers=num_workers,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        pin_memory=pin_memory,
        drop_last=bool(loader_cfg.get("drop_last", True)),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=rpc_scene_collate_fn,
    )
    _log("train_loader_constructed")
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(loader_cfg.get("val_batch_size", 1)),
        num_workers=num_workers,
        sampler=val_sampler,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=rpc_scene_collate_fn,
    )
    _log("val_loader_constructed")
    return train_loader, val_loader


def run_sanity_once(model, renderer, objective, loader, device) -> None:
    """闭环自检：仅前向一次，不反向。"""
    batch = next(iter(loader))
    from engine.distributed import move_batch_to_device

    batch = move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        render_outputs = renderer.render_paths(outputs, batch, mode="train")
        total_loss, scalar_dict, _ = objective(outputs, batch, global_step=0, epoch=0, render_outputs=render_outputs, mode="train")

    print("[sanity] outputs keys:", sorted(list(outputs.keys()))[:10], "...")
    print("[sanity] loss_total:", float(total_loss.detach().cpu().item()))
    print("[sanity] render rpc targets:", int(render_outputs["rpc"].get("num_targets", 0)))
    print("[sanity] render point targets:", int(render_outputs["point"].get("num_targets", 0)))
    print("[sanity] scalar sample:", {k: scalar_dict[k] for k in list(scalar_dict.keys())[:8]})


def main() -> None:
    from engine import (
        TensorBoardMonitor,
        Trainer,
        auto_resume_latest,
        configure_cuda_runtime,
        destroy_distributed,
        init_distributed,
        is_main_process,
        resume_from_checkpoint,
        seed_everything,
        wrap_ddp,
    )

    args = parse_args()
    cfg = apply_cli_overrides(load_cfg(args.config), args)
    startup_t0 = time.perf_counter()
    startup_last = startup_t0

    def _startup_log(stage: str, rank: int | None = None) -> None:
        nonlocal startup_last
        now = time.perf_counter()
        step_sec = now - startup_last
        total_sec = now - startup_t0
        rank_str = "?" if rank is None else str(rank)
        print(
            f"[startup][rank={rank_str}] {stage} | step={step_sec:.2f}s total={total_sec:.2f}s",
            flush=True,
        )
        startup_last = now

    _startup_log("config_loaded")

    system_cfg = cfg.get("system", {})
    work_dir = Path(system_cfg.get("work_dir", "work_dirs/default"))
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoints_dir = Path(checkpoint_cfg.get("checkpoints_dir", str(work_dir / "checkpoints")))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    _startup_log("workdir_and_checkpoint_dirs_ready")

    dist_state = init_distributed(backend=str(system_cfg.get("ddp_backend", "nccl")))
    device = dist_state["device"]
    _startup_log("distributed_initialized", rank=int(dist_state["rank"]))

    seed_everything(int(system_cfg.get("seed", 42)))
    configure_cuda_runtime(
        {
            "cudnn_benchmark": bool(system_cfg.get("cudnn_benchmark", True)),
            "allow_tf32": bool(system_cfg.get("allow_tf32", True)),
        }
    )
    _startup_log("seed_and_cuda_runtime_configured", rank=int(dist_state["rank"]))

    train_loader, val_loader = build_dataloaders(cfg, distributed=dist_state["distributed"])
    _startup_log("dataloaders_built", rank=int(dist_state["rank"]))

    model = build_model(cfg).to(device)
    _startup_log("model_built_and_to_device", rank=int(dist_state["rank"]))
    renderer = build_renderer(cfg, model.rpc_ops)
    _startup_log("renderer_built", rank=int(dist_state["rank"]))
    objective = build_objective(cfg, model.rpc_ops, model.patch_matcher)
    _startup_log("objective_built", rank=int(dist_state["rank"]))
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    _startup_log("optimizer_and_scheduler_built", rank=int(dist_state["rank"]))

    amp_dtype = str(cfg.get("train", {}).get("amp_dtype", "fp16"))
    use_scaler = bool(cfg.get("train", {}).get("enable_grad_scaler", True)) and amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    _startup_log("grad_scaler_ready", rank=int(dist_state["rank"]))

    model = wrap_ddp(model, device)

    if args.sanity_only:
        run_sanity_once(model, renderer, objective, train_loader, device)
        destroy_distributed()
        return

    monitor = None
    if is_main_process() and bool(cfg.get("logging", {}).get("tensorboard", {}).get("enable", True)):
        tb_cfg = cfg.get("logging", {}).get("tensorboard", {})
        log_dir = tb_cfg.get("log_dir", str(work_dir / "tb"))
        monitor = TensorBoardMonitor(
            log_dir=log_dir,
            is_enabled=True,
            enable_mesh=bool(tb_cfg.get("enable_mesh", True)),
            image_max_views=int(tb_cfg.get("image_max_views", 4)),
            max_pointcloud_points=int(tb_cfg.get("max_pointcloud_points", 8192)),
            flush_secs=int(tb_cfg.get("flush_secs", 30)),
            min_free_mb=float(tb_cfg.get("min_free_mb", 100.0)),
            disk_check_interval_sec=float(tb_cfg.get("disk_check_interval_sec", 1.0)),
            low_disk_warn_interval_sec=float(tb_cfg.get("low_disk_warn_interval_sec", 300.0)),
            reopen_interval_sec=float(tb_cfg.get("reopen_interval_sec", 30.0)),
            skip_when_low_disk=bool(tb_cfg.get("skip_when_low_disk", True)),
        )

    resume_state = None
    resume_path = str(system_cfg.get("resume_path", ""))
    checkpoint_path = str(system_cfg.get("checkpoint_path", ""))
    auto_resume = bool(system_cfg.get("auto_resume", True))
    checkpoint_load_model_only = bool(system_cfg.get("checkpoint_load_model_only", False))
    checkpoint_model_strict = bool(system_cfg.get("checkpoint_model_strict", False))

    # 新增：按 shape 部分加载模型参数
    checkpoint_match_by_shape = bool(system_cfg.get("checkpoint_match_by_shape", False))
    checkpoint_strip_prefixes = tuple(system_cfg.get("checkpoint_strip_prefixes", ["module."]))
    checkpoint_ignore_prefixes = tuple(system_cfg.get("checkpoint_ignore_prefixes", []))
    verbose_model_load = bool(system_cfg.get("verbose_model_load", True))

    # 结构发生变化并启用按 shape 加载时，强制按“仅加载模型参数”处理
    effective_load_model_only = bool(checkpoint_load_model_only or checkpoint_match_by_shape)

    if checkpoint_path:
        resume_path = checkpoint_path
    elif args.checkpoint:
        resume_path = args.checkpoint
    elif args.resume:
        resume_path = args.resume
    elif resume_path == "" and auto_resume:
        # 统一 checkpoint 配置入口：优先从 checkpoint.checkpoints_dir 自动恢复。
        last_ckpt = checkpoints_dir / "last.pt"
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
        else:
            cands = sorted(checkpoints_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            resume_path = str(cands[0]) if cands else ""
            if resume_path == "":
                resume_path = auto_resume_latest(str(work_dir)) or ""

    if resume_path:
        resume_state = resume_from_checkpoint(
            resume_path,
            model=model,
            optimizer=None if effective_load_model_only else optimizer,
            scheduler=None if effective_load_model_only else scheduler,
            scaler=None if effective_load_model_only else scaler,
            map_location="cpu",
            model_strict=checkpoint_model_strict,
            load_model_only=effective_load_model_only,
            restore_rng=(not effective_load_model_only),
            model_match_by_shape=checkpoint_match_by_shape,
            model_strip_prefixes=checkpoint_strip_prefixes,
            model_ignore_prefixes=checkpoint_ignore_prefixes,
            verbose_model_load=verbose_model_load,
        )

        # 对“只加载模型参数”的场景，不把 checkpoint 的 epoch/step 等状态交给 Trainer
        if effective_load_model_only:
            resume_state = None
        elif resume_state is not None:
            resume_state["resume_path"] = str(resume_path)

        if is_main_process() and monitor is not None:
            suffix = " (match_by_shape)" if checkpoint_match_by_shape else ""
            monitor.log_text("events/resume", f"resume from {resume_path}{suffix}", 0)

    trainer_cfg = dict(cfg.get("train", {}))
    trainer_cfg["best_metric"] = checkpoint_cfg.get("best_metric_name", trainer_cfg.get("best_metric", "loss_total"))
    trainer_cfg["best_mode"] = checkpoint_cfg.get("best_metric_mode", trainer_cfg.get("best_mode", "min"))
    trainer_cfg["checkpoints_dir"] = str(checkpoints_dir)

    trainer = Trainer(
        cfg=trainer_cfg,
        model=model,
        renderer=renderer,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        monitor=monitor,
        scaler=scaler,
        distributed_state=dist_state,
        work_dir=str(work_dir),
        log_profile="train",
        resume_state=resume_state,
    )

    if bool(system_cfg.get("eval_only", False)) or args.eval_only:
        trainer.validate()
    else:
        trainer.fit()

    if monitor is not None:
        monitor.close()
    destroy_distributed()


if __name__ == "__main__":
    main()

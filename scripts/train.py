"""Sat2World 训练入口。

核心约定提示：
- 磁盘 RPC 为 GT 已平差 RPC；
- dataset 中 rpc_init 由 GT RPC 注入 forward 扰动（true->observed）得到；
- 模型预测 affine_pred 为 correction（observed->true）；
- 渲染使用 outputs['rpc_corrected']，双路径中心分别来自 rpc+height 与 point_abs。
"""

from __future__ import annotations

import argparse
import math
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

    scfg.affine_head.diag_scale = float(m.get("affine_diag_scale", scfg.affine_head.diag_scale))
    scfg.affine_head.offdiag_scale = float(m.get("affine_offdiag_scale", scfg.affine_head.offdiag_scale))
    scfg.affine_head.trans_scale = float(m.get("affine_trans_scale", scfg.affine_head.trans_scale))

    scfg.height_bins = int(m.get("height_bins", scfg.height_bins))
    scfg.point_bins = int(m.get("point_bins", scfg.point_bins))
    scfg.sh_dim = int(m.get("sh_dim", scfg.sh_dim))
    scfg.height_bin_size = float(m.get("height_bin_size", scfg.height_bin_size))
    scfg.height_fine_range = float(m.get("height_fine_range", scfg.height_fine_range))
    scfg.point_bin_size_xy = float(m.get("point_bin_size_xy", scfg.point_bin_size_xy))
    scfg.point_bin_size_z = float(m.get("point_bin_size_z", scfg.point_bin_size_z))
    scfg.point_fine_range_xy = float(m.get("point_fine_range_xy", scfg.point_fine_range_xy))
    scfg.point_fine_range_z = float(m.get("point_fine_range_z", scfg.point_fine_range_z))
    return Sat2World(scfg)


def build_renderer(cfg: dict[str, Any], geometry_ops: Any) -> RPCGaussianRenderer:
    from render import RPCGaussianRenderer, RPCGaussianRendererCfg

    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)
    return RPCGaussianRenderer(geometry_ops, rcfg)


def build_objective(cfg: dict[str, Any], geometry_ops: Any) -> RPCAnySplatTrainingObjective:
    from loss.affine_loss import AffineGridLossCfg, AffinePairwiseGeometryLossCfg
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
        base_weights={
            "lambda_affine_grid": float(lcfg.get("lambda_affine_grid", 1.0)),
            "lambda_affine_pair": float(lcfg.get("lambda_affine_pair", 1.0)),
            "lambda_affine_reg": float(lcfg.get("lambda_affine_reg", 0.1)),
            "lambda_affine_ref": float(lcfg.get("lambda_affine_ref", 0.1)),
            "lambda_height": float(lcfg.get("lambda_height", 1.0)),
            "lambda_point": float(lcfg.get("lambda_point", 1.0)),
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
    return RPCAnySplatTrainingObjective(
        geometry_ops=geometry_ops,
        affine_grid_cfg=grid_cfg,
        affine_pair_cfg=pair_cfg,
        height_beta=float(lcfg.get("height_beta", 1.0)),
        point_beta=float(lcfg.get("point_beta", 1.0)),
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

    train_dataset = build_dataset(mode="train", **data_cfg.get("train", {}))
    val_dataset = build_dataset(mode="val", **data_cfg.get("val", {}))

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    num_workers = int(loader_cfg.get("num_workers", 4))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False))
    prefetch_factor = loader_cfg.get("prefetch_factor", None)

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

    system_cfg = cfg.get("system", {})
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

    train_loader, val_loader = build_dataloaders(cfg, distributed=dist_state["distributed"])

    model = build_model(cfg).to(device)
    renderer = build_renderer(cfg, model.rpc_ops)
    objective = build_objective(cfg, model.rpc_ops)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

    amp_dtype = str(cfg.get("train", {}).get("amp_dtype", "fp16"))
    use_scaler = bool(cfg.get("train", {}).get("enable_grad_scaler", True)) and amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

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
        )

    resume_state = None
    resume_path = str(system_cfg.get("resume_path", ""))
    checkpoint_path = str(system_cfg.get("checkpoint_path", ""))
    auto_resume = bool(system_cfg.get("auto_resume", True))

    if checkpoint_path:
        resume_path = checkpoint_path
    elif args.checkpoint:
        resume_path = args.checkpoint
    elif args.resume:
        resume_path = args.resume
    elif resume_path == "" and auto_resume:
        resume_path = auto_resume_latest(str(work_dir)) or ""

    if resume_path:
        resume_state = resume_from_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        if is_main_process() and monitor is not None:
            monitor.log_text("events/resume", f"resume from {resume_path}", 0)

    trainer_cfg = dict(cfg.get("train", {}))
    trainer_cfg["best_metric"] = cfg.get("checkpoint", {}).get("best_metric_name", trainer_cfg.get("best_metric", "loss_total"))
    trainer_cfg["best_mode"] = cfg.get("checkpoint", {}).get("best_metric_mode", trainer_cfg.get("best_mode", "min"))

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

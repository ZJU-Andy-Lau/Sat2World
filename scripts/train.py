"""训练入口脚本。

用法示例：
torchrun --nproc_per_node=4 scripts/train.py --config configs/train.yaml --work-dir work_dirs/exp1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from dataset import build_dataset, rpc_scene_collate_fn
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
)
from engine.distributed import wrap_ddp
from geometry import RPCGeometryOps
from loss.total_loss import RPCAnySplatTrainingObjective
from model import Sat2World, Sat2WorldCfg
from render import RPCGaussianRenderer, RPCGaussianRendererCfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World Train")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--work-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-rank", type=int, default=0)
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def build_model(cfg: dict[str, Any]) -> Sat2World:
    mcfg = Sat2WorldCfg()
    backbone = cfg.get("model", {}).get("backbone", {})
    if "dino_weight_path" in backbone:
        mcfg.backbone.dino_weight_path = backbone["dino_weight_path"]
    return Sat2World(mcfg)


def build_renderer(cfg: dict[str, Any], geometry_ops: RPCGeometryOps) -> RPCGaussianRenderer:
    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)
    return RPCGaussianRenderer(geometry_ops, rcfg)


def build_objective(cfg: dict[str, Any], geometry_ops: RPCGeometryOps) -> RPCAnySplatTrainingObjective:
    lcfg = cfg.get("loss", {})
    return RPCAnySplatTrainingObjective(
        geometry_ops=geometry_ops,
        height_beta=float(lcfg.get("height_beta", 1.0)),
        point_beta=float(lcfg.get("point_beta", 1.0)),
        scale_min=float(lcfg.get("scale_min", 1e-4)),
        scale_max=float(lcfg.get("scale_max", 0.5)),
    )


def build_optimizer_and_scheduler(cfg: dict[str, Any], model: torch.nn.Module):
    ocfg = cfg.get("optim", {})
    lr = float(ocfg.get("lr", 1e-4))
    wd = float(ocfg.get("weight_decay", 0.01))
    backbone_mult = float(ocfg.get("backbone_lr_mult", 1.0))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = n.endswith("bias") or ("norm" in n.lower()) or ("bn" in n.lower())
        group_lr = lr * backbone_mult if "backbone" in n else lr
        item = (p, group_lr)
        if is_no_decay:
            no_decay.append(item)
        else:
            decay.append(item)

    param_groups = []
    if decay:
        param_groups.append({"params": [x[0] for x in decay], "lr": lr, "weight_decay": wd})
    if no_decay:
        param_groups.append({"params": [x[0] for x in no_decay], "lr": lr, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    scfg = cfg.get("scheduler", {})
    max_steps = int(scfg.get("max_steps", 100000))
    warmup_steps = int(scfg.get("warmup_steps", 1000))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        t = min(max(t, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    dist_state = init_distributed(backend=str(cfg.get("ddp", {}).get("backend", "nccl")))
    device = dist_state["device"]

    seed_everything(int(args.seed))
    configure_cuda_runtime(cfg.get("runtime", {}))

    train_dataset = build_dataset(mode="train", **cfg.get("dataset", {}).get("train", {}))
    val_dataset = build_dataset(mode="val", **cfg.get("dataset", {}).get("val", {}))

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if dist_state["distributed"] else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist_state["distributed"] else None

    loader_cfg = cfg.get("dataloader", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        num_workers=int(loader_cfg.get("num_workers", 4)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        collate_fn=rpc_scene_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(loader_cfg.get("val_batch_size", 1)),
        num_workers=int(loader_cfg.get("num_workers", 4)),
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=False,
        collate_fn=rpc_scene_collate_fn,
    )

    model = build_model(cfg).to(device)
    geometry_ops = model.rpc_ops
    renderer = build_renderer(cfg, geometry_ops)
    objective = build_objective(cfg, geometry_ops)

    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

    amp_dtype = str(cfg.get("trainer", {}).get("amp_dtype", "fp16"))
    use_scaler = (amp_dtype == "fp16" and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    model = wrap_ddp(model, device)

    monitor = None
    if is_main_process():
        monitor = TensorBoardMonitor(log_dir=str(work_dir / "tb"), is_enabled=True)

    resume_state = None
    resume_path = args.resume or ""
    if resume_path == "" and bool(cfg.get("auto_resume", True)):
        resume_path = auto_resume_latest(str(work_dir)) or ""
    if args.checkpoint:
        resume_path = args.checkpoint
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

    trainer = Trainer(
        cfg=cfg.get("trainer", {}),
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

    if args.eval_only:
        trainer.validate()
    else:
        trainer.fit()

    if monitor is not None:
        monitor.close()
    destroy_distributed()


if __name__ == "__main__":
    main()

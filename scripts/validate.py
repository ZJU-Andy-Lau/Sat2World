"""独立验证脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from dataset import build_dataset, rpc_scene_collate_fn
from engine import TensorBoardMonitor, Trainer, destroy_distributed, init_distributed, is_main_process, resume_from_checkpoint
from engine.distributed import configure_cuda_runtime, seed_everything, wrap_ddp
from loss.total_loss import RPCAnySplatTrainingObjective
from model import Sat2World, Sat2WorldCfg
from render import RPCGaussianRenderer, RPCGaussianRendererCfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World Validate")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--work-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-rank", type=int, default=0)
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(cfg: dict[str, Any]) -> Sat2World:
    mcfg = Sat2WorldCfg()
    backbone = cfg.get("model", {}).get("backbone", {})
    if "dino_weight_path" in backbone:
        mcfg.backbone.dino_weight_path = backbone["dino_weight_path"]
    return Sat2World(mcfg)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    dist_state = init_distributed(backend=str(cfg.get("ddp", {}).get("backend", "nccl")))
    device = dist_state["device"]
    seed_everything(args.seed)
    configure_cuda_runtime(cfg.get("runtime", {}))

    val_dataset = build_dataset(mode="val", **cfg.get("dataset", {}).get("val", {}))
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist_state["distributed"] else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("dataloader", {}).get("val_batch_size", 1)),
        num_workers=int(cfg.get("dataloader", {}).get("num_workers", 4)),
        sampler=val_sampler,
        shuffle=False,
        collate_fn=rpc_scene_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    renderer_cfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(renderer_cfg, k):
            setattr(renderer_cfg, k, v)
    renderer = RPCGaussianRenderer(model.rpc_ops, renderer_cfg)
    objective = RPCAnySplatTrainingObjective(geometry_ops=model.rpc_ops)

    model = wrap_ddp(model, device)

    monitor = TensorBoardMonitor(log_dir=str(work_dir / "tb_val"), is_enabled=is_main_process()) if is_main_process() else None

    _ = resume_from_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location="cpu",
    )

    trainer = Trainer(
        cfg=cfg.get("trainer", {}),
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

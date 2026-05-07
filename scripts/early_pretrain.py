"""Sat2World early pretrain phase-1 scaffold.

This entry disables Gaussian/render training and uses a minimal objective over
existing patch NCE and patch-match outputs only. Dedicated early heads are left
for later phases.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World Early Pretrain")
    p.add_argument("--config", type=str, default="default_configs/early_pretrain.yaml")
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


def build_model(cfg: dict[str, Any]) -> Any:
    from scripts.train import build_model as _build_model

    return _build_model(cfg)


def build_objective(cfg: dict[str, Any], model: torch.nn.Module) -> Any:
    from loss.early_pretrain_objective import EarlyPretrainObjective, EarlyPretrainWeightCfg
    from loss.feature_nce_loss import FeatureInfoNCELossCfg
    from loss.patch_match_loss import PatchInternalMatchLossCfg

    lcfg = cfg.get("loss", {})
    weights = EarlyPretrainWeightCfg(
        lambda_feature_nce=float(lcfg.get("lambda_feature_nce", 1.0)),
        lambda_patch_match=float(lcfg.get("lambda_patch_match", 1.0)),
        lambda_match_coord=float(lcfg.get("lambda_match_coord", 1.0)),
        lambda_match_ce=float(lcfg.get("lambda_match_ce", 1.0)),
        lambda_match_cycle=float(lcfg.get("lambda_match_cycle", 0.0)),
        lambda_projection=float(lcfg.get("lambda_projection", 1.0)),
        lambda_early_height=float(lcfg.get("lambda_early_height", 1.0)),
        pixel_loss_norm=float(lcfg.get("pixel_loss_norm", cfg.get("model", {}).get("detail_patch_size", 16))),
        early_height_scale=float(lcfg.get("early_height_scale", cfg.get("model", {}).get("early_height_scale", 1000.0))),
    )
    feature_nce_cfg = FeatureInfoNCELossCfg(
        temperature=float(lcfg.get("feature_nce_temperature", 0.1)),
        max_pairs=int(lcfg.get("feature_nce_max_pairs", 4096)),
        match_max_pair=int(lcfg.get("feature_nce_match_max_pair", lcfg.get("match_max_pair", 12))),
    )
    patch_match_cfg = PatchInternalMatchLossCfg(
        patch_size=int(cfg.get("model", {}).get("detail_patch_size", 16)),
        subpix_weight=float(lcfg.get("patch_match_subpix_weight", 0.25)),
        max_pairs=int(lcfg.get("patch_match_max_pairs", 4096)),
        match_max_pair=int(lcfg.get("patch_match_match_max_pair", lcfg.get("match_max_pair", 12))),
    )
    return EarlyPretrainObjective(
        geometry_ops=model.rpc_ops,
        patch_matcher=model.patch_matcher,
        early_height_head=model.early_height_head,
        feature_nce_cfg=feature_nce_cfg,
        patch_match_cfg=patch_match_cfg,
        weights=weights,
    )

def build_optimizer_and_scheduler(cfg: dict[str, Any], model: torch.nn.Module):
    from scripts.train import build_optimizer_and_scheduler as _build_optimizer_and_scheduler

    return _build_optimizer_and_scheduler(cfg, model)


def build_dataloaders(cfg: dict[str, Any], distributed: bool):
    from scripts.train import build_dataloaders as _build_dataloaders

    return _build_dataloaders(cfg, distributed)


def run_sanity_once(model, objective, loader, device) -> None:
    """early pretrain 闭环自检：仅 model + objective 前向一次。"""
    from engine.distributed import move_batch_to_device

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        total_loss, scalar_dict, _ = objective(outputs, batch, global_step=0, epoch=0, render_outputs=None, mode="train")

    print("[early pretrain sanity] outputs keys:", sorted(list(outputs.keys()))[:10], "...")
    print("[early pretrain sanity] loss_total:", float(total_loss.detach().cpu().item()))
    print("[early pretrain sanity] scalar sample:", {k: scalar_dict[k] for k in list(scalar_dict.keys())[:8]})


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
        print(f"[startup][rank={rank_str}] {stage} | step={step_sec:.2f}s total={total_sec:.2f}s", flush=True)
        startup_last = now

    _startup_log("config_loaded")

    system_cfg = cfg.get("system", {})
    work_dir = Path(system_cfg.get("work_dir", "work_dirs/early_pretrain_default"))
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

    # Early pretrain 阶段禁用 Gaussian/render 分支（phase 1 仍复用现有 forward 主流程）。
    model_cfg = cfg.get("model", {})
    if bool(model_cfg.get("enable_gaussian_branch", True)):
        _startup_log("force_disable_gaussian_branch_for_early_pretrain", rank=int(dist_state["rank"]))
    model_cfg["enable_gaussian_branch"] = False
    cfg["model"] = model_cfg

    model = build_model(cfg).to(device)
    train_detail_encoder = bool(cfg.get("early_pretrain", {}).get("train_detail_encoder", False))
    if hasattr(model, "set_early_pretrain_only"):
        model.set_early_pretrain_only(True, train_detail_encoder=train_detail_encoder)
    else:
        raise AttributeError("Sat2World model must provide set_early_pretrain_only for early pretrain.")
    if int(dist_state["rank"]) == 0 and hasattr(model, "trainable_parameter_summary"):
        print(f"[early_pretrain] trainable_parameter_summary={model.trainable_parameter_summary()}", flush=True)
    _startup_log("model_built_and_to_device", rank=int(dist_state["rank"]))
    objective = build_objective(cfg, model)
    _startup_log("objective_built", rank=int(dist_state["rank"]))
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    _startup_log("optimizer_and_scheduler_built", rank=int(dist_state["rank"]))

    amp_dtype = str(cfg.get("train", {}).get("amp_dtype", "fp16"))
    use_scaler = bool(cfg.get("train", {}).get("enable_grad_scaler", True)) and amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    _startup_log("grad_scaler_ready", rank=int(dist_state["rank"]))

    model = wrap_ddp(model, device)

    if args.sanity_only:
        run_sanity_once(model, objective, train_loader, device)
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
    trainer_cfg["enable_render_train"] = False
    trainer_cfg["enable_render_val"] = False
    trainer_cfg["best_metric"] = checkpoint_cfg.get("best_metric_name", trainer_cfg.get("best_metric", "loss_total"))
    trainer_cfg["best_mode"] = checkpoint_cfg.get("best_metric_mode", trainer_cfg.get("best_mode", "min"))
    trainer_cfg["checkpoints_dir"] = str(checkpoints_dir)

    trainer = Trainer(
        cfg=trainer_cfg,
        model=model,
        renderer=None,
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
        log_profile="early_pretrain",
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

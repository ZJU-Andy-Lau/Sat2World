"""engine.checkpoint

Checkpoint 保存/恢复工具。

恢复语义：
- 支持恢复到 epoch 中间（epoch + step_in_epoch）；
- 支持恢复优化器、调度器、scaler、best_metric 与随机数状态；
- Trainer 可据此跳过已完成 step，避免重跑。
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from engine.distributed import is_main_process, unwrap_model


def capture_rng_state() -> dict[str, Any]:
    """采集 Python/NumPy/Torch 的随机数状态。"""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """恢复随机数状态。"""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda", None) is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _state_dict_or_none(obj: Any) -> Any:
    if obj is None:
        return None
    return obj.state_dict()


def save_checkpoint(
    path: str | os.PathLike,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_metric: float,
    cfg: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    """保存 checkpoint（主进程执行，临时文件+原子替换）。"""
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": _state_dict_or_none(optimizer),
        "scheduler": _state_dict_or_none(scheduler),
        "scaler": _state_dict_or_none(scaler),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "cfg": cfg,
        "rng_state": capture_rng_state(),
        "extra_state": extra_state or {},
    }

    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_checkpoint(
    path: str | os.PathLike,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """加载 checkpoint 并恢复对象状态。"""
    ckpt = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(ckpt["model"], strict=False)

    if optimizer is not None and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler", None) is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "step_in_epoch": int(ckpt.get("step_in_epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_metric": float(ckpt.get("best_metric", float("inf"))),
        "cfg": ckpt.get("cfg", {}),
        "extra_state": ckpt.get("extra_state", {}),
        "rng_state": ckpt.get("rng_state", None),
    }


def auto_resume_latest(work_dir: str | os.PathLike) -> str | None:
    """自动寻找最近 checkpoint。

    优先：
    1) checkpoints/last.pt
    2) checkpoints/epoch_*.pt 中最新修改时间
    """
    ckpt_dir = Path(work_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    last = ckpt_dir / "last.pt"
    if last.exists():
        return str(last)
    cands = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else None


def resume_from_checkpoint(
    path: str | os.PathLike,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """恢复 checkpoint 并恢复随机状态。"""
    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=map_location,
    )
    restore_rng_state(state.get("rng_state", None))
    return state

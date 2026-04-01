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
import time
from pathlib import Path
from typing import Any
import errno

import numpy as np
import torch

from engine.distributed import is_main_process, unwrap_model
from engine.utils import has_sufficient_disk_space


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


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    base = unwrap_model(model)
    for p in base.parameters():
        return p.device
    for _, b in base.named_buffers():
        return b.device
    return torch.device("cpu")


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """把 optimizer.state 中的 tensor 迁移到目标 device。"""
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=device)


def _safe_unlink(path: Path, retries: int = 5, base_sleep_sec: float = 0.1) -> None:
    """尽力删除临时文件；NFS 上 EBUSY 时重试并降级为告警。"""
    for i in range(max(int(retries), 1)):
        try:
            path.unlink(missing_ok=True)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            if e.errno not in (errno.EBUSY, errno.EPERM, errno.EACCES):
                raise
            if i >= retries - 1:
                print(f"[checkpoint] warning: unable to unlink temp file {path}: {e}")
                return
            time.sleep(base_sleep_sec * (2**i))


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
    min_disk_gb: float = 2.0,
) -> bool:
    """保存 checkpoint（主进程执行，临时文件+原子替换）。

    返回:
        success: 是否保存成功（空间不足或 IO 失败返回 False）。
    """
    if not is_main_process():
        return True
    path = Path(path)

    # 预检磁盘空间，避免写入中途由于磁盘满载导致崩溃
    if not has_sufficient_disk_space(path.parent, min_gb=min_disk_gb):
        print(f"[WARN] Disk space critically low (<{min_disk_gb}GB). Skipping checkpoint: {path.name}")
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Failed to create directory {path.parent}: {e}")
        return False

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
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint to {path}: {e}")
        return False
    finally:
        if tmp_path.exists():
            _safe_unlink(tmp_path)


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
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["model"], strict=False)

    if optimizer is not None and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        _move_optimizer_state_to_device(optimizer, _infer_model_device(model))
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

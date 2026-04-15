"""engine.checkpoint

Checkpoint 保存/恢复工具。

恢复语义：
- 支持恢复到 epoch 中间（epoch + step_in_epoch）；
- 支持恢复优化器、调度器、scaler、best_metric 与随机数状态；
- Trainer 可据此跳过已完成 step，避免重跑。
"""

from __future__ import annotations

import errno
import os
import random
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

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


def _strip_key_prefix(key: str, strip_prefixes: tuple[str, ...]) -> str:
    """按顺序剥离 checkpoint key 的常见前缀，例如 'module.'。"""
    for prefix in strip_prefixes:
        if prefix and key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _format_shape(x: Any) -> tuple[int, ...] | None:
    """把 tensor/parameter 的 shape 统一转成 tuple，非 tensor 返回 None。"""
    if torch.is_tensor(x):
        return tuple(int(v) for v in x.shape)
    return None


def load_state_dict_match_by_shape(
    model: torch.nn.Module,
    checkpoint_state_dict: dict[str, Any],
    *,
    strip_prefixes: tuple[str, ...] = ("module.",),
    ignore_prefixes: tuple[str, ...] = (),
    verbose: bool = True,
    max_report_items: int = 50,
) -> dict[str, Any]:
    """只加载 key 存在且 shape 完全匹配的参数/缓冲区。

    设计目标：
    - 允许模型结构发生变化（增删模块、修改维度）后，仍复用旧 checkpoint 中能复用的部分；
    - 对于同名但 shape 不匹配的层，直接跳过；
    - 对于 checkpoint 中存在、当前模型中不存在的层，直接跳过；
    - 返回详细报告，便于审计实际加载了哪些层。

    参数:
        model:
            当前模型（可带 DDP 包装）。
        checkpoint_state_dict:
            checkpoint 中的 model state_dict。
        strip_prefixes:
            尝试剥离的 key 前缀，例如 DDP 常见的 "module."。
        ignore_prefixes:
            需要显式忽略的 key 前缀。
        verbose:
            是否打印摘要。
        max_report_items:
            打印时每类最多展示多少项，防止日志过长。

    返回:
        report:
            {
                "num_model_keys": int,
                "num_ckpt_keys": int,
                "num_loaded": int,
                "loaded_keys": list[str],
                "missing_keys": list[str],
                "unexpected_keys": list[str],
                "shape_mismatch": list[dict],
                "ignored_keys": list[str],
            }
    """
    model_ref = unwrap_model(model)
    model_state = model_ref.state_dict()

    filtered_state = OrderedDict()

    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    ignored_keys: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []

    matched_model_keys: set[str] = set()

    for ckpt_key, ckpt_val in checkpoint_state_dict.items():
        norm_key = _strip_key_prefix(str(ckpt_key), strip_prefixes)

        if any(norm_key.startswith(p) for p in ignore_prefixes):
            ignored_keys.append(norm_key)
            continue

        if norm_key not in model_state:
            unexpected_keys.append(norm_key)
            continue

        model_val = model_state[norm_key]

        ckpt_shape = _format_shape(ckpt_val)
        model_shape = _format_shape(model_val)

        # 非 tensor 项通常不应出现在 state_dict 中；保守跳过
        if ckpt_shape is None or model_shape is None:
            unexpected_keys.append(norm_key)
            continue

        if ckpt_shape != model_shape:
            shape_mismatch.append(
                {
                    "key": norm_key,
                    "ckpt_shape": ckpt_shape,
                    "model_shape": model_shape,
                    "ckpt_dtype": str(getattr(ckpt_val, "dtype", "unknown")),
                    "model_dtype": str(getattr(model_val, "dtype", "unknown")),
                }
            )
            continue

        filtered_state[norm_key] = ckpt_val
        loaded_keys.append(norm_key)
        matched_model_keys.add(norm_key)

    missing_keys = [k for k in model_state.keys() if k not in matched_model_keys and not any(k.startswith(p) for p in ignore_prefixes)]

    # 这里只对“已经筛过且 shape 匹配”的部分做真正加载
    model_ref.load_state_dict(filtered_state, strict=False)

    report = {
        "num_model_keys": int(len(model_state)),
        "num_ckpt_keys": int(len(checkpoint_state_dict)),
        "num_loaded": int(len(loaded_keys)),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
        "ignored_keys": ignored_keys,
    }

    if verbose and is_main_process():
        print(
            "[checkpoint] partial model load by shape: "
            f"loaded={len(loaded_keys)} / model_keys={len(model_state)} / ckpt_keys={len(checkpoint_state_dict)}",
            flush=True,
        )

        if shape_mismatch:
            print(f"[checkpoint] shape mismatches (showing up to {max_report_items}):", flush=True)
            for item in shape_mismatch[:max_report_items]:
                print(
                    "  - {key}: ckpt={ckpt_shape}, model={model_shape}, "
                    "ckpt_dtype={ckpt_dtype}, model_dtype={model_dtype}".format(**item),
                    flush=True,
                )

        if unexpected_keys:
            print(f"[checkpoint] unexpected ckpt keys (showing up to {max_report_items}):", flush=True)
            for k in unexpected_keys[:max_report_items]:
                print(f"  - {k}", flush=True)

        if missing_keys:
            print(f"[checkpoint] missing model keys (showing up to {max_report_items}):", flush=True)
            for k in missing_keys[:max_report_items]:
                print(f"  - {k}", flush=True)

    return report


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
    model_strict: bool = False,
    load_model_only: bool = False,
    model_match_by_shape: bool = False,
    model_strip_prefixes: tuple[str, ...] = ("module.",),
    model_ignore_prefixes: tuple[str, ...] = (),
    verbose_model_load: bool = True,
) -> dict[str, Any]:
    """加载 checkpoint 并恢复对象状态。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model_load_report = None

    if bool(model_match_by_shape):
        # 部分加载通常只适合“加载预训练模型参数”，不适合恢复优化器/调度器/scaler。
        if not load_model_only:
            if is_main_process():
                print(
                    "[checkpoint] model_match_by_shape=True, force load_model_only=True "
                    "to avoid restoring optimizer/scheduler/scaler with changed model structure.",
                    flush=True,
                )
            load_model_only = True

        model_load_report = load_state_dict_match_by_shape(
            model,
            ckpt["model"],
            strip_prefixes=tuple(model_strip_prefixes),
            ignore_prefixes=tuple(model_ignore_prefixes),
            verbose=bool(verbose_model_load),
        )
    else:
        unwrap_model(model).load_state_dict(ckpt["model"], strict=bool(model_strict))

    if (not load_model_only) and optimizer is not None and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        _move_optimizer_state_to_device(optimizer, _infer_model_device(model))
    if (not load_model_only) and scheduler is not None and ckpt.get("scheduler", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if (not load_model_only) and scaler is not None and ckpt.get("scaler", None) is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "step_in_epoch": int(ckpt.get("step_in_epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_metric": float(ckpt.get("best_metric", float("inf"))),
        "cfg": ckpt.get("cfg", {}),
        "extra_state": ckpt.get("extra_state", {}),
        "rng_state": ckpt.get("rng_state", None),
        "model_load_report": model_load_report,
        "load_model_only_effective": bool(load_model_only),
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
    model_strict: bool = False,
    load_model_only: bool = False,
    restore_rng: bool = True,
    model_match_by_shape: bool = False,
    model_strip_prefixes: tuple[str, ...] = ("module.",),
    model_ignore_prefixes: tuple[str, ...] = (),
    verbose_model_load: bool = True,
) -> dict[str, Any]:
    """恢复 checkpoint 并恢复随机状态。"""
    if model_match_by_shape and not load_model_only:
        load_model_only = True
        restore_rng = False

    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=map_location,
        model_strict=model_strict,
        load_model_only=load_model_only,
        model_match_by_shape=model_match_by_shape,
        model_strip_prefixes=model_strip_prefixes,
        model_ignore_prefixes=model_ignore_prefixes,
        verbose_model_load=verbose_model_load,
    )
    if restore_rng and (not load_model_only):
        restore_rng_state(state.get("rng_state", None))
    return state
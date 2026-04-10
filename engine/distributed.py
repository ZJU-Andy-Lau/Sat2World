"""engine.distributed

分布式训练与设备管理基础设施。

本模块只负责 DDP 相关的通用能力，不包含 trainer 业务逻辑。
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def init_distributed(backend: str = "nccl") -> dict[str, Any]:
    """初始化分布式环境。

    说明：
    - 若环境变量未提供 RANK/WORLD_SIZE，则按单进程模式运行。
    - 多卡时设置当前 CUDA device = LOCAL_RANK。
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def destroy_distributed() -> None:
    """安全销毁进程组。"""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        finally:
            dist.destroy_process_group()


def is_distributed() -> bool:
    """当前是否处于已初始化分布式模式。"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """返回全局 rank。"""
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """返回本地 rank（从环境变量读取）。"""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """返回 world size。"""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """是否主进程。"""
    return get_rank() == 0


def barrier() -> None:
    """分布式同步屏障。"""
    if is_distributed():
        dist.barrier()


def seed_everything(base_seed: int) -> int:
    """设置全局随机种子，并按 rank 偏移。"""
    seed = int(base_seed) + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def configure_cuda_runtime(cfg: dict[str, Any] | None = None) -> None:
    """设置 CUDA 运行时性能开关。"""
    cfg = cfg or {}
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def _reduce_tensor_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return x
    y = x
    if dist.get_backend() == "nccl" and (not y.is_cuda):
        if torch.cuda.is_available():
            y = y.to(device=torch.device("cuda", get_local_rank()))
    y = y.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y = y / float(get_world_size())
    return y


def all_reduce_mean(value: Any) -> Any:
    """跨卡平均标量或标量字典。"""
    if not is_distributed():
        return value

    if torch.is_tensor(value):
        return _reduce_tensor_mean(value)
    if isinstance(value, (int, float)):
        if torch.cuda.is_available():
            dev = torch.device("cuda", get_local_rank())
        else:
            dev = torch.device("cpu")
        t = torch.tensor(float(value), device=dev)
        return float(_reduce_tensor_mean(t).item())
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[k] = all_reduce_mean(v)
        return out
    return value


_RPC_KEYS = {"rpc_gt", "rpc_init", "rpc_corrected"}


def _is_rpc_like(obj: Any) -> bool:
    return hasattr(obj, "to_gpu") and hasattr(obj, "device")


def _move_rpc_obj_to_device(rpc_obj: Any, device: torch.device) -> Any:
    if not _is_rpc_like(rpc_obj):
        return rpc_obj
    cur_dev = getattr(rpc_obj, "device", None)
    if isinstance(cur_dev, torch.device) and cur_dev == device:
        return rpc_obj
    if device.type == "cuda":
        rpc_obj.to_gpu(device)
    else:
        rpc_obj.to_gpu("cpu")
    return rpc_obj


def _move_rpc_nested_to_device(obj: Any, device: torch.device) -> Any:
    if _is_rpc_like(obj):
        return _move_rpc_obj_to_device(obj, device)
    if isinstance(obj, list):
        return [_move_rpc_nested_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_rpc_nested_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _move_rpc_nested_to_device(v, device) for k, v in obj.items()}
    return obj


def move_batch_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """递归移动 batch 到 device，并确保 RPC 对象与目标 device 对齐。"""
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if k in _RPC_KEYS:
                out[k] = _move_rpc_nested_to_device(v, device)
            else:
                out[k] = move_batch_to_device(v, device, non_blocking=non_blocking)
        return out
    if isinstance(batch, list):
        return [move_batch_to_device(v, device, non_blocking=non_blocking) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device, non_blocking=non_blocking) for v in batch)
    return batch


def assert_tensor_tree_device(obj: Any, expected_device: torch.device, prefix: str = "") -> None:
    """递归检查嵌套结构中所有 tensor 的 device 一致性。"""
    if torch.is_tensor(obj):
        if obj.device != expected_device:
            raise RuntimeError(f"Device mismatch at {prefix or '<root>'}: got {obj.device}, expect {expected_device}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_tensor_tree_device(v, expected_device, prefix=f"{prefix}.{k}" if prefix else str(k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_tensor_tree_device(v, expected_device, prefix=f"{prefix}[{i}]")
        return


def assert_rpc_tree_device(obj: Any, expected_device: torch.device, prefix: str = "") -> None:
    """递归检查嵌套结构中 RPC 对象的 device 一致性。"""
    if _is_rpc_like(obj):
        rpc_dev = getattr(obj, "device", None)
        if not isinstance(rpc_dev, torch.device):
            raise RuntimeError(f"RPC device invalid at {prefix or '<root>'}: got {rpc_dev}")
        if rpc_dev != expected_device:
            raise RuntimeError(f"RPC device mismatch at {prefix or '<root>'}: got {rpc_dev}, expect {expected_device}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_rpc_tree_device(v, expected_device, prefix=f"{prefix}.{k}" if prefix else str(k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_rpc_tree_device(v, expected_device, prefix=f"{prefix}[{i}]")
        return


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """若模型为 DDP，返回其 module。"""
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """按当前分布式状态包装 DDP。"""
    if not is_distributed():
        return model
    if device.type == "cuda":
        return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    return DistributedDataParallel(model, find_unused_parameters=False)


@contextlib.contextmanager
def maybe_no_sync(model: torch.nn.Module, enabled: bool):
    """梯度累积时的 no_sync 上下文。"""
    if enabled and isinstance(model, DistributedDataParallel):
        with model.no_sync():
            yield
    else:
        yield


def get_cuda_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """返回当前 CUDA 显存统计（MB）。"""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    device = device or torch.device("cuda", get_local_rank())
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
    }

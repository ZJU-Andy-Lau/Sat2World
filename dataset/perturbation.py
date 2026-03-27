"""dataset.perturbation

本文件负责在线仿射扰动与 synthetic rpc_init 构造。

关键约定（必须严格一致）：
1. 磁盘 RPC 为 GT（正确 RPC）。
2. affine_gt_forward 表示 true pixel -> observed pixel。
3. affine_gt_correction 是其逆，表示 observed pixel -> true pixel。
4. 参考视图永远不扰动，forward 与 correction 都为单位阵。
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from geometry.rpc import RPCModelParameterTorch


@dataclass
class PerturbationConfig:
    """在线仿射扰动范围配置。

    字段说明:
    - tx_range / ty_range: 平移范围（像素）。
    - scale_range: 对角缩放残差范围（围绕1，且建议 <= 1e-3）。
    - shear_range: shear 范围（建议 <= 1e-3）。
    """

    tx_range: tuple[float, float] = (-2.0, 2.0)
    ty_range: tuple[float, float] = (-2.0, 2.0)
    scale_range: tuple[float, float] = (-1e-4, 1e-4)
    shear_range: tuple[float, float] = (-1e-4, 1e-4)


def affine_2x3_to_3x3(affine: torch.Tensor) -> torch.Tensor:
    """将 2x3 仿射张量转换为 3x3 齐次矩阵，支持批量。"""
    if affine.shape[-2:] != (2, 3):
        raise ValueError(f"affine must end with (2,3), got {tuple(affine.shape)}")

    tail = affine.shape[:-2]
    bottom = torch.zeros((*tail, 1, 3), dtype=affine.dtype, device=affine.device)
    bottom[..., 0, 2] = 1.0
    return torch.cat([affine, bottom], dim=-2)


def affine_3x3_to_2x3(homo: torch.Tensor) -> torch.Tensor:
    """将 3x3 齐次矩阵转换为 2x3 仿射，支持批量。"""
    if homo.shape[-2:] != (3, 3):
        raise ValueError(f"homo must end with (3,3), got {tuple(homo.shape)}")
    return homo[..., :2, :]


def identity_affine_2x3(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """生成单位 2x3 仿射矩阵。"""
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)


def invert_affine_2x3(affine: torch.Tensor) -> torch.Tensor:
    """求 2x3 仿射逆矩阵，支持批量输入。"""
    homo = affine_2x3_to_3x3(affine)
    inv = torch.linalg.inv(homo)
    return affine_3x3_to_2x3(inv)


def sample_forward_affine(
    rng: np.random.Generator,
    cfg: PerturbationConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """采样 forward 扰动（true pixel -> observed pixel）。

    说明:
    - 线性部分强制接近单位阵（scale/shear 在小范围内）。
    - 不引入旋转参数，直接使用一般 2x3 仿射表达。
    """
    sx = rng.uniform(cfg.scale_range[0], cfg.scale_range[1])
    sy = rng.uniform(cfg.scale_range[0], cfg.scale_range[1])
    shx = rng.uniform(cfg.shear_range[0], cfg.shear_range[1])
    shy = rng.uniform(cfg.shear_range[0], cfg.shear_range[1])
    tx = rng.uniform(cfg.tx_range[0], cfg.tx_range[1])
    ty = rng.uniform(cfg.ty_range[0], cfg.ty_range[1])

    affine = torch.tensor(
        [[1.0 + sx, shx, tx], [shy, 1.0 + sy, ty]],
        device=device,
        dtype=dtype,
    )
    return affine


def make_deterministic_rng(base_seed: int, epoch: int, scene_id: int, sample_index: int) -> np.random.Generator:
    """构造确定性随机数生成器，避免依赖全局随机状态。"""
    key = f"{base_seed}_{epoch}_{scene_id}_{sample_index}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    seed = int(digest, 16) % (2**32)
    return np.random.default_rng(seed)


def _clone_rpc(rpc_obj: "RPCModelParameterTorch") -> "RPCModelParameterTorch":
    """深拷贝 RPC 对象，避免原地污染 GT RPC。"""
    return copy.deepcopy(rpc_obj)


def build_synthetic_rpc_inputs(
    rpc_gt_views: Sequence["RPCModelParameterTorch"],
    ref_view_idx: int,
    rng: np.random.Generator,
    perturb_cfg: PerturbationConfig,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[list["RPCModelParameterTorch"], torch.Tensor, torch.Tensor]:
    """根据 GT RPC 构造 synthetic 输入 RPC（rpc_init）与仿射标签。

    参数:
        rpc_gt_views: 当前 sample 的 GT RPC 列表。
        ref_view_idx: 参考视图索引。
        rng: 确定性随机生成器。
        perturb_cfg: 扰动范围配置。

    返回:
        rpc_init_views: 扰动后的输入 RPC 列表。
        affine_gt_forward: [V,2,3]，true -> observed。
        affine_gt_correction: [V,2,3]，observed -> true（forward 逆）。
    """
    v = len(rpc_gt_views)
    if v == 0:
        raise ValueError("rpc_gt_views cannot be empty")
    if not (0 <= ref_view_idx < v):
        raise ValueError(f"Invalid ref_view_idx={ref_view_idx} for V={v}")

    aff_forward = []
    rpc_init = []

    for vi, rpc_gt in enumerate(rpc_gt_views):
        if vi == ref_view_idx:
            a_fwd = identity_affine_2x3(device=device, dtype=dtype)
        else:
            a_fwd = sample_forward_affine(rng, perturb_cfg, dtype=dtype, device=device)

        # 通过 forward 扰动在 clone 上注入“观测几何误差”，构造 rpc_init。
        rpc_init_i = _clone_rpc(rpc_gt)
        rpc_init_i.Update_Adjust(a_fwd.to(dtype=torch.double, device=rpc_init_i.device))

        aff_forward.append(a_fwd)
        rpc_init.append(rpc_init_i)

    affine_gt_forward = torch.stack(aff_forward, dim=0)
    affine_gt_correction = invert_affine_2x3(affine_gt_forward)
    return rpc_init, affine_gt_forward, affine_gt_correction

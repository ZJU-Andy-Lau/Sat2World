"""scene_geometry.py

本文件提供与 RPC 数学求解本身解耦的“场景级几何辅助”函数。
这些函数不修改 RPC 参数，也不执行高阶相机模型计算，职责是：
1) 生成像素网格与 patch 中心；
2) 推断/扩展 height_ref；
3) 处理参考视图仿射恒等约束；
4) 提供少量形状整理工具。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch


class _NoValue:
    """内部哨兵类型：用于区分用户显式传入 None 与未传值。"""


NO_VALUE = _NoValue()


def make_image_grid(
    height: int,
    width: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构建完整分辨率的像素坐标网格。

    参数:
        height: 图像高度 H。
        width: 图像宽度 W。
        device: 输出张量所在设备。
        dtype: 输出数据类型。

    返回:
        grid: 形状 [H, W, 2]，最后一维顺序固定为 (line, samp)。
              line 对应行坐标 [0..H-1]，samp 对应列坐标 [0..W-1]。
    """
    lines = torch.arange(height, device=device, dtype=dtype)
    samps = torch.arange(width, device=device, dtype=dtype)
    line_grid, samp_grid = torch.meshgrid(lines, samps, indexing="ij")
    return torch.stack([line_grid, samp_grid], dim=-1)


def make_patch_centers(
    orig_hw: Tuple[int, int],
    padded_hw: Tuple[int, int],
    grid_hw: Tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """根据原图尺寸、padding 后尺寸与 patch 网格尺寸，生成 patch 中心与有效掩码。

    设计要点:
    - patch 中心先在 padded 空间按规则网格生成；
    - 再 clamp 回原图有效像素范围，保证几何坐标具有原图意义；
    - 同时输出 patch_valid_mask 指示该 patch 是否来自原图有效区域。

    参数:
        orig_hw: 原始输入图高宽 (H, W)。
        padded_hw: backbone padding 后高宽 (Hp, Wp)。
        grid_hw: patch token 网格高宽 (Gh, Gw)。
        device: 输出设备。
        dtype: 输出数据类型。

    返回:
        patch_centers: [N, 2]，N=Gh*Gw，顺序 (line, samp)。
        patch_valid_mask: [N]，bool，True 表示真实图像 patch，False 表示 padding 区域 patch。
    """
    h, w = orig_hw
    hp, wp = padded_hw
    gh, gw = grid_hw

    if gh <= 0 or gw <= 0:
        raise ValueError(f"grid_hw must be positive, got {(gh, gw)}")

    patch_h = hp / gh
    patch_w = wp / gw

    row_ids = torch.arange(gh, device=device, dtype=dtype)
    col_ids = torch.arange(gw, device=device, dtype=dtype)
    row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing="ij")

    line_center = (row_grid + 0.5) * patch_h - 0.5
    samp_center = (col_grid + 0.5) * patch_w - 0.5

    patch_valid = (line_center >= 0) & (line_center <= (h - 1)) & (samp_center >= 0) & (samp_center <= (w - 1))

    line_center = line_center.clamp(0, max(h - 1, 0))
    samp_center = samp_center.clamp(0, max(w - 1, 0))

    centers = torch.stack([line_center, samp_center], dim=-1).reshape(-1, 2)
    return centers, patch_valid.reshape(-1)


def infer_height_ref(rpc_obj: object) -> float:
    """从单个 RPC 对象推断 height_ref。

    优先读取 `HEIGHT_OFF`；若对象缺失该字段或读取失败，则返回 0.0。

    参数:
        rpc_obj: 单个 RPCModelParameterTorch 或兼容对象。

    返回:
        h_ref: float 标量。
    """
    if rpc_obj is None:
        return 0.0

    if hasattr(rpc_obj, "HEIGHT_OFF"):
        h = getattr(rpc_obj, "HEIGHT_OFF")
        if torch.is_tensor(h):
            if h.numel() == 0:
                return 0.0
            return float(h.detach().reshape(-1)[0].item())
        try:
            return float(h)
        except Exception:
            return 0.0
    return 0.0


def infer_height_ref_batch(
    rpc_batch: Sequence[Sequence[object]],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """从 RPC batch 推断每个视图的 h_ref。

    参数:
        rpc_batch: 长度为 B 的列表，每个元素是长度为 V 的 RPC 对象列表。
        device: 输出张量设备。
        dtype: 输出张量类型。

    返回:
        height_ref: [B, V]。
    """
    b = len(rpc_batch)
    if b == 0:
        return torch.empty(0, 0, device=device, dtype=dtype)
    v = len(rpc_batch[0])

    out = torch.zeros((b, v), device=device, dtype=dtype)
    for bi in range(b):
        if len(rpc_batch[bi]) != v:
            raise ValueError("All batch entries must have same number of views")
        for vi in range(v):
            out[bi, vi] = infer_height_ref(rpc_batch[bi][vi])
    return out


def expand_height_ref_map(
    height_ref: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """把 [B, V] 的 h_ref 扩展为 [B, V, 1, H, W]。

    参数:
        height_ref: [B, V]。
        height: 目标图像高度。
        width: 目标图像宽度。

    返回:
        h_ref_map: [B, V, 1, H, W]。
    """
    if height_ref.ndim != 2:
        raise ValueError(f"height_ref must be [B,V], got {tuple(height_ref.shape)}")
    return height_ref[:, :, None, None, None].expand(-1, -1, 1, height, width)


def enforce_reference_affine_identity(
    affine: torch.Tensor,
    ref_view_idx: int | torch.Tensor | Sequence[int] = 0,
) -> torch.Tensor:
    """强制参考视图 affine 为单位阵。

    参数:
        affine: [B, V, 2, 3] 的仿射校正张量。
        ref_view_idx: 参考视图索引，可为:
            - 单个 int（全 batch 共用）；
            - [B] tensor 或序列（每个 batch 一个索引）。

    返回:
        affine_fixed: 与输入同形状的新张量，参考视图位置被写为单位阵。
    """
    if affine.ndim != 4 or affine.shape[-2:] != (2, 3):
        raise ValueError(f"affine must be [B,V,2,3], got {tuple(affine.shape)}")

    b, v = affine.shape[:2]
    out = affine.clone()

    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=affine.device, dtype=affine.dtype)

    if isinstance(ref_view_idx, int):
        idx = max(0, min(v - 1, ref_view_idx))
        out[:, idx, :, :] = identity
        return out

    if isinstance(ref_view_idx, torch.Tensor):
        idx_tensor = ref_view_idx.to(device=affine.device, dtype=torch.long).view(-1)
    else:
        idx_tensor = torch.as_tensor(list(ref_view_idx), device=affine.device, dtype=torch.long).view(-1)

    if idx_tensor.numel() != b:
        raise ValueError(f"ref_view_idx must provide B={b} indices, got {idx_tensor.numel()}")

    idx_tensor = idx_tensor.clamp(0, v - 1)
    for bi in range(b):
        out[bi, idx_tensor[bi], :, :] = identity
    return out


def reshape_bv_to_bvchw(x: torch.Tensor, b: int, v: int) -> torch.Tensor:
    """将 [B*V, C, H, W] 还原为 [B, V, C, H, W]。"""
    if x.ndim != 4:
        raise ValueError(f"Expected [B*V,C,H,W], got {tuple(x.shape)}")
    bv = x.shape[0]
    if bv != b * v:
        raise ValueError(f"Leading dim mismatch: got {bv}, expect {b*v}")
    return x.view(b, v, *x.shape[1:])


def reshape_bv_to_bvn(x: torch.Tensor, b: int, v: int) -> torch.Tensor:
    """将 [B*V, N] 还原为 [B, V, N]。"""
    if x.ndim != 2:
        raise ValueError(f"Expected [B*V,N], got {tuple(x.shape)}")
    bv = x.shape[0]
    if bv != b * v:
        raise ValueError(f"Leading dim mismatch: got {bv}, expect {b*v}")
    return x.view(b, v, x.shape[1])

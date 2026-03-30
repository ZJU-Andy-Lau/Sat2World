"""loss.common

本文件集中实现 loss 子模块共用的数值工具函数。

重要约定（请在所有 loss 中保持一致）：
- affine_gt_forward 表示 true pixel -> observed pixel；
- affine_pred 表示 observed pixel -> true pixel（correction affine）。

这些函数均基于 PyTorch 实现，支持自动求导与 GPU 计算。
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Literal

import torch
import torch.nn.functional as F


def masked_reduce(value: torch.Tensor, mask: torch.Tensor | None = None, reduce: Literal["mean", "sum"] = "mean") -> torch.Tensor:
    """对张量执行带 mask 的聚合。

    输入:
        value:
            待聚合张量，形状任意。
        mask:
            可广播到 value 的掩码；None 表示全有效。
        reduce:
            聚合方式，仅支持 "mean" 与 "sum"。

    输出:
        标量张量。

    功能:
        - 支持广播掩码；
        - 当有效元素数量为 0 时返回 0，避免 NaN；
        - 统一被所有 masked loss 调用，保证行为一致。
    """
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce}")

    if mask is None:
        return value.mean() if reduce == "mean" else value.sum()

    m = mask.to(dtype=value.dtype)
    weighted = value * m
    if reduce == "sum":
        return weighted.sum()

    denom = m.sum()
    if denom.detach().item() <= 0:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return weighted.sum() / denom


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, beta: float = 1.0) -> torch.Tensor:
    """计算带 mask 的 Huber 损失。

    公式:
        e = pred - target
        if |e| <= beta: 0.5 * e^2 / beta
        else:          |e| - 0.5 * beta

    输入:
        pred/target:
            同形状张量。
        mask:
            可广播掩码；None 时退化为普通均值。
        beta:
            Huber 分段阈值，必须 > 0。

    输出:
        标量 loss。
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    e = pred - target
    abs_e = e.abs()
    quad = 0.5 * e.square() / beta
    lin = abs_e - 0.5 * beta
    huber = torch.where(abs_e <= beta, quad, lin)
    return masked_reduce(huber, mask=mask, reduce="mean")


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """计算带 mask 的 L1 损失。"""
    return masked_reduce((pred - target).abs(), mask=mask, reduce="mean")


def masked_l2_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """计算带 mask 的 L2 损失（MSE 形式）。"""
    return masked_reduce((pred - target).square(), mask=mask, reduce="mean")


def ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """计算轻量 SSIM map。

    输入:
        x/y:
            [B,C,H,W]，通常范围 [0,1]。
        kernel_size:
            均值池化窗口大小（3 或 5 常用）。

    输出:
        [B,1,H,W] 的 SSIM map。
    """
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("x and y must be [B,C,H,W]")
    if x.shape != y.shape:
        raise ValueError("x and y must share shape")

    pad = kernel_size // 2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=pad)

    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=pad) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim = num / den.clamp_min(eps)
    return ssim.mean(dim=1, keepdim=True)


def ssim_loss_from_map(ssim: torch.Tensor) -> torch.Tensor:
    """由 SSIM map 计算 SSIM 损失（1-SSIM）的平均值。"""
    return (1.0 - ssim).mean()


def psnr_from_mse(mse: torch.Tensor | float, eps: float = 1e-8) -> torch.Tensor:
    """根据 MSE 计算 PSNR，假设图像范围 [0,1]。"""
    mse_t = mse if torch.is_tensor(mse) else torch.tensor(float(mse))
    return -10.0 * torch.log10(mse_t.clamp_min(eps))


def softmax_entropy(logits: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """计算 softmax 熵 H=-sum(p log p)。"""
    p = torch.softmax(logits, dim=dim)
    return -(p * torch.log(p.clamp_min(eps))).sum(dim=dim)


def apply_affine_to_points(points: torch.Tensor, affine_2x3: torch.Tensor) -> torch.Tensor:
    """对二维点应用仿射变换。

    输入:
        points:
            最后一维为 2，顺序 [line, samp]。
        affine_2x3:
            最后两维为 [2,3]。

    输出:
        与 points 广播后的同批次二维点，顺序仍为 [line, samp]。
    """
    if points.shape[-1] != 2:
        raise ValueError("points last dim must be 2")
    if affine_2x3.shape[-2:] != (2, 3):
        raise ValueError("affine_2x3 last dims must be [2,3]")

    # 兼容常见误传形状：例如 [B,1,2,3] + [B,N,2]，自动压掉多余 singleton 维。
    while affine_2x3.ndim > points.ndim and affine_2x3.shape[-3] == 1:
        affine_2x3 = affine_2x3.squeeze(-3)

    ones = torch.ones_like(points[..., :1])
    homo = torch.cat([points, ones], dim=-1)
    out = torch.matmul(homo, affine_2x3.transpose(-1, -2))
    if out.shape[-1] != 2 or out.ndim != points.ndim:
        raise ValueError(
            "apply_affine_to_points produced unexpected shape. "
            f"points={tuple(points.shape)}, affine={tuple(affine_2x3.shape)}, out={tuple(out.shape)}"
        )
    return out


def make_uniform_grid_points(
    h: int,
    w: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """构造覆盖整图的均匀网格点，输出 [N,2] (line,samp)。"""
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid_h/grid_w must be > 0")
    ys = torch.linspace(0.0, float(max(h - 1, 0)), steps=grid_h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, float(max(w - 1, 0)), steps=grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)


def normalize_sampling_grid(line_samp: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """把 [line,samp] 坐标转换为 grid_sample 所需的归一化 [x,y] 坐标。

    输入支持:
        [N,2] 或 [B,N,2]

    输出:
        [B,1,N,2] 形式，适配 F.grid_sample。
    """
    if line_samp.ndim == 2:
        line_samp = line_samp.unsqueeze(0)
    if line_samp.ndim != 3 or line_samp.shape[-1] != 2:
        raise ValueError("line_samp must be [N,2] or [B,N,2]")

    line = line_samp[..., 0]
    samp = line_samp[..., 1]

    x = 2.0 * samp / max(w - 1, 1) - 1.0
    y = 2.0 * line / max(h - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=-1)
    return grid.unsqueeze(1)


def sample_map_bilinear(
    map_tensor: torch.Tensor,
    line_samp: torch.Tensor,
    align_corners: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 map 执行双线性采样。

    输入:
        map_tensor:
            [B,C,H,W]。
        line_samp:
            [B,N,2]，顺序 [line,samp]。

    输出:
        sampled:
            [B,C,N]。
        in_bounds:
            [B,N]，表示采样点是否在图内。
    """
    if map_tensor.ndim != 4:
        raise ValueError("map_tensor must be [B,C,H,W]")
    if line_samp.ndim != 3 or line_samp.shape[-1] != 2:
        raise ValueError("line_samp must be [B,N,2]")

    b, _, h, w = map_tensor.shape
    if line_samp.shape[0] != b:
        raise ValueError("batch size mismatch between map_tensor and line_samp")

    line = line_samp[..., 0]
    samp = line_samp[..., 1]
    in_bounds = (line >= 0) & (line <= (h - 1)) & (samp >= 0) & (samp <= (w - 1))

    grid = normalize_sampling_grid(line_samp, h=h, w=w)
    sampled = F.grid_sample(
        map_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    return sampled.squeeze(2), in_bounds


def pairwise_view_pairs(v: int, max_pairs: int | None = None) -> list[tuple[int, int]]:
    """生成无序视图对列表。"""
    all_pairs = list(combinations(range(v), 2))
    if max_pairs is None or max_pairs >= len(all_pairs):
        return all_pairs
    if max_pairs <= 0:
        return []
    idx = torch.linspace(0, len(all_pairs) - 1, steps=max_pairs).round().long().tolist()
    return [all_pairs[i] for i in idx]


def safe_rmse(squared_error: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """计算安全 RMSE: sqrt(masked_mean(se)+eps)。"""
    mse = masked_reduce(squared_error, mask=mask, reduce="mean")
    return torch.sqrt(mse + eps)

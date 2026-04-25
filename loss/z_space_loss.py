"""loss.z_space_loss

有界 sinh 偏移解码器对应的 z-space 监督工具。
"""

from __future__ import annotations

import torch

from loss.common import masked_huber_loss


def meter_beta_to_z_beta(beta_meter: float, scale: torch.Tensor | float) -> torch.Tensor:
    """把米制 Huber beta 映射到 z-space beta。"""
    if torch.is_tensor(scale):
        s = scale.float().clamp_min(1e-12)
        b = torch.as_tensor(float(beta_meter), device=s.device, dtype=s.dtype)
        return torch.asinh(b / s).clamp_min(1e-6)
    s = max(float(scale), 1e-12)
    b = torch.tensor(float(beta_meter), dtype=torch.float32)
    return torch.asinh(b / s).clamp_min(1e-6)


def offset_meter_to_target_z(
    offset_meter: torch.Tensor,
    *,
    scale: torch.Tensor | float,
    z_max: torch.Tensor | float,
) -> torch.Tensor:
    """把米制 offset 反变换到 z-space，并截断到 [-z_max, z_max]。"""
    offset_fp32 = offset_meter.float()
    if torch.is_tensor(scale):
        s = scale.to(device=offset_fp32.device, dtype=offset_fp32.dtype)
    else:
        s = torch.tensor(float(scale), device=offset_fp32.device, dtype=offset_fp32.dtype)
    if torch.is_tensor(z_max):
        zm = z_max.to(device=offset_fp32.device, dtype=offset_fp32.dtype)
    else:
        zm = torch.tensor(float(z_max), device=offset_fp32.device, dtype=offset_fp32.dtype)
    target_z = torch.asinh(offset_fp32 / s.clamp_min(1e-12))
    return target_z.clamp(min=-zm*0.98, max=zm*0.98)


def masked_z_huber_loss(
    pred_z: torch.Tensor,
    target_offset_meter: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    beta_meter: float,
    scale: torch.Tensor | float,
    z_max: torch.Tensor | float,
) -> torch.Tensor:
    """在 z-space 计算 masked SmoothL1/Huber。内部强制 float32。"""
    pred_fp32 = pred_z.float()
    target_z = offset_meter_to_target_z(target_offset_meter, scale=scale, z_max=z_max)
    beta_z = meter_beta_to_z_beta(beta_meter=beta_meter, scale=scale)
    beta_z_scalar = float(beta_z.detach().mean().item())
    mask_fp32 = None if mask is None else mask.to(device=pred_fp32.device, dtype=pred_fp32.dtype)
    return masked_huber_loss(pred_fp32, target_z, mask=mask_fp32, beta=beta_z_scalar)

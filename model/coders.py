"""coders.py

本文件实现“物理锚点 + 有界残差”解码策略：
- SymmetricBinScalarCoder: 标量残差解码器；
- HeightCoder: 绝对高程解码；
- PointCoder: 绝对点云解码。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SymmetricBinCoderCfg:
    """对称分桶残差解码器配置。"""

    num_bins: int = 33
    bin_size: float = 1.0
    fine_range: float = 0.5


class SymmetricBinScalarCoder(nn.Module):
    """对称离散锚点 + fine 残差 的标量解码器。

    解码公式:
        coarse = softmax(logits) · anchors
        fine = tanh(fine_raw) * fine_range
        residual = coarse + fine
    """

    def __init__(self, cfg: SymmetricBinCoderCfg) -> None:
        """初始化并注册离散锚点。"""
        super().__init__()
        if cfg.num_bins < 3 or cfg.num_bins % 2 == 0:
            raise ValueError("num_bins must be odd and >= 3")
        self.cfg = cfg
        half = cfg.num_bins // 2
        anchors = torch.arange(-half, half + 1, dtype=torch.float32) * cfg.bin_size
        self.register_buffer("anchors", anchors, persistent=True)

    def decode(
        self,
        coarse_logits: torch.Tensor,
        fine_raw: torch.Tensor,
        *,
        channel_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行残差解码。

        参数:
            coarse_logits: 包含离散 logits 的张量，channel_dim 对应 bins 维。
            fine_raw: 对应 fine 分量，shape 为 coarse 去掉 bins 维后的形状。
            channel_dim: coarse_logits 中 bins 所在维。

        返回:
            residual: 总残差。
            coarse_value: coarse 部分残差。
            fine_value: fine 部分残差。
        """
        if channel_dim < 0:
            channel_dim = coarse_logits.ndim + channel_dim

        probs = torch.softmax(coarse_logits, dim=channel_dim)
        view_shape = [1] * coarse_logits.ndim
        view_shape[channel_dim] = self.anchors.numel()
        anchors = self.anchors.view(*view_shape).to(device=coarse_logits.device, dtype=coarse_logits.dtype)

        coarse_value = (probs * anchors).sum(dim=channel_dim)
        fine_value = torch.tanh(fine_raw) * self.cfg.fine_range
        if fine_value.ndim == coarse_value.ndim + 1:
            fine_value = fine_value.squeeze(channel_dim)

        residual = coarse_value + fine_value
        return residual, coarse_value, fine_value


class HeightCoder(nn.Module):
    """高程解码器：直接把网络输出解码为绝对高程。"""

    def __init__(self, cfg: SymmetricBinCoderCfg) -> None:
        """初始化 HeightCoder。"""
        super().__init__()
        self.scalar = SymmetricBinScalarCoder(cfg)

    def forward(
        self,
        coarse_logits: torch.Tensor,
        fine_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """解码绝对高程。

        参数:
            coarse_logits: [B,V,K,H,W]。
            fine_raw: [B,V,1,H,W]。

        返回:
            dict:
                h_abs: [B,V,1,H,W]
                h_coarse: [B,V,1,H,W]
                h_fine: [B,V,1,H,W]
        """
        residual, coarse, fine = self.scalar.decode(coarse_logits, fine_raw, channel_dim=2)
        h_abs = residual.unsqueeze(2)
        return {
            "h_abs": h_abs,
            "h_coarse": coarse.unsqueeze(2),
            "h_fine": fine.unsqueeze(2),
        }


class PointCoder(nn.Module):
    """三轴点云解码器：点云锚点 + 三轴有界残差。"""

    def __init__(self, cfg_x: SymmetricBinCoderCfg, cfg_y: SymmetricBinCoderCfg, cfg_z: SymmetricBinCoderCfg) -> None:
        """初始化 PointCoder。"""
        super().__init__()
        self.coder_x = SymmetricBinScalarCoder(cfg_x)
        self.coder_y = SymmetricBinScalarCoder(cfg_y)
        self.coder_z = SymmetricBinScalarCoder(cfg_z)

    def forward(
        self,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
        z_logits: torch.Tensor,
        x_fine: torch.Tensor,
        y_fine: torch.Tensor,
        z_fine: torch.Tensor,
        point_anchor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """解码绝对点云。

        参数:
            x/y/z_logits: [B,V,K,H,W]。
            x/y/z_fine: [B,V,1,H,W]。
            point_anchor: [B,V,3,H,W]。

        返回:
            dict:
                point_abs: [B,V,3,H,W]
                delta_xyz_coarse: [B,V,3,H,W]
                delta_xyz_fine: [B,V,3,H,W]
        """
        dx, dx_c, dx_f = self.coder_x.decode(x_logits, x_fine, channel_dim=2)
        dy, dy_c, dy_f = self.coder_y.decode(y_logits, y_fine, channel_dim=2)
        dz, dz_c, dz_f = self.coder_z.decode(z_logits, z_fine, channel_dim=2)

        delta = torch.stack([dx, dy, dz], dim=2)
        delta_coarse = torch.stack([dx_c, dy_c, dz_c], dim=2)
        delta_fine = torch.stack([dx_f, dy_f, dz_f], dim=2)

        point_abs = point_anchor + delta
        return {
            "point_abs": point_abs,
            "delta_xyz_coarse": delta_coarse,
            "delta_xyz_fine": delta_fine,
        }

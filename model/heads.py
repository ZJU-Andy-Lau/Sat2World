"""heads.py

本文件实现 Sat2World 的任务头与共享 dense 解码器：
- SharedDenseDecoder
- AffineHead
- HeightHead
- PointHead
- GaussianHead
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """通用卷积块：Conv-BN-GELU。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        """初始化卷积块。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算。"""
        return self.net(x)


class SharedDenseDecoder(nn.Module):
    """共享 dense 解码器。

    输入:
        patch_map [B*V,C,Gh,Gw] 与原图 [B*V,3,H,W]
    输出:
        dense feature [B*V,256,H,W]
    """

    def __init__(self, in_ch: int = 1024, out_ch: int = 256) -> None:
        """初始化共享解码器。"""
        super().__init__()
        self.token_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        self.up_block1 = ConvBlock(out_ch, out_ch)
        self.up_block2 = ConvBlock(out_ch, out_ch)
        self.up_block3 = ConvBlock(out_ch, out_ch)

        self.rgb_stem = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
        )

        self.fuse = nn.Sequential(
            ConvBlock(out_ch + 64, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, patch_map: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """把 patch map 解码到全分辨率 dense 特征。

        参数:
            patch_map: [B*V,C,Gh,Gw]
            image: [B*V,3,H,W]

        返回:
            dense: [B*V,256,H,W]
        """
        _, _, h, w = image.shape

        x = self.token_proj(patch_map)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_block1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_block2(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.up_block3(x)

        rgb_feat = self.rgb_stem(image)
        x = torch.cat([x, rgb_feat], dim=1)
        x = self.fuse(x)
        return x


@dataclass
class AffineHeadCfg:
    """仿射头范围配置。"""

    diag_scale: float = 0.05
    offdiag_scale: float = 0.05
    trans_scale: float = 4.0


class AffineHead(nn.Module):
    """仿射校正头。

    输入 [B,V,C] 的 view token，输出 [B,V,2,3] 的有界仿射矩阵。
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, cfg: AffineHeadCfg | None = None) -> None:
        """初始化仿射头。"""
        super().__init__()
        self.cfg = cfg or AffineHeadCfg()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        """预测有界 affine correction。

        参数:
            view_tokens: [B,V,C]

        返回:
            affine: [B,V,2,3]
        """
        raw = self.mlp(view_tokens)
        t = torch.tanh(raw)

        a00 = 1.0 + self.cfg.diag_scale * t[..., 0]
        a01 = self.cfg.offdiag_scale * t[..., 1]
        tx = self.cfg.trans_scale * t[..., 2]
        a10 = self.cfg.offdiag_scale * t[..., 3]
        a11 = 1.0 + self.cfg.diag_scale * t[..., 4]
        ty = self.cfg.trans_scale * t[..., 5]

        row0 = torch.stack([a00, a01, tx], dim=-1)
        row1 = torch.stack([a10, a11, ty], dim=-1)
        return torch.stack([row0, row1], dim=-2)


class HeightHead(nn.Module):
    """高程头：预测 coarse logits 与 fine residual。"""

    def __init__(self, in_ch: int = 256, num_bins: int = 33) -> None:
        """初始化高程头。"""
        super().__init__()
        self.trunk = ConvBlock(in_ch, in_ch)
        self.logits = nn.Conv2d(in_ch, num_bins, kernel_size=1)
        self.fine = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """预测高程编码量。

        参数:
            feat: [B*V,256,H,W]

        返回:
            dict:
                logits: [B*V,K,H,W]
                fine: [B*V,1,H,W]
        """
        x = self.trunk(feat)
        return {"logits": self.logits(x), "fine": self.fine(x)}


class PointHead(nn.Module):
    """点云头：分别预测 x/y/z 三轴的 coarse logits 与 fine residual。"""

    def __init__(self, in_ch: int = 256, num_bins: int = 33) -> None:
        """初始化点云头。"""
        super().__init__()
        self.trunk = ConvBlock(in_ch, in_ch)

        self.x_logits = nn.Conv2d(in_ch, num_bins, kernel_size=1)
        self.y_logits = nn.Conv2d(in_ch, num_bins, kernel_size=1)
        self.z_logits = nn.Conv2d(in_ch, num_bins, kernel_size=1)

        self.x_fine = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.y_fine = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.z_fine = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """预测点云编码量。"""
        x = self.trunk(feat)
        return {
            "x_logits": self.x_logits(x),
            "y_logits": self.y_logits(x),
            "z_logits": self.z_logits(x),
            "x_fine": self.x_fine(x),
            "y_fine": self.y_fine(x),
            "z_fine": self.z_fine(x),
        }


class GaussianHead(nn.Module):
    """高斯属性头（不输出中心）。"""

    def __init__(self, in_ch: int = 256, sh_dim: int = 48, scale_max: float = 0.3, scale_eps: float = 1e-4) -> None:
        """初始化高斯属性头。"""
        super().__init__()
        self.scale_max = scale_max
        self.scale_eps = scale_eps

        self.trunk = ConvBlock(in_ch, in_ch)
        self.opacity = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.scale = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.rotation = nn.Conv2d(in_ch, 4, kernel_size=1)
        self.sh = nn.Conv2d(in_ch, sh_dim, kernel_size=1)
        self.conf_rpc = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.conf_point = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """预测高斯属性并完成必要激活。

        参数:
            feat: [B*V,256,H,W]

        返回:
            dict:
                opacity: [B*V,1,H,W]
                scale: [B*V,3,H,W]
                rotation: [B*V,4,H,W]
                sh: [B*V,sh_dim,H,W]
                confidence_rpc: [B*V,1,H,W]
                confidence_point: [B*V,1,H,W]
        """
        x = self.trunk(feat)

        opacity = torch.sigmoid(self.opacity(x))
        scale = F.softplus(self.scale(x)) + self.scale_eps
        scale = scale.clamp_max(self.scale_max)

        rotation_raw = self.rotation(x)
        rotation = F.normalize(rotation_raw, p=2, dim=1, eps=1e-8)

        sh = self.sh(x)
        conf_rpc = torch.sigmoid(self.conf_rpc(x))
        conf_point = torch.sigmoid(self.conf_point(x))

        return {
            "opacity": opacity,
            "scale": scale,
            "rotation": rotation,
            "sh": sh,
            "confidence_rpc": conf_rpc,
            "confidence_point": conf_point,
        }

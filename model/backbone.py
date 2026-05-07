"""backbone.py

本文件包含三类核心模块：
1) DINOv3Backbone: 负责视觉 patch token 提取与 padding/掩码管理；
2) GeometryTokenMLP: 把 30D patch 几何特征编码到 256D；
3) VisualGeometryFuser: 融合视觉 token 与几何 token，再压缩回 1024D。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DINOv3BackboneCfg:
    """DINOv3Backbone 配置。

    字段说明:
        dino_weight_path: dinov3 预训练参数路径。
    """

    dino_weight_path: str = ""


class DINOv3Backbone(nn.Module):
    """DINOv3 视觉 backbone 工程封装。

    功能:
    - 自动加载 third_party/dinov3 模型；
    - 对任意输入 [B*V,3,H,W] 执行右下 padding 到 patch_size 整数倍；
    - 输出 patch token、patch map 及有效掩码；
    - 处理 cls/register token，仅向后续输出 patch token。

    成员变量:
        cfg: 配置对象。
        model: 真实 DINOv3 模型。
        patch_size: patch 大小。
        embed_dim: token 维度。
    """

    def __init__(self, cfg: DINOv3BackboneCfg) -> None:
        """初始化 DINOv3Backbone 并加载模型。"""
        super().__init__()
        self.cfg = cfg
        self.model = self._build_model(cfg)
        self.patch_size = int(getattr(self.model, "patch_size", 16))
        self.embed_dim = int(getattr(self.model, "embed_dim", 1024))

        # DINOv3 当前阶段始终冻结，不参与训练。
        self.model.eval()
        self.model.requires_grad_(False)

    def _build_model(self, cfg: DINOv3BackboneCfg) -> nn.Module:
        """按约定方式加载本地 DINOv3 ViT-L/16 模型。"""
        if cfg.dino_weight_path == "":
            raise ValueError("DINOv3BackboneCfg.dino_weight_path 不能为空，请传入本地预训练权重路径。")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dino_repo = os.path.join(repo_root, "third_party", "dinov3")
        model = torch.hub.load(
            dino_repo,
            "dinov3_vitl16",
            source="local",
            weights=cfg.dino_weight_path,
        )
        return model

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """将输入在右侧和下侧 padding 到 patch_size 的整数倍。"""
        _, _, h, w = x.shape
        p = self.patch_size
        hp = ((h + p - 1) // p) * p
        wp = ((w + p - 1) // p) * p
        pad_h = hp - h
        pad_w = wp - w
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x_pad, (h, w), (hp, wp)

    def _extract_patch_tokens(self, model_out: Any, batch: int, gh: int, gw: int) -> torch.Tensor:
        """从不同风格输出中稳健提取 patch token。"""
        n_patch = gh * gw

        if isinstance(model_out, dict):
            if "x_norm_patchtokens" in model_out:
                return model_out["x_norm_patchtokens"]
            if "patch_tokens" in model_out:
                return model_out["patch_tokens"]

        if torch.is_tensor(model_out):
            tokens = model_out
            if tokens.shape[1] >= n_patch:
                return tokens[:, -n_patch:, :]

        if hasattr(model_out, "x_norm_patchtokens"):
            return getattr(model_out, "x_norm_patchtokens")

        raise RuntimeError("Cannot parse patch tokens from DINOv3 output")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        """提取 patch token 并返回元信息。

        参数:
            images: [B*V, 3, H, W]。

        返回:
            dict 包含:
            - orig_hw: (H,W)
            - pad_hw: (Hp,Wp)
            - grid_hw: (Gh,Gw)
            - patch_tokens: [B*V,N,C]
            - patch_map: [B*V,C,Gh,Gw]
            - patch_valid_mask: [B*V,N]
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B*V,3,H,W], got {tuple(images.shape)}")

        images_pad, orig_hw, pad_hw = self._pad_to_patch_multiple(images)
        hp, wp = pad_hw
        gh, gw = hp // self.patch_size, wp // self.patch_size
        b_all = images.shape[0]

        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(images_pad)
        else:
            out = self.model(images_pad, is_training=True)

        patch_tokens = self._extract_patch_tokens(out, b_all, gh, gw)
        c = patch_tokens.shape[-1]
        patch_map = patch_tokens.view(b_all, gh, gw, c).permute(0, 3, 1, 2).contiguous()

        device = images.device
        dtype = images.dtype

        row_ids = torch.arange(gh, device=device, dtype=dtype)
        col_ids = torch.arange(gw, device=device, dtype=dtype)
        row_grid, col_grid = torch.meshgrid(row_ids, col_ids, indexing="ij")

        center_line = (row_grid + 0.5) * self.patch_size - 0.5
        center_samp = (col_grid + 0.5) * self.patch_size - 0.5

        h, w = orig_hw
        patch_valid = (center_line >= 0) & (center_line <= h - 1) & (center_samp >= 0) & (center_samp <= w - 1)
        patch_valid_mask = patch_valid.reshape(1, gh * gw).expand(b_all, -1).clone()

        return {
            "orig_hw": orig_hw,
            "pad_hw": pad_hw,
            "grid_hw": (gh, gw),
            "patch_tokens": patch_tokens,
            "patch_map": patch_map,
            "patch_valid_mask": patch_valid_mask,
        }


class GeometryTokenMLP(nn.Module):
    """几何 token 编码器：30D -> 256D。

    结构:
        Linear(30->hidden) + GELU + LayerNorm + Linear(hidden->256) + LayerNorm。
    """

    def __init__(self, in_dim: int = 30, hidden_dim: int = 256, out_dim: int = 256) -> None:
        """初始化几何特征 MLP。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, geom_feat: torch.Tensor) -> torch.Tensor:
        """编码几何特征。

        参数:
            geom_feat: [B,V,N,30]。

        返回:
            geom_token: [B,V,N,256]。
        """
        return self.net(geom_feat)


@dataclass
class LocalPatchDetailEncoderCfg:
    """Patch 内细节编码器配置。"""

    patch_size: int = 16
    in_channels: int = 3
    hidden_dim: int = 1024
    out_dim: int = 1024


class LocalPatchDetailEncoder(nn.Module):
    """严格 patch-local 细节编码器。

    实现约束：
    - 先做与 DINO 一致的右下 padding；
    - PixelUnshuffle(patch_size) 将每个 patch 重排到通道；
    - 后续仅用 1x1 Conv 做 channel mixing，不做任何空间混合。
    """

    def __init__(self, cfg: LocalPatchDetailEncoderCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LocalPatchDetailEncoderCfg()
        self.patch_size = int(self.cfg.patch_size)
        in_dim = int(self.cfg.in_channels) * self.patch_size * self.patch_size
        hidden = int(self.cfg.hidden_dim)
        out_dim = int(self.cfg.out_dim)

        self.pixel_unshuffle = nn.PixelUnshuffle(self.patch_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.norm = nn.LayerNorm(out_dim)

    def _pad_to_patch_multiple(self, x: torch.Tensor, pad_hw: tuple[int, int] | None = None) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        _, _, h, w = x.shape
        if pad_hw is not None:
            hp, wp = int(pad_hw[0]), int(pad_hw[1])
            if hp < h or wp < w:
                raise ValueError(f"pad_hw must be >= input hw, got pad_hw={pad_hw}, input={(h, w)}")
            pad_h = hp - h
            pad_w = wp - w
        else:
            p = self.patch_size
            hp = ((h + p - 1) // p) * p
            wp = ((w + p - 1) // p) * p
            pad_h = hp - h
            pad_w = wp - w
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        return x_pad, (h, w), (hp, wp)

    def forward(
        self,
        images: torch.Tensor,
        *,
        orig_hw: tuple[int, int] | None = None,
        pad_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | tuple[int, int]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B*V,3,H,W], got {tuple(images.shape)}")

        x_pad, inferred_orig_hw, inferred_pad_hw = self._pad_to_patch_multiple(images, pad_hw=pad_hw)
        if orig_hw is not None and tuple(orig_hw) != tuple(inferred_orig_hw):
            raise ValueError(f"orig_hw mismatch: got {orig_hw}, inferred {inferred_orig_hw}")

        hp, wp = inferred_pad_hw
        gh, gw = hp // self.patch_size, wp // self.patch_size
        b_all = images.shape[0]

        # 先严格 patch-local 重排，再做纯 channel mixing（1x1）。
        x = self.pixel_unshuffle(x_pad)  # [B*V, 3*P*P, Gh, Gw]
        x = self.encoder(x)
        c = x.shape[1]
        tokens = x.permute(0, 2, 3, 1).reshape(b_all, gh * gw, c).contiguous()
        tokens = self.norm(tokens)

        return {
            "orig_hw": inferred_orig_hw,
            "pad_hw": inferred_pad_hw,
            "grid_hw": (gh, gw),
            "patch_tokens_detail": tokens,
        }


# class VisualGeometryFuser(nn.Module):
#     """视觉-几何 token 融合模块。

#     功能:
#     - 拼接视觉 token (1024) 与几何 token (256) 到 1280；
#     - 线性压回 1024；
#     - LayerNorm 稳定分布。
#     """

#     def __init__(self, visual_dim: int = 1024, geom_dim: int = 256, out_dim: int = 1024) -> None:
#         """初始化融合模块。"""
#         super().__init__()
#         self.proj = nn.Linear(visual_dim + geom_dim, out_dim)
#         self.norm = nn.LayerNorm(out_dim)

#     def forward(self, visual_tokens: torch.Tensor, geom_tokens: torch.Tensor) -> torch.Tensor:
#         """融合视觉与几何 token。

#         参数:
#             visual_tokens: [B,V,N,1024]。
#             geom_tokens: [B,V,N,256]。

#         返回:
#             fused_tokens: [B,V,N,1024]。
#         """
#         x = torch.cat([visual_tokens, geom_tokens], dim=-1)
#         return self.norm(self.proj(x))


class VisualGeometryDetailFuser(nn.Module):
    """视觉 / 细节 / 几何三路 token 直接拼接融合。"""

    def __init__(self, visual_dim: int = 1024, detail_dim: int = 1024, geom_dim: int = 256, out_dim: int = 1024) -> None:
        super().__init__()
        total_dim = visual_dim + detail_dim + geom_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim,total_dim),
            nn.GELU(),
            nn.Linear(total_dim,out_dim)
        )
        # self.proj = nn.Linear(visual_dim + detail_dim + geom_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, visual_tokens: torch.Tensor, detail_tokens: torch.Tensor, geom_tokens: torch.Tensor) -> torch.Tensor:
        x = torch.cat([visual_tokens, detail_tokens, geom_tokens], dim=-1)
        return self.norm(self.proj(x))

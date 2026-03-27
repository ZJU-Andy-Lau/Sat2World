"""encoder.py

本文件实现 Sat2World 的多视图交替编码器：
- IntraViewBlock: 单视图内 self-attention；
- CrossViewBlock: 轻量跨视图信息交换；
- AlternatingEncoder: 交替堆叠 Intra/Cross。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AlternatingEncoderCfg:
    """交替编码器配置。"""

    dim: int = 1024
    num_heads: int = 8
    ffn_ratio: float = 4.0
    num_layers: int = 4
    dropout: float = 0.0


class _PreNormFFN(nn.Module):
    """预归一化前馈网络模块。"""

    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        """初始化 FFN。"""
        super().__init__()
        hidden = int(dim * ratio)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 x + FFN(LN(x)) 残差结构。"""
        return x + self.ffn(self.norm(x))


class IntraViewBlock(nn.Module):
    """单视图内部注意力块。

    输入为每个视图序列 `[view_token + patch_tokens]`，在视图内独立执行 self-attention 与 FFN。
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """初始化 IntraViewBlock。"""
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = _PreNormFFN(dim=dim, ratio=ffn_ratio, dropout=dropout)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        view_tokens: torch.Tensor,
        patch_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行单视图内注意力。

        参数:
            patch_tokens: [B,V,N,C]。
            view_tokens: [B,V,C]。
            patch_valid_mask: [B,V,N]，True 表示有效。

        返回:
            patch_tokens_new: [B,V,N,C]。
            view_tokens_new: [B,V,C]。
        """
        b, v, n, c = patch_tokens.shape
        seq = torch.cat([view_tokens.unsqueeze(2), patch_tokens], dim=2).reshape(b * v, n + 1, c)

        key_padding = torch.zeros((b, v, n + 1), device=patch_tokens.device, dtype=torch.bool)
        key_padding[:, :, 1:] = ~patch_valid_mask
        key_padding = key_padding.reshape(b * v, n + 1)

        seq_norm = self.norm_attn(seq)
        attn_out, _ = self.attn(seq_norm, seq_norm, seq_norm, key_padding_mask=key_padding, need_weights=False)
        seq = seq + self.dropout(attn_out)
        seq = self.ffn(seq)

        seq = seq.reshape(b, v, n + 1, c)
        view_out = seq[:, :, 0, :]
        patch_out = seq[:, :, 1:, :]
        return patch_out, view_out


class CrossViewBlock(nn.Module):
    """轻量跨视图交互块。

    交互流程:
    1) 对每个视图 patch token 做 masked mean pooling，得到 summary；
    2) 组装 [scene_token, all_view_tokens, all_summary_tokens] 做全局自注意力；
    3) 将更新后的 scene/view/summary 通过线性投影注入各视图 patch token。
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """初始化 CrossViewBlock。"""
        super().__init__()
        self.norm_global = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.global_ffn = _PreNormFFN(dim=dim, ratio=ffn_ratio, dropout=dropout)

        self.proj_scene = nn.Linear(dim, dim)
        self.proj_view = nn.Linear(dim, dim)
        self.proj_summary = nn.Linear(dim, dim)
        self.patch_ffn = _PreNormFFN(dim=dim, ratio=2.0, dropout=dropout)

    def _masked_mean(self, patch_tokens: torch.Tensor, patch_valid_mask: torch.Tensor) -> torch.Tensor:
        """对 patch token 做掩码均值池化。"""
        mask = patch_valid_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=2).clamp_min(1.0)
        return (patch_tokens * mask).sum(dim=2) / denom

    def forward(
        self,
        patch_tokens: torch.Tensor,
        view_tokens: torch.Tensor,
        scene_token: torch.Tensor,
        patch_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行跨视图交互。

        参数:
            patch_tokens: [B,V,N,C]。
            view_tokens: [B,V,C]。
            scene_token: [B,C]。
            patch_valid_mask: [B,V,N]。

        返回:
            patch_tokens_new: [B,V,N,C]。
            view_tokens_new: [B,V,C]。
            scene_token_new: [B,C]。
        """
        b, v, n, c = patch_tokens.shape
        summary = self._masked_mean(patch_tokens, patch_valid_mask)  # [B,V,C]

        global_seq = torch.cat([
            scene_token.unsqueeze(1),
            view_tokens,
            summary,
        ], dim=1)  # [B,1+V+V,C]

        global_seq_norm = self.norm_global(global_seq)
        global_out, _ = self.global_attn(global_seq_norm, global_seq_norm, global_seq_norm, need_weights=False)
        global_seq = global_seq + global_out
        global_seq = self.global_ffn(global_seq)

        scene_new = global_seq[:, 0, :]
        view_new = global_seq[:, 1 : 1 + v, :]
        summary_new = global_seq[:, 1 + v :, :]

        inject_scene = self.proj_scene(scene_new).unsqueeze(1).unsqueeze(2)  # [B,1,1,C]
        inject_view = self.proj_view(view_new).unsqueeze(2)  # [B,V,1,C]
        inject_sum = self.proj_summary(summary_new).unsqueeze(2)  # [B,V,1,C]

        patch_new = patch_tokens + inject_scene + inject_view + inject_sum
        patch_new = self.patch_ffn(patch_new)

        return patch_new, view_new, scene_new


class AlternatingEncoder(nn.Module):
    """多视图交替编码器。

    模块职责:
    - 管理可学习 scene/view token 初值；
    - 交替执行 IntraViewBlock 与 CrossViewBlock；
    - 输出最终 patch/view/scene token。
    """

    def __init__(self, cfg: AlternatingEncoderCfg) -> None:
        """初始化 AlternatingEncoder。"""
        super().__init__()
        self.cfg = cfg
        # 参考视图与其他视图分别使用不同的初始化 token（借鉴 VGGT 的双 token 思路）。
        self.scene_token_ref = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.scene_token_other = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.view_token_ref = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.view_token_other = nn.Parameter(torch.zeros(1, 1, cfg.dim))

        self.intra_blocks = nn.ModuleList(
            [IntraViewBlock(cfg.dim, cfg.num_heads, cfg.ffn_ratio, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.cross_blocks = nn.ModuleList(
            [CrossViewBlock(cfg.dim, cfg.num_heads, cfg.ffn_ratio, cfg.dropout) for _ in range(cfg.num_layers)]
        )

        nn.init.trunc_normal_(self.scene_token_ref, std=0.02)
        nn.init.trunc_normal_(self.scene_token_other, std=0.02)
        nn.init.trunc_normal_(self.view_token_ref, std=0.02)
        nn.init.trunc_normal_(self.view_token_other, std=0.02)

    @staticmethod
    def patch_tokens_to_map(patch_tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """把 [B,V,N,C] patch token 还原为 [B,V,C,Gh,Gw]。"""
        b, v, n, c = patch_tokens.shape
        gh, gw = grid_hw
        if n != gh * gw:
            raise ValueError(f"Token number mismatch: N={n}, Gh*Gw={gh*gw}")
        return patch_tokens.view(b, v, gh, gw, c).permute(0, 1, 4, 2, 3).contiguous()

    def _build_ref_mask(self, b: int, v: int, ref_view_idx: int | torch.Tensor) -> torch.Tensor:
        """构建 [B,V] 的参考视图掩码。"""
        if isinstance(ref_view_idx, int):
            idx = torch.full((b,), ref_view_idx, dtype=torch.long, device=self.scene_token_ref.device)
        else:
            idx = ref_view_idx.to(device=self.scene_token_ref.device, dtype=torch.long).view(-1)
            if idx.numel() != b:
                raise ValueError(f"ref_view_idx must have B={b} entries, got {idx.numel()}")
        idx = idx.clamp(0, v - 1)
        mask = torch.zeros((b, v), device=self.scene_token_ref.device, dtype=torch.bool)
        mask[torch.arange(b, device=mask.device), idx] = True
        return mask

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        ref_view_idx: int | torch.Tensor = 0,
    ) -> dict[str, torch.Tensor]:
        """执行交替编码。

        参数:
            patch_tokens: [B,V,N,C]，融合后的视觉-几何 token。
            patch_valid_mask: [B,V,N]，True 为有效 patch。
            ref_view_idx: 参考视图索引，支持 int 或 [B]。

        返回:
            dict:
                patch_tokens: [B,V,N,C]
                view_tokens: [B,V,C]
                scene_token: [B,C]
        """
        if patch_tokens.ndim != 4:
            raise ValueError(f"patch_tokens must be [B,V,N,C], got {tuple(patch_tokens.shape)}")
        if patch_valid_mask.ndim != 3:
            raise ValueError(f"patch_valid_mask must be [B,V,N], got {tuple(patch_valid_mask.shape)}")

        b, v, _, c = patch_tokens.shape
        if c != self.cfg.dim:
            raise ValueError(f"Channel mismatch: got {c}, expect {self.cfg.dim}")

        ref_mask = self._build_ref_mask(b, v, ref_view_idx).to(device=patch_tokens.device)

        view_ref = self.view_token_ref.expand(b, v, -1).to(device=patch_tokens.device)
        view_other = self.view_token_other.expand(b, v, -1).to(device=patch_tokens.device)
        view = torch.where(ref_mask.unsqueeze(-1), view_ref, view_other).contiguous()

        scene_ref = self.scene_token_ref.expand(b, v, -1).to(device=patch_tokens.device)
        scene_other = self.scene_token_other.expand(b, v, -1).to(device=patch_tokens.device)
        scene_per_view = torch.where(ref_mask.unsqueeze(-1), scene_ref, scene_other)
        scene = scene_per_view.mean(dim=1).contiguous()
        patch = patch_tokens

        for intra, cross in zip(self.intra_blocks, self.cross_blocks):
            patch, view = intra(patch, view, patch_valid_mask)
            patch, view, scene = cross(patch, view, scene, patch_valid_mask)

        return {
            "patch_tokens": patch,
            "view_tokens": view,
            "scene_token": scene,
        }

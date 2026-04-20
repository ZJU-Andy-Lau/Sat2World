"""patch_matcher.py

轻量 patch 内匹配器：
- 输入一批 src/ref patch token 对 [M,1024]（来自 encoder 同层 patch token）；
- 使用联合双 token 序列 self-attn + ref<-src cross-attn 交替更新；
- 仅从更新后的 ref token 解码 16x16 heatmap logits。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model.encoder import AttentionSDPA, Mlp


@dataclass
class PatchMatcherCfg:
    in_dim: int = 1024
    match_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ffn_ratio: float = 2.0
    dropout: float = 0.0


class CrossAttention(nn.Module):
    """轻量 cross-attn：query <- kv。"""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=float(dropout), batch_first=True)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, kv, kv, need_weights=False)
        return out


class PatchInteractionBlock(nn.Module):
    """单层交互块。

    设计说明：
    1) self-attn 在 [src, ref] 长度为 2 的联合序列上做，避免单 token self-attn 退化；
    2) cross-attn 显式保留 ref <- src 主路径；
    3) src/ref 各自独立 FFN，均采用 LN + residual 写法。
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_joint = nn.LayerNorm(dim)
        self.self_attn = AttentionSDPA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            qk_norm=True,
            fused_attn=True,
            rope=None,
        )

        self.norm_ref_q = nn.LayerNorm(dim)
        self.norm_src_kv = nn.LayerNorm(dim)
        self.cross_ref_from_src = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)

        self.norm_src_ffn = nn.LayerNorm(dim)
        self.norm_ref_ffn = nn.LayerNorm(dim)
        self.src_ffn = Mlp(in_features=dim, hidden_features=int(dim * ffn_ratio), drop=dropout, bias=True)
        self.ref_ffn = Mlp(in_features=dim, hidden_features=int(dim * ffn_ratio), drop=dropout, bias=True)

    def forward(self, src: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [M,2,C] 联合 self-attn
        joint = torch.stack([src, ref], dim=1)
        joint = joint + self.self_attn(self.norm_joint(joint), pos=None)
        src, ref = joint[:, 0], joint[:, 1]

        # ref <- src cross-attn（主路径）
        ref = ref + self.cross_ref_from_src(self.norm_ref_q(ref).unsqueeze(1), self.norm_src_kv(src).unsqueeze(1)).squeeze(1)

        src = src + self.src_ffn(self.norm_src_ffn(src))
        ref = ref + self.ref_ffn(self.norm_ref_ffn(ref))
        return src, ref


class PatchHeatmapMatcher(nn.Module):
    """patch token 对 -> 16x16 heatmap logits。"""

    def __init__(self, cfg: PatchMatcherCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PatchMatcherCfg()

        self.src_in = nn.Linear(self.cfg.in_dim, self.cfg.match_dim)
        self.ref_in = nn.Linear(self.cfg.in_dim, self.cfg.match_dim)

        self.blocks = nn.ModuleList(
            [
                PatchInteractionBlock(
                    dim=self.cfg.match_dim,
                    num_heads=self.cfg.num_heads,
                    ffn_ratio=self.cfg.ffn_ratio,
                    dropout=self.cfg.dropout,
                )
                for _ in range(max(int(self.cfg.num_layers), 1))
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.cfg.match_dim),
            nn.Linear(self.cfg.match_dim, self.cfg.match_dim),
            nn.GELU(),
            nn.Linear(self.cfg.match_dim, 16 * 16),
        )

    def forward(self, src_token: torch.Tensor, ref_token: torch.Tensor) -> torch.Tensor:
        if src_token.ndim != 2 or ref_token.ndim != 2:
            raise ValueError("src/ref tokens must be [M,C]")
        if src_token.shape != ref_token.shape:
            raise ValueError(f"src/ref shape mismatch: {tuple(src_token.shape)} vs {tuple(ref_token.shape)}")

        src = self.src_in(src_token)
        ref = self.ref_in(ref_token)
        for blk in self.blocks:
            src, ref = blk(src, ref)

        logits = self.head(ref)
        return logits.view(-1, 16, 16)

"""patch_matcher.py

轻量 patch 内匹配器：
- 输入一批 src/ref patch token 对 [M,1024]（来自 encoder 同层 patch token）；
- 分别投影 src/ref token 后直接拼接；
- 使用多层 MLP 解码 16x16 heatmap logits。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PatchMatcherCfg:
    in_dim: int = 1024
    match_dim: int = 256
    num_layers: int = 2


class PatchHeatmapMatcher(nn.Module):
    """patch token 对 -> 16x16 heatmap logits。"""

    def __init__(self, cfg: PatchMatcherCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PatchMatcherCfg()

        self.src_in = nn.Linear(self.cfg.in_dim, self.cfg.match_dim)
        self.ref_in = nn.Linear(self.cfg.in_dim, self.cfg.match_dim)

        num_layers = max(int(self.cfg.num_layers), 1)
        layers: list[nn.Module] = [nn.LayerNorm(self.cfg.match_dim * 2)]

        in_dim = self.cfg.match_dim * 2
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, self.cfg.match_dim))
            layers.append(nn.GELU())
            in_dim = self.cfg.match_dim

        layers.append(nn.Linear(self.cfg.match_dim, 16 * 16))
        self.head = nn.Sequential(*layers)

    def forward(self, src_token: torch.Tensor, ref_token: torch.Tensor) -> torch.Tensor:
        if src_token.ndim != 2 or ref_token.ndim != 2:
            raise ValueError("src/ref tokens must be [M,C]")
        if src_token.shape != ref_token.shape:
            raise ValueError(f"src/ref shape mismatch: {tuple(src_token.shape)} vs {tuple(ref_token.shape)}")

        src = self.src_in(src_token)
        ref = self.ref_in(ref_token)
        feat = torch.cat([src, ref], dim=-1)

        logits = self.head(feat)
        return logits.view(-1, 16, 16)
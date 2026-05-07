"""Early encoder pretraining heads for Sat2World.

All pixel coordinates emitted by these heads are in the crop image coordinate
system and ordered as ``line/samp``.  The heads are intentionally lightweight and
operate only on encoder-side patch/view tokens; they do not depend on dense
geometry, point, Gaussian or render branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GlobalPatchMatchHeadCfg:
    """Configuration for global patch matching.

    ``match_dim`` controls the dimension used for patch-to-patch similarity.
    ``residual_scale`` is measured in pixels and bounds the optional coordinate
    refinement around the soft coarse coordinate.
    """

    match_dim: int = 256
    residual_hidden_dim: int = 512
    residual_scale: float = 16.0
    enable_residual: bool = True


@dataclass
class ProjectionPredictionHeadCfg:
    """Configuration for the rpc_init projection prediction head."""

    hidden_dim: int = 512


@dataclass
class CrossViewHeightHeadCfg:
    """Configuration for cross-view height regression.

    The head returns normalized height ``x`` such that metric height is
    ``x * height_scale + HEIGHT_OFF``.  The default ``height_scale=1000`` follows
    the early-pretrain objective convention.
    """

    hidden_dim: int = 512
    height_scale: float = 1000.0


class GlobalPatchMatchHead(nn.Module):
    """Predict bidirectional global patch correspondences between two views.

    Args:
        patch_tokens: ``[B,2,N,C]`` encoder patch tokens.
        patch_centers: ``[N,2]`` target patch centers in ``line/samp`` order.
        patch_valid_mask: ``[B,2,N]`` valid token mask.
        image_hw: crop image height/width used for debug only.

    Returns:
        A dict containing ``pred_0_to_1`` and ``pred_1_to_0`` pixel coordinates
        with shape ``[B,N,2]`` in ``line/samp`` order, plus matching logits
        ``[B,N,N]`` for both directions.
    """

    def __init__(self, in_dim: int, cfg: GlobalPatchMatchHeadCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GlobalPatchMatchHeadCfg()
        md = int(self.cfg.match_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, md)
        self.logit_scale = nn.Parameter(torch.tensor(float(md) ** -0.5))
        hid = int(self.cfg.residual_hidden_dim)
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(in_dim * 4 + 4),
            nn.Linear(in_dim * 4 + 4, hid),
            nn.GELU(),
            nn.Linear(hid, 2),
        )
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def _direction(
        self,
        src_tok: torch.Tensor,
        tgt_tok: torch.Tensor,
        src_centers: torch.Tensor,
        tgt_centers: torch.Tensor,
        src_valid: torch.Tensor,
        tgt_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.normalize(self.proj(self.norm(src_tok)), dim=-1, eps=1e-6)
        k = F.normalize(self.proj(self.norm(tgt_tok)), dim=-1, eps=1e-6)
        logits = torch.matmul(q, k.transpose(-1, -2)) / self.logit_scale.abs().clamp_min(1e-6)
        valid_logits = src_valid.unsqueeze(-1) & tgt_valid.unsqueeze(-2)
        logits = logits.masked_fill(~valid_logits, -1.0e4)
        prob = torch.softmax(logits.float(), dim=-1).to(dtype=src_tok.dtype)
        prob = prob * tgt_valid.to(dtype=prob.dtype).unsqueeze(1)
        denom = prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        prob = prob / denom
        coarse = torch.matmul(prob, tgt_centers.to(device=src_tok.device, dtype=src_tok.dtype))
        soft_tgt = torch.matmul(prob, tgt_tok)
        if bool(self.cfg.enable_residual):
            b = int(src_tok.shape[0])
            src_c = src_centers.to(device=src_tok.device, dtype=src_tok.dtype).view(1, -1, 2).expand(b, -1, -1)
            inp = torch.cat([src_tok, soft_tgt, src_tok - soft_tgt, src_tok * soft_tgt, coarse, src_c], dim=-1)
            residual = torch.tanh(self.residual_mlp(inp)) * float(self.cfg.residual_scale)
        else:
            residual = torch.zeros_like(coarse)
        pred = coarse + residual
        return pred, logits, coarse

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        image_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        del image_hw
        if patch_tokens.ndim != 4 or patch_tokens.shape[1] != 2:
            raise ValueError(f"GlobalPatchMatchHead expects patch_tokens [B,2,N,C], got {tuple(patch_tokens.shape)}")
        centers = patch_centers.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
        pred01, logits01, coarse01 = self._direction(
            patch_tokens[:, 0], patch_tokens[:, 1], centers, centers, patch_valid_mask[:, 0], patch_valid_mask[:, 1]
        )
        pred10, logits10, coarse10 = self._direction(
            patch_tokens[:, 1], patch_tokens[:, 0], centers, centers, patch_valid_mask[:, 1], patch_valid_mask[:, 0]
        )
        return {
            "pred_0_to_1": pred01,
            "pred_1_to_0": pred10,
            "logits_0_to_1": logits01,
            "logits_1_to_0": logits10,
            "coarse_0_to_1": coarse01,
            "coarse_1_to_0": coarse10,
        }


class ProjectionPredictionHead(nn.Module):
    """Predict rpc_init target-view projection coordinates from patch/view tokens.

    For direction ``i->j``, the input is source patch tokens ``[B,N,C]`` from
    view ``i`` and the target view token ``[B,C]`` from view ``j``.  The shared
    MLP predicts normalized coordinates, which are mapped to crop pixels in
    ``line/samp`` order.
    """

    def __init__(self, in_dim: int, cfg: ProjectionPredictionHeadCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ProjectionPredictionHeadCfg()
        hid = int(self.cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim * 2),
            nn.Linear(in_dim * 2, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _direction(self, src_patch: torch.Tensor, tgt_view: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
        b, n, _ = src_patch.shape
        tgt = tgt_view.view(b, 1, -1).expand(b, n, -1)
        raw = self.mlp(torch.cat([src_patch, tgt], dim=-1))
        norm = torch.sigmoid(raw)
        h, w = int(image_hw[0]), int(image_hw[1])
        line = norm[..., 0] * float(max(h - 1, 1))
        samp = norm[..., 1] * float(max(w - 1, 1))
        return torch.stack([line, samp], dim=-1)

    def forward(self, patch_tokens: torch.Tensor, view_tokens: torch.Tensor, image_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 4 or patch_tokens.shape[1] != 2:
            raise ValueError(f"ProjectionPredictionHead expects patch_tokens [B,2,N,C], got {tuple(patch_tokens.shape)}")
        return {
            "pred_0_to_1": self._direction(patch_tokens[:, 0], view_tokens[:, 1], image_hw),
            "pred_1_to_0": self._direction(patch_tokens[:, 1], view_tokens[:, 0], image_hw),
        }


class CrossViewHeightHead(nn.Module):
    """Regress normalized source height from source and sampled target tokens.

    Inputs are source patch token and target-view token sampled at the rpc_gt
    correspondence pixel.  The returned scalar is normalized height ``x``; the
    metric-height relation is handled by the objective as
    ``height_m = x * height_scale + source_rpc.HEIGHT_OFF``.
    """

    def __init__(self, in_dim: int, cfg: CrossViewHeightHeadCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CrossViewHeightHeadCfg()
        hid = int(self.cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim * 4),
            nn.Linear(in_dim * 4, hid),
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, src_token: torch.Tensor, tgt_sampled_token: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([src_token, tgt_sampled_token, src_token - tgt_sampled_token, src_token * tgt_sampled_token], dim=-1)
        return self.mlp(inp).squeeze(-1)

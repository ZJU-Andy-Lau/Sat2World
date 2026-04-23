"""loss.feature_nce_loss

基于中间层 patch 特征的双向 InfoNCE 监督：
- 对每个样本先从无序视图对集合随机采样若干对；
- 每个视图对同时累积 i->j 与 j->i 两个方向；
- mask: 仅保留投影落在目标图像内、且源 patch 有效的样本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import random_pairwise_view_pairs, sample_map_bilinear


@dataclass
class FeatureInfoNCELossCfg:
    """Feature InfoNCE 配置。"""

    temperature: float = 0.1
    max_pairs: int = 4096
    view_pair_max_pairs: int | None = None


class FeatureInfoNCELoss:
    """中间层 patch 特征单向 InfoNCE。"""

    def __init__(self, geometry_ops: Any, cfg: FeatureInfoNCELossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or FeatureInfoNCELossCfg()

    def __call__(
        self,
        *,
        patch_tokens_proj: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_grid_hw: tuple[int, int],
        patch_padded_hw: tuple[int, int] | None,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """计算单向 InfoNCE。

        参数:
            patch_tokens_proj: [B,V,N,D]，已投影特征。
            patch_valid_mask: [B,V,N]。
            patch_centers: [N,2]，(line,samp)。
            patch_grid_hw: (Gh,Gw)。
            batch: 需包含 rpc_gt/height_gt/height_valid_mask。
        """
        b, v, n, d = patch_tokens_proj.shape
        gh, gw = patch_grid_hw
        if n != gh * gw:
            raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh * gw}")

        device = patch_tokens_proj.device
        dtype = patch_tokens_proj.dtype
        hgt = batch["height_gt"].to(device=device, dtype=dtype)
        hmask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        rpc_gt = batch["rpc_gt"]
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)

        anchors_all: list[torch.Tensor] = []
        positives_all: list[torch.Tensor] = []
        valid_pairs_total = 0

        centers = patch_centers.to(device=device, dtype=dtype).view(1, n, 2)
        feat_map_all = patch_tokens_proj.view(b, v, gh, gw, d).permute(0, 1, 4, 2, 3).contiguous()
        # 关键：将像素坐标投影点映射到 patch-grid 坐标系后再采样 ref_feat_map。
        if patch_padded_hw is not None:
            hp, wp = int(patch_padded_hw[0]), int(patch_padded_hw[1])
        else:
            hp = int(batch["images"].shape[-2])
            wp = int(batch["images"].shape[-1])
        patch_h = float(hp) / float(max(gh, 1))
        patch_w = float(wp) / float(max(gw, 1))
        loss_stub = patch_tokens_proj.sum() * 0.0

        def _collect_direction(bi: int, view_i: int, view_j: int) -> None:
            nonlocal valid_pairs_total
            src_valid = patch_valid_mask[bi : bi + 1, view_i]
            if not bool(src_valid.any()):
                return

            h_src, in_h = sample_map_bilinear(hgt[bi : bi + 1, view_i], centers)
            m_src, in_m = sample_map_bilinear(hmask[bi : bi + 1, view_i], centers)
            h_src = h_src[:, 0]
            valid_src = src_valid & in_h & in_m & (m_src[:, 0] > 0.5)
            if not bool(valid_src.any()):
                return

            pts_src = centers[:, valid_src[0]]
            h_valid = h_src[:, valid_src[0]]
            xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                rpc_batch=[[rpc_gt[bi][view_i]]],
                lines=pts_src[..., 0].view(1, 1, -1),
                samps=pts_src[..., 1].view(1, 1, -1),
                heights=h_valid.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            l_tgt, s_tgt = self.geometry_ops.xy_to_linesamp_batch(
                rpc_batch=[[rpc_gt[bi][view_j]]],
                xs=xs,
                ys=ys,
                heights=h_valid.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            pts_tgt_pix = torch.stack([l_tgt.view(1, -1), s_tgt.view(1, -1)], dim=-1).to(device=device, dtype=dtype)
            pts_tgt_patch = pts_tgt_pix.clone()
            pts_tgt_patch[..., 0] = (pts_tgt_patch[..., 0] + 0.5) / patch_h - 0.5
            pts_tgt_patch[..., 1] = (pts_tgt_patch[..., 1] + 0.5) / patch_w - 0.5
            tgt_feat, in_tgt = sample_map_bilinear(feat_map_all[bi : bi + 1, view_j], pts_tgt_patch)
            if not bool(in_tgt.any()):
                return

            pos = tgt_feat[0].transpose(0, 1)[in_tgt[0]]
            anchor = patch_tokens_proj[bi, view_i, valid_src[0]][in_tgt[0]]
            if pos.numel() == 0 or anchor.numel() == 0:
                return
            anchors_all.append(anchor)
            positives_all.append(pos)
            valid_pairs_total += int(pos.shape[0])

        for bi in range(b):
            view_pairs = random_pairwise_view_pairs(v=v, max_pairs=self.cfg.view_pair_max_pairs, device=device)
            for view_i, view_j in view_pairs:
                _collect_direction(bi, view_i, view_j)
                _collect_direction(bi, view_j, view_i)

        if len(anchors_all) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return loss_stub, {"feature_nce_valid_pairs": zero}, {"feature_nce_valid_pairs": 0}

        anchors = torch.cat(anchors_all, dim=0)
        positives = torch.cat(positives_all, dim=0)

        if anchors.shape[0] > int(self.cfg.max_pairs):
            perm = torch.randperm(anchors.shape[0], device=device)[: int(self.cfg.max_pairs)]
            anchors = anchors[perm]
            positives = positives[perm]

        anchors = F.normalize(anchors, dim=-1, eps=1e-6)
        positives = F.normalize(positives, dim=-1, eps=1e-6)

        logits = torch.matmul(anchors.float(), positives.float().transpose(0, 1)) / float(self.cfg.temperature)
        targets = torch.arange(logits.shape[0], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits, targets).to(dtype=dtype) + loss_stub

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == targets).float().mean()
            valid_pairs = torch.tensor(float(anchors.shape[0]), device=device, dtype=dtype)

        probe = {
            "feature_nce_valid_pairs": valid_pairs.detach(),
            "feature_nce_acc_top1": acc.detach().to(dtype=dtype),
        }
        aux = {
            "feature_nce_valid_pairs": int(anchors.shape[0]),
            "feature_nce_valid_pairs_before_cap": int(valid_pairs_total),
        }
        return loss, probe, aux

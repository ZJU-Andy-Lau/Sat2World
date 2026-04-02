"""loss.feature_nce_loss

基于中间层 patch 特征的单向 InfoNCE 监督：
- anchor: 非参考视图 patch 特征；
- positive: 通过 rpc_gt + h_gt 投影到参考视图后，从参考特征图双线性采样得到的特征；
- mask: 仅保留投影落在参考图像内、且源 patch 有效的样本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import sample_map_bilinear


@dataclass
class FeatureInfoNCELossCfg:
    """Feature InfoNCE 配置。"""

    temperature: float = 0.1
    max_pairs: int = 4096


class FeatureInfoNCELoss:
    """中间层 patch 特征单向 InfoNCE。"""

    def __init__(self, geometry_ops: Any, cfg: FeatureInfoNCELossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or FeatureInfoNCELossCfg()

    @staticmethod
    def _expand_ref_idx(ref_view_idx: torch.Tensor | None, b: int, v: int, device: torch.device) -> torch.Tensor:
        if ref_view_idx is None:
            return torch.zeros((b,), dtype=torch.long, device=device)
        ref = ref_view_idx.long().to(device=device).view(-1)
        if ref.numel() == 1:
            ref = ref.expand(b)
        return ref.clamp(0, v - 1)

    def __call__(
        self,
        *,
        patch_tokens_proj: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_grid_hw: tuple[int, int],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """计算单向 InfoNCE。

        参数:
            patch_tokens_proj: [B,V,N,D]，已投影特征。
            patch_valid_mask: [B,V,N]。
            patch_centers: [N,2]，(line,samp)。
            patch_grid_hw: (Gh,Gw)。
            batch: 需包含 rpc_gt/height_gt/height_valid_mask/ref_view_idx。
        """
        b, v, n, d = patch_tokens_proj.shape
        gh, gw = patch_grid_hw
        if n != gh * gw:
            raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh * gw}")

        device = patch_tokens_proj.device
        dtype = patch_tokens_proj.dtype
        ref_idx = self._expand_ref_idx(batch.get("ref_view_idx", None), b, v, device)

        hgt = batch["height_gt"].to(device=device, dtype=dtype)
        hmask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        rpc_gt = batch["rpc_gt"]
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)

        anchors_all: list[torch.Tensor] = []
        positives_all: list[torch.Tensor] = []
        valid_pairs_total = 0

        centers = patch_centers.to(device=device, dtype=dtype).view(1, n, 2)
        ref_feat_map_all = patch_tokens_proj.view(b, v, gh, gw, d).permute(0, 1, 4, 2, 3).contiguous()

        for bi in range(b):
            ref = int(ref_idx[bi].item())
            ref_feat_map = ref_feat_map_all[bi : bi + 1, ref]  # [1,D,Gh,Gw]

            for vi in range(v):
                if vi == ref:
                    continue

                src_valid = patch_valid_mask[bi : bi + 1, vi]  # [1,N]
                if not bool(src_valid.any()):
                    continue

                h_src, in_h = sample_map_bilinear(hgt[bi : bi + 1, vi], centers)
                m_src, in_m = sample_map_bilinear(hmask[bi : bi + 1, vi], centers)
                h_src = h_src[:, 0]
                valid_src = src_valid & in_h & in_m & (m_src[:, 0] > 0.5)
                if not bool(valid_src.any()):
                    continue

                pts_src = centers[:, valid_src[0]]
                h_valid = h_src[:, valid_src[0]]

                xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][vi]]],
                    lines=pts_src[..., 0].view(1, 1, -1),
                    samps=pts_src[..., 1].view(1, 1, -1),
                    heights=h_valid.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                l_ref, s_ref = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][ref]]],
                    xs=xs,
                    ys=ys,
                    heights=h_valid.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                pts_ref = torch.stack([l_ref.view(1, -1), s_ref.view(1, -1)], dim=-1).to(device=device, dtype=dtype)
                pos_feat, in_ref = sample_map_bilinear(ref_feat_map, pts_ref)
                if not bool(in_ref.any()):
                    continue

                pos = pos_feat[0].transpose(0, 1)[in_ref[0]]  # [M,D]
                src_proj = patch_tokens_proj[bi, vi, valid_src[0]][in_ref[0]]  # [M,D]
                if pos.numel() == 0 or src_proj.numel() == 0:
                    continue
                anchors_all.append(src_proj)
                positives_all.append(pos)
                valid_pairs_total += int(pos.shape[0])

        if len(anchors_all) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, {"feature_nce_valid_pairs": zero}, {"feature_nce_valid_pairs": 0}

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
        loss = F.cross_entropy(logits, targets).to(dtype=dtype)

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

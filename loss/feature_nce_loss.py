"""loss.feature_nce_loss

基于中间层 patch 特征的单向 InfoNCE 监督：
- anchor: 源视图 patch 特征；
- positive: 通过 rpc_gt + h_gt 投影到目标视图后，从目标特征图双线性采样得到的特征；
- mask: 仅保留几何与采样均有效且数值有限的样本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import sample_map_bilinear
from loss.correspondence_utils import build_patch_correspondence_gt


@dataclass
class FeatureInfoNCELossCfg:
    """Feature InfoNCE 配置。"""

    temperature: float = 0.1
    max_pairs: int = 4096
    match_max_pair: int = 12


class FeatureInfoNCELoss:
    """中间层 patch 特征单向 InfoNCE。"""

    def __init__(self, geometry_ops: Any, cfg: FeatureInfoNCELossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or FeatureInfoNCELossCfg()

    @staticmethod
    def _sample_view_pairs(v: int, max_pairs: int, device: torch.device) -> list[tuple[int, int]]:
        if v < 2:
            return []
        candidates: list[tuple[int, int]] = [(i, j) for i in range(v) for j in range(v) if i != j]
        if len(candidates) == 0:
            return []
        if max_pairs > 0 and len(candidates) > max_pairs:
            perm = torch.randperm(len(candidates), device=device)[:max_pairs].tolist()
            sampled = [candidates[idx] for idx in perm]
        else:
            sampled = candidates

        # 强制包含 (0,1)，并去重。
        forced = (0, 1)
        out = [forced]
        for pair in sampled:
            if pair != forced:
                out.append(pair)
        seen: set[tuple[int, int]] = set()
        dedup: list[tuple[int, int]] = []
        for pair in out:
            if pair not in seen:
                dedup.append(pair)
                seen.add(pair)
        return dedup

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
        """计算单向 InfoNCE。"""
        b, v, n, d = patch_tokens_proj.shape
        gh, gw = patch_grid_hw
        if n != gh * gw:
            raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh * gw}")

        device = patch_tokens_proj.device
        dtype = patch_tokens_proj.dtype

        anchors_all: list[torch.Tensor] = []
        positives_all: list[torch.Tensor] = []
        valid_pairs_total = 0
        valid_pairs_after_filter = 0
        sampled_pairs_debug: dict[int, list[tuple[int, int]]] = {}

        feat_map_all = patch_tokens_proj.view(b, v, gh, gw, d).permute(0, 1, 4, 2, 3).contiguous()
        if patch_padded_hw is not None:
            hp, wp = int(patch_padded_hw[0]), int(patch_padded_hw[1])
        else:
            hp = int(batch["images"].shape[-2])
            wp = int(batch["images"].shape[-1])
        patch_h = float(hp) / float(max(gh, 1))
        patch_w = float(wp) / float(max(gw, 1))
        loss_stub = patch_tokens_proj.sum() * 0.0

        view_pairs = self._sample_view_pairs(v, int(self.cfg.match_max_pair), device=device)
        for bi in range(b):
            sampled_pairs_debug[int(bi)] = view_pairs
        for vi, vj in view_pairs:
            corr = build_patch_correspondence_gt(
                geometry_ops=self.geometry_ops,
                batch=batch,
                patch_centers=patch_centers,
                patch_valid_mask=patch_valid_mask,
                patch_grid_hw=patch_grid_hw,
                patch_padded_hw=patch_padded_hw,
                src_view_idx=vi,
                tgt_view_idx=vj,
                rpc_key="rpc_gt",
                require_target_patch_valid=False,
            )
            valid_pairs_total += int(corr.before_filter_count)
            if corr.num_valid == 0:
                continue
            for bi in corr.batch_indices.unique(sorted=True).tolist():
                sel = corr.batch_indices == int(bi)
                if not bool(sel.any()):
                    continue
                pts_i = corr.tgt_pixels[sel].view(1, -1, 2).to(device=device, dtype=dtype).clone()
                pts_i[..., 0] = (pts_i[..., 0] + 0.5) / patch_h - 0.5
                pts_i[..., 1] = (pts_i[..., 1] + 0.5) / patch_w - 0.5
                pos_feat, in_tgt = sample_map_bilinear(feat_map_all[int(bi) : int(bi) + 1, vj], pts_i)
                if not bool(in_tgt.any()):
                    continue
                src_proj = patch_tokens_proj[int(bi), vi, corr.src_patch_indices[sel]][in_tgt[0]]
                pos = pos_feat[0].transpose(0, 1)[in_tgt[0]]
                finite_pair = torch.isfinite(src_proj).all(dim=-1) & torch.isfinite(pos).all(dim=-1)
                if not bool(finite_pair.any()):
                    continue
                anchors_all.append(src_proj[finite_pair])
                positives_all.append(pos[finite_pair])
                valid_pairs_after_filter += int(finite_pair.sum().item())

        if len(anchors_all) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return loss_stub, {"feature_nce_valid_pairs": zero}, {
                "feature_nce_valid_pairs": 0,
                "feature_nce_valid_pairs_before_cap": 0,
                "feature_nce_valid_pairs_before_filter": int(valid_pairs_total),
                "feature_nce_sampled_view_pairs": sampled_pairs_debug,
            }

        anchors = torch.cat(anchors_all, dim=0)
        positives = torch.cat(positives_all, dim=0)

        if anchors.shape[0] > int(self.cfg.max_pairs):
            perm = torch.randperm(anchors.shape[0], device=device)[: int(self.cfg.max_pairs)]
            anchors = anchors[perm]
            positives = positives[perm]

        finite_final = torch.isfinite(anchors).all(dim=-1) & torch.isfinite(positives).all(dim=-1)
        anchors = anchors[finite_final]
        positives = positives[finite_final]
        if anchors.shape[0] == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return loss_stub, {"feature_nce_valid_pairs": zero}, {
                "feature_nce_valid_pairs": 0,
                "feature_nce_valid_pairs_before_cap": int(valid_pairs_after_filter),
                "feature_nce_valid_pairs_before_filter": int(valid_pairs_total),
                "feature_nce_sampled_view_pairs": sampled_pairs_debug,
            }

        anchors = F.normalize(anchors, dim=-1, eps=1e-6)
        positives = F.normalize(positives, dim=-1, eps=1e-6)

        temp = max(float(self.cfg.temperature), 1e-6)
        logits = torch.matmul(anchors.float(), positives.float().transpose(0, 1)) / temp
        if not bool(torch.isfinite(logits).all()):
            # 保持 InfoNCE 对角线语义：行/列必须同步过滤，避免 targets 错位。
            finite_rows = torch.isfinite(logits).all(dim=1)
            finite_cols = torch.isfinite(logits).all(dim=0)
            keep = finite_rows & finite_cols
            anchors = anchors[keep]
            positives = positives[keep]
            if anchors.shape[0] == 0:
                zero = torch.zeros((), device=device, dtype=dtype)
                return loss_stub, {"feature_nce_valid_pairs": zero}, {
                    "feature_nce_valid_pairs": 0,
                    "feature_nce_valid_pairs_before_cap": int(valid_pairs_after_filter),
                    "feature_nce_valid_pairs_before_filter": int(valid_pairs_total),
                    "feature_nce_sampled_view_pairs": sampled_pairs_debug,
                }
            logits = torch.matmul(anchors.float(), positives.float().transpose(0, 1)) / temp

        if logits.shape[0] == 0 or not bool(torch.isfinite(logits).all()):
            zero = torch.zeros((), device=device, dtype=dtype)
            return loss_stub, {"feature_nce_valid_pairs": zero}, {
                "feature_nce_valid_pairs": 0,
                "feature_nce_valid_pairs_before_cap": int(valid_pairs_after_filter),
                "feature_nce_valid_pairs_before_filter": int(valid_pairs_total),
                "feature_nce_sampled_view_pairs": sampled_pairs_debug,
            }
        targets = torch.arange(logits.shape[0], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits, targets).to(dtype=dtype) + loss_stub

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == targets).float().mean()
            valid_pairs = torch.tensor(float(logits.shape[0]), device=device, dtype=dtype)

        probe = {
            "feature_nce_valid_pairs": valid_pairs.detach(),
            "feature_nce_acc_top1": acc.detach().to(dtype=dtype),
        }
        aux = {
            "feature_nce_valid_pairs": int(logits.shape[0]),
            "feature_nce_valid_pairs_before_cap": int(valid_pairs_after_filter),
            "feature_nce_valid_pairs_before_filter": int(valid_pairs_total),
            "feature_nce_sampled_view_pairs": sampled_pairs_debug,
        }
        return loss, probe, aux

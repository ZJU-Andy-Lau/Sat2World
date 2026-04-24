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

        hgt = batch["height_gt"].to(device=device, dtype=dtype)
        hmask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        rpc_gt = batch["rpc_gt"]
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)

        anchors_all: list[torch.Tensor] = []
        positives_all: list[torch.Tensor] = []
        valid_pairs_total = 0
        valid_pairs_after_filter = 0
        sampled_pairs_debug: dict[int, list[tuple[int, int]]] = {}

        centers = patch_centers.to(device=device, dtype=dtype).view(1, n, 2)
        feat_map_all = patch_tokens_proj.view(b, v, gh, gw, d).permute(0, 1, 4, 2, 3).contiguous()
        if patch_padded_hw is not None:
            hp, wp = int(patch_padded_hw[0]), int(patch_padded_hw[1])
        else:
            hp = int(batch["images"].shape[-2])
            wp = int(batch["images"].shape[-1])
        patch_h = float(hp) / float(max(gh, 1))
        patch_w = float(wp) / float(max(gw, 1))
        loss_stub = patch_tokens_proj.sum() * 0.0

        for bi in range(b):
            view_pairs = self._sample_view_pairs(v, int(self.cfg.match_max_pair), device=device)
            sampled_pairs_debug[int(bi)] = view_pairs
            for vi, vj in view_pairs:
                tgt_feat_map = feat_map_all[bi : bi + 1, vj]  # [1,D,Gh,Gw]
                src_valid = patch_valid_mask[bi : bi + 1, vi]  # [1,N]
                if not bool(src_valid.any()):
                    continue

                h_src, in_h = sample_map_bilinear(hgt[bi : bi + 1, vi], centers)
                m_src, in_m = sample_map_bilinear(hmask[bi : bi + 1, vi], centers)
                h_src = h_src[:, 0]
                valid_src = src_valid & in_h & in_m & (m_src[:, 0] > 0.5)
                finite_src = torch.isfinite(centers[..., 0]) & torch.isfinite(centers[..., 1]) & torch.isfinite(h_src)
                valid_src = valid_src & finite_src
                if not bool(valid_src.any()):
                    continue

                pts_src = centers[:, valid_src[0]]
                h_valid = h_src[:, valid_src[0]]
                if not bool(torch.isfinite(pts_src).all()) or not bool(torch.isfinite(h_valid).all()):
                    continue

                xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][vi]]],
                    lines=pts_src[..., 0].view(1, 1, -1),
                    samps=pts_src[..., 1].view(1, 1, -1),
                    heights=h_valid.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                l_tgt, s_tgt = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][vj]]],
                    xs=xs,
                    ys=ys,
                    heights=h_valid.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                pts_tgt_pix = torch.stack([l_tgt.view(1, -1), s_tgt.view(1, -1)], dim=-1).to(device=device, dtype=dtype)
                finite_tgt = torch.isfinite(pts_tgt_pix[..., 0]) & torch.isfinite(pts_tgt_pix[..., 1])
                if not bool(finite_tgt.any()):
                    continue
                pts_tgt_pix = pts_tgt_pix[:, finite_tgt[0]]
                src_proj_all = patch_tokens_proj[bi, vi, valid_src[0]][finite_tgt[0]]
                valid_pairs_total += int(src_proj_all.shape[0])
                if pts_tgt_pix.shape[1] == 0:
                    continue

                pts_tgt_patch = pts_tgt_pix.clone()
                pts_tgt_patch[..., 0] = (pts_tgt_patch[..., 0] + 0.5) / patch_h - 0.5
                pts_tgt_patch[..., 1] = (pts_tgt_patch[..., 1] + 0.5) / patch_w - 0.5
                pos_feat, in_tgt = sample_map_bilinear(tgt_feat_map, pts_tgt_patch)
                if not bool(in_tgt.any()):
                    continue

                pos = pos_feat[0].transpose(0, 1)[in_tgt[0]]
                src_proj = src_proj_all[in_tgt[0]]
                finite_pair = torch.isfinite(src_proj).all(dim=-1) & torch.isfinite(pos).all(dim=-1)
                if not bool(finite_pair.any()):
                    continue
                src_proj = src_proj[finite_pair]
                pos = pos[finite_pair]
                if src_proj.numel() == 0 or pos.numel() == 0:
                    continue
                anchors_all.append(src_proj)
                positives_all.append(pos)
                valid_pairs_after_filter += int(pos.shape[0])

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

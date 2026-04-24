"""loss.patch_match_loss

基于交替注意力编码器同层 patch token 的 patch 内匹配监督。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import sample_map_bilinear


@dataclass
class PatchInternalMatchLossCfg:
    patch_size: int = 16
    subpix_weight: float = 0.25
    max_pairs: int = 4096
    match_max_pair: int = 12


class PatchInternalMatchLoss:
    def __init__(self, geometry_ops: Any, patch_matcher: torch.nn.Module, cfg: PatchInternalMatchLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.patch_matcher = patch_matcher
        self.cfg = cfg or PatchInternalMatchLossCfg()

    @staticmethod
    def _sample_view_pairs(v: int, max_pairs: int, device: torch.device) -> list[tuple[int, int]]:
        if v < 2:
            return []
        candidates: list[tuple[int, int]] = [(i, j) for i in range(v) for j in range(v) if i != j]
        if max_pairs > 0 and len(candidates) > max_pairs:
            perm = torch.randperm(len(candidates), device=device)[:max_pairs].tolist()
            sampled = [candidates[idx] for idx in perm]
        else:
            sampled = candidates
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
        patch_tokens_match: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_grid_hw: tuple[int, int],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        b, v, n, _ = patch_tokens_match.shape
        gh, gw = int(patch_grid_hw[0]), int(patch_grid_hw[1])
        if n != gh * gw:
            raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh * gw}")

        device = patch_tokens_match.device
        dtype = patch_tokens_match.dtype
        p = int(self.cfg.patch_size)

        hgt = batch["height_gt"].to(device=device, dtype=dtype)
        hmask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        rpc_gt = batch["rpc_gt"]
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        h_img = int(batch["images"].shape[-2])
        w_img = int(batch["images"].shape[-1])

        centers = patch_centers.to(device=device, dtype=dtype).view(1, n, 2)

        src_tok_all: list[torch.Tensor] = []
        tgt_tok_all: list[torch.Tensor] = []
        tgt_local_all: list[torch.Tensor] = []
        meta_batch_all: list[torch.Tensor] = []
        meta_src_view_all: list[torch.Tensor] = []
        meta_tgt_view_all: list[torch.Tensor] = []
        src_global_pts_all: list[torch.Tensor] = []
        tgt_global_pts_all: list[torch.Tensor] = []
        tgt_patch_topleft_all: list[torch.Tensor] = []
        sampled_pairs_debug: dict[int, list[tuple[int, int]]] = {}
        total_before_filter = 0

        for bi in range(b):
            view_pairs = self._sample_view_pairs(v, int(self.cfg.match_max_pair), device=device)
            sampled_pairs_debug[int(bi)] = view_pairs
            for vi, vj in view_pairs:
                src_valid = patch_valid_mask[bi : bi + 1, vi]
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

                src_idx = valid_src[0].nonzero(as_tuple=False).squeeze(1)
                src_pts = centers[:, src_idx]
                h_valid = h_src[:, src_idx]
                if not bool(torch.isfinite(src_pts).all()) or not bool(torch.isfinite(h_valid).all()):
                    continue

                xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][vi]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
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
                l_tgt = l_tgt.view(-1)
                s_tgt = s_tgt.view(-1)
                finite_tgt = torch.isfinite(l_tgt) & torch.isfinite(s_tgt)
                if not bool(finite_tgt.any()):
                    continue
                src_idx = src_idx[finite_tgt]
                l_tgt = l_tgt[finite_tgt]
                s_tgt = s_tgt[finite_tgt]
                total_before_filter += int(src_idx.shape[0])

                in_tgt = (l_tgt >= 0) & (l_tgt <= (h_img - 1)) & (s_tgt >= 0) & (s_tgt <= (w_img - 1))
                if not bool(in_tgt.any()):
                    continue

                src_idx = src_idx[in_tgt]
                l_tgt = l_tgt[in_tgt]
                s_tgt = s_tgt[in_tgt]

                tgt_row = torch.round((l_tgt + 0.5) / float(p) - 0.5).long().clamp(0, gh - 1)
                tgt_col = torch.round((s_tgt + 0.5) / float(p) - 0.5).long().clamp(0, gw - 1)
                tgt_flat = tgt_row * gw + tgt_col

                tgt_valid = patch_valid_mask[bi, vj, tgt_flat]
                if not bool(tgt_valid.any()):
                    continue

                src_idx = src_idx[tgt_valid]
                tgt_flat = tgt_flat[tgt_valid]
                l_tgt = l_tgt[tgt_valid]
                s_tgt = s_tgt[tgt_valid]
                tgt_row = tgt_row[tgt_valid]
                tgt_col = tgt_col[tgt_valid]

                top_line = tgt_row.to(dtype=dtype) * float(p)
                top_samp = tgt_col.to(dtype=dtype) * float(p)
                local_line = (l_tgt - top_line).clamp(0.0, float(p) - 1e-4)
                local_samp = (s_tgt - top_samp).clamp(0.0, float(p) - 1e-4)
                tgt_local = torch.stack([local_line, local_samp], dim=-1)
                finite_local = torch.isfinite(tgt_local).all(dim=-1)
                if not bool(finite_local.any()):
                    continue

                src_idx = src_idx[finite_local]
                tgt_flat = tgt_flat[finite_local]
                l_tgt = l_tgt[finite_local]
                s_tgt = s_tgt[finite_local]
                tgt_local = tgt_local[finite_local]
                top_line = top_line[finite_local]
                top_samp = top_samp[finite_local]

                src_tok_part = patch_tokens_match[bi, vi, src_idx]
                tgt_tok_part = patch_tokens_match[bi, vj, tgt_flat]
                finite_tok = torch.isfinite(src_tok_part).all(dim=-1) & torch.isfinite(tgt_tok_part).all(dim=-1)
                if not bool(finite_tok.any()):
                    continue

                src_tok_part = src_tok_part[finite_tok]
                tgt_tok_part = tgt_tok_part[finite_tok]
                tgt_local = tgt_local[finite_tok]
                l_tgt = l_tgt[finite_tok]
                s_tgt = s_tgt[finite_tok]
                top_line = top_line[finite_tok]
                top_samp = top_samp[finite_tok]
                src_idx = src_idx[finite_tok]

                src_tok_all.append(src_tok_part)
                tgt_tok_all.append(tgt_tok_part)
                tgt_local_all.append(tgt_local)

                n_pair = int(src_tok_part.shape[0])
                src_global = centers[0, src_idx]
                tgt_global = torch.stack([l_tgt, s_tgt], dim=-1)
                tgt_patch_topleft = torch.stack([top_line, top_samp], dim=-1)
                meta_batch_all.append(torch.full((n_pair,), int(bi), dtype=torch.long, device=device))
                meta_src_view_all.append(torch.full((n_pair,), int(vi), dtype=torch.long, device=device))
                meta_tgt_view_all.append(torch.full((n_pair,), int(vj), dtype=torch.long, device=device))
                src_global_pts_all.append(src_global)
                tgt_global_pts_all.append(tgt_global)
                tgt_patch_topleft_all.append(tgt_patch_topleft)

        param_stub = torch.zeros((), device=device, dtype=dtype)
        for p_match in self.patch_matcher.parameters():
            param_stub = param_stub + p_match.sum().to(device=device, dtype=dtype) * 0.0
        loss_stub = patch_tokens_match.sum() * 0.0 + param_stub

        if len(src_tok_all) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            probe = {
                "patch_match_valid_pairs": zero,
                "patch_match_acc_top1_1px": zero,
                "patch_match_l1_px": zero,
            }
            aux = {
                "patch_match_valid_pairs": 0,
                "patch_match_valid_pairs_before_cap": 0,
                "patch_match_valid_pairs_before_filter": int(total_before_filter),
                "patch_match_sampled_view_pairs": sampled_pairs_debug,
            }
            return loss_stub, probe, aux

        src_tok = torch.cat(src_tok_all, dim=0)
        tgt_tok = torch.cat(tgt_tok_all, dim=0)
        tgt_local = torch.cat(tgt_local_all, dim=0)
        meta_batch = torch.cat(meta_batch_all, dim=0)
        meta_src_view = torch.cat(meta_src_view_all, dim=0)
        meta_tgt_view = torch.cat(meta_tgt_view_all, dim=0)
        src_global_pts = torch.cat(src_global_pts_all, dim=0)
        tgt_global_pts = torch.cat(tgt_global_pts_all, dim=0)
        tgt_patch_topleft = torch.cat(tgt_patch_topleft_all, dim=0)
        total_before_cap = int(src_tok.shape[0])

        if src_tok.shape[0] > int(self.cfg.max_pairs):
            perm = torch.randperm(src_tok.shape[0], device=device)[: int(self.cfg.max_pairs)]
            src_tok = src_tok[perm]
            tgt_tok = tgt_tok[perm]
            tgt_local = tgt_local[perm]
            meta_batch = meta_batch[perm]
            meta_src_view = meta_src_view[perm]
            meta_tgt_view = meta_tgt_view[perm]
            src_global_pts = src_global_pts[perm]
            tgt_global_pts = tgt_global_pts[perm]
            tgt_patch_topleft = tgt_patch_topleft[perm]

        finite_all = (
            torch.isfinite(src_tok).all(dim=-1)
            & torch.isfinite(tgt_tok).all(dim=-1)
            & torch.isfinite(tgt_local).all(dim=-1)
            & torch.isfinite(src_global_pts).all(dim=-1)
            & torch.isfinite(tgt_global_pts).all(dim=-1)
            & torch.isfinite(tgt_patch_topleft).all(dim=-1)
        )
        src_tok = src_tok[finite_all]
        tgt_tok = tgt_tok[finite_all]
        tgt_local = tgt_local[finite_all]
        meta_batch = meta_batch[finite_all]
        meta_src_view = meta_src_view[finite_all]
        meta_tgt_view = meta_tgt_view[finite_all]
        src_global_pts = src_global_pts[finite_all]
        tgt_global_pts = tgt_global_pts[finite_all]
        tgt_patch_topleft = tgt_patch_topleft[finite_all]

        if src_tok.shape[0] == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            probe = {
                "patch_match_valid_pairs": zero,
                "patch_match_acc_top1_1px": zero,
                "patch_match_l1_px": zero,
            }
            aux = {
                "patch_match_valid_pairs": 0,
                "patch_match_valid_pairs_before_cap": total_before_cap,
                "patch_match_valid_pairs_before_filter": int(total_before_filter),
                "patch_match_sampled_view_pairs": sampled_pairs_debug,
            }
            return loss_stub, probe, aux

        logits = self.patch_matcher(src_tok, tgt_tok)
        logits_flat = logits.view(logits.shape[0], -1)
        finite_logits = torch.isfinite(logits_flat).all(dim=-1) & torch.isfinite(tgt_local).all(dim=-1)
        logits_flat = logits_flat[finite_logits]
        tgt_local = tgt_local[finite_logits]
        meta_batch = meta_batch[finite_logits]
        meta_src_view = meta_src_view[finite_logits]
        meta_tgt_view = meta_tgt_view[finite_logits]
        src_global_pts = src_global_pts[finite_logits]
        tgt_global_pts = tgt_global_pts[finite_logits]
        tgt_patch_topleft = tgt_patch_topleft[finite_logits]

        if logits_flat.shape[0] == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            probe = {
                "patch_match_valid_pairs": zero,
                "patch_match_acc_top1_1px": zero,
                "patch_match_l1_px": zero,
            }
            aux = {
                "patch_match_valid_pairs": 0,
                "patch_match_valid_pairs_before_cap": total_before_cap,
                "patch_match_valid_pairs_before_filter": int(total_before_filter),
                "patch_match_sampled_view_pairs": sampled_pairs_debug,
            }
            return loss_stub, probe, aux

        tgt_y = torch.floor(tgt_local[:, 0]).long().clamp(0, p - 1)
        tgt_x = torch.floor(tgt_local[:, 1]).long().clamp(0, p - 1)
        tgt_cls = tgt_y * p + tgt_x
        loss_ce = F.cross_entropy(logits_flat.float(), tgt_cls).to(dtype=dtype)

        prob = torch.softmax(logits_flat.float(), dim=-1).view(-1, p, p)
        yy = torch.arange(p, device=device, dtype=prob.dtype).view(1, p, 1)
        xx = torch.arange(p, device=device, dtype=prob.dtype).view(1, 1, p)
        pred_y = (prob * yy).sum(dim=(1, 2))
        pred_x = (prob * xx).sum(dim=(1, 2))
        pred_local = torch.stack([pred_y, pred_x], dim=-1).to(dtype=dtype)
        loss_subpix = F.smooth_l1_loss(pred_local, tgt_local)

        loss = loss_ce + float(self.cfg.subpix_weight) * loss_subpix + loss_stub

        with torch.no_grad():
            argmax = logits_flat.argmax(dim=-1)
            am_y = (argmax // p).to(dtype=dtype)
            am_x = (argmax % p).to(dtype=dtype)
            err = torch.sqrt((am_y - tgt_local[:, 0]) ** 2 + (am_x - tgt_local[:, 1]) ** 2)
            acc_1px = (err <= 1.0).to(dtype=dtype).mean()
            l1_px = (pred_local - tgt_local).abs().sum(dim=-1).mean()
            valid_pairs = torch.tensor(float(logits_flat.shape[0]), device=device, dtype=dtype)

        probe = {
            "patch_match_valid_pairs": valid_pairs.detach(),
            "patch_match_acc_top1_1px": acc_1px.detach(),
            "patch_match_l1_px": l1_px.detach(),
        }
        aux = {
            "patch_match_valid_pairs": int(logits_flat.shape[0]),
            "patch_match_valid_pairs_before_cap": total_before_cap,
            "patch_match_valid_pairs_before_filter": int(total_before_filter),
            "patch_match_sampled_view_pairs": sampled_pairs_debug,
        }

        with torch.no_grad():
            if meta_batch.numel() > 0:
                unique_pairs = torch.stack([meta_batch, meta_src_view, meta_tgt_view], dim=-1)
                pair_count: dict[tuple[int, int, int], int] = {}
                for row in unique_pairs.tolist():
                    key = (int(row[0]), int(row[1]), int(row[2]))
                    pair_count[key] = pair_count.get(key, 0) + 1
                keys = list(pair_count.keys())
                sel = keys[int(torch.randint(0, len(keys), (1,), device=device).item())]
                bi_sel, src_sel, tgt_sel = sel
                sel_mask = (meta_batch == bi_sel) & (meta_src_view == src_sel) & (meta_tgt_view == tgt_sel)
                if bool(sel_mask.any()):
                    src_vis = src_global_pts[sel_mask].detach()
                    tgt_gt_vis = tgt_global_pts[sel_mask].detach()
                    pred_vis = (tgt_patch_topleft[sel_mask] + pred_local[sel_mask]).detach()
                    aux["patch_match_vis"] = {
                        "batch_index": int(bi_sel),
                        "src_view_index": int(src_sel),
                        "tgt_view_index": int(tgt_sel),
                        "src_points": src_vis,
                        "tgt_points_gt": tgt_gt_vis,
                        "tgt_points_pred": pred_vis,
                        "num_pairs_selected_view_pair": int(src_vis.shape[0]),
                        "num_pairs_total_after_cap": int(logits_flat.shape[0]),
                        "num_pairs_total_before_cap": int(total_before_cap),
                        "num_pairs_total_before_filter": int(total_before_filter),
                    }
        return loss, probe, aux

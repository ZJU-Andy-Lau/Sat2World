"""loss.patch_match_loss

基于交替注意力编码器同层 patch token 的 patch 内匹配监督。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import random_pairwise_view_pairs, sample_map_bilinear


@dataclass
class PatchInternalMatchLossCfg:
    patch_size: int = 16
    subpix_weight: float = 0.25
    max_pairs: int = 4096
    view_pair_max_pairs: int | None = None


class PatchInternalMatchLoss:
    def __init__(self, geometry_ops: Any, patch_matcher: torch.nn.Module, cfg: PatchInternalMatchLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.patch_matcher = patch_matcher
        self.cfg = cfg or PatchInternalMatchLossCfg()

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
        ref_tok_all: list[torch.Tensor] = []
        tgt_local_all: list[torch.Tensor] = []
        meta_batch_all: list[torch.Tensor] = []
        meta_view_i_all: list[torch.Tensor] = []
        meta_view_j_all: list[torch.Tensor] = []
        points_i_all: list[torch.Tensor] = []
        points_j_gt_all: list[torch.Tensor] = []
        top_line_all: list[torch.Tensor] = []
        top_samp_all: list[torch.Tensor] = []
        dir_total = 0
        dir_no_src_valid = 0
        dir_no_valid_src = 0
        dir_zero_overlap = 0
        dir_no_tgt_valid_patch = 0
        dir_used = 0

        def _collect_direction(bi: int, view_i: int, view_j: int) -> None:
            nonlocal dir_total, dir_no_src_valid, dir_no_valid_src, dir_zero_overlap, dir_no_tgt_valid_patch, dir_used
            dir_total += 1
            src_valid = patch_valid_mask[bi : bi + 1, view_i]
            if not bool(src_valid.any()):
                dir_no_src_valid += 1
                return

            h_src, in_h = sample_map_bilinear(hgt[bi : bi + 1, view_i], centers)
            m_src, in_m = sample_map_bilinear(hmask[bi : bi + 1, view_i], centers)
            h_src = h_src[:, 0]
            valid_src = src_valid & in_h & in_m & (m_src[:, 0] > 0.5)
            if not bool(valid_src.any()):
                dir_no_valid_src += 1
                return

            src_idx = valid_src[0].nonzero(as_tuple=False).squeeze(1)
            src_pts = centers[:, src_idx]
            h_valid = h_src[:, src_idx]

            xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                rpc_batch=[[rpc_gt[bi][view_i]]],
                lines=src_pts[..., 0].view(1, 1, -1),
                samps=src_pts[..., 1].view(1, 1, -1),
                heights=h_valid.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            l_j, s_j = self.geometry_ops.xy_to_linesamp_batch(
                rpc_batch=[[rpc_gt[bi][view_j]]],
                xs=xs,
                ys=ys,
                heights=h_valid.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            l_j = l_j.view(-1)
            s_j = s_j.view(-1)

            in_j = (l_j >= 0) & (l_j <= (h_img - 1)) & (s_j >= 0) & (s_j <= (w_img - 1))
            if not bool(in_j.any()):
                dir_zero_overlap += 1
                return

            src_idx = src_idx[in_j]
            l_j = l_j[in_j]
            s_j = s_j[in_j]
            view_j_row = torch.round((l_j + 0.5) / float(p) - 0.5).long().clamp(0, gh - 1)
            view_j_col = torch.round((s_j + 0.5) / float(p) - 0.5).long().clamp(0, gw - 1)
            view_j_flat = view_j_row * gw + view_j_col

            view_j_valid = patch_valid_mask[bi, view_j, view_j_flat]
            if not bool(view_j_valid.any()):
                dir_no_tgt_valid_patch += 1
                return

            src_idx = src_idx[view_j_valid]
            view_j_flat = view_j_flat[view_j_valid]
            l_j = l_j[view_j_valid]
            s_j = s_j[view_j_valid]
            view_j_row = view_j_row[view_j_valid]
            view_j_col = view_j_col[view_j_valid]
            top_line = view_j_row.to(dtype=dtype) * float(p)
            top_samp = view_j_col.to(dtype=dtype) * float(p)
            local_line = (l_j - top_line).clamp(0.0, float(p) - 1e-4)
            local_samp = (s_j - top_samp).clamp(0.0, float(p) - 1e-4)
            tgt_local = torch.stack([local_line, local_samp], dim=-1)

            src_tok_all.append(patch_tokens_match[bi, view_i, src_idx])
            ref_tok_all.append(patch_tokens_match[bi, view_j, view_j_flat])
            tgt_local_all.append(tgt_local)

            n_pair = int(src_idx.shape[0])
            meta_batch_all.append(torch.full((n_pair,), int(bi), dtype=torch.long, device=device))
            meta_view_i_all.append(torch.full((n_pair,), int(view_i), dtype=torch.long, device=device))
            meta_view_j_all.append(torch.full((n_pair,), int(view_j), dtype=torch.long, device=device))
            points_i_all.append(centers[0, src_idx])
            points_j_gt_all.append(torch.stack([l_j, s_j], dim=-1))
            top_line_all.append(top_line)
            top_samp_all.append(top_samp)
            dir_used += 1

        for bi in range(b):
            view_pairs = random_pairwise_view_pairs(v=v, max_pairs=self.cfg.view_pair_max_pairs, device=device)
            for view_i, view_j in view_pairs:
                _collect_direction(bi, view_i, view_j)
                _collect_direction(bi, view_j, view_i)

        # 空样本安全：与 matcher 参数保持图连接，避免 DDP unused parameter
        param_stub = torch.zeros((), device=device, dtype=dtype)
        for p_match in self.patch_matcher.parameters():
            param_stub = param_stub + p_match.sum().to(device=device, dtype=dtype) * 0.0
        loss_stub = patch_tokens_match.sum() * 0.0 + param_stub
        input_nonfinite_ratio = (~torch.isfinite(patch_tokens_match)).to(dtype=dtype).mean()

        if len(src_tok_all) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            dir_total_t = torch.tensor(float(dir_total), device=device, dtype=dtype)
            dir_zero_overlap_t = torch.tensor(float(dir_zero_overlap), device=device, dtype=dtype)
            dir_used_t = torch.tensor(float(dir_used), device=device, dtype=dtype)
            probe = {
                "patch_match_valid_pairs": zero,
                "patch_match_acc_top1_1px": zero,
                "patch_match_l1_px": zero,
                "patch_match_input_nonfinite_ratio": input_nonfinite_ratio.detach(),
                "patch_match_src_nonfinite_ratio": zero,
                "patch_match_ref_nonfinite_ratio": zero,
                "patch_match_logits_nonfinite_ratio": zero,
                "patch_match_num_directions_total": dir_total_t,
                "patch_match_num_directions_zero_overlap": dir_zero_overlap_t,
                "patch_match_num_directions_used": dir_used_t,
            }
            aux = {
                "patch_match_valid_pairs": 0,
                "patch_match_valid_pairs_before_cap": 0,
                "patch_match_num_directions_total": dir_total,
                "patch_match_num_directions_no_src_valid": dir_no_src_valid,
                "patch_match_num_directions_no_valid_src": dir_no_valid_src,
                "patch_match_num_directions_zero_overlap": dir_zero_overlap,
                "patch_match_num_directions_no_tgt_valid_patch": dir_no_tgt_valid_patch,
                "patch_match_num_directions_used": dir_used,
            }
            return loss_stub, probe, aux

        src_tok = torch.cat(src_tok_all, dim=0)
        ref_tok = torch.cat(ref_tok_all, dim=0)
        tgt_local = torch.cat(tgt_local_all, dim=0)
        meta_batch = torch.cat(meta_batch_all, dim=0)
        meta_view_i = torch.cat(meta_view_i_all, dim=0)
        meta_view_j = torch.cat(meta_view_j_all, dim=0)
        points_i = torch.cat(points_i_all, dim=0)
        points_j_gt = torch.cat(points_j_gt_all, dim=0)
        top_line = torch.cat(top_line_all, dim=0)
        top_samp = torch.cat(top_samp_all, dim=0)
        total_before_cap = int(src_tok.shape[0])

        if src_tok.shape[0] > int(self.cfg.max_pairs):
            perm = torch.randperm(src_tok.shape[0], device=device)[: int(self.cfg.max_pairs)]
            src_tok = src_tok[perm]
            ref_tok = ref_tok[perm]
            tgt_local = tgt_local[perm]
            meta_batch = meta_batch[perm]
            meta_view_i = meta_view_i[perm]
            meta_view_j = meta_view_j[perm]
            points_i = points_i[perm]
            points_j_gt = points_j_gt[perm]
            top_line = top_line[perm]
            top_samp = top_samp[perm]
        src_nonfinite_ratio = (~torch.isfinite(src_tok)).to(dtype=dtype).mean()
        ref_nonfinite_ratio = (~torch.isfinite(ref_tok)).to(dtype=dtype).mean()

        logits = self.patch_matcher(src_tok, ref_tok)  # [M,16,16]
        logits_flat = logits.view(logits.shape[0], -1)
        logits_nonfinite_ratio = (~torch.isfinite(logits_flat)).to(dtype=dtype).mean()

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
        points_j_pred = torch.stack([top_line + pred_local[:, 0], top_samp + pred_local[:, 1]], dim=-1)
        loss_subpix = F.smooth_l1_loss(pred_local, tgt_local)

        loss = loss_ce + float(self.cfg.subpix_weight) * loss_subpix + loss_stub

        with torch.no_grad():
            argmax = logits_flat.argmax(dim=-1)
            am_y = (argmax // p).to(dtype=dtype)
            am_x = (argmax % p).to(dtype=dtype)
            err = torch.sqrt((am_y - tgt_local[:, 0]) ** 2 + (am_x - tgt_local[:, 1]) ** 2)
            acc_1px = (err <= 1.0).to(dtype=dtype).mean()
            l1_px = (pred_local - tgt_local).abs().sum(dim=-1).mean()
            valid_pairs = torch.tensor(float(src_tok.shape[0]), device=device, dtype=dtype)
            dir_total_t = torch.tensor(float(dir_total), device=device, dtype=dtype)
            dir_zero_overlap_t = torch.tensor(float(dir_zero_overlap), device=device, dtype=dtype)
            dir_used_t = torch.tensor(float(dir_used), device=device, dtype=dtype)

        probe = {
            "patch_match_valid_pairs": valid_pairs.detach(),
            "patch_match_acc_top1_1px": acc_1px.detach(),
            "patch_match_l1_px": l1_px.detach(),
            "patch_match_input_nonfinite_ratio": input_nonfinite_ratio.detach(),
            "patch_match_src_nonfinite_ratio": src_nonfinite_ratio.detach(),
            "patch_match_ref_nonfinite_ratio": ref_nonfinite_ratio.detach(),
            "patch_match_logits_nonfinite_ratio": logits_nonfinite_ratio.detach(),
            "patch_match_num_directions_total": dir_total_t,
            "patch_match_num_directions_zero_overlap": dir_zero_overlap_t,
            "patch_match_num_directions_used": dir_used_t,
        }
        aux = {
            "patch_match_valid_pairs": int(src_tok.shape[0]),
            "patch_match_valid_pairs_before_cap": total_before_cap,
            "patch_match_num_directions_total": dir_total,
            "patch_match_num_directions_no_src_valid": dir_no_src_valid,
            "patch_match_num_directions_no_valid_src": dir_no_valid_src,
            "patch_match_num_directions_zero_overlap": dir_zero_overlap,
            "patch_match_num_directions_no_tgt_valid_patch": dir_no_tgt_valid_patch,
            "patch_match_num_directions_used": dir_used,
        }

        # 从“截断后真正参与 loss 的匹配集合”中选择一个真实视图对，供 TensorBoard 可视化。
        with torch.no_grad():
            if meta_batch.numel() > 0:
                pair_keys = torch.stack([meta_batch, meta_view_i, meta_view_j], dim=1)
                uniq_pairs = torch.unique(pair_keys, dim=0)
                pick = torch.randint(0, int(uniq_pairs.shape[0]), size=(1,), device=device)[0]
                pair_sel = uniq_pairs[pick]
                bi_sel = int(pair_sel[0].item())
                view_i_sel = int(pair_sel[1].item())
                view_j_sel = int(pair_sel[2].item())
                sel_mask = (meta_batch == bi_sel) & (meta_view_i == view_i_sel) & (meta_view_j == view_j_sel)
                if bool(sel_mask.any()):
                    points_i_vis = points_i[sel_mask].detach()
                    points_j_gt_vis = points_j_gt[sel_mask].detach()
                    points_j_pred_vis = points_j_pred[sel_mask].detach()
                    aux["patch_match_vis"] = {
                        "batch_index": bi_sel,
                        "view_i_index": view_i_sel,
                        "view_j_index": view_j_sel,
                        "points_i": points_i_vis,
                        "points_j_gt": points_j_gt_vis,
                        "points_j_pred": points_j_pred_vis,
                        "num_pairs_selected_view_pair": int(points_i_vis.shape[0]),
                        "num_pairs_total_after_cap": int(src_tok.shape[0]),
                        "num_pairs_total_before_cap": int(total_before_cap),
                    }
        return loss, probe, aux

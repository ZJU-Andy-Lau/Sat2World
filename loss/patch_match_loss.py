"""loss.patch_match_loss

基于 patch-local detail token 的 patch 内匹配监督。
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


class PatchInternalMatchLoss:
    def __init__(self, geometry_ops: Any, patch_matcher: torch.nn.Module, cfg: PatchInternalMatchLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.patch_matcher = patch_matcher
        self.cfg = cfg or PatchInternalMatchLossCfg()

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
        patch_tokens_detail: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        patch_centers: torch.Tensor,
        patch_grid_hw: tuple[int, int],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        b, v, n, _ = patch_tokens_detail.shape
        gh, gw = int(patch_grid_hw[0]), int(patch_grid_hw[1])
        if n != gh * gw:
            raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh * gw}")

        device = patch_tokens_detail.device
        dtype = patch_tokens_detail.dtype
        p = int(self.cfg.patch_size)
        ref_idx = self._expand_ref_idx(batch.get("ref_view_idx", None), b, v, device)

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

        for bi in range(b):
            ref = int(ref_idx[bi].item())
            for vi in range(v):
                if vi == ref:
                    continue
                src_valid = patch_valid_mask[bi : bi + 1, vi]
                if not bool(src_valid.any()):
                    continue

                h_src, in_h = sample_map_bilinear(hgt[bi : bi + 1, vi], centers)
                m_src, in_m = sample_map_bilinear(hmask[bi : bi + 1, vi], centers)
                h_src = h_src[:, 0]
                valid_src = src_valid & in_h & in_m & (m_src[:, 0] > 0.5)
                if not bool(valid_src.any()):
                    continue

                src_idx = valid_src[0].nonzero(as_tuple=False).squeeze(1)
                src_pts = centers[:, src_idx]
                h_valid = h_src[:, src_idx]

                xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][vi]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
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
                l_ref = l_ref.view(-1)
                s_ref = s_ref.view(-1)

                in_ref = (l_ref >= 0) & (l_ref <= (h_img - 1)) & (s_ref >= 0) & (s_ref <= (w_img - 1))
                if not bool(in_ref.any()):
                    continue

                src_idx = src_idx[in_ref]
                l_ref = l_ref[in_ref]
                s_ref = s_ref[in_ref]

                ref_row = torch.round((l_ref + 0.5) / float(p) - 0.5).long().clamp(0, gh - 1)
                ref_col = torch.round((s_ref + 0.5) / float(p) - 0.5).long().clamp(0, gw - 1)
                ref_flat = ref_row * gw + ref_col

                ref_valid = patch_valid_mask[bi, ref, ref_flat]
                if not bool(ref_valid.any()):
                    continue

                src_idx = src_idx[ref_valid]
                ref_flat = ref_flat[ref_valid]
                l_ref = l_ref[ref_valid]
                s_ref = s_ref[ref_valid]

                top_line = ref_row[ref_valid].to(dtype=dtype) * float(p)
                top_samp = ref_col[ref_valid].to(dtype=dtype) * float(p)
                local_line = (l_ref - top_line).clamp(0.0, float(p) - 1e-4)
                local_samp = (s_ref - top_samp).clamp(0.0, float(p) - 1e-4)
                tgt_local = torch.stack([local_line, local_samp], dim=-1)

                src_tok_all.append(patch_tokens_detail[bi, vi, src_idx])
                ref_tok_all.append(patch_tokens_detail[bi, ref, ref_flat])
                tgt_local_all.append(tgt_local)

        # 空样本安全：与 matcher 参数保持图连接，避免 DDP unused parameter
        param_stub = torch.zeros((), device=device, dtype=dtype)
        for p_match in self.patch_matcher.parameters():
            param_stub = param_stub + p_match.sum().to(device=device, dtype=dtype) * 0.0
            break
        loss_stub = patch_tokens_detail.sum() * 0.0 + param_stub

        if len(src_tok_all) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            probe = {
                "patch_match_valid_pairs": zero,
                "patch_match_acc_top1_1px": zero,
                "patch_match_l1_px": zero,
            }
            aux = {"patch_match_valid_pairs": 0, "patch_match_valid_pairs_before_cap": 0}
            return loss_stub, probe, aux

        src_tok = torch.cat(src_tok_all, dim=0)
        ref_tok = torch.cat(ref_tok_all, dim=0)
        tgt_local = torch.cat(tgt_local_all, dim=0)
        total_before_cap = int(src_tok.shape[0])

        if src_tok.shape[0] > int(self.cfg.max_pairs):
            perm = torch.randperm(src_tok.shape[0], device=device)[: int(self.cfg.max_pairs)]
            src_tok = src_tok[perm]
            ref_tok = ref_tok[perm]
            tgt_local = tgt_local[perm]

        logits = self.patch_matcher(src_tok, ref_tok)  # [M,16,16]
        logits_flat = logits.view(logits.shape[0], -1)

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
            valid_pairs = torch.tensor(float(src_tok.shape[0]), device=device, dtype=dtype)

        probe = {
            "patch_match_valid_pairs": valid_pairs.detach(),
            "patch_match_acc_top1_1px": acc_1px.detach(),
            "patch_match_l1_px": l1_px.detach(),
        }
        aux = {
            "patch_match_valid_pairs": int(src_tok.shape[0]),
            "patch_match_valid_pairs_before_cap": total_before_cap,
        }
        return loss, probe, aux

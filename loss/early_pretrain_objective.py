"""Full early encoder-pretraining objective for Sat2World.

This objective only consumes early-forward outputs: encoder patch/view tokens,
patch masks/centers, and early-head predictions.  It does not depend on dense
geometry outputs such as affine predictions, corrected RPCs, dense height maps,
point maps, Gaussian attributes or renderer outputs.

RPC supervision semantics:
- feature NCE, patch-local match, global match and cross-view height use
  ``rpc_gt`` because they supervise true cross-view geometry;
- projection prediction uses ``rpc_init`` because it learns the initialized RPC
  projection before affine correction.

All pixel coordinates are crop-coordinate ``line/samp`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import sample_map_bilinear
from loss.correspondence_utils import PatchCorrespondenceGT, build_patch_correspondence_gt
from loss.feature_nce_loss import FeatureInfoNCELoss, FeatureInfoNCELossCfg
from loss.patch_match_loss import PatchInternalMatchLoss, PatchInternalMatchLossCfg


@dataclass
class EarlyPretrainWeightCfg:
    lambda_feature_nce: float = 1.0
    lambda_patch_match: float = 1.0
    lambda_match_coord: float = 1.0
    lambda_match_ce: float = 1.0
    lambda_match_cycle: float = 0.0
    lambda_projection: float = 1.0
    lambda_early_height: float = 1.0
    pixel_loss_norm: float = 16.0
    early_height_scale: float = 1000.0


class EarlyPretrainObjective:
    """Complete two-view early-pretrain objective.

    The objective combines existing patch-level feature losses with three early
    heads: global patch matching (``rpc_gt``), projection prediction
    (``rpc_init``), and cross-view height regression (``rpc_gt``).  It requires
    exactly two views and returns finite graph-connected zero losses when a
    batch has no valid correspondence.
    """

    def __init__(
        self,
        *,
        geometry_ops: Any,
        patch_matcher: torch.nn.Module,
        early_height_head: torch.nn.Module,
        feature_nce_cfg: FeatureInfoNCELossCfg | None = None,
        patch_match_cfg: PatchInternalMatchLossCfg | None = None,
        weights: EarlyPretrainWeightCfg | None = None,
    ) -> None:
        self.geometry_ops = geometry_ops
        self.feature_nce_loss = FeatureInfoNCELoss(geometry_ops=geometry_ops, cfg=feature_nce_cfg or FeatureInfoNCELossCfg())
        self.patch_match_loss = PatchInternalMatchLoss(
            geometry_ops=geometry_ops,
            patch_matcher=patch_matcher,
            cfg=patch_match_cfg or PatchInternalMatchLossCfg(),
        )
        self.early_height_head = early_height_head
        self.weights = weights or EarlyPretrainWeightCfg()

    @staticmethod
    def _require_keys(data: dict[str, Any], keys: list[str], name: str) -> None:
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"Missing required {name} keys for EarlyPretrainObjective: {missing}")

    @staticmethod
    def _zero_like_stub(stub: torch.Tensor) -> torch.Tensor:
        return stub.sum() * 0.0

    @staticmethod
    def _rpc_height_off(rpc_obj: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h = getattr(rpc_obj, "HEIGHT_OFF", 0.0)
        if torch.is_tensor(h):
            return h.to(device=device, dtype=dtype).reshape(()).clone()
        return torch.tensor(float(h), device=device, dtype=dtype)

    def _corr(self, outputs: dict[str, Any], batch: dict[str, Any], src: int, tgt: int, rpc_key: str, require_target_patch_valid: bool) -> PatchCorrespondenceGT:
        return build_patch_correspondence_gt(
            geometry_ops=self.geometry_ops,
            batch=batch,
            patch_centers=outputs["patch_centers"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_grid_hw=outputs["patch_grid_hw"],
            patch_padded_hw=outputs.get("patch_padded_hw", None),
            src_view_idx=src,
            tgt_view_idx=tgt,
            rpc_key=rpc_key,
            require_target_patch_valid=require_target_patch_valid,
        )

    def _graph_stub(self, outputs: dict[str, Any]) -> torch.Tensor:
        stub = outputs["patch_tokens_final"].sum() * 0.0 + outputs["patch_tokens_nce_proj"].sum() * 0.0
        for group in ("early_match", "early_projection"):
            for val in outputs.get(group, {}).values():
                if torch.is_tensor(val):
                    stub = stub + val.sum() * 0.0
        for k in ("early_height_dummy_0_to_1", "early_height_dummy_1_to_0"):
            if k in outputs and torch.is_tensor(outputs[k]):
                stub = stub + outputs[k].sum() * 0.0
        return stub

    def _global_match_losses(self, outputs: dict[str, Any], batch: dict[str, Any], stub: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        dtype = outputs["patch_tokens_final"].dtype
        device = outputs["patch_tokens_final"].device
        norm = max(float(self.weights.pixel_loss_norm), 1e-6)
        coord_losses: list[torch.Tensor] = []
        ce_losses: list[torch.Tensor] = []
        sq_errs: list[torch.Tensor] = []
        abs_errs: list[torch.Tensor] = []
        accs: list[torch.Tensor] = []
        counts: dict[str, int] = {}
        for src, tgt, suffix in [(0, 1, "0_to_1"), (1, 0, "1_to_0")]:
            corr = self._corr(outputs, batch, src, tgt, rpc_key="rpc_gt", require_target_patch_valid=True)
            counts[suffix] = corr.num_valid
            if corr.num_valid == 0:
                continue
            pred = outputs["early_match"][f"pred_{suffix}"][corr.batch_indices, corr.src_patch_indices]
            logits = outputs["early_match"][f"logits_{suffix}"][corr.batch_indices, corr.src_patch_indices]
            finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(corr.tgt_pixels).all(dim=-1) & torch.isfinite(logits).all(dim=-1)
            if not bool(finite.any()):
                continue
            pred = pred[finite]
            tgt_pix = corr.tgt_pixels[finite].to(device=device, dtype=dtype)
            logits = logits[finite]
            tgt_cls = corr.tgt_patch_indices[finite].to(device=device, dtype=torch.long)
            coord_losses.append(F.smooth_l1_loss(pred / norm, tgt_pix / norm))
            ce_losses.append(F.cross_entropy(logits.float(), tgt_cls))
            err = torch.linalg.vector_norm(pred.detach() - tgt_pix.detach(), dim=-1)
            abs_errs.append(err)
            sq_errs.append(err.square())
            accs.append((logits.argmax(dim=-1) == tgt_cls).to(dtype=dtype))
        loss_coord = torch.stack(coord_losses).mean() if coord_losses else stub
        loss_ce = torch.stack(ce_losses).mean().to(dtype=dtype) if ce_losses else stub
        if abs_errs:
            abs_all = torch.cat(abs_errs)
            sq_all = torch.cat(sq_errs)
            acc_all = torch.cat(accs)
            valid = torch.tensor(float(abs_all.numel()), device=device, dtype=dtype)
            px_mean = abs_all.mean().to(dtype=dtype)
            px_rmse = torch.sqrt(sq_all.mean().clamp_min(0.0)).to(dtype=dtype)
            acc = acc_all.mean().to(dtype=dtype)
        else:
            valid = torch.zeros((), device=device, dtype=dtype)
            px_mean = torch.zeros((), device=device, dtype=dtype)
            px_rmse = torch.zeros((), device=device, dtype=dtype)
            acc = torch.zeros((), device=device, dtype=dtype)
        return {
            "loss_match_coord": loss_coord,
            "loss_match_ce": loss_ce,
            "loss_match_cycle": stub,
            "metric_match_valid_pairs": valid,
            "metric_match_px_mean": px_mean,
            "metric_match_px_rmse": px_rmse,
            "metric_match_acc_patch_top1": acc,
        }, {"early_match_valid_by_direction": counts}

    def _projection_loss(self, outputs: dict[str, Any], batch: dict[str, Any], stub: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if "rpc_init" not in batch:
            raise KeyError("Early projection prediction requires batch['rpc_init'] for rpc_init GT.")
        dtype = outputs["patch_tokens_final"].dtype
        device = outputs["patch_tokens_final"].device
        norm = max(float(self.weights.pixel_loss_norm), 1e-6)
        losses: list[torch.Tensor] = []
        abs_errs: list[torch.Tensor] = []
        sq_errs: list[torch.Tensor] = []
        counts: dict[str, int] = {}
        for src, tgt, suffix in [(0, 1, "0_to_1"), (1, 0, "1_to_0")]:
            corr = self._corr(outputs, batch, src, tgt, rpc_key="rpc_init", require_target_patch_valid=False)
            counts[suffix] = corr.num_valid
            if corr.num_valid == 0:
                continue
            pred = outputs["early_projection"][f"pred_{suffix}"][corr.batch_indices, corr.src_patch_indices]
            finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(corr.tgt_pixels).all(dim=-1)
            if not bool(finite.any()):
                continue
            pred = pred[finite]
            tgt_pix = corr.tgt_pixels[finite].to(device=device, dtype=dtype)
            losses.append(F.smooth_l1_loss(pred / norm, tgt_pix / norm))
            err = torch.linalg.vector_norm(pred.detach() - tgt_pix.detach(), dim=-1)
            abs_errs.append(err)
            sq_errs.append(err.square())
        loss = torch.stack(losses).mean() if losses else stub
        if abs_errs:
            abs_all = torch.cat(abs_errs)
            sq_all = torch.cat(sq_errs)
            valid = torch.tensor(float(abs_all.numel()), device=device, dtype=dtype)
            px_mean = abs_all.mean().to(dtype=dtype)
            px_rmse = torch.sqrt(sq_all.mean().clamp_min(0.0)).to(dtype=dtype)
        else:
            valid = torch.zeros((), device=device, dtype=dtype)
            px_mean = torch.zeros((), device=device, dtype=dtype)
            px_rmse = torch.zeros((), device=device, dtype=dtype)
        return {"loss_projection": loss, "metric_projection_valid_pairs": valid, "metric_projection_px_mean": px_mean, "metric_projection_px_rmse": px_rmse}, {"early_projection_valid_by_direction": counts}

    def _sample_target_tokens(self, outputs: dict[str, Any], corr: PatchCorrespondenceGT, tgt_view: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = outputs["patch_tokens_final"]
        b, _, _, c = tokens.shape
        gh, gw = int(outputs["patch_grid_hw"][0]), int(outputs["patch_grid_hw"][1])
        hp, wp = outputs.get("patch_padded_hw", (int(outputs.get("image_hw", (gh, gw))[0]), int(outputs.get("image_hw", (gh, gw))[1])))
        patch_h = float(int(hp)) / float(max(gh, 1))
        patch_w = float(int(wp)) / float(max(gw, 1))
        feat_map = tokens[:, tgt_view].view(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()
        sampled_parts: list[torch.Tensor] = []
        keep_parts: list[torch.Tensor] = []
        for bi in corr.batch_indices.unique(sorted=True).tolist():
            sel = corr.batch_indices == int(bi)
            pts = corr.tgt_pixels[sel].view(1, -1, 2).to(device=tokens.device, dtype=tokens.dtype).clone()
            pts[..., 0] = (pts[..., 0] + 0.5) / patch_h - 0.5
            pts[..., 1] = (pts[..., 1] + 0.5) / patch_w - 0.5
            smp, in_bounds = sample_map_bilinear(feat_map[int(bi) : int(bi) + 1], pts)
            sampled_parts.append(smp[0].transpose(0, 1))
            keep_parts.append(in_bounds[0])
        if not sampled_parts:
            empty_tok = torch.empty((0, c), device=tokens.device, dtype=tokens.dtype)
            empty_keep = torch.empty((0,), device=tokens.device, dtype=torch.bool)
            return empty_tok, empty_keep
        return torch.cat(sampled_parts, dim=0), torch.cat(keep_parts, dim=0)

    def _early_height_loss(self, outputs: dict[str, Any], batch: dict[str, Any], stub: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        dtype = outputs["patch_tokens_final"].dtype
        device = outputs["patch_tokens_final"].device
        scale = max(float(self.weights.early_height_scale), 1e-6)
        losses: list[torch.Tensor] = []
        abs_m: list[torch.Tensor] = []
        sq_m: list[torch.Tensor] = []
        counts: dict[str, int] = {}
        for src, tgt, suffix in [(0, 1, "0_to_1"), (1, 0, "1_to_0")]:
            corr = self._corr(outputs, batch, src, tgt, rpc_key="rpc_gt", require_target_patch_valid=False)
            counts[suffix] = corr.num_valid
            if corr.num_valid == 0:
                continue
            tgt_tok, in_sample = self._sample_target_tokens(outputs, corr, tgt_view=tgt)
            if tgt_tok.numel() == 0:
                continue
            src_tok = outputs["patch_tokens_final"][corr.batch_indices, corr.src_view_indices, corr.src_patch_indices]
            src_tok = src_tok[in_sample]
            tgt_tok = tgt_tok[in_sample]
            heights = corr.heights[in_sample].to(device=device, dtype=dtype)
            batch_idx = corr.batch_indices[in_sample]
            h_offsets = torch.stack(
                [self._rpc_height_off(batch["rpc_gt"][int(bi)][src], device=device, dtype=dtype) for bi in batch_idx.tolist()], dim=0
            )
            finite = torch.isfinite(src_tok).all(dim=-1) & torch.isfinite(tgt_tok).all(dim=-1) & torch.isfinite(heights) & torch.isfinite(h_offsets)
            if not bool(finite.any()):
                continue
            x_pred = self.early_height_head(src_tok[finite], tgt_tok[finite])
            target_x = (heights[finite] - h_offsets[finite]) / scale
            losses.append(F.smooth_l1_loss(x_pred, target_x))
            pred_h = x_pred.detach() * scale + h_offsets[finite].detach()
            err = (pred_h - heights[finite].detach()).abs()
            abs_m.append(err)
            sq_m.append(err.square())
        loss = torch.stack(losses).mean() if losses else stub
        if abs_m:
            abs_all = torch.cat(abs_m)
            sq_all = torch.cat(sq_m)
            valid = torch.tensor(float(abs_all.numel()), device=device, dtype=dtype)
            mae = abs_all.mean().to(dtype=dtype)
            rmse = torch.sqrt(sq_all.mean().clamp_min(0.0)).to(dtype=dtype)
        else:
            valid = torch.zeros((), device=device, dtype=dtype)
            mae = torch.zeros((), device=device, dtype=dtype)
            rmse = torch.zeros((), device=device, dtype=dtype)
        return {"loss_early_height": loss, "metric_early_height_valid_pairs": valid, "metric_early_height_mae_m": mae, "metric_early_height_rmse_m": rmse}, {"early_height_valid_by_direction": counts}

    def __call__(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        global_step: int,
        epoch: int = 0,
        render_outputs: dict[str, dict[str, torch.Tensor | None]] | None = None,
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        del global_step, epoch, render_outputs, mode
        self._require_keys(
            outputs,
            [
                "patch_tokens_final",
                "patch_tokens_nce_proj",
                "patch_tokens_match",
                "view_tokens_final",
                "patch_valid_mask",
                "patch_centers",
                "patch_grid_hw",
                "patch_padded_hw",
                "early_match",
                "early_projection",
            ],
            "outputs",
        )
        self._require_keys(batch, ["height_gt", "height_valid_mask", "rpc_gt", "rpc_init", "images"], "batch")
        view_num = int(batch["images"].shape[1])
        if view_num != 2:
            raise ValueError(f"EarlyPretrainObjective requires exactly 2 views per batch, got V={view_num}")
        forbidden = ["affine_pred", "rpc_corrected", "height_abs", "gaussian_opacity"]
        present_forbidden = [k for k in forbidden if k in outputs]
        if present_forbidden:
            raise KeyError(f"Early forward outputs must not contain dense geometry/render keys: {present_forbidden}")

        stub = self._graph_stub(outputs)
        l_nce, p_nce, aux_nce = self.feature_nce_loss(
            patch_tokens_proj=outputs["patch_tokens_nce_proj"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_centers=outputs["patch_centers"],
            patch_grid_hw=outputs["patch_grid_hw"],
            patch_padded_hw=outputs["patch_padded_hw"],
            batch=batch,
        )
        l_patch, p_patch, aux_patch = self.patch_match_loss(
            patch_tokens_match=outputs["patch_tokens_match"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_centers=outputs["patch_centers"],
            patch_grid_hw=outputs["patch_grid_hw"],
            batch=batch,
        )
        match_dict, aux_match = self._global_match_losses(outputs, batch, stub)
        proj_dict, aux_proj = self._projection_loss(outputs, batch, stub)
        height_dict, aux_height = self._early_height_loss(outputs, batch, stub)

        w = self.weights
        total = (
            float(w.lambda_feature_nce) * l_nce
            + float(w.lambda_patch_match) * l_patch
            + float(w.lambda_match_coord) * match_dict["loss_match_coord"]
            + float(w.lambda_match_ce) * match_dict["loss_match_ce"]
            + float(w.lambda_match_cycle) * match_dict["loss_match_cycle"]
            + float(w.lambda_projection) * proj_dict["loss_projection"]
            + float(w.lambda_early_height) * height_dict["loss_early_height"]
            + stub
        )
        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        scalar: dict[str, Any] = {
            "loss_total": total,
            "loss_feature_nce": l_nce,
            "loss_patch_match": l_patch,
            **match_dict,
            **proj_dict,
            **height_dict,
            "metric_feature_nce_valid_pairs": p_nce.get("feature_nce_valid_pairs", zero),
            "metric_feature_nce_acc_top1": p_nce.get("feature_nce_acc_top1", zero),
            "metric_patch_match_valid_pairs": p_patch.get("patch_match_valid_pairs", zero),
            "metric_patch_match_acc_top1_1px": p_patch.get("patch_match_acc_top1_1px", zero),
            "metric_patch_match_l1_px": p_patch.get("patch_match_l1_px", zero),
            "weight_feature_nce": float(w.lambda_feature_nce),
            "weight_patch_match": float(w.lambda_patch_match),
            "weight_match_coord": float(w.lambda_match_coord),
            "weight_match_ce": float(w.lambda_match_ce),
            "weight_match_cycle": float(w.lambda_match_cycle),
            "weight_projection": float(w.lambda_projection),
            "weight_early_height": float(w.lambda_early_height),
        }
        aux: dict[str, Any] = {}
        aux.update(aux_nce)
        aux.update(aux_patch)
        aux.update(aux_match)
        aux.update(aux_proj)
        aux.update(aux_height)
        return total, scalar, aux

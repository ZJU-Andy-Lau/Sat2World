"""loss.pretrain_objective

几何预训练目标：仅优化几何主线（仿射 / 高程 / 点云），
不引入 3DGS 渲染与高斯属性相关训练项。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from loss.affine_loss import (
    AffineGridLoss,
    AffineGridLossCfg,
    AffineLinearRegularization,
    AffinePairwiseGeometryLoss,
    AffinePairwiseGeometryLossCfg,
)
from loss.feature_nce_loss import FeatureInfoNCELoss, FeatureInfoNCELossCfg
from loss.height_loss import HeightHuberLoss
from loss.normal_loss import PointNormalLoss, PointNormalLossCfg
from loss.patch_match_loss import PatchInternalMatchLoss, PatchInternalMatchLossCfg
from loss.point_pair_loss import HeightReprojectionLoss, PointPairwiseConsistencyLoss, PointPairwiseLossCfg, PointReprojectionLoss
from loss.point_loss import PointMapLoss
from loss.z_space_loss import masked_z_huber_loss


@dataclass
class GeometryPretrainWeightCfg:
    """几何预训练损失权重。"""

    lambda_affine_grid: float = 1.0
    lambda_affine_pair: float = 1.0
    lambda_affine_reg: float = 0.1
    lambda_height: float = 1.0
    lambda_height_anchor: float = 0.5
    lambda_point: float = 1.0
    lambda_point_xy: float | None = None
    lambda_point_z: float | None = None
    lambda_height_meter_aux: float = 1.0e-3
    lambda_height_anchor_meter_aux: float = 1.0e-3
    lambda_point_z_meter_aux: float = 1.0e-3
    lambda_point_reproj: float = 0.2
    lambda_height_reproj: float = 0.2
    lambda_point_pair: float = 0.1
    lambda_normal_height: float = 0.2
    lambda_normal_point: float = 0.2
    lambda_feature_nce: float = 0.1
    lambda_patch_match: float = 0.5
    abs_keep_steps: int = 5000


class GeometryPretrainObjective:
    """几何预训练目标组合器。

    仅包含：
    - AffineGridLoss
    - AffinePairwiseGeometryLoss
    - AffineLinearRegularization
    - HeightHuberLoss
    - PointMapLoss
    """

    def __init__(
        self,
        geometry_ops: Any,
        *,
        affine_grid_cfg: AffineGridLossCfg | None = None,
        affine_pair_cfg: AffinePairwiseGeometryLossCfg | None = None,
        point_pair_cfg: PointPairwiseLossCfg | None = None,
        feature_nce_cfg: FeatureInfoNCELossCfg | None = None,
        patch_match_cfg: PatchInternalMatchLossCfg | None = None,
        patch_matcher: torch.nn.Module | None = None,
        normal_cfg: PointNormalLossCfg | None = None,
        height_beta: float = 1.0,
        point_beta: float = 1.0,
        height_z_beta_meter: float | None = None,
        height_anchor_z_beta_meter: float = 5.0,
        point_z_beta_meter: float | None = None,
        weights: GeometryPretrainWeightCfg | None = None,
    ) -> None:
        self.geometry_ops = geometry_ops
        self.affine_grid = AffineGridLoss(affine_grid_cfg)
        self.affine_pair = AffinePairwiseGeometryLoss(geometry_ops, affine_pair_cfg)
        self.affine_reg = AffineLinearRegularization()
        self.height_loss = HeightHuberLoss(beta=height_beta)
        self.height_anchor_loss = torch.nn.SmoothL1Loss(reduction="mean")
        self.point_loss = PointMapLoss(geometry_ops=geometry_ops, beta=point_beta)
        self.point_reproj_loss = PointReprojectionLoss(geometry_ops=geometry_ops, cfg=point_pair_cfg or PointPairwiseLossCfg())
        self.point_pair_loss = PointPairwiseConsistencyLoss(geometry_ops=geometry_ops, cfg=point_pair_cfg or PointPairwiseLossCfg())
        self.height_reproj_loss = HeightReprojectionLoss(geometry_ops=geometry_ops, cfg=point_pair_cfg or PointPairwiseLossCfg())
        self.feature_nce_loss = FeatureInfoNCELoss(geometry_ops=geometry_ops, cfg=feature_nce_cfg or FeatureInfoNCELossCfg())
        if patch_matcher is None:
            raise ValueError("patch_matcher must be provided for PatchInternalMatchLoss.")
        self.patch_match_loss = PatchInternalMatchLoss(
            geometry_ops=geometry_ops,
            patch_matcher=patch_matcher,
            cfg=patch_match_cfg or PatchInternalMatchLossCfg(),
        )
        self.normal_loss = PointNormalLoss(geometry_ops=geometry_ops, cfg=normal_cfg or PointNormalLossCfg())
        self.weights = weights or GeometryPretrainWeightCfg()
        self.height_beta_meter = float(height_beta)
        self.point_beta_meter = float(point_beta)
        self.height_z_beta_meter = float(height_beta if height_z_beta_meter is None else height_z_beta_meter)
        self.height_anchor_z_beta_meter = float(height_anchor_z_beta_meter)
        self.point_z_beta_meter = float(point_beta if point_z_beta_meter is None else point_z_beta_meter)

    def _require_keys(self, data: dict[str, Any], keys: list[str], name: str) -> None:
        miss = [k for k in keys if k not in data]
        if miss:
            raise KeyError(f"Missing required {name} keys: {miss}")

    def _to_float_scalar_dict(self, scalar_dict: dict[str, Any]) -> dict[str, float | torch.Tensor]:
        out: dict[str, float | torch.Tensor] = {}
        for k, v in scalar_dict.items():
            if torch.is_tensor(v) and v.ndim == 0:
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = v
        return out

    def _build_nan_probe(self, scalar_dict: dict[str, Any]) -> dict[str, Any]:
        """构建 NaN/Inf 探针摘要。"""
        nonfinite_losses: list[str] = []
        nonfinite_metrics: list[str] = []
        loss_snapshot: dict[str, float | str] = {}

        def _is_finite_value(x: Any) -> bool:
            if torch.is_tensor(x):
                if x.numel() == 0:
                    return True
                return bool(torch.isfinite(x).all().item())
            if isinstance(x, (int, float)):
                return math.isfinite(float(x))
            return True

        def _to_scalar_repr(x: Any) -> float | str:
            if torch.is_tensor(x):
                if x.numel() == 0:
                    return "empty"
                if x.ndim == 0:
                    val = float(x.detach().cpu().item())
                    if math.isnan(val):
                        return "nan"
                    if math.isinf(val):
                        return "inf" if val > 0 else "-inf"
                    return val
                return f"tensor(shape={tuple(x.shape)})"
            if isinstance(x, (int, float)):
                val = float(x)
                if math.isnan(val):
                    return "nan"
                if math.isinf(val):
                    return "inf" if val > 0 else "-inf"
                return val
            return str(type(x).__name__)

        for k, v in scalar_dict.items():
            if not (k.startswith("loss_") or k.startswith("metric_")):
                continue
            finite = _is_finite_value(v)
            if k.startswith("loss_"):
                loss_snapshot[k] = _to_scalar_repr(v)
                if not finite:
                    nonfinite_losses.append(k)
            elif not finite:
                nonfinite_metrics.append(k)

        return {
            "nan_probe_nonfinite_losses": nonfinite_losses,
            "nan_probe_nonfinite_metrics": nonfinite_metrics,
            "nan_probe_first_bad_loss": nonfinite_losses[0] if len(nonfinite_losses) > 0 else "",
            "nan_probe_total_is_finite": len(nonfinite_losses) == 0,
            "nan_probe_loss_snapshot": loss_snapshot,
        }

    def _replace_ref_affine_with_identity(self, affine_pred: torch.Tensor, ref_view_idx: torch.Tensor | None) -> torch.Tensor:
        b, v = affine_pred.shape[:2]
        out = affine_pred.clone()
        eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=out.dtype, device=out.device)
        if ref_view_idx is None:
            out[:, 0] = eye
            return out
        ref = ref_view_idx.long().view(-1).to(device=out.device)
        if ref.numel() == 1:
            ref = ref.expand(b)
        ref = ref.clamp(0, v - 1)
        out[torch.arange(b, device=out.device), ref] = eye
        return out

    def forward(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        global_step: int,
        epoch: int = 0,
        render_outputs: dict[str, dict[str, torch.Tensor | None]] | None = None,
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor], dict[str, Any]]:
        del epoch, render_outputs, mode

        self._require_keys(
            outputs,
            [
                "affine_pred",
                "height_abs",
                "height_anchor",
                "height_local_z",
                "height_anchor_z",
                "height_ref_anchor",
                "point_abs",
                "point_anchor",
                "point_z_local_z",
                "height_local_scale",
                "height_anchor_scale",
                "height_z_max",
                "rpc_corrected",
                "patch_tokens_nce_proj",
                "patch_valid_mask",
                "patch_centers",
                "patch_grid_hw",
                "patch_padded_hw",
                "patch_tokens_match",
            ],
            "outputs",
        )
        self._require_keys(batch, ["height_gt", "height_valid_mask", "height_anchor_gt", "affine_gt_forward", "rpc_gt", "images"], "batch")

        affine_pred = outputs["affine_pred"]
        ref_idx = batch.get("ref_view_idx", None)
        affine_pred_direct = affine_pred
        affine_pred_relative = self._replace_ref_affine_with_identity(affine_pred, ref_idx)
        outputs_for_affine_pair = dict(outputs)
        outputs_for_affine_pair["affine_pred"] = affine_pred_relative
        if "rpc_init" in batch:
            outputs_for_affine_pair["rpc_corrected"] = self.geometry_ops.apply_affine_correction_batch(batch["rpc_init"], affine_pred_relative)
        image_hw = (int(outputs["height_abs"].shape[-2]), int(outputs["height_abs"].shape[-1]))

        l_aff_grid, p_aff_grid = self.affine_grid(
            affine_pred=affine_pred_direct,
            affine_gt_forward=batch["affine_gt_forward"].to(device=affine_pred.device, dtype=affine_pred.dtype),
            image_hw=image_hw,
            ref_view_idx=ref_idx,
        )
        l_aff_pair, p_aff_pair, aux_pair = self.affine_pair(outputs_for_affine_pair, batch)
        l_aff_reg, p_aff_reg = self.affine_reg(affine_pred_direct, ref_view_idx=ref_idx)

        l_h, p_h = self.height_loss(
            outputs["height_abs"],
            batch["height_gt"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype),
            batch["height_valid_mask"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype),
        )
        l_h_anchor = self.height_anchor_loss(
            outputs["height_anchor"],
            batch["height_anchor_gt"].to(device=outputs["height_anchor"].device, dtype=outputs["height_anchor"].dtype),
        )
        p_h_anchor_mae = (outputs["height_anchor"] - batch["height_anchor_gt"].to(device=outputs["height_anchor"].device, dtype=outputs["height_anchor"].dtype)).abs().mean().detach()
        z_max_val = float(outputs.get("height_z_max", torch.tensor(0.0, device=outputs["height_abs"].device)).detach().item())
        z_thr = z_max_val * 0.98 if z_max_val > 0 else 0.0
        h_local_z = outputs.get("height_local_z", None)
        pz_local_z = outputs.get("point_z_local_z", None)
        h_local_boundary = (
            (h_local_z.abs() >= z_thr).to(h_local_z.dtype).mean().detach()
            if h_local_z is not None and z_thr > 0
            else torch.zeros((), device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype)
        )
        pz_local_boundary = (
            (pz_local_z.abs() >= z_thr).to(pz_local_z.dtype).mean().detach()
            if pz_local_z is not None and z_thr > 0
            else torch.zeros((), device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype)
        )

        l_p, p_p, aux_point = self.point_loss(outputs["point_abs"], outputs["point_anchor"], batch, return_aux=True)
        l_p_xy = aux_point.get("loss_point_xy_meter", l_p)
        l_p_z_meter = aux_point.get("loss_point_z_meter", torch.zeros_like(l_p))
        gt_point_z_meter = aux_point.pop("gt_point_z_meter", None)
        aux_point.pop("gt_point_map_metric", None)
        aux_point.pop("gt_point_map", None)
        if gt_point_z_meter is None:
            raise KeyError("PointMapLoss(return_aux=True) must provide gt_point_z_meter for z-space point supervision.")

        height_valid_mask = batch["height_valid_mask"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype)
        height_anchor_detached = outputs["height_anchor"].detach()
        height_z_max = outputs["height_z_max"]
        height_local_scale = outputs["height_local_scale"]
        height_anchor_scale = outputs["height_anchor_scale"]

        l_h_z = masked_z_huber_loss(
            outputs["height_local_z"],
            batch["height_gt"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype) - height_anchor_detached.view(-1, 1, 1, 1, 1),
            mask=height_valid_mask,
            beta_meter=self.height_z_beta_meter,
            scale=height_local_scale,
            z_max=height_z_max,
        )
        l_h_anchor_z = masked_z_huber_loss(
            outputs["height_anchor_z"].view(-1, 1, 1, 1, 1),
            batch["height_anchor_gt"].to(device=outputs["height_anchor"].device, dtype=outputs["height_anchor"].dtype).view(-1, 1, 1, 1, 1)
            - outputs["height_ref_anchor"].to(device=outputs["height_anchor"].device, dtype=outputs["height_anchor"].dtype).view(-1, 1, 1, 1, 1),
            mask=None,
            beta_meter=self.height_anchor_z_beta_meter,
            scale=height_anchor_scale,
            z_max=height_z_max,
        )
        l_p_z_z = masked_z_huber_loss(
            outputs["point_z_local_z"],
            gt_point_z_meter - height_anchor_detached.view(-1, 1, 1, 1, 1),
            mask=height_valid_mask,
            beta_meter=self.point_z_beta_meter,
            scale=height_local_scale,
            z_max=height_z_max,
        )
        del gt_point_z_meter
        l_preproj, p_preproj, aux_preproj = self.point_reproj_loss(outputs["point_abs"], batch)
        l_ppair, p_ppair, aux_ppair = self.point_pair_loss(outputs["point_abs"], batch)
        l_nh, l_np, p_norm, aux_norm = self.normal_loss(
            height_abs=outputs["height_abs"],
            point_abs=outputs["point_abs"],
            batch=batch,
        )
        l_hreproj, p_hreproj, aux_hreproj = self.height_reproj_loss(outputs["height_abs"], batch)
        l_nce, p_nce, aux_nce = self.feature_nce_loss(
            patch_tokens_proj=outputs["patch_tokens_nce_proj"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_centers=outputs["patch_centers"],
            patch_grid_hw=outputs["patch_grid_hw"],
            patch_padded_hw=outputs["patch_padded_hw"],
            batch=batch,
        )
        l_patch_match, p_patch_match, aux_patch_match = self.patch_match_loss(
            patch_tokens_match=outputs["patch_tokens_match"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_centers=outputs["patch_centers"],
            patch_grid_hw=outputs["patch_grid_hw"],
            batch=batch,
        )

        w = self.weights
        abs_mul = 1.0 if int(global_step) < int(w.abs_keep_steps) else 0.1
        w_h_abs = float(w.lambda_height) * abs_mul
        w_p_xy_base = float(w.lambda_point if w.lambda_point_xy is None else w.lambda_point_xy)
        w_p_z_base = float(w.lambda_point if w.lambda_point_z is None else w.lambda_point_z)
        w_p_xy = w_p_xy_base * abs_mul
        w_p_z = w_p_z_base * abs_mul
        w_h_meter_aux = float(w.lambda_height_meter_aux)
        w_h_anchor_meter_aux = float(w.lambda_height_anchor_meter_aux)
        w_p_z_meter_aux = float(w.lambda_point_z_meter_aux)

        l_h_optim = l_h_z + w_h_meter_aux * l_h
        l_h_anchor_optim = l_h_anchor_z + w_h_anchor_meter_aux * l_h_anchor
        l_p_xy_optim = l_p_xy
        l_p_z_optim = l_p_z_z + w_p_z_meter_aux * l_p_z_meter
        total = (
            w.lambda_affine_grid * l_aff_grid
            + w.lambda_affine_pair * l_aff_pair
            + w.lambda_affine_reg * l_aff_reg
            + w_h_abs * l_h_optim
            + float(w.lambda_height_anchor) * l_h_anchor_optim
            + w_p_xy * l_p_xy_optim
            + w_p_z * l_p_z_optim
            + w.lambda_point_reproj * l_preproj
            + w.lambda_height_reproj * l_hreproj
            + w.lambda_point_pair * l_ppair
            + w.lambda_normal_height * l_nh
            + w.lambda_normal_point * l_np
            + w.lambda_feature_nce * l_nce
            + w.lambda_patch_match * l_patch_match
        )

        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        scalar_dict: dict[str, Any] = {
            "loss_total": total,
            "loss_affine_grid": l_aff_grid,
            "loss_affine_pair": l_aff_pair,
            "loss_affine_reg": l_aff_reg,
            "loss_height": l_h,
            "loss_height_anchor": l_h_anchor,
            "loss_height_z": l_h_z,
            "loss_height_anchor_z": l_h_anchor_z,
            "loss_height_optim": l_h_optim,
            "loss_height_anchor_optim": l_h_anchor_optim,
            "loss_height_rel": zero,
            "loss_point": l_p,
            "loss_point_xy": l_p_xy,
            "loss_point_z": l_p_z_meter,
            "loss_point_z_z": l_p_z_z,
            "loss_point_z_optim": l_p_z_optim,
            "loss_point_reproj": l_preproj,
            "loss_height_reproj": l_hreproj,
            "loss_point_pair": l_ppair,
            "loss_normal_height": l_nh,
            "loss_normal_point": l_np,
            "loss_feature_nce": l_nce,
            "loss_patch_match": l_patch_match,
            "metric_affine_grid_error_px_mean": p_aff_grid.get("affine_grid_error_px_mean", zero),
            "metric_affine_grid_error_px_rmse": p_aff_grid.get("affine_grid_error_px_rmse", zero),
            "metric_affine_grid_ref_error_px_mean": p_aff_grid.get("affine_grid_ref_error_px_mean", zero),
            "metric_affine_grid_ref_error_px_rmse": p_aff_grid.get("affine_grid_ref_error_px_rmse", zero),
            "metric_affine_grid_nonref_error_px_mean": p_aff_grid.get("affine_grid_nonref_error_px_mean", zero),
            "metric_affine_grid_nonref_error_px_rmse": p_aff_grid.get("affine_grid_nonref_error_px_rmse", zero),
            "metric_affine_pair_error_px_mean": p_aff_pair.get("affine_pair_error_px_mean", zero),
            "probe_affine_linear_frob_mean": p_aff_reg.get("affine_linear_frob_mean", zero),
            "probe_affine_linear_ref_frob_mean": p_aff_reg.get("affine_linear_ref_frob_mean", zero),
            "probe_affine_linear_nonref_frob_mean": p_aff_reg.get("affine_linear_nonref_frob_mean", zero),
            "metric_height_rmse": p_h.get("height_rmse", zero),
            "metric_height_mae": p_h.get("height_mae", zero),
            "metric_height_bias": p_h.get("height_bias", zero),
            "metric_height_anchor_mae": p_h_anchor_mae,
            "metric_height_rel_consistency": zero,
            "metric_height_rel_cycle_px": zero,
            "metric_height_rel_cycle_px_rmse": zero,
            "metric_height_rel_pairs_used": zero,
            "metric_height_reproj_px_mean": p_hreproj.get("height_reproj_px_mean", zero),
            "metric_height_reproj_num_pairs_used": p_hreproj.get("height_reproj_num_pairs_used", zero),
            "metric_point_xyz_rmse": p_p.get("point_xyz_rmse", zero),
            "metric_point_xy_rmse": p_p.get("point_xy_rmse", zero),
            "metric_point_z_rmse": p_p.get("point_z_rmse", zero),
            "metric_point_anchor_displacement_mean": p_p.get("point_anchor_displacement_mean", zero),
            "metric_point_reproj_px_mean": p_preproj.get("point_reproj_px_mean", zero),
            "metric_point_reproj_num_pairs_used": p_preproj.get("point_reproj_num_pairs_used", zero),
            "metric_point_pair_dist_mean": p_ppair.get("point_pair_dist_mean", zero),
            "metric_point_pair_num_pairs_used": p_ppair.get("point_pair_num_pairs_used", zero),
            "metric_normal_h_cos_mean": p_norm.get("normal_h_cos_mean", zero),
            "metric_normal_h_ang_deg_mean": p_norm.get("normal_h_ang_deg_mean", zero),
            "metric_normal_p_cos_mean": p_norm.get("normal_p_cos_mean", zero),
            "metric_normal_p_ang_deg_mean": p_norm.get("normal_p_ang_deg_mean", zero),
            "metric_normal_valid_ratio": p_norm.get("normal_valid_ratio", zero),
            "metric_patch_match_valid_pairs": p_patch_match.get("patch_match_valid_pairs", zero),
            "metric_patch_match_acc_top1_1px": p_patch_match.get("patch_match_acc_top1_1px", zero),
            "metric_patch_match_l1_px": p_patch_match.get("patch_match_l1_px", zero),
            "probe_height_local_z_boundary_ratio": h_local_boundary,
            "probe_point_z_local_z_boundary_ratio": pz_local_boundary,
            "weight_affine_grid": float(w.lambda_affine_grid),
            "weight_affine_pair": float(w.lambda_affine_pair),
            "weight_affine_reg": float(w.lambda_affine_reg),
            "weight_height": float(w_h_abs),
            "weight_height_anchor": float(w.lambda_height_anchor),
            "weight_height_meter_aux": float(w_h_meter_aux),
            "weight_height_anchor_meter_aux": float(w_h_anchor_meter_aux),
            "weight_height_rel": 0.0,
            "weight_point": float(float(w.lambda_point) * abs_mul),
            "weight_point_xy": float(w_p_xy),
            "weight_point_z": float(w_p_z),
            "weight_point_z_meter_aux": float(w_p_z_meter_aux),
            "weight_point_reproj": float(w.lambda_point_reproj),
            "weight_height_reproj": float(w.lambda_height_reproj),
            "weight_point_pair": float(w.lambda_point_pair),
            "weight_normal_height": float(w.lambda_normal_height),
            "weight_normal_point": float(w.lambda_normal_point),
            "weight_feature_nce": float(w.lambda_feature_nce),
            "weight_patch_match": float(w.lambda_patch_match),
            # 显式输出关闭项，便于日志检查
            "weight_center": 0.0,
            "weight_opacity_reg": 0.0,
            "weight_scale_reg": 0.0,
            "weight_render_rpc": 0.0,
            "weight_render_point": 0.0,
            "loss_center_consistency": zero,
            "loss_gaussian_opacity_reg": zero,
            "loss_gaussian_scale_reg": zero,
            "loss_render_rpc": zero,
            "loss_render_point": zero,
            "loss_ssim": zero,
        }

        aux_dict: dict[str, Any] = {
            "num_pairs_used": aux_pair.get("num_pairs_used", 0),
            "render_rpc_num_targets": 0.0,
            "render_point_num_targets": 0.0,
        }
        aux_dict.update(aux_point)
        aux_dict.update(aux_preproj)
        aux_dict.update(aux_ppair)
        aux_dict.update(aux_norm)
        aux_dict.update(aux_hreproj)
        aux_dict.update(aux_nce)
        aux_dict.update(aux_patch_match)
        aux_dict.update(self._build_nan_probe(scalar_dict))
        scalar_dict["metric_feature_nce_valid_pairs"] = p_nce.get("feature_nce_valid_pairs", zero)
        scalar_dict["metric_feature_nce_acc_top1"] = p_nce.get("feature_nce_acc_top1", zero)

        return total, self._to_float_scalar_dict(scalar_dict), aux_dict

    __call__ = forward

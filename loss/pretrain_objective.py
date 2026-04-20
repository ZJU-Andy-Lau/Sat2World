"""loss.pretrain_objective

几何预训练目标：仅优化几何主线（仿射 / 高程 / 点云），
不引入 3DGS 渲染与高斯属性相关训练项。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from loss.affine_loss import (
    AffineGridLoss,
    AffineGridLossCfg,
    AffineLinearRegularization,
    AffinePairwiseGeometryLoss,
    AffinePairwiseGeometryLossCfg,
    RefAffineIdentityLoss,
)
from loss.feature_nce_loss import FeatureInfoNCELoss, FeatureInfoNCELossCfg
from loss.height_loss import HeightHuberLoss
from loss.normal_loss import PointNormalLoss, PointNormalLossCfg
from loss.patch_match_loss import PatchInternalMatchLoss, PatchInternalMatchLossCfg
from loss.point_pair_loss import HeightReprojectionLoss, PointPairwiseConsistencyLoss, PointPairwiseLossCfg, PointReprojectionLoss
from loss.point_loss import PointMapLoss


@dataclass
class GeometryPretrainWeightCfg:
    """几何预训练损失权重。"""

    lambda_affine_grid: float = 1.0
    lambda_affine_pair: float = 1.0
    lambda_affine_reg: float = 0.1
    lambda_affine_ref: float = 0.1
    lambda_height: float = 1.0
    lambda_point: float = 1.0
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
    - RefAffineIdentityLoss
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
        weights: GeometryPretrainWeightCfg | None = None,
    ) -> None:
        self.geometry_ops = geometry_ops
        self.affine_grid = AffineGridLoss(affine_grid_cfg)
        self.affine_pair = AffinePairwiseGeometryLoss(geometry_ops, affine_pair_cfg)
        self.affine_reg = AffineLinearRegularization()
        self.affine_ref = RefAffineIdentityLoss()
        self.height_loss = HeightHuberLoss(beta=height_beta)
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
                "point_abs",
                "point_anchor",
                "rpc_corrected",
                "patch_tokens_nce_proj",
                "patch_valid_mask",
                "patch_centers",
                "patch_grid_hw",
                "patch_padded_hw",
                "patch_tokens_detail",
            ],
            "outputs",
        )
        self._require_keys(batch, ["height_gt", "height_valid_mask", "affine_gt_forward", "rpc_gt", "images"], "batch")

        affine_pred = outputs["affine_pred"]
        ref_idx = batch.get("ref_view_idx", None)
        affine_pred_for_loss = self._replace_ref_affine_with_identity(affine_pred, ref_idx)
        outputs_for_affine_pair = dict(outputs)
        outputs_for_affine_pair["affine_pred"] = affine_pred_for_loss
        if "rpc_init" in batch:
            outputs_for_affine_pair["rpc_corrected"] = self.geometry_ops.apply_affine_correction_batch(batch["rpc_init"], affine_pred_for_loss)
        image_hw = (int(outputs["height_abs"].shape[-2]), int(outputs["height_abs"].shape[-1]))

        l_aff_grid, p_aff_grid = self.affine_grid(
            affine_pred=affine_pred_for_loss,
            affine_gt_forward=batch["affine_gt_forward"].to(device=affine_pred.device, dtype=affine_pred.dtype),
            image_hw=image_hw,
            ref_view_idx=ref_idx,
        )
        l_aff_pair, p_aff_pair, aux_pair = self.affine_pair(outputs_for_affine_pair, batch)
        l_aff_reg, p_aff_reg = self.affine_reg(affine_pred_for_loss, ref_view_idx=ref_idx)
        l_aff_ref, p_aff_ref = self.affine_ref(
            affine_pred,
            image_hw=image_hw,
            ref_view_idx=ref_idx,
        )

        l_h, p_h = self.height_loss(
            outputs["height_abs"],
            batch["height_gt"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype),
            batch["height_valid_mask"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype),
        )

        l_p, p_p, aux_point = self.point_loss(outputs["point_abs"], outputs["point_anchor"], batch, return_aux=False)
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
            patch_tokens_detail=outputs["patch_tokens_detail"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_centers=outputs["patch_centers"],
            patch_grid_hw=outputs["patch_grid_hw"],
            batch=batch,
        )

        w = self.weights
        abs_mul = 1.0 if int(global_step) < int(w.abs_keep_steps) else 0.1
        w_h_abs = float(w.lambda_height) * abs_mul
        w_p_abs = float(w.lambda_point) * abs_mul
        total = (
            w.lambda_affine_grid * l_aff_grid
            + w.lambda_affine_pair * l_aff_pair
            + w.lambda_affine_reg * l_aff_reg
            + w.lambda_affine_ref * l_aff_ref
            + w_h_abs * l_h
            + w_p_abs * l_p
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
            "loss_affine_ref": l_aff_ref,
            "loss_height": l_h,
            "loss_height_rel": zero,
            "loss_point": l_p,
            "loss_point_reproj": l_preproj,
            "loss_height_reproj": l_hreproj,
            "loss_point_pair": l_ppair,
            "loss_normal_height": l_nh,
            "loss_normal_point": l_np,
            "loss_feature_nce": l_nce,
            "loss_patch_match": l_patch_match,
            "metric_affine_grid_error_px_mean": p_aff_grid.get("affine_grid_error_px_mean", zero),
            "metric_affine_pair_error_px_mean": p_aff_pair.get("affine_pair_error_px_mean", zero),
            "metric_ref_affine_identity_l2": p_aff_ref.get("ref_affine_identity_l2", zero),
            "metric_height_rmse": p_h.get("height_rmse", zero),
            "metric_height_mae": p_h.get("height_mae", zero),
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
            "weight_affine_grid": float(w.lambda_affine_grid),
            "weight_affine_pair": float(w.lambda_affine_pair),
            "weight_affine_reg": float(w.lambda_affine_reg),
            "weight_affine_ref": float(w.lambda_affine_ref),
            "weight_height": float(w_h_abs),
            "weight_height_rel": 0.0,
            "weight_point": float(w_p_abs),
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
        scalar_dict["metric_feature_nce_valid_pairs"] = p_nce.get("feature_nce_valid_pairs", zero)
        scalar_dict["metric_feature_nce_acc_top1"] = p_nce.get("feature_nce_acc_top1", zero)

        return total, self._to_float_scalar_dict(scalar_dict), aux_dict

    __call__ = forward

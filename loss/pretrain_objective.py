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
from loss.point_pair_loss import PointPairwiseConsistencyLoss, PointPairwiseLossCfg
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
    lambda_point_pair: float = 0.1
    lambda_feature_nce: float = 0.1


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
        self.point_pair_loss = PointPairwiseConsistencyLoss(geometry_ops=geometry_ops, cfg=point_pair_cfg or PointPairwiseLossCfg())
        self.feature_nce_loss = FeatureInfoNCELoss(geometry_ops=geometry_ops, cfg=feature_nce_cfg or FeatureInfoNCELossCfg())
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
        del global_step, epoch, render_outputs, mode

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
            ],
            "outputs",
        )
        self._require_keys(batch, ["height_gt", "height_valid_mask", "affine_gt_forward", "rpc_gt", "images"], "batch")

        affine_pred = outputs["affine_pred"]
        ref_idx = batch.get("ref_view_idx", None)
        affine_pred_for_loss = self._replace_ref_affine_with_identity(affine_pred, ref_idx)
        image_hw = (int(outputs["height_abs"].shape[-2]), int(outputs["height_abs"].shape[-1]))

        l_aff_grid, p_aff_grid = self.affine_grid(
            affine_pred=affine_pred_for_loss,
            affine_gt_forward=batch["affine_gt_forward"].to(device=affine_pred.device, dtype=affine_pred.dtype),
            image_hw=image_hw,
            ref_view_idx=ref_idx,
        )
        l_aff_pair, p_aff_pair, aux_pair = self.affine_pair(outputs, batch)
        l_aff_reg, p_aff_reg = self.affine_reg(affine_pred_for_loss, ref_view_idx=ref_idx)
        l_aff_ref, p_aff_ref = self.affine_ref(affine_pred, ref_view_idx=ref_idx)

        l_h, p_h = self.height_loss(
            outputs["height_abs"],
            batch["height_gt"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype),
            batch["height_valid_mask"].to(device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype),
        )

        l_p, p_p, aux_point = self.point_loss(outputs["point_abs"], outputs["point_anchor"], batch, return_aux=False)
        l_ppair, p_ppair, aux_ppair = self.point_pair_loss(outputs["point_abs"], batch)
        l_nce, p_nce, aux_nce = self.feature_nce_loss(
            patch_tokens_proj=outputs["patch_tokens_nce_proj"],
            patch_valid_mask=outputs["patch_valid_mask"],
            patch_centers=outputs["patch_centers"],
            patch_grid_hw=outputs["patch_grid_hw"],
            batch=batch,
        )

        w = self.weights
        total = (
            w.lambda_affine_grid * l_aff_grid
            + w.lambda_affine_pair * l_aff_pair
            + w.lambda_affine_reg * l_aff_reg
            + w.lambda_affine_ref * l_aff_ref
            + w.lambda_height * l_h
            + w.lambda_point * l_p
            + w.lambda_point_pair * l_ppair
            + w.lambda_feature_nce * l_nce
        )

        zero = torch.zeros((), device=total.device, dtype=total.dtype)
        scalar_dict: dict[str, Any] = {
            "loss_total": total,
            "loss_affine_grid": l_aff_grid,
            "loss_affine_pair": l_aff_pair,
            "loss_affine_reg": l_aff_reg,
            "loss_affine_ref": l_aff_ref,
            "loss_height": l_h,
            "loss_point": l_p,
            "loss_point_pair": l_ppair,
            "loss_feature_nce": l_nce,
            "metric_affine_grid_error_px_mean": p_aff_grid.get("affine_grid_error_px_mean", zero),
            "metric_affine_pair_error_px_mean": p_aff_pair.get("affine_pair_error_px_mean", zero),
            "metric_ref_affine_identity_l2": p_aff_ref.get("ref_affine_identity_l2", zero),
            "metric_height_rmse": p_h.get("height_rmse", zero),
            "metric_height_mae": p_h.get("height_mae", zero),
            "metric_point_xyz_rmse": p_p.get("point_xyz_rmse", zero),
            "metric_point_xy_rmse": p_p.get("point_xy_rmse", zero),
            "metric_point_z_rmse": p_p.get("point_z_rmse", zero),
            "metric_point_anchor_displacement_mean": p_p.get("point_anchor_displacement_mean", zero),
            "metric_point_pair_dist_mean": p_ppair.get("point_pair_dist_mean", zero),
            "metric_point_pair_num_pairs_used": p_ppair.get("point_pair_num_pairs_used", zero),
            "weight_affine_grid": float(w.lambda_affine_grid),
            "weight_affine_pair": float(w.lambda_affine_pair),
            "weight_affine_reg": float(w.lambda_affine_reg),
            "weight_affine_ref": float(w.lambda_affine_ref),
            "weight_height": float(w.lambda_height),
            "weight_point": float(w.lambda_point),
            "weight_point_pair": float(w.lambda_point_pair),
            "weight_feature_nce": float(w.lambda_feature_nce),
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
        }

        aux_dict: dict[str, Any] = {
            "num_pairs_used": aux_pair.get("num_pairs_used", 0),
            "render_rpc_num_targets": 0.0,
            "render_point_num_targets": 0.0,
        }
        aux_dict.update(aux_point)
        aux_dict.update(aux_ppair)
        aux_dict.update(aux_nce)
        scalar_dict["metric_feature_nce_valid_pairs"] = p_nce.get("feature_nce_valid_pairs", zero)
        scalar_dict["metric_feature_nce_acc_top1"] = p_nce.get("feature_nce_acc_top1", zero)

        return total, self._to_float_scalar_dict(scalar_dict), aux_dict

    __call__ = forward

"""loss.total_loss

总训练目标：
- LossWeightScheduler
- RPCAnySplatTrainingObjective

方向约定（反复强调）：
- affine_gt_forward: true pixel -> observed pixel
- affine_pred:       observed pixel -> true pixel
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from loss.height_loss import HeightHuberLoss
from loss.point_loss import PointMapLoss
from loss.regularization_loss import CenterConsistencyLoss, CoderProbe, GaussianRegularizationLoss
from loss.render_loss import RenderPathLoss


@dataclass
class LossWeightScheduler:
    """动态权重调度器。

    功能:
        前期只训练几何主线，render 权重为 0；
        随后在 ramp 阶段将 render 权重平滑提升到目标值。

    成员变量:
        warmup_steps_geom_only: 几何预热步数。
        render_ramp_steps: render 权重爬升步数。
        base_weights: 各项基础权重字典。
        ramp_mode: "linear" 或 "cosine"。
    """

    warmup_steps_geom_only: int = 1000
    render_ramp_steps: int = 2000
    base_weights: dict[str, float] = field(
        default_factory=lambda: {
            "lambda_affine_grid": 1.0,
            "lambda_affine_pair": 1.0,
            "lambda_affine_reg": 0.1,
            "lambda_affine_ref": 0.1,
            "lambda_height": 1.0,
            "lambda_point": 1.0,
            "lambda_center": 0.2,
            "lambda_opacity_reg": 0.01,
            "lambda_scale_reg": 0.01,
            "lambda_render_rpc": 1.0,
            "lambda_render_point": 1.0,
            "render_rgb_l1": 1.0,
            "render_rgb_ssim": 0.2,
            "render_height": 0.5,
            "render_alpha": 0.01,
        }
    )
    ramp_mode: str = "linear"

    def _render_multiplier(self, global_step: int) -> float:
        """计算 render 权重乘子。"""
        step = int(global_step)
        if step < self.warmup_steps_geom_only:
            return 0.0
        if self.render_ramp_steps <= 0:
            return 1.0
        if step >= self.warmup_steps_geom_only + self.render_ramp_steps:
            return 1.0

        t = (step - self.warmup_steps_geom_only) / float(self.render_ramp_steps)
        t = max(0.0, min(1.0, t))
        if self.ramp_mode == "cosine":
            return float(0.5 - 0.5 * torch.cos(torch.tensor(t * torch.pi)).item())
        return float(t)

    def __call__(self, global_step: int, epoch: int = 0) -> tuple[dict[str, float], dict[str, float]]:
        """返回当前 step 的权重与调度 probe。"""
        w = dict(self.base_weights)
        mul = self._render_multiplier(global_step)
        w["lambda_render_rpc"] = w["lambda_render_rpc"] * mul
        w["lambda_render_point"] = w["lambda_render_point"] * mul
        return w, {"schedule_render_multiplier": mul}


class RPCAnySplatTrainingObjective:
    """Sat2World 总损失组合器。

    功能:
        统一调用各子 loss 并输出 total_loss、scalar_dict、aux_dict。

    成员变量:
        geometry_ops: 几何接口实例（来自 geometry.rpc_geometry.RPCGeometryOps）。
        affine_grid/affine_pair/affine_reg: 仿射相关损失模块。
        height_loss/point_loss: 高程与点云监督模块。
        center_loss/gaussian_reg/coder_probe: 正则与探针模块。
        render_rpc/render_point: 两条路径渲染损失模块。
        scheduler: 权重调度器。

    重要字段约定（outputs 期望）:
        必需:
            affine_pred, height_abs, point_abs, point_anchor,
            gaussian_centers_rpc, gaussian_centers_point,
            gaussian_opacity, gaussian_scale, gaussian_rotation,
            gaussian_confidence_rpc, gaussian_confidence_point,
            rpc_corrected
        可选（用于 coder probe）:
            height_logits / height_coarse_logits,
            height_fine_raw,
            point_logits / point_coarse_logits,
            point_fine_raw。
    """

    def __init__(
        self,
        geometry_ops: Any,
        *,
        affine_grid_cfg: AffineGridLossCfg | None = None,
        affine_pair_cfg: AffinePairwiseGeometryLossCfg | None = None,
        height_beta: float = 1.0,
        point_beta: float = 1.0,
        scale_min: float = 1e-4,
        scale_max: float = 0.5,
        scheduler: LossWeightScheduler | None = None,
    ) -> None:
        """初始化总 objective。"""
        self.geometry_ops = geometry_ops
        self.affine_grid = AffineGridLoss(affine_grid_cfg)
        self.affine_pair = AffinePairwiseGeometryLoss(geometry_ops, affine_pair_cfg)
        self.affine_reg = AffineLinearRegularization()
        self.affine_ref = RefAffineIdentityLoss()

        self.height_loss = HeightHuberLoss(beta=height_beta)
        self.point_loss = PointMapLoss(geometry_ops=geometry_ops, beta=point_beta)

        self.center_loss = CenterConsistencyLoss()
        self.gaussian_reg = GaussianRegularizationLoss(scale_min=scale_min, scale_max=scale_max)
        self.coder_probe = CoderProbe(default_zero=True)

        self.scheduler = scheduler or LossWeightScheduler()

        bw = self.scheduler.base_weights
        self.render_rpc = RenderPathLoss(
            w_l1=bw.get("render_rgb_l1", 1.0),
            w_ssim=bw.get("render_rgb_ssim", 0.2),
            w_h=bw.get("render_height", 0.0),
            w_alpha=bw.get("render_alpha", 0.0),
        )
        self.render_point = RenderPathLoss(
            w_l1=bw.get("render_rgb_l1", 1.0),
            w_ssim=bw.get("render_rgb_ssim", 0.2),
            w_h=bw.get("render_height", 0.0),
            w_alpha=bw.get("render_alpha", 0.0),
        )

    def _require_keys(self, data: dict[str, Any], keys: list[str], name: str) -> None:
        """检查必需字段并在缺失时抛出清晰错误。"""
        miss = [k for k in keys if k not in data]
        if miss:
            raise KeyError(f"Missing required {name} keys: {miss}")

    def _to_float_scalar_dict(self, scalar_dict: dict[str, Any]) -> dict[str, float | torch.Tensor]:
        """把可 item 的标量张量转换为 Python float，便于日志系统使用。"""
        out: dict[str, float | torch.Tensor] = {}
        for k, v in scalar_dict.items():
            if torch.is_tensor(v) and v.ndim == 0:
                out[k] = float(v.detach().cpu().item())
            else:
                out[k] = v
        return out

    def _replace_ref_affine_with_identity(self, affine_pred: torch.Tensor, ref_view_idx: torch.Tensor | None) -> torch.Tensor:
        """在 loss 计算阶段把参考视图 affine 替换为单位阵。"""
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
        """执行总损失组合。

        输入:
            outputs: model.forward 输出字典。
            batch: dataset/collate 输出字典。
            global_step: 当前训练步。
            epoch: 当前 epoch。
            render_outputs:
                可选，格式约定为 {"rpc": path_result, "point": path_result}。
            mode: "train"/"val" 等，当前仅用于接口保留。

        输出:
            total_loss: 标量张量。
            scalar_dict: 标量字典（适合 TensorBoard）。
            aux_dict: 轻量辅助字典。
        """
        self._require_keys(
            outputs,
            [
                "affine_pred",
                "height_abs",
                "point_abs",
                "point_anchor",
                "gaussian_centers_rpc",
                "gaussian_centers_point",
                "gaussian_opacity",
                "gaussian_scale",
                "gaussian_rotation",
                "rpc_corrected",
            ],
            "outputs",
        )
        self._require_keys(
            batch,
            [
                "height_gt",
                "height_valid_mask",
                "affine_gt_forward",
                "rpc_gt",
                "images",
            ],
            "batch",
        )

        weights, schedule_probe = self.scheduler(global_step=global_step, epoch=epoch)
        image_hw = (int(outputs["height_abs"].shape[-2]), int(outputs["height_abs"].shape[-1]))
        ref_idx = batch.get("ref_view_idx", None)
        affine_pred = outputs["affine_pred"]
        affine_pred_for_loss = self._replace_ref_affine_with_identity(affine_pred, ref_idx)

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

        l_center, p_center = self.center_loss(
            outputs["gaussian_centers_rpc"],
            outputs["gaussian_centers_point"],
            batch.get("height_valid_mask", None),
        )

        reg_dict, p_gauss = self.gaussian_reg(
            outputs["gaussian_opacity"],
            outputs["gaussian_scale"],
            outputs["gaussian_rotation"],
            outputs.get("gaussian_confidence_rpc", None),
            outputs.get("gaussian_confidence_point", None),
        )
        l_opacity = reg_dict["opacity_reg_loss"]
        l_scale = reg_dict["scale_reg_loss"]

        p_coder = self.coder_probe(outputs, valid_mask=batch.get("height_valid_mask", None))

        zero = torch.zeros((), device=l_h.device, dtype=l_h.dtype)
        l_render_rpc = zero
        l_render_point = zero
        p_render_rpc = {
            "render_l1": zero,
            "render_ssim_loss": zero,
            "render_psnr": zero,
            "render_height_huber": zero,
            "render_alpha_mean": zero,
            "render_alpha_coverage": zero,
            "render_num_targets": zero,
        }
        p_render_point = dict(p_render_rpc)

        if render_outputs is not None:
            if "rpc" in render_outputs:
                l_render_rpc, p_render_rpc = self.render_rpc(render_outputs["rpc"])
            if "point" in render_outputs:
                l_render_point, p_render_point = self.render_point(render_outputs["point"])

        total = (
            weights["lambda_affine_grid"] * l_aff_grid
            + weights["lambda_affine_pair"] * l_aff_pair
            + weights["lambda_affine_reg"] * l_aff_reg
            + weights.get("lambda_affine_ref", 1.0) * l_aff_ref
            + weights["lambda_height"] * l_h
            + weights["lambda_point"] * l_p
            + weights["lambda_center"] * l_center
            + weights["lambda_opacity_reg"] * l_opacity
            + weights["lambda_scale_reg"] * l_scale
            + weights["lambda_render_rpc"] * l_render_rpc
            + weights["lambda_render_point"] * l_render_point
        )

        scalar_dict: dict[str, Any] = {
            "loss_total": total,
            "loss_affine_grid": l_aff_grid,
            "loss_affine_pair": l_aff_pair,
            "loss_affine_reg": l_aff_reg,
            "loss_affine_ref": l_aff_ref,
            "loss_height": l_h,
            "loss_point": l_p,
            "loss_center_consistency": l_center,
            "loss_gaussian_opacity_reg": l_opacity,
            "loss_gaussian_scale_reg": l_scale,
            "loss_render_rpc": l_render_rpc,
            "loss_render_point": l_render_point,
            "metric_affine_grid_error_px_mean": p_aff_grid.get("affine_grid_error_px_mean", zero),
            "metric_affine_pair_error_px_mean": p_aff_pair.get("affine_pair_error_px_mean", zero),
            "metric_ref_affine_identity_l2": p_aff_ref.get("ref_affine_identity_l2", zero),
            "metric_height_rmse": p_h.get("height_rmse", zero),
            "metric_height_mae": p_h.get("height_mae", zero),
            "metric_point_xyz_rmse": p_p.get("point_xyz_rmse", zero),
            "metric_point_xy_rmse": p_p.get("point_xy_rmse", zero),
            "metric_point_z_rmse": p_p.get("point_z_rmse", zero),
            "metric_point_anchor_displacement_mean": p_p.get("point_anchor_displacement_mean", zero),
            "metric_center_dist_mean": p_center.get("center_dist_mean", zero),
            "metric_center_dist_rmse": p_center.get("center_dist_rmse", zero),
            "probe_gaussian_opacity_mean": p_gauss.get("gaussian_opacity_mean", zero),
            "probe_gaussian_scale_mean": p_gauss.get("gaussian_scale_mean", zero),
            "probe_gaussian_confidence_rpc_mean": p_gauss.get("gaussian_confidence_rpc_mean", zero),
            "probe_gaussian_confidence_point_mean": p_gauss.get("gaussian_confidence_point_mean", zero),
            "probe_gaussian_quat_norm_error": p_gauss.get("gaussian_quat_norm_error", zero),
            "probe_height_coarse_entropy": p_coder.get("height_coarse_entropy", zero),
            "probe_height_fine_abs_mean": p_coder.get("height_fine_abs_mean", zero),
            "probe_point_x_coarse_entropy": p_coder.get("point_x_coarse_entropy", zero),
            "probe_point_y_coarse_entropy": p_coder.get("point_y_coarse_entropy", zero),
            "probe_point_z_coarse_entropy": p_coder.get("point_z_coarse_entropy", zero),
            "probe_point_x_fine_abs_mean": p_coder.get("point_x_fine_abs_mean", zero),
            "probe_point_y_fine_abs_mean": p_coder.get("point_y_fine_abs_mean", zero),
            "probe_point_z_fine_abs_mean": p_coder.get("point_z_fine_abs_mean", zero),
            "schedule_render_multiplier": schedule_probe["schedule_render_multiplier"],
            "weight_affine_grid": weights["lambda_affine_grid"],
            "weight_affine_pair": weights["lambda_affine_pair"],
            "weight_affine_reg": weights["lambda_affine_reg"],
            "weight_affine_ref": weights.get("lambda_affine_ref", 1.0),
            "weight_height": weights["lambda_height"],
            "weight_point": weights["lambda_point"],
            "weight_center": weights["lambda_center"],
            "weight_opacity_reg": weights["lambda_opacity_reg"],
            "weight_scale_reg": weights["lambda_scale_reg"],
            "weight_render_rpc": weights["lambda_render_rpc"],
            "weight_render_point": weights["lambda_render_point"],
        }

        aux_dict: dict[str, Any] = {
            "num_pairs_used": aux_pair.get("num_pairs_used", 0),
            "render_rpc_num_targets": float(p_render_rpc.get("render_num_targets", zero).detach().item()) if torch.is_tensor(p_render_rpc.get("render_num_targets", None)) else p_render_rpc.get("render_num_targets", 0),
            "render_point_num_targets": float(p_render_point.get("render_num_targets", zero).detach().item()) if torch.is_tensor(p_render_point.get("render_num_targets", None)) else p_render_point.get("render_num_targets", 0),
        }
        aux_dict.update(aux_point)

        return total, self._to_float_scalar_dict(scalar_dict), aux_dict

    __call__ = forward

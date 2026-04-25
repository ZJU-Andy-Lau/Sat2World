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
import math
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
from loss.point_loss import PointMapLoss
from loss.point_pair_loss import HeightReprojectionLoss, PointPairwiseConsistencyLoss, PointPairwiseLossCfg, PointReprojectionLoss
from loss.regularization_loss import CenterConsistencyLoss, CoderProbe, GaussianRegularizationLoss
from loss.render_loss import RenderPathLoss
from loss.z_space_loss import masked_z_huber_loss


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
            "lambda_height_anchor": 0.5,
            "lambda_point": 1.0,
            "lambda_point_xy": 1.0,
            "lambda_point_z": 1.0,
            "lambda_height_meter_aux": 1.0e-3,
            "lambda_height_anchor_meter_aux": 1.0e-3,
            "lambda_point_z_meter_aux": 1.0e-3,
            "lambda_point_reproj": 0.2,
            "lambda_height_reproj": 0.2,
            "lambda_point_pair": 0.2,
            "lambda_normal_height": 0.2,
            "lambda_normal_point": 0.2,
            "lambda_feature_nce": 0.1,
            "lambda_patch_match": 0.5,
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
    stage1_steps: int = 5000
    stage2_steps: int = 20000
    abs_keep_steps: int = 5000

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
        st1 = int(self.stage1_steps)
        st2 = int(self.stage2_steps)
        if global_step < st1:
            # Stage-1 现在保留 point 分支监督（用户要求：第一阶段加入 point_map_loss）。
            # 其余较“后期”的项仍保持关闭，以避免渲染与高斯正则过早干扰几何主线。
            w["lambda_center"] = 0.0
            w["lambda_opacity_reg"] = 0.0
            w["lambda_scale_reg"] = 0.0
            w["lambda_render_rpc"] = 0.0
            w["lambda_render_point"] = 0.0
            w["lambda_point_pair"] = w.get("lambda_point_pair", 0.0) * 0.2
            stage_mul = 0.0
        elif global_step < st2:
            stage_mul = 0.35
            p_base = w.get("lambda_point", 0.0)
            pxy_base = w.get("lambda_point_xy", p_base)
            pz_base = w.get("lambda_point_z", p_base)
            w["lambda_point"] = p_base * stage_mul
            w["lambda_point_xy"] = pxy_base * stage_mul
            w["lambda_point_z"] = pz_base * stage_mul
            w["lambda_point_pair"] = w.get("lambda_point_pair", 0.0) * 0.6
            w["lambda_center"] = w["lambda_center"] * stage_mul
            w["lambda_opacity_reg"] = w["lambda_opacity_reg"] * stage_mul
            w["lambda_scale_reg"] = w["lambda_scale_reg"] * stage_mul
            w["lambda_render_rpc"] = w["lambda_render_rpc"] * stage_mul
            w["lambda_render_point"] = w["lambda_render_point"] * stage_mul
        else:
            stage_mul = 1.0
        if global_step >= int(self.abs_keep_steps):
            h_base = w.get("lambda_height", 0.0)
            p_base = w.get("lambda_point", 0.0)
            pxy_base = w.get("lambda_point_xy", p_base)
            pz_base = w.get("lambda_point_z", p_base)
            w["lambda_height"] = h_base * 0.1
            w["lambda_point"] = p_base * 0.1
            w["lambda_point_xy"] = pxy_base * 0.1
            w["lambda_point_z"] = pz_base * 0.1
        return w, {"schedule_render_multiplier": mul, "schedule_detail_stage_multiplier": stage_mul}


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
        可选（用于 probe）:
            height_anchor, height_local_z, point_z_local_z, height_z_max。
    """

    def __init__(
        self,
        geometry_ops: Any,
        *,
        affine_grid_cfg: AffineGridLossCfg | None = None,
        affine_pair_cfg: AffinePairwiseGeometryLossCfg | None = None,
        height_beta: float = 1.0,
        point_beta: float = 1.0,
        height_z_beta_meter: float | None = None,
        height_anchor_z_beta_meter: float = 5.0,
        point_z_beta_meter: float | None = None,
        point_pair_cfg: PointPairwiseLossCfg | None = None,
        feature_nce_cfg: FeatureInfoNCELossCfg | None = None,
        patch_match_cfg: PatchInternalMatchLossCfg | None = None,
        patch_matcher: torch.nn.Module | None = None,
        normal_cfg: PointNormalLossCfg | None = None,
        scale_min: float = 1e-4,
        scale_max: float = 0.5,
        scheduler: LossWeightScheduler | None = None,
    ) -> None:
        """初始化总 objective。"""
        self.geometry_ops = geometry_ops
        self.affine_grid = AffineGridLoss(affine_grid_cfg)
        self.affine_pair = AffinePairwiseGeometryLoss(geometry_ops, affine_pair_cfg)
        self.affine_reg = AffineLinearRegularization()
        # ref loss 改为“网格版 identity 约束”，并复用 affine_grid_cfg 的网格分辨率
        self.affine_ref = RefAffineIdentityLoss(affine_grid_cfg)

        self.height_loss = HeightHuberLoss(beta=height_beta)
        self.height_anchor_loss = torch.nn.SmoothL1Loss(reduction="mean")
        self.point_loss = PointMapLoss(geometry_ops=geometry_ops, beta=point_beta)
        self.height_z_beta_meter = float(height_beta if height_z_beta_meter is None else height_z_beta_meter)
        self.height_anchor_z_beta_meter = float(height_anchor_z_beta_meter)
        self.point_z_beta_meter = float(point_beta if point_z_beta_meter is None else point_z_beta_meter)
        self.point_pair_loss = PointPairwiseConsistencyLoss(geometry_ops=geometry_ops, cfg=point_pair_cfg or PointPairwiseLossCfg())
        self.point_reproj_loss = PointReprojectionLoss(geometry_ops=geometry_ops, cfg=point_pair_cfg or PointPairwiseLossCfg())
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
                "height_anchor",
                "height_anchor_z",
                "height_ref_anchor",
                "height_local_z",
                "point_abs",
                "point_anchor",
                "point_z_local_z",
                "gaussian_centers_rpc",
                "gaussian_centers_point",
                "gaussian_opacity",
                "gaussian_scale",
                "gaussian_rotation",
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
        self._require_keys(
            batch,
            [
                "height_gt",
                "height_valid_mask",
                "height_anchor_gt",
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
        outputs_for_affine_pair = dict(outputs)
        outputs_for_affine_pair["affine_pred"] = affine_pred_for_loss
        if "rpc_init" in batch:
            outputs_for_affine_pair["rpc_corrected"] = self.geometry_ops.apply_affine_correction_batch(batch["rpc_init"], affine_pred_for_loss)

        l_aff_grid, p_aff_grid = self.affine_grid(
            affine_pred=affine_pred_for_loss,
            affine_gt_forward=batch["affine_gt_forward"].to(device=affine_pred.device, dtype=affine_pred.dtype),
            image_hw=image_hw,
            ref_view_idx=ref_idx,
        )
        l_aff_pair, p_aff_pair, aux_pair = self.affine_pair(outputs_for_affine_pair, batch)
        l_aff_reg, p_aff_reg = self.affine_reg(affine_pred_for_loss, ref_view_idx=ref_idx)
        # 这里改为网格版参考视图 identity 约束
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
        l_h_anchor = self.height_anchor_loss(
            outputs["height_anchor"],
            batch["height_anchor_gt"].to(device=outputs["height_anchor"].device, dtype=outputs["height_anchor"].dtype),
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
        height_z_max = outputs.get("height_z_max", torch.tensor(4.0, device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype))
        height_local_scale = outputs.get("height_local_scale", torch.tensor(30.0, device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype))
        height_anchor_scale = outputs.get("height_anchor_scale", torch.tensor(200.0, device=outputs["height_abs"].device, dtype=outputs["height_abs"].dtype))

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

        probe_outputs = dict(outputs)
        probe_outputs["height_anchor_gt"] = batch["height_anchor_gt"].to(
            device=outputs["height_anchor"].device,
            dtype=outputs["height_anchor"].dtype,
        )
        p_coder = self.coder_probe(probe_outputs, valid_mask=batch.get("height_valid_mask", None))

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
        l_ssim = 0.5 * (p_render_rpc.get("render_ssim_loss", zero) + p_render_point.get("render_ssim_loss", zero))

        w_point_xy = weights.get("lambda_point_xy", weights["lambda_point"])
        w_point_z = weights.get("lambda_point_z", weights["lambda_point"])
        w_h_meter_aux = weights.get("lambda_height_meter_aux", 1.0e-3)
        w_h_anchor_meter_aux = weights.get("lambda_height_anchor_meter_aux", 1.0e-3)
        w_p_z_meter_aux = weights.get("lambda_point_z_meter_aux", 1.0e-3)

        l_h_optim = l_h_z + w_h_meter_aux * l_h
        l_h_anchor_optim = l_h_anchor_z + w_h_anchor_meter_aux * l_h_anchor
        l_p_z_optim = l_p_z_z + w_p_z_meter_aux * l_p_z_meter

        total = (
            weights["lambda_affine_grid"] * l_aff_grid
            + weights["lambda_affine_pair"] * l_aff_pair
            + weights["lambda_affine_reg"] * l_aff_reg
            + weights.get("lambda_affine_ref", 1.0) * l_aff_ref
            + weights["lambda_height"] * l_h_optim
            + weights.get("lambda_height_anchor", 0.5) * l_h_anchor_optim
            + w_point_xy * l_p_xy
            + w_point_z * l_p_z_optim
            + weights.get("lambda_point_reproj", 0.0) * l_preproj
            + weights.get("lambda_height_reproj", 0.0) * l_hreproj
            + weights.get("lambda_point_pair", 0.0) * l_ppair
            + weights.get("lambda_normal_height", 0.0) * l_nh
            + weights.get("lambda_normal_point", 0.0) * l_np
            + weights.get("lambda_feature_nce", 0.0) * l_nce
            + weights.get("lambda_patch_match", 0.0) * l_patch_match
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
            "loss_center_consistency": l_center,
            "loss_gaussian_opacity_reg": l_opacity,
            "loss_gaussian_scale_reg": l_scale,
            "loss_render_rpc": l_render_rpc,
            "loss_render_point": l_render_point,
            "loss_ssim": l_ssim,
            "metric_affine_grid_error_px_mean": p_aff_grid.get("affine_grid_error_px_mean", zero),
            "metric_affine_pair_error_px_mean": p_aff_pair.get("affine_pair_error_px_mean", zero),
            # 新增：更准确地记录 ref 网格误差
            "metric_ref_affine_grid_error_px_mean": p_aff_ref.get("ref_affine_grid_error_px_mean", zero),
            "metric_ref_affine_grid_error_px_rmse": p_aff_ref.get("ref_affine_grid_error_px_rmse", zero),
            # 兼容旧 tag，值现在等于 ref 网格误差均值
            "metric_ref_affine_identity_l2": p_aff_ref.get("ref_affine_identity_l2", zero),
            "metric_height_rmse": p_h.get("height_rmse", zero),
            "metric_height_mae": p_h.get("height_mae", zero),
            "metric_height_bias": p_h.get("height_bias", zero),
            "metric_height_anchor_mae": p_coder.get("height_anchor_mae", zero),
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
            "metric_feature_nce_valid_pairs": p_nce.get("feature_nce_valid_pairs", zero),
            "metric_feature_nce_acc_top1": p_nce.get("feature_nce_acc_top1", zero),
            "metric_patch_match_valid_pairs": p_patch_match.get("patch_match_valid_pairs", zero),
            "metric_patch_match_acc_top1_1px": p_patch_match.get("patch_match_acc_top1_1px", zero),
            "metric_patch_match_l1_px": p_patch_match.get("patch_match_l1_px", zero),
            "metric_center_dist_mean": p_center.get("center_dist_mean", zero),
            "metric_center_dist_rmse": p_center.get("center_dist_rmse", zero),
            "probe_gaussian_opacity_mean": p_gauss.get("gaussian_opacity_mean", zero),
            "probe_gaussian_scale_mean": p_gauss.get("gaussian_scale_mean", zero),
            "probe_gaussian_confidence_rpc_mean": p_gauss.get("gaussian_confidence_rpc_mean", zero),
            "probe_gaussian_confidence_point_mean": p_gauss.get("gaussian_confidence_point_mean", zero),
            "probe_gaussian_quat_norm_error": p_gauss.get("gaussian_quat_norm_error", zero),
            "probe_height_local_z_boundary_ratio": p_coder.get("height_local_z_boundary_ratio", zero),
            "probe_point_z_local_z_boundary_ratio": p_coder.get("point_z_local_z_boundary_ratio", zero),
            "schedule_render_multiplier": schedule_probe["schedule_render_multiplier"],
            "schedule_detail_stage_multiplier": schedule_probe.get("schedule_detail_stage_multiplier", 1.0),
            "weight_affine_grid": weights["lambda_affine_grid"],
            "weight_affine_pair": weights["lambda_affine_pair"],
            "weight_affine_reg": weights["lambda_affine_reg"],
            "weight_affine_ref": weights.get("lambda_affine_ref", 1.0),
            "weight_height": weights["lambda_height"],
            "weight_height_anchor": weights.get("lambda_height_anchor", 0.5),
            "weight_height_meter_aux": w_h_meter_aux,
            "weight_height_anchor_meter_aux": w_h_anchor_meter_aux,
            "weight_height_rel": 0.0,
            "weight_point": weights["lambda_point"],
            "weight_point_xy": w_point_xy,
            "weight_point_z": w_point_z,
            "weight_point_z_meter_aux": w_p_z_meter_aux,
            "weight_point_reproj": weights.get("lambda_point_reproj", 0.0),
            "weight_height_reproj": weights.get("lambda_height_reproj", 0.0),
            "weight_point_pair": weights.get("lambda_point_pair", 0.0),
            "weight_normal_height": weights.get("lambda_normal_height", 0.0),
            "weight_normal_point": weights.get("lambda_normal_point", 0.0),
            "weight_feature_nce": weights.get("lambda_feature_nce", 0.0),
            "weight_patch_match": weights.get("lambda_patch_match", 0.0),
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
        aux_dict.update(aux_preproj)
        aux_dict.update(aux_ppair)
        aux_dict.update(aux_norm)
        aux_dict.update(aux_hreproj)
        aux_dict.update(aux_nce)
        aux_dict.update(aux_patch_match)
        aux_dict.update(self._build_nan_probe(scalar_dict))

        return total, self._to_float_scalar_dict(scalar_dict), aux_dict

    __call__ = forward

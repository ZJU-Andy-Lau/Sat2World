"""loss.regularization_loss

实现非主监督项：
- CenterConsistencyLoss
- GaussianRegularizationLoss
- CoderProbe
"""

from __future__ import annotations

from typing import Any

import torch

from loss.common import masked_l1_loss, masked_reduce, safe_rmse, softmax_entropy


class CenterConsistencyLoss:
    """双路径中心一致性损失。

    功能:
        约束 gaussian_centers_rpc 与 gaussian_centers_point 不发生大幅漂离。
    """

    def __init__(self) -> None:
        """初始化（无超参数）。"""

    def __call__(
        self,
        gaussian_centers_rpc: torch.Tensor,
        gaussian_centers_point: torch.Tensor,
        height_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算中心一致性损失与 probe。"""
        loss = masked_l1_loss(gaussian_centers_rpc, gaussian_centers_point, mask=height_valid_mask)
        d = gaussian_centers_rpc - gaussian_centers_point
        dist = torch.linalg.norm(d, dim=2, keepdim=True)
        probe = {
            "center_dist_mean": masked_reduce(dist, mask=height_valid_mask, reduce="mean").detach(),
            "center_dist_rmse": safe_rmse(dist.square(), mask=height_valid_mask).detach(),
        }
        return loss, probe


class GaussianRegularizationLoss:
    """高斯健康性正则。

    功能:
        - opacity 稀疏正则
        - scale 范围正则
        同时返回 gaussian 统计 probe。
    """

    def __init__(self, scale_min: float = 1e-4, scale_max: float = 0.5) -> None:
        """初始化正则参数。"""
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

    def __call__(
        self,
        gaussian_opacity: torch.Tensor,
        gaussian_scale: torch.Tensor,
        gaussian_rotation: torch.Tensor,
        gaussian_confidence_rpc: torch.Tensor | None = None,
        gaussian_confidence_point: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """计算正则项与 probe。

        输出:
            reg_loss_dict:
                至少包含 opacity_reg_loss 与 scale_reg_loss。
            probe_dict:
                各类统计量。
        """
        opacity_reg = gaussian_opacity.mean()
        low = torch.relu(self.scale_min - gaussian_scale).square()
        high = torch.relu(gaussian_scale - self.scale_max).square()
        scale_reg = (low + high).mean()

        q_norm = torch.linalg.norm(gaussian_rotation, dim=2, keepdim=True)
        quat_norm_err = (q_norm - 1.0).abs().mean()

        conf_rpc_mean = gaussian_confidence_rpc.mean() if gaussian_confidence_rpc is not None else torch.zeros_like(opacity_reg)
        conf_point_mean = gaussian_confidence_point.mean() if gaussian_confidence_point is not None else torch.zeros_like(opacity_reg)

        reg = {
            "opacity_reg_loss": opacity_reg,
            "scale_reg_loss": scale_reg,
        }
        probe = {
            "gaussian_opacity_mean": gaussian_opacity.mean().detach(),
            "gaussian_opacity_std": gaussian_opacity.std(unbiased=False).detach(),
            "gaussian_scale_mean": gaussian_scale.mean().detach(),
            "gaussian_scale_std": gaussian_scale.std(unbiased=False).detach(),
            "gaussian_scale_min": gaussian_scale.min().detach(),
            "gaussian_scale_max": gaussian_scale.max().detach(),
            "gaussian_confidence_rpc_mean": conf_rpc_mean.detach(),
            "gaussian_confidence_point_mean": conf_point_mean.detach(),
            "gaussian_quat_norm_error": quat_norm_err.detach(),
        }
        return reg, probe


class CoderProbe:
    """Coder 探针指标。

    功能:
        读取 model outputs 中可选的 raw logits/fine 张量，
        仅输出指标，不产生训练损失。

    兼容字段:
        - height_logits 或 height_coarse_logits
        - height_fine_raw
        - point_logits(dict) / point_coarse_logits(合并)
        - point_fine_raw(dict) / point_fine_raw(合并)
    """

    def __init__(self, default_zero: bool = True) -> None:
        """初始化。

        输入:
            default_zero: 缺失字段时是否返回 0 指标（推荐 True，便于接口稳定）。
        """
        self.default_zero = bool(default_zero)

    def _zeros(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        return {
            "height_coarse_entropy": z,
            "height_fine_abs_mean": z,
            "point_x_coarse_entropy": z,
            "point_y_coarse_entropy": z,
            "point_z_coarse_entropy": z,
            "point_x_fine_abs_mean": z,
            "point_y_fine_abs_mean": z,
            "point_z_fine_abs_mean": z,
        }

    def __call__(self, outputs: dict[str, Any], valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """计算 coder probe 指标。"""
        ref_tensor = None
        for v in outputs.values():
            if torch.is_tensor(v):
                ref_tensor = v
                break
        if ref_tensor is None:
            return {}

        out = self._zeros(ref_tensor) if self.default_zero else {}

        h_logits = outputs.get("height_logits", outputs.get("height_coarse_logits", None))
        if h_logits is not None:
            h_ent = softmax_entropy(h_logits, dim=2)
            mask = None
            if valid_mask is not None:
                mask = valid_mask.squeeze(2)
            out["height_coarse_entropy"] = masked_reduce(h_ent, mask=mask, reduce="mean").detach()

        h_fine = outputs.get("height_fine_raw", None)
        if h_fine is not None:
            out["height_fine_abs_mean"] = masked_reduce(h_fine.abs(), mask=valid_mask, reduce="mean").detach()

        p_logits_dict = outputs.get("point_logits", None)
        if p_logits_dict is not None and isinstance(p_logits_dict, dict):
            for axis in ("x", "y", "z"):
                if axis in p_logits_dict:
                    ent = softmax_entropy(p_logits_dict[axis], dim=2)
                    mask = None if valid_mask is None else valid_mask.squeeze(2)
                    out[f"point_{axis}_coarse_entropy"] = masked_reduce(ent, mask=mask, reduce="mean").detach()
        else:
            p_logits = outputs.get("point_coarse_logits", None)
            if p_logits is not None and p_logits.ndim >= 3 and p_logits.shape[2] % 3 == 0:
                k = p_logits.shape[2] // 3
                for idx, axis in enumerate(("x", "y", "z")):
                    sub = p_logits[:, :, idx * k : (idx + 1) * k]
                    ent = softmax_entropy(sub, dim=2)
                    mask = None if valid_mask is None else valid_mask.squeeze(2)
                    out[f"point_{axis}_coarse_entropy"] = masked_reduce(ent, mask=mask, reduce="mean").detach()

        p_fine = outputs.get("point_fine_raw", None)
        if isinstance(p_fine, dict):
            for axis in ("x", "y", "z"):
                if axis in p_fine:
                    out[f"point_{axis}_fine_abs_mean"] = masked_reduce(p_fine[axis].abs(), mask=valid_mask, reduce="mean").detach()
        elif p_fine is not None and p_fine.shape[2] == 3:
            out["point_x_fine_abs_mean"] = masked_reduce(p_fine[:, :, 0:1].abs(), mask=valid_mask, reduce="mean").detach()
            out["point_y_fine_abs_mean"] = masked_reduce(p_fine[:, :, 1:2].abs(), mask=valid_mask, reduce="mean").detach()
            out["point_z_fine_abs_mean"] = masked_reduce(p_fine[:, :, 2:3].abs(), mask=valid_mask, reduce="mean").detach()

        return out

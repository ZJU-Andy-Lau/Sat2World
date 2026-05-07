"""loss.regularization_loss

实现非主监督项：
- CenterConsistencyLoss
- GaussianRegularizationLoss
- CoderProbe
"""

from __future__ import annotations

from typing import Any

import torch

from loss.common import masked_l1_loss, masked_reduce, safe_rmse


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
    """高度表示探针指标（无训练损失）。"""

    def __init__(self, default_zero: bool = True) -> None:
        """初始化。

        输入:
            default_zero: 缺失字段时是否返回 0 指标（推荐 True，便于接口稳定）。
        """
        self.default_zero = bool(default_zero)

    def _zeros(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        z = torch.zeros((), device=ref.device, dtype=ref.dtype)
        return {
            "height_anchor_mae": z,
            "height_local_z_boundary_ratio": z,
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

        height_anchor = outputs.get("height_anchor", None)
        height_anchor_gt = outputs.get("height_anchor_gt", None)
        if height_anchor is not None and height_anchor_gt is not None:
            out["height_anchor_mae"] = (height_anchor - height_anchor_gt).abs().mean().detach()

        z_max_cfg = outputs.get("height_z_max", None)
        z_max_val = float(z_max_cfg.detach().item()) if torch.is_tensor(z_max_cfg) and z_max_cfg.numel() == 1 else None

        def _boundary_ratio(z_map: torch.Tensor) -> torch.Tensor:
            z_abs = z_map.abs()
            z_max = z_max_val if z_max_val is not None else (float(z_abs.detach().max().item()) if z_abs.numel() > 0 else 0.0)
            if z_max <= 0:
                return torch.zeros((), device=z_map.device, dtype=z_map.dtype)
            thr = z_max * 0.98
            return (z_abs >= thr).to(z_map.dtype).mean().detach()

        h_local_z = outputs.get("height_local_z", None)
        if h_local_z is not None:
            out["height_local_z_boundary_ratio"] = _boundary_ratio(h_local_z)
        return out

"""loss.normal_loss

点云法向监督损失：
- GT 法向来自 rpc_gt + height_gt 构造的 GT 点云；
- 预测法向来自两条路径：
  1) rpc_gt + height_abs
  2) point_abs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from loss.common import masked_reduce
from loss.point_pair_loss import point_map_to_metric


@dataclass
class PointNormalLossCfg:
    """法向损失配置。"""

    w_cos: float = 1.0
    w_l1: float = 0.5
    eps: float = 1e-6
    sign_invariant: bool = True
    detach_gt: bool = True


def compute_normals_from_point_map(
    point_map: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """由点云图计算法向图。

    输入:
        point_map: [B,V,3,H,W]
        valid_mask: [B,V,1,H,W] 或 None
    输出:
        normals: [B,V,3,H,W]
        normal_valid_mask: [B,V,1,H,W]
    """
    if point_map.ndim != 5 or point_map.shape[2] != 3:
        raise ValueError(f"point_map must be [B,V,3,H,W], got {tuple(point_map.shape)}")

    b, v, _, h, w = point_map.shape
    normals = torch.zeros_like(point_map)
    normal_valid = torch.zeros((b, v, 1, h, w), device=point_map.device, dtype=point_map.dtype)
    if h < 3 or w < 3:
        return normals, normal_valid

    dx = point_map[:, :, :, 2:, 1:-1] - point_map[:, :, :, :-2, 1:-1]
    dy = point_map[:, :, :, 1:-1, 2:] - point_map[:, :, :, 1:-1, :-2]
    n_inner = torch.cross(dx, dy, dim=2)
    n_inner = F.normalize(n_inner, p=2, dim=2, eps=float(eps))
    normals[:, :, :, 1:-1, 1:-1] = n_inner

    if valid_mask is None:
        normal_valid[:, :, :, 1:-1, 1:-1] = 1.0
        return normals, normal_valid

    vm = (valid_mask > 0).to(dtype=point_map.dtype)
    center = vm[:, :, :, 1:-1, 1:-1]
    up = vm[:, :, :, :-2, 1:-1]
    down = vm[:, :, :, 2:, 1:-1]
    left = vm[:, :, :, 1:-1, :-2]
    right = vm[:, :, :, 1:-1, 2:]
    inner_valid = center * up * down * left * right
    normal_valid[:, :, :, 1:-1, 1:-1] = inner_valid
    return normals, normal_valid


def normal_alignment_terms(
    pred_n: torch.Tensor,
    gt_n: torch.Tensor,
    mask: torch.Tensor,
    *,
    sign_invariant: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """计算法向对齐损失项（角度项 + L1 项）。"""
    if pred_n.shape != gt_n.shape:
        raise ValueError("pred_n and gt_n must share shape")
    if mask.ndim != pred_n.ndim:
        raise ValueError("mask ndim mismatch")

    dot = (pred_n * gt_n).sum(dim=2, keepdim=True)
    dot = dot.clamp(-1.0, 1.0)
    if sign_invariant:
        loss_cos_map = 1.0 - dot.abs()
        sign = torch.sign(dot.detach())
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        gt_for_l1 = gt_n * sign
    else:
        loss_cos_map = 1.0 - dot
        gt_for_l1 = gt_n

    loss_l1_map = (pred_n - gt_for_l1).abs().mean(dim=2, keepdim=True)
    loss_cos = masked_reduce(loss_cos_map, mask=mask, reduce="mean")
    loss_l1 = masked_reduce(loss_l1_map, mask=mask, reduce="mean")

    ang_deg = torch.rad2deg(torch.acos(dot.abs() if sign_invariant else dot))
    probe = {
        "normal_cos_mean": masked_reduce((dot.abs() if sign_invariant else dot), mask=mask, reduce="mean").detach(),
        "normal_ang_deg_mean": masked_reduce(ang_deg, mask=mask, reduce="mean").detach(),
        "normal_valid_ratio": mask.to(dtype=pred_n.dtype).mean().detach(),
    }
    return torch.nan_to_num(loss_cos, nan=0.0, posinf=0.0, neginf=0.0), torch.nan_to_num(
        loss_l1, nan=0.0, posinf=0.0, neginf=0.0
    ), probe


class PointNormalLoss:
    """基于点云图法向的双路径监督。"""

    def __init__(self, geometry_ops: Any, cfg: PointNormalLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or PointNormalLossCfg()
        self._grid_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def _get_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, str(device), str(dtype))
        if key not in self._grid_cache:
            yy = torch.arange(h, device=device, dtype=dtype)
            xx = torch.arange(w, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            self._grid_cache[key] = torch.stack([gy, gx], dim=-1)
        return self._grid_cache[key]

    def __call__(
        self,
        *,
        height_abs: torch.Tensor,
        point_abs: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """返回两条路径的法向损失与 probe。"""
        if height_abs.ndim != 5 or point_abs.ndim != 5:
            raise ValueError("height_abs and point_abs must be [B,V,C,H,W]")
        if point_abs.shape[2] != 3:
            raise ValueError("point_abs channel dim must be 3")

        device = point_abs.device
        dtype = point_abs.dtype
        _, _, _, h, w = point_abs.shape

        height_gt = batch["height_gt"].to(device=device, dtype=dtype)
        valid_mask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        rpc_gt = batch["rpc_gt"]
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        if torch.is_tensor(scene_xy_scale):
            scene_xy_scale = scene_xy_scale.to(device=device, dtype=dtype)

        image_grid = self._get_grid(h, w, device, dtype)
        gt_point = self.geometry_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=rpc_gt,
            pixel_grid=image_grid,
            height_abs=height_gt,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        pred_point_h = self.geometry_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=rpc_gt,
            pixel_grid=image_grid,
            height_abs=height_abs,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        pred_point_p = point_abs

        metric_scale = scene_xy_scale if torch.is_tensor(scene_xy_scale) else None
        gt_metric = point_map_to_metric(gt_point, metric_scale)
        pred_h_metric = point_map_to_metric(pred_point_h, metric_scale)
        pred_p_metric = point_map_to_metric(pred_point_p, metric_scale)

        gt_n, nmask = compute_normals_from_point_map(gt_metric, valid_mask=valid_mask, eps=float(self.cfg.eps))
        if bool(self.cfg.detach_gt):
            gt_n = gt_n.detach()
        pred_h_n, _ = compute_normals_from_point_map(pred_h_metric, valid_mask=valid_mask, eps=float(self.cfg.eps))
        pred_p_n, _ = compute_normals_from_point_map(pred_p_metric, valid_mask=valid_mask, eps=float(self.cfg.eps))

        l_h_cos, l_h_l1, p_h = normal_alignment_terms(
            pred_h_n,
            gt_n,
            nmask,
            sign_invariant=bool(self.cfg.sign_invariant),
            eps=float(self.cfg.eps),
        )
        l_p_cos, l_p_l1, p_p = normal_alignment_terms(
            pred_p_n,
            gt_n,
            nmask,
            sign_invariant=bool(self.cfg.sign_invariant),
            eps=float(self.cfg.eps),
        )
        w_cos = float(self.cfg.w_cos)
        w_l1 = float(self.cfg.w_l1)
        loss_h = torch.nan_to_num(w_cos * l_h_cos + w_l1 * l_h_l1, nan=0.0, posinf=0.0, neginf=0.0)
        loss_p = torch.nan_to_num(w_cos * l_p_cos + w_l1 * l_p_l1, nan=0.0, posinf=0.0, neginf=0.0)

        probe = {
            "normal_h_cos_mean": p_h["normal_cos_mean"],
            "normal_h_ang_deg_mean": p_h["normal_ang_deg_mean"],
            "normal_p_cos_mean": p_p["normal_cos_mean"],
            "normal_p_ang_deg_mean": p_p["normal_ang_deg_mean"],
            "normal_valid_ratio": p_h["normal_valid_ratio"],
        }
        aux = {
            "normal_mask": nmask,
        }
        return loss_h, loss_p, probe, aux

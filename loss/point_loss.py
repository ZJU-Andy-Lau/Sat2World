"""loss.point_loss

实现点云监督损失 PointMapLoss。
"""

from __future__ import annotations

from typing import Any

import torch

from loss.common import masked_huber_loss, masked_l1_loss, masked_reduce, safe_rmse
from loss.point_pair_loss import point_map_to_metric


class PointMapLoss:
    """点云图监督损失。

    功能:
        在线使用 rpc_gt + height_gt 构造 gt_point_map，
        对 point_abs 进行监督，并输出点云误差与 anchor 偏移指标。

    成员变量:
        geometry_ops: 几何接口实例，用于 RPC 投影/反投影。
        beta: Huber 参数。
        _grid_cache: 图像网格缓存，key=(H,W,device,dtype)。
    """

    def __init__(self, geometry_ops: Any, beta: float = 1.0) -> None:
        """初始化损失类。"""
        self.geometry_ops = geometry_ops
        self.beta = float(beta)
        self._grid_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}

    def _get_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取或创建 [H,W,2] 像素网格（line,samp）。"""
        key = (h, w, str(device), str(dtype))
        if key not in self._grid_cache:
            yy = torch.arange(h, device=device, dtype=dtype)
            xx = torch.arange(w, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            self._grid_cache[key] = torch.stack([gy, gx], dim=-1)
        return self._grid_cache[key]

    def __call__(
        self,
        point_abs: torch.Tensor,
        point_anchor: torch.Tensor,
        batch: dict[str, Any],
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """计算点云损失。

        输入:
            point_abs: [B,V,3,H,W]。
            point_anchor: [B,V,3,H,W]。
            batch: 至少包含 rpc_gt/height_gt/height_valid_mask/scene_xy_center/scene_xy_scale。
            return_aux: 是否返回 gt_point_map（默认 False）。

        输出:
            loss: 标量。
            probe: 误差指标字典。
            aux: 辅助字典，默认为空；return_aux=True 时可含 gt_point_map。
        """
        height_gt = batch["height_gt"].to(device=point_abs.device, dtype=point_abs.dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=point_abs.device, dtype=point_abs.dtype)
        rpc_gt = batch["rpc_gt"]
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        if scene_xy_scale is not None and torch.is_tensor(scene_xy_scale):
            scene_xy_scale = scene_xy_scale.to(device=point_abs.device, dtype=point_abs.dtype)

        b, v, _, h, w = point_abs.shape
        image_grid = self._get_grid(h, w, point_abs.device, point_abs.dtype)

        gt_point_map = self.geometry_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=rpc_gt,
            pixel_grid=image_grid,
            height_abs=height_gt,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )

        point_abs_metric = point_map_to_metric(point_abs, scene_xy_scale if torch.is_tensor(scene_xy_scale) else None)
        gt_point_map_metric = point_map_to_metric(gt_point_map, scene_xy_scale if torch.is_tensor(scene_xy_scale) else None)
        point_anchor_metric = point_map_to_metric(point_anchor, scene_xy_scale if torch.is_tensor(scene_xy_scale) else None)

        loss_xy = masked_huber_loss(point_abs_metric[:, :, 0:2], gt_point_map_metric[:, :, 0:2], mask=height_valid_mask, beta=self.beta)
        loss_z = masked_huber_loss(point_abs_metric[:, :, 2:3], gt_point_map_metric[:, :, 2:3], mask=height_valid_mask, beta=self.beta)
        loss = loss_xy + loss_z

        d = point_abs_metric - gt_point_map_metric
        dx2 = d[:, :, 0:1].square()
        dy2 = d[:, :, 1:2].square()
        dz2 = d[:, :, 2:3].square()

        norm_xyz = torch.sqrt((d.square().sum(dim=2, keepdim=True)).clamp_min(1e-8))
        anchor_delta = point_abs_metric - point_anchor_metric
        anchor_norm = torch.sqrt((anchor_delta.square().sum(dim=2, keepdim=True)).clamp_min(1e-8))

        probe = {
            "point_xy_meter_loss": loss_xy.detach(),
            "point_z_meter_loss": loss_z.detach(),
            "point_xyz_rmse": safe_rmse(dx2 + dy2 + dz2, mask=height_valid_mask).detach(),
            "point_xy_rmse": safe_rmse(dx2 + dy2, mask=height_valid_mask).detach(),
            "point_z_rmse": safe_rmse(dz2, mask=height_valid_mask).detach(),
            "point_xyz_mae": masked_reduce(d.abs().mean(dim=2, keepdim=True), mask=height_valid_mask, reduce="mean").detach(),
            "point_anchor_displacement_mean": masked_reduce(anchor_norm, mask=height_valid_mask, reduce="mean").detach(),
            "point_anchor_displacement_z_mean": masked_reduce(anchor_delta[:, :, 2:3].abs(), mask=height_valid_mask, reduce="mean").detach(),
        }

        aux: dict[str, Any] = {}
        aux["loss_point_xy_meter"] = loss_xy
        aux["loss_point_z_meter"] = loss_z
        aux["loss_point_meter"] = loss
        if return_aux:
            aux["gt_point_map_metric"] = gt_point_map_metric
            aux["gt_point_map"] = gt_point_map_metric
        return loss, probe, aux

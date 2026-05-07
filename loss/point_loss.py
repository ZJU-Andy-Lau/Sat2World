"""Point supervision losses for normalized raw lat/lon plane coordinates."""

from __future__ import annotations

from typing import Any

import torch

from loss.common import masked_huber_loss, masked_l1_loss, masked_reduce, safe_rmse


class PointLatLonLoss:
    """Plane-coordinate point loss in normalized raw lat/lon space.

    The point head predicts ``point_latlon_norm`` with shape [B,V,2,H,W]
    (lat_norm/lon_norm).  GT is built from ``rpc_gt + height_gt`` by direct raw
    lat/lon inverse projection and normalized with batch ``scene_latlon_*``.
    Metric-plane errors are approximate logging metrics only and are not used as
    optimization targets.
    """

    def __init__(self, geometry_ops: Any, beta: float = 1.0) -> None:
        self.geometry_ops = geometry_ops
        self.beta = float(beta)
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
        point_latlon_norm: torch.Tensor,
        point_latlon_anchor: torch.Tensor,
        batch: dict[str, Any],
        return_aux: bool = False,
        aux_include_full_gt_map: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        del aux_include_full_gt_map
        if point_latlon_norm.ndim != 5 or point_latlon_norm.shape[2] != 2:
            raise ValueError(f"point_latlon_norm must be [B,V,2,H,W], got {tuple(point_latlon_norm.shape)}")
        device = point_latlon_norm.device
        dtype = point_latlon_norm.dtype
        b, v, _, h, w = point_latlon_norm.shape
        height_gt = batch["height_gt"].to(device=device, dtype=dtype)
        mask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        center = batch["scene_latlon_center"].to(device=device, dtype=dtype)
        scale = batch["scene_latlon_scale"].to(device=device, dtype=dtype).clamp_min(1e-12)
        grid = self._get_grid(h, w, device, dtype)
        line = grid[..., 0].reshape(-1).view(1, 1, -1).expand(b, v, -1)
        samp = grid[..., 1].reshape(-1).view(1, 1, -1).expand(b, v, -1)
        heights = height_gt[:, :, 0].reshape(b, v, -1)
        lat, lon = self.geometry_ops.linesamp_to_latlon_batch(batch["rpc_gt"], line, samp, heights)
        lat = lat.to(device=device, dtype=dtype).view(b, v, h, w)
        lon = lon.to(device=device, dtype=dtype).view(b, v, h, w)
        gt = torch.stack(
            [
                (lat - center[:, 0].view(b, 1, 1, 1)) / scale[:, 0].view(b, 1, 1, 1),
                (lon - center[:, 1].view(b, 1, 1, 1)) / scale[:, 1].view(b, 1, 1, 1),
            ],
            dim=2,
        )
        loss = masked_huber_loss(point_latlon_norm, gt, mask=mask, beta=self.beta)
        diff = point_latlon_norm.detach() - gt.detach()
        norm_l1 = masked_l1_loss(point_latlon_norm.detach(), gt.detach(), mask=mask)
        norm_rmse = safe_rmse(diff.square().sum(dim=2, keepdim=True), mask=mask)
        lat_center_rad = torch.deg2rad(center[:, 0].view(b, 1, 1, 1))
        dlat_m = diff[:, :, 0:1] * scale[:, 0].view(b, 1, 1, 1, 1) * 111320.0
        dlon_m = diff[:, :, 1:2] * scale[:, 1].view(b, 1, 1, 1, 1) * 111320.0 * torch.cos(lat_center_rad).view(b, 1, 1, 1, 1)
        plane_m = torch.sqrt((dlat_m.square() + dlon_m.square()).clamp_min(1e-8))
        anchor_delta = point_latlon_norm.detach() - point_latlon_anchor.detach()
        probe = {
            "point_latlon_norm_loss": loss.detach(),
            "point_latlon_norm_rmse": norm_rmse.detach(),
            "point_latlon_norm_mae": norm_l1.detach(),
            "point_plane_error_m_mean": masked_reduce(plane_m, mask=mask, reduce="mean").detach(),
            "point_plane_error_m_rmse": safe_rmse(plane_m.square(), mask=mask).detach(),
            "point_latlon_anchor_displacement_mean": masked_reduce(anchor_delta.abs().mean(dim=2, keepdim=True), mask=mask, reduce="mean").detach(),
        }
        aux: dict[str, Any] = {"loss_point_latlon_norm": loss}
        if return_aux:
            aux["gt_point_latlon_norm"] = gt.detach()
        return loss, probe, aux

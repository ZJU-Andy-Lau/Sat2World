"""loss.point_pair_loss

跨视图点云一致性损失：
- 在源视图上均匀采样 K 个像素点（默认 64x64）；
- 使用 rpc_gt + h_gt 建立源视图到参考视图的对应关系；
- 点云头只提供归一化 raw lat/lon 平面坐标；高程来自 height branch。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from loss.common import sample_map_bilinear


@dataclass
class PointPairwiseLossCfg:
    """点云跨视一致性损失配置。"""

    grid_h: int = 64
    grid_w: int = 64


class PointPairwiseConsistencyLoss:
    """Cross-view consistency in normalized lat/lon plane coordinates.

    The point branch no longer predicts height.  Correspondences are still found
    with rpc_gt + height_gt, but the consistency distance is computed only in the
    normalized lat/lon plane sampled from ``point_latlon_norm``.
    """

    def __init__(self, geometry_ops: Any, cfg: PointPairwiseLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or PointPairwiseLossCfg()
        self._grid_cache: dict[tuple[int, int, int, int, str, str], torch.Tensor] = {}

    def _get_uniform_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, int(self.cfg.grid_h), int(self.cfg.grid_w), str(device), str(dtype))
        if key in self._grid_cache:
            return self._grid_cache[key]
        ys = torch.linspace(0.0, float(max(h - 1, 0)), steps=int(self.cfg.grid_h), device=device, dtype=dtype)
        xs = torch.linspace(0.0, float(max(w - 1, 0)), steps=int(self.cfg.grid_w), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        self._grid_cache[key] = pts
        return pts

    @staticmethod
    def _expand_ref_idx(ref_view_idx: torch.Tensor | None, b: int, v: int, device: torch.device) -> torch.Tensor:
        if ref_view_idx is None:
            return torch.zeros((b,), dtype=torch.long, device=device)
        ref = ref_view_idx.long().to(device=device).view(-1)
        if ref.numel() == 1:
            ref = ref.expand(b)
        return ref.clamp(0, v - 1)

    def __call__(self, point_latlon_norm: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        b, v, c, h, w = point_latlon_norm.shape
        if c != 2:
            raise ValueError("point_latlon_norm must be [B,V,2,H,W]")
        device = point_latlon_norm.device
        dtype = point_latlon_norm.dtype
        ref_idx = self._expand_ref_idx(batch.get("ref_view_idx", None), b, v, device)
        rpc_gt = batch["rpc_gt"]
        height_gt = batch["height_gt"].to(device=device, dtype=dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        grid = self._get_uniform_grid(h, w, device=device, dtype=dtype).view(1, -1, 2)
        pair_losses: list[torch.Tensor] = []
        valid_pairs = 0
        for bi in range(b):
            ref = int(ref_idx[bi].item())
            ref_map = point_latlon_norm[bi : bi + 1, ref]
            for vi in range(v):
                if vi == ref:
                    continue
                src_map = point_latlon_norm[bi : bi + 1, vi]
                h_src, in_h = sample_map_bilinear(height_gt[bi : bi + 1, vi], grid)
                m_src, in_m = sample_map_bilinear(height_valid_mask[bi : bi + 1, vi], grid)
                valid_src = in_h & in_m & (m_src[:, 0] > 0.5)
                if not bool(valid_src.any()):
                    continue
                src_pts = grid[:, valid_src[0]]
                h_vals = h_src[:, 0][:, valid_src[0]]
                lat, lon = self.geometry_ops.linesamp_to_latlon_batch(
                    rpc_batch=[[rpc_gt[bi][vi]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
                    heights=h_vals.view(1, 1, -1),
                )
                l_ref, s_ref = self.geometry_ops.latlon_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][ref]]],
                    lats=lat,
                    lons=lon,
                    heights=h_vals.view(1, 1, -1),
                )
                ref_pts = torch.stack([l_ref.view(1, -1), s_ref.view(1, -1)], dim=-1).to(device=device, dtype=dtype)
                src_feat, _ = sample_map_bilinear(src_map, src_pts)
                ref_feat, in_ref = sample_map_bilinear(ref_map, ref_pts)
                if not bool(in_ref.any()):
                    continue
                p_src = src_feat[0].transpose(0, 1)[in_ref[0]]
                p_ref = ref_feat[0].transpose(0, 1)[in_ref[0]]
                if p_src.numel() == 0:
                    continue
                pair_losses.append(torch.linalg.norm(p_src - p_ref, dim=-1).mean())
                valid_pairs += 1
        if len(pair_losses) == 0:
            zero = point_latlon_norm.sum() * 0.0
            zero_detached = torch.zeros((), device=device, dtype=dtype)
            return zero, {"point_pair_latlon_norm_dist_mean": zero_detached, "point_pair_num_pairs_used": zero_detached}, {"point_pair_num_pairs_used": 0}
        loss = torch.stack(pair_losses).mean()
        probe = {
            "point_pair_latlon_norm_dist_mean": loss.detach(),
            "point_pair_num_pairs_used": torch.tensor(float(valid_pairs), device=device, dtype=dtype),
        }
        return loss, probe, {"point_pair_num_pairs_used": valid_pairs}


class PointReprojectionLoss:
    """Point-plane reprojection using point_latlon_norm + height_abs.

    The point head supplies only lat/lon; height is provided by the independent
    height branch.  Normalized lat/lon is unnormalized with scene_latlon_* before
    projection through target-view rpc_gt.
    """

    def __init__(self, geometry_ops: Any, cfg: PointPairwiseLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or PointPairwiseLossCfg()
        self._grid_cache: dict[tuple[int, int, int, int, str, str], torch.Tensor] = {}

    def _get_uniform_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, int(self.cfg.grid_h), int(self.cfg.grid_w), str(device), str(dtype))
        if key in self._grid_cache:
            return self._grid_cache[key]
        ys = torch.linspace(0.0, float(max(h - 1, 0)), steps=int(self.cfg.grid_h), device=device, dtype=dtype)
        xs = torch.linspace(0.0, float(max(w - 1, 0)), steps=int(self.cfg.grid_w), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        self._grid_cache[key] = pts
        return pts

    def __call__(
        self,
        point_latlon_norm: torch.Tensor,
        height_abs: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        b, v, c, h, w = point_latlon_norm.shape
        if c != 2:
            raise ValueError("point_latlon_norm must be [B,V,2,H,W]")
        device = point_latlon_norm.device
        dtype = point_latlon_norm.dtype
        if v < 2:
            zero = (point_latlon_norm.sum() + height_abs.sum()) * 0.0
            zero_detached = torch.zeros((), device=device, dtype=dtype)
            return zero, {"point_reproj_px_mean": zero_detached, "point_reproj_num_pairs_used": zero_detached}, {"point_reproj_num_pairs_used": 0}
        rpc_gt = batch["rpc_gt"]
        height_gt = batch["height_gt"].to(device=device, dtype=dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        center = batch["scene_latlon_center"].to(device=device, dtype=dtype)
        scale = batch["scene_latlon_scale"].to(device=device, dtype=dtype).clamp_min(1e-12)
        grid = self._get_uniform_grid(h, w, device=device, dtype=dtype).view(1, -1, 2)
        losses: list[torch.Tensor] = []
        num_pairs_used = 0
        for bi in range(b):
            c_b = center[bi]
            s_b = scale[bi]
            for i in range(v):
                j = (i + 1) % v
                h_i, in_h = sample_map_bilinear(height_gt[bi : bi + 1, i], grid)
                m_i, in_m = sample_map_bilinear(height_valid_mask[bi : bi + 1, i], grid)
                h_pred_i, in_hp = sample_map_bilinear(height_abs[bi : bi + 1, i], grid)
                p_i, in_p = sample_map_bilinear(point_latlon_norm[bi : bi + 1, i], grid)
                valid_i = in_h & in_m & in_hp & in_p & (m_i[:, 0] > 0.5)
                if not bool(valid_i.any()):
                    continue
                src_pts = grid[:, valid_i[0]]
                h_i_gt = h_i[:, 0][:, valid_i[0]]
                h_i_pred = h_pred_i[:, 0][:, valid_i[0]]
                lat_gt, lon_gt = self.geometry_ops.linesamp_to_latlon_batch(
                    rpc_batch=[[rpc_gt[bi][i]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
                    heights=h_i_gt.view(1, 1, -1),
                )
                l_gt, s_gt = self.geometry_ops.latlon_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    lats=lat_gt,
                    lons=lon_gt,
                    heights=h_i_gt.view(1, 1, -1),
                )
                point_gt = torch.stack([l_gt.view(-1), s_gt.view(-1)], dim=-1)
                p_norm = p_i[0].transpose(0, 1)[valid_i[0]]
                lat_pred = p_norm[:, 0].view(1, 1, -1) * s_b[0] + c_b[0]
                lon_pred = p_norm[:, 1].view(1, 1, -1) * s_b[1] + c_b[1]
                l_proj, s_proj = self.geometry_ops.latlon_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    lats=lat_pred,
                    lons=lon_pred,
                    heights=h_i_pred.view(1, 1, -1),
                )
                point_proj = torch.stack([l_proj.view(-1), s_proj.view(-1)], dim=-1)
                finite = torch.isfinite(point_proj).all(dim=-1) & torch.isfinite(point_gt).all(dim=-1)
                in_bound = (
                    (point_proj[:, 0] >= 0.0) & (point_proj[:, 0] <= float(max(h - 1, 0)))
                    & (point_proj[:, 1] >= 0.0) & (point_proj[:, 1] <= float(max(w - 1, 0)))
                    & (point_gt[:, 0] >= 0.0) & (point_gt[:, 0] <= float(max(h - 1, 0)))
                    & (point_gt[:, 1] >= 0.0) & (point_gt[:, 1] <= float(max(w - 1, 0)))
                )
                keep = finite & in_bound
                if not bool(keep.any()):
                    continue
                losses.append(torch.linalg.norm(point_proj[keep] - point_gt[keep], dim=-1).mean())
                num_pairs_used += 1
        if len(losses) == 0:
            zero = (point_latlon_norm.sum() + height_abs.sum()) * 0.0
            zero_detached = torch.zeros((), device=device, dtype=dtype)
            return zero, {"point_reproj_px_mean": zero_detached, "point_reproj_num_pairs_used": zero_detached}, {"point_reproj_num_pairs_used": 0}
        loss = torch.stack(losses).mean()
        probe = {
            "point_reproj_px_mean": loss.detach(),
            "point_reproj_num_pairs_used": torch.tensor(float(num_pairs_used), device=device, dtype=dtype),
        }
        return loss, probe, {"point_reproj_num_pairs_used": int(num_pairs_used)}


class HeightReprojectionLoss:
    """高程重投影损失。

    定义：
    - 对每个视图 i，选择 j=(i+1)%V；
    - 使用 rpc_gt + h_pred(i) 从 src 像素采样点构造世界点，再投影到 view_j 得到 point_proj；
    - 使用 rpc_gt + h_gt(i) 生成同一 src 采样点在 view_j 的 GT 投影 point_gt；
    - 以两者像素距离作为重投影误差。
    """

    def __init__(self, geometry_ops: Any, cfg: PointPairwiseLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or PointPairwiseLossCfg()
        self._grid_cache: dict[tuple[int, int, int, int, str, str], torch.Tensor] = {}

    def _get_uniform_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, int(self.cfg.grid_h), int(self.cfg.grid_w), str(device), str(dtype))
        if key in self._grid_cache:
            return self._grid_cache[key]
        ys = torch.linspace(0.0, float(max(h - 1, 0)), steps=int(self.cfg.grid_h), device=device, dtype=dtype)
        xs = torch.linspace(0.0, float(max(w - 1, 0)), steps=int(self.cfg.grid_w), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
        self._grid_cache[key] = pts
        return pts

    def __call__(self, height_abs: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        b, v, _, h, w = height_abs.shape
        device = height_abs.device
        dtype = height_abs.dtype
        if v < 2:
            zero = height_abs.sum() * 0.0
            zero_detached = torch.zeros((), device=device, dtype=dtype)
            return zero, {"height_reproj_px_mean": zero_detached, "height_reproj_num_pairs_used": zero_detached}, {"height_reproj_num_pairs_used": 0}

        rpc_gt = batch["rpc_gt"]
        height_gt = batch["height_gt"].to(device=device, dtype=dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        if scene_xy_scale is not None and torch.is_tensor(scene_xy_scale):
            scene_xy_scale = scene_xy_scale.to(device=device, dtype=dtype)

        grid = self._get_uniform_grid(h, w, device=device, dtype=dtype).view(1, -1, 2)
        losses: list[torch.Tensor] = []
        num_pairs_used = 0

        for bi in range(b):
            for i in range(v):
                j = (i + 1) % v
                h_gt_i, in_h = sample_map_bilinear(height_gt[bi : bi + 1, i], grid)
                m_i, in_m = sample_map_bilinear(height_valid_mask[bi : bi + 1, i], grid)
                h_pred_i, in_hp = sample_map_bilinear(height_abs[bi : bi + 1, i], grid)
                valid_i = in_h & in_m & in_hp & (m_i[:, 0] > 0.5)
                if not bool(valid_i.any()):
                    continue
                src_pts = grid[:, valid_i[0]]
                h_i_gt = h_gt_i[:, 0][:, valid_i[0]]
                h_i_pred = h_pred_i[:, 0][:, valid_i[0]]

                # GT world(来自 rpc_gt + h_gt) -> view_j
                xs_gt, ys_gt = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][i]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
                    heights=h_i_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                l_gt, s_gt = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    xs=xs_gt,
                    ys=ys_gt,
                    heights=h_i_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                point_gt = torch.stack([l_gt.view(-1), s_gt.view(-1)], dim=-1)

                # 预测高程(view_i) -> world -> view_j
                xs_pred, ys_pred = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][i]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
                    heights=h_i_pred.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                l_proj, s_proj = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    xs=xs_pred,
                    ys=ys_pred,
                    heights=h_i_pred.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                point_proj = torch.stack([l_proj.view(-1), s_proj.view(-1)], dim=-1)

                finite = torch.isfinite(point_proj).all(dim=-1) & torch.isfinite(point_gt).all(dim=-1)
                in_bound = (
                    (point_proj[:, 0] >= 0.0)
                    & (point_proj[:, 0] <= float(max(h - 1, 0)))
                    & (point_proj[:, 1] >= 0.0)
                    & (point_proj[:, 1] <= float(max(w - 1, 0)))
                    & (point_gt[:, 0] >= 0.0)
                    & (point_gt[:, 0] <= float(max(h - 1, 0)))
                    & (point_gt[:, 1] >= 0.0)
                    & (point_gt[:, 1] <= float(max(w - 1, 0)))
                )
                keep = finite & in_bound
                if not bool(keep.any()):
                    continue
                dist = torch.linalg.norm(point_proj[keep] - point_gt[keep], dim=-1)
                losses.append(dist.mean())
                num_pairs_used += 1

        if len(losses) == 0:
            zero = height_abs.sum() * 0.0
            zero_detached = torch.zeros((), device=device, dtype=dtype)
            return zero, {"height_reproj_px_mean": zero_detached, "height_reproj_num_pairs_used": zero_detached}, {"height_reproj_num_pairs_used": 0}

        loss = torch.stack(losses).mean()
        probe = {
            "height_reproj_px_mean": loss.detach(),
            "height_reproj_num_pairs_used": torch.tensor(float(num_pairs_used), device=device, dtype=dtype),
        }
        aux = {"height_reproj_num_pairs_used": int(num_pairs_used)}
        return loss, probe, aux

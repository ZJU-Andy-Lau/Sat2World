"""loss.point_pair_loss

跨视图点云一致性损失：
- 在源视图上均匀采样 K 个像素点（默认 64x64）；
- 使用 rpc_gt + h_gt 建立源视图到参考视图的对应关系；
- 取 point_abs 在源点与参考投影点的三维点，计算欧氏距离并求平均。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from loss.common import sample_map_bilinear


def point_map_to_metric(point_map: torch.Tensor, scene_xy_scale: torch.Tensor | None) -> torch.Tensor:
    """把点云图从 (x_norm, y_norm, h_m) 转为统一米制近似坐标。

    约定:
    - point_map 通道顺序为 (x, y, h)；
    - scene_xy_scale 顺序为 (y, x)。
    """
    if scene_xy_scale is None:
        return point_map
    if scene_xy_scale.ndim != 2 or scene_xy_scale.shape[-1] != 2:
        raise ValueError(f"scene_xy_scale must be [B,2], got {tuple(scene_xy_scale.shape)}")

    out = point_map.clone()
    scale_y = scene_xy_scale[:, 0].to(device=point_map.device, dtype=point_map.dtype).view(-1, 1, 1, 1, 1)
    scale_x = scene_xy_scale[:, 1].to(device=point_map.device, dtype=point_map.dtype).view(-1, 1, 1, 1, 1)
    out[:, :, 0:1] = out[:, :, 0:1] * scale_x
    out[:, :, 1:2] = out[:, :, 1:2] * scale_y
    return out


@dataclass
class PointPairwiseLossCfg:
    """点云跨视一致性损失配置。"""

    grid_h: int = 64
    grid_w: int = 64


class PointPairwiseConsistencyLoss:
    """Image_i 与 Image_ref 的对应点三维欧氏距离损失。"""

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

    def __call__(
        self,
        point_abs: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        b, v, _, h, w = point_abs.shape
        device = point_abs.device
        dtype = point_abs.dtype
        ref_idx = self._expand_ref_idx(batch.get("ref_view_idx", None), b, v, device)
        rpc_gt = batch["rpc_gt"]
        height_gt = batch["height_gt"].to(device=device, dtype=dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=device, dtype=dtype)
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        if scene_xy_scale is not None and torch.is_tensor(scene_xy_scale):
            scene_xy_scale = scene_xy_scale.to(device=device, dtype=dtype)

        point_metric = point_map_to_metric(point_abs, scene_xy_scale if torch.is_tensor(scene_xy_scale) else None)

        grid = self._get_uniform_grid(h, w, device=device, dtype=dtype).view(1, -1, 2)
        pair_losses: list[torch.Tensor] = []
        pair_dist_means: list[torch.Tensor] = []
        valid_pairs = 0

        for bi in range(b):
            ref = int(ref_idx[bi].item())
            ref_map = point_metric[bi : bi + 1, ref]

            for vi in range(v):
                if vi == ref:
                    continue
                src_map = point_metric[bi : bi + 1, vi]

                h_src, in_h = sample_map_bilinear(height_gt[bi : bi + 1, vi], grid)
                m_src, in_m = sample_map_bilinear(height_valid_mask[bi : bi + 1, vi], grid)
                valid_src = in_h & in_m & (m_src[:, 0] > 0.5)
                if not bool(valid_src.any()):
                    continue

                src_pts = grid[:, valid_src[0]]
                h_vals = h_src[:, 0][:, valid_src[0]]

                xs, ys = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][vi]]],
                    lines=src_pts[..., 0].view(1, 1, -1),
                    samps=src_pts[..., 1].view(1, 1, -1),
                    heights=h_vals.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                l_ref, s_ref = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][ref]]],
                    xs=xs,
                    ys=ys,
                    heights=h_vals.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
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

                dist = torch.linalg.norm(p_src - p_ref, dim=-1)
                pair_losses.append(dist.mean())
                pair_dist_means.append(dist.mean().detach())
                valid_pairs += 1

        if len(pair_losses) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, {"point_pair_dist_mean": zero, "point_pair_num_pairs_used": zero}, {"point_pair_num_pairs_used": 0}

        loss = torch.stack(pair_losses).mean()
        dist_mean = torch.stack(pair_dist_means).mean()
        probe = {
            "point_pair_dist_mean": dist_mean,
            "point_pair_num_pairs_used": torch.tensor(float(valid_pairs), device=device, dtype=dtype),
        }
        aux = {"point_pair_num_pairs_used": valid_pairs}
        return loss, probe, aux

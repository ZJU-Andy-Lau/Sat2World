"""loss.height_pair_loss

跨视图高程相对一致性损失：
1) 同一物点在两视图预测高程的一致性（Huber）；
2) 基于预测高程的跨视图投影闭环一致性（像平面误差）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from loss.common import make_uniform_grid_points, masked_huber_loss, pairwise_view_pairs, safe_rmse, sample_map_bilinear


@dataclass
class HeightPairwiseLossCfg:
    """高程相对损失配置。"""

    anchors_per_pair: int = 256
    max_pairs: int | None = None
    sample_from_valid_only: bool = True
    beta: float = 1.0
    lambda_consistency: float = 1.0
    lambda_cycle: float = 1.0


class HeightPairwiseRelativeLoss:
    """高程相对损失（双向 i<->j）。"""

    def __init__(self, geometry_ops: Any, cfg: HeightPairwiseLossCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or HeightPairwiseLossCfg()
        self._anchor_cache: dict[tuple[int, int, int, str, str], torch.Tensor] = {}

    def _get_anchor_points(self, h: int, w: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, n, str(device), str(dtype))
        if key not in self._anchor_cache:
            gh = int(max(1, round(n**0.5)))
            gw = int(max(1, (n + gh - 1) // gh))
            grid = make_uniform_grid_points(h, w, gh, gw, device=device, dtype=dtype)
            if grid.shape[0] > n:
                grid = grid[:n]
            self._anchor_cache[key] = grid
        return self._anchor_cache[key]

    def _dir_loss(
        self,
        *,
        bi: int,
        src: int,
        tgt: int,
        anchors_src: torch.Tensor,  # [1,N,2] true-domain points on src
        h_src_gt: torch.Tensor,  # [1,N]
        rpc_gt: Any,
        height_abs: torch.Tensor,  # [B,V,1,H,W]
        height_valid_mask: torch.Tensor,  # [B,V,1,H,W]
        scene_xy_center: torch.Tensor | None,
        scene_xy_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        device, dtype = height_abs.device, height_abs.dtype

        # 过滤源视图有效 anchor
        m_src, in_src = sample_map_bilinear(height_valid_mask[bi : bi + 1, src], anchors_src)
        keep_src = in_src & (m_src[:, 0] > 0.5)
        if keep_src.sum() == 0:
            return None, None, 0
        anchors_src = anchors_src[:, keep_src[0]]
        h_src_gt = h_src_gt[:, keep_src[0]]

        # rpc_gt + h_gt 建立跨视图 GT 对应点：anchor_src -> anchor_src2tgt_gt
        xs, ys = self.geometry_ops.linesamp_to_xy_batch(
            rpc_batch=[[rpc_gt[bi][src]]],
            lines=anchors_src[..., 0].view(1, 1, -1),
            samps=anchors_src[..., 1].view(1, 1, -1),
            heights=h_src_gt.view(1, 1, -1),
            scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
            scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
        )
        l_t, s_t = self.geometry_ops.xy_to_linesamp_batch(
            rpc_batch=[[rpc_gt[bi][tgt]]],
            xs=xs,
            ys=ys,
            heights=h_src_gt.view(1, 1, -1),
            scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
            scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
        )
        anchors_tgt = torch.stack([l_t.view(1, -1), s_t.view(1, -1)], dim=-1).to(device=device, dtype=dtype)

        # 过滤目标视图可采样与有效区域
        m_tgt, in_tgt = sample_map_bilinear(height_valid_mask[bi : bi + 1, tgt], anchors_tgt)
        keep = in_tgt & (m_tgt[:, 0] > 0.5)
        if keep.sum() == 0:
            return None, None, 0
        anchors_src = anchors_src[:, keep[0]]
        anchors_tgt = anchors_tgt[:, keep[0]]

        # (1) 同物点高程一致性：h_i(anchor) vs h_j(anchor_i2j_gt)
        h_src_pred, in_hs = sample_map_bilinear(height_abs[bi : bi + 1, src], anchors_src)
        h_tgt_pred, in_ht = sample_map_bilinear(height_abs[bi : bi + 1, tgt], anchors_tgt)
        keep_h = in_hs & in_ht
        if keep_h.sum() == 0:
            return None, None, 0
        anchors_src = anchors_src[:, keep_h[0]]
        anchors_tgt = anchors_tgt[:, keep_h[0]]
        h_src_pred = h_src_pred[:, 0, keep_h[0]]
        h_tgt_pred = h_tgt_pred[:, 0, keep_h[0]]

        loss_cons = masked_huber_loss(h_src_pred, h_tgt_pred, mask=None, beta=float(self.cfg.beta))

        # (2) 预测高程驱动闭环：anchor_tgt -> world(h_tgt_pred) -> src(h_src_pred)
        xs2, ys2 = self.geometry_ops.linesamp_to_xy_batch(
            rpc_batch=[[rpc_gt[bi][tgt]]],
            lines=anchors_tgt[..., 0].view(1, 1, -1),
            samps=anchors_tgt[..., 1].view(1, 1, -1),
            heights=h_tgt_pred.view(1, 1, -1),
            scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
            scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
        )
        l_back, s_back = self.geometry_ops.xy_to_linesamp_batch(
            rpc_batch=[[rpc_gt[bi][src]]],
            xs=xs2,
            ys=ys2,
            heights=h_src_pred.view(1, 1, -1),
            scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
            scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
        )
        anchors_back = torch.stack([l_back.view(1, -1), s_back.view(1, -1)], dim=-1).to(device=device, dtype=dtype)
        cycle_err = torch.linalg.norm(anchors_back - anchors_src, dim=-1)
        loss_cycle = cycle_err.mean()
        return loss_cons, loss_cycle, int(cycle_err.numel())

    def __call__(
        self,
        height_abs: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        rpc_gt = batch["rpc_gt"]
        height_gt = batch["height_gt"].to(device=height_abs.device, dtype=height_abs.dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=height_abs.device, dtype=height_abs.dtype)
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)

        b, v, _, h, w = height_abs.shape
        use_precomputed_anchors = ("anchor_line_samp_true" in batch and "anchor_height_true" in batch)
        if use_precomputed_anchors:
            anchor_v = int(batch["anchor_line_samp_true"].shape[1])
            anchor_h_v = int(batch["anchor_height_true"].shape[1])
            if anchor_v != v or anchor_h_v != v:
                use_precomputed_anchors = False

        cons_terms: list[torch.Tensor] = []
        cycle_terms: list[torch.Tensor] = []
        pairs_used = 0
        anchors_used = 0

        for bi in range(b):
            pairs = pairwise_view_pairs(v, max_pairs=self.cfg.max_pairs)
            for i, j in pairs:
                if use_precomputed_anchors:
                    a_i = batch["anchor_line_samp_true"][bi : bi + 1, i].to(device=height_abs.device, dtype=height_abs.dtype)
                    a_j = batch["anchor_line_samp_true"][bi : bi + 1, j].to(device=height_abs.device, dtype=height_abs.dtype)
                    h_i_gt = batch["anchor_height_true"][bi : bi + 1, i].to(device=height_abs.device, dtype=height_abs.dtype)
                    h_j_gt = batch["anchor_height_true"][bi : bi + 1, j].to(device=height_abs.device, dtype=height_abs.dtype)
                else:
                    a_i = self._get_anchor_points(h, w, int(self.cfg.anchors_per_pair), height_abs.device, height_abs.dtype).unsqueeze(0)
                    a_j = self._get_anchor_points(h, w, int(self.cfg.anchors_per_pair), height_abs.device, height_abs.dtype).unsqueeze(0)
                    h_i_gt, _ = sample_map_bilinear(height_gt[bi : bi + 1, i], a_i)
                    h_j_gt, _ = sample_map_bilinear(height_gt[bi : bi + 1, j], a_j)
                    h_i_gt = h_i_gt[:, 0]
                    h_j_gt = h_j_gt[:, 0]

                c_ij, y_ij, n_ij = self._dir_loss(
                    bi=bi,
                    src=i,
                    tgt=j,
                    anchors_src=a_i,
                    h_src_gt=h_i_gt,
                    rpc_gt=rpc_gt,
                    height_abs=height_abs,
                    height_valid_mask=height_valid_mask,
                    scene_xy_center=scene_xy_center,
                    scene_xy_scale=scene_xy_scale,
                )
                c_ji, y_ji, n_ji = self._dir_loss(
                    bi=bi,
                    src=j,
                    tgt=i,
                    anchors_src=a_j,
                    h_src_gt=h_j_gt,
                    rpc_gt=rpc_gt,
                    height_abs=height_abs,
                    height_valid_mask=height_valid_mask,
                    scene_xy_center=scene_xy_center,
                    scene_xy_scale=scene_xy_scale,
                )

                valid_dirs = 0
                if c_ij is not None and y_ij is not None:
                    cons_terms.append(c_ij)
                    cycle_terms.append(y_ij)
                    anchors_used += n_ij
                    valid_dirs += 1
                if c_ji is not None and y_ji is not None:
                    cons_terms.append(c_ji)
                    cycle_terms.append(y_ji)
                    anchors_used += n_ji
                    valid_dirs += 1
                if valid_dirs > 0:
                    pairs_used += 1

        zero = torch.zeros((), device=height_abs.device, dtype=height_abs.dtype)
        if len(cons_terms) == 0 or len(cycle_terms) == 0:
            probe = {
                "height_rel_consistency": zero,
                "height_rel_cycle_px": zero,
                "height_rel_cycle_px_rmse": zero,
                "height_rel_num_pairs_used": zero,
            }
            aux = {"height_rel_pairs_used": 0, "height_rel_anchors_used": 0}
            return zero, probe, aux

        l_cons = torch.stack(cons_terms).mean()
        l_cycle = torch.stack(cycle_terms).mean()
        total = float(self.cfg.lambda_consistency) * l_cons + float(self.cfg.lambda_cycle) * l_cycle

        cycle_sq = torch.stack([x * x for x in cycle_terms], dim=0)
        probe = {
            "height_rel_consistency": l_cons.detach(),
            "height_rel_cycle_px": l_cycle.detach(),
            "height_rel_cycle_px_rmse": safe_rmse(cycle_sq).detach(),
            "height_rel_num_pairs_used": torch.tensor(float(pairs_used), device=height_abs.device, dtype=height_abs.dtype),
        }
        aux = {"height_rel_pairs_used": int(pairs_used), "height_rel_anchors_used": int(anchors_used)}
        return total, probe, aux


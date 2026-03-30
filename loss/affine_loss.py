"""loss.affine_loss

实现仿射相关三类损失：
1) AffineGridLoss
2) AffinePairwiseGeometryLoss
3) AffineLinearRegularization

方向约定（必须保持）：
- affine_gt_forward: true pixel -> observed pixel
- affine_pred:       observed pixel -> true pixel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from loss.common import (
    apply_affine_to_points,
    make_uniform_grid_points,
    pairwise_view_pairs,
    sample_map_bilinear,
    safe_rmse,
)


@dataclass
class AffineGridLossCfg:
    """AffineGridLoss 配置。"""

    grid_h: int = 16
    grid_w: int = 16


class AffineGridLoss:
    """仿射网格恢复损失。

    功能:
        在 true 坐标网格上先施加 GT forward 扰动（true->observed），
        再施加模型 correction（observed->true），期望恢复原坐标。

    成员变量:
        cfg: 网格采样配置。
        _grid_cache: 按 (H,W,grid_h,grid_w,device,dtype) 缓存网格。
    """

    def __init__(self, cfg: AffineGridLossCfg | None = None) -> None:
        """初始化损失类。"""
        self.cfg = cfg or AffineGridLossCfg()
        self._grid_cache: dict[tuple[int, int, int, int, str, str], torch.Tensor] = {}

    def _get_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取或创建均匀采样网格 [N,2]。"""
        key = (h, w, self.cfg.grid_h, self.cfg.grid_w, str(device), str(dtype))
        if key not in self._grid_cache:
            self._grid_cache[key] = make_uniform_grid_points(h, w, self.cfg.grid_h, self.cfg.grid_w, device, dtype)
        return self._grid_cache[key]

    def __call__(
        self,
        affine_pred: torch.Tensor,
        affine_gt_forward: torch.Tensor,
        image_hw: tuple[int, int],
        ref_view_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算 AffineGridLoss。

        输入:
            affine_pred: [B,V,2,3]，correction affine（observed->true）。
            affine_gt_forward: [B,V,2,3]，forward 扰动（true->observed）。
            image_hw: (H,W)。
            ref_view_idx: [B] 或 None，参考视图索引。

        输出:
            loss: 标量。
            probe: 包含平均误差、rmse、参考视图误差。
        """
        if affine_pred.shape != affine_gt_forward.shape:
            raise ValueError("affine_pred and affine_gt_forward must share shape")
        b, v = affine_pred.shape[:2]
        h, w = image_hw
        grid = self._get_grid(h, w, affine_pred.device, affine_pred.dtype)
        n = grid.shape[0]

        g_true = grid.view(1, 1, n, 2).expand(b, v, n, 2)
        g_obs = apply_affine_to_points(g_true, affine_gt_forward)
        g_rec = apply_affine_to_points(g_obs, affine_pred)

        err = torch.linalg.norm(g_rec - g_true, dim=-1)  # [B,V,N]

        view_mask = torch.ones((b, v, 1), device=err.device, dtype=err.dtype)
        if ref_view_idx is not None:
            ref = ref_view_idx.long().view(-1)
            if ref.numel() == 1:
                ref = ref.expand(b)
            view_mask[torch.arange(b, device=err.device), ref, 0] = 0.0

        view_mask_exp = view_mask.expand_as(err)
        loss = (err * view_mask_exp).sum() / view_mask_exp.sum().clamp_min(1.0)

        ref_err_mean = torch.zeros((), device=err.device, dtype=err.dtype)
        if ref_view_idx is not None:
            ref_err = err[torch.arange(b, device=err.device), ref]
            ref_err_mean = ref_err.mean()

        probe = {
            "affine_grid_error_px_mean": loss.detach(),
            "affine_grid_error_px_rmse": safe_rmse(err.square(), mask=view_mask).detach(),
            "reference_affine_grid_error_px_mean": ref_err_mean.detach(),
        }
        return loss, probe


@dataclass
class AffinePairwiseGeometryLossCfg:
    """AffinePairwiseGeometryLoss 配置。"""

    anchors_per_pair: int = 256
    max_pairs: int | None = None
    sample_from_valid_only: bool = True


class AffinePairwiseGeometryLoss:
    """仿射-几何 pairwise 一致性损失。

    功能:
        同时依赖 affine_pred、height_abs、rpc_corrected，按 pair 约束双向像空间一致性。

    成员变量:
        cfg: pairwise 采样配置。
        _anchor_cache: 缓存 anchor 网格，避免重复构造。
    """

    def __init__(self, geometry_ops: Any, cfg: AffinePairwiseGeometryLossCfg | None = None) -> None:
        """初始化损失类。"""
        self.geometry_ops = geometry_ops
        self.cfg = cfg or AffinePairwiseGeometryLossCfg()
        self._anchor_cache: dict[tuple[int, int, int, str, str], torch.Tensor] = {}

    def _get_anchor_points(self, h: int, w: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """返回 [N,2] true 像素 anchor（line,samp）。"""
        key = (h, w, n, str(device), str(dtype))
        if key not in self._anchor_cache:
            gh = int(max(1, round(n**0.5)))
            gw = int(max(1, (n + gh - 1) // gh))
            grid = make_uniform_grid_points(h, w, gh, gw, device=device, dtype=dtype)
            if grid.shape[0] > n:
                grid = grid[:n]
            self._anchor_cache[key] = grid
        return self._anchor_cache[key]

    def _prepare_ref_idx(self, ref_view_idx: torch.Tensor | None, b: int, device: torch.device) -> torch.Tensor:
        """标准化参考视图索引为 [B]。"""
        if ref_view_idx is None:
            return torch.zeros((b,), dtype=torch.long, device=device)
        ref = ref_view_idx.long().to(device=device).view(-1)
        if ref.numel() == 1:
            ref = ref.expand(b)
        return ref

    def __call__(self, outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """计算 pairwise 几何一致性损失。

        输入:
            outputs: 至少包含 affine_pred/height_abs/rpc_corrected。
            batch: 至少包含 rpc_gt/affine_gt_forward/height_gt/height_valid_mask/scene_xy_center/scene_xy_scale/ref_view_idx。

        输出:
            loss: 标量。
            probe: 误差与有效率统计。
            aux: 轻量辅助信息（如有效 pair 数）。
        """
        affine_pred = outputs["affine_pred"]
        height_abs = outputs["height_abs"]
        rpc_corrected = outputs["rpc_corrected"]

        rpc_gt = batch["rpc_gt"]
        affine_gt_forward = batch["affine_gt_forward"].to(device=affine_pred.device, dtype=affine_pred.dtype)
        height_gt = batch["height_gt"].to(device=affine_pred.device, dtype=affine_pred.dtype)
        height_valid_mask = batch["height_valid_mask"].to(device=affine_pred.device, dtype=affine_pred.dtype)
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)

        b, v, _, h, w = height_abs.shape
        ref_idx = self._prepare_ref_idx(batch.get("ref_view_idx", None), b, affine_pred.device)

        pair_losses: list[torch.Tensor] = []
        pair_errors: list[torch.Tensor] = []
        valid_anchor_ratios: list[torch.Tensor] = []
        world_consistency: list[torch.Tensor] = []
        num_pairs_used = 0
        use_precomputed_anchors = ("anchor_line_samp_true" in batch and "anchor_height_true" in batch)
        if use_precomputed_anchors:
            anchor_v = int(batch["anchor_line_samp_true"].shape[1])
            anchor_h_v = int(batch["anchor_height_true"].shape[1])
            if anchor_v != v or anchor_h_v != v:
                use_precomputed_anchors = False

        for bi in range(b):
            pairs = pairwise_view_pairs(v, self.cfg.max_pairs)
            for i, j in pairs:
                if use_precomputed_anchors:
                    anchors_i_true = batch["anchor_line_samp_true"][bi : bi + 1, i].to(device=affine_pred.device, dtype=affine_pred.dtype)
                    h_i_gt = batch["anchor_height_true"][bi : bi + 1, i].to(device=affine_pred.device, dtype=affine_pred.dtype)
                    in_i = torch.ones((1, anchors_i_true.shape[1]), dtype=torch.bool, device=affine_pred.device)
                else:
                    anchors_i_true = self._get_anchor_points(h, w, self.cfg.anchors_per_pair, affine_pred.device, affine_pred.dtype).unsqueeze(0)

                    # 可选：从有效区域二次过滤（不逐点循环）
                    if self.cfg.sample_from_valid_only:
                        mask_i = height_valid_mask[bi : bi + 1, i]
                        _, valid_i = sample_map_bilinear(mask_i, anchors_i_true)
                        keep = valid_i & (sample_map_bilinear(mask_i, anchors_i_true)[0][:, 0] > 0.5)
                        if keep.sum() > 0:
                            anchors_i_true = anchors_i_true[:, keep[0]]

                    if anchors_i_true.shape[1] == 0:
                        continue

                    h_i_gt, in_i = sample_map_bilinear(height_gt[bi : bi + 1, i], anchors_i_true)
                    h_i_gt = h_i_gt[:, 0]

                # Step3: i true -> world
                xs_i, ys_i = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][i]]],
                    lines=anchors_i_true[..., 0].view(1, 1, -1),
                    samps=anchors_i_true[..., 1].view(1, 1, -1),
                    heights=h_i_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )

                # Step4: world -> j true
                line_j_true, samp_j_true = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    xs=xs_i,
                    ys=ys_i,
                    heights=h_i_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                anchors_j_true = torch.stack([line_j_true.view(1, -1), samp_j_true.view(1, -1)], dim=-1)
                anchors_j_true = anchors_j_true.to(device=affine_pred.device, dtype=affine_pred.dtype)

                # Step5: 过滤有效点
                _, in_j_img = sample_map_bilinear(height_gt[bi : bi + 1, j], anchors_j_true)
                h_j_gt, _ = sample_map_bilinear(height_gt[bi : bi + 1, j], anchors_j_true)
                m_j, _ = sample_map_bilinear(height_valid_mask[bi : bi + 1, j], anchors_j_true)
                m_i = in_i & (sample_map_bilinear(height_valid_mask[bi : bi + 1, i], anchors_i_true)[0][:, 0] > 0.5)
                valid = m_i & in_j_img & (m_j[:, 0] > 0.5)

                if valid.sum() == 0:
                    continue

                anchors_i_true = anchors_i_true[:, valid[0]]
                anchors_j_true = anchors_j_true[:, valid[0]]

                # Step6 true->obs
                anchor_i_obs = apply_affine_to_points(anchors_i_true, affine_gt_forward[bi : bi + 1, i])
                anchor_j_obs = apply_affine_to_points(anchors_j_true, affine_gt_forward[bi : bi + 1, j])

                # Step7 在 obs 域采样预测高程（避免对点与 RPC 的双重 correction）
                h_i_pred, in_i_pred = sample_map_bilinear(height_abs[bi : bi + 1, i], anchor_i_obs)
                h_j_pred, in_j_pred = sample_map_bilinear(height_abs[bi : bi + 1, j], anchor_j_obs)
                h_i_pred = h_i_pred[:, 0]
                h_j_pred = h_j_pred[:, 0]

                valid_pred = in_i_pred & in_j_pred
                if valid_pred.sum() == 0:
                    continue

                anchor_i_obs = anchor_i_obs[:, valid_pred[0]]
                anchor_j_obs = anchor_j_obs[:, valid_pred[0]]
                h_i_pred = h_i_pred[:, valid_pred[0]]
                h_j_pred = h_j_pred[:, valid_pred[0]]

                # Step8 反投影得到 world（输入保持 obs 域坐标）
                xs_i_pred, ys_i_pred = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_corrected[bi][i]]],
                    lines=anchor_i_obs[..., 0].view(1, 1, -1),
                    samps=anchor_i_obs[..., 1].view(1, 1, -1),
                    heights=h_i_pred.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                xs_j_pred, ys_j_pred = self.geometry_ops.linesamp_to_xy_batch(
                    rpc_batch=[[rpc_corrected[bi][j]]],
                    lines=anchor_j_obs[..., 0].view(1, 1, -1),
                    samps=anchor_j_obs[..., 1].view(1, 1, -1),
                    heights=h_j_pred.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )

                # Step9 交叉投影（输出为 obs 域像素）
                l_i2j, s_i2j = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_corrected[bi][j]]],
                    xs=xs_i_pred,
                    ys=ys_i_pred,
                    heights=h_i_pred.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                l_j2i, s_j2i = self.geometry_ops.xy_to_linesamp_batch(
                    rpc_batch=[[rpc_corrected[bi][i]]],
                    xs=xs_j_pred,
                    ys=ys_j_pred,
                    heights=h_j_pred.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )
                proj_i2j = torch.stack([l_i2j.view(1, -1), s_i2j.view(1, -1)], dim=-1)
                proj_j2i = torch.stack([l_j2i.view(1, -1), s_j2i.view(1, -1)], dim=-1)
                proj_i2j = proj_i2j.to(device=affine_pred.device, dtype=affine_pred.dtype)
                proj_j2i = proj_j2i.to(device=affine_pred.device, dtype=affine_pred.dtype)

                # Step10 pair loss（obs 域双向一致性）
                e_i2j = torch.linalg.norm(proj_i2j - anchor_j_obs, dim=-1)
                e_j2i = torch.linalg.norm(proj_j2i - anchor_i_obs, dim=-1)
                loss_pair = 0.5 * e_i2j.mean() + 0.5 * e_j2i.mean()

                pair_losses.append(loss_pair)
                pair_errors.append(torch.cat([e_i2j, e_j2i], dim=-1))
                valid_anchor_ratios.append(torch.tensor(float(valid_pred.sum().item()) / float(self.cfg.anchors_per_pair), device=loss_pair.device))

                p_i = torch.stack([xs_i_pred.view(-1), ys_i_pred.view(-1), h_i_pred.view(-1)], dim=-1)
                p_j = torch.stack([xs_j_pred.view(-1), ys_j_pred.view(-1), h_j_pred.view(-1)], dim=-1)
                world_consistency.append(torch.linalg.norm(p_i - p_j, dim=-1).mean())
                num_pairs_used += 1

        if len(pair_losses) == 0:
            zero = torch.zeros((), device=affine_pred.device, dtype=affine_pred.dtype)
            probe = {
                "affine_pair_error_px_mean": zero,
                "affine_pair_error_px_rmse": zero,
                "pairwise_valid_anchor_ratio": zero,
                "pairwise_num_pairs_used": zero,
                "pairwise_world_xyz_consistency_mean": zero,
            }
            return zero, probe, {"num_pairs_used": 0}

        loss = torch.stack(pair_losses).mean()
        all_err = torch.cat(pair_errors, dim=-1)
        all_err_sq = all_err.square()

        probe = {
            "affine_pair_error_px_mean": all_err.mean().detach(),
            "affine_pair_error_px_rmse": safe_rmse(all_err_sq).detach(),
            "pairwise_valid_anchor_ratio": torch.stack(valid_anchor_ratios).mean().detach(),
            "pairwise_num_pairs_used": torch.tensor(float(num_pairs_used), device=loss.device),
            "pairwise_world_xyz_consistency_mean": torch.stack(world_consistency).mean().detach() if len(world_consistency) > 0 else torch.zeros_like(loss),
        }
        aux = {"num_pairs_used": num_pairs_used}
        return loss, probe, aux


class AffineLinearRegularization:
    """仿射线性部分正则。

    功能:
        仅约束 affine 的 2x2 线性部分接近单位阵，不约束平移项。
    """

    def __call__(self, affine_pred: torch.Tensor, ref_view_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算线性正则项。

        输入:
            affine_pred: [B,V,2,3]。
            ref_view_idx: [B] 或 None。

        输出:
            loss 与 probe 字典。
        """
        b, v = affine_pred.shape[:2]
        lin = affine_pred[..., :2]
        trans = affine_pred[..., 2]

        eye = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=affine_pred.device, dtype=affine_pred.dtype).view(1, 1, 2, 2)
        frob_sq = (lin - eye).square().sum(dim=(-2, -1))

        mask = torch.ones((b, v), device=affine_pred.device, dtype=affine_pred.dtype)
        if ref_view_idx is not None:
            ref = ref_view_idx.long().view(-1)
            if ref.numel() == 1:
                ref = ref.expand(b)
            mask[torch.arange(b, device=mask.device), ref] = 0.0

        denom = mask.sum().clamp_min(1.0)
        loss = (frob_sq * mask).sum() / denom

        probe = {
            "affine_linear_frob_mean": torch.sqrt((frob_sq * mask).sum() / denom).detach(),
            "affine_translation_abs_mean": (trans.abs().sum(dim=-1) * mask).sum().detach() / denom,
        }
        return loss, probe


class RefAffineIdentityLoss:
    """参考视图仿射约束。

    功能:
        仅对 ref_view_idx 位置施加约束，推动其 affine_pred 接近单位阵。
    """

    def __call__(self, affine_pred: torch.Tensor, ref_view_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算参考视图仿射约束。"""
        b, v = affine_pred.shape[:2]
        if ref_view_idx is None:
            ref = torch.zeros((b,), dtype=torch.long, device=affine_pred.device)
        else:
            ref = ref_view_idx.long().view(-1).to(device=affine_pred.device)
            if ref.numel() == 1:
                ref = ref.expand(b)
        ref = ref.clamp(0, v - 1)

        pred_ref = affine_pred[torch.arange(b, device=affine_pred.device), ref]  # [B,2,3]
        eye = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=pred_ref.device,
            dtype=pred_ref.dtype,
        ).view(1, 2, 3).expand_as(pred_ref)
        diff = pred_ref - eye
        loss = diff.square().mean()
        probe = {
            "ref_affine_identity_l2": loss.detach(),
            "ref_affine_translation_abs_mean": pred_ref[..., 2].abs().mean().detach(),
            "ref_affine_linear_abs_mean": (pred_ref[..., :2] - eye[..., :2]).abs().mean().detach(),
        }
        return loss, probe

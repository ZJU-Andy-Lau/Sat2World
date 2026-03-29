"""engine.tensorboard_vis

TensorBoard 可视化统一入口。

职责：
- 标量/直方图/图像面板/点云统一记录；
- trainer 只需调用公开接口，无需关心绘图细节；
- 支持 add_mesh 不可用时的二维可视化 fallback。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

from loss.common import apply_affine_to_points, make_uniform_grid_points


class TensorBoardMonitor:
    """TensorBoard 监控器。

    成员变量说明：
    - writer: SummaryWriter 实例；
    - enable_mesh: 是否尝试 add_mesh；
    - image_max_views: 图像面板最多显示视图数；
    - max_pointcloud_points: 点云可视化最多点数；
    - is_enabled: 是否启用（非主进程可禁用）。
    """

    def __init__(
        self,
        log_dir: str,
        *,
        is_enabled: bool = True,
        enable_mesh: bool = True,
        image_max_views: int = 4,
        max_pointcloud_points: int = 8192,
        flush_secs: int = 30,
        writer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """初始化 monitor。"""
        self.is_enabled = bool(is_enabled)
        self.enable_mesh = bool(enable_mesh)
        self.image_max_views = int(image_max_views)
        self.max_pointcloud_points = int(max_pointcloud_points)
        self.writer = None
        if self.is_enabled and SummaryWriter is not None:
            kwargs = writer_kwargs or {}
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs, **kwargs)
        elif self.is_enabled:
            self.is_enabled = False

    def _to_float(self, v: Any) -> float | None:
        if torch.is_tensor(v):
            if v.numel() != 1:
                return None
            v = float(v.detach().cpu().item())
        if isinstance(v, (int, float)):
            if math.isfinite(float(v)):
                return float(v)
        return None

    def log_scalars(self, tag_prefix: str, scalar_dict: dict[str, Any], global_step: int) -> None:
        """记录标量。"""
        if not self.is_enabled or self.writer is None:
            return
        for k, v in scalar_dict.items():
            fv = self._to_float(v)
            if fv is None:
                continue
            self.writer.add_scalar(f"{tag_prefix}/{k}", fv, global_step)

    def log_histograms(self, hist_dict: dict[str, Any], global_step: int) -> None:
        """记录直方图。"""
        if not self.is_enabled or self.writer is None:
            return
        for k, v in hist_dict.items():
            if torch.is_tensor(v) and v.numel() > 0:
                self.writer.add_histogram(k, v.detach().float().cpu(), global_step)

    def make_rgb_montage(self, images_vchw: torch.Tensor) -> torch.Tensor:
        """将 [V,3,H,W] 多视图图像拼接成横向 montage。"""
        v = min(images_vchw.shape[0], self.image_max_views)
        imgs = images_vchw[:v].clamp(0, 1)
        return torch.cat([imgs[i] for i in range(v)], dim=-1)

    def make_colormap_image(self, x: torch.Tensor, vmin: float | None = None, vmax: float | None = None) -> torch.Tensor:
        """把单通道图转为伪彩色（简单 heatmap）。"""
        t = x.detach().float()
        if t.ndim == 3:
            t = t[0]
        mn = float(t.min().item()) if vmin is None else vmin
        mx = float(t.max().item()) if vmax is None else vmax
        if mx <= mn:
            mx = mn + 1e-6
        n = (t - mn) / (mx - mn)
        n = n.clamp(0, 1)
        r = n
        g = 1.0 - (n - 0.5).abs() * 2.0
        b = 1.0 - n
        return torch.stack([r, g.clamp(0, 1), b], dim=0)

    def make_abs_error_map(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """构建绝对误差热力图。"""
        err = (pred - gt).abs()
        if err.ndim == 3:
            err = err[0]
        elif err.ndim == 4:
            err = err[0, 0]
        return self.make_colormap_image(err)

    def make_affine_residual_heatmap(self, affine_gt_forward: torch.Tensor, affine_pred: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """仿射组合残差热力图。

        方向约定：
        - affine_gt_forward: true -> observed
        - affine_pred: observed -> true
        """
        grid = make_uniform_grid_points(h, w, 24, 24, device=affine_pred.device, dtype=affine_pred.dtype)  # [N,2]
        n = grid.shape[0]
        g = grid.unsqueeze(0)
        obs = apply_affine_to_points(g, affine_gt_forward.unsqueeze(0))[0]
        rec = apply_affine_to_points(obs.unsqueeze(0), affine_pred.unsqueeze(0))[0]
        d = (rec - grid).norm(dim=-1)
        dmap = d.view(24, 24).unsqueeze(0).unsqueeze(0)
        dmap = F.interpolate(dmap, size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        return self.make_colormap_image(dmap)

    def make_pairwise_error_matrix(self, aux: dict[str, Any], v: int) -> torch.Tensor | None:
        """构建 pairwise 误差矩阵热图（若 aux 中有矩阵则可视化）。"""
        m = aux.get("pairwise_error_matrix", None)
        if m is None:
            return None
        if torch.is_tensor(m):
            if m.ndim == 2:
                return self.make_colormap_image(m)
        return None

    def log_visual_panels(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any],
        render_outputs: dict[str, Any] | None,
        aux: dict[str, Any],
        global_step: int,
        split: str,
    ) -> None:
        """记录固定监控样本图像面板。"""
        if not self.is_enabled or self.writer is None:
            return

        images = batch["images"][0]  # [V,3,H,W]
        height_gt = batch.get("height_gt", None)
        pred_h = outputs.get("height_abs", None)
        pred_p = outputs.get("point_abs", None)

        self.writer.add_image(f"vis/{split}/input_rgb", self.make_rgb_montage(images), global_step)

        if height_gt is not None and pred_h is not None:
            gt = height_gt[0, 0]
            ph = pred_h[0, 0]
            self.writer.add_image(f"vis/{split}/height_gt", self.make_colormap_image(gt), global_step)
            self.writer.add_image(f"vis/{split}/height_pred", self.make_colormap_image(ph), global_step)
            self.writer.add_image(f"vis/{split}/height_error", self.make_abs_error_map(ph, gt), global_step)

        if pred_p is not None and aux.get("gt_point_map", None) is not None:
            pz = pred_p[0, 0, 2]
            gz = aux["gt_point_map"][0, 0, 2]
            self.writer.add_image(f"vis/{split}/point_z_pred", self.make_colormap_image(pz), global_step)
            self.writer.add_image(f"vis/{split}/point_z_gt", self.make_colormap_image(gz), global_step)
            self.writer.add_image(f"vis/{split}/point_z_error", self.make_abs_error_map(pz, gz), global_step)

        if render_outputs is not None and render_outputs.get("rpc", None) is not None:
            rr = render_outputs["rpc"]
            rp = render_outputs["point"]
            if rr.get("num_targets", 0) > 0:
                self.writer.add_image(f"vis/{split}/render_rpc_rgb", rr["rendered_rgb"][0], global_step)
                self.writer.add_image(f"vis/{split}/render_target_rgb", rr["target_rgb"][0], global_step)
                self.writer.add_image(f"vis/{split}/render_rpc_alpha", rr["rendered_alpha"][0], global_step)
                self.writer.add_image(f"vis/{split}/render_rpc_height", self.make_colormap_image(rr["rendered_height"][0]), global_step)
            if rp.get("num_targets", 0) > 0:
                self.writer.add_image(f"vis/{split}/render_point_rgb", rp["rendered_rgb"][0], global_step)
                self.writer.add_image(f"vis/{split}/render_point_alpha", rp["rendered_alpha"][0], global_step)
                self.writer.add_image(f"vis/{split}/render_point_height", self.make_colormap_image(rp["rendered_height"][0]), global_step)

        if "gaussian_confidence_rpc" in outputs:
            self.writer.add_image(f"vis/{split}/gaussian_conf_rpc", self.make_colormap_image(outputs["gaussian_confidence_rpc"][0, 0]), global_step)
        if "gaussian_confidence_point" in outputs:
            self.writer.add_image(f"vis/{split}/gaussian_conf_point", self.make_colormap_image(outputs["gaussian_confidence_point"][0, 0]), global_step)
        if "gaussian_opacity" in outputs:
            self.writer.add_image(f"vis/{split}/gaussian_opacity", self.make_colormap_image(outputs["gaussian_opacity"][0, 0]), global_step)
        if "gaussian_scale" in outputs:
            mag = outputs["gaussian_scale"][0, 0].norm(dim=0, keepdim=True)
            self.writer.add_image(f"vis/{split}/gaussian_scale_mag", self.make_colormap_image(mag), global_step)
        if "gaussian_centers_rpc" in outputs and "gaussian_centers_point" in outputs:
            d = (outputs["gaussian_centers_rpc"][0, 0] - outputs["gaussian_centers_point"][0, 0]).norm(dim=0, keepdim=True)
            self.writer.add_image(f"vis/{split}/center_disagreement", self.make_colormap_image(d), global_step)

        if "affine_gt_forward" in batch and "affine_pred" in outputs:
            h, w = images.shape[-2:]
            hm = self.make_affine_residual_heatmap(batch["affine_gt_forward"][0, 0], outputs["affine_pred"][0, 0], h, w)
            self.writer.add_image(f"vis/{split}/affine_residual", hm, global_step)

        pairwise = self.make_pairwise_error_matrix(aux, v=int(images.shape[0]))
        if pairwise is not None:
            self.writer.add_image(f"vis/{split}/pairwise_error_matrix", pairwise, global_step)

    def _sample_pointcloud(self, xyz_hw3: torch.Tensor, rgb_hw3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = xyz_hw3.shape[:2]
        xyz = xyz_hw3.reshape(-1, 3)
        rgb = rgb_hw3.reshape(-1, 3)
        n = xyz.shape[0]
        if n > self.max_pointcloud_points:
            step = max(1, n // self.max_pointcloud_points)
            idx = torch.arange(0, n, step, device=xyz.device)[: self.max_pointcloud_points]
            xyz = xyz[idx]
            rgb = rgb[idx]
        return xyz, rgb

    def log_pointclouds(self, batch: dict[str, Any], outputs: dict[str, Any], aux: dict[str, Any], global_step: int, split: str) -> None:
        """记录 GT/pred/rpc-center 点云（优先 add_mesh，失败 fallback）。"""
        if not self.is_enabled or self.writer is None:
            return
        if "point_abs" not in outputs or "gaussian_centers_rpc" not in outputs:
            return

        img = batch["images"][0, 0].permute(1, 2, 0).detach().float().cpu()
        pred = outputs["point_abs"][0, 0].permute(1, 2, 0).detach().float().cpu()
        rpc = outputs["gaussian_centers_rpc"][0, 0].permute(1, 2, 0).detach().float().cpu()
        gt = aux.get("gt_point_map", None)
        if gt is not None:
            gt = gt[0, 0].permute(1, 2, 0).detach().float().cpu()

        def _mesh(tag: str, xyz_hw3: torch.Tensor):
            xyz, rgb = self._sample_pointcloud(xyz_hw3, img)
            if self.enable_mesh:
                try:
                    self.writer.add_mesh(tag, vertices=xyz.unsqueeze(0), colors=(rgb.clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0), global_step=global_step)
                    return
                except Exception:
                    pass
            # fallback: BEV
            bev = torch.zeros(3, 512, 512)
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            xn = (x - x.min()) / (x.max() - x.min() + 1e-6)
            yn = (y - y.min()) / (y.max() - y.min() + 1e-6)
            zi = (z - z.min()) / (z.max() - z.min() + 1e-6)
            xi = (xn * 511).long().clamp(0, 511)
            yi = (yn * 511).long().clamp(0, 511)
            bev[0, yi, xi] = zi
            bev[1, yi, xi] = 1.0 - zi
            bev[2, yi, xi] = 0.5
            self.writer.add_image(tag + "_bev", bev, global_step)

        if gt is not None:
            _mesh(f"vis/{split}/pc_gt", gt)
        _mesh(f"vis/{split}/pc_pred", pred)
        _mesh(f"vis/{split}/pc_rpc_center", rpc)

    def log_optimizer_state(self, state: dict[str, Any], global_step: int) -> None:
        """记录优化器与系统状态。"""
        self.log_scalars("optim", state, global_step)

    def log_text(self, tag: str, text: str, global_step: int) -> None:
        """记录文本事件。"""
        if self.is_enabled and self.writer is not None:
            self.writer.add_text(tag, text, global_step)

    def flush(self) -> None:
        """强制 flush。"""
        if self.is_enabled and self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """关闭 writer。"""
        if self.is_enabled and self.writer is not None:
            self.writer.close()

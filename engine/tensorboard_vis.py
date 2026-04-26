"""engine.tensorboard_vis

TensorBoard 可视化统一入口。

职责：
- 标量/直方图/图像面板/点云统一记录；
- trainer 只需调用公开接口，无需关心绘图细节；
- 支持 add_mesh 不可用时的二维可视化 fallback。
"""

from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

from loss.common import apply_affine_to_points, make_uniform_grid_points, sample_map_bilinear


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
        min_free_mb: float = 100.0,
        disk_check_interval_sec: float = 1.0,
        low_disk_warn_interval_sec: float = 300.0,
        reopen_interval_sec: float = 30.0,
        skip_when_low_disk: bool = True,
    ) -> None:
        """初始化 monitor。"""
        self.is_enabled = bool(is_enabled)
        self.enable_mesh = bool(enable_mesh)
        self.image_max_views = int(image_max_views)
        self.max_pointcloud_points = int(max_pointcloud_points)
        self.log_dir = Path(log_dir)
        self.flush_secs = int(flush_secs)
        self.writer_kwargs = dict(writer_kwargs or {})
        self.min_free_bytes = int(float(min_free_mb) * 1024 * 1024)
        self.skip_when_low_disk = bool(skip_when_low_disk)
        self.disk_check_interval_sec = float(disk_check_interval_sec)
        self.low_disk_warn_interval_sec = float(low_disk_warn_interval_sec)
        self.reopen_interval_sec = float(reopen_interval_sec)
        self._last_disk_check_time = 0.0
        self._last_disk_ok = True
        self._last_free_bytes: int | None = None
        self._last_low_disk_warn_time = 0.0
        self._last_reopen_attempt_time = 0.0
        self._writer_closed_by_disk_error = False
        self.writer = None
        if self.is_enabled and SummaryWriter is not None:
            self._create_writer(op="init", tag=None)
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

    def _is_disk_write_error(self, exc: Exception) -> bool:
        """判断是否为磁盘空间/配额类写入错误。"""
        if isinstance(exc, OSError):
            if exc.errno in (errno.ENOSPC, errno.EDQUOT):
                return True
        msg = str(exc).lower()
        return ("no space left on device" in msg) or ("disk quota exceeded" in msg) or ("quota exceeded" in msg)

    def _warn_write_failure(self, op: str, tag: str | None, exc: Exception) -> None:
        tag_msg = f", tag={tag}" if tag is not None else ""
        if self._is_disk_write_error(exc):
            warnings.warn(
                f"TensorBoard {op} failed due to disk issue{tag_msg}, log_dir={self.log_dir}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(f"TensorBoard {op} failed{tag_msg}, log_dir={self.log_dir}: {exc}", RuntimeWarning, stacklevel=2)

    def _resolve_statvfs_path(self) -> Path:
        p = self.log_dir
        while not p.exists():
            if p.parent == p:
                break
            p = p.parent
        return p

    def _check_disk_space(self, *, op: str, tag: str | None = None) -> bool:
        if not self.skip_when_low_disk:
            return True
        now = time.monotonic()
        should_refresh = (now - self._last_disk_check_time) >= self.disk_check_interval_sec
        if should_refresh:
            self._last_disk_check_time = now
            try:
                stat_path = self._resolve_statvfs_path()
                st = os.statvfs(stat_path)
                free_bytes = int(st.f_bavail) * int(st.f_frsize)
                self._last_free_bytes = free_bytes
                self._last_disk_ok = free_bytes >= self.min_free_bytes
            except Exception as exc:
                self._last_disk_ok = False
                self._last_free_bytes = None
                warnings.warn(
                    f"TensorBoard disk check failed for op={op}, tag={tag}, log_dir={self.log_dir}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self._last_disk_ok:
            return True
        if (now - self._last_low_disk_warn_time) >= self.low_disk_warn_interval_sec:
            free_txt = "unknown" if self._last_free_bytes is None else str(self._last_free_bytes)
            warnings.warn(
                f"Skip TensorBoard {op} due to low disk space, tag={tag}, "
                f"log_dir={self.log_dir}, free_bytes={free_txt}, threshold_bytes={self.min_free_bytes}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._last_low_disk_warn_time = now
        return False

    def _create_writer(self, *, op: str, tag: str | None) -> bool:
        try:
            self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=self.flush_secs, **self.writer_kwargs)
            self._writer_closed_by_disk_error = False
            return True
        except Exception as exc:
            self._warn_write_failure(op, tag, exc)
            self.writer = None
            if self._is_disk_write_error(exc):
                self._writer_closed_by_disk_error = True
            return False

    def _ensure_writer(self, op: str, tag: str | None = None) -> bool:
        if not self.is_enabled or SummaryWriter is None:
            return False
        if not self._check_disk_space(op=op, tag=tag):
            return False
        if self.writer is not None:
            return True
        now = time.monotonic()
        if self._writer_closed_by_disk_error and (now - self._last_reopen_attempt_time) < self.reopen_interval_sec:
            return False
        self._last_reopen_attempt_time = now
        return self._create_writer(op=f"{op}/reopen", tag=tag)

    def _close_writer_noexcept(self, op: str = "close") -> None:
        if self.writer is None:
            return
        writer = self.writer
        self.writer = None
        try:
            writer.close()
        except Exception as exc:
            self._warn_write_failure(op, None, exc)

    def _safe_writer_call(self, op: str, fn: Callable[[], None], *, tag: str | None = None) -> bool:
        if not self._ensure_writer(op=op, tag=tag):
            return False
        try:
            fn()
            return True
        except Exception as exc:
            self._warn_write_failure(op, tag, exc)
            if self._is_disk_write_error(exc):
                self._writer_closed_by_disk_error = True
                self._close_writer_noexcept(op=f"{op}/close_after_error")
            return False

    def _route_scalar_tag(self, split: str, key: str) -> str | None:
        """将标量键路由到结构化 tag；返回 None 表示丢弃。"""
        essential_losses = {
            "loss_total",
            "loss_affine_grid",
            "loss_affine_pair",
            "loss_affine_reg",
            "loss_affine_ref",
            "loss_height",
            "loss_height_anchor",
            "loss_point",
            "loss_point_reproj",
            "loss_height_reproj",
            "loss_point_pair",
            "loss_normal_point",
            "loss_feature_nce",
            "loss_patch_match",
            # "loss_render_rpc",
            # "loss_render_point",
            # "loss_ssim",
        }
        essential_metrics = {
            "metric_affine_grid_error_px_mean",
            "metric_affine_pair_error_px_mean",
            # "metric_height_rmse",
            # "metric_height_mae",
            # "metric_point_xyz_rmse",
            # "metric_point_z_rmse",
            "metric_point_reproj_px_mean",
            "metric_height_reproj_px_mean",
            "metric_patch_match_acc_top1_1px",
            "metric_feature_nce_acc_top1",
            # "metric_center_dist_mean",
        }
        if key in essential_losses:
            return f"{split}/loss/{key.removeprefix('loss_')}"
        if key in essential_metrics:
            return f"{split}/metrics/{key.removeprefix('metric_')}"
        if key.startswith("optim/") or key.startswith("time/") or key.startswith("system/"):
            return key
        return None

    def log_scalars(self, tag_prefix: str, scalar_dict: dict[str, Any], global_step: int) -> None:
        """记录标量。"""
        if not self.is_enabled:
            return
        for k, v in scalar_dict.items():
            fv = self._to_float(v)
            if fv is None:
                continue
            if tag_prefix in {"train", "val"}:
                tag = self._route_scalar_tag(tag_prefix, k)
                if tag is None:
                    continue
            else:
                tag = k if "/" in k else f"{tag_prefix}/{k}"
            self._safe_writer_call("add_scalar", lambda tag=tag, fv=fv: self.writer.add_scalar(tag, fv, global_step), tag=tag)

    def log_histograms(self, hist_dict: dict[str, Any], global_step: int) -> None:
        """记录直方图。"""
        if not self.is_enabled:
            return
        for k, v in hist_dict.items():
            if torch.is_tensor(v) and v.numel() > 0:
                self._safe_writer_call("add_histogram", lambda k=k, v=v: self.writer.add_histogram(k, v.detach().float().cpu(), global_step), tag=k)

    def make_rgb_montage(self, images_vchw: torch.Tensor) -> torch.Tensor:
        """将 [V,3,H,W] 多视图图像拼接成横向 montage。"""
        v = min(images_vchw.shape[0], self.image_max_views)
        imgs = images_vchw[:v].clamp(0, 1)
        return torch.cat([imgs[i] for i in range(v)], dim=-1)

    def make_batch_view_montage(self, images_bvchw: torch.Tensor) -> torch.Tensor:
        """将 [B,V,3,H,W] 拼接为“每行一个样本、每列一个视图”的大图。"""
        if images_bvchw.ndim != 5:
            raise ValueError("images_bvchw must be [B,V,3,H,W]")
        b, v = int(images_bvchw.shape[0]), int(images_bvchw.shape[1])
        rows = []
        for bi in range(b):
            row = torch.cat([images_bvchw[bi, vi].clamp(0, 1) for vi in range(v)], dim=-1)
            rows.append(row)
        return torch.cat(rows, dim=-2)

    def _draw_label_on_image(
        self,
        image_chw: torch.Tensor,
        label: str,
        *,
        xy: tuple[int, int] = (4, 4),
        text_color: tuple[int, int, int] = (255, 255, 255),
        box_color: tuple[int, int, int] = (0, 0, 0),
    ) -> torch.Tensor:
        img = image_chw.detach().float().clamp(0, 1).cpu()
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError("image_chw must be [3,H,W]")
        h, w = int(img.shape[-2]), int(img.shape[-1])
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255.0).astype("uint8"))
        draw = ImageDraw.Draw(pil)
        font = ImageFont.load_default()
        x0, y0 = int(xy[0]), int(xy[1])
        x0 = max(0, min(x0, max(w - 1, 0)))
        y0 = max(0, min(y0, max(h - 1, 0)))
        bbox = draw.textbbox((x0, y0), label, font=font)
        pad = 2
        bx0 = max(0, bbox[0] - pad)
        by0 = max(0, bbox[1] - pad)
        bx1 = min(w - 1, bbox[2] + pad)
        by1 = min(h - 1, bbox[3] + pad)
        draw.rectangle([bx0, by0, bx1, by1], fill=box_color)
        draw.text((x0, y0), label, fill=text_color, font=font)
        out = torch.from_numpy(np.array(pil))
        out = out.permute(2, 0, 1).to(torch.float32) / 255.0
        return out.clamp(0, 1)

    def _invert_affine_2x3(self, affine_2x3: torch.Tensor) -> torch.Tensor:
        """求 2x3 仿射逆（单视图）。"""
        if affine_2x3.shape != (2, 3):
            raise ValueError("affine_2x3 must be [2,3]")
        a = affine_2x3[:, :2]
        t = affine_2x3[:, 2:]
        a_inv = torch.linalg.inv(a)
        t_inv = -(a_inv @ t)
        return torch.cat([a_inv, t_inv], dim=1)

    def _warp_crop_by_affine_forward(
        self,
        image_chw: torch.Tensor,
        affine_forward_2x3: torch.Tensor,
        affine_correction_2x3: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把 I_crop 按 affine_gt_forward 变换得到 I_crop_af。"""
        if image_chw.ndim != 3 or image_chw.shape[0] != 3:
            raise ValueError("image_chw must be [3,H,W]")
        h, w = int(image_chw.shape[-2]), int(image_chw.shape[-1])
        img = image_chw.unsqueeze(0).detach().float()
        aff_corr = affine_correction_2x3 if affine_correction_2x3 is not None else self._invert_affine_2x3(affine_forward_2x3)
        aff_corr = aff_corr.to(device=img.device, dtype=img.dtype)

        line = torch.arange(h, device=img.device, dtype=img.dtype)
        samp = torch.arange(w, device=img.device, dtype=img.dtype)
        yy, xx = torch.meshgrid(line, samp, indexing="ij")
        obs_grid = torch.stack([yy, xx], dim=-1).view(1, -1, 2)  # [1,HW,2], [line,samp]
        src_grid = apply_affine_to_points(obs_grid, aff_corr.unsqueeze(0)).view(1, h, w, 2)

        x = 2.0 * src_grid[..., 1] / max(w - 1, 1) - 1.0
        y = 2.0 * src_grid[..., 0] / max(h - 1, 1) - 1.0
        grid = torch.stack([x, y], dim=-1)
        out = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return out[0].clamp(0, 1)

    def _make_checkerboard_blend(self, image_a_chw: torch.Tensor, image_b_chw: torch.Tensor, tile_size: int = 32) -> torch.Tensor:
        """构造棋盘格融合图。"""
        if image_a_chw.shape != image_b_chw.shape:
            raise ValueError("image_a_chw and image_b_chw must share shape")
        if image_a_chw.ndim != 3 or image_a_chw.shape[0] != 3:
            raise ValueError("image_a_chw must be [3,H,W]")
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        h, w = int(image_a_chw.shape[-2]), int(image_a_chw.shape[-1])
        yy = torch.arange(h, device=image_a_chw.device).view(h, 1).expand(h, w)
        xx = torch.arange(w, device=image_a_chw.device).view(1, w).expand(h, w)
        mask = ((yy // tile_size + xx // tile_size) % 2).to(dtype=image_a_chw.dtype)
        mask = mask.unsqueeze(0)
        return (image_a_chw * (1.0 - mask) + image_b_chw * mask).clamp(0, 1)

    def _make_pairwise_affine_rpc_checkerboard(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Any],
        *,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]] | None:
        """随机抽取一对视图，构造“j图 vs i投影到j”的棋盘格可视化。

        返回:
            (checker, image_j_corrected, image_i_projected_to_j, (i,j)) 或 None。
        """
        need_keys_batch = {"images", "rpc_gt", "height_gt", "affine_gt_forward"}
        need_keys_outputs = {"affine_pred"}
        if any(k not in batch for k in need_keys_batch) or any(k not in outputs for k in need_keys_outputs):
            return None

        images = batch["images"]  # [B,V,3,H,W]
        aff_gt_forward = batch["affine_gt_forward"]
        aff_pred = outputs["affine_pred"]
        if images.ndim != 5 or aff_gt_forward.ndim != 4 or aff_pred.ndim != 4:
            return None

        b, v, _, h, w = images.shape
        if b <= 0 or v < 2:
            return None

        bi = 0
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(global_step) + 20260330)
        perm = torch.randperm(v, generator=gen)
        i = int(perm[0].item())
        j = int(perm[1].item())

        image_i_true = images[bi, i].detach().float()
        image_j_true = images[bi, j].detach().float()

        # Step2: GT forward 扰动到 observed 域
        image_i_obs = self._warp_crop_by_affine_forward(image_i_true, aff_gt_forward[bi, i].detach().float())
        image_j_obs = self._warp_crop_by_affine_forward(image_j_true, aff_gt_forward[bi, j].detach().float())

        # Step3: 用预测仿射做纠正（obs -> true）
        image_i_corr = self._warp_crop_by_affine_forward(image_i_obs, aff_pred[bi, i].detach().float())
        image_j_corr = self._warp_crop_by_affine_forward(image_j_obs, aff_pred[bi, j].detach().float())

        # Step4: 在 j 像空间把 i 采样投影到 j（使用 h_j_gt + rpc_j_gt/rpc_i_gt）
        rpc_j = batch["rpc_gt"][bi][j]
        rpc_i = batch["rpc_gt"][bi][i]

        # [N] 的 j 像素坐标 (line,samp)
        line = torch.arange(h, device=image_j_corr.device, dtype=torch.float32)
        samp = torch.arange(w, device=image_j_corr.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(line, samp, indexing="ij")
        line_flat = yy.reshape(-1)
        samp_flat = xx.reshape(-1)

        h_j_map = batch["height_gt"][bi, j]
        if h_j_map.ndim == 3 and h_j_map.shape[0] == 1:
            h_j_map = h_j_map[0]
        h_j_flat = h_j_map.detach().to(device=image_j_corr.device, dtype=torch.float32).reshape(-1)

        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        xy_center_j = None if scene_xy_center is None else scene_xy_center[bi]
        xy_scale_j = None if scene_xy_scale is None else scene_xy_scale[bi]

        x_world, y_world = rpc_j.RPC_LINESAMP2XY(
            line_in=line_flat,
            samp_in=samp_flat,
            h_in=h_j_flat,
            output_type="tensor",
            xy_center=xy_center_j,
            xy_scale=xy_scale_j,
        )
        line_i_proj, samp_i_proj = rpc_i.RPC_XY2LINESAMP(
            x_in=x_world,
            y_in=y_world,
            h_in=h_j_flat,
            output_type="tensor",
            xy_center=xy_center_j,
            xy_scale=xy_scale_j,
        )
        line_i_proj = line_i_proj.to(device=image_i_corr.device, dtype=image_i_corr.dtype)
        samp_i_proj = samp_i_proj.to(device=image_i_corr.device, dtype=image_i_corr.dtype)
        proj_points_i = torch.stack([line_i_proj, samp_i_proj], dim=-1).unsqueeze(0)  # [1,N,2]

        sampled_i_to_j, _ = sample_map_bilinear(image_i_corr.unsqueeze(0), proj_points_i)
        image_i_to_j = sampled_i_to_j.view(1, 3, h, w)[0].clamp(0, 1)

        # Step5: 与 j 构造棋盘格
        checker = self._make_checkerboard_blend(image_j_corr, image_i_to_j, tile_size=32)
        return checker, image_j_corr.clamp(0, 1), image_i_to_j, (i, j)

    def _overlay_points_on_image(
        self,
        image_chw: torch.Tensor,
        points_n2: torch.Tensor,
        *,
        color: tuple[float, float, float] = (1.0, 0.2, 0.2),
        radius: int = 1,
    ) -> torch.Tensor:
        """在图像上叠加 [line,samp] 点。"""
        if image_chw.ndim != 3 or image_chw.shape[0] != 3:
            raise ValueError("image_chw must be [3,H,W]")
        out = image_chw.detach().float().clone()
        h, w = int(out.shape[-2]), int(out.shape[-1])
        pts = points_n2.detach().to(device=out.device, dtype=torch.float32)
        line = pts[:, 0].round().long()
        samp = pts[:, 1].round().long()
        valid = (line >= 0) & (line < h) & (samp >= 0) & (samp < w)
        line = line[valid]
        samp = samp[valid]
        if line.numel() == 0:
            return out.clamp(0, 1)
        for dl in range(-radius, radius + 1):
            for ds in range(-radius, radius + 1):
                li = (line + dl).clamp(0, h - 1)
                si = (samp + ds).clamp(0, w - 1)
                out[0, li, si] = color[0]
                out[1, li, si] = color[1]
                out[2, li, si] = color[2]
        return out.clamp(0, 1)

    def make_colormap_image(
        self,
        x: torch.Tensor,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap_name: str = "YlGnBu_r",
    ) -> torch.Tensor:
        """把单通道图转为伪彩色。"""
        t = x.detach().float()
        if t.ndim == 3:
            t = t[0]
        mn = float(t.min().item()) if vmin is None else vmin
        mx = float(t.max().item()) if vmax is None else vmax
        if mx <= mn:
            mx = mn + 1e-6
        try:
            from matplotlib import cm, colors

            arr = t.cpu().numpy().astype(np.float32)
            norm = colors.Normalize(vmin=float(mn), vmax=float(mx), clip=True)
            cmap = cm.get_cmap(cmap_name)
            rgba = cmap(norm(arr))
            rgb = torch.from_numpy(rgba[..., :3]).permute(2, 0, 1).to(torch.float32)
            return rgb.clamp(0, 1)
        except Exception:
            n = ((t - mn) / (mx - mn)).clamp(0, 1)
            r = n
            g = 1.0 - (n - 0.5).abs() * 2.0
            b = 1.0 - n
            return torch.stack([r, g.clamp(0, 1), b], dim=0)

    def _group_vrange(
        self,
        maps: list[torch.Tensor],
        *,
        masks: list[torch.Tensor] | None = None,
        q_low: float = 0.05,
        q_high: float = 0.95,
    ) -> tuple[float, float]:
        vals: list[torch.Tensor] = []
        for idx, m in enumerate(maps):
            mm = m.detach().float()
            if mm.ndim == 3:
                mm = mm[0]
            if masks is not None and idx < len(masks) and masks[idx] is not None:
                mk = masks[idx].detach().float()
                if mk.ndim == 3:
                    mk = mk[0]
                vals.append(mm[mk > 0.5].reshape(-1))
            else:
                vals.append(mm.reshape(-1))
        if len(vals) == 0:
            return 0.0, 1.0
        nz = [x for x in vals if x.numel() > 0]
        if len(nz) == 0:
            return 0.0, 1.0
        joint = torch.cat(nz, dim=0)
        if joint.numel() == 0:
            return 0.0, 1.0
        lo = float(torch.quantile(joint, float(q_low)).item())
        hi = float(torch.quantile(joint, float(q_high)).item())
        if hi <= lo + 1e-6:
            hi = lo + 1e-6
        return lo, hi

    def make_abs_error_map(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """构建绝对误差热力图。"""
        err = (pred - gt).abs()
        if err.ndim == 3:
            err = err[0]
        elif err.ndim == 4:
            err = err[0, 0]
        return self.make_colormap_image(err, cmap_name="magma")

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

    def _crop_patch_with_boundary(self, image_chw: torch.Tensor, center: torch.Tensor, patch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = int(image_chw.shape[-2]), int(image_chw.shape[-1])
        half = int(patch_size // 2)
        cy = float(center[0].item())
        cx = float(center[1].item())
        y0 = int(round(cy)) - half
        x0 = int(round(cx)) - half
        y0 = max(0, min(y0, max(h - patch_size, 0)))
        x0 = max(0, min(x0, max(w - patch_size, 0)))
        y1 = min(y0 + patch_size, h)
        x1 = min(x0 + patch_size, w)
        patch = torch.zeros((3, patch_size, patch_size), dtype=image_chw.dtype)
        crop = image_chw[:, y0:y1, x0:x1]
        patch[:, : crop.shape[-2], : crop.shape[-1]] = crop
        local = torch.tensor([cy - float(y0), cx - float(x0)], dtype=torch.float32)
        return patch, local

    def _draw_cross_on_patch(self, patch_chw: torch.Tensor, local_xy: torch.Tensor, color: tuple[int, int, int] = (255, 64, 64)) -> torch.Tensor:
        patch = patch_chw.detach().float().cpu().clamp(0, 1)
        py = int(round(float(local_xy[0].item())))
        px = int(round(float(local_xy[1].item())))
        py = max(0, min(py, int(patch.shape[-2]) - 1))
        px = max(0, min(px, int(patch.shape[-1]) - 1))
        pil = Image.fromarray((patch.permute(1, 2, 0).numpy() * 255.0).astype("uint8"))
        draw = ImageDraw.Draw(pil)
        arm = 6
        draw.line([(px - arm, py), (px + arm, py)], fill=color, width=2)
        draw.line([(px, py - arm), (px, py + arm)], fill=color, width=2)
        out = torch.from_numpy(np.array(pil)).permute(2, 0, 1).to(torch.float32) / 255.0
        return out.clamp(0, 1)

    def _make_patch_match_prediction_panel(
        self,
        batch: dict[str, Any],
        aux: dict[str, Any],
        *,
        global_step: int,
    ) -> tuple[torch.Tensor, str] | None:
        vis = aux.get("patch_match_vis", None)
        if not isinstance(vis, dict) or "images" not in batch:
            return None
        try:
            bi = int(vis.get("batch_index", 0))
            src_view = int(vis.get("src_view_index", 0))
            tgt_view = int(vis.get("tgt_view_index", 0))
            src_points = vis.get("src_points", None)
            tgt_points_gt = vis.get("tgt_points_gt", None)
            tgt_points_pred = vis.get("tgt_points_pred", None)
            if not torch.is_tensor(src_points) or not torch.is_tensor(tgt_points_gt) or not torch.is_tensor(tgt_points_pred):
                return None
            images = batch["images"]
            if images.ndim != 5:
                return None
            b, v = int(images.shape[0]), int(images.shape[1])
            if bi < 0 or bi >= b or src_view < 0 or src_view >= v or tgt_view < 0 or tgt_view >= v:
                return None
        except Exception:
            return None

        n = min(int(src_points.shape[0]), int(tgt_points_gt.shape[0]), int(tgt_points_pred.shape[0]))
        if n <= 0:
            return None
        src_points = src_points[:n].detach().float().cpu()
        tgt_points_gt = tgt_points_gt[:n].detach().float().cpu()
        tgt_points_pred = tgt_points_pred[:n].detach().float().cpu()

        src_img = batch["images"][bi, src_view].detach().float().cpu().clamp(0, 1)
        tgt_img = batch["images"][bi, tgt_view].detach().float().cpu().clamp(0, 1)
        hs, ws = int(src_img.shape[-2]), int(src_img.shape[-1])
        ht, wt = int(tgt_img.shape[-2]), int(tgt_img.shape[-1])

        valid = (
            torch.isfinite(src_points).all(dim=-1)
            & torch.isfinite(tgt_points_gt).all(dim=-1)
            & torch.isfinite(tgt_points_pred).all(dim=-1)
            & (src_points[:, 0] >= 0)
            & (src_points[:, 0] <= hs - 1)
            & (src_points[:, 1] >= 0)
            & (src_points[:, 1] <= ws - 1)
            & (tgt_points_gt[:, 0] >= 0)
            & (tgt_points_gt[:, 0] <= ht - 1)
            & (tgt_points_gt[:, 1] >= 0)
            & (tgt_points_gt[:, 1] <= wt - 1)
        )
        if not bool(valid.any()):
            return None
        src_points = src_points[valid]
        tgt_points_gt = tgt_points_gt[valid]
        tgt_points_pred = tgt_points_pred[valid]
        n_valid = int(src_points.shape[0])
        n_draw = min(16, n_valid)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(global_step) + 20260424 + bi * 97 + src_view * 37 + tgt_view * 13)
        if n_valid > n_draw:
            idx = torch.randperm(n_valid, generator=gen)[:n_draw]
            src_points = src_points[idx]
            tgt_points_gt = tgt_points_gt[idx]
            tgt_points_pred = tgt_points_pred[idx]

        src_canvas = torch.zeros((3, 512, 512), dtype=torch.float32)
        tgt_canvas = torch.zeros((3, 512, 512), dtype=torch.float32)
        patch_size = 128
        for k in range(int(src_points.shape[0])):
            r = k // 4
            c = k % 4
            y0 = r * patch_size
            x0 = c * patch_size
            src_patch, src_local = self._crop_patch_with_boundary(src_img, src_points[k], patch_size=patch_size)
            tgt_patch, tgt_local_gt = self._crop_patch_with_boundary(tgt_img, tgt_points_gt[k], patch_size=patch_size)
            pred_local = torch.tensor(
                [
                    float(tgt_local_gt[0].item()) + float(tgt_points_pred[k, 0].item()) - float(tgt_points_gt[k, 0].item()),
                    float(tgt_local_gt[1].item()) + float(tgt_points_pred[k, 1].item()) - float(tgt_points_gt[k, 1].item()),
                ],
                dtype=torch.float32,
            )
            pred_local[0] = pred_local[0].clamp(0, patch_size - 1)
            pred_local[1] = pred_local[1].clamp(0, patch_size - 1)
            src_canvas[:, y0 : y0 + patch_size, x0 : x0 + patch_size] = self._draw_cross_on_patch(src_patch, src_local, color=(64, 255, 64))
            tgt_canvas[:, y0 : y0 + patch_size, x0 : x0 + patch_size] = self._draw_cross_on_patch(tgt_patch, pred_local, color=(255, 64, 64))

        panel = torch.cat([src_canvas, tgt_canvas], dim=-1).clamp(0, 1)
        view_ids = batch.get("view_ids", None)
        if torch.is_tensor(view_ids) and view_ids.ndim >= 2 and bi < int(view_ids.shape[0]):
            src_view_id = int(view_ids[bi, src_view].item())
            tgt_view_id = int(view_ids[bi, tgt_view].item())
        else:
            src_view_id = int(src_view)
            tgt_view_id = int(tgt_view)
        panel = self._draw_label_on_image(panel, f"src view={src_view} (id={src_view_id})", xy=(6, 6))
        panel = self._draw_label_on_image(panel, f"tgt view={tgt_view} (id={tgt_view_id})", xy=(518, 6))
        meta = (
            f"batch_index={bi}, src_view_index={src_view}, tgt_view_index={tgt_view}, "
            f"src_view_id={src_view_id}, tgt_view_id={tgt_view_id}, "
            f"num_pairs_selected_view_pair={int(vis.get('num_pairs_selected_view_pair', n_valid))}, "
            f"num_pairs_total_after_cap={int(vis.get('num_pairs_total_after_cap', n_valid))}, "
            f"num_pairs_total_before_cap={int(vis.get('num_pairs_total_before_cap', n_valid))}, "
            f"num_pairs_drawn={int(src_points.shape[0])}"
        )
        return panel, meta

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
        if not self.is_enabled:
            return

        images = batch["images"]  # [B,V,3,H,W]
        view_ids = batch.get("view_ids", None)
        height_gt = batch.get("height_gt", None)
        pred_h = outputs.get("height_abs", None)
        pred_p = outputs.get("point_abs", None)

        # 兼容旧面板（第一个样本） + 新面板（整个 batch 的全部视图）。
        self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/input_rgb", self.make_rgb_montage(images[0]), global_step), tag=f"vis/{split}/input_rgb")
        b, v = int(images.shape[0]), int(images.shape[1])
        labeled_rows = []
        for bi in range(b):
            row = []
            for vi in range(v):
                vid = int(view_ids[bi, vi].item()) if torch.is_tensor(view_ids) else vi
                row.append(self._draw_label_on_image(images[bi, vi], f"view={vid}"))
            labeled_rows.append(torch.cat(row, dim=-1))
        self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/I_crop", torch.cat(labeled_rows, dim=-2), global_step), tag=f"vis/{split}/I_crop")

        if "affine_gt_forward" in batch:
            aff_fwd = batch["affine_gt_forward"]
            aff_corr = batch.get("affine_gt_correction", None)
            warped = []
            for bi in range(b):
                warped_row = []
                for vi in range(v):
                    aff_corr_i = aff_corr[bi, vi] if aff_corr is not None else None
                    wi = self._warp_crop_by_affine_forward(images[bi, vi], aff_fwd[bi, vi], affine_correction_2x3=aff_corr_i)
                    vid = int(view_ids[bi, vi].item()) if torch.is_tensor(view_ids) else vi
                    warped_row.append(self._draw_label_on_image(wi, f"view={vid}"))
                warped.append(torch.cat(warped_row, dim=-1))
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/I_crop_af", torch.cat(warped, dim=-2), global_step), tag=f"vis/{split}/I_crop_af")

        if "anchor_line_samp_true" in batch:
            anchors = batch["anchor_line_samp_true"]  # [B,V,K,2]
            b = int(images.shape[0])
            v = min(int(images.shape[1]), int(anchors.shape[1]))
            overlays = []
            for bi in range(b):
                row = [self._overlay_points_on_image(images[bi, vi], anchors[bi, vi]) for vi in range(v)]
                overlays.append(torch.cat(row, dim=-1))
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/crop_anchor_overlay", torch.cat(overlays, dim=-2), global_step), tag=f"vis/{split}/crop_anchor_overlay")

        if height_gt is not None and pred_h is not None:
            gt = height_gt[0, 0]
            ph = pred_h[0, 0]
            hmask = batch.get("height_valid_mask", None)
            vm = hmask[0, 0] if torch.is_tensor(hmask) else None
            vmin, vmax = self._group_vrange([gt, ph], masks=[vm, vm], q_low=0.05, q_high=0.95)
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/height_gt", self.make_colormap_image(gt, vmin=vmin, vmax=vmax), global_step), tag=f"vis/{split}/height_gt")
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/height_pred", self.make_colormap_image(ph, vmin=vmin, vmax=vmax), global_step), tag=f"vis/{split}/height_pred")
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/height_error", self.make_abs_error_map(ph, gt), global_step), tag=f"vis/{split}/height_error")

        if pred_p is not None and aux.get("gt_point_map", None) is not None:
            pz = pred_p[0, 0, 2]
            gz = aux["gt_point_map"][0, 0, 2]
            hmask = batch.get("height_valid_mask", None)
            vm = hmask[0, 0] if torch.is_tensor(hmask) else None
            vmin, vmax = self._group_vrange([pz, gz], masks=[vm, vm], q_low=0.05, q_high=0.95)
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/point_z_pred", self.make_colormap_image(pz, vmin=vmin, vmax=vmax), global_step), tag=f"vis/{split}/point_z_pred")
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/point_z_gt", self.make_colormap_image(gz, vmin=vmin, vmax=vmax), global_step), tag=f"vis/{split}/point_z_gt")
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/point_z_error", self.make_abs_error_map(pz, gz), global_step), tag=f"vis/{split}/point_z_error")

        if render_outputs is not None and render_outputs.get("rpc", None) is not None:
            rr = render_outputs["rpc"]
            rp = render_outputs["point"]
            if rr.get("num_targets", 0) > 0:
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_rpc_rgb", rr["rendered_rgb"][0], global_step), tag=f"vis/{split}/render_rpc_rgb")
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_target_rgb", rr["target_rgb"][0], global_step), tag=f"vis/{split}/render_target_rgb")
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_rpc_alpha", rr["rendered_alpha"][0], global_step), tag=f"vis/{split}/render_rpc_alpha")
                if not rr["rendered_height"] is None:
                    rh_maps = [rr["rendered_height"][0]]
                    if rp is not None and rp.get("rendered_height", None) is not None:
                        rh_maps.append(rp["rendered_height"][0])
                    rvmin, rvmax = self._group_vrange(rh_maps, q_low=0.05, q_high=0.95)
                    self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_rpc_height", self.make_colormap_image(rr["rendered_height"][0], vmin=rvmin, vmax=rvmax), global_step), tag=f"vis/{split}/render_rpc_height")
            if rp.get("num_targets", 0) > 0:
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_point_rgb", rp["rendered_rgb"][0], global_step), tag=f"vis/{split}/render_point_rgb")
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_point_alpha", rp["rendered_alpha"][0], global_step), tag=f"vis/{split}/render_point_alpha")
                if not rp["rendered_height"] is None:
                    rh_maps = [rp["rendered_height"][0]]
                    if rr is not None and rr.get("rendered_height", None) is not None:
                        rh_maps.append(rr["rendered_height"][0])
                    rvmin, rvmax = self._group_vrange(rh_maps, q_low=0.05, q_high=0.95)
                    self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/render_point_height", self.make_colormap_image(rp["rendered_height"][0], vmin=rvmin, vmax=rvmax), global_step), tag=f"vis/{split}/render_point_height")

        if "gaussian_confidence_rpc" in outputs:
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/gaussian_conf_rpc", self.make_colormap_image(outputs["gaussian_confidence_rpc"][0, 0]), global_step), tag=f"vis/{split}/gaussian_conf_rpc")
        if "gaussian_confidence_point" in outputs:
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/gaussian_conf_point", self.make_colormap_image(outputs["gaussian_confidence_point"][0, 0]), global_step), tag=f"vis/{split}/gaussian_conf_point")
        if "gaussian_opacity" in outputs:
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/gaussian_opacity", self.make_colormap_image(outputs["gaussian_opacity"][0, 0]), global_step), tag=f"vis/{split}/gaussian_opacity")
        if "gaussian_scale" in outputs:
            mag = outputs["gaussian_scale"][0, 0].norm(dim=0, keepdim=True)
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/gaussian_scale_mag", self.make_colormap_image(mag), global_step), tag=f"vis/{split}/gaussian_scale_mag")
        if "gaussian_centers_rpc" in outputs and "gaussian_centers_point" in outputs:
            d = (outputs["gaussian_centers_rpc"][0, 0] - outputs["gaussian_centers_point"][0, 0]).norm(dim=0, keepdim=True)
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/center_disagreement", self.make_colormap_image(d), global_step), tag=f"vis/{split}/center_disagreement")

        if "affine_gt_forward" in batch and "affine_pred" in outputs:
            h, w = images.shape[-2:]
            hm = self.make_affine_residual_heatmap(batch["affine_gt_forward"][0, 0], outputs["affine_pred"][0, 0], h, w)
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/affine_residual", hm, global_step), tag=f"vis/{split}/affine_residual")

            checker_pack = self._make_pairwise_affine_rpc_checkerboard(batch, outputs, global_step=global_step)
            if checker_pack is not None:
                checker, image_j_corr, image_i_to_j, (i, j) = checker_pack
                if torch.is_tensor(view_ids):
                    view_i = int(view_ids[0, i].item())
                    view_j = int(view_ids[0, j].item())
                else:
                    view_i, view_j = int(i), int(j)
                checker_labeled = self._draw_label_on_image(checker, f"checker: view_i={view_i} / view_j={view_j}")
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/pair_rpc_j_corr", image_j_corr, global_step), tag=f"vis/{split}/pair_rpc_j_corr")
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/pair_rpc_i_to_j", image_i_to_j, global_step), tag=f"vis/{split}/pair_rpc_i_to_j")
                self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/pair_rpc_checker", checker_labeled, global_step), tag=f"vis/{split}/pair_rpc_checker")
                self._safe_writer_call("add_text", lambda: self.writer.add_text(f"vis/{split}/pair_rpc_meta", f"pair(i,j)=({i},{j}), view=({view_i},{view_j})", global_step), tag=f"vis/{split}/pair_rpc_meta")

        pairwise = self.make_pairwise_error_matrix(aux, v=int(images.shape[0]))
        if pairwise is not None:
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/pairwise_error_matrix", pairwise, global_step), tag=f"vis/{split}/pairwise_error_matrix")

        patch_match_panel = self._make_patch_match_prediction_panel(batch, aux, global_step=global_step)
        if patch_match_panel is not None:
            panel, meta_text = patch_match_panel
            self._safe_writer_call("add_image", lambda: self.writer.add_image(f"vis/{split}/patch_match_pred_patch_grid", panel, global_step), tag=f"vis/{split}/patch_match_pred_patch_grid")
            self._safe_writer_call("add_text", lambda: self.writer.add_text(f"vis/{split}/patch_match_meta", meta_text, global_step), tag=f"vis/{split}/patch_match_meta")

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
        if not self.is_enabled:
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
                ok = self._safe_writer_call(
                    "add_mesh",
                    lambda: self.writer.add_mesh(tag, vertices=xyz.unsqueeze(0), colors=(rgb.clamp(0, 1) * 255).to(torch.uint8).unsqueeze(0), global_step=global_step),
                    tag=tag,
                )
                if ok:
                    return
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
            self._safe_writer_call("add_image", lambda: self.writer.add_image(tag + "_bev", bev, global_step), tag=tag + "_bev")

        if gt is not None:
            _mesh(f"vis/{split}/pc_gt", gt)
        _mesh(f"vis/{split}/pc_pred", pred)
        _mesh(f"vis/{split}/pc_rpc_center", rpc)

    def log_optimizer_state(self, state: dict[str, Any], global_step: int) -> None:
        """记录优化器与系统状态。"""
        self.log_scalars("optim", state, global_step)

    def log_text(self, tag: str, text: str, global_step: int) -> None:
        """记录文本事件。"""
        if self.is_enabled:
            self._safe_writer_call("add_text", lambda: self.writer.add_text(tag, text, global_step), tag=tag)

    def flush(self) -> None:
        """强制 flush。"""
        if self.is_enabled:
            self._safe_writer_call("flush", lambda: self.writer.flush())

    def close(self) -> None:
        """关闭 writer。"""
        if self.is_enabled:
            self._close_writer_noexcept(op="close")

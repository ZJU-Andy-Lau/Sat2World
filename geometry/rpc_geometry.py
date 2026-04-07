"""rpc_geometry.py

本文件实现 RPCGeometryOps：Sat2World 的 RPC 工程接口层。
核心原则：
1) 不重写 rpc.py 数学；
2) 统一 batch/list 组织、dtype/设备管理；
3) 为模型提供稳定、清晰、可复用的几何调用接口。
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch

from geometry.rpc import RPCModelParameterTorch
from geometry.scene_geometry import make_image_grid


class RPCGeometryOps:
    """RPC 工程化操作集合。

    该类负责在模型与 rpc.py 之间建立稳定桥接层：
    - RPC 对象安全复制；
    - 仿射校正的批量应用；
    - 基于 RPC 的投影/反投影批量接口；
    - patch 级几何特征提取；
    - 两条高斯中心路径所需的 3D 中心构造。

    成员变量:
        rpc_dtype: RPC 内部计算 dtype，默认 torch.double。
        net_dtype: 输出到网络时的 dtype，默认 torch.float32。
    """

    def __init__(
        self,
        rpc_dtype: torch.dtype = torch.double,
        net_dtype: torch.dtype = torch.float32,
    ) -> None:
        """初始化 RPCGeometryOps。

        参数:
            rpc_dtype: RPC 相关计算精度，建议 torch.double。
            net_dtype: 返回网络张量时使用的精度，建议 torch.float32。
        """
        self.rpc_dtype = rpc_dtype
        self.net_dtype = net_dtype

    def clone_rpc(self, rpc_obj: RPCModelParameterTorch) -> RPCModelParameterTorch:
        """安全复制单个 RPC 对象，避免修改原对象。"""
        return copy.deepcopy(rpc_obj)

    def clone_rpc_batch(
        self,
        rpc_batch: Sequence[Sequence[RPCModelParameterTorch]],
    ) -> list[list[RPCModelParameterTorch]]:
        """安全复制 RPC batch。

        参数:
            rpc_batch: 长度为 B 的列表，每个元素是长度为 V 的 RPC 列表。

        返回:
            cloned: 同结构深拷贝列表。
        """
        return [[self.clone_rpc(rpc_obj) for rpc_obj in rpc_views] for rpc_views in rpc_batch]

    def apply_affine_correction(
        self,
        rpc_obj: RPCModelParameterTorch,
        affine_correction: torch.Tensor,
    ) -> RPCModelParameterTorch:
        """对单个 RPC 应用仿射校正，并返回新对象。

        约定:
            affine_correction 是“把当前观测像素坐标校正回原始正确坐标”的矩阵，
            与 rpc.py 中 `Update_Adjust` 方向保持一致。

        参数:
            rpc_obj: 原始 RPC 对象。
            affine_correction: [2,3]。

        返回:
            corrected_rpc: clone 后应用校正的新 RPC。
        """
        if affine_correction.shape != (2, 3):
            raise ValueError(f"affine_correction must be [2,3], got {tuple(affine_correction.shape)}")

        corrected = self.clone_rpc(rpc_obj)
        corrected.Update_Adjust(affine_correction.detach().to(dtype=self.rpc_dtype, device=corrected.device))
        return corrected

    def apply_affine_correction_batch(
        self,
        rpc_batch: Sequence[Sequence[RPCModelParameterTorch]],
        affine_batch: torch.Tensor,
    ) -> list[list[RPCModelParameterTorch]]:
        """批量应用仿射校正。

        参数:
            rpc_batch: [B][V] RPC 列表。
            affine_batch: [B,V,2,3]。

        返回:
            corrected_batch: [B][V] 新 RPC 列表（深拷贝后更新）。
        """
        if affine_batch.ndim != 4 or affine_batch.shape[-2:] != (2, 3):
            raise ValueError(f"affine_batch must be [B,V,2,3], got {tuple(affine_batch.shape)}")

        b, v = affine_batch.shape[:2]
        if len(rpc_batch) != b:
            raise ValueError(f"rpc_batch B mismatch: {len(rpc_batch)} vs {b}")

        out: list[list[RPCModelParameterTorch]] = []
        for bi in range(b):
            if len(rpc_batch[bi]) != v:
                raise ValueError(f"rpc_batch[{bi}] view count mismatch")
            out_views: list[RPCModelParameterTorch] = []
            for vi in range(v):
                out_views.append(self.apply_affine_correction(rpc_batch[bi][vi], affine_batch[bi, vi]))
            out.append(out_views)
        return out

    def linesamp_to_xy(
        self,
        rpc_obj: RPCModelParameterTorch,
        lines: torch.Tensor,
        samps: torch.Tensor,
        heights: torch.Tensor,
        *,
        xy_center: Optional[torch.Tensor | Sequence[float]] = None,
        xy_scale: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单 RPC: (line,samp,h) -> (x,y)。

        返回张量与输入 `heights` 保持可微关系（若 heights requires_grad=True）。
        """
        x, y = rpc_obj.RPC_LINESAMP2XY(
            line_in=lines.to(dtype=self.rpc_dtype, device=rpc_obj.device),
            samp_in=samps.to(dtype=self.rpc_dtype, device=rpc_obj.device),
            h_in=heights.to(dtype=self.rpc_dtype, device=rpc_obj.device),
            output_type="tensor",
            xy_center=self._to_rpc_param(xy_center, rpc_obj.device),
            xy_scale=self._to_rpc_param(xy_scale, rpc_obj.device),
        )
        return x, y

    def xy_to_linesamp(
        self,
        rpc_obj: RPCModelParameterTorch,
        xs: torch.Tensor,
        ys: torch.Tensor,
        heights: torch.Tensor,
        *,
        xy_center: Optional[torch.Tensor | Sequence[float]] = None,
        xy_scale: Optional[torch.Tensor | Sequence[float]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单 RPC: (x,y,h) -> (line,samp)。"""
        lines, samps = rpc_obj.RPC_XY2LINESAMP(
            x_in=xs.to(dtype=self.rpc_dtype, device=rpc_obj.device),
            y_in=ys.to(dtype=self.rpc_dtype, device=rpc_obj.device),
            h_in=heights.to(dtype=self.rpc_dtype, device=rpc_obj.device),
            output_type="tensor",
            xy_center=self._to_rpc_param(xy_center, rpc_obj.device),
            xy_scale=self._to_rpc_param(xy_scale, rpc_obj.device),
        )
        return lines, samps

    def linesamp_to_xy_batch(
        self,
        rpc_batch: Sequence[Sequence[RPCModelParameterTorch]],
        lines: torch.Tensor,
        samps: torch.Tensor,
        heights: torch.Tensor,
        *,
        scene_xy_center: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        scene_xy_scale: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """批量 RPC 反投影接口。

        参数:
            rpc_batch: [B][V]。
            lines/samps/heights: [B,V,N]。
            scene_xy_center/scene_xy_scale: 每个 batch 一个 (y,x)。

        返回:
            xs, ys: 均为 [B,V,N]，float32。
        """
        b, v, n = lines.shape
        centers, scales = self._normalize_scene_xy(scene_xy_center, scene_xy_scale, b, lines.device)

        xs = torch.empty((b, v, n), device=lines.device, dtype=self.net_dtype)
        ys = torch.empty((b, v, n), device=lines.device, dtype=self.net_dtype)

        for bi in range(b):
            for vi in range(v):
                x, y = self.linesamp_to_xy(
                    rpc_batch[bi][vi],
                    lines[bi, vi],
                    samps[bi, vi],
                    heights[bi, vi],
                    xy_center=centers[bi],
                    xy_scale=scales[bi],
                )
                xs[bi, vi] = x.to(dtype=self.net_dtype, device=lines.device)
                ys[bi, vi] = y.to(dtype=self.net_dtype, device=lines.device)
        return xs, ys

    def xy_to_linesamp_batch(
        self,
        rpc_batch: Sequence[Sequence[RPCModelParameterTorch]],
        xs: torch.Tensor,
        ys: torch.Tensor,
        heights: torch.Tensor,
        *,
        scene_xy_center: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        scene_xy_scale: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """批量 RPC 正投影接口。"""
        b, v, n = xs.shape
        centers, scales = self._normalize_scene_xy(scene_xy_center, scene_xy_scale, b, xs.device)

        lines = torch.empty((b, v, n), device=xs.device, dtype=self.net_dtype)
        samps = torch.empty((b, v, n), device=xs.device, dtype=self.net_dtype)

        for bi in range(b):
            for vi in range(v):
                l, s = self.xy_to_linesamp(
                    rpc_batch[bi][vi],
                    xs[bi, vi],
                    ys[bi, vi],
                    heights[bi, vi],
                    xy_center=centers[bi],
                    xy_scale=scales[bi],
                )
                lines[bi, vi] = l.to(dtype=self.net_dtype, device=xs.device)
                samps[bi, vi] = s.to(dtype=self.net_dtype, device=xs.device)
        return lines, samps

    def compute_patch_geometry_features_batch(
        self,
        rpc_batch: Sequence[Sequence[RPCModelParameterTorch]],
        patch_centers: torch.Tensor,
        height_ref: torch.Tensor,
        *,
        scene_xy_center: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        scene_xy_scale: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
    ) -> torch.Tensor:
        """批量计算 patch 级 43 维几何特征。

        参数:
            rpc_batch: [B][V] RPC 列表。
            patch_centers: [N,2] 或 [B,V,N,2]，顺序 (line,samp)。
            height_ref: [B,V]。
            scene_xy_center/scene_xy_scale: [B,2] 或等价 list。

        返回:
            geom_feat: [B,V,N,43]，float32。

        说明:
            本函数默认不对 affine 保梯度，内部使用 no_grad 调用几何特征接口，
            以控制显存与图复杂度。
        """
        if height_ref.ndim != 2:
            raise ValueError(f"height_ref must be [B,V], got {tuple(height_ref.shape)}")

        b, v = height_ref.shape
        centers, scales = self._normalize_scene_xy(scene_xy_center, scene_xy_scale, b, height_ref.device)

        if patch_centers.ndim == 2:
            n = patch_centers.shape[0]
            centers_bv = patch_centers[None, None].expand(b, v, n, 2)
        elif patch_centers.ndim == 4:
            if tuple(patch_centers.shape[:2]) != (b, v):
                raise ValueError("patch_centers [B,V,N,2] shape mismatch")
            n = patch_centers.shape[2]
            centers_bv = patch_centers
        else:
            raise ValueError(f"Unsupported patch_centers shape: {tuple(patch_centers.shape)}")

        out = torch.empty((b, v, n, 43), device=height_ref.device, dtype=self.net_dtype)
        h_offsets = torch.tensor([-50.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 50.0], dtype=self.rpc_dtype)

        for bi in range(b):
            for vi in range(v):
                rpc_obj = rpc_batch[bi][vi]
                coords = centers_bv[bi, vi].to(dtype=self.rpc_dtype, device=rpc_obj.device)
                h_ref_scalar = float(height_ref[bi, vi].item())
                dem = torch.full((n,), h_ref_scalar, dtype=self.rpc_dtype, device=rpc_obj.device)
                xy_center = self._to_rpc_param(centers[bi], rpc_obj.device)
                xy_scale = self._to_rpc_param(scales[bi], rpc_obj.device)

                feat = rpc_obj.compute_geometry_features(
                    Coords=coords,
                    dem=dem,
                    xy_center=xy_center,
                    xy_scale=xy_scale,
                )
                h_ref_digits = self._encode_height_ref_digits(h_ref_scalar, device=rpc_obj.device)
                h_ref_digits_n = h_ref_digits.unsqueeze(0).expand(n, -1)  # [N,5]

                h_anchors = dem.unsqueeze(1) + h_offsets.to(device=rpc_obj.device).unsqueeze(0)  # [N,9]
                line_rep = coords[:, 0].unsqueeze(1).expand(-1, 9).reshape(-1)
                samp_rep = coords[:, 1].unsqueeze(1).expand(-1, 9).reshape(-1)
                h_rep = h_anchors.reshape(-1)
                x_norm_rep, y_norm_rep = rpc_obj.RPC_LINESAMP2XY(
                    line_in=line_rep,
                    samp_in=samp_rep,
                    h_in=h_rep,
                    output_type="tensor",
                    xy_center=xy_center,
                    xy_scale=xy_scale,
                )
                xy_anchors = torch.stack([x_norm_rep, y_norm_rep], dim=-1).view(n, 9, 2).reshape(n, 18)  # [N,18]

                feat43 = torch.cat([feat, h_ref_digits_n, xy_anchors], dim=-1)
                out[bi, vi] = feat43.detach().to(dtype=self.net_dtype, device=height_ref.device)

        return out

    def _encode_height_ref_digits(self, h_ref: float, *, device: torch.device) -> torch.Tensor:
        """将 h_ref 编码为 5 维：[千位, 百位, 十位, 个位, 小数]，并归一化到 [0,1]。"""
        h_abs = abs(float(h_ref))
        int_part = int(h_abs)
        frac_part = h_abs - float(int_part)
        sym = h_ref / h_abs

        thousands = (int_part // 1000) % 10
        hundreds = (int_part // 100) % 10
        tens = (int_part // 10) % 10
        ones = int_part % 10

        return torch.tensor(
            [
                sym * float(thousands) / 9.0,
                sym * float(hundreds) / 9.0,
                sym * float(tens) / 9.0,
                sym * float(ones) / 9.0,
                sym * float(frac_part),
            ],
            dtype=self.rpc_dtype,
            device=device,
        )

    def centers_from_rpc_and_height_batch(
        self,
        corrected_rpc_batch: Sequence[Sequence[RPCModelParameterTorch]],
        pixel_grid: torch.Tensor,
        height_abs: torch.Tensor,
        *,
        scene_xy_center: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        scene_xy_scale: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        downsample_factor: int = 1,
    ) -> torch.Tensor:
        """由 corrected RPC + 绝对高程图生成三维中心。

        参数:
            corrected_rpc_batch: [B][V]。
            pixel_grid: [H,W,2]，(line,samp)。
            height_abs: [B,V,1,H,W]，必须保留梯度。

        返回:
            centers: [B,V,3,H,W]，通道顺序 (x,y,h)。
        """
        if pixel_grid.ndim != 3 or pixel_grid.shape[-1] != 2:
            raise ValueError("pixel_grid must be [H,W,2]")
        if height_abs.ndim != 5:
            raise ValueError("height_abs must be [B,V,1,H,W]")

        b, v, _, h, w = height_abs.shape
        ds = max(int(downsample_factor), 1)
        if ds > 1:
            h_ds = max(h // ds, 1)
            w_ds = max(w // ds, 1)
            height_abs_work = torch.nn.functional.interpolate(
                height_abs.view(b * v, 1, h, w),
                size=(h_ds, w_ds),
                mode="bilinear",
                align_corners=False,
            ).view(b, v, 1, h_ds, w_ds)
            # 关键：下采样后仍需保持“原图坐标系”的 line/samp 射线。
            # 不能重建 0..h_ds-1 / 0..w_ds-1 的局部网格，否则会把射线压缩到左上角。
            pixel_grid_full = pixel_grid.to(device=height_abs.device, dtype=height_abs.dtype)
            pixel_grid_work = torch.nn.functional.interpolate(
                pixel_grid_full.permute(2, 0, 1).unsqueeze(0),  # [1,2,H,W]
                size=(h_ds, w_ds),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0).contiguous()  # [h_ds,w_ds,2]
        else:
            h_ds, w_ds = h, w
            height_abs_work = height_abs
            pixel_grid_work = pixel_grid

        centers, scales = self._normalize_scene_xy(scene_xy_center, scene_xy_scale, b, height_abs.device)

        line = pixel_grid_work[..., 0].reshape(-1).to(device=height_abs.device)
        samp = pixel_grid_work[..., 1].reshape(-1).to(device=height_abs.device)

        out = torch.empty((b, v, 3, h_ds, w_ds), device=height_abs.device, dtype=self.net_dtype)

        for bi in range(b):
            for vi in range(v):
                rpc_obj = corrected_rpc_batch[bi][vi]
                h_flat = height_abs_work[bi, vi, 0].reshape(-1)

                x, y = self.linesamp_to_xy(
                    rpc_obj,
                    lines=line,
                    samps=samp,
                    heights=h_flat,
                    xy_center=centers[bi],
                    xy_scale=scales[bi],
                )
                x = x.reshape(h_ds, w_ds).to(dtype=self.net_dtype, device=height_abs.device)
                y = y.reshape(h_ds, w_ds).to(dtype=self.net_dtype, device=height_abs.device)
                z = height_abs_work[bi, vi, 0].to(dtype=self.net_dtype)
                out[bi, vi] = torch.stack([x, y, z], dim=0)

        if ds > 1:
            out = torch.nn.functional.interpolate(
                out.view(b * v, 3, h_ds, w_ds),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).view(b, v, 3, h, w)

        return out

    def build_point_anchor_map_batch(
        self,
        rpc_init_batch: Sequence[Sequence[RPCModelParameterTorch]],
        pixel_grid: torch.Tensor,
        height_ref: torch.Tensor,
        *,
        scene_xy_center: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
        scene_xy_scale: Optional[torch.Tensor | Sequence[Sequence[float]]] = None,
    ) -> torch.Tensor:
        """基于初始 RPC 与 h_ref 构造点云绝对锚点。

        参数:
            rpc_init_batch: [B][V] 初始 RPC。
            pixel_grid: [H,W,2]。
            height_ref: [B,V]。

        返回:
            point_anchor: [B,V,3,H,W]，(x,y,h_ref)。
        """
        if pixel_grid.ndim != 3 or pixel_grid.shape[-1] != 2:
            raise ValueError("pixel_grid must be [H,W,2]")
        if height_ref.ndim != 2:
            raise ValueError("height_ref must be [B,V]")

        b, v = height_ref.shape
        h, w = pixel_grid.shape[:2]
        centers, scales = self._normalize_scene_xy(scene_xy_center, scene_xy_scale, b, height_ref.device)

        line = pixel_grid[..., 0].reshape(-1)
        samp = pixel_grid[..., 1].reshape(-1)

        out = torch.empty((b, v, 3, h, w), device=height_ref.device, dtype=self.net_dtype)

        for bi in range(b):
            for vi in range(v):
                rpc_obj = rpc_init_batch[bi][vi]
                h_val = float(height_ref[bi, vi].item())
                h_flat = torch.full((h * w,), h_val, device=height_ref.device, dtype=self.net_dtype)

                x, y = self.linesamp_to_xy(
                    rpc_obj,
                    lines=line,
                    samps=samp,
                    heights=h_flat,
                    xy_center=centers[bi],
                    xy_scale=scales[bi],
                )
                x = x.reshape(h, w).to(dtype=self.net_dtype, device=height_ref.device)
                y = y.reshape(h, w).to(dtype=self.net_dtype, device=height_ref.device)
                z = torch.full((h, w), h_val, device=height_ref.device, dtype=self.net_dtype)
                out[bi, vi] = torch.stack([x, y, z], dim=0)

        return out

    def _to_rpc_param(
        self,
        value: Optional[torch.Tensor | Sequence[float]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """将中心/尺度参数转为 RPC 可用 double 张量。"""
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device, dtype=self.rpc_dtype)
        return torch.as_tensor(value, device=device, dtype=self.rpc_dtype)

    def _normalize_scene_xy(
        self,
        scene_xy_center: Optional[torch.Tensor | Sequence[Sequence[float]]],
        scene_xy_scale: Optional[torch.Tensor | Sequence[Sequence[float]]],
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[Optional[torch.Tensor]], list[Optional[torch.Tensor]]]:
        """把 scene 级 (y,x) 中心/尺度整理为长度 B 的列表。"""

        def _normalize_one(
            data: Optional[torch.Tensor | Sequence[Sequence[float]]],
            name: str,
        ) -> list[Optional[torch.Tensor]]:
            if data is None:
                return [None for _ in range(batch_size)]
            if torch.is_tensor(data):
                t = data.to(device=device)
                if t.ndim == 1 and t.numel() == 2:
                    return [t for _ in range(batch_size)]
                if t.ndim == 2 and t.shape[0] == batch_size and t.shape[1] == 2:
                    return [t[i] for i in range(batch_size)]
                raise ValueError(f"{name} tensor shape must be [2] or [B,2], got {tuple(t.shape)}")

            seq = list(data)
            if len(seq) == 2 and not isinstance(seq[0], (list, tuple, torch.Tensor)):
                t = torch.as_tensor(seq, device=device, dtype=self.net_dtype)
                return [t for _ in range(batch_size)]
            if len(seq) != batch_size:
                raise ValueError(f"{name} list length must be B={batch_size}, got {len(seq)}")
            return [torch.as_tensor(seq[i], device=device, dtype=self.net_dtype) for i in range(batch_size)]

        return _normalize_one(scene_xy_center, "scene_xy_center"), _normalize_one(scene_xy_scale, "scene_xy_scale")

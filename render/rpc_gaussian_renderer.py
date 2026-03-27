"""render.rpc_gaussian_renderer

本文件实现 Sat2World 的轻量级可微渲染器，用于训练阶段的闭环监督与可视化探针。

设计定位：
1) 严格沿用 RPC 几何投影，不引入针孔近似；
2) 采用可微的双线性 splat，提供稳定梯度；
3) 支持双路径中心（RPC+高程路径、独立点云路径）分别渲染。

注意：
- 该实现是“训练基线渲染器”，强调稳定与接口清晰；
- 并非最终高保真高斯光栅器，后续可替换为更高性能/更物理一致版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from geometry import RPCGeometryOps


@dataclass
class RPCGaussianRendererCfg:
    """渲染器配置。

    成员变量：
        background:
            背景颜色标量（0~1）。渲染完成后会把前景与该背景进行 alpha 合成。
        use_confidence:
            是否启用置信度路径。如果为 True，渲染权重会保留最小权重下界，
            以减小训练初期“完全空洞像素”导致的梯度不稳定。
        confidence_floor:
            置信度/权重下界，仅在 use_confidence=True 时生效。
    """

    background: float = 0.0
    use_confidence: bool = True
    confidence_floor: float = 1e-3


class RPCGaussianRenderer:
    """RPC 投影 + 双线性 splat 的轻量渲染器。

    功能概述：
    - 接收模型输出中的高斯中心与属性；
    - 使用 corrected RPC 将 3D 中心投影回像素平面；
    - 对每个“高斯样本”执行双线性 splat，把贡献写入像素网格；
    - 输出两条路径（rpc/point）的渲染 RGB 与 alpha。

    成员变量：
        cfg:
            渲染配置实例，控制背景值、置信度开关与权重下界。
        rpc_ops:
            RPCGeometryOps 几何接口实例，负责 batched xy->linesamp 投影。
    """

    def __init__(self, cfg: RPCGaussianRendererCfg | None = None) -> None:
        """初始化渲染器。

        输入：
            cfg:
                渲染配置；为 None 时使用默认配置 RPCGaussianRendererCfg()。

        输出：
            无显式返回值；构建好渲染器内部配置与 RPC 几何操作对象。
        """
        self.cfg = cfg or RPCGaussianRendererCfg()
        self.rpc_ops = RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32)

    def _project_centers(
        self,
        centers_xyz: torch.Tensor,
        rpc_batch: list[list[Any]],
        scene_xy_center: torch.Tensor | None,
        scene_xy_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将三维中心批量投影到像素平面。

        输入：
            centers_xyz:
                三维中心，形状 [B,V,3,H,W]，通道顺序为 (x,y,z)。
            rpc_batch:
                批量 RPC 对象，结构为 list[B][V]。
            scene_xy_center:
                场景级 xy 中心，形状可为 [B,2] 或兼容形式；可为 None。
            scene_xy_scale:
                场景级 xy 尺度，形状可为 [B,2] 或兼容形式；可为 None。

        输出：
            lines, samps:
                投影后的像素坐标，形状均为 [B,V,N]，N=H*W。

        功能说明：
            该函数仅完成几何投影，不做任何栅格化与颜色融合。
        """
        b, v, _, h, w = centers_xyz.shape
        xyz = centers_xyz.permute(0, 1, 3, 4, 2).reshape(b, v, h * w, 3)
        xs = xyz[..., 0]
        ys = xyz[..., 1]
        hs = xyz[..., 2]
        lines, samps = self.rpc_ops.xy_to_linesamp_batch(
            rpc_batch=rpc_batch,
            xs=xs,
            ys=ys,
            heights=hs,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        return lines, samps

    def _rasterize_bilinear(
        self,
        lines: torch.Tensor,
        samps: torch.Tensor,
        value: torch.Tensor,
        weight: torch.Tensor,
        hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行双线性 splat 栅格化。

        输入：
            lines/samps:
                每个样本的投影坐标，形状 [B,V,N]。
            value:
                待累积数值，形状 [B,V,C,N]，例如 RGB*alpha。
            weight:
                累积权重，形状 [B,V,1,N]，通常为 alpha（或其修正）。
            hw:
                输出图像大小 (H,W)。

        输出：
            out:
                分子累积图，形状 [B,V,C,H,W]。
            acc:
                权重累积图，形状 [B,V,1,H,W]。

        功能说明：
            - 对每个样本在四个邻域像素执行双线性分配；
            - 使用 scatter_add_ 进行可微累积；
            - 越界样本会被有效 mask 自动丢弃。
        """
        b, v, _ = lines.shape
        h, w = hw
        out = torch.zeros((b, v, value.shape[2], h, w), device=value.device, dtype=value.dtype)
        acc = torch.zeros((b, v, 1, h, w), device=value.device, dtype=value.dtype)

        line0 = torch.floor(lines)
        samp0 = torch.floor(samps)
        dy = (lines - line0).clamp(0.0, 1.0)
        dx = (samps - samp0).clamp(0.0, 1.0)

        line0 = line0.long()
        samp0 = samp0.long()
        corners = (
            (line0, samp0, (1.0 - dy) * (1.0 - dx)),
            (line0 + 1, samp0, dy * (1.0 - dx)),
            (line0, samp0 + 1, (1.0 - dy) * dx),
            (line0 + 1, samp0 + 1, dy * dx),
        )

        for yi, xi, wb in corners:
            valid = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            if not valid.any():
                continue
            flat_idx = yi * w + xi
            contrib_w = (wb * weight.squeeze(2)) * valid
            for c in range(value.shape[2]):
                contrib_v = value[:, :, c] * contrib_w
                out[:, :, c].view(b, v, -1).scatter_add_(2, flat_idx, contrib_v)
            acc[:, :, 0].view(b, v, -1).scatter_add_(2, flat_idx, contrib_w)

        return out, acc

    def render_path(
        self,
        centers_xyz: torch.Tensor,
        opacity: torch.Tensor,
        sh: torch.Tensor,
        rpc_batch: list[list[Any]],
        scene_xy_center: torch.Tensor | None,
        scene_xy_scale: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """渲染单一路径中心。

        输入：
            centers_xyz:
                当前路径的中心点，形状 [B,V,3,H,W]。
            opacity:
                不透明度，形状 [B,V,1,H,W]。
            sh:
                球谐系数，形状 [B,V,S,H,W]。本基线仅使用前 3 维近似 RGB DC 项。
            rpc_batch:
                corrected RPC，结构 list[B][V]。
            scene_xy_center/scene_xy_scale:
                场景归一化参数，可为 None。

        输出：
            dict，包含：
                rgb: [B,V,3,H,W] 渲染图；
                alpha: [B,V,1,H,W] 累积透明度；
                line/samp: [B,V,N] 投影坐标（便于调试）。

        功能说明：
            该函数把“投影 -> splat -> 归一化 -> 背景合成”串成一条路径。
        """
        b, v, _, h, w = centers_xyz.shape
        lines, samps = self._project_centers(centers_xyz, rpc_batch, scene_xy_center, scene_xy_scale)

        rgb = torch.sigmoid(sh[:, :, :3]).reshape(b, v, 3, h * w)
        alpha = opacity.reshape(b, v, 1, h * w)

        value = rgb * alpha
        weight = alpha
        if self.cfg.use_confidence:
            weight = weight.clamp_min(self.cfg.confidence_floor)

        num, den = self._rasterize_bilinear(lines, samps, value, weight, (h, w))
        rgb_render = num / den.clamp_min(1e-6)
        alpha_render = den.clamp(0.0, 1.0)

        bg = torch.full_like(rgb_render, self.cfg.background)
        rgb_comp = rgb_render * alpha_render + bg * (1.0 - alpha_render)
        return {"rgb": rgb_comp, "alpha": alpha_render, "line": lines, "samp": samps}

    def __call__(self, model_out: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """渲染双路径输出并返回统一结果。

        输入：
            model_out:
                模型输出字典，至少应包含：
                - rpc_corrected
                - gaussian_centers_rpc / gaussian_centers_point
                - gaussian_opacity
                - gaussian_sh
                - gaussian_confidence_rpc / gaussian_confidence_point
            batch:
                batch 字典，读取 scene_xy_center / scene_xy_scale 等几何上下文。

        输出：
            dict，包含：
                rgb_rpc / alpha_rpc:
                    RPC+高程路径渲染结果。
                rgb_point / alpha_point:
                    独立点云路径渲染结果。

        功能说明：
            这是训练主循环调用入口，会分别对两条中心路径执行 render_path。
        """
        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        rpc_corrected = model_out["rpc_corrected"]

        opacity = model_out["gaussian_opacity"]
        sh = model_out["gaussian_sh"]

        conf_rpc = model_out["gaussian_confidence_rpc"]
        conf_point = model_out["gaussian_confidence_point"]

        rpc_path = self.render_path(
            centers_xyz=model_out["gaussian_centers_rpc"],
            opacity=opacity * conf_rpc,
            sh=sh,
            rpc_batch=rpc_corrected,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        point_path = self.render_path(
            centers_xyz=model_out["gaussian_centers_point"],
            opacity=opacity * conf_point,
            sh=sh,
            rpc_batch=rpc_corrected,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        return {
            "rgb_rpc": rpc_path["rgb"],
            "alpha_rpc": rpc_path["alpha"],
            "rgb_point": point_path["rgb"],
            "alpha_point": point_path["alpha"],
        }

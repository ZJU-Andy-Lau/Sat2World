"""loss.sat2world_loss

本文件实现 Sat2World 的组合损失模块，覆盖以下训练目标：
1) 仿射纠正监督（affine）；
2) 绝对高程监督（height）；
3) 独立点云监督（point）；
4) 双路径渲染监督（render_rpc/render_point）；
5) 训练探针正则（probe）。

设计要点：
- 各分量保持解耦，便于阶段化调权；
- 点云 GT 使用“GT RPC + GT 高程”在线构造，保证几何一致；
- 保留 raw logits/fine 的 probe 约束接口，便于监控网络早期行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from geometry import RPCGeometryOps, make_image_grid


@dataclass
class Sat2WorldLossCfg:
    """损失权重与鲁棒项配置。

    成员变量：
        w_affine_l2:
            affine 拟合项权重（pred vs gt_correction）。
        w_affine_reg:
            affine 线性部分接近单位阵的正则权重。
        w_height:
            高程监督分量权重。
        w_point:
            点云监督分量权重。
        w_render_rpc:
            RPC 路径渲染监督权重。
        w_render_point:
            Point 路径渲染监督权重。
        w_probe:
            探针正则权重（logits 熵与 fine 幅值约束）。
        huber_delta_height:
            高程 Huber 损失 delta。
        huber_delta_point:
            点云 Huber 损失 delta。
    """

    w_affine_l2: float = 1.0
    w_affine_reg: float = 0.1
    w_height: float = 1.0
    w_point: float = 1.0
    w_render_rpc: float = 0.1
    w_render_point: float = 0.1
    w_probe: float = 0.02
    huber_delta_height: float = 2.0
    huber_delta_point: float = 3.0


class Sat2WorldLoss:
    """Sat2World 训练总损失。

    功能概述：
    - 将模型输出、batch 标签和渲染结果汇总成可反向传播的总损失；
    - 输出总损失及各子损失，便于日志监控与分阶段调参。

    成员变量：
        cfg:
            Sat2WorldLossCfg 配置实例。
        rpc_ops:
            RPCGeometryOps 接口实例，用于基于 GT RPC 构造点云真值。
    """

    def __init__(self, cfg: Sat2WorldLossCfg | None = None) -> None:
        """初始化损失模块。

        输入：
            cfg:
                损失配置；若为 None 则使用默认 Sat2WorldLossCfg()。

        输出：
            无显式返回值；完成内部配置与 RPC 工具初始化。
        """
        self.cfg = cfg or Sat2WorldLossCfg()
        self.rpc_ops = RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32)

    def _affine_loss(self, pred: torch.Tensor, gt_corr: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算 affine 子损失。

        输入：
            pred:
                预测仿射矩阵，形状 [B,V,2,3]。
            gt_corr:
                仿射纠正 GT，形状 [B,V,2,3]。

        输出：
            loss:
                affine 总分量（拟合 + 线性正则）。
            parts:
                细分项字典，包含 affine_l2、affine_reg。

        功能说明：
            - affine_l2: pred 与 gt_corr 的 smooth L1；
            - affine_reg: 线性 2x2 部分贴近单位阵的约束。
        """
        l2 = F.smooth_l1_loss(pred, gt_corr, beta=0.5)

        eye = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=pred.dtype, device=pred.device).view(1, 1, 2, 2)
        lin = pred[..., :2]
        reg = (lin - eye).abs().mean()

        loss = self.cfg.w_affine_l2 * l2 + self.cfg.w_affine_reg * reg
        return loss, {"affine_l2": l2, "affine_reg": reg}

    def _height_loss(self, pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """计算高程子损失（mask 版 Huber）。

        输入：
            pred:
                预测绝对高程，形状 [B,V,1,H,W]。
            gt:
                GT 高程，形状 [B,V,1,H,W]。
            valid_mask:
                高程有效掩码，形状 [B,V,1,H,W]，1 表示有效。

        输出：
            标量高程损失。

        功能说明：
            仅在有效像素上进行 Huber 聚合，自动按有效像素数量归一化。
        """
        diff = F.huber_loss(pred, gt, delta=self.cfg.huber_delta_height, reduction="none")
        m = valid_mask.to(diff.dtype)
        return (diff * m).sum() / m.sum().clamp_min(1.0)

    def _point_gt_from_rpc(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """用 GT RPC 与 GT 高程在线构造点云真值。

        输入：
            batch:
                训练 batch，需包含 height_gt、rpc_gt、scene_xy_center/scale。
            device:
                目标计算设备。

        输出：
            gt_point:
                物方真值点云，形状 [B,V,3,H,W]。

        功能说明：
            该构造方式与数据几何链一致，避免离线点云缓存带来的坐标系偏差。
        """
        h_gt = batch["height_gt"].to(device=device, dtype=torch.float32)
        _, _, _, h, w = h_gt.shape
        pixel_grid = make_image_grid(h, w, device=device, dtype=torch.float32)
        centers = self.rpc_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=batch["rpc_gt"],
            pixel_grid=pixel_grid,
            height_abs=h_gt,
            scene_xy_center=batch.get("scene_xy_center", None),
            scene_xy_scale=batch.get("scene_xy_scale", None),
        )
        return centers

    def _point_loss(self, pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """计算点云子损失（mask 版 Huber）。

        输入：
            pred:
                预测绝对点云，形状 [B,V,3,H,W]。
            gt:
                GT 绝对点云，形状 [B,V,3,H,W]。
            valid_mask:
                高程有效掩码 [B,V,1,H,W]，会扩展到 3 通道。

        输出：
            标量点云损失。

        功能说明：
            使用与高程一致的有效区域约束，保证监督区域对齐。
        """
        m = valid_mask.to(dtype=pred.dtype).expand_as(pred)
        diff = F.huber_loss(pred, gt, delta=self.cfg.huber_delta_point, reduction="none")
        return (diff * m).sum() / m.sum().clamp_min(1.0)

    def _render_loss(self, render_out: dict[str, torch.Tensor], target_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算双路径渲染损失。

        输入：
            render_out:
                渲染器输出，需包含 rgb_rpc 与 rgb_point。
            target_rgb:
                目标 RGB 图像，形状 [B,V,3,H,W]。

        输出：
            l_rpc:
                RPC 路径渲染损失。
            l_point:
                Point 路径渲染损失。

        功能说明：
            当前采用 L1 重建误差；后续可扩展 SSIM/感知损失等。
        """
        l_rpc = F.l1_loss(render_out["rgb_rpc"], target_rgb)
        l_point = F.l1_loss(render_out["rgb_point"], target_rgb)
        return l_rpc, l_point

    def _probe_loss(self, model_out: dict[str, Any]) -> torch.Tensor:
        """计算探针正则损失。

        输入：
            model_out:
                模型输出字典，若包含 raw logits/fine 张量会参与 probe 计算。

        输出：
            标量 probe 损失。

        功能说明：
            - 对 logits 计算熵，抑制过早塌陷；
            - 对 fine raw 计算幅值正则，抑制异常大残差；
            - 若无探针字段，则返回 0。
        """
        probes = []
        if "height_logits" in model_out:
            p = model_out["height_logits"].float()
            probs = torch.softmax(p, dim=2)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=2).mean()
            probes.append(entropy)
        if "point_logits" in model_out:
            for k in ("x", "y", "z"):
                p = model_out["point_logits"][k].float()
                probs = torch.softmax(p, dim=2)
                probes.append((-(probs * probs.clamp_min(1e-8).log()).sum(dim=2)).mean())
        if "height_fine_raw" in model_out:
            probes.append(model_out["height_fine_raw"].abs().mean())
        if "point_fine_raw" in model_out:
            probes.append(
                (model_out["point_fine_raw"]["x"].abs().mean() + model_out["point_fine_raw"]["y"].abs().mean() + model_out["point_fine_raw"]["z"].abs().mean())
                / 3.0
            )
        if not probes:
            return torch.zeros((), device=next(iter(model_out.values())).device)
        return torch.stack([x if x.ndim == 0 else x.mean() for x in probes]).mean()

    def __call__(self, model_out: dict[str, Any], batch: dict[str, Any], render_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """组合并返回总损失与各分量。

        输入：
            model_out:
                模型输出字典。
            batch:
                数据 batch 字典，包含 GT 仿射/高程/有效掩码/RPC 等。
            render_out:
                渲染器输出字典。

        输出：
            dict[str, torch.Tensor]：
                total、affine、affine_l2、affine_reg、height、point、
                render_rpc、render_point、probe。

        功能说明：
            这是训练阶段唯一公开入口：
            - 内部按固定顺序计算所有子损失；
            - 应用配置权重后得到 total；
            - 返回细分项供日志与调试使用。
        """
        device = model_out["height_abs"].device
        gt_affine = batch["affine_gt_correction"].to(device=device, dtype=torch.float32)
        gt_height = batch["height_gt"].to(device=device, dtype=torch.float32)
        gt_mask = batch["height_valid_mask"].to(device=device, dtype=torch.float32)

        l_aff, aff_parts = self._affine_loss(model_out["affine_pred"], gt_affine)
        l_height = self._height_loss(model_out["height_abs"], gt_height, gt_mask)

        gt_point = self._point_gt_from_rpc(batch, device)
        l_point = self._point_loss(model_out["point_abs"], gt_point, gt_mask)

        target_rgb = batch["images"].to(device=device, dtype=torch.float32)
        l_render_rpc, l_render_point = self._render_loss(render_out, target_rgb)

        l_probe = self._probe_loss(model_out)

        total = (
            l_aff
            + self.cfg.w_height * l_height
            + self.cfg.w_point * l_point
            + self.cfg.w_render_rpc * l_render_rpc
            + self.cfg.w_render_point * l_render_point
            + self.cfg.w_probe * l_probe
        )

        out = {
            "total": total,
            "affine": l_aff,
            "affine_l2": aff_parts["affine_l2"],
            "affine_reg": aff_parts["affine_reg"],
            "height": l_height,
            "point": l_point,
            "render_rpc": l_render_rpc,
            "render_point": l_render_point,
            "probe": l_probe,
        }
        return out

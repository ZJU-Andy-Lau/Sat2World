"""loss.render_loss

实现单路径渲染损失 RenderPathLoss。

注意：
- 本文件只处理“已经渲染好的结果字典”；
- 不依赖 renderer 的具体实现。
"""

from __future__ import annotations

import torch

from loss.common import masked_huber_loss, psnr_from_mse, ssim_loss_from_map, ssim_map


class RenderPathLoss:
    """单路径渲染损失。

    功能:
        计算 L1 + SSIM + 可选高度 Huber + alpha 正则，并输出常用渲染 probe。

    成员变量:
        w_l1/w_ssim/w_h/w_alpha: 子项权重。
        height_beta: 高程 Huber beta。
        alpha_thresh: alpha coverage 阈值。
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_ssim: float = 0.2,
        w_h: float = 0.0,
        w_alpha: float = 0.0,
        height_beta: float = 1.0,
        alpha_thresh: float = 0.1,
    ) -> None:
        """初始化渲染 loss。"""
        self.w_l1 = float(w_l1)
        self.w_ssim = float(w_ssim)
        self.w_h = float(w_h)
        self.w_alpha = float(w_alpha)
        self.height_beta = float(height_beta)
        self.alpha_thresh = float(alpha_thresh)

    def __call__(self, path_result: dict[str, torch.Tensor | None]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算单路径渲染损失。

        输入:
            path_result:
                包含 rendered_rgb/rendered_alpha/target_rgb，可选 rendered_height/target_height/target_valid_mask。

        输出:
            loss: 标量。
            probe: 渲染指标字典。
        """
        rendered_rgb = path_result.get("rendered_rgb", None)
        rendered_alpha = path_result.get("rendered_alpha", None)
        target_rgb = path_result.get("target_rgb", None)

        if rendered_rgb is None or rendered_alpha is None or target_rgb is None:
            raise ValueError("path_result must contain rendered_rgb, rendered_alpha, target_rgb")

        m = int(rendered_rgb.shape[0])
        if m == 0:
            zero = torch.zeros((), device=rendered_rgb.device, dtype=rendered_rgb.dtype)
            return zero, {
                "render_l1": zero,
                "render_ssim_loss": zero,
                "render_psnr": zero,
                "render_height_huber": zero,
                "render_alpha_mean": zero,
                "render_alpha_coverage": zero,
                "render_num_targets": torch.tensor(0.0, device=zero.device),
            }

        l1 = (rendered_rgb - target_rgb).abs().mean()
        smap = ssim_map(rendered_rgb, target_rgb)
        l_ssim = ssim_loss_from_map(smap)

        rendered_height = path_result.get("rendered_height", None)
        target_height = path_result.get("target_height", None)
        target_valid_mask = path_result.get("target_valid_mask", None)

        if rendered_height is not None and target_height is not None:
            l_height = masked_huber_loss(
                rendered_height,
                target_height,
                mask=target_valid_mask,
                beta=self.height_beta,
            )
        else:
            l_height = torch.zeros((), device=rendered_rgb.device, dtype=rendered_rgb.dtype)

        l_alpha = rendered_alpha.mean()
        total = self.w_l1 * l1 + self.w_ssim * l_ssim + self.w_h * l_height + self.w_alpha * l_alpha

        mse = ((rendered_rgb - target_rgb) ** 2).mean()
        probe = {
            "render_l1": l1.detach(),
            "render_ssim_loss": l_ssim.detach(),
            "render_psnr": psnr_from_mse(mse).detach(),
            "render_height_huber": l_height.detach(),
            "render_alpha_mean": l_alpha.detach(),
            "render_alpha_coverage": (rendered_alpha > self.alpha_thresh).to(rendered_alpha.dtype).mean().detach(),
            "render_num_targets": torch.tensor(float(m), device=rendered_rgb.device),
        }
        return total, probe

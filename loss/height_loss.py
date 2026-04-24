"""loss.height_loss

实现高程监督损失 HeightHuberLoss。
"""

from __future__ import annotations

import torch

from loss.common import masked_huber_loss, masked_l1_loss, safe_rmse


class HeightHuberLoss:
    """高程 Huber 监督。

    功能:
        对 height_abs 与 height_gt 在 valid_mask 区域计算 Huber 损失，
        并输出 rmse/mae/bias 等指标。

    方向约定提示:
        affine 方向约定仍然是
        - affine_gt_forward: true pixel -> observed pixel
        - affine_pred: observed pixel -> true pixel
        该类虽不直接使用 affine，但与同一训练目标保持一致。
    """

    def __init__(self, beta: float = 1.0) -> None:
        """初始化。

        输入:
            beta: Huber 分段阈值。
        """
        self.beta = float(beta)

    def __call__(
        self,
        height_abs: torch.Tensor,
        height_gt: torch.Tensor,
        height_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算高程监督损失与指标。

        输入:
            height_abs: [B,V,1,H,W]。
            height_gt: [B,V,1,H,W]。
            height_valid_mask: [B,V,1,H,W]。

        输出:
            loss: 标量。
            probe: 包含 height_rmse/height_mae/height_bias。
        """
        if height_abs.isnan().any():
            print(f"height abs pred has nan")
        loss = masked_huber_loss(height_abs, height_gt, mask=height_valid_mask, beta=self.beta)

        diff = height_abs - height_gt
        probe = {
            "height_rmse": safe_rmse(diff.square(), mask=height_valid_mask).detach(),
            "height_mae": masked_l1_loss(height_abs, height_gt, mask=height_valid_mask).detach(),
            "height_bias": (diff * height_valid_mask.to(diff.dtype)).sum().detach() / height_valid_mask.to(diff.dtype).sum().clamp_min(1.0),
        }
        return loss, probe

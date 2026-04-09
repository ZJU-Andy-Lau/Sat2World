from dataclasses import dataclass
import torch,math
from jaxtyping import Float
from torch import Tensor

from .loss import Loss
import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_and_visualize_depth(rendered_depth: torch.Tensor,
                             save_path: str = "rendered_depth.png"):
    """
    rendered_depth: torch.Tensor, shape [H, W] 或 [1, H, W]
    save_path: 保存路径
    """
    # 1. 转到 CPU 并去掉多余维度
    depth = rendered_depth.detach().cpu().squeeze()

    # 2. 可选：归一化到 [0,1] 方便显示
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    # 3. 可视化并保存
    plt.figure(figsize=(6,6))
    plt.imshow(depth_norm, cmap='magma')  # 也可以换成 'viridis' 等
    plt.colorbar(label='Normalized Depth')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()  # 如果只想保存不显示，可以去掉这行

@dataclass
class LossRelHeightCfg:
    weight: float


@dataclass
class LossRelHeightCfgWrapper:
    relheight: LossRelHeightCfg


class LossRelHeight(Loss[LossRelHeightCfg, LossRelHeightCfgWrapper]):
    """
    PCC loss between predicted relative height and GT height.
    Needs gaussians.hei and batch["height"]
    """

    def pcc_loss(self, x, y):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        return 1 - torch.sum(x_centered * y_centered) / (
            torch.sqrt(torch.sum(x_centered ** 2)) * torch.sqrt(torch.sum(y_centered ** 2)) + 1e-8
        )

    def forward(
            self,
            prediction,
            batch,
            gaussians,
            global_step,
            total_steps
    ) -> Float[Tensor, ""]:
        b, v, _, _, _ = batch["context"]["image"].shape
        _, N = gaussians.hei.shape
        hw = N // v
        h = w = int(math.sqrt(hw))
        # ===============================
        # 1. prediction
        # ===============================
        pred_height = gaussians.hei.reshape(b, v, h, w)
        # ===============================
        # 2. GT
        # ===============================
        gt_height = batch["context"]["height"]
        gt_height_pseudo = batch["context"]["height_pseudoLabel"]
        # ===============================
        # 3. resize
        # ===============================
        if gt_height.shape[-2:] != (h, w):
            gt_height = F.interpolate(gt_height, size=(h, w), mode="bilinear", align_corners=False)
        if gt_height_pseudo.shape[-2:] != (h, w):
            gt_height_pseudo = F.interpolate(gt_height_pseudo, size=(h, w), mode="bilinear", align_corners=False)
        # ===============================
        # 4. mask
        # ===============================
        mask_abs = ~torch.isnan(gt_height_pseudo)
        mask_rel = ~mask_abs
        # ===============================
        # 5. absolute loss（带稳定性保护）
        # ===============================
        if mask_abs.sum() < 2:
            abs_depth_loss = torch.zeros((), device=pred_height.device)
        else:
            σ = gt_height_pseudo[mask_abs].std()
            if σ < 1e-4:
                abs_depth_loss = torch.zeros((), device=pred_height.device)
            else:
                μ = gt_height_pseudo[mask_abs].mean()
                pred_abs = (pred_height - μ) / σ
                gt_abs = (gt_height_pseudo - μ) / σ
                abs_depth_loss = F.smooth_l1_loss(
                    pred_abs[mask_abs],
                    gt_abs[mask_abs]
                )
        # ===============================
        # 6. relative loss（仅 mask_rel + 防 NaN）
        # ===============================
        if mask_rel.sum() < 2:
            rel_depth_loss = torch.zeros((), device=pred_height.device)
        else:
            pred_rel = pred_height[mask_rel]
            gt_rel = gt_height[mask_rel]
            pred_std = pred_rel.std()
            gt_std = gt_rel.std()
            if pred_std < 1e-4 or gt_std < 1e-4:
                rel_depth_loss = torch.zeros((), device=pred_height.device)
            else:
                pred_rel = (pred_rel - pred_rel.mean()) / (pred_std + 1e-6)
                gt_rel = (gt_rel - gt_rel.mean()) / (gt_std + 1e-6)

                rel_depth_loss = self.pcc_loss(pred_rel, gt_rel)
        # ===============================
        # 7. final loss（无 alpha）
        # ===============================
        loss = abs_depth_loss + rel_depth_loss
        return self.cfg.weight * loss
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from einops import rearrange
from .loss import Loss


@dataclass
class LossFeatureMseCfg:
    weight_sam: float


@dataclass
class LossFeatureMseCfgWrapper:
    feature_mse: LossFeatureMseCfg


class LossFeatureMse(Loss[LossFeatureMseCfg, LossFeatureMseCfgWrapper]):
    """
    MSE loss between rendered SAM features and GT SAM features.
    prediction["render_SAM_feats"]: (B, C, H, W)
    batch["context"]["SAM_feats"]: (B, V, C, H, W)
    """

    def forward(
        self,
        prediction,
        batch,
        gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:

        rendered_sam = prediction["render_SAM_feats"]  # (B, C, H, W)
        gt_sam = batch["context"]["SAM_feats"]          # (B, V, C, H, W)

        bs, view = gt_sam.size(0), gt_sam.size(1)
        gt_sam = rearrange(gt_sam, "b v c h w -> (b v) c h w")

        # ---- flatten to (N, C) ----
        rendered_feat = rendered_sam.permute(0, 2, 3, 1).reshape(-1, rendered_sam.size(1))
        gt_feat = gt_sam.permute(0, 2, 3, 1).reshape(-1, gt_sam.size(1))

        # ---- MSE loss ----
        sam_mse_loss = F.mse_loss(rendered_feat, gt_feat)

        # 计算整体幅值（RMS）
        # rms_rendered = rendered_feat.pow(2).mean().sqrt()
        # rms_gt = gt_feat.pow(2).mean().sqrt()
        # 用 MSE 约束 RMS
        # sam_mse_loss = F.mse_loss(rms_rendered, rms_gt)

        return self.cfg.weight_sam * sam_mse_loss

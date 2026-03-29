"""loss.init

loss 目录导出入口。
"""

from loss.affine_loss import AffineGridLoss, AffineLinearRegularization, AffinePairwiseGeometryLoss, RefAffineIdentityLoss
from loss.common import masked_huber_loss, masked_l1_loss, masked_l2_loss, psnr_from_mse, softmax_entropy, ssim_map
from loss.height_loss import HeightHuberLoss
from loss.point_loss import PointMapLoss
from loss.regularization_loss import CenterConsistencyLoss, CoderProbe, GaussianRegularizationLoss
from loss.render_loss import RenderPathLoss
from loss.total_loss import LossWeightScheduler, RPCAnySplatTrainingObjective

__all__ = [
    "masked_huber_loss",
    "masked_l1_loss",
    "masked_l2_loss",
    "ssim_map",
    "psnr_from_mse",
    "softmax_entropy",
    "AffineGridLoss",
    "AffinePairwiseGeometryLoss",
    "AffineLinearRegularization",
    "RefAffineIdentityLoss",
    "HeightHuberLoss",
    "PointMapLoss",
    "RenderPathLoss",
    "GaussianRegularizationLoss",
    "CenterConsistencyLoss",
    "CoderProbe",
    "RPCAnySplatTrainingObjective",
    "LossWeightScheduler",
]

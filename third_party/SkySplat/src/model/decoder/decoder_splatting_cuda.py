from dataclasses import dataclass
from typing import Literal

import torch, os
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F
from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput
import torch.nn as nn
import torch.nn as nn
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_feature_map(
    feat: torch.Tensor,
    save_path: str = "pca_feature.png"
):
    # ---- 1. 统一成 (H*W, C) ----
    x = feat.detach().cpu().numpy()  # (H*W, C)
    n_pixels, C = feat.shape
    H = W = int(n_pixels ** 0.5)
    # ---- 2. PCA -> 3 通道 ----
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x)                      # (H*W, 3)
    x_pca = x_pca.reshape(H, W, 3)
    # ---- 3. 归一化到 [0,1] ----
    x_min, x_max = x_pca.min(), x_pca.max()
    x_pca = (x_pca - x_min) / (x_max - x_min + 1e-8)
    # ---- 4. 保存图片 ----
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.imsave(save_path, x_pca)

def downsample_features(feats, out_size=64):
    # feats: [B, V, H*W, C] 其中 H=W=256
    V, N, C = feats.shape
    H = W = int(N ** 0.5)  # 256
    feats = feats.view(V, H, W, C).permute(0, 3, 1, 2)  # [B*V, C, H, W]
    feats_down = F.interpolate(feats, size=(out_size, out_size), mode='bilinear', align_corners=False)
    feats_down = feats_down.permute(0, 2, 3, 1).reshape(V, out_size * out_size, C)
    return feats_down

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

        # feats_MLP: 64 + 1024 + 64 -> 512 -> 64
        feats_channels = 1024
        mid_channels = 512
        feats_dim = 64
        self.feats_MLP = nn.Sequential(
            nn.Linear(feats_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, feats_dim)
        )
        # SAM_MLP:  64 + -> 512 -> 1024
        in_dim, sam_dim = 64, 1024
        self.SAM_MLP = nn.Sequential(
            nn.Linear(in_dim, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, sam_dim)
        )

    def forward(
        self,
        gaussians,
        kwargs,
        image_shape: tuple[int, int],
        mode = "train"
    ):
        means3d = gaussians.means
        ### 渲染RGB
        bs_ref, view_ref = means3d.size(0), means3d.size(1)
        bs, view = bs_ref, kwargs['cam2img'].size(1)  # target的shape
        backgrounds = torch.zeros(bs, view, 3, device=means3d.device) # 保证最后一个维度为3，否则可能导致渲染颜色异常，如蓝色
        if "JAX" in kwargs['ref_filename'][0][0]:
            near = 5e5 * torch.ones(1, view, device=means3d.device).expand(bs, view)
            far  = 1e6 * torch.ones(1, view, device=means3d.device).expand(bs, view)
        elif "OMA" in kwargs['ref_filename'][0][0]:
            near = 1e6 * torch.ones(1, view, device=means3d.device).expand(bs, view)
            far  = 2e6 * torch.ones(1, view, device=means3d.device).expand(bs, view)

        color = render_cuda(
            rearrange(kwargs["cam2enu"], "b v i j -> (b v) i j"),
            rearrange(kwargs["cam2img"][..., :3, :3], "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            rearrange(backgrounds, "b v i -> (b v) i"),
            rearrange(gaussians.means, "b (v i) j -> (b v) i j", v=3),
            rearrange(gaussians.covariances, "b (v i) m n -> (b v) i m n", v=3),
            rearrange(gaussians.harmonics, "b (v g) d_sh c -> (b v) g d_sh c", v=3),
            rearrange(gaussians.opacities, "b (v i) -> (b v) (i)", v=3),
            mode = mode,
            scale_invariant  = True,  # 缩放near、far、mean3d、scales和conv等
            filename=kwargs['ref_filename']
        ) # 物方 -> 像方，SPLAT
        color = rearrange(color, "(b v) c h w -> b v c h w", b=bs, v=view)
        return {"render_color": color}


    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)

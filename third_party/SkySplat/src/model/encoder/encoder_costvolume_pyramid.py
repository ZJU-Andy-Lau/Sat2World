from dataclasses import dataclass
from typing import Literal, Optional, List

import torch,copy
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians,Gaussians_features

from .backbone import BackbonePyramid

from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfgPyramid:
    name: str
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]


class EncoderCostVolumePyramid(Encoder[EncoderCostVolumeCfgPyramid]):
    backbone: BackbonePyramid
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfgPyramid) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        self.backbone = BackbonePyramid(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                self.backbone.load_state_dict(updated_state_dict, strict=False)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        depth_min,depth_max = context['hei_min_max'][:,:,0] , context['hei_min_max'][:,:,1] # 缩小误差(绝对高度基准)
        near,far = torch.ones((b, v), dtype=torch.float32, device=device)*depth_min, torch.ones((b, v), dtype=torch.float32, device=device)*depth_max # 后续需要更改成和shunping程序的深度范围一致
        # Encode the context images.
        epipolar_kwargs = None
        features_list = self.backbone(
            context,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        ) # 前者（trans_features）经过了cross_view attn的融合，而后者（cnn_features）没有

        # Sample depths from the resulting features.
        in_feats = features_list
        extra_info = {}
        extra_info['images'] = rearrange(context['image'], "b v c h w -> (v b) c h w") # 注意，这里要使用没有归一化的images。只除以255即可，因为其用于产生高斯的scales和RGB球谐函数，需要与监督（/255）的一致
        extra_info["scene_names"] = scene_names
        extra_info['global_step'] = global_step
        gpp = self.cfg.gaussians_per_pixel
        gaussian_dict, result_dict = self.depth_predictor(
            in_feats, # 经过了cross_view attn融合后的特征
            copy.deepcopy(context["rpc_proj_matrices"]),
            near,
            far,
            context = context,
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            encoder=self,
        ) # 像方的深度、密度和初始高斯核建立完毕；下面投影到3D
        return gaussian_dict, result_dict


    def convert_to_gaussians_single_stge(
        self,
        raw_gaussians,
        densities,
        heights,
        image_size,
        context,
        global_step,
        opacity_multiplier=1.0,
        stage_id=0,
    ):
        device = raw_gaussians.device
        h, w = image_size[0], image_size[1]
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(raw_gaussians,"... (srf c) -> ... srf c",srf=self.cfg.num_surfaces,)
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w - 1, h - 1), dtype=torch.float32, device=device)
        # offset_xy = torch.full_like(offset_xy, 0.5) # 用于测试
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size # 在像方上offset_xy.其中,(offset_xy - 0.5)的目的是将offset范围变为 [-0.5, +0.5],允许正负偏移
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            context,
            context["enu2latlon"],
            copy.deepcopy(context["rpc_proj_matrices"]),
            copy.deepcopy(rearrange(context["cam2enu"], "b v i j -> b v () () () i j")),
            copy.deepcopy(rearrange(context["cam2img"][..., :3, :3], "b v i j -> b v () () () i j")),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            heights,
            self.map_pdf_to_opacity(densities, global_step) / gpp, # 将 概率密度（pdf） 映射到 不透明度（opacity），用于体渲染。且随着global_step动态变化
            rearrange(gaussians[..., 2:],"b v r srf c -> b v r srf () c",),
            (h, w),
            stage_id=stage_id,
        ) # 将2D高斯投影到3D世界空间中

        # Optionally apply a per-pixel opacity.
        scales = rearrange(gaussians.scales, "b v spp xyz -> b (v spp) xyz")
        rotations = rearrange(gaussians.rotations, "b v spp xyzw -> b (v spp) xyzw")
        opacity_multiplier = 1

        return Gaussians_features(
                rearrange(gaussians.utmmean,"b v spp xyz -> b (v spp) xyz"),
                gaussians.hei.reshape(-1).unsqueeze(dim=0),
                rearrange(gaussians.means,"b v spp xyz -> b (v spp) xyz"),
                rearrange(gaussians.scales,"b v spp xyz -> b (v spp) xyz"),
                rearrange(gaussians.rotations,"b v spp xyzw -> b (v spp) xyzw"),
                rearrange(opacity_multiplier * gaussians.opacities,"b v spp x -> b (v spp x)"),
                rearrange(gaussians.harmonics,"b v spp c d_sh -> b (v spp) c d_sh"),
                rearrange(gaussians.covariances,"b v spp i j -> b (v spp) i j")
        ), scales, rotations

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None

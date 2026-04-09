import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel
from .warping import RPC_Photo2Obj,RPC_Obj2Photo
from simfeatup_dev.upsamplers import get_upsampler
from torchvision import transforms
from .red_regularization import RED_Regularization
from ...types import Gaussians_features


def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values, coef):
    # src_fea: [B, C, H, W]
    # src_rpc: [B, 170]
    # ref_rpc: [B, 170]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    # import time
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.double, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.double, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y = y.view(1, 1, height, width).repeat(batch, num_depth, 1, 1) # (B, ndepth, H, W)
        x = x.view(1, 1, height, width).repeat(batch, num_depth, 1, 1)

        if len(depth_values.shape) == 2:
            h = depth_values.view(batch, num_depth, 1, 1).double().repeat(1, 1, height, width) # (B, ndepth, H, W)
        else:
            h = depth_values # (B, ndepth, H, W)

        x = x.view(batch, -1)
        y = y.view(batch, -1)
        h = h.view(batch, -1)
        h = h.double()

        # start = time.time()
        lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc, coef)
        samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc, coef) # (B, ndepth*H*W)
        # end = time.time()

        # print(torch.mean(samp - x), torch.var(samp - x))
        # print(torch.mean(line - y), torch.var(line - y))

        samp = samp.float()
        line = line.float()

        proj_x_normalized = samp / ((width - 1) / 2) - 1
        proj_y_normalized = line / ((height - 1) / 2) - 1
        proj_x_normalized = proj_x_normalized.view(batch, num_depth, height * width)
        proj_y_normalized = proj_y_normalized.view(batch, num_depth, height * width)

        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    grid_np = grid.view(batch, num_depth * height, width, 2).cpu().numpy()
    # # # # 可视化
    # import numpy as np
    # import cv2
    # warped_feature_np = warped_src_fea.squeeze(2)  # 去掉 Depth 维度，变为 [3, 3, 256, 256]
    # warped_feature_np = warped_feature_np.permute(0, 2, 3, 1).cpu().numpy().astype(
    #     np.uint8)  # 调整为 [B, H, W, C]
    # # # 显示三张图片
    # for i in range(3):
    #     cv2.imwrite(f"/media/pc2080ti/0A9AD66165F33762/XJHuang/project_sat_seg_3DGS/ours/GaussTR_MVSplat_RPC/work_dirs/test/Image_test_{i + 1}_ori.jpg", warped_feature_np[i])
    return warped_src_fea,grid_np

def rpc_prepare_feat_proj_data_lists(
    features, rpc_proj_matrices, near, far, num_samples=64, depth_inteval_pixel=4, pre_depth=None
):
    # prepare features
    b, v, _, h, w = features.shape
    feat_lists = []
    ref_proj_list = []
    src_proj_list = []

    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order] # 反转，配合作为ref_feat对应的src_feat
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_proj_list = []
            cur_src_proj_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                ref_proj = copy.deepcopy(rpc_proj_matrices[:, v0])
                src_proj = copy.deepcopy(rpc_proj_matrices[:, v1])
                ### 缩放RPC (*64/256)
                ref_proj[:, 0] = ref_proj[:, 0] * float(w) / 256.0
                ref_proj[:, 1] = ref_proj[:, 1] * float(h) / 256.0
                ref_proj[:, 5] = ref_proj[:, 5] * float(w) / 256.0
                ref_proj[:, 6] = ref_proj[:, 6] * float(h) / 256.0

                src_proj[:, 0] = src_proj[:, 0] * float(w) / 256.0
                src_proj[:, 1] = src_proj[:, 1] * float(h) / 256.0
                src_proj[:, 5] = src_proj[:, 5] * float(w) / 256.0
                src_proj[:, 6] = src_proj[:, 6] * float(h) / 256.0
                cur_ref_proj_list.append(ref_proj)
                cur_src_proj_list.append(src_proj)


            ### 存储batch的ref_intr和src_intr
            cur_ref_proj_to_v0s = torch.cat(cur_ref_proj_list, dim=0)  # (vxb c h w)
            ref_proj_list.append(cur_ref_proj_to_v0s)
            cur_src_proj_to_v0s = torch.cat(cur_src_proj_list, dim=0)  # (vxb c h w)
            src_proj_list.append(cur_src_proj_to_v0s)
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper


    # prepare depth bound (inverse depth) [v*b, d]
    if pre_depth == None:
        cur_depth_min = near.reshape(b*v)  # (B,)
        cur_depth_max = far.reshape(b*v)
        new_interval = (cur_depth_max - cur_depth_min) / (num_samples - 1)  # (B, )
        depth_candi_curr = cur_depth_min.unsqueeze(1) + (
                torch.arange(0, num_samples, device=cur_depth_min.device, dtype=cur_depth_min.dtype,
                             requires_grad=False).reshape(1, -1).repeat(3, 1) * new_interval.unsqueeze(1))  # (B, D)
        depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    else:
        pre_depth = F.interpolate(pre_depth,[h, w], mode='bilinear', align_corners=False)
        cur_depth_min = (pre_depth - num_samples / 2 * depth_inteval_pixel)  # (B, H, W)
        cur_depth_max = (pre_depth + num_samples / 2 * depth_inteval_pixel)
        new_interval = (cur_depth_max - cur_depth_min) / (num_samples - 1)  # (B, H, W)

        depth_candi_curr = cur_depth_min.unsqueeze(1) + (
                torch.arange(0, num_samples, device=pre_depth.device, dtype=pre_depth.dtype,
                             requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
        
    return feat_lists, ref_proj_list, src_proj_list, depth_candi_curr.squeeze(dim=1) # feat_lists中，[0]表示ref_feat，[1]表示对应的src_feat，[2]也表示对应的src_feat.

class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # attention 0, 1, 2 -> 64*64, 128*128, 256*256
        channel_list = [32, 64, 128]
        # channel_list = [64, 96, 128]
        depth_candi_list = [64, 32, 8]
        depth_inteval_pixel_list = [4, 2, 1]
        # depth_inteval_pixel_list = [2, 1, 0.5]
        self.depth_candi_list = depth_candi_list
        depth_predictor_list = []
        for stage_id in range(len(channel_list)):
            depth_predictor_list.append(
                DepthPredictorRefine(
                    channel_list,
                    depth_candi_list,
                    depth_inteval_pixel_list,
                    depth_unet_feat_dim,
                    depth_unet_attn_res,
                    depth_unet_channel_mult,
                    gaussian_raw_channels,
                    gaussians_per_pixel,
                    num_views,
                    stage_id,
                )
            )
        self.depth_predictor = nn.ModuleList(depth_predictor_list)

    def forward(
        self,
        features,
        rpc_proj_matrices,
        near,
        far,
        context = None,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        encoder=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""
        cnn_feature, trans_feature = features[0], features[1]
        b, v, c, h, w = trans_feature[0].shape
        cnn_feature = [rearrange(f, "(b v) c h w -> (v b) c h w", b=b) for f in cnn_feature]
        # for different resolution's depth and gaussians
        # attention: it returns b v
        result_dict = {"stage0": {}, "stage1": {}, "stage2": {}}
        gaussian_dict = {k: {} for k in result_dict.keys()}
        for i in range(len(self.depth_predictor)):
            depth_size = cnn_feature[i].size()[-2:]
            depth_predictor = self.depth_predictor[i]
            trans_feature_i = trans_feature[i]

            depths, densities, raw_gaussians, pdf_max  = depth_predictor(
                trans_feature_i,
                cnn_feature[i],
                copy.deepcopy(rpc_proj_matrices),
                near,
                far,
                extra_info=extra_info
            )
            extra_info["pdf_max"] = pdf_max
            extra_info["final_depth"] = rearrange(depths, "b v (h w) () () -> (v b) () h w", h=depth_size[0])
            result_dict[f"stage{i}"]["depths"] = depths
            result_dict[f"stage{i}"]["densities"] = densities
            result_dict[f"stage{i}"]["raw_gaussians"] = raw_gaussians
            image_size = cnn_feature[i].shape[-2:]
            return_gaussians, scales, rotations = encoder.convert_to_gaussians_single_stge(
                result_dict[f"stage{i}"]["raw_gaussians"],
                result_dict[f"stage{i}"]["densities"],
                result_dict[f"stage{i}"]["depths"],
                image_size,
                context,
                extra_info["global_step"],
                opacity_multiplier=1.0,
                stage_id=i,
            )

            concat_gaussians = return_gaussians if i == 0 else self.concat_gaussians(concat_gaussians,return_gaussians)
            concat_scales = scales if i == 0 else torch.cat([concat_scales, scales], dim=1)
            concat_rotations = rotations if i == 0 else torch.cat([concat_rotations, rotations], dim=1)
            gaussian_dict[f"stage{i}"]["gaussians"] = concat_gaussians
            gaussian_dict[f"stage{i}"]["depths"] = depths
            gaussian_dict[f"stage{i}"]["scales"] = concat_scales
            gaussian_dict[f"stage{i}"]["rotations"] = concat_rotations
        return gaussian_dict, result_dict

    def concat_gaussians(self, ga: Gaussians_features, gb: Gaussians_features):
        return Gaussians_features(
            torch.cat([ga.utmmean, gb.utmmean], dim=1),
            gb.hei,
            # torch.cat([ga.hei, gb.hei], dim=1),
            torch.cat([ga.means, gb.means], dim=1),
            torch.cat([ga.scales, gb.scales], dim=1),
            torch.cat([ga.rotations, gb.rotations], dim=1),
            torch.cat([ga.opacities, gb.opacities], dim=1),
            torch.cat([ga.harmonics, gb.harmonics], dim=1),
            torch.cat([ga.covariances, gb.covariances], dim=1),
        )

class DepthPredictorRefine(nn.Module):
    # Include the prediction and refinement of depth
    # Attention 0, 1, 2 -> 64*64, 128*128, 256*256
    # Channel_list = [32, 64, 128]
    # Depth_candi_list = [128, 64, 32]
    def __init__(
        self,
        channel_list,
        depth_candi_list,
        depth_inteval_pixel_list,
        depth_unet_feat_dim,
        depth_unet_attn_res,
        depth_unet_channel_mult,
        gaussian_raw_channels,
        gaussians_per_pixel,
        num_views: int,
        stage_id: int,
    ):
        super(DepthPredictorRefine, self).__init__()
        self.channel_list = channel_list
        self.depth_candi_list = depth_candi_list
        self.num_depth_candidates = depth_candi_list[stage_id]
        self.depth_inteval_pixel = depth_inteval_pixel_list[stage_id]

        self.num_views = num_views
        self.gaussians_per_pixel = gaussians_per_pixel
        stage_num = len(self.channel_list)
        channel_stage_id = int(stage_num - 1 - stage_id)
        self.stage_id = stage_id
        self.channel_stage_id = channel_stage_id
        self.upscale_factor = 4
        self.feature_channels = channel_list[channel_stage_id]
        depth_unet_feat_dim = depth_unet_feat_dim * 2 ** (2 - stage_id)
        # Depth estimation: project features to get softmax based coarse depth
        # CNN-based feature upsampler
        proj_in_channels = self.feature_channels + self.feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, self.feature_channels, 3, 1, 1),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(self.feature_channels, depth_unet_feat_dim, 3, 1, 1)
        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        self.refine_unet = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, 1, 1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=depth_unet_attn_res,
                channel_mult=depth_unet_channel_mult,
                num_head_channels=32,
                dims=2,
                postnorm=True,
                num_frames=num_views,
                use_cross_view_self_attn=True,
            ),
        )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + self.feature_channels
        gaussian_raw_channels = gaussian_raw_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1),
        )
        # Gaussians prediction: centers, opacity
        channels = depth_unet_feat_dim
        disps_models = [
            nn.Conv2d(channels, channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
        ]
        self.to_disparity = nn.Sequential(*disps_models)
        self.cost_regularization = RED_Regularization(in_channels=self.feature_channels, base_channels=8)

    def forward(
        self,
        features,
        cnn_features,
        rpc_proj_matrices,
        near,
        far,
        gaussians_per_pixel=1,
        extra_info=None,
    ):
        guidedHeight = True
        device = features.device
        b, v, c, h, w = features.shape

        ### 如果不使用guidedHeight的时候，仅self.stage_id == 0时，显示估计高度。后续层，直接插值即可。
        if not guidedHeight:
            if self.stage_id == 0:
                feat_comb_lists, ref_proj, src_proj, disp_candi_curr = (
                    rpc_prepare_feat_proj_data_lists(
                        features,
                        rpc_proj_matrices,
                        near,
                        far,
                        num_samples=self.num_depth_candidates,
                    )
                )
                feat01 = feat_comb_lists[0]
                ref_volume = feat_comb_lists[0].unsqueeze(2).repeat(1, 1, self.num_depth_candidates, 1, 1)

                ### shunping的高度估计代码
                num_views = len(feat01)
                volume_sum = ref_volume
                volume_sq_sum = ref_volume ** 2  # 方差
                # step 1. ref_features 和 src_features的获得
                for feat10, per_src_proj, per_ref_proj in zip(feat_comb_lists[1:], src_proj,
                                                              ref_proj):  # src_proj(1 2 0 和2 0 1的src_rpc参数);ref_proj(0 1 2的ref_rpc参数)
                    depth_values = disp_candi_curr.repeat(1, 1, h, w)  # (B, D, H, W)
                    coef = torch.ones((b * v, h * w * self.num_depth_candidates, 20), dtype=torch.double).to(device)
                    feat01_warped, grid_np = rpc_warping(feat10, per_src_proj, per_ref_proj, depth_values, coef)
                    volume_sum = volume_sum + feat01_warped  # 这一步类似于残差操作
                    volume_sq_sum = volume_sq_sum + feat01_warped ** 2  # 计算的方差，后续将其正则化
                    del feat01_warped

                # step 2. aggregate multiple feature volumes by variance
                volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
                # step 3. cost volume regularization
                refined_volume_variance = self.cost_regularization(volume_variance)
                # refined_volume_variance = F.interpolate(refined_volume_variance, scale_factor=self.upscale_factor,
                #                                         mode="bilinear", align_corners=True, )
                # refined_volume_variance = self.upsampler_cost_volume(refined_volume_variance) # 使用CNN-based上采样，不要直接F.interpolate，会导致深度图可视化有问题
                # step 4. cost volume regularization
                prob_volume = F.softmax(refined_volume_variance, dim=1)
                depth = depth_regression(prob_volume, depth_values=depth_values).unsqueeze(1)
                photometric_confidence, indices = prob_volume.max(1)
                photometric_confidence = photometric_confidence.unsqueeze(1)
                pdf_max = photometric_confidence
                final_depth = depth
                ### shunping的高度估计代码
            else:
                pdf_max = F.interpolate(extra_info["pdf_max"], size=cnn_features.shape[-2:], mode="bilinear", align_corners=True)
                final_depth = F.interpolate(extra_info["final_depth"], size=cnn_features.shape[-2:], mode="bilinear", align_corners=True)
                feat01 = rearrange(features, "b v c h w -> (v b) c h w")

        ### 使用guidedHeight时，每一层都需要显示估计高度。其中，下一层高度的估计区间([near, far])，由上一层提供高度先验。
        if guidedHeight:
            feat_comb_lists, ref_proj, src_proj, disp_candi_curr = (
                rpc_prepare_feat_proj_data_lists(
                    features,
                    rpc_proj_matrices,
                    near,
                    far,
                    num_samples=self.num_depth_candidates,
                    depth_inteval_pixel=self.depth_inteval_pixel,
                    pre_depth=extra_info["final_depth"] if self.stage_id != 0 else None
                )
            )
            feat01 = feat_comb_lists[0]
            ref_volume = feat_comb_lists[0].unsqueeze(2).repeat(1, 1, self.num_depth_candidates, 1, 1)

            ### shunping的高度估计代码
            num_views = len(feat01)
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2  # 方差
            # step 1. ref_features 和 src_features的获得
            for feat10, per_src_proj, per_ref_proj in zip(feat_comb_lists[1:], src_proj,
                                                          ref_proj):  # src_proj(1 2 0 和2 0 1的src_rpc参数);ref_proj(0 1 2的ref_rpc参数)
                if self.stage_id == 0:
                    depth_values = disp_candi_curr.repeat(1, 1, h, w)  # (B, D, H, W)
                else:
                    depth_values = disp_candi_curr  # (B, D, H, W)
                coef = torch.ones((b * v, h * w * self.num_depth_candidates, 20), dtype=torch.double).to(device)
                feat01_warped, grid_np = rpc_warping(feat10, per_src_proj, per_ref_proj, depth_values, coef)
                volume_sum = volume_sum + feat01_warped  # 这一步类似于残差操作
                volume_sq_sum = volume_sq_sum + feat01_warped ** 2  # 计算的方差，后续将其正则化
                del feat01_warped

            # step 2. aggregate multiple feature volumes by variance
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            # step 3. cost volume regularization
            refined_volume_variance = self.cost_regularization(volume_variance)
            # refined_volume_variance = F.interpolate(refined_volume_variance, scale_factor=self.upscale_factor,
            #                                         mode="bilinear", align_corners=True, )
            # refined_volume_variance = self.upsampler_cost_volume(refined_volume_variance) # 使用CNN-based上采样，不要直接F.interpolate，会导致深度图可视化有问题
            # step 4. cost volume regularization
            prob_volume = F.softmax(refined_volume_variance, dim=1)
            depth = depth_regression(prob_volume, depth_values=depth_values).unsqueeze(1)
            photometric_confidence, indices = prob_volume.max(1)
            photometric_confidence = photometric_confidence.unsqueeze(1)
            pdf_max = photometric_confidence
            final_depth = depth
            ### shunping的高度估计代码

        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        extra_img = F.interpolate(extra_info["images"], scale_factor=0.5**self.channel_stage_id, mode="bilinear", align_corners=True)
        # depth refinement
        refine_out = self.refine_unet(torch.cat((extra_img, proj_feature, final_depth, pdf_max), dim=1))  # 把ref_imgs，对应的ref_feats，对应的深度估计结果和pdf输入，用于一起细化。下面用于输出delta_disps, raw_densities，raw_gaussians_in
        # gaussians head
        raw_gaussians_in = [refine_out, extra_img, proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b)
        # delta fine depth and density
        delta_disps_density = self.to_disparity(refine_out)
        _, raw_densities = delta_disps_density.split(gaussians_per_pixel, dim=1)
        # combine coarse and fine info and match shape
        densities = repeat(F.sigmoid(raw_densities), "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1, )
        # fine_disps = (fullres_disps + delta_disps)# 初始深度估计+修正delta结果
        depths = final_depth
        depths = depths.clamp(rearrange(near, "b v -> (v b) () () ()"), rearrange(far, "b v -> (v b) () () ()"), )
        depths = repeat(depths, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1, )
        return depths, densities, raw_gaussians, pdf_max

def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    else:
        depth_values = F.interpolate(depth_values, [p.shape[2], p.shape[3]], mode='bilinear', align_corners=False)
    depth = torch.sum(p * depth_values, 1)
    return depth
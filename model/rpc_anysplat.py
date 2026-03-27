"""rpc_anysplat.py

Sat2World 主模型入口：RPCAnySplat。

前向主流程（两次 refinement）：
1) 提取视觉 patch token；
2) 第一次几何融合编码 -> affine_coarse；
3) 用 affine_coarse 校正 RPC，再做第二次几何融合编码；
4) 输出 affine_pred / height_abs / point_abs / gaussian attributes；
5) 构造两条高斯中心路径：
   - centers_rpc = corrected RPC + height_abs
   - centers_point = point_abs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from geometry import (
    RPCGeometryOps,
    enforce_reference_affine_identity,
    expand_height_ref_map,
    infer_height_ref_batch,
    make_image_grid,
    make_patch_centers,
    reshape_bv_to_bvchw,
    reshape_bv_to_bvn,
)
from model.backbone import DINOv3Backbone, DINOv3BackboneCfg, GeometryTokenMLP, VisualGeometryFuser
from model.coders import HeightCoder, PointCoder, SymmetricBinCoderCfg
from model.encoder import AlternatingEncoder, AlternatingEncoderCfg
from model.heads import (
    AffineHead,
    AffineHeadCfg,
    GaussianHead,
    HeightHead,
    PointHead,
    SharedDenseDecoder,
)


@dataclass
class RPCAnySplatCfg:
    """RPCAnySplat 总配置。"""

    backbone: DINOv3BackboneCfg = field(default_factory=DINOv3BackboneCfg)
    encoder: AlternatingEncoderCfg = field(default_factory=AlternatingEncoderCfg)
    affine_head: AffineHeadCfg = field(default_factory=AffineHeadCfg)

    height_bins: int = 33
    point_bins: int = 33
    sh_dim: int = 48

    height_bin_size: float = 1.0
    point_bin_size_xy: float = 2.0
    point_bin_size_z: float = 1.0

    height_fine_range: float = 0.5
    point_fine_range_xy: float = 1.0
    point_fine_range_z: float = 0.5


class RPCAnySplat(nn.Module):
    """Sat2World 主模型。

    成员模块:
        - backbone: DINOv3 patch token 提取；
        - geom_mlp/fuser: 几何-视觉融合；
        - encoder: 共享权重交替编码器（前向中调用两次）；
        - affine/height/point/gaussian heads;
        - height/point coders;
        - rpc_ops: RPC 工程接口。
    """

    def __init__(self, cfg: RPCAnySplatCfg) -> None:
        """初始化主模型并构建所有子模块。"""
        super().__init__()
        self.cfg = cfg

        self.backbone = DINOv3Backbone(cfg.backbone)
        self.geom_mlp = GeometryTokenMLP(in_dim=20, hidden_dim=256, out_dim=256)
        self.fuser = VisualGeometryFuser(visual_dim=self.backbone.embed_dim, geom_dim=256, out_dim=self.backbone.embed_dim)

        encoder_cfg = cfg.encoder
        if encoder_cfg.dim != self.backbone.embed_dim:
            encoder_cfg = AlternatingEncoderCfg(
                dim=self.backbone.embed_dim,
                num_heads=encoder_cfg.num_heads,
                ffn_ratio=encoder_cfg.ffn_ratio,
                num_layers=encoder_cfg.num_layers,
                dropout=encoder_cfg.dropout,
            )
        self.encoder = AlternatingEncoder(encoder_cfg)

        self.dense_decoder = SharedDenseDecoder(in_ch=self.backbone.embed_dim, out_ch=256)
        self.affine_head = AffineHead(in_dim=self.backbone.embed_dim, hidden_dim=512, cfg=cfg.affine_head)
        self.height_head = HeightHead(in_ch=256, num_bins=cfg.height_bins)
        self.point_head = PointHead(in_ch=256, num_bins=cfg.point_bins)
        self.gaussian_head = GaussianHead(in_ch=256, sh_dim=cfg.sh_dim)

        self.height_coder = HeightCoder(
            SymmetricBinCoderCfg(num_bins=cfg.height_bins, bin_size=cfg.height_bin_size, fine_range=cfg.height_fine_range)
        )
        self.point_coder = PointCoder(
            cfg_x=SymmetricBinCoderCfg(num_bins=cfg.point_bins, bin_size=cfg.point_bin_size_xy, fine_range=cfg.point_fine_range_xy),
            cfg_y=SymmetricBinCoderCfg(num_bins=cfg.point_bins, bin_size=cfg.point_bin_size_xy, fine_range=cfg.point_fine_range_xy),
            cfg_z=SymmetricBinCoderCfg(num_bins=cfg.point_bins, bin_size=cfg.point_bin_size_z, fine_range=cfg.point_fine_range_z),
        )

        self.rpc_ops = RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32)

    def _prepare_height_ref(self, batch: dict[str, Any], rpc_init: list[list[Any]], device: torch.device) -> torch.Tensor:
        """准备 [B,V] 的 height_ref；优先读取 batch，缺失则从 RPC 推断。"""
        if "height_ref" in batch and batch["height_ref"] is not None:
            h_ref = batch["height_ref"]
            if not torch.is_tensor(h_ref):
                h_ref = torch.as_tensor(h_ref, device=device, dtype=torch.float32)
            else:
                h_ref = h_ref.to(device=device, dtype=torch.float32)
            return h_ref
        return infer_height_ref_batch(rpc_init, device=device, dtype=torch.float32)

    def _reshape_logits_to_bv(self, x: torch.Tensor, b: int, v: int) -> torch.Tensor:
        """把 [B*V,C,H,W] 整理为 [B,V,C,H,W]。"""
        return reshape_bv_to_bvchw(x, b, v)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """执行 RPCAnySplat 前向。

        约定输入:
            batch["images"]: [B,V,3,H,W]
            batch["rpc_init"]: 长度 B 的 list，每项是长度 V 的 RPC 列表
            batch["scene_xy_center"]: 可选 [B,2] 或 list，顺序 (y,x)
            batch["scene_xy_scale"]: 可选 [B,2] 或 list，顺序 (y,x)
            batch["height_ref"]: 可选 [B,V]
            batch["ref_view_idx"]: 可选，int 或 [B]

        返回:
            包含仿射、高程、点云、高斯属性、双路径中心与关键中间量的字典。
        """
        images: torch.Tensor = batch["images"].to(dtype=torch.float32)
        rpc_init = batch["rpc_init"]

        if images.ndim != 5:
            raise ValueError(f"images must be [B,V,3,H,W], got {tuple(images.shape)}")

        b, v, _, h, w = images.shape
        device = images.device

        scene_xy_center = batch.get("scene_xy_center", None)
        scene_xy_scale = batch.get("scene_xy_scale", None)
        ref_view_idx = batch.get("ref_view_idx", 0)

        # 1) Backbone 特征提取
        images_bv = images.view(b * v, 3, h, w)
        backbone_out = self.backbone(images_bv)
        patch_tokens_vis = backbone_out["patch_tokens"].view(b, v, -1, self.backbone.embed_dim)
        patch_valid_mask_backbone = reshape_bv_to_bvn(backbone_out["patch_valid_mask"], b, v)

        gh, gw = backbone_out["grid_hw"]
        patch_centers, patch_valid_mask_scene = make_patch_centers(
            orig_hw=backbone_out["orig_hw"],
            padded_hw=backbone_out["pad_hw"],
            grid_hw=backbone_out["grid_hw"],
            device=device,
            dtype=torch.float32,
        )

        patch_valid_mask = patch_valid_mask_backbone & patch_valid_mask_scene.view(1, 1, -1)

        # 2) height_ref
        height_ref = self._prepare_height_ref(batch, rpc_init, device=device)

        # 3) 第一次几何特征 + 编码
        geom_feat_1 = self.rpc_ops.compute_patch_geometry_features_batch(
            rpc_batch=rpc_init,
            patch_centers=patch_centers,
            height_ref=height_ref,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        geom_tok_1 = self.geom_mlp(geom_feat_1)
        fused_tok_1 = self.fuser(patch_tokens_vis, geom_tok_1)

        enc_out_1 = self.encoder(fused_tok_1, patch_valid_mask)
        affine_coarse = self.affine_head(enc_out_1["view_tokens"])
        affine_coarse = enforce_reference_affine_identity(affine_coarse, ref_view_idx)

        # 4) 粗仿射修正 RPC
        corrected_rpc_coarse = self.rpc_ops.apply_affine_correction_batch(rpc_init, affine_coarse)

        # 5) 第二次几何特征 + 同一编码器再次 refinement
        geom_feat_2 = self.rpc_ops.compute_patch_geometry_features_batch(
            rpc_batch=corrected_rpc_coarse,
            patch_centers=patch_centers,
            height_ref=height_ref,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        geom_tok_2 = self.geom_mlp(geom_feat_2)
        fused_tok_2 = self.fuser(patch_tokens_vis, geom_tok_2)

        enc_out_2 = self.encoder(fused_tok_2, patch_valid_mask)
        patch_tokens_final = enc_out_2["patch_tokens"]
        view_tokens_final = enc_out_2["view_tokens"]
        scene_token_final = enc_out_2["scene_token"]

        # 6) 最终仿射
        affine_pred = self.affine_head(view_tokens_final)
        affine_pred = enforce_reference_affine_identity(affine_pred, ref_view_idx)
        corrected_rpc_final = self.rpc_ops.apply_affine_correction_batch(rpc_init, affine_pred)

        # 7) patch -> dense
        patch_map_final = self.encoder.patch_tokens_to_map(patch_tokens_final, (gh, gw))
        patch_map_final_bv = patch_map_final.view(b * v, self.backbone.embed_dim, gh, gw)
        dense_feat = self.dense_decoder(patch_map_final_bv, images_bv)

        # 8) 高程分支
        height_pred = self.height_head(dense_feat)
        h_logits = self._reshape_logits_to_bv(height_pred["logits"], b, v)
        h_fine = self._reshape_logits_to_bv(height_pred["fine"], b, v)

        h_ref_map = expand_height_ref_map(height_ref, h, w)
        height_decoded = self.height_coder(h_logits, h_fine, h_ref_map)
        height_abs = height_decoded["h_abs"]

        # 9) 点云分支（独立 anchor）
        image_grid = make_image_grid(h, w, device=device, dtype=torch.float32)
        point_anchor = self.rpc_ops.build_point_anchor_map_batch(
            rpc_init_batch=rpc_init,
            pixel_grid=image_grid,
            height_ref=height_ref,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )

        point_pred = self.point_head(dense_feat)
        x_logits = self._reshape_logits_to_bv(point_pred["x_logits"], b, v)
        y_logits = self._reshape_logits_to_bv(point_pred["y_logits"], b, v)
        z_logits = self._reshape_logits_to_bv(point_pred["z_logits"], b, v)
        x_fine = self._reshape_logits_to_bv(point_pred["x_fine"], b, v)
        y_fine = self._reshape_logits_to_bv(point_pred["y_fine"], b, v)
        z_fine = self._reshape_logits_to_bv(point_pred["z_fine"], b, v)

        point_decoded = self.point_coder(
            x_logits=x_logits,
            y_logits=y_logits,
            z_logits=z_logits,
            x_fine=x_fine,
            y_fine=y_fine,
            z_fine=z_fine,
            point_anchor=point_anchor,
        )
        point_abs = point_decoded["point_abs"]

        # 10) 高斯属性分支
        gauss_pred = self.gaussian_head(dense_feat)
        gaussian_opacity = self._reshape_logits_to_bv(gauss_pred["opacity"], b, v)
        gaussian_scale = self._reshape_logits_to_bv(gauss_pred["scale"], b, v)
        gaussian_rotation = self._reshape_logits_to_bv(gauss_pred["rotation"], b, v)
        gaussian_sh = self._reshape_logits_to_bv(gauss_pred["sh"], b, v)
        gaussian_conf_rpc = self._reshape_logits_to_bv(gauss_pred["confidence_rpc"], b, v)
        gaussian_conf_point = self._reshape_logits_to_bv(gauss_pred["confidence_point"], b, v)

        # 11) 双路径中心
        centers_rpc = self.rpc_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=corrected_rpc_final,
            pixel_grid=image_grid,
            height_abs=height_abs,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        centers_point = point_abs

        return {
            "affine_coarse": affine_coarse,
            "affine_pred": affine_pred,
            "rpc_corrected": corrected_rpc_final,
            "height_ref": height_ref,
            "height_abs": height_abs,
            "height_coarse": height_decoded["delta_h_coarse"],
            "height_fine": height_decoded["delta_h_fine"],
            "point_anchor": point_anchor,
            "point_abs": point_abs,
            "point_delta_coarse": point_decoded["delta_xyz_coarse"],
            "point_delta_fine": point_decoded["delta_xyz_fine"],
            "gaussian_opacity": gaussian_opacity,
            "gaussian_scale": gaussian_scale,
            "gaussian_rotation": gaussian_rotation,
            "gaussian_sh": gaussian_sh,
            "gaussian_confidence_rpc": gaussian_conf_rpc,
            "gaussian_confidence_point": gaussian_conf_point,
            "gaussian_centers_rpc": centers_rpc,
            "gaussian_centers_point": centers_point,
            "patch_valid_mask": patch_valid_mask,
            "patch_tokens_final": patch_tokens_final,
            "view_tokens_final": view_tokens_final,
            "scene_token_final": scene_token_final,
        }

"""sat2world.py

Sat2World 主模型入口：Sat2World。

前向主流程（单阶段）：
1) 提取视觉 patch token；
2) 计算一次几何特征并与视觉 token 融合；
3) 单次编码后直接输出 affine_pred；
4) 用 affine_pred 校正 rpc_init，得到 rpc_corrected；
5) 输出 height_abs / point_abs / gaussian attributes；
6) 构造两条高斯中心路径：
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
    infer_height_ref_batch,
    make_image_grid,
    make_patch_centers,
    reshape_bv_to_bvchw,
    reshape_bv_to_bvn,
)
from model.backbone import (
    DINOv3Backbone,
    DINOv3BackboneCfg,
    GeometryTokenMLP,
    LocalPatchDetailEncoder,
    LocalPatchDetailEncoderCfg,
    VisualGeometryDetailFuser,
)
from model.coders import BoundedSinhOffsetDecoder, SymmetricBinCoderCfg, SymmetricBinScalarCoder
from model.encoder import AlternatingEncoder, AlternatingEncoderCfg
from model.heads import (
    AffineHead,
    AffineHeadCfg,
    DPTDenseDecoder,
    DPTDenseDecoderCfg,
    GaussianHead,
    DenseHeightLocalHead,
    PointXYHead,
    PointZLocalHead,
    SceneHeightAnchorHead,
    TaskAdapter,
)
from model.patch_matcher import PatchHeatmapMatcher, PatchMatcherCfg
from model.utils import check_nan


@dataclass
class Sat2WorldCfg:
    """Sat2World 总配置。"""

    backbone: DINOv3BackboneCfg = field(default_factory=DINOv3BackboneCfg)
    encoder: AlternatingEncoderCfg = field(default_factory=AlternatingEncoderCfg)
    affine_head: AffineHeadCfg = field(default_factory=AffineHeadCfg)

    point_bins_xy: int = 33
    sh_dim: int = 48

    height_anchor_scale: float = 200.0
    height_local_scale: float = 30.0
    height_z_max: float = 4.0
    point_bin_size_xy: float = 2.0
    point_fine_range_xy: float = 1.0
    center_downsample_stage_steps: tuple[int, ...] = (0, 20000, 60000)
    center_downsample_factors: tuple[int, ...] = (4, 2, 1)

    enable_gaussian_branch: bool = True
    intermediate_layer_idx: tuple[int, int, int, int] = (5, 11, 17, 23)
    dense_pos_embed: bool = True
    dense_down_ratio: int = 1
    dense_frames_chunk_size: int = 8
    task_adapter_depth: int = 2
    # 预训练中间层对比监督配置
    nce_layer_index: int = 5  # 0-based，第 6 层
    nce_projector_dim: int = 256
    nce_projector_hidden_dim: int = 512
    detail_patch_size: int = 16
    detail_token_dim: int = 1024
    patch_match_dim: int = 256
    patch_match_heads: int = 4
    patch_match_layers: int = 2


class Sat2World(nn.Module):
    """Sat2World 主模型。

    成员模块:
        - backbone: DINOv3 patch token 提取；
        - geom_mlp/fuser: 几何-视觉融合；
        - encoder: 共享权重交替编码器（前向中调用一次）；
        - affine/height/point/gaussian heads;
        - height/point coders;
        - rpc_ops: RPC 工程接口。
    """

    def __init__(self, cfg: Sat2WorldCfg) -> None:
        """初始化主模型并构建所有子模块。"""
        super().__init__()
        self.cfg = cfg

        self.backbone = DINOv3Backbone(cfg.backbone)
        self.detail_encoder = LocalPatchDetailEncoder(
            LocalPatchDetailEncoderCfg(
                patch_size=int(cfg.detail_patch_size),
                in_channels=3,
                hidden_dim=int(cfg.detail_token_dim),
                out_dim=int(cfg.detail_token_dim),
            )
        )
        self.geom_mlp = GeometryTokenMLP(in_dim=45, hidden_dim=256, out_dim=256)
        self.fuser = VisualGeometryDetailFuser(
            visual_dim=self.backbone.embed_dim,
            detail_dim=int(cfg.detail_token_dim),
            geom_dim=256,
            out_dim=self.backbone.embed_dim,
        )
        self.patch_matcher = PatchHeatmapMatcher(
            PatchMatcherCfg(
                in_dim=self.backbone.embed_dim,
                match_dim=int(cfg.patch_match_dim),
                num_heads=int(cfg.patch_match_heads),
                num_layers=int(cfg.patch_match_layers),
            )
        )

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

        self.dense_decoder = DPTDenseDecoder(
            in_ch=self.backbone.embed_dim,
            cfg=DPTDenseDecoderCfg(
                out_ch=256,
                intermediate_layer_idx=tuple(int(x) for x in cfg.intermediate_layer_idx),
                pos_embed=bool(cfg.dense_pos_embed),
                down_ratio=int(cfg.dense_down_ratio),
                frames_chunk_size=int(cfg.dense_frames_chunk_size),
            ),
        )
        self.height_adapter = TaskAdapter(ch=256, depth=int(cfg.task_adapter_depth))
        self.point_adapter = TaskAdapter(ch=256, depth=int(cfg.task_adapter_depth))
        self.gaussian_adapter = TaskAdapter(ch=256, depth=int(cfg.task_adapter_depth))
        self.affine_head = AffineHead(in_dim=self.backbone.embed_dim, hidden_dim=512, cfg=cfg.affine_head)
        self.height_anchor_head = SceneHeightAnchorHead(in_dim=self.backbone.embed_dim, hidden_dim=512)
        self.height_local_head = DenseHeightLocalHead(in_ch=256)
        self.point_xy_head = PointXYHead(in_ch=256, num_bins_xy=cfg.point_bins_xy)
        self.point_z_local_head = PointZLocalHead(in_ch=256)
        self.gaussian_head = GaussianHead(in_ch=256, sh_dim=cfg.sh_dim)
        self.nce_projector = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, int(cfg.nce_projector_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(cfg.nce_projector_hidden_dim), int(cfg.nce_projector_dim)),
        )

        self.height_offset_decoder = BoundedSinhOffsetDecoder(z_max=cfg.height_z_max)
        self.point_xy_coder_x = SymmetricBinScalarCoder(
            SymmetricBinCoderCfg(num_bins=cfg.point_bins_xy, bin_size=cfg.point_bin_size_xy, fine_range=cfg.point_fine_range_xy)
        )
        self.point_xy_coder_y = SymmetricBinScalarCoder(
            SymmetricBinCoderCfg(num_bins=cfg.point_bins_xy, bin_size=cfg.point_bin_size_xy, fine_range=cfg.point_fine_range_xy)
        )

        self.rpc_ops = RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32)
        self.runtime_global_step: int = 0
        self.runtime_mode: str = "train"
        self.runtime_pretrain_geometry_only: bool = False
        self._set_gaussian_trainable(bool(self.cfg.enable_gaussian_branch))

    def set_runtime_context(self, *, global_step: int, mode: str) -> None:
        self.runtime_global_step = int(global_step)
        self.runtime_mode = str(mode)

    def _set_gaussian_trainable(self, trainable: bool) -> None:
        trainable = bool(trainable)
        self.gaussian_adapter.requires_grad_(trainable)
        self.gaussian_head.requires_grad_(trainable)

    def set_pretrain_geometry_only(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.runtime_pretrain_geometry_only = enabled

        # 预训练几何模式下冻结高斯分支参数，避免 DDP(find_unused_parameters=False)
        # 因分支跳过而触发“unused parameters”归约错误。
        if enabled:
            self._set_gaussian_trainable(False)
        else:
            # 仅在配置允许高斯分支时恢复可训练。
            if bool(self.cfg.enable_gaussian_branch):
                self._set_gaussian_trainable(True)

    def _current_center_downsample(self) -> int:
        steps = tuple(int(x) for x in self.cfg.center_downsample_stage_steps)
        factors = tuple(int(x) for x in self.cfg.center_downsample_factors)
        if len(steps) == 0 or len(factors) == 0:
            return 1
        idx = 0
        for si, st in enumerate(steps):
            if self.runtime_global_step >= st:
                idx = si
            else:
                break
        idx = min(idx, len(factors) - 1)
        if self.runtime_mode != "train":
            return 1
        return max(1, factors[idx])

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

    def _prepare_height_ref_anchor(
        self,
        batch: dict[str, Any],
        height_ref: torch.Tensor,
        ref_view_idx: torch.Tensor | int,
    ) -> torch.Tensor:
        if "height_ref_anchor" in batch and batch["height_ref_anchor"] is not None:
            t = batch["height_ref_anchor"]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=height_ref.device, dtype=torch.float32)
            else:
                t = t.to(device=height_ref.device, dtype=torch.float32)
            return t.view(height_ref.shape[0], 1)

        b, v = height_ref.shape
        if torch.is_tensor(ref_view_idx):
            ref = ref_view_idx.to(device=height_ref.device, dtype=torch.long).view(-1)
        else:
            ref = torch.full((b,), int(ref_view_idx), device=height_ref.device, dtype=torch.long)
        if ref.numel() == 1:
            ref = ref.expand(b)
        ref = ref.clamp(0, v - 1)
        return height_ref.gather(1, ref.view(b, 1))

    def _point_map_norm_to_local_meter(
        self,
        point_map: torch.Tensor,
        scene_xy_scale: Any,
    ) -> torch.Tensor:
        """把点云从 (x_norm,y_norm,h_m) 转为局部米制 (x_l,y_l,h_m)。

        约定：
        - 局部米制定义为相对 scene center 的 ENU/墨卡托局部平面坐标，不含平移项；
        - 仅对 x/y 乘以 scene scale（顺序为 [y, x]），z 保持不变。
        """
        if scene_xy_scale is None:
            return point_map
        if not torch.is_tensor(scene_xy_scale):
            return point_map
        if scene_xy_scale.ndim != 2 or scene_xy_scale.shape[-1] != 2:
            return point_map

        out = point_map.clone()
        b = int(point_map.shape[0])
        sx = scene_xy_scale[:, 1].to(device=point_map.device, dtype=point_map.dtype).view(b, 1, 1, 1, 1)
        sy = scene_xy_scale[:, 0].to(device=point_map.device, dtype=point_map.dtype).view(b, 1, 1, 1, 1)
        out[:, :, 0:1] = out[:, :, 0:1] * sx
        out[:, :, 1:2] = out[:, :, 1:2] * sy
        return out

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """执行 Sat2World 前向。

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
        detail_out = self.detail_encoder(
            images_bv,
            orig_hw=backbone_out["orig_hw"],
            pad_hw=backbone_out["pad_hw"],
        )
        patch_tokens_detail = detail_out["patch_tokens_detail"].view(b, v, -1, int(self.cfg.detail_token_dim))
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

        # 3) 单次几何特征 + 编码
        geom_feat = self.rpc_ops.compute_patch_geometry_features_batch(
            rpc_batch=rpc_init,
            patch_centers=patch_centers,
            height_ref=height_ref,
            scene_xy_center=scene_xy_center,
            scene_xy_scale=scene_xy_scale,
        )
        geom_tok = self.geom_mlp(geom_feat)
        fused_tok = self.fuser(patch_tokens_vis, patch_tokens_detail, geom_tok)

        enc_out = self.encoder(
            fused_tok,
            patch_valid_mask,
            ref_view_idx=ref_view_idx,
            patch_grid_hw=(gh, gw),
            return_all_layers=True,
        )
        patch_tokens_final = enc_out["patch_tokens"]
        view_tokens_final = enc_out["view_tokens"]
        scene_token_final = enc_out["scene_token"]
        patch_tokens_layers = enc_out["patch_tokens_layers"]
        layer_idx = int(self.cfg.nce_layer_index)
        if layer_idx < 0:
            layer_idx = patch_tokens_layers.shape[1] + layer_idx
        layer_idx = max(0, min(layer_idx, patch_tokens_layers.shape[1] - 1))
        patch_tokens_layer_sel = patch_tokens_layers[:, layer_idx]
        patch_tokens_nce_proj = self.nce_projector(patch_tokens_layer_sel)

        with torch.autocast(device_type=device.type, enabled=False):
            # 4) 单阶段仿射预测（参考视图约束在 loss 阶段处理）
            affine_pred = self.affine_head(view_tokens_final.float())

            # 5) affine_pred 修正 rpc_init
            rpc_corrected = self.rpc_ops.apply_affine_correction_batch(rpc_init, affine_pred)

        # 6) patch layers -> DPT dense
        dense_feat = self.dense_decoder(
            patch_tokens_layers=patch_tokens_layers,
            images=images_bv,
            patch_grid_hw=(gh, gw),
        )
        with torch.autocast(device_type=device.type, enabled=False):
            # 7) 高程分支：scene-level anchor + dense local
            height_ref_anchor = self._prepare_height_ref_anchor(batch, height_ref, ref_view_idx).float()
            check_nan(height_ref_anchor,"height_ref_anchor")
            height_anchor_raw_z = self.height_anchor_head(scene_token_final.float())
            check_nan(height_anchor_raw_z,"height_anchor_raw_z")
            height_anchor_dec = self.height_offset_decoder(height_anchor_raw_z, scale=self.cfg.height_anchor_scale)
            height_anchor_z = height_anchor_dec["z"]
            check_nan(height_anchor_z,"height_anchor_z")
            height_anchor_offset = height_anchor_dec["offset"]
            check_nan(height_anchor_offset,"height_anchor_offset")
            height_anchor = height_ref_anchor + height_anchor_offset
            check_nan(height_anchor,"height_anchor")

            check_nan(dense_feat,"dense_feat")
            height_feat = self.height_adapter(dense_feat.float())
            check_nan(height_feat,"height_adapter_feat")
            height_local_raw_z_bv = self.height_local_head(height_feat)
            check_nan(height_local_raw_z_bv,"height_local_raw_z_bv")
            height_local_raw_z = self._reshape_logits_to_bv(height_local_raw_z_bv, b, v)
            height_local_dec = self.height_offset_decoder(height_local_raw_z, scale=self.cfg.height_local_scale)
            height_local_z = height_local_dec["z"]
            check_nan(height_local_z,"height_local_z")
            height_local_offset = height_local_dec["offset"]
            check_nan(height_local_offset,"height_local_offset")
            height_anchor_map = height_anchor.view(b, 1, 1, 1, 1).expand(b, v, 1, h, w)
            check_nan(height_anchor_map,"height_anchor_map")
            height_abs = height_anchor_map + height_local_offset
            check_nan(height_abs,"height_abs")

            # 8) 点云分支（独立 anchor）
            image_grid = make_image_grid(h, w, device=device, dtype=torch.float32)
            point_anchor = self.rpc_ops.build_point_anchor_map_batch(
                rpc_init_batch=rpc_init,
                pixel_grid=image_grid,
                height_ref=height_ref,
                scene_xy_center=scene_xy_center,
                scene_xy_scale=scene_xy_scale,
            )

            point_feat = self.point_adapter(dense_feat.float())
            point_xy_pred = self.point_xy_head(point_feat)
            x_logits = self._reshape_logits_to_bv(point_xy_pred["x_logits"], b, v)
            y_logits = self._reshape_logits_to_bv(point_xy_pred["y_logits"], b, v)
            x_fine = self._reshape_logits_to_bv(point_xy_pred["x_fine"], b, v)
            y_fine = self._reshape_logits_to_bv(point_xy_pred["y_fine"], b, v)
            dx, dx_coarse, dx_fine = self.point_xy_coder_x.decode(x_logits, x_fine, channel_dim=2)
            dy, dy_coarse, dy_fine = self.point_xy_coder_y.decode(y_logits, y_fine, channel_dim=2)
            point_x = point_anchor[:, :, 0:1] + dx.unsqueeze(2)
            point_y = point_anchor[:, :, 1:2] + dy.unsqueeze(2)

            point_z_local_raw_z_bv = self.point_z_local_head(point_feat)
            point_z_local_raw_z = self._reshape_logits_to_bv(point_z_local_raw_z_bv, b, v)
            point_z_local_dec = self.height_offset_decoder(point_z_local_raw_z, scale=self.cfg.height_local_scale)
            point_z_local_z = point_z_local_dec["z"]
            point_z_local_offset = point_z_local_dec["offset"]
            point_z = height_anchor_map + point_z_local_offset
            point_abs = torch.cat([point_x, point_y, point_z], dim=2)

        gaussian_enabled = bool(self.cfg.enable_gaussian_branch) and (not self.runtime_pretrain_geometry_only)

        out = {
            "affine_pred": affine_pred,
            "rpc_corrected": rpc_corrected,
            "height_ref": height_ref,
            "height_ref_anchor": height_ref_anchor,
            "height_abs": height_abs,
            "height_anchor": height_anchor,
            "height_anchor_offset": height_anchor_offset,
            "height_anchor_z": height_anchor_z,
            "height_local_z": height_local_z,
            "height_local_offset": height_local_offset,
            "height_anchor_raw_z": height_anchor_raw_z,
            "height_local_raw_z": height_local_raw_z,
            "point_anchor": point_anchor,
            "point_abs": point_abs,
            "point_delta_xy_coarse": torch.cat([dx_coarse.unsqueeze(2), dy_coarse.unsqueeze(2)], dim=2),
            "point_delta_xy_fine": torch.cat([dx_fine.unsqueeze(2), dy_fine.unsqueeze(2)], dim=2),
            "point_logits": {"x": x_logits, "y": y_logits},
            "point_fine_raw": {"x": x_fine, "y": y_fine},
            "point_z_local_z": point_z_local_z,
            "point_z_local_offset": point_z_local_offset,
            "point_z_local_raw_z": point_z_local_raw_z,
            "height_z_max": torch.tensor(float(self.cfg.height_z_max), device=device, dtype=height_abs.dtype),
            "height_anchor_scale": torch.tensor(float(self.cfg.height_anchor_scale), device=device, dtype=height_abs.dtype),
            "height_local_scale": torch.tensor(float(self.cfg.height_local_scale), device=device, dtype=height_abs.dtype),
            "patch_valid_mask": patch_valid_mask,
            "patch_tokens_final": patch_tokens_final,
            "patch_tokens_match": patch_tokens_layer_sel,
            "patch_tokens_layer_for_nce": patch_tokens_layer_sel,
            "patch_tokens_nce_proj": patch_tokens_nce_proj,
            "view_tokens_final": view_tokens_final,
            "scene_token_final": scene_token_final,
            "patch_centers": patch_centers,
            "patch_grid_hw": (gh, gw),
            "patch_padded_hw": backbone_out["pad_hw"],
            "gaussian_branch_enabled": gaussian_enabled,
        }

        if gaussian_enabled:
            # 9) 高斯属性分支
            gauss_pred = self.gaussian_head(self.gaussian_adapter(dense_feat))
            gaussian_opacity = self._reshape_logits_to_bv(gauss_pred["opacity"], b, v)
            gaussian_scale = self._reshape_logits_to_bv(gauss_pred["scale"], b, v)
            gaussian_rotation = self._reshape_logits_to_bv(gauss_pred["rotation"], b, v)
            gaussian_sh = self._reshape_logits_to_bv(gauss_pred["sh"], b, v)
            gaussian_conf_rpc = self._reshape_logits_to_bv(gauss_pred["confidence_rpc"], b, v)
            gaussian_conf_point = self._reshape_logits_to_bv(gauss_pred["confidence_point"], b, v)

            # 10) 双路径中心
            # 3DGS 坐标语义统一为“局部米制”：
            # - rpc+height 路径：仅做 offset（scene center），不做 scale；
            # - point 路径：x/y 从归一化域乘回米制 scale，不反加 offset。
            scene_scale_for_rpc = None
            if scene_xy_center is not None and torch.is_tensor(scene_xy_center):
                scene_scale_for_rpc = torch.ones_like(scene_xy_center)
            centers_rpc = self.rpc_ops.centers_from_rpc_and_height_batch(
                corrected_rpc_batch=rpc_corrected,
                pixel_grid=image_grid,
                height_abs=height_abs,
                scene_xy_center=scene_xy_center,
                scene_xy_scale=scene_scale_for_rpc,
                downsample_factor=self._current_center_downsample(),
            )
            centers_point = self._point_map_norm_to_local_meter(point_abs, scene_xy_scale)
            out.update(
                {
                    "gaussian_opacity": gaussian_opacity,
                    "gaussian_scale": gaussian_scale,
                    "gaussian_rotation": gaussian_rotation,
                    "gaussian_sh": gaussian_sh,
                    "gaussian_confidence_rpc": gaussian_conf_rpc,
                    "gaussian_confidence_point": gaussian_conf_point,
                    "gaussian_centers_rpc": centers_rpc,
                    "gaussian_centers_point": centers_point,
                }
            )

        return out

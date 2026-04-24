"""heads.py

本文件实现 Sat2World 的任务头与 DPT 风格 dense 解码器：
- DPTDenseDecoder (AnySplat/VGGT 风格)
- AffineHead
- GaussianHead
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """通用卷积块：Conv-Norm-GELU（默认 GroupNorm，避免小 batch 下 BN 统计不稳定）。"""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        # 选择可整除的最大组数（上限 32），保证对任意通道数都可用。
        groups = 32
        while groups > 1 and (out_ch % groups != 0):
            groups //= 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        self.has_residual = has_residual
        self.res1 = ResidualConvUnit(features) if has_residual else nn.Identity()
        self.res2 = ResidualConvUnit(features)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, size: tuple[int, int] | None = None) -> torch.Tensor:
        out = x
        if residual is not None and self.has_residual:
            res = self.res1(residual)
            if size is not None and tuple(res.shape[-2:]) != tuple(size):
                res = F.interpolate(res, size=size, mode="bilinear", align_corners=True)
            if tuple(out.shape[-2:]) != tuple(res.shape[-2:]):
                out = F.interpolate(out, size=res.shape[-2:], mode="bilinear", align_corners=True)
            out = out + res
        out = self.res2(out)
        if size is not None and tuple(out.shape[-2:]) != tuple(size):
            out = F.interpolate(out, size=size, mode="bilinear", align_corners=True)
        return out


@dataclass
class DPTDenseDecoderCfg:
    out_ch: int = 256
    out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    intermediate_layer_idx: tuple[int, int, int, int] = (5, 11, 17, 23)
    pos_embed: bool = True
    down_ratio: int = 1
    frames_chunk_size: int = 8
    image_merge_ch: int = 128


class DPTDenseDecoder(nn.Module):
    """AnySplat/VGGT 风格 DPT dense 解码器。

    输入:
        patch_tokens_layers: [B,Layers,V,N,C]
        images: [B*V,3,H,W]
        patch_grid_hw: (Gh,Gw)
    输出:
        dense feature [B*V,out_ch,H/down_ratio,W/down_ratio]
    """

    def __init__(self, in_ch: int = 1024, cfg: DPTDenseDecoderCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DPTDenseDecoderCfg()
        oc = list(self.cfg.out_channels)
        feat = int(self.cfg.out_ch)

        self.norm = nn.LayerNorm(in_ch)
        self.projects = nn.ModuleList([nn.Conv2d(in_ch, c, kernel_size=1, stride=1, padding=0) for c in oc])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(oc[0], oc[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(oc[1], oc[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(oc[3], oc[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.layer_rn = nn.ModuleList([
            nn.Conv2d(oc[0], feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(oc[1], feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(oc[2], feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(oc[3], feat, kernel_size=3, stride=1, padding=1, bias=False),
        ])

        self.refinenet4 = FeatureFusionBlock(feat, has_residual=False)
        self.refinenet3 = FeatureFusionBlock(feat, has_residual=True)
        self.refinenet2 = FeatureFusionBlock(feat, has_residual=True)
        self.refinenet1 = FeatureFusionBlock(feat, has_residual=True)

        self.output_conv1 = nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1)
        self.input_merger = nn.Sequential(
            nn.Conv2d(3, int(self.cfg.image_merge_ch), kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.cfg.image_merge_ch), feat, kernel_size=3, stride=1, padding=1),
        )

    @staticmethod
    def _create_uv_grid(w: int, h: int, *, aspect_ratio: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        x = x * aspect_ratio
        return torch.stack([x, y], dim=-1)

    def _apply_pos_embed(self, x: torch.Tensor, w_full: int, h_full: int, ratio: float = 0.1) -> torch.Tensor:
        h, w = x.shape[-2:]
        grid = self._create_uv_grid(w, h, aspect_ratio=float(w_full) / float(max(h_full, 1)), device=x.device, dtype=x.dtype)
        # 简单傅里叶嵌入风格映射
        ch = x.shape[1]
        freq = torch.linspace(1.0, 8.0, max(ch // 4, 1), device=x.device, dtype=x.dtype)
        gx = grid[..., 0:1] * freq
        gy = grid[..., 1:2] * freq
        emb = torch.cat([torch.sin(gx), torch.cos(gx), torch.sin(gy), torch.cos(gy)], dim=-1)
        emb = emb[..., :ch]
        emb = emb.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return x + ratio * emb

    def _select_layers(self, patch_tokens_layers: torch.Tensor) -> list[torch.Tensor]:
        # patch_tokens_layers: [B,Layers,V,N,C]
        b, l_all, _, _, _ = patch_tokens_layers.shape
        del b
        selected = []
        for idx in self.cfg.intermediate_layer_idx:
            i = int(idx)
            if 0 <= i < l_all:
                selected.append(patch_tokens_layers[:, i])
            else:
                # 当 encoder 层数不足时，按等间距回退到可用层，保证可运行。
                rank = list(self.cfg.intermediate_layer_idx).index(idx)
                mapped = min(max(int(round((l_all - 1) * rank / 3.0)), 0), l_all - 1)
                selected.append(patch_tokens_layers[:, mapped])
        return selected

    def _forward_impl(
        self,
        patch_tokens_layers: torch.Tensor,
        images: torch.Tensor,
        patch_grid_hw: tuple[int, int],
        views_start: int | None = None,
        views_end: int | None = None,
    ) -> torch.Tensor:
        gh, gw = int(patch_grid_hw[0]), int(patch_grid_hw[1])
        b, _, v, n, c = patch_tokens_layers.shape
        if views_start is not None and views_end is not None:
            patch_tokens_layers = patch_tokens_layers[:, :, views_start:views_end].contiguous()
            v = views_end - views_start
            images = images.view(b, -1, *images.shape[1:])[:, views_start:views_end].reshape(b * v, *images.shape[1:])

        if n != gh * gw:
            raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh*gw}")

        selected_layers = self._select_layers(patch_tokens_layers)
        out_feats: list[torch.Tensor] = []

        for li, x in enumerate(selected_layers):
            # x: [B,V,N,C]
            x = x.reshape(b * v, n, c)
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(b * v, c, gh, gw)
            x = self.projects[li](x)
            if self.cfg.pos_embed:
                _, _, h_im, w_im = images.shape
                x = self._apply_pos_embed(x, w_im, h_im)
            x = self.resize_layers[li](x)
            out_feats.append(x)

        l1 = self.layer_rn[0](out_feats[0])
        l2 = self.layer_rn[1](out_feats[1])
        l3 = self.layer_rn[2](out_feats[2])
        l4 = self.layer_rn[3](out_feats[3])

        out = self.refinenet4(l4, size=l3.shape[-2:])
        out = self.refinenet3(out, l3, size=l2.shape[-2:])
        out = self.refinenet2(out, l2, size=l1.shape[-2:])
        out = self.refinenet1(out, l1)
        out = self.output_conv1(out)

        _, _, h_im, w_im = images.shape
        out_h = int(h_im // max(int(self.cfg.down_ratio), 1))
        out_w = int(w_im // max(int(self.cfg.down_ratio), 1))
        out = F.interpolate(out, size=(out_h, out_w), mode="bilinear", align_corners=True)

        img_feat = self.input_merger(images)
        if tuple(img_feat.shape[-2:]) != tuple(out.shape[-2:]):
            img_feat = F.interpolate(img_feat, size=out.shape[-2:], mode="bilinear", align_corners=True)
        out = out + img_feat
        return out

    def forward(self, patch_tokens_layers: torch.Tensor, images: torch.Tensor, patch_grid_hw: tuple[int, int]) -> torch.Tensor:
        b, _, v, _, _ = patch_tokens_layers.shape
        _, _, h, w = images.shape
        images_bv = images.view(b, v, 3, h, w)

        chunk = int(self.cfg.frames_chunk_size)
        if chunk <= 0 or chunk >= v:
            return self._forward_impl(patch_tokens_layers, images, patch_grid_hw)

        outs = []
        for s in range(0, v, chunk):
            e = min(s + chunk, v)
            out = self._forward_impl(
                patch_tokens_layers,
                images_bv.reshape(b * v, 3, h, w),
                patch_grid_hw,
                views_start=s,
                views_end=e,
            )
            outs.append(out.view(b, e - s, out.shape[1], out.shape[2], out.shape[3]))
        out_cat = torch.cat(outs, dim=1)
        return out_cat.view(b * v, out_cat.shape[2], out_cat.shape[3], out_cat.shape[4])


class TaskAdapter(nn.Module):
    """任务专属轻量适配器。"""

    def __init__(self, ch: int = 256, depth: int = 2) -> None:
        super().__init__()
        blocks = [ConvBlock(ch, ch) for _ in range(max(int(depth), 1))]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 兼容旧命名，内部直接切到 DPT 解码器。
class SharedDenseDecoder(DPTDenseDecoder):
    def __init__(self, in_ch: int = 1024, out_ch: int = 256) -> None:
        super().__init__(in_ch=in_ch, cfg=DPTDenseDecoderCfg(out_ch=out_ch))


@dataclass
class AffineHeadCfg:
    diag_scale: float = 0.01
    offdiag_scale: float = 0.01
    trans_scale: float = 50.0


class AffineHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, cfg: AffineHeadCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AffineHeadCfg()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(view_tokens)
        t = torch.tanh(raw)
        a00 = 1.0 + self.cfg.diag_scale * t[..., 0]
        a01 = self.cfg.offdiag_scale * t[..., 1]
        tx = self.cfg.trans_scale * t[..., 2]
        a10 = self.cfg.offdiag_scale * t[..., 3]
        a11 = 1.0 + self.cfg.diag_scale * t[..., 4]
        ty = self.cfg.trans_scale * t[..., 5]
        row0 = torch.stack([a00, a01, tx], dim=-1)
        row1 = torch.stack([a10, a11, ty], dim=-1)
        return torch.stack([row0, row1], dim=-2)


class SceneHeightAnchorHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, scene_token: torch.Tensor) -> torch.Tensor:
        return self.net(scene_token)


class DenseHeightLocalHead(nn.Module):
    def __init__(self, in_ch: int = 256, hidden_ch: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden_ch),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class PointXYHead(nn.Module):
    def __init__(self, in_ch: int = 256, num_bins_xy: int = 33) -> None:
        super().__init__()
        self.trunk = ConvBlock(in_ch, in_ch)
        self.x_logits = nn.Conv2d(in_ch, num_bins_xy, kernel_size=1)
        self.y_logits = nn.Conv2d(in_ch, num_bins_xy, kernel_size=1)
        self.x_fine = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.y_fine = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.trunk(feat)
        return {
            "x_logits": self.x_logits(x),
            "y_logits": self.y_logits(x),
            "x_fine": self.x_fine(x),
            "y_fine": self.y_fine(x),
        }


class PointZLocalHead(nn.Module):
    def __init__(self, in_ch: int = 256, hidden_ch: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_ch, hidden_ch),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class GaussianHead(nn.Module):
    def __init__(self, in_ch: int = 256, sh_dim: int = 48, scale_max: float = 0.3, scale_eps: float = 1e-4) -> None:
        super().__init__()
        self.scale_max = scale_max
        self.scale_eps = scale_eps
        self.trunk = ConvBlock(in_ch, in_ch)
        self.opacity = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.scale = nn.Conv2d(in_ch, 3, kernel_size=1)
        self.rotation = nn.Conv2d(in_ch, 4, kernel_size=1)
        self.sh = nn.Conv2d(in_ch, sh_dim, kernel_size=1)
        self.conf_rpc = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.conf_point = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.trunk(feat)
        opacity = torch.sigmoid(self.opacity(x))
        scale = F.softplus(self.scale(x)) + self.scale_eps
        scale = scale.clamp_max(self.scale_max)
        rotation = F.normalize(self.rotation(x), p=2, dim=1, eps=1e-8)
        sh = self.sh(x)
        conf_rpc = torch.sigmoid(self.conf_rpc(x))
        conf_point = torch.sigmoid(self.conf_point(x))
        return {
            "opacity": opacity,
            "scale": scale,
            "rotation": rotation,
            "sh": sh,
            "confidence_rpc": conf_rpc,
            "confidence_point": conf_point,
        }

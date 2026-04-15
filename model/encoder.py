"""encoder.py

AnySplat/VGGT 风格交替注意力编码器（Sat2World 版本）。

核心实现：
- 每视图 token 组织为 [view_token(1), scene_tokens(4), patch_tokens(L)]；
- Frame Attention:  (B*V, P, C)
- Global Attention: (B, V*P, C)
- 对 patch token 使用 2D RoPE；special tokens 位置为 0；
- 引入 LayerScale / DropPath / qk_norm；
- 注意力计算采用 F.scaled_dot_product_attention。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlternatingEncoderCfg:
    """交替编码器配置（AnySplat 对齐版）。"""

    dim: int = 1024
    num_heads: int = 8
    ffn_ratio: float = 4.0
    num_layers: int = 4
    dropout: float = 0.0

    # AnySplat 风格增强项
    num_scene_tokens: int = 4
    aa_order: tuple[str, ...] = ("frame", "global")
    aa_block_size: int = 1
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    qk_norm: bool = True
    fused_attn: bool = True
    rope_freq: float = 100.0
    drop_path_rate: float = 0.0
    init_values: float = 1e-2


class DropPath(nn.Module):
    """Stochastic Depth。"""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor


class LayerScale(nn.Module):
    """LayerScale。"""

    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * float(init_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PositionGetter:
    """生成并缓存 2D patch 位置。"""

    def __init__(self) -> None:
        self.cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self.cache:
            ys = torch.arange(height, device=device)
            xs = torch.arange(width, device=device)
            pos = torch.cartesian_prod(ys, xs)  # [H*W,2]
            self.cache[key] = pos
        return self.cache[key].view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D RoPE。"""

    def __init__(self, frequency: float = 100.0) -> None:
        super().__init__()
        self.base_frequency = float(frequency)
        self.cache: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _get_freq(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (dim, seq_len, device, dtype)
        if key not in self.cache:
            exponents = torch.arange(0, dim, 2, device=device, dtype=torch.float32) / float(dim)
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat([angles, angles], dim=-1)
            self.cache[key] = (angles.cos(), angles.sin())
        return self.cache[key]

    def _apply_1d(self, t: torch.Tensor, p: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # t: [B,H,N,D], p:[B,N]
        cos_p = F.embedding(p, cos)[:, None, :, :]
        sin_p = F.embedding(p, sin)[:, None, :, :]
        return t * cos_p + self._rotate_half(t) * sin_p

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # tokens: [B,H,N,D], positions:[B,N,2]
        if positions is None:
            return tokens
        d = tokens.shape[-1]
        if d % 2 != 0:
            return tokens
        d_half = d // 2
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        cos, sin = self._get_freq(d_half, max_pos, tokens.device, tokens.dtype)
        tv, th = tokens.chunk(2, dim=-1)
        pv = positions[..., 0].long().clamp_min(0)
        ph = positions[..., 1].long().clamp_min(0)
        tv = self._apply_1d(tv, pv, cos, sin)
        th = self._apply_1d(th, ph, cos, sin)
        return torch.cat([tv, th], dim=-1)


class Mlp(nn.Module):
    """FFN。"""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionSDPA(nn.Module):
    """QKNorm + RoPE + SDPA Attention。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope: RotaryPositionEmbedding2D | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = bool(fused_attn)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AAttnBlock(nn.Module):
    """AnySplat 风格 Transformer Block。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        init_values: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        fused_attn: bool,
        rope: RotaryPositionEmbedding2D | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionSDPA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=dropout,
            proj_drop=dropout,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values > 0 else nn.Identity()
        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=dropout,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values > 0 else nn.Identity()
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dp1(self.ls1(self.attn(self.norm1(x), pos=pos)))
        x = x + self.dp2(self.ls2(self.mlp(self.norm2(x))))
        return x


class AlternatingEncoder(nn.Module):
    """AnySplat 对齐版交替编码器。"""

    def __init__(self, cfg: AlternatingEncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if cfg.aa_block_size <= 0:
            raise ValueError("aa_block_size must be > 0")
        if cfg.num_layers % cfg.aa_block_size != 0:
            raise ValueError("num_layers must be divisible by aa_block_size")

        self.num_scene_tokens = int(cfg.num_scene_tokens)
        self.patch_start_idx = 1 + self.num_scene_tokens

        # 特殊 token（参考视图 / 其他视图）
        self.view_token_ref = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.view_token_other = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.scene_tokens_ref = nn.Parameter(torch.zeros(1, self.num_scene_tokens, cfg.dim))
        self.scene_tokens_other = nn.Parameter(torch.zeros(1, self.num_scene_tokens, cfg.dim))
        nn.init.trunc_normal_(self.view_token_ref, std=0.02)
        nn.init.trunc_normal_(self.view_token_other, std=0.02)
        nn.init.trunc_normal_(self.scene_tokens_ref, std=0.02)
        nn.init.trunc_normal_(self.scene_tokens_other, std=0.02)

        rope = RotaryPositionEmbedding2D(frequency=cfg.rope_freq) if cfg.rope_freq > 0 else None
        self.position_getter = PositionGetter() if rope is not None else None

        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.num_layers).tolist()
        self.frame_blocks = nn.ModuleList(
            [
                AAttnBlock(
                    dim=cfg.dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.ffn_ratio,
                    dropout=cfg.dropout,
                    drop_path=float(dpr[i]),
                    init_values=cfg.init_values,
                    qkv_bias=cfg.qkv_bias,
                    proj_bias=cfg.proj_bias,
                    ffn_bias=cfg.ffn_bias,
                    qk_norm=cfg.qk_norm,
                    fused_attn=cfg.fused_attn,
                    rope=rope,
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.global_blocks = nn.ModuleList(
            [
                AAttnBlock(
                    dim=cfg.dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.ffn_ratio,
                    dropout=cfg.dropout,
                    drop_path=float(dpr[i]),
                    init_values=cfg.init_values,
                    qkv_bias=cfg.qkv_bias,
                    proj_bias=cfg.proj_bias,
                    ffn_bias=cfg.ffn_bias,
                    qk_norm=cfg.qk_norm,
                    fused_attn=cfg.fused_attn,
                    rope=rope,
                )
                for i in range(cfg.num_layers)
            ]
        )

    @staticmethod
    def patch_tokens_to_map(patch_tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """把 [B,V,N,C] 还原为 [B,V,C,Gh,Gw]。"""
        b, v, n, c = patch_tokens.shape
        gh, gw = grid_hw
        if n != gh * gw:
            raise ValueError(f"Token number mismatch: N={n}, Gh*Gw={gh*gw}")
        return patch_tokens.view(b, v, gh, gw, c).permute(0, 1, 4, 2, 3).contiguous()

    def _build_ref_mask(self, b: int, v: int, ref_view_idx: int | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(ref_view_idx, int):
            idx = torch.full((b,), ref_view_idx, dtype=torch.long, device=device)
        else:
            idx = ref_view_idx.to(device=device, dtype=torch.long).view(-1)
            if idx.numel() != b:
                raise ValueError(f"ref_view_idx must have B={b} entries, got {idx.numel()}")
        idx = idx.clamp(0, v - 1)
        mask = torch.zeros((b, v), dtype=torch.bool, device=device)
        mask[torch.arange(b, device=device), idx] = True
        return mask

    def _build_special_tokens(self, b: int, v: int, ref_mask: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        view_ref = self.view_token_ref.to(device=device).expand(b, v, -1)
        view_other = self.view_token_other.to(device=device).expand(b, v, -1)
        view_tok = torch.where(ref_mask.unsqueeze(-1), view_ref, view_other).unsqueeze(2)  # [B,V,1,C]

        scene_ref = self.scene_tokens_ref.to(device=device).expand(b, v, -1, -1)
        scene_other = self.scene_tokens_other.to(device=device).expand(b, v, -1, -1)
        scene_tok = torch.where(ref_mask.unsqueeze(-1).unsqueeze(-1), scene_ref, scene_other)  # [B,V,S,C]
        return view_tok, scene_tok

    def _build_pos(self, b: int, v: int, l: int, grid_hw: tuple[int, int], device: torch.device) -> torch.Tensor | None:
        if self.position_getter is None:
            return None
        gh, gw = grid_hw
        if l != gh * gw:
            raise ValueError(f"patch token count mismatch for RoPE: L={l}, Gh*Gw={gh*gw}")
        patch_pos = self.position_getter(b * v, gh, gw, device=device).view(b, v, l, 2)
        # 为 special tokens 注入视图索引位置信息，避免跨视图时 special token 位置完全相同。
        # pos[...,0] 使用 view_idx；pos[...,1] 使用 special token 的槽位编号（0..patch_start_idx-1）。
        view_idx = torch.arange(v, device=device, dtype=patch_pos.dtype).view(1, v, 1).expand(b, v, self.patch_start_idx)
        slot_idx = torch.arange(self.patch_start_idx, device=device, dtype=patch_pos.dtype).view(1, 1, self.patch_start_idx).expand(b, v, self.patch_start_idx)
        special_pos = torch.stack([view_idx, slot_idx], dim=-1)
        return torch.cat([special_pos, patch_pos], dim=2)

    def _run_frame(self, tokens: torch.Tensor, pos: torch.Tensor | None, layer_idx: int) -> torch.Tensor:
        b, v, p, c = tokens.shape
        t = tokens.view(b * v, p, c)
        pp = None if pos is None else pos.view(b * v, p, 2)
        t = self.frame_blocks[layer_idx](t, pos=pp)
        return t.view(b, v, p, c)

    def _run_global(self, tokens: torch.Tensor, pos: torch.Tensor | None, layer_idx: int) -> torch.Tensor:
        b, v, p, c = tokens.shape
        t = tokens.view(b, v * p, c)
        pp = None if pos is None else pos.view(b, v * p, 2)
        t = self.global_blocks[layer_idx](t, pos=pp)
        return t.view(b, v, p, c)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        ref_view_idx: int | torch.Tensor = 0,
        patch_grid_hw: tuple[int, int] | None = None,
        return_all_layers: bool = False,
    ) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 4:
            raise ValueError(f"patch_tokens must be [B,V,N,C], got {tuple(patch_tokens.shape)}")
        if patch_valid_mask.ndim != 3:
            raise ValueError(f"patch_valid_mask must be [B,V,N], got {tuple(patch_valid_mask.shape)}")

        b, v, l, c = patch_tokens.shape
        if c != self.cfg.dim:
            raise ValueError(f"Channel mismatch: got {c}, expect {self.cfg.dim}")

        # RoPE 需要 patch 网格尺寸；优先使用显式传入的 patch_grid_hw。
        if patch_grid_hw is not None:
            gh, gw = int(patch_grid_hw[0]), int(patch_grid_hw[1])
            if gh * gw != l:
                raise ValueError(f"patch_grid_hw mismatch: Gh*Gw={gh*gw}, L={l}")
        else:
            gh = int(round(l**0.5))
            gw = gh
            if gh * gw != l:
                # fallback 为单行网格，保证兼容历史调用。
                gh, gw = 1, l

        ref_mask = self._build_ref_mask(b, v, ref_view_idx, device=patch_tokens.device)
        view_tok, scene_tok = self._build_special_tokens(b, v, ref_mask, device=patch_tokens.device)

        tokens = torch.cat([view_tok, scene_tok, patch_tokens], dim=2)  # [B,V,L+5,C]
        pos = self._build_pos(b, v, l, (gh, gw), device=patch_tokens.device)

        patch_layers: list[torch.Tensor] = []
        for li in range(self.cfg.num_layers):
            for attn_type in self.cfg.aa_order:
                if attn_type == "frame":
                    tokens = self._run_frame(tokens, pos, li)
                elif attn_type == "global":
                    tokens = self._run_global(tokens, pos, li)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            if return_all_layers:
                patch_layers.append(tokens[:, :, self.patch_start_idx :, :])

        patch_out = tokens[:, :, self.patch_start_idx :, :]
        view_out = tokens[:, :, 0, :]
        scene_tokens_out = tokens[:, :, 1 : self.patch_start_idx, :]
        scene_out = scene_tokens_out.mean(dim=(1, 2))

        out = {
            "patch_tokens": patch_out,
            "view_tokens": view_out,
            "scene_token": scene_out,
            "scene_tokens": scene_tokens_out,
        }
        if return_all_layers:
            out["patch_tokens_layers"] = torch.stack(patch_layers, dim=1)
        return out

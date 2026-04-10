"""基于 diff-gaussian-rasterization 的 RPC 渲染后端。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from render.rpc_gaussian_renderer import (
    RPCGaussianRenderer,
    RPCGaussianRendererCfg,
    VirtualPinholeCamera,
    cov3d_from_scale_rotation,
)


@dataclass
class RPCGaussianRendererDGRCfg(RPCGaussianRendererCfg):
    dgr_debug: bool = False
    dgr_prefiltered: bool = False
    dgr_scale_modifier: float = 1.0
    dgr_near_plane_min: float = 1.0e-3
    dgr_depth_eps: float = 1.0e-6


class RPCGaussianRendererDGR(RPCGaussianRenderer):
    """复用 Sat2World 现有上层流程，仅替换底层 CUDA 光栅化后端。"""

    def __init__(self, geometry_ops, cfg: RPCGaussianRendererDGRCfg | None = None) -> None:
        super().__init__(geometry_ops=geometry_ops, cfg=cfg or RPCGaussianRendererDGRCfg())

    @staticmethod
    def _projection_from_k(K: torch.Tensor, width: int, height: int, near: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        proj = torch.zeros((4, 4), dtype=torch.float32, device=K.device)
        proj[0, 0] = 2.0 * fx / float(max(width, 1))
        proj[0, 2] = 2.0 * cx / float(max(width, 1)) - 1.0
        proj[1, 1] = 2.0 * fy / float(max(height, 1))
        proj[1, 2] = 2.0 * cy / float(max(height, 1)) - 1.0
        proj[2, 2] = far / (far - near)
        proj[2, 3] = -(far * near) / (far - near)
        proj[3, 2] = 1.0
        return proj

    def _render_cuda(
        self,
        centers: torch.Tensor,
        opacity: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        rgb: torch.Tensor,
        cam: VirtualPinholeCamera,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 diff_gaussian_rasterization，无法进行 DGR 渲染。"
            ) from e

        h, w = image_hw
        device = centers.device
        means = centers.to(dtype=torch.float32)
        op = opacity[:, 0].to(dtype=torch.float32).clamp(0.0, 1.0)
        colors = rgb.to(dtype=torch.float32).clamp(0.0, 1.0)

        cov3d = cov3d_from_scale_rotation(scale.to(torch.float32), rotation.to(torch.float32))
        row, col = torch.triu_indices(3, 3, device=device)
        cov3d_upper = cov3d[:, row, col].contiguous()

        ones = torch.ones((means.shape[0], 1), device=device, dtype=means.dtype)
        means_h = torch.cat([means, ones], dim=-1)
        cam_space = (cam.w2c.to(device=device, dtype=means.dtype) @ means_h.T).T
        depth = cam_space[:, 2]
        valid = depth > float(getattr(self.cfg, "dgr_depth_eps", 1.0e-6))

        if not bool(valid.any()):
            rgb_out = torch.zeros((3, h, w), device=device, dtype=torch.float32)
            alpha_out = torch.zeros((1, h, w), device=device, dtype=torch.float32)
            depth_out = torch.zeros((1, h, w), device=device, dtype=torch.float32)
            return rgb_out, alpha_out, depth_out

        near = depth[valid].min() * 0.5
        near = near.clamp_min(float(getattr(self.cfg, "dgr_near_plane_min", 1.0e-3)))
        far = depth[valid].max() * 1.5
        far = torch.maximum(far, near + 1.0)

        K = cam.K.to(device=device, dtype=torch.float32)
        fx = K[0, 0].clamp_min(1.0e-8)
        fy = K[1, 1].clamp_min(1.0e-8)
        tanfovx = float((0.5 * float(max(w, 1))) / float(fx.item()))
        tanfovy = float((0.5 * float(max(h, 1))) / float(fy.item()))

        w2c = cam.w2c.to(device=device, dtype=torch.float32)
        c2w = torch.linalg.inv(w2c)
        view_matrix = w2c.transpose(0, 1).contiguous()
        proj_matrix = self._projection_from_k(K, w, h, near, far).transpose(0, 1).contiguous()
        full_projection = (view_matrix @ proj_matrix).contiguous()

        means2d = torch.zeros_like(means, requires_grad=True)
        try:
            means2d.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros((3,), device=device, dtype=torch.float32),
            scale_modifier=float(getattr(self.cfg, "dgr_scale_modifier", 1.0)),
            viewmatrix=view_matrix,
            projmatrix=full_projection,
            sh_degree=0,
            campos=c2w[:3, 3],
            prefiltered=bool(getattr(self.cfg, "dgr_prefiltered", False)),
            debug=bool(getattr(self.cfg, "dgr_debug", False)),
        )
        rasterizer = GaussianRasterizer(settings)

        rendered_image, _radii = rasterizer(
            means3D=means,
            means2D=means2d,
            shs=None,
            colors_precomp=colors.contiguous(),
            opacities=op.unsqueeze(-1),
            cov3D_precomp=cov3d_upper,
        )

        # 为得到与颜色无关的 coverage/alpha 统计，再执行一次“白色渲染”近似 alpha。
        ones_rgb = torch.ones_like(colors)
        alpha_img, _ = rasterizer(
            means3D=means,
            means2D=means2d,
            shs=None,
            colors_precomp=ones_rgb.contiguous(),
            opacities=op.unsqueeze(-1),
            cov3D_precomp=cov3d_upper,
        )

        rgb_out = rendered_image.contiguous().clamp(0.0, 1.0)
        alpha_out = alpha_img.mean(dim=0, keepdim=True).contiguous().clamp(0.0, 1.0)
        depth_out = torch.zeros((1, h, w), device=device, dtype=torch.float32)
        return rgb_out, alpha_out, depth_out

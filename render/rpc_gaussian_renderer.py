"""render.rpc_gaussian_renderer

本文件实现 Sat2World 的统一 RPC Gaussian 渲染接口层。

设计目标：
1) renderer 不负责构造中心，中心来自 model outputs；
2) renderer 不负责计算 render loss，只输出结构化渲染结果；
3) 目标成像模型固定使用 outputs['rpc_corrected']；
4) 默认排除 target 自身 source Gaussian；
5) 当前只使用 gaussian_sh 前 3 通道作为 base RGB；
6) 当前使用纯 PyTorch 近似 splatting，优先训练闭环可用与接口稳定。

坐标与方向约定：
- 像空间使用 (line, samp)；
- 世界中心使用 (x, y, h)；
- scene_xy_center / scene_xy_scale 顺序为 (y, x)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """归一化四元数。

    输入:
        q: [...,4]。
    输出:
        [...,4]，L2 归一化结果。
    """
    if q.shape[-1] != 4:
        raise ValueError("quaternion last dim must be 4")
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """四元数转旋转矩阵。

    输入:
        q: [...,4]，按 (w,x,y,z) 解释。
    输出:
        [...,3,3]。
    """
    q = normalize_quaternion(q)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


def cov3d_from_scale_rotation(scale: torch.Tensor, quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """由 scale 与 quaternion 构造 3D 协方差。

    公式:
        Sigma_3d = R * diag(scale^2) * R^T
    """
    s = scale.clamp_min(eps)
    r = quaternion_to_matrix(quaternion)
    d = torch.diag_embed(s * s)
    return r @ d @ r.transpose(-1, -2)


def sh_basecolor_to_rgb(gaussian_sh: torch.Tensor) -> torch.Tensor:
    """将 SH 前 3 通道解释为 base RGB。

    说明:
        当前仅使用 degree-0/base color；若 sh_dim>3 其余通道忽略。
    """
    if gaussian_sh.shape[-1] < 3:
        raise ValueError("gaussian_sh last dim must be >=3")
    return torch.sigmoid(gaussian_sh[..., :3])


def affine_scale_for_render_size(h: int, w: int, h_out: int, w_out: int) -> tuple[float, float]:
    """计算渲染分辨率缩放系数 (line_scale, samp_scale)。"""
    return float(h_out) / float(max(h, 1)), float(w_out) / float(max(w, 1))


def project_centers_to_view(
    geometry_ops: Any,
    centers_world: torch.Tensor,
    target_rpc: Any,
    scene_xy_center: torch.Tensor | None,
    scene_xy_scale: torch.Tensor | None,
) -> torch.Tensor:
    """将世界中心投影到目标视图像空间。

    输入:
        centers_world: [N,3] (x,y,h)。
    输出:
        mean_2d: [N,2] (line,samp)。
    """
    if centers_world.numel() == 0:
        return centers_world.new_zeros((0, 2))

    x = centers_world[:, 0].view(1, 1, -1)
    y = centers_world[:, 1].view(1, 1, -1)
    h = centers_world[:, 2].view(1, 1, -1)

    lines, samps = geometry_ops.xy_to_linesamp_batch(
        rpc_batch=[[target_rpc]],
        xs=x,
        ys=y,
        heights=h,
        scene_xy_center=None if scene_xy_center is None else scene_xy_center.view(1, 2),
        scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale.view(1, 2),
    )
    return torch.stack([lines.view(-1), samps.view(-1)], dim=-1)


def estimate_local_view_direction(
    geometry_ops: Any,
    target_rpc: Any,
    image_h: int,
    image_w: int,
    h_ref: float,
    scene_xy_center: torch.Tensor | None,
    scene_xy_scale: torch.Tensor | None,
    delta_h_dir: float = 10.0,
) -> torch.Tensor:
    """估计目标视图局部视线方向。

    方法:
        在中心像素处取 h_ref 和 h_ref+delta_h 两个高度反投影，方向取差分单位向量。
    """
    base_device = getattr(target_rpc, "device", torch.device("cpu"))
    c_line = torch.tensor([(image_h - 1) * 0.5], dtype=torch.float32, device=base_device)
    c_samp = torch.tensor([(image_w - 1) * 0.5], dtype=torch.float32, device=base_device)

    h0 = torch.tensor([float(h_ref)], dtype=torch.float32, device=base_device)
    h1 = torch.tensor([float(h_ref + delta_h_dir)], dtype=torch.float32, device=base_device)

    x0, y0 = geometry_ops.linesamp_to_xy(
        target_rpc,
        lines=c_line,
        samps=c_samp,
        heights=h0,
        xy_center=scene_xy_center,
        xy_scale=scene_xy_scale,
    )
    x1, y1 = geometry_ops.linesamp_to_xy(
        target_rpc,
        lines=c_line,
        samps=c_samp,
        heights=h1,
        xy_center=scene_xy_center,
        xy_scale=scene_xy_scale,
    )
    v = torch.stack([x1 - x0, y1 - y0, h1 - h0], dim=-1).view(3)
    return v / v.norm().clamp_min(1e-8)


def compute_depth_proxy(centers_world: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
    """计算深度排序代理值。"""
    return (centers_world * view_dir.view(1, 3)).sum(dim=-1)


def jacobian_rpc_projection_finite_diff(
    geometry_ops: Any,
    centers_world: torch.Tensor,
    target_rpc: Any,
    scene_xy_center: torch.Tensor | None,
    scene_xy_scale: torch.Tensor | None,
    eps_xy: float,
    eps_h: float,
) -> torch.Tensor:
    """有限差分估计 RPC 投影 Jacobian J=d(line,samp)/d(x,y,h)。"""
    n = centers_world.shape[0]
    if n == 0:
        return centers_world.new_zeros((0, 2, 3))

    c = centers_world
    ex = torch.tensor([eps_xy, 0.0, 0.0], device=c.device, dtype=c.dtype)
    ey = torch.tensor([0.0, eps_xy, 0.0], device=c.device, dtype=c.dtype)
    eh = torch.tensor([0.0, 0.0, eps_h], device=c.device, dtype=c.dtype)

    def _proj(p: torch.Tensor) -> torch.Tensor:
        return project_centers_to_view(geometry_ops, p, target_rpc, scene_xy_center, scene_xy_scale)

    p_xp = _proj(c + ex)
    p_xm = _proj(c - ex)
    p_yp = _proj(c + ey)
    p_ym = _proj(c - ey)
    p_hp = _proj(c + eh)
    p_hm = _proj(c - eh)

    d_dx = (p_xp - p_xm) / (2.0 * eps_xy)
    d_dy = (p_yp - p_ym) / (2.0 * eps_xy)
    d_dh = (p_hp - p_hm) / (2.0 * eps_h)

    return torch.stack([d_dx, d_dy, d_dh], dim=-1)  # [N,2,3]


def project_cov3d_to_cov2d(sigma_3d: torch.Tensor, jacobian: torch.Tensor, eps_cov: float) -> torch.Tensor:
    """协方差投影 Sigma_2d = J Sigma_3d J^T + eps I。"""
    s2 = jacobian @ sigma_3d @ jacobian.transpose(-1, -2)
    eye = torch.eye(2, device=s2.device, dtype=s2.dtype).view(1, 2, 2)
    return s2 + eps_cov * eye


def rescale_projection_for_output_size(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    h: int,
    w: int,
    h_out: int,
    w_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将投影均值与协方差重标定到输出分辨率。"""
    sl, ss = affine_scale_for_render_size(h, w, h_out, w_out)
    s = torch.tensor([[sl, 0.0], [0.0, ss]], device=mean_2d.device, dtype=mean_2d.dtype).view(1, 2, 2)
    m = mean_2d.clone()
    m[:, 0] = m[:, 0] * sl
    m[:, 1] = m[:, 1] * ss
    c = s @ cov_2d @ s.transpose(-1, -2)
    return m, c


def compute_bbox_from_cov2d(mean_2d: torch.Tensor, cov_2d: torch.Tensor, image_h: int, image_w: int, nsigma: float) -> torch.Tensor:
    """根据 2D 协方差估计 axis-aligned bbox。"""
    if mean_2d.shape[0] == 0:
        return mean_2d.new_zeros((0, 4), dtype=torch.long)
    s_line = cov_2d[:, 0, 0].clamp_min(1e-8).sqrt() * nsigma
    s_samp = cov_2d[:, 1, 1].clamp_min(1e-8).sqrt() * nsigma

    l0 = torch.floor(mean_2d[:, 0] - s_line).long()
    l1 = torch.ceil(mean_2d[:, 0] + s_line).long()
    s0 = torch.floor(mean_2d[:, 1] - s_samp).long()
    s1 = torch.ceil(mean_2d[:, 1] + s_samp).long()

    l0 = l0.clamp(0, max(image_h - 1, 0))
    l1 = l1.clamp(0, max(image_h - 1, 0))
    s0 = s0.clamp(0, max(image_w - 1, 0))
    s1 = s1.clamp(0, max(image_w - 1, 0))
    return torch.stack([l0, l1, s0, s1], dim=-1)


def rasterize_projected_gaussians(
    mean_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    depth_proxy: torch.Tensor,
    rgb: torch.Tensor,
    opacity: torch.Tensor,
    height_value: torch.Tensor,
    image_h: int,
    image_w: int,
    nsigma: float,
    chunk_size: int,
    alpha_clamp_max: float,
    depth_sort_descending: bool,
    alpha_cov_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """纯 PyTorch 前向 splatting。

    说明:
        当前实现是“局部 bbox 逐 Gaussian 前向合成”的近似可微 rasterizer，
        重点是训练链路可用与接口稳定，不追求 CUDA 级性能。
    """
    device = mean_2d.device
    dtype = mean_2d.dtype
    n = mean_2d.shape[0]

    canvas_rgb = torch.zeros((3, image_h, image_w), device=device, dtype=dtype)
    canvas_alpha = torch.zeros((1, image_h, image_w), device=device, dtype=dtype)
    canvas_height = torch.zeros((1, image_h, image_w), device=device, dtype=dtype)
    trans = torch.ones((1, image_h, image_w), device=device, dtype=dtype)

    if n == 0:
        stats = {
            "num_input_gaussians": torch.tensor(0.0, device=device),
            "num_rasterized_gaussians": torch.tensor(0.0, device=device),
            "mean_transmittance_final": trans.mean().detach(),
            "alpha_coverage": torch.tensor(0.0, device=device),
        }
        return canvas_rgb, canvas_alpha, canvas_height, stats

    order = torch.argsort(depth_proxy, descending=depth_sort_descending)
    mean_2d = mean_2d[order]
    cov_2d = cov_2d[order]
    rgb = rgb[order]
    opacity = opacity[order, 0]
    height_value = height_value[order, 0]

    bboxes = compute_bbox_from_cov2d(mean_2d, cov_2d, image_h, image_w, nsigma)
    rasterized = 0

    eye = torch.eye(2, device=device, dtype=dtype)
    step = max(int(chunk_size), 1)
    for g0 in range(0, n, step):
        g1 = min(g0 + step, n)
        for g in range(g0, g1):
            l0, l1, s0, s1 = [int(x.item()) for x in bboxes[g]]
            if l1 < l0 or s1 < s0:
                continue
            rasterized += 1

            yy = torch.arange(l0, l1 + 1, device=device, dtype=dtype)
            xx = torch.arange(s0, s1 + 1, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")

            d0 = gy - mean_2d[g, 0]
            d1 = gx - mean_2d[g, 1]
            d = torch.stack([d0, d1], dim=-1)  # [ph,pw,2]

            # 低精度（fp16/bf16）下 torch.linalg.inv 不受支持，且数值稳定性也更差。
            # 这里局部提升到 fp32 做 2x2 逆与 Mahalanobis 计算，再回落到渲染 dtype。
            inv_cov32 = torch.linalg.inv((cov_2d[g].to(torch.float32) + 1e-8 * eye.to(torch.float32)))
            m = torch.einsum("...i,ij,...j->...", d.to(torch.float32), inv_cov32, d.to(torch.float32))
            w = torch.exp(-0.5 * m).to(dtype=dtype)

            alpha = torch.clamp(opacity[g] * w, 0.0, alpha_clamp_max).unsqueeze(0)
            trans_patch = trans[:, l0 : l1 + 1, s0 : s1 + 1].clone()
            contrib = trans_patch * alpha

            rgb_patch = canvas_rgb[:, l0 : l1 + 1, s0 : s1 + 1].clone()
            h_patch = canvas_height[:, l0 : l1 + 1, s0 : s1 + 1].clone()
            a_patch = canvas_alpha[:, l0 : l1 + 1, s0 : s1 + 1].clone()

            canvas_rgb[:, l0 : l1 + 1, s0 : s1 + 1] = rgb_patch + contrib * rgb[g].view(3, 1, 1)
            canvas_height[:, l0 : l1 + 1, s0 : s1 + 1] = h_patch + contrib * height_value[g]
            canvas_alpha[:, l0 : l1 + 1, s0 : s1 + 1] = a_patch + contrib
            trans[:, l0 : l1 + 1, s0 : s1 + 1] = trans_patch * (1.0 - alpha)

    rendered_height = canvas_height / canvas_alpha.clamp_min(1e-8)
    rendered_height = torch.where(canvas_alpha > 1e-6, rendered_height, torch.zeros_like(rendered_height))

    stats = {
        "num_input_gaussians": torch.tensor(float(n), device=device),
        "num_rasterized_gaussians": torch.tensor(float(rasterized), device=device),
        "mean_transmittance_final": trans.mean().detach(),
        "alpha_coverage": (canvas_alpha > alpha_cov_thresh).to(dtype).mean().detach(),
    }
    return canvas_rgb, canvas_alpha, rendered_height, stats


def project_gaussians_to_view(
    geometry_ops: Any,
    centers_world: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    target_rpc: Any,
    scene_xy_center: torch.Tensor | None,
    scene_xy_scale: torch.Tensor | None,
    eps_xy_fd: float,
    eps_h_fd: float,
    eps_cov: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """高斯 3D->2D 投影统一函数，返回 mean_2d 与 cov_2d。"""
    mean_2d = project_centers_to_view(geometry_ops, centers_world, target_rpc, scene_xy_center, scene_xy_scale)
    sigma_3d = cov3d_from_scale_rotation(scale, rotation)
    jac = jacobian_rpc_projection_finite_diff(
        geometry_ops,
        centers_world,
        target_rpc,
        scene_xy_center,
        scene_xy_scale,
        eps_xy=eps_xy_fd,
        eps_h=eps_h_fd,
    )
    cov_2d = project_cov3d_to_cov2d(sigma_3d, jac, eps_cov=eps_cov)
    return mean_2d, cov_2d


@dataclass
class RPCGaussianRendererCfg:
    """RPCGaussianRenderer 配置。"""

    train_num_target_views: int = 2
    val_num_target_views: int = 2
    use_all_targets_in_val: bool = False

    exclude_self_source: bool = True
    allow_self_source_if_single_view: bool = True

    source_stride: int = 2
    confidence_threshold: float = 0.05
    topk_per_target: int | None = 20000

    enable_voxelization: bool = False
    voxel_xy: float = 2.0
    voxel_z: float = 1.0

    render_downsample_factor_train: int = 2
    render_downsample_factor_val: int = 1

    nsigma: float = 2.5
    alpha_clamp_max: float = 0.98

    eps_xy_fd: float = 1e-2
    eps_h_fd: float = 1e-1
    eps_cov: float = 1e-4

    depth_dir_delta_h: float = 10.0
    depth_sort_descending: bool = False

    chunk_size: int = 512
    alpha_cov_thresh: float = 0.05
    render_compute_dtype: str = "fp16"

    deterministic_target_selection: bool = True


class RPCGaussianRenderer:
    """Sat2World 统一渲染接口。

    本类职责：
    - 接收 model outputs 与 batch，选择 target views；
    - 为每个 target 收集 source Gaussian；
    - 在 corrected RPC 下执行双路径（rpc/point）可微渲染；
    - 输出可直接喂给 RenderPathLoss 与可视化模块的结构化结果。

    非职责：
    - 不构造 Gaussian 中心（来自 outputs）；
    - 不计算 render loss；
    - 不修改 model/dataset/geometry 主体逻辑。

    重要约定：
    - affine_gt_forward: true->observed（在 batch 中）；
    - affine_pred: observed->true（在 outputs 中）；
    - 本 renderer 渲染时只使用 outputs['rpc_corrected'] 作为目标成像 RPC。
    """

    def __init__(self, geometry_ops: Any, cfg: RPCGaussianRendererCfg | None = None) -> None:
        """初始化 renderer。

        输入:
            geometry_ops: geometry.rpc_geometry.RPCGeometryOps 实例。
            cfg: 渲染配置。
        """
        self.geometry_ops = geometry_ops
        self.cfg = cfg or RPCGaussianRendererCfg()

    def _get_render_dtype(self, device: torch.device) -> torch.dtype:
        mode = str(getattr(self.cfg, "render_compute_dtype", "fp16")).lower()
        if mode == "bf16":
            return torch.bfloat16 if device.type == "cuda" else torch.float32
        if mode == "fp32":
            return torch.float32
        return torch.float16 if device.type == "cuda" else torch.float32

    def select_target_views(
        self,
        b: int,
        v: int,
        mode: str,
        ref_view_idx: torch.Tensor | None,
    ) -> list[list[int]]:
        """选择每个 sample 的 target views。"""
        device = ref_view_idx.device if torch.is_tensor(ref_view_idx) else torch.device("cpu")
        if ref_view_idx is None:
            ref = torch.zeros((b,), dtype=torch.long, device=device)
        else:
            ref = ref_view_idx.long().view(-1)
            if ref.numel() == 1:
                ref = ref.expand(b)

        out: list[list[int]] = []
        for bi in range(b):
            non_ref = [i for i in range(v) if i != int(ref[bi])]
            if mode == "train":
                k = self.cfg.train_num_target_views
                if len(non_ref) == 0:
                    if self.cfg.allow_self_source_if_single_view:
                        out.append([int(ref[bi])])
                    else:
                        out.append([])
                else:
                    out.append(non_ref[:k])
            else:
                if self.cfg.use_all_targets_in_val:
                    cand = non_ref
                else:
                    cand = non_ref[: self.cfg.val_num_target_views]
                if len(cand) == 0:
                    cand = [int(ref[bi])]
                out.append(cand)
        return out

    def voxelize_gaussians_world(
        self,
        centers: torch.Tensor,
        opacity: torch.Tensor,
        scale: torch.Tensor,
        rotation: torch.Tensor,
        rgb: torch.Tensor,
        confidence: torch.Tensor,
        voxel_xy: float,
        voxel_z: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """简单体素聚合（CPU dict 分桶）。"""
        n = centers.shape[0]
        if n == 0:
            return centers, opacity, scale, rotation, rgb, confidence

        c_cpu = centers.detach().cpu()
        key_xyz = torch.stack(
            [
                torch.floor(c_cpu[:, 0] / voxel_xy),
                torch.floor(c_cpu[:, 1] / voxel_xy),
                torch.floor(c_cpu[:, 2] / voxel_z),
            ],
            dim=-1,
        ).long()

        buckets: dict[tuple[int, int, int], list[int]] = {}
        for idx, k in enumerate(key_xyz.tolist()):
            buckets.setdefault((k[0], k[1], k[2]), []).append(idx)

        idx_groups = list(buckets.values())
        dev = centers.device

        out_centers, out_opacity, out_scale, out_rot, out_rgb, out_conf = [], [], [], [], [], []
        for g in idx_groups:
            ids = torch.tensor(g, device=dev, dtype=torch.long)
            conf = confidence[ids, 0]
            w = torch.softmax(conf, dim=0).view(-1, 1)
            out_centers.append((centers[ids] * w).sum(dim=0, keepdim=True))
            out_opacity.append((opacity[ids] * w).sum(dim=0, keepdim=True))
            out_scale.append((scale[ids] * w).sum(dim=0, keepdim=True))
            out_rgb.append((rgb[ids] * w).sum(dim=0, keepdim=True))
            q = normalize_quaternion((rotation[ids] * w).sum(dim=0, keepdim=True))
            out_rot.append(q)
            out_conf.append((confidence[ids] * w).sum(dim=0, keepdim=True))

        return (
            torch.cat(out_centers, dim=0),
            torch.cat(out_opacity, dim=0),
            torch.cat(out_scale, dim=0),
            torch.cat(out_rot, dim=0),
            torch.cat(out_rgb, dim=0),
            torch.cat(out_conf, dim=0),
        )

    def collect_source_gaussians_for_target(
        self,
        centers_path: torch.Tensor,
        gaussian_opacity: torch.Tensor,
        gaussian_scale: torch.Tensor,
        gaussian_rotation: torch.Tensor,
        gaussian_sh: torch.Tensor,
        path_confidence: torch.Tensor,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        target_view_indices: list[list[int]],
    ) -> list[dict[str, Any]]:
        """为每个 target 整理 source Gaussian 集合。"""
        b, v, _, h, w = centers_path.shape
        rgb_all = sh_basecolor_to_rgb(gaussian_sh.permute(0, 1, 3, 4, 2).contiguous())  # [B,V,H,W,3]

        packs: list[dict[str, Any]] = []
        for bi in range(b):
            for tv in target_view_indices[bi]:
                source_ids = list(range(v))
                if self.cfg.exclude_self_source and (v > 1 or not self.cfg.allow_self_source_if_single_view):
                    source_ids = [sid for sid in source_ids if sid != tv]

                if len(source_ids) == 0:
                    packs.append(
                        {
                            "centers_world": centers_path.new_zeros((0, 3)),
                            "opacity": centers_path.new_zeros((0, 1)),
                            "scale": centers_path.new_zeros((0, 3)),
                            "rotation": centers_path.new_zeros((0, 4)),
                            "rgb": centers_path.new_zeros((0, 3)),
                            "confidence": centers_path.new_zeros((0, 1)),
                            "source_view_ids": torch.zeros((0,), dtype=torch.long, device=centers_path.device),
                            "batch_index": bi,
                            "target_view_index": tv,
                            "target_rpc": outputs["rpc_corrected"][bi][tv],
                            "target_height_ref": float(batch["height_ref"][bi, tv].item()) if "height_ref" in batch else 0.0,
                            "raw_num_gaussians": 0,
                            "after_stride_num_gaussians": 0,
                            "after_conf_filter_num_gaussians": 0,
                            "after_voxel_num_gaussians": 0,
                        }
                    )
                    continue

                c = centers_path[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 3)
                o = gaussian_opacity[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 1)
                s = gaussian_scale[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 3)
                r = gaussian_rotation[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 4)
                col = rgb_all[bi, source_ids].reshape(-1, 3)
                conf = path_confidence[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 1)

                raw_n = c.shape[0]
                stride = max(int(self.cfg.source_stride), 1)
                idx = torch.arange(0, raw_n, stride, device=c.device)
                c, o, s, r, col, conf = c[idx], o[idx], s[idx], r[idx], col[idx], conf[idx]
                sid_map = torch.tensor(source_ids, device=c.device, dtype=torch.long)
                source_view_ids = sid_map.repeat_interleave(h * w)[idx]

                n_stride = c.shape[0]
                keep = conf[:, 0] >= float(self.cfg.confidence_threshold)
                if keep.any():
                    c, o, s, r, col, conf, source_view_ids = c[keep], o[keep], s[keep], r[keep], col[keep], conf[keep], source_view_ids[keep]
                else:
                    c = c[:0]
                    o = o[:0]
                    s = s[:0]
                    r = r[:0]
                    col = col[:0]
                    conf = conf[:0]
                    source_view_ids = source_view_ids[:0]

                n_conf = c.shape[0]
                if self.cfg.topk_per_target is not None and n_conf > self.cfg.topk_per_target:
                    topk = int(self.cfg.topk_per_target)
                    ids = torch.topk(conf[:, 0], k=topk, largest=True).indices
                    c, o, s, r, col, conf, source_view_ids = c[ids], o[ids], s[ids], r[ids], col[ids], conf[ids], source_view_ids[ids]

                if self.cfg.enable_voxelization and c.shape[0] > 0:
                    c, o, s, r, col, conf = self.voxelize_gaussians_world(
                        c,
                        o,
                        s,
                        r,
                        col,
                        conf,
                        voxel_xy=self.cfg.voxel_xy,
                        voxel_z=self.cfg.voxel_z,
                    )
                    source_view_ids = torch.zeros((c.shape[0],), dtype=torch.long, device=c.device) - 1

                packs.append(
                    {
                        "centers_world": c,
                        "opacity": o,
                        "scale": s,
                        "rotation": r,
                        "rgb": col,
                        "confidence": conf,
                        "source_view_ids": source_view_ids,
                        "batch_index": bi,
                        "target_view_index": tv,
                        "target_rpc": outputs["rpc_corrected"][bi][tv],
                        "target_height_ref": float(batch["height_ref"][bi, tv].item()) if "height_ref" in batch else 0.0,
                        "raw_num_gaussians": int(raw_n),
                        "after_stride_num_gaussians": int(n_stride),
                        "after_conf_filter_num_gaussians": int(n_conf),
                        "after_voxel_num_gaussians": int(c.shape[0]),
                    }
                )
        return packs

    def render_single_path_single_target(
        self,
        pack: dict[str, Any],
        batch: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        """渲染单路径单 target。

        输出字段与 RenderPathLoss / 可视化直接对接。
        """
        bi = int(pack["batch_index"])
        tv = int(pack["target_view_index"])

        target_rgb = batch["images"][bi, tv]
        target_h = batch["height_gt"][bi, tv] if "height_gt" in batch else None
        target_m = batch["height_valid_mask"][bi, tv] if "height_valid_mask" in batch else None

        h, w = int(target_rgb.shape[-2]), int(target_rgb.shape[-1])
        ds = self.cfg.render_downsample_factor_train if mode == "train" else self.cfg.render_downsample_factor_val
        ds = max(int(ds), 1)
        h_out, w_out = max(h // ds, 1), max(w // ds, 1)

        c = pack["centers_world"]
        if c.shape[0] == 0:
            rr = torch.zeros((3, h_out, w_out), device=target_rgb.device, dtype=target_rgb.dtype)
            ra = torch.zeros((1, h_out, w_out), device=target_rgb.device, dtype=target_rgb.dtype)
            rh = torch.zeros((1, h_out, w_out), device=target_rgb.device, dtype=target_rgb.dtype)
            return {
                "rendered_rgb": rr,
                "rendered_alpha": ra,
                "rendered_height": rh,
                "target_rgb": F.interpolate(target_rgb.unsqueeze(0), size=(h_out, w_out), mode="bilinear", align_corners=False).squeeze(0),
                "target_height": None if target_h is None else F.interpolate(target_h.unsqueeze(0), size=(h_out, w_out), mode="bilinear", align_corners=False).squeeze(0),
                "target_valid_mask": None if target_m is None else F.interpolate(target_m.unsqueeze(0), size=(h_out, w_out), mode="nearest").squeeze(0),
                "target_view_index": tv,
                "batch_index": bi,
                "num_gaussians_raw": pack["raw_num_gaussians"],
                "num_gaussians_after_stride": pack["after_stride_num_gaussians"],
                "num_gaussians_after_conf": pack["after_conf_filter_num_gaussians"],
                "num_gaussians_after_voxel": pack["after_voxel_num_gaussians"],
                "num_gaussians_visible": 0,
                "alpha_coverage": torch.tensor(0.0, device=target_rgb.device),
            }

        scene_c = batch.get("scene_xy_center", None)
        scene_s = batch.get("scene_xy_scale", None)
        scene_ci = None if scene_c is None else scene_c[bi]
        scene_si = None if scene_s is None else scene_s[bi]

        mean_2d, cov_2d = project_gaussians_to_view(
            self.geometry_ops,
            c,
            pack["scale"],
            pack["rotation"],
            pack["target_rpc"],
            scene_ci,
            scene_si,
            self.cfg.eps_xy_fd,
            self.cfg.eps_h_fd,
            self.cfg.eps_cov,
        )

        view_dir = estimate_local_view_direction(
            self.geometry_ops,
            pack["target_rpc"],
            h,
            w,
            pack["target_height_ref"],
            scene_ci,
            scene_si,
            delta_h_dir=self.cfg.depth_dir_delta_h,
        ).to(device=c.device, dtype=c.dtype)
        depth_proxy = compute_depth_proxy(c, view_dir)

        mean_2d, cov_2d = rescale_projection_for_output_size(mean_2d, cov_2d, h, w, h_out, w_out)

        # 粗可见性过滤
        visible = (mean_2d[:, 0] >= -2) & (mean_2d[:, 0] <= h_out + 1) & (mean_2d[:, 1] >= -2) & (mean_2d[:, 1] <= w_out + 1)
        mean_2d = mean_2d[visible]
        cov_2d = cov_2d[visible]
        depth_proxy = depth_proxy[visible]
        rgb = pack["rgb"][visible]
        opacity = torch.clamp(pack["opacity"][visible] * pack["confidence"][visible], 0.0, 1.0)
        h_val = c[visible, 2:3]

        rdtype = self._get_render_dtype(c.device)
        rr, ra, rh, stats = rasterize_projected_gaussians(
            mean_2d=mean_2d.to(rdtype),
            cov_2d=cov_2d.to(rdtype),
            depth_proxy=depth_proxy.to(rdtype),
            rgb=rgb.to(rdtype),
            opacity=opacity.to(rdtype),
            height_value=h_val.to(rdtype),
            image_h=h_out,
            image_w=w_out,
            nsigma=self.cfg.nsigma,
            chunk_size=self.cfg.chunk_size,
            alpha_clamp_max=self.cfg.alpha_clamp_max,
            depth_sort_descending=self.cfg.depth_sort_descending,
            alpha_cov_thresh=self.cfg.alpha_cov_thresh,
        )

        target_rgb_out = F.interpolate(target_rgb.unsqueeze(0), size=(h_out, w_out), mode="bilinear", align_corners=False).squeeze(0)
        target_h_out = None if target_h is None else F.interpolate(target_h.unsqueeze(0), size=(h_out, w_out), mode="bilinear", align_corners=False).squeeze(0)
        target_m_out = None if target_m is None else F.interpolate(target_m.unsqueeze(0), size=(h_out, w_out), mode="nearest").squeeze(0)

        return {
            "rendered_rgb": rr,
            "rendered_alpha": ra,
            "rendered_height": rh,
            "target_rgb": target_rgb_out,
            "target_height": target_h_out,
            "target_valid_mask": target_m_out,
            "target_view_index": tv,
            "batch_index": bi,
            "num_gaussians_raw": pack["raw_num_gaussians"],
            "num_gaussians_after_stride": pack["after_stride_num_gaussians"],
            "num_gaussians_after_conf": pack["after_conf_filter_num_gaussians"],
            "num_gaussians_after_voxel": pack["after_voxel_num_gaussians"],
            "num_gaussians_visible": int(visible.sum().item()),
            "alpha_coverage": stats["alpha_coverage"],
        }

    def render_single_path(
        self,
        centers_path: torch.Tensor,
        path_confidence: torch.Tensor,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        """渲染单一路径在全 batch 的所有目标视图。"""
        b, v = centers_path.shape[:2]
        target_views = self.select_target_views(b, v, mode, batch.get("ref_view_idx", None))

        packs = self.collect_source_gaussians_for_target(
            centers_path=centers_path,
            gaussian_opacity=outputs["gaussian_opacity"],
            gaussian_scale=outputs["gaussian_scale"],
            gaussian_rotation=outputs["gaussian_rotation"],
            gaussian_sh=outputs["gaussian_sh"],
            path_confidence=path_confidence,
            outputs=outputs,
            batch=batch,
            target_view_indices=target_views,
        )

        single_results = [self.render_single_path_single_target(p, batch, mode) for p in packs]

        if len(single_results) == 0:
            dev = centers_path.device
            return {
                "rendered_rgb": torch.zeros((0, 3, 0, 0), device=dev),
                "rendered_alpha": torch.zeros((0, 1, 0, 0), device=dev),
                "rendered_height": torch.zeros((0, 1, 0, 0), device=dev),
                "target_rgb": torch.zeros((0, 3, 0, 0), device=dev),
                "target_height": None,
                "target_valid_mask": None,
                "batch_indices": torch.zeros((0,), dtype=torch.long, device=dev),
                "target_view_indices": torch.zeros((0,), dtype=torch.long, device=dev),
                "num_targets": 0,
                "stats": {
                    "num_gaussians_raw_mean": 0.0,
                    "num_gaussians_after_conf_mean": 0.0,
                    "num_gaussians_after_voxel_mean": 0.0,
                    "num_gaussians_visible_mean": 0.0,
                    "alpha_coverage_mean": 0.0,
                },
            }

        rendered_rgb = torch.stack([x["rendered_rgb"] for x in single_results], dim=0)
        rendered_alpha = torch.stack([x["rendered_alpha"] for x in single_results], dim=0)
        rendered_height = torch.stack([x["rendered_height"] for x in single_results], dim=0)
        target_rgb = torch.stack([x["target_rgb"] for x in single_results], dim=0)

        any_height = all(x["target_height"] is not None for x in single_results)
        any_mask = all(x["target_valid_mask"] is not None for x in single_results)
        target_height = torch.stack([x["target_height"] for x in single_results], dim=0) if any_height else None
        target_valid_mask = torch.stack([x["target_valid_mask"] for x in single_results], dim=0) if any_mask else None

        batch_indices = torch.tensor([x["batch_index"] for x in single_results], device=rendered_rgb.device, dtype=torch.long)
        target_view_indices = torch.tensor([x["target_view_index"] for x in single_results], device=rendered_rgb.device, dtype=torch.long)

        stats = {
            "num_gaussians_raw_mean": float(sum(x["num_gaussians_raw"] for x in single_results) / max(len(single_results), 1)),
            "num_gaussians_after_conf_mean": float(sum(x["num_gaussians_after_conf"] for x in single_results) / max(len(single_results), 1)),
            "num_gaussians_after_voxel_mean": float(sum(x["num_gaussians_after_voxel"] for x in single_results) / max(len(single_results), 1)),
            "num_gaussians_visible_mean": float(sum(x["num_gaussians_visible"] for x in single_results) / max(len(single_results), 1)),
            "alpha_coverage_mean": float(torch.stack([x["alpha_coverage"] for x in single_results]).mean().item()),
        }

        return {
            "rendered_rgb": rendered_rgb,
            "rendered_alpha": rendered_alpha,
            "rendered_height": rendered_height,
            "target_rgb": target_rgb,
            "target_height": target_height,
            "target_valid_mask": target_valid_mask,
            "batch_indices": batch_indices,
            "target_view_indices": target_view_indices,
            "num_targets": int(len(single_results)),
            "stats": stats,
        }

    def render_paths(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        mode: str = "train",
    ) -> dict[str, dict[str, Any]]:
        """对外主接口：渲染 rpc path 与 point path。

        使用方式:
            outputs = model(batch)
            render_outputs = renderer.render_paths(outputs, batch, mode='train')
        """
        required = [
            "gaussian_centers_rpc",
            "gaussian_centers_point",
            "gaussian_opacity",
            "gaussian_scale",
            "gaussian_rotation",
            "gaussian_sh",
            "gaussian_confidence_rpc",
            "gaussian_confidence_point",
            "rpc_corrected",
        ]
        miss = [k for k in required if k not in outputs]
        if miss:
            raise KeyError(f"Renderer missing outputs keys: {miss}")

        rpc_path = self.render_single_path(
            centers_path=outputs["gaussian_centers_rpc"],
            path_confidence=outputs["gaussian_confidence_rpc"],
            outputs=outputs,
            batch=batch,
            mode=mode,
        )
        point_path = self.render_single_path(
            centers_path=outputs["gaussian_centers_point"],
            path_confidence=outputs["gaussian_confidence_point"],
            outputs=outputs,
            batch=batch,
            mode=mode,
        )
        return {"rpc": rpc_path, "point": point_path}

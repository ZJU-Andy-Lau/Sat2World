"""高性能 RPC->虚拟 pinhole 3DGS 渲染器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.spatial.transform
import torch
import torch.nn.functional as F


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if q.shape[-1] != 4:
        raise ValueError("quaternion last dim must be 4")
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
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


def axis_angle_to_matrix(rvec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Rodrigues axis-angle -> rotation matrix."""
    theta = torch.linalg.norm(rvec).clamp_min(eps)
    k = rvec / theta
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.stack(
        [
            torch.stack([torch.zeros_like(kx), -kz, ky]),
            torch.stack([kz, torch.zeros_like(kx), -kx]),
            torch.stack([-ky, kx, torch.zeros_like(kx)]),
        ]
    )
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    ct = torch.cos(theta)
    st = torch.sin(theta)
    return I + st * K + (1.0 - ct) * (K @ K)


def _normalize_points_2d(uv: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    mu = uv.mean(dim=0)
    d = torch.linalg.norm(uv - mu[None, :], dim=-1)
    s = torch.sqrt(torch.tensor(2.0, dtype=uv.dtype, device=uv.device)) / d.mean().clamp_min(eps)
    T = torch.eye(3, dtype=uv.dtype, device=uv.device)
    T[0, 0] = s
    T[1, 1] = s
    T[0, 2] = -s * mu[0]
    T[1, 2] = -s * mu[1]
    uv_h = torch.cat([uv, torch.ones((uv.shape[0], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    uv_n = (T @ uv_h.T).T[:, :2]
    return uv_n, T


def _normalize_points_3d(xyz: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    mu = xyz.mean(dim=0)
    d = torch.linalg.norm(xyz - mu[None, :], dim=-1)
    s = torch.sqrt(torch.tensor(3.0, dtype=xyz.dtype, device=xyz.device)) / d.mean().clamp_min(eps)
    U = torch.eye(4, dtype=xyz.dtype, device=xyz.device)
    U[0, 0] = s
    U[1, 1] = s
    U[2, 2] = s
    U[0, 3] = -s * mu[0]
    U[1, 3] = -s * mu[1]
    U[2, 3] = -s * mu[2]
    xyz_h = torch.cat([xyz, torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)], dim=-1)
    xyz_n = (U @ xyz_h.T).T[:, :3]
    return xyz_n, U


def _normalize_points_2d_np(uv: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    mu = uv.mean(axis=0)
    d = np.linalg.norm(uv - mu[None, :], axis=1)
    scale = np.sqrt(2.0) / max(float(d.mean()), eps)
    T = np.eye(3, dtype=np.float64)
    T[0, 0] = scale
    T[1, 1] = scale
    T[0, 2] = -scale * mu[0]
    T[1, 2] = -scale * mu[1]
    uv_h = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=np.float64)], axis=1)
    uv_n = (T @ uv_h.T).T[:, :2]
    return uv_n, T


def _normalize_points_3d_np(xyz: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    mu = xyz.mean(axis=0)
    d = np.linalg.norm(xyz - mu[None, :], axis=1)
    scale = np.sqrt(3.0) / max(float(d.mean()), eps)
    U = np.eye(4, dtype=np.float64)
    U[0, 0] = scale
    U[1, 1] = scale
    U[2, 2] = scale
    U[0, 3] = -scale * mu[0]
    U[1, 3] = -scale * mu[1]
    U[2, 3] = -scale * mu[2]
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    xyz_n = (U @ xyz_h.T).T[:, :3]
    return xyz_n, U


def cov3d_from_scale_rotation(scale: torch.Tensor, quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = scale.clamp_min(eps)
    r = quaternion_to_matrix(quaternion)
    d = torch.diag_embed(s * s)
    return r @ d @ r.transpose(-1, -2)


def sh_basecolor_to_rgb(gaussian_sh: torch.Tensor) -> torch.Tensor:
    if gaussian_sh.shape[-1] < 3:
        raise ValueError("gaussian_sh last dim must be >=3")
    return torch.sigmoid(gaussian_sh[..., :3])


def _scene_center_scale_for_rpc(scene_center: torch.Tensor | None) -> torch.Tensor | None:
    if scene_center is None:
        return None
    return torch.ones_like(scene_center)


def project_centers_to_view(
    geometry_ops: Any,
    centers_world: torch.Tensor,
    target_rpc: Any,
    scene_xy_center: torch.Tensor | None,
    scene_xy_scale: torch.Tensor | None,
) -> torch.Tensor:
    if centers_world.numel() == 0:
        return centers_world.new_zeros((0, 2))
    x = centers_world[:, 0].view(1, 1, -1)
    y = centers_world[:, 1].view(1, 1, -1)
    h = centers_world[:, 2].view(1, 1, -1)
    l, s = geometry_ops.xy_to_linesamp_batch(
        rpc_batch=[[target_rpc]],
        xs=x,
        ys=y,
        heights=h,
        scene_xy_center=None if scene_xy_center is None else scene_xy_center.view(1, 2),
        scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale.view(1, 2),
    )
    return torch.stack([l.view(-1), s.view(-1)], dim=-1)


def jacobian_rpc_projection_finite_diff(
    geometry_ops: Any,
    centers_world: torch.Tensor,
    target_rpc: Any,
    scene_xy_center: torch.Tensor | None,
    scene_xy_scale: torch.Tensor | None,
    eps_xy: float,
    eps_h: float,
) -> torch.Tensor:
    n = centers_world.shape[0]
    if n == 0:
        return centers_world.new_zeros((0, 2, 3))
    c = centers_world
    ex = torch.tensor([eps_xy, 0.0, 0.0], device=c.device, dtype=c.dtype)
    ey = torch.tensor([0.0, eps_xy, 0.0], device=c.device, dtype=c.dtype)
    eh = torch.tensor([0.0, 0.0, eps_h], device=c.device, dtype=c.dtype)

    def _proj(p: torch.Tensor) -> torch.Tensor:
        return project_centers_to_view(geometry_ops, p, target_rpc, scene_xy_center, scene_xy_scale)

    d_dx = (_proj(c + ex) - _proj(c - ex)) / (2.0 * eps_xy)
    d_dy = (_proj(c + ey) - _proj(c - ey)) / (2.0 * eps_xy)
    d_dh = (_proj(c + eh) - _proj(c - eh)) / (2.0 * eps_h)
    return torch.stack([d_dx, d_dy, d_dh], dim=-1)


def project_cov3d_to_cov2d(sigma_3d: torch.Tensor, jacobian: torch.Tensor, eps_cov: float) -> torch.Tensor:
    s2 = jacobian @ sigma_3d @ jacobian.transpose(-1, -2)
    eye = torch.eye(2, device=s2.device, dtype=s2.dtype).view(1, 2, 2)
    return s2 + eps_cov * eye


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
class VirtualPinholeCamera:
    K: torch.Tensor  # [3,3]
    w2c: torch.Tensor  # [4,4]
    fit_p50: float
    fit_p95: float
    fit_max: float
    diagnostics: dict[str, Any] | None = None


@dataclass
class RPCGaussianRendererCfg:
    train_num_target_views: int = 1
    val_num_target_views: int = 1
    use_all_targets_in_val: bool = False
    exclude_self_source: bool = True
    allow_self_source_if_single_view: bool = True
    source_stride: int = 4
    confidence_threshold: float = 0.1
    topk_per_target: int | None = 4000
    enable_voxelization: bool = False
    voxel_xy: float = 2.0
    voxel_z: float = 1.0
    render_downsample_factor_train: int = 4
    render_downsample_factor_val: int = 2
    deterministic_target_selection: bool = True
    target_selection_mode: str = "random"
    target_selection_seed_offset: int = 0

    # RPC->pinhole 拟合配置
    fit_grid_nx: int = 24
    fit_grid_ny: int = 24
    fit_grid_nz: int = 7
    fit_height_margin: float = 30.0
    fit_max_reproj_p95_px: float = 1.0
    fit_cache_enable: bool = True
    fit_z_eps: float = 1.0e-6
    fit_min_positive_depth_ratio: float = 0.7
    fit_constrained_enable: bool = True
    fit_constrained_iters: int = 250
    fit_constrained_lr: float = 5.0e-2
    fit_constrained_huber_delta: float = 1.0
    fit_constrained_lambda_fxy: float = 1.0e-2
    fit_constrained_lambda_depth: float = 5.0e-2
    fit_constrained_lambda_center: float = 1.0e-2
    fit_constrained_min_depth: float = 1.0e-2
    # 像方采样式 RPC->3D-2D 对应构建
    fit_pixel_inner_nx: int = 32
    fit_pixel_inner_ny: int = 32
    fit_pixel_edge_per_side: int = 64
    fit_pixel_center_dense_n: int = 128
    fit_pixel_center_dense_radius_ratio: float = 0.15
    fit_pixel_jitter_px: float = 0.25
    fit_pixel_random_state: int = 42
    fit_height_quantile_low: float = 0.01
    fit_height_quantile_high: float = 0.99
    fit_train_ratio: float = 0.8
    fit_random_state: int = 12345
    fit_enable_validation: bool = True
    fit_robust_loss: str = "huber"  # huber | soft_l1
    fit_robust_f_scale: float = 1.0
    fit_stage_a_max_nfev: int = 200
    fit_stage_b_max_nfev: int = 220
    fit_stage_c_max_nfev: int = 250
    fit_gamma_x: float = 0.5
    fit_gamma_y: float = 0.5
    fit_lambda_pp_center: float = 2.0
    fit_lambda_focal_ratio: float = 1.0
    fit_lambda_depth_barrier: float = 1.0
    fit_depth_barrier_k: float = 20.0
    fit_lambda_center_prior: float = 1.0e-3
    fit_max_focal_ratio: float = 1.5
    fit_rotation_orth_tol: float = 1.0e-4
    fit_det_tol: float = 1.0e-4


class RPCGaussianRenderer:
    def __init__(self, geometry_ops: Any, cfg: RPCGaussianRendererCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or RPCGaussianRendererCfg()
        self._virtual_cam_cache: dict[tuple[int, int, int, int], VirtualPinholeCamera] = {}

    def _make_target_selection_generator(self, *, batch: dict[str, Any], bi: int, global_step: int | None, epoch: int | None) -> torch.Generator:
        seed_offset = int(self.cfg.target_selection_seed_offset)
        gs = 0 if global_step is None else int(global_step)
        ep = 0 if epoch is None else int(epoch)
        sid = int(batch["scene_id"][bi].detach().cpu().item()) if torch.is_tensor(batch.get("scene_id", None)) else 0
        seed = (sid * 1000003 + bi * 9176 + gs * 1315423911 + ep * 2654435761 + seed_offset) & 0xFFFFFFFF
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return g

    def select_target_views(self, b: int, v: int, mode: str, ref_view_idx: torch.Tensor | None, *, batch: dict[str, Any], global_step: int | None, epoch: int | None) -> list[list[int]]:
        ref = torch.zeros((b,), dtype=torch.long) if ref_view_idx is None else ref_view_idx.long().view(-1)
        if ref.numel() == 1:
            ref = ref.expand(b)
        out: list[list[int]] = []
        for bi in range(b):
            non_ref = [i for i in range(v) if i != int(ref[bi])]
            if mode == "train":
                if not non_ref:
                    out.append([int(ref[bi])] if self.cfg.allow_self_source_if_single_view else [])
                    continue
                k = min(int(self.cfg.train_num_target_views), len(non_ref))
                if self.cfg.target_selection_mode == "random":
                    if self.cfg.deterministic_target_selection:
                        g = self._make_target_selection_generator(batch=batch, bi=bi, global_step=global_step, epoch=epoch)
                        ids = torch.randperm(len(non_ref), generator=g)[:k].tolist()
                    else:
                        ids = torch.randperm(len(non_ref))[:k].tolist()
                    out.append([non_ref[i] for i in ids])
                else:
                    out.append(non_ref[:k])
            else:
                if self.cfg.use_all_targets_in_val:
                    cand = non_ref
                else:
                    cand = non_ref[: int(self.cfg.val_num_target_views)]
                out.append(cand if cand else [int(ref[bi])])
        return out

    def _sample_pixels_for_view(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """采样像素点（四角/边缘/中心/内部），用于RPC反投影构建3D-2D对应。"""
        pts: list[torch.Tensor] = []
        # 锚点：中心 + 四角 + 四边中点
        anchor = torch.tensor(
            [
                [0.5 * (w - 1), 0.5 * (h - 1)],
                [0.0, 0.0],
                [float(w - 1), 0.0],
                [0.0, float(h - 1)],
                [float(w - 1), float(h - 1)],
                [0.5 * (w - 1), 0.0],
                [0.5 * (w - 1), float(h - 1)],
                [0.0, 0.5 * (h - 1)],
                [float(w - 1), 0.5 * (h - 1)],
            ],
            dtype=torch.float64,
            device=device,
        )
        pts.append(anchor)
        edge_n = max(int(self.cfg.fit_pixel_edge_per_side), 2)
        u_edge = torch.linspace(0.0, float(w - 1), edge_n, dtype=torch.float64, device=device)
        v_edge = torch.linspace(0.0, float(h - 1), edge_n, dtype=torch.float64, device=device)
        top = torch.stack([u_edge, torch.zeros_like(u_edge)], dim=-1)
        bottom = torch.stack([u_edge, torch.full_like(u_edge, float(h - 1))], dim=-1)
        left = torch.stack([torch.zeros_like(v_edge), v_edge], dim=-1)
        right = torch.stack([torch.full_like(v_edge, float(w - 1)), v_edge], dim=-1)
        pts.extend([top, bottom, left, right])
        # 内部均匀网格
        nx = max(int(self.cfg.fit_pixel_inner_nx), 2)
        ny = max(int(self.cfg.fit_pixel_inner_ny), 2)
        uu = torch.linspace(0.0, float(w - 1), nx, dtype=torch.float64, device=device)
        vv = torch.linspace(0.0, float(h - 1), ny, dtype=torch.float64, device=device)
        gv, gu = torch.meshgrid(vv, uu, indexing="ij")
        pts.append(torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1))
        # 中心加密
        n_center = max(int(self.cfg.fit_pixel_center_dense_n), 0)
        if n_center > 0:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(self.cfg.fit_pixel_random_state))
            radius_u = float(self.cfg.fit_pixel_center_dense_radius_ratio) * float(w)
            radius_v = float(self.cfg.fit_pixel_center_dense_radius_ratio) * float(h)
            rand = torch.rand((n_center, 2), generator=g, dtype=torch.float64)
            rand = 2.0 * rand - 1.0
            center = torch.tensor([0.5 * (w - 1), 0.5 * (h - 1)], dtype=torch.float64)
            dense = torch.stack([center[0] + rand[:, 0] * radius_u, center[1] + rand[:, 1] * radius_v], dim=-1).to(device=device)
            pts.append(dense)
        uv = torch.cat(pts, dim=0)
        # 抖动（锚点保留，其他抖动）
        jitter = float(self.cfg.fit_pixel_jitter_px)
        if jitter > 0 and uv.shape[0] > anchor.shape[0]:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(self.cfg.fit_pixel_random_state) + 17)
            noise = (torch.rand((uv.shape[0] - anchor.shape[0], 2), generator=g, dtype=torch.float64) * 2.0 - 1.0) * jitter
            uv[anchor.shape[0] :] += noise.to(device=device)
        uv[:, 0] = uv[:, 0].clamp(0.0, float(w - 1))
        uv[:, 1] = uv[:, 1].clamp(0.0, float(h - 1))
        # 去重
        uv_q = (torch.round(uv * 1000.0) / 1000.0).detach().cpu().numpy()
        _, idx = np.unique(uv_q, axis=0, return_index=True)
        idx_t = torch.from_numpy(np.sort(idx)).to(device=device, dtype=torch.long)
        return uv[idx_t]

    def _make_height_levels(self, batch: dict[str, Any], bi: int, tv: int, device: torch.device) -> torch.Tensor:
        h_ref = float(batch["height_ref"][bi, tv].item()) if "height_ref" in batch else 0.0
        h_min = h_ref - float(self.cfg.fit_height_margin)
        h_max = h_ref + float(self.cfg.fit_height_margin)
        if "height_gt" in batch:
            hv = batch["height_gt"][bi, tv, 0].to(dtype=torch.float64, device=device)
            mv = batch.get("height_valid_mask", None)
            if mv is not None:
                m = mv[bi, tv, 0].to(device=device) > 0.5
                hv = hv[m] if bool(m.any()) else hv
            if hv.numel() > 0:
                ql = float(self.cfg.fit_height_quantile_low)
                qh = float(self.cfg.fit_height_quantile_high)
                h_min = float(torch.quantile(hv, ql).item())
                h_max = float(torch.quantile(hv, qh).item())
        if not np.isfinite(h_min) or not np.isfinite(h_max) or h_max <= h_min:
            h_min, h_max = h_ref - float(self.cfg.fit_height_margin), h_ref + float(self.cfg.fit_height_margin)
        return torch.linspace(h_min, h_max, max(int(self.cfg.fit_grid_nz), 3), dtype=torch.float64, device=device)

    def _build_correspondence_from_view_pixels(
        self, batch: dict[str, Any], bi: int, tv: int, image_hw: tuple[int, int], fit_dev: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = image_hw
        rpc_t = batch["rpc_gt"][bi][tv]
        scene_center = batch["scene_xy_center"][bi].to(dtype=torch.float64, device=fit_dev)
        scene_scale = torch.ones_like(scene_center)
        uv = self._sample_pixels_for_view(h=h, w=w, device=fit_dev)
        heights = self._make_height_levels(batch=batch, bi=bi, tv=tv, device=fit_dev)
        nu = uv.shape[0]
        nh = heights.shape[0]
        u_rep = uv[:, 0].repeat_interleave(nh)
        v_rep = uv[:, 1].repeat_interleave(nh)
        h_rep = heights.repeat(nu)
        x, y = rpc_t.RPC_LINESAMP2XY(
            line_in=v_rep,
            samp_in=u_rep,
            h_in=h_rep,
            output_type="tensor",
            xy_center=scene_center,
            xy_scale=scene_scale,
        )
        xyz = torch.stack([x, y, h_rep], dim=-1).to(dtype=torch.float64, device=fit_dev)
        uv_rep = torch.stack([u_rep, v_rep], dim=-1).to(dtype=torch.float64, device=fit_dev)
        finite = torch.isfinite(xyz).all(dim=-1) & torch.isfinite(uv_rep).all(dim=-1)
        if not bool(finite.any()):
            raise RuntimeError("No finite 3D-2D correspondences from RPC_LINESAMP2XY")
        return xyz[finite], uv_rep[finite]

    def _precheck_correspondence_np(self, xyz: np.ndarray, uv: np.ndarray) -> list[str]:
        warn: list[str] = []
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N,3], got {xyz.shape}")
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"uv must be [N,2], got {uv.shape}")
        if xyz.shape[0] != uv.shape[0]:
            raise ValueError("xyz and uv size mismatch")
        if xyz.shape[0] < 20:
            raise ValueError(f"Need at least 20 correspondences, got {xyz.shape[0]}")
        if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
            raise ValueError("Non-finite values detected in correspondence")
        centered = xyz - xyz.mean(axis=0, keepdims=True)
        rank = int(np.linalg.matrix_rank(centered))
        if rank < 3:
            raise ValueError(f"3D points are degenerate (rank={rank})")
        cov = np.cov(centered.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eig[0] <= 1e-12:
            raise ValueError("3D spread is too small")
        if eig[2] / eig[0] < 1e-4:
            warn.append("3D points are near-coplanar")
        return warn

    def _split_train_val_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if not bool(self.cfg.fit_enable_validation):
            idx = np.arange(n, dtype=np.int64)
            return idx, np.empty((0,), dtype=np.int64)
        ratio = float(np.clip(self.cfg.fit_train_ratio, 0.5, 0.95))
        rng = np.random.default_rng(int(self.cfg.fit_random_state))
        perm = rng.permutation(n)
        n_train = int(max(16, min(n - 4, round(n * ratio))))
        return perm[:n_train], perm[n_train:]

    def _dlt_init_np(self, xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
        xyz_n, U = _normalize_points_3d_np(xyz)
        uv_n, T = _normalize_points_2d_np(uv)
        n = xyz.shape[0]
        X = np.concatenate([xyz_n, np.ones((n, 1), dtype=np.float64)], axis=1)
        u = uv_n[:, 0:1]
        v = uv_n[:, 1:2]
        O = np.zeros_like(X)
        A1 = np.concatenate([X, O, -u * X], axis=1)
        A2 = np.concatenate([O, X, -v * X], axis=1)
        A = np.concatenate([A1, A2], axis=0)
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        p = vh[-1]
        Pn = p.reshape(3, 4)
        P = np.linalg.inv(T) @ Pn @ U
        return P / max(np.linalg.norm(P), 1e-12)

    def _decompose_projection_np(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = P[:, :3]
        K, R = scipy.linalg.rq(M)
        diag = np.diag(K).copy()
        sign = np.where(diag >= 0, 1.0, -1.0)
        D = np.diag(sign)
        K = K @ D
        R = D @ R
        if np.linalg.det(R) < 0:
            K[:, 2] *= -1.0
            R[2, :] *= -1.0
        K = K / max(K[2, 2], 1e-12)
        _, _, vh = np.linalg.svd(P)
        C_h = vh[-1]
        C = C_h[:3] / max(C_h[3], 1e-12)
        return K, R, C

    @staticmethod
    def _project_np(xyz: np.ndarray, R: np.ndarray, C: np.ndarray, fx: float, fy: float, cx: float, cy: float, z_eps: float) -> tuple[np.ndarray, np.ndarray]:
        t = -R @ C
        cam = (R @ xyz.T).T + t[None, :]
        z = cam[:, 2]
        z_safe = np.maximum(z, z_eps)
        u = fx * (cam[:, 0] / z_safe) + cx
        v = fy * (cam[:, 1] / z_safe) + cy
        return np.stack([u, v], axis=1), z

    @staticmethod
    def _reproj_metrics_np(err_xy: np.ndarray) -> tuple[float, float, float, float]:
        if err_xy.size == 0:
            return float("inf"), float("inf"), float("inf"), float("inf")
        e = np.linalg.norm(err_xy, axis=1)
        rmse = float(np.sqrt(np.mean(e**2)))
        p50 = float(np.quantile(e, 0.5))
        p95 = float(np.quantile(e, 0.95))
        pmax = float(np.max(e))
        return rmse, p50, p95, pmax

    def _health_check_np(
        self,
        *,
        K: np.ndarray,
        R: np.ndarray,
        z_train: np.ndarray,
        z_val: np.ndarray | None,
        train_p95: float,
        val_p95: float | None,
        image_hw: tuple[int, int],
    ) -> tuple[bool, list[str]]:
        h, w = image_hw
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        reasons: list[str] = []
        if fx <= 0 or fy <= 0:
            reasons.append("fx/fy must be positive")
        rot_err = np.linalg.norm(R.T @ R - np.eye(3))
        if rot_err > float(self.cfg.fit_rotation_orth_tol):
            reasons.append(f"R is not orthonormal (err={rot_err:.3e})")
        det_r = float(np.linalg.det(R))
        if abs(det_r - 1.0) > float(self.cfg.fit_det_tol):
            reasons.append(f"det(R) not close to +1 (det={det_r:.6f})")
        gamma_x = float(self.cfg.fit_gamma_x) * float(w)
        gamma_y = float(self.cfg.fit_gamma_y) * float(h)
        if abs(cx - 0.5 * w) > 1.05 * gamma_x or abs(cy - 0.5 * h) > 1.05 * gamma_y:
            reasons.append("principal point out of bounded range")
        ratio = max(fx, fy) / max(min(fx, fy), 1e-12)
        if ratio > float(self.cfg.fit_max_focal_ratio):
            reasons.append(f"fx/fy too imbalanced (ratio={ratio:.3f})")
        pos_train = float(np.mean(z_train > float(self.cfg.fit_z_eps)))
        if pos_train < float(self.cfg.fit_min_positive_depth_ratio):
            reasons.append(f"train positive depth ratio too low ({pos_train:.4f})")
        if z_val is not None and z_val.size > 0:
            pos_val = float(np.mean(z_val > float(self.cfg.fit_z_eps)))
            if pos_val < float(self.cfg.fit_min_positive_depth_ratio):
                reasons.append(f"val positive depth ratio too low ({pos_val:.4f})")
        if train_p95 > float(self.cfg.fit_max_reproj_p95_px):
            reasons.append(f"train p95 too high ({train_p95:.4f})")
        if val_p95 is not None and val_p95 > float(self.cfg.fit_max_reproj_p95_px):
            reasons.append(f"val p95 too high ({val_p95:.4f})")
        return (len(reasons) == 0), reasons

    def _evaluate_camera_fit(
        self,
        *,
        xyz: torch.Tensor,
        uv: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        t_t: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        xyz_h = torch.cat([xyz, torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)], dim=-1)
        w2c = torch.eye(4, dtype=xyz.dtype, device=xyz.device)
        w2c[:3, :3] = R
        w2c[:3, 3] = t_t
        cam = (w2c[:3, :] @ xyz_h.T).T
        proj = (K @ cam.T).T
        z = proj[:, 2]
        valid = z > float(self.cfg.fit_z_eps)
        if not bool(valid.any()):
            return 0.0, float("inf"), float("inf"), float("inf")
        uv_fit = proj[valid, :2] / proj[valid, 2:3]
        err = torch.linalg.norm(uv_fit - uv[valid], dim=-1)
        return (
            float(valid.float().mean().item()),
            float(torch.quantile(err, 0.5).item()),
            float(torch.quantile(err, 0.95).item()),
            float(err.max().item()),
        )

    def _fit_virtual_camera(self, batch: dict[str, Any], bi: int, tv: int, image_hw: tuple[int, int]) -> VirtualPinholeCamera:
        h, w = image_hw
        cache_key = (int(batch["scene_id"][bi].item()) if torch.is_tensor(batch.get("scene_id", None)) else bi, tv, h, w)
        if self.cfg.fit_cache_enable and cache_key in self._virtual_cam_cache:
            return self._virtual_cam_cache[cache_key]

        fit_dev = batch["rpc_gt"][bi][tv].device
        xyz_t, uv_t = self._build_correspondence_from_view_pixels(batch=batch, bi=bi, tv=tv, image_hw=image_hw, fit_dev=fit_dev)
        xyz_np = xyz_t.detach().cpu().numpy().astype(np.float64)
        uv_np = uv_t.detach().cpu().numpy().astype(np.float64)
        pre_warn = self._precheck_correspondence_np(xyz_np, uv_np)
        train_idx, val_idx = self._split_train_val_indices(xyz_np.shape[0])
        xyz_train, uv_train = xyz_np[train_idx], uv_np[train_idx]
        xyz_val = xyz_np[val_idx] if val_idx.size > 0 else np.empty((0, 3), dtype=np.float64)
        uv_val = uv_np[val_idx] if val_idx.size > 0 else np.empty((0, 2), dtype=np.float64)

        P0 = self._dlt_init_np(xyz_train, uv_train)
        K0, R0, C0 = self._decompose_projection_np(P0)
        # 朝向修正：优先让训练点正深度比例更高
        _, z_pos = self._project_np(
            xyz_train,
            R=R0,
            C=C0,
            fx=max(abs(float(K0[0, 0])), 1.0),
            fy=max(abs(float(K0[1, 1])), 1.0),
            cx=float(K0[0, 2]),
            cy=float(K0[1, 2]),
            z_eps=float(self.cfg.fit_z_eps),
        )
        _, z_neg = self._project_np(
            xyz_train,
            R=-R0,
            C=C0,
            fx=max(abs(float(K0[0, 0])), 1.0),
            fy=max(abs(float(K0[1, 1])), 1.0),
            cx=float(K0[0, 2]),
            cy=float(K0[1, 2]),
            z_eps=float(self.cfg.fit_z_eps),
        )
        if np.mean(z_neg > float(self.cfg.fit_z_eps)) > np.mean(z_pos > float(self.cfg.fit_z_eps)):
            R0 = -R0

        rot0 = scipy.spatial.transform.Rotation.from_matrix(R0)
        rvec0 = rot0.as_rotvec()
        f0 = max(1.0, float(0.5 * (abs(K0[0, 0]) + abs(K0[1, 1]))))
        cx0 = float(K0[0, 2])
        cy0 = float(K0[1, 2])
        gamma_x = float(self.cfg.fit_gamma_x)
        gamma_y = float(self.cfg.fit_gamma_y)

        scene_scale = max(float(np.std(xyz_train[:, 0])), float(np.std(xyz_train[:, 1])), float(np.std(xyz_train[:, 2])), 1.0)

        def _solve_stage(stage: str, x0: np.ndarray, max_nfev: int) -> tuple[scipy.optimize.OptimizeResult, dict[str, float]]:
            def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
                if stage == "A":
                    rvec = params[:3]
                    C = params[3:6]
                    f = float(np.exp(params[6]))
                    cx, cy = 0.5 * w, 0.5 * h
                    return rvec, C, f, f, cx, cy
                if stage == "B":
                    rvec = params[:3]
                    C = params[3:6]
                    f = float(np.exp(params[6]))
                    bx, by = float(params[7]), float(params[8])
                    cx = 0.5 * w + gamma_x * w * np.tanh(bx)
                    cy = 0.5 * h + gamma_y * h * np.tanh(by)
                    return rvec, C, f, f, cx, cy
                rvec = params[:3]
                C = params[3:6]
                ax, ay = float(params[6]), float(params[7])
                bx, by = float(params[8]), float(params[9])
                fx, fy = float(np.exp(ax)), float(np.exp(ay))
                cx = 0.5 * w + gamma_x * w * np.tanh(bx)
                cy = 0.5 * h + gamma_y * h * np.tanh(by)
                return rvec, C, fx, fy, cx, cy

            def residual(params: np.ndarray) -> np.ndarray:
                rvec, C, fx, fy, cx, cy = unpack(params)
                R = scipy.spatial.transform.Rotation.from_rotvec(rvec).as_matrix()
                pred, z = self._project_np(
                    xyz_train,
                    R=R,
                    C=C,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    z_eps=float(self.cfg.fit_z_eps),
                )
                data = (pred - uv_train).reshape(-1)
                residuals: list[np.ndarray] = [data]
                lam_pp = float(self.cfg.fit_lambda_pp_center)
                if stage in ("B", "C"):
                    residuals.append(np.array([lam_pp * (cx - 0.5 * w) / max(float(w), 1.0), lam_pp * (cy - 0.5 * h) / max(float(h), 1.0)], dtype=np.float64))
                lam_fd = float(self.cfg.fit_lambda_depth_barrier)
                k = float(self.cfg.fit_depth_barrier_k)
                eps_d = float(self.cfg.fit_constrained_min_depth)
                depth_soft = np.log1p(np.exp(k * (eps_d - z))) / max(k, 1e-6)
                residuals.append(np.sqrt(lam_fd) * np.sqrt(np.maximum(depth_soft, 0.0)))
                lam_cp = float(self.cfg.fit_lambda_center_prior)
                residuals.append(np.sqrt(lam_cp) * ((C - C0) / scene_scale))
                if stage == "C":
                    lam_fr = float(self.cfg.fit_lambda_focal_ratio)
                    residuals.append(np.array([np.sqrt(lam_fr) * np.log(max(fx, 1e-12) / max(fy, 1e-12))], dtype=np.float64))
                return np.concatenate(residuals, axis=0)

            result = scipy.optimize.least_squares(
                residual,
                x0,
                method="trf",
                loss=str(self.cfg.fit_robust_loss),
                f_scale=float(self.cfg.fit_robust_f_scale),
                max_nfev=max_nfev,
            )
            rvec, C, fx, fy, cx, cy = unpack(result.x)
            return result, {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "rvec0": rvec[0], "rvec1": rvec[1], "rvec2": rvec[2], "C0": C[0], "C1": C[1], "C2": C[2]}

        # 初始化阶段参数
        x0_a = np.concatenate([rvec0, C0, np.array([np.log(f0)], dtype=np.float64)], axis=0)
        res_a, params_a = _solve_stage("A", x0_a, max_nfev=int(self.cfg.fit_stage_a_max_nfev))
        cx_anchor = np.clip((cx0 - 0.5 * w) / max(gamma_x * w, 1e-6), -0.999, 0.999)
        cy_anchor = np.clip((cy0 - 0.5 * h) / max(gamma_y * h, 1e-6), -0.999, 0.999)
        x0_b = np.concatenate([res_a.x[:7], np.array([np.arctanh(cx_anchor), np.arctanh(cy_anchor)], dtype=np.float64)], axis=0)
        res_b, params_b = _solve_stage("B", x0_b, max_nfev=int(self.cfg.fit_stage_b_max_nfev))
        x0_c = np.concatenate([res_b.x[:6], np.array([res_b.x[6], res_b.x[6], res_b.x[7], res_b.x[8]], dtype=np.float64)], axis=0)
        res_c, params_c = _solve_stage("C", x0_c, max_nfev=int(self.cfg.fit_stage_c_max_nfev))

        stage_pack = [("A", res_a, params_a), ("B", res_b, params_b), ("C", res_c, params_c)]
        diagnostics_stages: list[dict[str, Any]] = []
        chosen: tuple[str, dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None = None

        for name, res, p in stage_pack:
            rvec = np.array([p["rvec0"], p["rvec1"], p["rvec2"]], dtype=np.float64)
            C = np.array([p["C0"], p["C1"], p["C2"]], dtype=np.float64)
            R = scipy.spatial.transform.Rotation.from_rotvec(rvec).as_matrix()
            pred_train, z_train = self._project_np(
                xyz_train,
                R=R,
                C=C,
                fx=float(p["fx"]),
                fy=float(p["fy"]),
                cx=float(p["cx"]),
                cy=float(p["cy"]),
                z_eps=float(self.cfg.fit_z_eps),
            )
            train_rmse, train_p50, train_p95, train_max = self._reproj_metrics_np(pred_train - uv_train)
            if xyz_val.shape[0] > 0:
                pred_val, z_val = self._project_np(
                    xyz_val,
                    R=R,
                    C=C,
                    fx=float(p["fx"]),
                    fy=float(p["fy"]),
                    cx=float(p["cx"]),
                    cy=float(p["cy"]),
                    z_eps=float(self.cfg.fit_z_eps),
                )
                val_rmse, val_p50, val_p95, val_max = self._reproj_metrics_np(pred_val - uv_val)
            else:
                z_val = np.empty((0,), dtype=np.float64)
                val_rmse = val_p50 = val_p95 = val_max = None
            K_np = np.array([[p["fx"], 0.0, p["cx"]], [0.0, p["fy"], p["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
            ok, reasons = self._health_check_np(
                K=K_np,
                R=R,
                z_train=z_train,
                z_val=z_val if z_val.size > 0 else None,
                train_p95=train_p95,
                val_p95=val_p95,
                image_hw=image_hw,
            )
            stage_diag = {
                "stage": name,
                "optimizer_success": bool(res.success),
                "optimizer_message": str(res.message),
                "nfev": int(res.nfev),
                "train_rmse": train_rmse,
                "train_p50": train_p50,
                "train_p95": train_p95,
                "train_max": train_max,
                "train_pos_ratio": float(np.mean(z_train > float(self.cfg.fit_z_eps))),
                "val_rmse": val_rmse,
                "val_p50": val_p50,
                "val_p95": val_p95,
                "val_max": val_max,
                "val_pos_ratio": None if z_val.size == 0 else float(np.mean(z_val > float(self.cfg.fit_z_eps))),
                "health_pass": ok,
                "health_reasons": reasons,
            }
            diagnostics_stages.append(stage_diag)
            target_p95 = val_p95 if val_p95 is not None else train_p95
            if ok and target_p95 <= float(self.cfg.fit_max_reproj_p95_px):
                chosen = (name, stage_diag, K_np, R, -R @ C)
                break
            if chosen is None:
                chosen = (name, stage_diag, K_np, R, -R @ C)

        assert chosen is not None
        chosen_name, chosen_diag, K_best, R_best, t_best = chosen
        if not bool(chosen_diag["health_pass"]) or (
            (chosen_diag["val_p95"] is not None and float(chosen_diag["val_p95"]) > float(self.cfg.fit_max_reproj_p95_px))
            or (chosen_diag["val_p95"] is None and float(chosen_diag["train_p95"]) > float(self.cfg.fit_max_reproj_p95_px))
        ):
            import warnings

            warnings.warn(
                "该数据不适合用单个健康针孔模型在当前误差阈值下拟合；已返回最优阶段结果。",
                RuntimeWarning,
            )

        K = torch.from_numpy(K_best).to(dtype=torch.float64, device=fit_dev)
        R = torch.from_numpy(R_best).to(dtype=torch.float64, device=fit_dev)
        t_t = torch.from_numpy(t_best).to(dtype=torch.float64, device=fit_dev)
        pos_ratio, p50, p95, pmax = self._evaluate_camera_fit(xyz=xyz_t, uv=uv_t, K=K, R=R, t_t=t_t)
        w2c = torch.eye(4, dtype=torch.float64, device=fit_dev)
        w2c[:3, :3] = R
        w2c[:3, 3] = t_t
        diagnostics = {
            "precheck_warnings": pre_warn,
            "selected_stage": chosen_name,
            "stages": diagnostics_stages,
            "num_correspondences_total": int(xyz_np.shape[0]),
            "num_train": int(train_idx.shape[0]),
            "num_val": int(val_idx.shape[0]),
        }
        cam_obj = VirtualPinholeCamera(
            K=K.to(torch.float32),
            w2c=w2c.to(torch.float32),
            fit_p50=p50,
            fit_p95=p95,
            fit_max=pmax,
            diagnostics=diagnostics,
        )
        if self.cfg.fit_cache_enable:
            self._virtual_cam_cache[cache_key] = cam_obj
        return cam_obj

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
            from gsplat import rasterization  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("未安装gsplat，无法进行高性能渲染。请先安装gsplat。") from e

        h, w = image_hw
        means = centers.to(torch.float32)
        scales = scale.to(torch.float32).clamp_min(1e-6)
        rots = normalize_quaternion(rotation.to(torch.float32))
        op = opacity[:, 0].to(torch.float32).clamp(0.0, 1.0)
        colors = rgb.to(torch.float32)
        viewmats = cam.w2c.to(torch.float32).contiguous().unsqueeze(0).to(means.device)
        Ks = cam.K.to(torch.float32).unsqueeze(0).to(means.device)
        bg = torch.zeros((1, 3), device=means.device, dtype=torch.float32)

        rendering, alpha, _ = rasterization(
            means,
            rots,
            scales,
            op,
            colors,
            viewmats,
            Ks,
            w,
            h,
            sh_degree=None,
            render_mode="RGB+D",
            packed=False,
            near_plane=1e-6,
            backgrounds=bg,
            rasterize_mode="classic",
        )
        rgb_out = rendering[0, ..., :3].permute(2, 0, 1).contiguous().clamp(0.0, 1.0)
        depth_out = rendering[0, ..., 3:4].permute(2, 0, 1).contiguous()
        alpha_out = alpha[0].permute(2, 0, 1).contiguous() if alpha.ndim == 4 else alpha[0].unsqueeze(0)
        return rgb_out, alpha_out, depth_out

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
        b, v, _, h, w = centers_path.shape
        rgb_all = sh_basecolor_to_rgb(gaussian_sh.permute(0, 1, 3, 4, 2).contiguous())
        packs: list[dict[str, Any]] = []
        for bi in range(b):
            for tv in target_view_indices[bi]:
                source_ids = list(range(v))
                if self.cfg.exclude_self_source and (v > 1 or not self.cfg.allow_self_source_if_single_view):
                    source_ids = [sid for sid in source_ids if sid != tv]
                if len(source_ids) == 0:
                    c = centers_path.new_zeros((0, 3))
                    o = centers_path.new_zeros((0, 1))
                    s = centers_path.new_zeros((0, 3))
                    r = centers_path.new_zeros((0, 4))
                    col = centers_path.new_zeros((0, 3))
                    conf = centers_path.new_zeros((0, 1))
                else:
                    c = centers_path[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 3)
                    o = gaussian_opacity[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 1)
                    s = gaussian_scale[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 3)
                    r = gaussian_rotation[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 4)
                    col = rgb_all[bi, source_ids].reshape(-1, 3)
                    conf = path_confidence[bi, source_ids].permute(0, 2, 3, 1).reshape(-1, 1)

                    raw_n = c.shape[0]
                    idx = torch.arange(0, raw_n, max(1, int(self.cfg.source_stride)), device=c.device)
                    c, o, s, r, col, conf = c[idx], o[idx], s[idx], r[idx], col[idx], conf[idx]
                    keep = conf[:, 0] >= float(self.cfg.confidence_threshold)
                    c, o, s, r, col, conf = c[keep], o[keep], s[keep], r[keep], col[keep], conf[keep]
                    if self.cfg.topk_per_target is not None and c.shape[0] > int(self.cfg.topk_per_target):
                        ids = torch.topk(conf[:, 0], k=int(self.cfg.topk_per_target), largest=True).indices
                        c, o, s, r, col, conf = c[ids], o[ids], s[ids], r[ids], col[ids], conf[ids]

                packs.append(
                    {
                        "centers_world": c,
                        "opacity": o,
                        "scale": s,
                        "rotation": r,
                        "rgb": col,
                        "confidence": conf,
                        "batch_index": bi,
                        "target_view_index": tv,
                        "raw_num_gaussians": int(c.shape[0]),
                    }
                )
        return packs

    def fit_virtual_camera_for_target(self, batch: dict[str, Any], batch_index: int, target_view_index: int, image_hw: tuple[int, int] | None = None) -> VirtualPinholeCamera:
        if image_hw is None:
            rgb = batch["images"][batch_index, target_view_index]
            image_hw = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        return self._fit_virtual_camera(batch, batch_index, target_view_index, image_hw)

    def render_single_path_single_target(self, pack: dict[str, Any], batch: dict[str, Any], mode: str) -> dict[str, Any]:
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
            rh = None
            return {
                "rendered_rgb": rr,
                "rendered_alpha": ra,
                "rendered_height": rh,
                "target_rgb": F.interpolate(target_rgb.unsqueeze(0), size=(h_out, w_out), mode="bilinear", align_corners=False).squeeze(0),
                "target_height": None if target_h is None else F.interpolate(target_h.unsqueeze(0), size=(h_out, w_out), mode="bilinear", align_corners=False).squeeze(0),
                "target_valid_mask": None if target_m is None else F.interpolate(target_m.unsqueeze(0), size=(h_out, w_out), mode="nearest").squeeze(0),
                "target_view_index": tv,
                "batch_index": bi,
                "num_gaussians_raw": 0,
                "num_gaussians_after_conf": 0,
                "num_gaussians_after_voxel": 0,
                "num_gaussians_visible": 0,
                "alpha_coverage": torch.tensor(0.0, device=target_rgb.device),
                "fit_p95": 0.0,
            }

        cam = self._fit_virtual_camera(batch, bi, tv, (h, w))
        if cam.fit_p95 > float(self.cfg.fit_max_reproj_p95_px):
            raise RuntimeError(
                f"Virtual camera fit too inaccurate for scene={int(batch['scene_id'][bi].item()) if 'scene_id' in batch else bi}, "
                f"view={tv}: p95={cam.fit_p95:.4f}px > threshold={self.cfg.fit_max_reproj_p95_px:.4f}px"
            )

        # 修复: 虚拟相机在 full-res 拟合，渲染分辨率为 downsample 后时必须同步缩放内参。
        sx = float(w_out) / float(max(w, 1))
        sy = float(h_out) / float(max(h, 1))
        K_render = cam.K.clone()
        K_render[0, 0] = K_render[0, 0] * sx
        K_render[1, 1] = K_render[1, 1] * sy
        K_render[0, 2] = K_render[0, 2] * sx
        K_render[1, 2] = K_render[1, 2] * sy
        cam_render = VirtualPinholeCamera(
            K=K_render,
            w2c=cam.w2c,
            fit_p50=cam.fit_p50,
            fit_p95=cam.fit_p95,
            fit_max=cam.fit_max,
        )

        rr, ra, _depth_cam = self._render_cuda(
            centers=c,
            opacity=(pack["opacity"] * pack["confidence"]).clamp(0.0, 1.0),
            scale=pack["scale"],
            rotation=pack["rotation"],
            rgb=pack["rgb"],
            cam=cam_render,
            image_hw=(h_out, w_out),
        )
        # 修复: RGB+D 的 D 是相机深度，不等价于世界高程。
        # 为避免错误监督，当前不返回 rendered_height，RenderPathLoss 将自动跳过 height 项。
        rh = None

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
            "num_gaussians_raw": int(pack["raw_num_gaussians"]),
            "num_gaussians_after_conf": int(pack["raw_num_gaussians"]),
            "num_gaussians_after_voxel": int(pack["raw_num_gaussians"]),
            "num_gaussians_visible": int(pack["raw_num_gaussians"]),
            "alpha_coverage": (ra > 0.05).float().mean().detach(),
            "fit_p95": float(cam.fit_p95),
        }

    def render_single_path(self, centers_path: torch.Tensor, path_confidence: torch.Tensor, outputs: dict[str, Any], batch: dict[str, Any], mode: str, *, global_step: int | None = None, epoch: int | None = None) -> dict[str, Any]:
        b, v = centers_path.shape[:2]
        target_views = self.select_target_views(
            b,
            v,
            mode,
            batch.get("ref_view_idx", None),
            batch=batch,
            global_step=global_step,
            epoch=epoch,
        )
        packs = self.collect_source_gaussians_for_target(
            centers_path,
            outputs["gaussian_opacity"],
            outputs["gaussian_scale"],
            outputs["gaussian_rotation"],
            outputs["gaussian_sh"],
            path_confidence,
            outputs,
            batch,
            target_views,
        )
        single_results = [self.render_single_path_single_target(p, batch, mode) for p in packs]
        if not single_results:
            dev = centers_path.device
            return {
                "rendered_rgb": torch.zeros((0, 3, 0, 0), device=dev),
                "rendered_alpha": torch.zeros((0, 1, 0, 0), device=dev),
                "rendered_height": None,
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
                    "virtual_cam_fit_p95_mean": 0.0,
                },
            }

        rendered_rgb = torch.stack([x["rendered_rgb"] for x in single_results], dim=0)
        rendered_alpha = torch.stack([x["rendered_alpha"] for x in single_results], dim=0)
        any_render_height = all(x["rendered_height"] is not None for x in single_results)
        rendered_height = torch.stack([x["rendered_height"] for x in single_results], dim=0) if any_render_height else None
        target_rgb = torch.stack([x["target_rgb"] for x in single_results], dim=0)
        any_height = all(x["target_height"] is not None for x in single_results)
        any_mask = all(x["target_valid_mask"] is not None for x in single_results)
        target_height = torch.stack([x["target_height"] for x in single_results], dim=0) if any_height else None
        target_valid_mask = torch.stack([x["target_valid_mask"] for x in single_results], dim=0) if any_mask else None
        batch_indices = torch.tensor([x["batch_index"] for x in single_results], device=rendered_rgb.device, dtype=torch.long)
        target_view_indices = torch.tensor([x["target_view_index"] for x in single_results], device=rendered_rgb.device, dtype=torch.long)

        stats = {
            "num_gaussians_raw_mean": float(sum(x["num_gaussians_raw"] for x in single_results) / len(single_results)),
            "num_gaussians_after_conf_mean": float(sum(x["num_gaussians_after_conf"] for x in single_results) / len(single_results)),
            "num_gaussians_after_voxel_mean": float(sum(x["num_gaussians_after_voxel"] for x in single_results) / len(single_results)),
            "num_gaussians_visible_mean": float(sum(x["num_gaussians_visible"] for x in single_results) / len(single_results)),
            "alpha_coverage_mean": float(torch.stack([x["alpha_coverage"] for x in single_results]).mean().item()),
            "virtual_cam_fit_p95_mean": float(sum(x.get("fit_p95", 0.0) for x in single_results) / len(single_results)),
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

    def render_paths(self, outputs: dict[str, Any], batch: dict[str, Any], mode: str = "train", global_step: int | None = None, epoch: int | None = None) -> dict[str, dict[str, Any]]:
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
            global_step=global_step,
            epoch=epoch,
        )
        point_path = self.render_single_path(
            centers_path=outputs["gaussian_centers_point"],
            path_confidence=outputs["gaussian_confidence_point"],
            outputs=outputs,
            batch=batch,
            mode=mode,
            global_step=global_step,
            epoch=epoch,
        )
        return {"rpc": rpc_path, "point": point_path}

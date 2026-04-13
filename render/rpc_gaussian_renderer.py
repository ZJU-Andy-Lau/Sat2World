"""RPC 高斯渲染器（仅渲染，不负责相机拟合）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def cov3d_from_scale_rotation(scale: torch.Tensor, quaternion: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = scale.clamp_min(eps)
    r = quaternion_to_matrix(quaternion)
    d = torch.diag_embed(s * s)
    return r @ d @ r.transpose(-1, -2)


def sh_basecolor_to_rgb(gaussian_sh: torch.Tensor) -> torch.Tensor:
    if gaussian_sh.shape[-1] < 3:
        raise ValueError("gaussian_sh last dim must be >=3")
    return torch.sigmoid(gaussian_sh[..., :3])


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
    fit_p50: float = 0.0
    fit_p95: float = 0.0
    fit_max: float = 0.0
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


class RPCGaussianRenderer:
    def __init__(self, geometry_ops: Any, cfg: RPCGaussianRendererCfg | None = None) -> None:
        self.geometry_ops = geometry_ops
        self.cfg = cfg or RPCGaussianRendererCfg()

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

    def _get_view_camera(self, batch: dict[str, Any], bi: int, tv: int) -> VirtualPinholeCamera:
        if "view_pinhole_K" not in batch or "view_pinhole_w2c" not in batch:
            raise KeyError(
                "Batch missing pre-fitted pinhole camera fields: 'view_pinhole_K'/'view_pinhole_w2c'. "
                "Please fit pinhole camera in dataset construction."
            )
        K = batch["view_pinhole_K"][bi, tv]
        w2c = batch["view_pinhole_w2c"][bi, tv]

        p50 = 0.0
        p95 = 0.0
        pmax = 0.0
        if "view_pinhole_fit_p50" in batch:
            p50 = float(batch["view_pinhole_fit_p50"][bi, tv].item())
        if "view_pinhole_fit_p95" in batch:
            p95 = float(batch["view_pinhole_fit_p95"][bi, tv].item())
        if "view_pinhole_fit_max" in batch:
            pmax = float(batch["view_pinhole_fit_max"][bi, tv].item())

        diagnostics = None
        if "view_pinhole_diagnostics" in batch:
            diagnostics = batch["view_pinhole_diagnostics"][bi][tv]

        return VirtualPinholeCamera(
            K=K.to(torch.float32),
            w2c=w2c.to(torch.float32),
            fit_p50=p50,
            fit_p95=p95,
            fit_max=pmax,
            diagnostics=diagnostics,
        )

    def fit_virtual_camera_for_target(self, batch: dict[str, Any], batch_index: int, target_view_index: int, image_hw: tuple[int, int] | None = None) -> VirtualPinholeCamera:
        del image_hw
        return self._get_view_camera(batch, int(batch_index), int(target_view_index))

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
        del outputs, batch
        b, v, _, _, _ = centers_path.shape
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

        cam = self._get_view_camera(batch, bi, tv)

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
            diagnostics=cam.diagnostics,
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

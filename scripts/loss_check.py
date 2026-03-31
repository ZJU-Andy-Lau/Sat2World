"""scripts.loss_check

分阶段验证 loss 实现与解析真值是否一致（支持 DDP）。

四个阶段：
1) zero-noise
2) height/point-noise only
3) affine-noise only
4) all-noise
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.distributed import (
    all_reduce_mean,
    destroy_distributed,
    get_rank,
    init_distributed,
    is_main_process,
    move_batch_to_device,
    seed_everything,
)
from geometry import RPCGeometryOps
from geometry.scene_geometry import make_image_grid
from loss.affine_loss import (
    AffineGridLoss,
    AffineGridLossCfg,
    AffinePairwiseGeometryLoss,
    AffinePairwiseGeometryLossCfg,
)
from loss.common import apply_affine_to_points, masked_huber_loss, pairwise_view_pairs, sample_map_bilinear
from loss.height_loss import HeightHuberLoss
from loss.point_loss import PointMapLoss
from scripts.train import build_dataloaders, load_cfg


@dataclass
class NoiseCfg:
    affine_diag: float = 0.01
    affine_offdiag: float = 0.01
    affine_trans: float = 0.5
    height_std: float = 0.05
    point_std_xy: float = 0.02
    point_std_z: float = 0.05


def _eye_affine(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)


def _to_homo(aff: torch.Tensor) -> torch.Tensor:
    # aff: [...,2,3] -> [...,3,3]
    tail = torch.zeros((*aff.shape[:-2], 1, 3), device=aff.device, dtype=aff.dtype)
    tail[..., 0, 2] = 1.0
    return torch.cat([aff, tail], dim=-2)


def _compose_affine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """组合 2x3 仿射（先 first 后 second）。"""
    a = _to_homo(first)
    b = _to_homo(second)
    c = b @ a
    return c[..., :2, :]


def _invert_affine_bv(aff: torch.Tensor) -> torch.Tensor:
    # aff: [B,V,2,3]
    a = aff[..., :2]
    t = aff[..., 2:]
    a_inv = torch.linalg.inv(a)
    t_inv = -(a_inv @ t)
    return torch.cat([a_inv, t_inv], dim=-1)


def _make_affine_noise(affine_gt_correction: torch.Tensor, ref_view_idx: torch.Tensor, cfg: NoiseCfg) -> torch.Tensor:
    b, v = affine_gt_correction.shape[:2]
    device, dtype = affine_gt_correction.device, affine_gt_correction.dtype
    n = torch.zeros((b, v, 2, 3), device=device, dtype=dtype)
    eye = _eye_affine(device=device, dtype=dtype)
    n[:] = eye

    t = torch.tanh(torch.randn((b, v, 6), device=device, dtype=dtype))
    n[..., 0, 0] = 1.0 + cfg.affine_diag * t[..., 0]
    n[..., 0, 1] = cfg.affine_offdiag * t[..., 1]
    n[..., 0, 2] = cfg.affine_trans * t[..., 2]
    n[..., 1, 0] = cfg.affine_offdiag * t[..., 3]
    n[..., 1, 1] = 1.0 + cfg.affine_diag * t[..., 4]
    n[..., 1, 2] = cfg.affine_trans * t[..., 5]

    ref = ref_view_idx.long().view(-1)
    if ref.numel() == 1:
        ref = ref.expand(b)
    n[torch.arange(b, device=device), ref] = eye
    return n


def _build_gt_point_map(ops: RPCGeometryOps, batch_dev: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    _, _, _, h, w = batch_dev["height_gt"].shape
    grid = make_image_grid(h, w, device=device, dtype=dtype)
    return ops.centers_from_rpc_and_height_batch(
        corrected_rpc_batch=batch_dev["rpc_gt"],
        pixel_grid=grid,
        height_abs=batch_dev["height_gt"].to(device=device, dtype=dtype),
        scene_xy_center=batch_dev.get("scene_xy_center", None),
        scene_xy_scale=batch_dev.get("scene_xy_scale", None),
    )


def _affine_grid_truth(
    affine_noise: torch.Tensor,
    image_hw: tuple[int, int],
    ref_view_idx: torch.Tensor | None,
) -> torch.Tensor:
    b, v = affine_noise.shape[:2]
    h, w = image_hw
    yy = torch.linspace(0.0, float(max(h - 1, 0)), steps=16, device=affine_noise.device, dtype=affine_noise.dtype)
    xx = torch.linspace(0.0, float(max(w - 1, 0)), steps=16, device=affine_noise.device, dtype=affine_noise.dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
    n = grid.shape[0]
    g_true = grid.view(1, 1, n, 2).expand(b, v, n, 2)
    g_noise = apply_affine_to_points(g_true, affine_noise)
    err = torch.linalg.norm(g_noise - g_true, dim=-1)

    view_mask = torch.ones((b, v, 1), device=err.device, dtype=err.dtype)
    if ref_view_idx is not None:
        ref = ref_view_idx.long().view(-1)
        if ref.numel() == 1:
            ref = ref.expand(b)
        view_mask[torch.arange(b, device=err.device), ref, 0] = 0.0
    m = view_mask.expand_as(err)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def _affine_pair_truth(
    batch_dev: dict[str, Any],
    affine_noise: torch.Tensor,
    height_abs: torch.Tensor,
    max_pairs: int | None,
) -> torch.Tensor:
    rpc_gt = batch_dev["rpc_gt"]
    h_gt = batch_dev["height_gt"].to(device=height_abs.device, dtype=height_abs.dtype)
    h_mask = batch_dev["height_valid_mask"].to(device=height_abs.device, dtype=height_abs.dtype)
    scene_xy_center = batch_dev.get("scene_xy_center", None)
    scene_xy_scale = batch_dev.get("scene_xy_scale", None)
    ref_idx = batch_dev.get("ref_view_idx", None)

    b, v, _, h, w = height_abs.shape
    n_inv = _invert_affine_bv(affine_noise)
    pair_terms: list[torch.Tensor] = []

    for bi in range(b):
        ref_b = int(ref_idx[bi].item()) if ref_idx is not None else 0
        pairs = pairwise_view_pairs(v, max_pairs=max_pairs)
        for i, j in pairs:
            anchors_i = batch_dev["anchor_line_samp_true"][bi : bi + 1, i].to(device=height_abs.device, dtype=height_abs.dtype)
            anchors_j = batch_dev["anchor_line_samp_true"][bi : bi + 1, j].to(device=height_abs.device, dtype=height_abs.dtype)
            h_i_gt = batch_dev["anchor_height_true"][bi : bi + 1, i].to(device=height_abs.device, dtype=height_abs.dtype)
            h_j_gt = batch_dev["anchor_height_true"][bi : bi + 1, j].to(device=height_abs.device, dtype=height_abs.dtype)

            m_i, in_i = sample_map_bilinear(h_mask[bi : bi + 1, i], anchors_i)
            m_j, in_j = sample_map_bilinear(h_mask[bi : bi + 1, j], anchors_j)
            keep_i = in_i & (m_i[:, 0] > 0.5)
            keep_j = in_j & (m_j[:, 0] > 0.5)
            if keep_i.sum() == 0 or keep_j.sum() == 0:
                continue
            anchors_i = anchors_i[:, keep_i[0]]
            anchors_j = anchors_j[:, keep_j[0]]
            h_i_gt = h_i_gt[:, keep_i[0]]
            h_j_gt = h_j_gt[:, keep_j[0]]

            # GT cross projection
            xs_i, ys_i = RPCGeometryOps().linesamp_to_xy_batch(
                rpc_batch=[[rpc_gt[bi][i]]],
                lines=anchors_i[..., 0].view(1, 1, -1),
                samps=anchors_i[..., 1].view(1, 1, -1),
                heights=h_i_gt.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            xs_j, ys_j = RPCGeometryOps().linesamp_to_xy_batch(
                rpc_batch=[[rpc_gt[bi][j]]],
                lines=anchors_j[..., 0].view(1, 1, -1),
                samps=anchors_j[..., 1].view(1, 1, -1),
                heights=h_j_gt.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            l_i2j_gt, s_i2j_gt = RPCGeometryOps().xy_to_linesamp_batch(
                rpc_batch=[[rpc_gt[bi][j]]],
                xs=xs_i,
                ys=ys_i,
                heights=h_i_gt.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            l_j2i_gt, s_j2i_gt = RPCGeometryOps().xy_to_linesamp_batch(
                rpc_batch=[[rpc_gt[bi][i]]],
                xs=xs_j,
                ys=ys_j,
                heights=h_j_gt.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            gt_i2j = torch.stack([l_i2j_gt.view(1, -1), s_i2j_gt.view(1, -1)], dim=-1).to(height_abs.device, height_abs.dtype)
            gt_j2i = torch.stack([l_j2i_gt.view(1, -1), s_j2i_gt.view(1, -1)], dim=-1).to(height_abs.device, height_abs.dtype)

            # 按要求的“noise_i -> cross-project -> noise_j^-1”构造 pred truth
            ai_noisy = apply_affine_to_points(anchors_i, affine_noise[bi : bi + 1, i])
            aj_noisy = apply_affine_to_points(anchors_j, affine_noise[bi : bi + 1, j])

            li_n, si_n = RPCGeometryOps().xy_to_linesamp_batch(
                rpc_batch=[[rpc_gt[bi][j]]],
                xs=RPCGeometryOps().linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][i]]],
                    lines=ai_noisy[..., 0].view(1, 1, -1),
                    samps=ai_noisy[..., 1].view(1, 1, -1),
                    heights=h_i_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )[0],
                ys=RPCGeometryOps().linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][i]]],
                    lines=ai_noisy[..., 0].view(1, 1, -1),
                    samps=ai_noisy[..., 1].view(1, 1, -1),
                    heights=h_i_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )[1],
                heights=h_i_gt.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            lj_n, sj_n = RPCGeometryOps().xy_to_linesamp_batch(
                rpc_batch=[[rpc_gt[bi][i]]],
                xs=RPCGeometryOps().linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    lines=aj_noisy[..., 0].view(1, 1, -1),
                    samps=aj_noisy[..., 1].view(1, 1, -1),
                    heights=h_j_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )[0],
                ys=RPCGeometryOps().linesamp_to_xy_batch(
                    rpc_batch=[[rpc_gt[bi][j]]],
                    lines=aj_noisy[..., 0].view(1, 1, -1),
                    samps=aj_noisy[..., 1].view(1, 1, -1),
                    heights=h_j_gt.view(1, 1, -1),
                    scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                    scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
                )[1],
                heights=h_j_gt.view(1, 1, -1),
                scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
                scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
            )
            pred_i2j_noisy = torch.stack([li_n.view(1, -1), si_n.view(1, -1)], dim=-1).to(height_abs.device, height_abs.dtype)
            pred_j2i_noisy = torch.stack([lj_n.view(1, -1), sj_n.view(1, -1)], dim=-1).to(height_abs.device, height_abs.dtype)

            pred_i2j = apply_affine_to_points(pred_i2j_noisy, n_inv[bi : bi + 1, j])
            pred_j2i = apply_affine_to_points(pred_j2i_noisy, n_inv[bi : bi + 1, i])

            e_i2j = torch.linalg.norm(pred_i2j - gt_i2j, dim=-1)
            e_j2i = torch.linalg.norm(pred_j2i - gt_j2i, dim=-1)

            w_i2j = 0.0 if i == ref_b else 1.0
            w_j2i = 0.0 if j == ref_b else 1.0
            w_sum = w_i2j + w_j2i
            if w_sum <= 0.0:
                continue
            pair_terms.append((w_i2j * e_i2j.mean() + w_j2i * e_j2i.mean()) / w_sum)

    if len(pair_terms) == 0:
        return torch.zeros((), device=height_abs.device, dtype=height_abs.dtype)
    return torch.stack(pair_terms).mean()


def run_phase(
    phase_name: str,
    *,
    batch_dev: dict[str, Any],
    ops: RPCGeometryOps,
    losses: dict[str, Any],
    noise_cfg: NoiseCfg,
    add_affine_noise: bool,
    add_height_noise: bool,
    add_point_noise: bool,
) -> dict[str, torch.Tensor]:
    device = batch_dev["height_gt"].device
    dtype = batch_dev["height_gt"].dtype
    b, v, _, h, w = batch_dev["height_gt"].shape

    aff_corr_gt = batch_dev["affine_gt_correction"].to(device=device, dtype=dtype)
    ref = batch_dev.get("ref_view_idx", torch.zeros((b,), device=device, dtype=torch.long))
    aff_noise = _make_affine_noise(aff_corr_gt, ref, noise_cfg) if add_affine_noise else _eye_affine(device, dtype).view(1, 1, 2, 3).expand(b, v, 2, 3).clone()
    affine_pred = _compose_affine(aff_corr_gt, aff_noise)

    h_gt = batch_dev["height_gt"].to(device=device, dtype=dtype)
    h_noise = torch.randn_like(h_gt) * float(noise_cfg.height_std) if add_height_noise else torch.zeros_like(h_gt)
    height_abs = h_gt + h_noise

    gt_point_map = _build_gt_point_map(ops, batch_dev, device, dtype)
    point_noise = torch.zeros_like(gt_point_map)
    if add_point_noise:
        point_noise[:, :, 0:1] = torch.randn_like(point_noise[:, :, 0:1]) * float(noise_cfg.point_std_xy)
        point_noise[:, :, 1:2] = torch.randn_like(point_noise[:, :, 1:2]) * float(noise_cfg.point_std_xy)
        point_noise[:, :, 2:3] = torch.randn_like(point_noise[:, :, 2:3]) * float(noise_cfg.point_std_z)
    point_abs = gt_point_map + point_noise

    grid = make_image_grid(h, w, device=device, dtype=dtype)
    point_anchor = ops.build_point_anchor_map_batch(
        rpc_init_batch=batch_dev["rpc_init"],
        pixel_grid=grid,
        height_ref=batch_dev["height_ref"].to(device=device, dtype=dtype),
        scene_xy_center=batch_dev.get("scene_xy_center", None),
        scene_xy_scale=batch_dev.get("scene_xy_scale", None),
    )
    rpc_corrected = ops.apply_affine_correction_batch(batch_dev["rpc_init"], affine_pred)

    outputs = {"affine_pred": affine_pred, "height_abs": height_abs, "rpc_corrected": rpc_corrected}

    l_grid_impl, _ = losses["aff_grid"](
        affine_pred=affine_pred,
        affine_gt_forward=batch_dev["affine_gt_forward"].to(device=device, dtype=dtype),
        image_hw=(h, w),
        ref_view_idx=ref,
    )
    l_pair_impl, _, _ = losses["aff_pair"](outputs, batch_dev)
    l_h_impl, _ = losses["height"](height_abs, h_gt, batch_dev["height_valid_mask"].to(device=device, dtype=dtype))
    l_p_impl, _, _ = losses["point"](point_abs, point_anchor, batch_dev, return_aux=False)

    l_grid_true = _affine_grid_truth(aff_noise, (h, w), ref)
    l_pair_true = _affine_pair_truth(batch_dev, aff_noise, height_abs, max_pairs=losses["pair_cfg"].max_pairs)
    l_h_true = masked_huber_loss(h_gt + h_noise, h_gt, mask=batch_dev["height_valid_mask"].to(device=device, dtype=dtype), beta=losses["height_beta"])
    l_p_true = masked_huber_loss(point_abs, gt_point_map, mask=batch_dev["height_valid_mask"].to(device=device, dtype=dtype), beta=losses["point_beta"])

    out = {
        "phase": torch.tensor(0.0, device=device, dtype=dtype),
        "impl_affine_grid": l_grid_impl.detach(),
        "true_affine_grid": l_grid_true.detach(),
        "diff_affine_grid": (l_grid_impl - l_grid_true).abs().detach(),
        "impl_affine_pair": l_pair_impl.detach(),
        "true_affine_pair": l_pair_true.detach(),
        "diff_affine_pair": (l_pair_impl - l_pair_true).abs().detach(),
        "impl_height": l_h_impl.detach(),
        "true_height": l_h_true.detach(),
        "diff_height": (l_h_impl - l_h_true).abs().detach(),
        "impl_point": l_p_impl.detach(),
        "true_point": l_p_true.detach(),
        "diff_point": (l_p_impl - l_p_true).abs().detach(),
        "noise_affine_mean_abs": aff_noise[..., :2, :].abs().mean().detach(),
        "noise_height_std": h_noise.std().detach(),
        "noise_point_std": point_noise.std().detach(),
    }
    if is_main_process():
        print(f"\n[loss_check] ===== {phase_name} =====")
        for k in [
            "impl_affine_grid",
            "true_affine_grid",
            "diff_affine_grid",
            "impl_affine_pair",
            "true_affine_pair",
            "diff_affine_pair",
            "impl_height",
            "true_height",
            "diff_height",
            "impl_point",
            "true_point",
            "diff_point",
            "noise_affine_mean_abs",
            "noise_height_std",
            "noise_point_std",
        ]:
            print(f"{k}: {float(out[k].item()):.8f}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDP loss consistency checker")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase-batches", type=int, default=1, help="batches per phase on each rank")
    p.add_argument("--tol-grid", type=float, default=5e-4)
    p.add_argument("--tol-pair", type=float, default=2e-3)
    p.add_argument("--tol-height", type=float, default=1e-6)
    p.add_argument("--tol-point", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    dist_state = init_distributed(backend=str(cfg.get("system", {}).get("ddp_backend", "nccl")))
    device: torch.device = dist_state["device"]
    rank:int  = dist_state["rank"]
    seed_everything(args.seed)

    print(f"[rank:{rank}] start building dataloaders")
    train_loader, _ = build_dataloaders(cfg, distributed=dist_state["distributed"])
    it = iter(train_loader)

    ops = RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32)
    lcfg = cfg.get("loss", {})
    pair_cfg = AffinePairwiseGeometryLossCfg(
        anchors_per_pair=int(lcfg.get("anchors_per_pair", 256)),
        max_pairs=lcfg.get("max_pairs", None),
        sample_from_valid_only=bool(lcfg.get("sample_from_valid_only", True)),
    )
    losses = {
        "aff_grid": AffineGridLoss(AffineGridLossCfg(grid_h=int(lcfg.get("affine_grid_h", 16)), grid_w=int(lcfg.get("affine_grid_w", 16))),
        ),
        "aff_pair": AffinePairwiseGeometryLoss(ops, pair_cfg),
        "height": HeightHuberLoss(beta=float(lcfg.get("height_beta", 1.0))),
        "point": PointMapLoss(geometry_ops=ops, beta=float(lcfg.get("point_beta", 1.0))),
        "pair_cfg": pair_cfg,
        "height_beta": float(lcfg.get("height_beta", 1.0)),
        "point_beta": float(lcfg.get("point_beta", 1.0)),
    }
    noise_cfg = NoiseCfg()

    phases = [
        ("phase1_zero_noise", False, False, False),
        ("phase2_height_point_noise", False, True, True),
        ("phase3_affine_noise", True, False, False),
        ("phase4_all_noise", True, True, True),
    ]

    all_results = []
    for name, a_on, h_on, p_on in phases:
        print(f"[rank:{rank}] start {name}")
        r_acc: dict[str, torch.Tensor] | None = None
        n_used = 0
        for _ in range(max(int(args.phase_batches), 1)):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            batch_dev = move_batch_to_device(copy.deepcopy(batch), device)
            r_i = run_phase(
                name,
                batch_dev=batch_dev,
                ops=ops,
                losses=losses,
                noise_cfg=noise_cfg,
                add_affine_noise=a_on,
                add_height_noise=h_on,
                add_point_noise=p_on,
            )
            if r_acc is None:
                r_acc = {k: v.clone() for k, v in r_i.items()}
            else:
                for k in r_acc.keys():
                    r_acc[k] = r_acc[k] + r_i[k]
            n_used += 1
        assert r_acc is not None
        for k in r_acc.keys():
            r_acc[k] = r_acc[k] / float(n_used)
        r = all_reduce_mean(r_acc)
        all_results.append((name, r))

    if is_main_process():
        print("\n[loss_check] ===== summary (all-reduced) =====")
        for name, r in all_results:
            print(
                f"{name}: "
                f"grid_diff={float(r['diff_affine_grid']):.8f}, "
                f"pair_diff={float(r['diff_affine_pair']):.8f}, "
                f"height_diff={float(r['diff_height']):.8f}, "
                f"point_diff={float(r['diff_point']):.8f}"
            )

        ok = True
        for name, r in all_results:
            ok = ok and float(r["diff_affine_grid"]) <= args.tol_grid
            ok = ok and float(r["diff_affine_pair"]) <= args.tol_pair
            ok = ok and float(r["diff_height"]) <= args.tol_height
            ok = ok and float(r["diff_point"]) <= args.tol_point
            if not ok:
                print(f"[loss_check][WARN] phase={name} exceeds tolerance.")
        if not ok:
            raise SystemExit(2)
        print("[loss_check] PASS")

    destroy_distributed()


if __name__ == "__main__":
    main()

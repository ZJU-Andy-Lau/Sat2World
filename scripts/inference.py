"""Sat2World inference demo.

功能覆盖：
1) 读取单个 scene，随机抽取或指定多视图；
2) 按训练同款策略进行 512 联合裁切；
3) 可选给非参考视图注入随机仿射初始误差；
4) 前向得到 affine/height/point/3DGS 参数；
5) 计算 affine_grid / affine_pair 误差；
6) 可选导出双路径点云 ply；
7) 可选基于双路径构造两个 3DGS 场；
8) 构造覆盖全场的斜视 pinhole 相机并渲染两张 RGB。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.io import (  # noqa: E402
    compute_valid_crop_anchor_bbox,
    estimate_scene_xy_center_scale,
    linesamp_to_raw_xy,
    raw_xy_to_linesamp,
    read_height_tif,
    read_image_tif,
    read_rpc_file,
)
from dataset.perturbation import (  # noqa: E402
    PerturbationConfig,
    build_synthetic_rpc_inputs,
    identity_affine_2x3,
)
from engine.checkpoint import load_checkpoint  # noqa: E402
from loss.affine_loss import (  # noqa: E402
    AffineGridLoss,
    AffineGridLossCfg,
    AffinePairwiseGeometryLoss,
    AffinePairwiseGeometryLossCfg,
)
from model import Sat2World  # noqa: E402
from render import RPCGaussianRenderer, RPCGaussianRendererCfg  # noqa: E402
from render.rpc_gaussian_renderer import VirtualPinholeCamera, sh_basecolor_to_rgb  # noqa: E402
from scripts.train import build_model  # noqa: E402


@dataclass(frozen=True)
class ViewItem:
    view_id: int
    image_path: str
    height_path: str
    rpc_path: str
    full_hw: tuple[int, int]


@dataclass(frozen=True)
class SceneItem:
    scene_id: int
    views: list[ViewItem]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World Inference Demo")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scene-dir", type=str, required=True, help="path/to/scene_xxx")
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--view-num", type=int, default=4)
    p.add_argument("--view-idxs", type=str, default="", help="comma separated index in sorted views, e.g. 0,2,3")

    p.add_argument("--crop-size", type=int, default=512)

    p.add_argument("--apply-random-init-error", action="store_true")
    p.add_argument("--tx-range", type=float, nargs=2, default=[-20.0, 20.0])
    p.add_argument("--ty-range", type=float, nargs=2, default=[-20.0, 20.0])
    p.add_argument("--scale-range", type=float, nargs=2, default=[-1e-4, 1e-4])
    p.add_argument("--shear-range", type=float, nargs=2, default=[-1e-4, 1e-4])

    p.add_argument("--anchors-per-pair", type=int, default=256)
    p.add_argument("--max-pairs", type=int, default=32)
    p.add_argument("--sample-from-valid-only", action="store_true")

    p.add_argument("--export-pointcloud", action="store_true")
    p.add_argument("--pointcloud-sample-stride", type=int, default=2)

    p.add_argument("--render-3dgs", action="store_true")
    p.add_argument("--render-hw", type=int, nargs=2, default=[1024, 1024])
    p.add_argument("--render-margin-ratio", type=float, default=0.05)
    p.add_argument("--render-distance-factor", type=float, default=1.8)
    p.add_argument("--render-max-retries", type=int, default=8)
    p.add_argument("--render-confidence-thresh", type=float, default=0.1)
    p.add_argument("--render-topk", type=int, default=200000)

    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_scene_id(scene_dir: Path) -> int:
    m = re.fullmatch(r"scene_(\d+)", scene_dir.name)
    if m is None:
        raise ValueError(f"scene-dir name must match scene_<id>, got: {scene_dir.name}")
    return int(m.group(1))


def _read_view_item(scene_id: int, view_dir: Path) -> ViewItem:
    m = re.fullmatch(r"view_(\d+)", view_dir.name)
    if m is None:
        raise ValueError(f"Invalid view folder name: {view_dir}")
    vid = int(m.group(1))

    image = view_dir / f"image_{scene_id}_{vid}.tif"
    height = view_dir / f"height_{scene_id}_{vid}.tif"
    rpc = view_dir / f"rpc_{scene_id}_{vid}.txt"
    if not image.exists() or not height.exists() or not rpc.exists():
        raise FileNotFoundError(
            f"Missing files in {view_dir}: expect image/height/rpc with scene={scene_id}, view={vid}"
        )

    img_probe = read_image_tif(str(image))
    full_hw = (int(img_probe.shape[-2]), int(img_probe.shape[-1]))

    return ViewItem(
        view_id=vid,
        image_path=str(image),
        height_path=str(height),
        rpc_path=str(rpc),
        full_hw=full_hw,
    )


def load_scene(scene_dir: str) -> SceneItem:
    p = Path(scene_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"scene-dir not found: {p}")

    sid = _parse_scene_id(p)
    view_dirs = [x for x in p.iterdir() if x.is_dir() and re.fullmatch(r"view_\d+", x.name)]
    if len(view_dirs) == 0:
        raise RuntimeError(f"No view_* folder found in {p}")

    views = [_read_view_item(sid, vd) for vd in sorted(view_dirs, key=lambda x: int(x.name.split("_")[1]))]
    return SceneItem(scene_id=sid, views=views)


def parse_view_indices(arg: str) -> list[int]:
    if arg.strip() == "":
        return []
    out = []
    for s in arg.split(","):
        ss = s.strip()
        if ss == "":
            continue
        out.append(int(ss))
    if len(set(out)) != len(out):
        raise ValueError(f"view-idxs has duplicates: {out}")
    return out


def select_views(scene: SceneItem, view_num: int, view_idxs: list[int], seed: int) -> list[ViewItem]:
    views = list(scene.views)
    v_total = len(views)
    if len(view_idxs) > 0:
        for i in view_idxs:
            if i < 0 or i >= v_total:
                raise IndexError(f"view idx out of range: {i}, total={v_total}")
        return [views[i] for i in view_idxs]

    k = int(view_num)
    if k <= 0:
        raise ValueError("view-num must be > 0")
    if k > v_total:
        raise ValueError(f"view-num={k} exceeds total views={v_total}")

    rng = np.random.default_rng(seed)
    ids = rng.choice(v_total, size=k, replace=False).tolist()
    return [views[int(i)] for i in ids]


def _infer_height_ref(rpc_views: Sequence[Any]) -> torch.Tensor:
    out: list[float] = []
    for rpc_obj in rpc_views:
        if hasattr(rpc_obj, "HEIGHT_OFF"):
            h = rpc_obj.HEIGHT_OFF
            if torch.is_tensor(h):
                out.append(float(h.detach().reshape(-1)[0].item()))
            else:
                out.append(float(h))
        else:
            out.append(0.0)
    return torch.tensor(out, dtype=torch.float32)


def _crop_rpc_offsets(rpc_obj: Any, top: int, left: int) -> Any:
    rpc_new = copy.deepcopy(rpc_obj)
    rpc_new.LINE_OFF = rpc_new.LINE_OFF - torch.as_tensor(float(top), dtype=rpc_new.LINE_OFF.dtype, device=rpc_new.LINE_OFF.device)
    rpc_new.SAMP_OFF = rpc_new.SAMP_OFF - torch.as_tensor(float(left), dtype=rpc_new.SAMP_OFF.dtype, device=rpc_new.SAMP_OFF.device)
    return rpc_new


def _make_support_intersection_anchor(
    rpc_full_views: Sequence[Any],
    full_hws: Sequence[tuple[int, int]],
    crop_size: int,
    h_anchor: float,
    rng: np.random.Generator,
) -> tuple[float, float] | None:
    boxes = [
        compute_valid_crop_anchor_bbox(
            rpc_obj=rpc,
            full_h=int(hw[0]),
            full_w=int(hw[1]),
            crop_size=int(crop_size),
            h_anchor=float(h_anchor),
            support_grid_size=3,
        )
        for rpc, hw in zip(rpc_full_views, full_hws)
    ]
    inter_x_min = max(b[0] for b in boxes)
    inter_x_max = min(b[1] for b in boxes)
    inter_y_min = max(b[2] for b in boxes)
    inter_y_max = min(b[3] for b in boxes)
    if inter_x_min <= inter_x_max and inter_y_min <= inter_y_max:
        x = float(rng.uniform(inter_x_min, inter_x_max)) if inter_x_max > inter_x_min else float(inter_x_min)
        y = float(rng.uniform(inter_y_min, inter_y_max)) if inter_y_max > inter_y_min else float(inter_y_min)
        return x, y
    return None


def select_joint_crop_windows(
    rpc_full_views: Sequence[Any],
    full_hws: Sequence[tuple[int, int]],
    crop_size: int,
    h_anchor: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, int, int]], dict[str, float]]:
    half = float(crop_size) * 0.5
    for h, w in full_hws:
        if h < crop_size or w < crop_size:
            raise RuntimeError(f"full_hw=({h},{w}) smaller than crop_size={crop_size}")

    anchor_xy = _make_support_intersection_anchor(
        rpc_full_views=rpc_full_views,
        full_hws=full_hws,
        crop_size=crop_size,
        h_anchor=h_anchor,
        rng=rng,
    )
    if anchor_xy is None:
        h_ref, w_ref = full_hws[0]
        line_ref = float(rng.uniform(half, float(h_ref) - half))
        samp_ref = float(rng.uniform(half, float(w_ref) - half))
        ref_rpc = rpc_full_views[0]
        l = torch.tensor([line_ref], dtype=torch.double, device=ref_rpc.device)
        s = torch.tensor([samp_ref], dtype=torch.double, device=ref_rpc.device)
        hh = torch.tensor([h_anchor], dtype=torch.double, device=ref_rpc.device)
        xy_ref = linesamp_to_raw_xy(ref_rpc, line=l, samp=s, h=hh)[0]
        anchor_xy = (float(xy_ref[0].item()), float(xy_ref[1].item()))

    x_anchor, y_anchor = anchor_xy
    windows: list[tuple[int, int, int, int]] = []
    centers: list[torch.Tensor] = []
    for rpc, (h_full, w_full) in zip(rpc_full_views, full_hws):
        x = torch.tensor([x_anchor], dtype=torch.double, device=rpc.device)
        y = torch.tensor([y_anchor], dtype=torch.double, device=rpc.device)
        hh = torch.tensor([h_anchor], dtype=torch.double, device=rpc.device)
        ls = raw_xy_to_linesamp(rpc, x=x, y=y, h=hh)[0]
        line_c = float(ls[0].item())
        samp_c = float(ls[1].item())

        top = int(round(line_c - half))
        left = int(round(samp_c - half))
        top = min(max(top, 0), int(h_full - crop_size))
        left = min(max(left, 0), int(w_full - crop_size))
        windows.append((top, left, crop_size, crop_size))

        line_real = torch.tensor([float(top) + half], dtype=torch.double, device=rpc.device)
        samp_real = torch.tensor([float(left) + half], dtype=torch.double, device=rpc.device)
        xy = linesamp_to_raw_xy(rpc, line=line_real, samp=samp_real, h=hh)[0].detach().cpu()
        centers.append(xy)

    center_stack = torch.stack(centers, dim=0)
    d = torch.cdist(center_stack, center_stack, p=2)
    max_center_dist = float(d.max().item()) if d.numel() > 0 else 0.0
    dbg = {
        "crop_anchor_x": float(x_anchor),
        "crop_anchor_y": float(y_anchor),
        "crop_anchor_h": float(h_anchor),
        "max_center_distance_m": max_center_dist,
    }
    return windows, dbg


def build_inference_batch(
    selected_views: Sequence[ViewItem],
    scene_id: int,
    crop_size: int,
    seed: int,
    perturb_cfg: PerturbationConfig,
    apply_random_init_error: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ref_view_idx = 0

    rpc_gt_full = [read_rpc_file(v.rpc_path) for v in selected_views]
    full_hws = [v.full_hw for v in selected_views]

    h_refs_full = _infer_height_ref(rpc_gt_full)
    h_anchor = float(h_refs_full.mean().item()) if h_refs_full.numel() > 0 else 0.0
    crop_windows, crop_dbg = select_joint_crop_windows(
        rpc_full_views=rpc_gt_full,
        full_hws=full_hws,
        crop_size=int(crop_size),
        h_anchor=h_anchor,
        rng=rng,
    )

    images, heights, masks = [], [], []
    rpc_gt_views: list[Any] = []
    crop_tops, crop_lefts = [], []
    for vr, rpc_full, win in zip(selected_views, rpc_gt_full, crop_windows):
        top, left, ch, cw = win
        img = read_image_tif(vr.image_path, window=win)
        hgt, msk = read_height_tif(vr.height_path, window=win)
        rpc_crop = _crop_rpc_offsets(rpc_full, top=top, left=left)

        if tuple(img.shape[-2:]) != (ch, cw):
            raise RuntimeError("Cropped image shape mismatch")
        if tuple(hgt.shape[-2:]) != (ch, cw):
            raise RuntimeError("Cropped height shape mismatch")

        images.append(img)
        heights.append(hgt)
        masks.append(msk)
        rpc_gt_views.append(rpc_crop)
        crop_tops.append(int(top))
        crop_lefts.append(int(left))

    images_t = torch.stack(images, dim=0).to(torch.float32)
    height_gt_t = torch.stack(heights, dim=0).to(torch.float32)
    height_mask_t = torch.stack(masks, dim=0).to(torch.float32)

    if apply_random_init_error:
        rpc_init_views, aff_fwd, aff_corr = build_synthetic_rpc_inputs(
            rpc_gt_views=rpc_gt_views,
            ref_view_idx=ref_view_idx,
            rng=rng,
            perturb_cfg=perturb_cfg,
            dtype=torch.float32,
            device=images_t.device,
        )
    else:
        rpc_init_views = [copy.deepcopy(r) for r in rpc_gt_views]
        eye = identity_affine_2x3(device=images_t.device, dtype=torch.float32)
        v = len(rpc_gt_views)
        aff_fwd = eye.unsqueeze(0).repeat(v, 1, 1)
        aff_corr = eye.unsqueeze(0).repeat(v, 1, 1)

    height_ref = _infer_height_ref(rpc_init_views)
    image_shapes_crop = [(crop_size, crop_size) for _ in selected_views]
    scene_xy_center, scene_xy_scale = estimate_scene_xy_center_scale(
        selected_views_rpc_gt=rpc_init_views,
        selected_image_shapes=image_shapes_crop,
        selected_height_ref=height_ref,
        ref_view_idx=ref_view_idx,
    )

    batch = {
        "images": images_t.unsqueeze(0),
        "height_gt": height_gt_t.unsqueeze(0),
        "height_valid_mask": height_mask_t.unsqueeze(0),
        "rpc_gt": [rpc_gt_views],
        "rpc_init": [rpc_init_views],
        "affine_gt_forward": aff_fwd.unsqueeze(0).to(torch.float32),
        "affine_gt_correction": aff_corr.unsqueeze(0).to(torch.float32),
        "height_ref": height_ref.unsqueeze(0).to(torch.float32),
        "scene_xy_center": scene_xy_center.unsqueeze(0).to(torch.float32),
        "scene_xy_scale": scene_xy_scale.unsqueeze(0).to(torch.float32),
        "ref_view_idx": torch.tensor([ref_view_idx], dtype=torch.long),
        "scene_id": torch.tensor([int(scene_id)], dtype=torch.long),
        "view_ids": torch.tensor([[v.view_id for v in selected_views]], dtype=torch.long),
        "image_paths": [[v.image_path for v in selected_views]],
        "crop_tops": torch.tensor([crop_tops], dtype=torch.long),
        "crop_lefts": torch.tensor([crop_lefts], dtype=torch.long),
        "crop_anchor_xy": torch.tensor([[crop_dbg["crop_anchor_x"], crop_dbg["crop_anchor_y"]]], dtype=torch.float32),
        "crop_anchor_height": torch.tensor([crop_dbg["crop_anchor_h"]], dtype=torch.float32),
        "max_center_distance_m": torch.tensor([crop_dbg["max_center_distance_m"]], dtype=torch.float32),
    }

    diagnostics = {
        "crop_windows": crop_windows,
        "crop_debug": crop_dbg,
    }
    return batch, diagnostics


def move_tensor_fields_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def build_affine_losses(args: argparse.Namespace, geometry_ops: Any) -> tuple[AffineGridLoss, AffinePairwiseGeometryLoss]:
    grid_loss = AffineGridLoss(AffineGridLossCfg(grid_h=16, grid_w=16))
    pair_loss = AffinePairwiseGeometryLoss(
        geometry_ops=geometry_ops,
        cfg=AffinePairwiseGeometryLossCfg(
            anchors_per_pair=int(args.anchors_per_pair),
            max_pairs=int(args.max_pairs) if args.max_pairs > 0 else None,
            sample_from_valid_only=bool(args.sample_from_valid_only),
        ),
    )
    return grid_loss, pair_loss


def save_rgb_tensor(path: Path, chw_rgb: torch.Tensor) -> None:
    arr = chw_rgb.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr_u8).save(path)


def write_ply_xyz(path: Path, xyz: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")


def flatten_centers_map(centers_bv3hw: torch.Tensor, mask_bv1hw: torch.Tensor | None = None, stride: int = 1) -> np.ndarray:
    c = centers_bv3hw[0].permute(0, 2, 3, 1).reshape(-1, 3)
    if mask_bv1hw is not None:
        m = mask_bv1hw[0].reshape(-1) > 0.5
        c = c[m]
    st = max(int(stride), 1)
    c = c[::st]
    return c.detach().cpu().numpy().astype(np.float32)


def _look_at_w2c_np(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    r = np.cross(f, up_hint)
    if float(np.linalg.norm(r)) < 1e-12:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        r = np.cross(f, alt)
    r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, f)
    R = np.stack([r, u, f], axis=0)
    t = -R @ eye
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def _fit_intrinsics_cover_points_np(points_xyz: np.ndarray, w2c: np.ndarray, width: int, height: int, margin_ratio: float) -> np.ndarray:
    cxy = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    xyz_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (w2c[:3, :] @ xyz_h.T).T
    z = cam[:, 2]
    if not np.all(z > 1e-6):
        raise RuntimeError("cover camera invalid: some points behind camera")
    xz = np.abs(cam[:, 0] / z)
    yz = np.abs(cam[:, 1] / z)
    usable_w = max(float(width) * (1.0 - 2.0 * margin_ratio), 1.0)
    usable_h = max(float(height) * (1.0 - 2.0 * margin_ratio), 1.0)
    fx_max = (usable_w / 2.0) / max(float(np.max(xz)), 1e-8)
    fy_max = (usable_h / 2.0) / max(float(np.max(yz)), 1e-8)
    fx = 0.98 * fx_max
    fy = 0.98 * fy_max
    return np.array([[fx, 0.0, cxy[0]], [0.0, fy, cxy[1]], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_cover_camera_45deg(
    xyz_world: np.ndarray,
    image_hw: tuple[int, int],
    device: torch.device,
    margin_ratio: float,
    distance_factor: float,
    max_retries: int,
) -> VirtualPinholeCamera:
    h, w = image_hw
    xyz = xyz_world.astype(np.float64)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    extent = xyz_max - xyz_min
    diag = float(np.linalg.norm(extent) + 1e-8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    view_dir = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)

    last_err: Exception | None = None
    for i in range(max(int(max_retries), 1)):
        d = max(diag * float(distance_factor) * (1.6**i), 1.0)
        eye = center + d * view_dir
        w2c = _look_at_w2c_np(eye=eye, target=center, up_hint=up)
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
        cam = (w2c[:3, :] @ xyz_h.T).T
        if not np.all(cam[:, 2] > 1e-6):
            last_err = RuntimeError("points behind camera")
            continue
        try:
            K = _fit_intrinsics_cover_points_np(xyz, w2c, w, h, float(margin_ratio))
            return VirtualPinholeCamera(
                K=torch.from_numpy(K).to(device=device, dtype=torch.float32),
                w2c=torch.from_numpy(w2c).to(device=device, dtype=torch.float32),
                fit_p50=float("nan"),
                fit_p95=float("nan"),
                fit_max=float("nan"),
            )
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"failed to build cover camera after {max_retries} retries: {last_err}")


def build_flattened_3dgs_field(
    outputs: dict[str, Any],
    path: str,
    confidence_thresh: float,
    topk: int,
) -> dict[str, torch.Tensor]:
    if path not in {"rpc", "point"}:
        raise ValueError("path must be rpc or point")

    centers_key = "gaussian_centers_rpc" if path == "rpc" else "gaussian_centers_point"
    conf_key = "gaussian_confidence_rpc" if path == "rpc" else "gaussian_confidence_point"

    centers = outputs[centers_key][0].permute(0, 2, 3, 1).reshape(-1, 3)
    opacity = outputs["gaussian_opacity"][0].permute(0, 2, 3, 1).reshape(-1, 1)
    scale = outputs["gaussian_scale"][0].permute(0, 2, 3, 1).reshape(-1, 3)
    rotation = outputs["gaussian_rotation"][0].permute(0, 2, 3, 1).reshape(-1, 4)
    sh = outputs["gaussian_sh"][0].permute(0, 2, 3, 1).reshape(-1, outputs["gaussian_sh"].shape[2])
    conf = outputs[conf_key][0].permute(0, 2, 3, 1).reshape(-1, 1)

    keep = conf[:, 0] >= float(confidence_thresh)
    centers, opacity, scale, rotation, sh, conf = centers[keep], opacity[keep], scale[keep], rotation[keep], sh[keep], conf[keep]

    if int(topk) > 0 and centers.shape[0] > int(topk):
        ids = torch.topk(conf[:, 0], k=int(topk), largest=True).indices
        centers, opacity, scale, rotation, sh, conf = centers[ids], opacity[ids], scale[ids], rotation[ids], sh[ids], conf[ids]

    rgb = sh_basecolor_to_rgb(sh)
    opacity_eff = (opacity * conf).clamp(0.0, 1.0)

    return {
        "centers": centers,
        "opacity": opacity_eff,
        "scale": scale,
        "rotation": rotation,
        "rgb": rgb,
        "num": torch.tensor([centers.shape[0]], device=centers.device),
    }


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed))
    device = torch.device(args.device)

    cfg = load_cfg(args.config)
    model: Sat2World = build_model(cfg).to(device)
    load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location="cpu",
        model_strict=bool(cfg.get("system", {}).get("checkpoint_model_strict", False)),
        load_model_only=True,
    )
    model.eval()

    scene = load_scene(args.scene_dir)
    view_idxs = parse_view_indices(args.view_idxs)
    selected_views = select_views(scene, args.view_num, view_idxs, seed=int(args.seed))

    perturb_cfg = PerturbationConfig(
        tx_range=(float(args.tx_range[0]), float(args.tx_range[1])),
        ty_range=(float(args.ty_range[0]), float(args.ty_range[1])),
        scale_range=(float(args.scale_range[0]), float(args.scale_range[1])),
        shear_range=(float(args.shear_range[0]), float(args.shear_range[1])),
    )

    batch_cpu, diagnostics = build_inference_batch(
        selected_views=selected_views,
        scene_id=int(scene.scene_id),
        crop_size=int(args.crop_size),
        seed=int(args.seed),
        perturb_cfg=perturb_cfg,
        apply_random_init_error=bool(args.apply_random_init_error),
    )
    batch = move_tensor_fields_to_device(batch_cpu, device=device)

    with torch.no_grad():
        outputs = model(batch)

    grid_loss_fn, pair_loss_fn = build_affine_losses(args, geometry_ops=model.rpc_ops)
    with torch.no_grad():
        l_grid, p_grid = grid_loss_fn(
            affine_pred=outputs["affine_pred"],
            affine_gt_forward=batch["affine_gt_forward"],
            image_hw=(int(batch["images"].shape[-2]), int(batch["images"].shape[-1])),
            ref_view_idx=batch["ref_view_idx"],
        )
        l_pair, p_pair, aux_pair = pair_loss_fn(outputs, batch)

    results: dict[str, Any] = {
        "scene_id": int(scene.scene_id),
        "selected_view_ids": [int(v.view_id) for v in selected_views],
        "crop_windows": diagnostics["crop_windows"],
        "crop_debug": diagnostics["crop_debug"],
        "loss_affine_grid": float(l_grid.detach().item()),
        "loss_affine_pair": float(l_pair.detach().item()),
        "probe_affine_grid": {k: float(v.detach().item()) for k, v in p_grid.items()},
        "probe_affine_pair": {k: float(v.detach().item()) for k, v in p_pair.items()},
        "aux_affine_pair": {"num_pairs_used": int(aux_pair.get("num_pairs_used", 0))},
    }

    if bool(args.export_pointcloud):
        image_h = int(batch["images"].shape[-2])
        image_w = int(batch["images"].shape[-1])
        pixel_grid = torch.stack(
            torch.meshgrid(
                torch.arange(image_h, device=device, dtype=torch.float32),
                torch.arange(image_w, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )

        scene_scale_for_rpc = torch.ones_like(batch["scene_xy_center"])
        centers_rpc = model.rpc_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=outputs["rpc_corrected"],
            pixel_grid=pixel_grid,
            height_abs=outputs["height_abs"],
            scene_xy_center=batch["scene_xy_center"],
            scene_xy_scale=scene_scale_for_rpc,
            downsample_factor=1,
        )
        centers_point = outputs["point_abs"]

        stride = max(int(args.pointcloud_sample_stride), 1)
        xyz_rpc = flatten_centers_map(centers_rpc, mask_bv1hw=batch["height_valid_mask"], stride=stride)
        xyz_point = flatten_centers_map(centers_point, mask_bv1hw=batch["height_valid_mask"], stride=stride)

        ply_rpc = save_dir / "cloud_rpc_height.ply"
        ply_point = save_dir / "cloud_point_path.ply"
        write_ply_xyz(ply_rpc, xyz_rpc)
        write_ply_xyz(ply_point, xyz_point)
        results["pointcloud"] = {
            "rpc_height_ply": str(ply_rpc),
            "point_path_ply": str(ply_point),
            "num_points_rpc": int(xyz_rpc.shape[0]),
            "num_points_point": int(xyz_point.shape[0]),
            "stride": int(stride),
        }

    if bool(args.render_3dgs):
        if not bool(outputs.get("gaussian_branch_enabled", False)):
            raise RuntimeError("gaussian branch disabled in model output, cannot render 3DGS")

        renderer = RPCGaussianRenderer(model.rpc_ops, RPCGaussianRendererCfg())

        field_rpc = build_flattened_3dgs_field(
            outputs=outputs,
            path="rpc",
            confidence_thresh=float(args.render_confidence_thresh),
            topk=int(args.render_topk),
        )
        field_point = build_flattened_3dgs_field(
            outputs=outputs,
            path="point",
            confidence_thresh=float(args.render_confidence_thresh),
            topk=int(args.render_topk),
        )

        if int(field_rpc["num"].item()) <= 0 or int(field_point["num"].item()) <= 0:
            raise RuntimeError("No gaussians after confidence/topk filtering, cannot render")

        image_hw = (int(args.render_hw[0]), int(args.render_hw[1]))

        cam_rpc = build_cover_camera_45deg(
            xyz_world=field_rpc["centers"].detach().cpu().numpy(),
            image_hw=image_hw,
            device=device,
            margin_ratio=float(args.render_margin_ratio),
            distance_factor=float(args.render_distance_factor),
            max_retries=int(args.render_max_retries),
        )
        cam_point = build_cover_camera_45deg(
            xyz_world=field_point["centers"].detach().cpu().numpy(),
            image_hw=image_hw,
            device=device,
            margin_ratio=float(args.render_margin_ratio),
            distance_factor=float(args.render_distance_factor),
            max_retries=int(args.render_max_retries),
        )

        with torch.no_grad():
            rgb_rpc, alpha_rpc, _depth_rpc = renderer._render_cuda(
                centers=field_rpc["centers"],
                opacity=field_rpc["opacity"],
                scale=field_rpc["scale"],
                rotation=field_rpc["rotation"],
                rgb=field_rpc["rgb"],
                cam=cam_rpc,
                image_hw=image_hw,
            )
            rgb_point, alpha_point, _depth_point = renderer._render_cuda(
                centers=field_point["centers"],
                opacity=field_point["opacity"],
                scale=field_point["scale"],
                rotation=field_point["rotation"],
                rgb=field_point["rgb"],
                cam=cam_point,
                image_hw=image_hw,
            )

        out_rpc = save_dir / "render_rpc_field.png"
        out_point = save_dir / "render_point_field.png"
        save_rgb_tensor(out_rpc, rgb_rpc)
        save_rgb_tensor(out_point, rgb_point)

        results["render_3dgs"] = {
            "rpc_rgb": str(out_rpc),
            "point_rgb": str(out_point),
            "num_gaussians_rpc": int(field_rpc["num"].item()),
            "num_gaussians_point": int(field_point["num"].item()),
            "alpha_cov_rpc": float((alpha_rpc > 1e-4).to(torch.float32).mean().item()),
            "alpha_cov_point": float((alpha_point > 1e-4).to(torch.float32).mean().item()),
            "render_hw": [int(image_hw[0]), int(image_hw[1])],
        }

    report_path = save_dir / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    results["report_path"] = str(report_path)
    return results


def main() -> None:
    args = parse_args()
    results = run_inference(args)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

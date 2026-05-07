"""Cross-view patch correspondence ground-truth utilities.

The helpers in this module only build geometry correspondences; they do not
compute any loss.  All pixel coordinates are in the crop image coordinate system
used by the current batch, and every two-coordinate tensor is ordered as
``line/samp`` (row/column), never ``x/y``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from loss.common import sample_map_bilinear


@dataclass
class PatchCorrespondenceGT:
    """Directed patch correspondence GT for one source->target view pair.

    Coordinate convention:
    - ``src_pixels``, ``tgt_pixels`` and ``tgt_local_pixels`` are ``[M, 2]`` in
      ``line/samp`` order.
    - Pixel coordinates are expressed in the post-crop image coordinate system
      carried by ``batch['images']``.
    - ``tgt_patch_indices`` indexes the flattened target patch grid
      ``row * grid_w + col``. ``tgt_local_pixels`` is the target location inside
      that patch in pixel units and is clamped to ``[0, patch_size)``.
    """

    batch_indices: torch.Tensor
    src_view_indices: torch.Tensor
    tgt_view_indices: torch.Tensor
    src_patch_indices: torch.Tensor
    tgt_patch_indices: torch.Tensor
    src_pixels: torch.Tensor
    tgt_pixels: torch.Tensor
    tgt_local_pixels: torch.Tensor
    heights: torch.Tensor
    valid: torch.Tensor
    before_filter_count: int = 0
    after_filter_count: int = 0

    @property
    def num_valid(self) -> int:
        return int(self.batch_indices.numel())


def _empty(device: torch.device, dtype: torch.dtype, before_filter_count: int = 0) -> PatchCorrespondenceGT:
    long0 = torch.empty((0,), device=device, dtype=torch.long)
    pix0 = torch.empty((0, 2), device=device, dtype=dtype)
    h0 = torch.empty((0,), device=device, dtype=dtype)
    v0 = torch.empty((0,), device=device, dtype=torch.bool)
    return PatchCorrespondenceGT(long0, long0, long0, long0, long0, pix0, pix0, pix0, h0, v0, before_filter_count, 0)


def build_patch_correspondence_gt(
    *,
    geometry_ops: Any,
    batch: dict[str, Any],
    patch_centers: torch.Tensor,
    patch_valid_mask: torch.Tensor,
    patch_grid_hw: tuple[int, int],
    patch_padded_hw: tuple[int, int] | None = None,
    image_hw: tuple[int, int] | None = None,
    src_view_idx: int,
    tgt_view_idx: int,
    rpc_key: str = "rpc_gt",
    require_target_patch_valid: bool = True,
    max_points: int | None = None,
) -> PatchCorrespondenceGT:
    """Build directed source-to-target patch correspondences from RPC geometry.

    For every valid source patch center, this function samples source GT height,
    back-projects ``(line, samp, height)`` through ``batch[rpc_key]`` for the
    source view, projects the object point to the target view with the same RPC
    key, filters invalid/non-finite/out-of-image points, and converts the target
    pixel to a flattened target patch index plus patch-local ``line/samp``.

    ``rpc_key`` may be ``"rpc_gt"`` or ``"rpc_init"``.  The repository keeps both
    as nested lists shaped ``batch_size x num_views``; this function preserves and
    consumes that convention directly.
    """

    if rpc_key not in {"rpc_gt", "rpc_init"}:
        raise ValueError(f"rpc_key must be 'rpc_gt' or 'rpc_init', got {rpc_key!r}")
    if rpc_key not in batch:
        raise KeyError(f"batch does not contain {rpc_key!r}")

    device = patch_valid_mask.device
    dtype = patch_centers.dtype if torch.is_floating_point(patch_centers) else torch.float32
    patch_centers = patch_centers.to(device=device, dtype=dtype)
    patch_valid_mask = patch_valid_mask.to(device=device)
    b, v, n = patch_valid_mask.shape
    gh, gw = int(patch_grid_hw[0]), int(patch_grid_hw[1])
    if n != gh * gw:
        raise ValueError(f"patch token count mismatch: N={n}, Gh*Gw={gh * gw}")
    if not (0 <= int(src_view_idx) < v and 0 <= int(tgt_view_idx) < v):
        raise ValueError(f"view index out of range: src={src_view_idx}, tgt={tgt_view_idx}, V={v}")
    if int(src_view_idx) == int(tgt_view_idx):
        return _empty(device, dtype)

    h_img = int(batch["images"].shape[-2]) if image_hw is None else int(image_hw[0])
    w_img = int(batch["images"].shape[-1]) if image_hw is None else int(image_hw[1])
    if patch_padded_hw is not None:
        hp, wp = int(patch_padded_hw[0]), int(patch_padded_hw[1])
    else:
        hp, wp = h_img, w_img
    patch_h = float(hp) / float(max(gh, 1))
    patch_w = float(wp) / float(max(gw, 1))

    hgt = batch["height_gt"].to(device=device, dtype=dtype)
    hmask = batch["height_valid_mask"].to(device=device, dtype=dtype)
    rpc_batch = batch[rpc_key]
    scene_xy_center = batch.get("scene_xy_center", None)
    scene_xy_scale = batch.get("scene_xy_scale", None)

    centers = patch_centers.view(1, n, 2)
    all_batch: list[torch.Tensor] = []
    all_src_view: list[torch.Tensor] = []
    all_tgt_view: list[torch.Tensor] = []
    all_src_idx: list[torch.Tensor] = []
    all_tgt_idx: list[torch.Tensor] = []
    all_src_pix: list[torch.Tensor] = []
    all_tgt_pix: list[torch.Tensor] = []
    all_tgt_local: list[torch.Tensor] = []
    all_heights: list[torch.Tensor] = []
    before_filter = 0

    for bi in range(b):
        src_valid = patch_valid_mask[bi : bi + 1, int(src_view_idx)]
        if not bool(src_valid.any()):
            continue
        h_src, in_h = sample_map_bilinear(hgt[bi : bi + 1, int(src_view_idx)], centers)
        m_src, in_m = sample_map_bilinear(hmask[bi : bi + 1, int(src_view_idx)], centers)
        h_src = h_src[:, 0]
        valid_src = src_valid & in_h & in_m & (m_src[:, 0] > 0.5)
        valid_src = valid_src & torch.isfinite(centers[..., 0]) & torch.isfinite(centers[..., 1]) & torch.isfinite(h_src)
        if not bool(valid_src.any()):
            continue
        src_idx = valid_src[0].nonzero(as_tuple=False).squeeze(1)
        if max_points is not None and int(max_points) > 0 and src_idx.numel() > int(max_points):
            perm = torch.randperm(src_idx.numel(), device=device)[: int(max_points)]
            src_idx = src_idx[perm]
        src_pts = centers[:, src_idx]
        h_valid = h_src[:, src_idx]
        if src_pts.numel() == 0 or h_valid.numel() == 0:
            continue

        xs, ys = geometry_ops.linesamp_to_xy_batch(
            rpc_batch=[[rpc_batch[bi][int(src_view_idx)]]],
            lines=src_pts[..., 0].view(1, 1, -1),
            samps=src_pts[..., 1].view(1, 1, -1),
            heights=h_valid.view(1, 1, -1),
            scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
            scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
        )
        l_tgt, s_tgt = geometry_ops.xy_to_linesamp_batch(
            rpc_batch=[[rpc_batch[bi][int(tgt_view_idx)]]],
            xs=xs,
            ys=ys,
            heights=h_valid.view(1, 1, -1),
            scene_xy_center=None if scene_xy_center is None else scene_xy_center[bi : bi + 1],
            scene_xy_scale=None if scene_xy_scale is None else scene_xy_scale[bi : bi + 1],
        )
        l_tgt = l_tgt.view(-1).to(device=device, dtype=dtype)
        s_tgt = s_tgt.view(-1).to(device=device, dtype=dtype)
        h_flat = h_valid.view(-1).to(device=device, dtype=dtype)
        finite = torch.isfinite(l_tgt) & torch.isfinite(s_tgt) & torch.isfinite(h_flat)
        if not bool(finite.any()):
            continue
        src_idx = src_idx[finite]
        l_tgt = l_tgt[finite]
        s_tgt = s_tgt[finite]
        h_flat = h_flat[finite]
        before_filter += int(src_idx.shape[0])

        in_img = (l_tgt >= 0) & (l_tgt <= (h_img - 1)) & (s_tgt >= 0) & (s_tgt <= (w_img - 1))
        if not bool(in_img.any()):
            continue
        src_idx = src_idx[in_img]
        l_tgt = l_tgt[in_img]
        s_tgt = s_tgt[in_img]
        h_flat = h_flat[in_img]

        # Use continuous padded-grid coordinates, then choose nearest target patch
        # exactly as the historical PatchInternalMatchLoss did for square 16px
        # patches when padded_hw == image_hw.
        row_cont = (l_tgt + 0.5) / patch_h - 0.5
        col_cont = (s_tgt + 0.5) / patch_w - 0.5
        tgt_row = torch.round(row_cont).long().clamp(0, gh - 1)
        tgt_col = torch.round(col_cont).long().clamp(0, gw - 1)
        tgt_flat = tgt_row * gw + tgt_col
        if require_target_patch_valid:
            tgt_valid = patch_valid_mask[bi, int(tgt_view_idx), tgt_flat]
            if not bool(tgt_valid.any()):
                continue
            src_idx = src_idx[tgt_valid]
            l_tgt = l_tgt[tgt_valid]
            s_tgt = s_tgt[tgt_valid]
            h_flat = h_flat[tgt_valid]
            tgt_row = tgt_row[tgt_valid]
            tgt_col = tgt_col[tgt_valid]
            tgt_flat = tgt_flat[tgt_valid]

        top_line = tgt_row.to(dtype=dtype) * patch_h
        top_samp = tgt_col.to(dtype=dtype) * patch_w
        local_line = (l_tgt - top_line).clamp(0.0, patch_h - 1e-4)
        local_samp = (s_tgt - top_samp).clamp(0.0, patch_w - 1e-4)
        tgt_local = torch.stack([local_line, local_samp], dim=-1)
        finite_local = torch.isfinite(tgt_local).all(dim=-1)
        if not bool(finite_local.any()):
            continue
        src_idx = src_idx[finite_local]
        l_tgt = l_tgt[finite_local]
        s_tgt = s_tgt[finite_local]
        h_flat = h_flat[finite_local]
        tgt_flat = tgt_flat[finite_local]
        tgt_local = tgt_local[finite_local]
        count = int(src_idx.shape[0])
        if count == 0:
            continue
        all_batch.append(torch.full((count,), bi, device=device, dtype=torch.long))
        all_src_view.append(torch.full((count,), int(src_view_idx), device=device, dtype=torch.long))
        all_tgt_view.append(torch.full((count,), int(tgt_view_idx), device=device, dtype=torch.long))
        all_src_idx.append(src_idx.to(device=device, dtype=torch.long))
        all_tgt_idx.append(tgt_flat.to(device=device, dtype=torch.long))
        all_src_pix.append(centers[0, src_idx])
        all_tgt_pix.append(torch.stack([l_tgt, s_tgt], dim=-1))
        all_tgt_local.append(tgt_local)
        all_heights.append(h_flat)

    if len(all_batch) == 0:
        return _empty(device, dtype, before_filter_count=before_filter)
    out = PatchCorrespondenceGT(
        batch_indices=torch.cat(all_batch, dim=0),
        src_view_indices=torch.cat(all_src_view, dim=0),
        tgt_view_indices=torch.cat(all_tgt_view, dim=0),
        src_patch_indices=torch.cat(all_src_idx, dim=0),
        tgt_patch_indices=torch.cat(all_tgt_idx, dim=0),
        src_pixels=torch.cat(all_src_pix, dim=0),
        tgt_pixels=torch.cat(all_tgt_pix, dim=0),
        tgt_local_pixels=torch.cat(all_tgt_local, dim=0),
        heights=torch.cat(all_heights, dim=0),
        valid=torch.ones((sum(x.numel() for x in all_batch),), device=device, dtype=torch.bool),
        before_filter_count=before_filter,
        after_filter_count=sum(int(x.numel()) for x in all_batch),
    )
    return out

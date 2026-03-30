"""dataset.rpc_scene_dataset

本文件实现 Sat2World 的主 Dataset、collate_fn 与构建便捷函数。

设计目标：
- 输出字段与 Sat2World.forward 约定严格兼容；
- 训练模式支持“在线视图采样 + 在线 RPC 仿射扰动”；
- 验证/测试默认不扰动，支持可选确定性合成扰动评测；
- rpc_init/rpc_gt 保持为嵌套 list[RPCModelParameterTorch]，不做 tensor 化。
- 数据读取策略改为：先基于 full-view RPC 一次性规划 crop，再 window 读取 512×512 patch。
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from geometry.rpc import RPCModelParameterTorch

from .io import (
    SceneRecord,
    ViewRecord,
    compute_valid_crop_anchor_bbox,
    estimate_scene_xy_center_scale,
    linesamp_to_raw_xy,
    raw_xy_to_linesamp,
    read_height_tif,
    read_image_tif,
    read_rpc_file,
    scan_dataset_root,
)
from .perturbation import (
    PerturbationConfig,
    build_synthetic_rpc_inputs,
    identity_affine_2x3,
    make_deterministic_rng,
)


class RPCSceneDataset(Dataset):
    """RPC 场景级多视图数据集。

    关键约定（请在训练/损失实现时保持一致）：
    1) 磁盘上的 RPC 是 GT 已平差 full-view RPC（正确模型）。
    2) dataset 先用 full-view RPC 规划 crop，再读 patch，并同步修正 RPC offset。
    3) rpc_init 由 crop 后 GT RPC 注入 forward 仿射扰动得到。
    4) affine_gt_forward: true patch pixel -> observed patch pixel。
    5) affine_gt_correction: observed patch pixel -> true patch pixel（forward 的逆）。
    6) 参考视图（ref_view_idx=0）永不扰动，forward/correction 均为单位阵。
    """

    def __init__(
        self,
        root: str,
        mode: str = "train",
        num_views: Optional[int] = None,
        max_view_num: Optional[int] = None,
        samples_per_scene: int = 1,
        min_views: int = 2,
        min_view_num: int = 1,
        apply_perturbation: bool = True,
        synthetic_perturbation_in_eval: bool = False,
        perturb_cfg: Optional[PerturbationConfig | Mapping[str, Any]] = None,
        cache_rpc: bool = True,
        cache_image: bool = False,
        cache_height: bool = False,
        base_seed: int = 0,
        reference_policy: str = "first",
        strict_same_hw: bool = True,
        crop_size: int = 512,
        anchors_per_view_per_sample: int = 0,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be one of train/val/test, got {mode}")

        self.root = root
        self.mode = mode
        self.max_view_num = max_view_num if max_view_num is not None else num_views
        self.samples_per_scene = int(samples_per_scene)
        self.min_views = int(min_views)
        self.min_view_num = int(min_view_num)
        if self.min_view_num <= 0:
            raise ValueError("min_view_num must be > 0")
        self.apply_perturbation = bool(apply_perturbation)
        self.synthetic_perturbation_in_eval = bool(synthetic_perturbation_in_eval)
        if perturb_cfg is None:
            self.perturb_cfg = PerturbationConfig()
        elif isinstance(perturb_cfg, PerturbationConfig):
            self.perturb_cfg = perturb_cfg
        elif isinstance(perturb_cfg, Mapping):
            self.perturb_cfg = PerturbationConfig.from_mapping(perturb_cfg)
        else:
            raise TypeError(
                "perturb_cfg must be PerturbationConfig | mapping | None, "
                f"got {type(perturb_cfg).__name__}"
            )

        self.cache_rpc = bool(cache_rpc)
        self.cache_image = bool(cache_image)
        self.cache_height = bool(cache_height)

        self.base_seed = int(base_seed)
        self.epoch = 0

        self.reference_policy = reference_policy
        if self.reference_policy != "first":
            raise ValueError("Current stage only supports reference_policy='first'")

        self.strict_same_hw = bool(strict_same_hw)
        self.crop_size = int(crop_size)
        if self.crop_size <= 0:
            raise ValueError("crop_size must be > 0")
        self.anchors_per_view_per_sample = int(anchors_per_view_per_sample)

        all_scenes = scan_dataset_root(root)
        scenes = [s for s in all_scenes if len(s.views) >= self.min_views]
        if len(scenes) == 0:
            raise RuntimeError(
                f"No valid scenes after filtering min_views={self.min_views}. "
                f"root={root}, total_scenes={len(all_scenes)}"
            )

        self.scenes = sorted(scenes, key=lambda s: s.scene_id)

        if self.mode == "train":
            if self.max_view_num is None:
                raise ValueError("max_view_num (or num_views) must be specified in train mode")
            if self.max_view_num <= 0:
                raise ValueError("max_view_num must be > 0")
            if self.samples_per_scene <= 0:
                raise ValueError("samples_per_scene must be > 0 in train mode")

        self.rpc_cache: dict[str, "RPCModelParameterTorch"] = {}
        self.image_cache: dict[str, torch.Tensor] = {}
        self.height_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.scenes) * self.samples_per_scene
        return len(self.scenes)

    def _scene_from_index(self, index: int) -> SceneRecord:
        if self.mode == "train":
            scene_idx = index // self.samples_per_scene
        else:
            scene_idx = index
        return self.scenes[scene_idx]

    def _select_views(self, scene_record: SceneRecord, rng) -> tuple[list[ViewRecord], int]:
        views = list(scene_record.views)
        views.sort(key=lambda v: v.view_id)
        if self.mode == "train":
            return views, 0
        selected = views if self.max_view_num is None else views
        return selected, 0

    def _load_rpc_full(self, rpc_path: str) -> "RPCModelParameterTorch":
        if self.cache_rpc and rpc_path in self.rpc_cache:
            return copy.deepcopy(self.rpc_cache[rpc_path])
        rpc = read_rpc_file(rpc_path)
        if self.cache_rpc:
            self.rpc_cache[rpc_path] = copy.deepcopy(rpc)
        return rpc

    def _infer_height_ref(self, rpc_views: Sequence["RPCModelParameterTorch"]) -> torch.Tensor:
        out = []
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

    def _crop_rpc_offsets(self, rpc_obj: "RPCModelParameterTorch", top: int, left: int) -> "RPCModelParameterTorch":
        """生成 crop 后 RPC（仅改像平面偏移）。"""
        rpc_new = copy.deepcopy(rpc_obj)
        rpc_new.LINE_OFF = rpc_new.LINE_OFF - torch.as_tensor(float(top), dtype=rpc_new.LINE_OFF.dtype, device=rpc_new.LINE_OFF.device)
        rpc_new.SAMP_OFF = rpc_new.SAMP_OFF - torch.as_tensor(float(left), dtype=rpc_new.SAMP_OFF.dtype, device=rpc_new.SAMP_OFF.device)
        return rpc_new

    def _make_support_intersection_anchor(
        self,
        rpc_full_views: Sequence["RPCModelParameterTorch"],
        full_hws: Sequence[tuple[int, int]],
        h_anchor: float,
        rng,
    ) -> tuple[float, float] | None:
        """通过各视图合法物方 bbox 交集一次性采样共同 anchor。"""
        boxes = [
            compute_valid_crop_anchor_bbox(
                rpc_obj=rpc,
                full_h=int(hw[0]),
                full_w=int(hw[1]),
                crop_size=self.crop_size,
                h_anchor=h_anchor,
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

    def _sample_view_anchors(
        self,
        height_gt: torch.Tensor,
        height_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为单个视图从有效高程区域抽样控制点。

        输入:
            height_gt: [1,H,W]
            height_valid_mask: [1,H,W]
        返回:
            line_samp: [K,2]
            h_true: [K]
        """
        k = self.anchors_per_view_per_sample
        if k <= 0:
            return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)

        valid = (height_valid_mask[0] > 0.5).nonzero(as_tuple=False)
        h, w = height_gt.shape[-2:]
        if valid.numel() == 0:
            ys = torch.randint(low=0, high=h, size=(k,), dtype=torch.long)
            xs = torch.randint(low=0, high=w, size=(k,), dtype=torch.long)
        else:
            if valid.shape[0] >= k:
                ids = torch.randperm(valid.shape[0])[:k]
                pick = valid[ids]
            else:
                rep = torch.randint(low=0, high=valid.shape[0], size=(k,), dtype=torch.long)
                pick = valid[rep]
            ys = pick[:, 0]
            xs = pick[:, 1]

        line_samp = torch.stack([ys.to(torch.float32), xs.to(torch.float32)], dim=-1)
        h_true = height_gt[0, ys, xs].to(torch.float32)
        return line_samp, h_true

    def _select_joint_crop_windows(
        self,
        rpc_full_views: Sequence["RPCModelParameterTorch"],
        full_hws: Sequence[tuple[int, int]],
        h_anchor: float,
        rng,
    ) -> tuple[list[tuple[int, int, int, int]], dict[str, float]]:
        """一次性确定所有视图 crop window（无 rejection 循环）。"""
        half = float(self.crop_size) * 0.5
        for h, w in full_hws:
            if h < self.crop_size or w < self.crop_size:
                raise RuntimeError(f"full_hw=({h},{w}) smaller than crop_size={self.crop_size}")

        anchor_xy = self._make_support_intersection_anchor(rpc_full_views, full_hws, h_anchor, rng)

        if anchor_xy is None:
            # 解析 fallback：在参考视图合法中心内随机，反投影为共同 anchor。
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
        center_xy_world: list[torch.Tensor] = []
        for rpc, (h_full, w_full) in zip(rpc_full_views, full_hws):
            x = torch.tensor([x_anchor], dtype=torch.double, device=rpc.device)
            y = torch.tensor([y_anchor], dtype=torch.double, device=rpc.device)
            hh = torch.tensor([h_anchor], dtype=torch.double, device=rpc.device)
            ls = raw_xy_to_linesamp(rpc, x=x, y=y, h=hh)[0]  # [2], line,samp
            line_c = float(ls[0].item())
            samp_c = float(ls[1].item())

            top = int(round(line_c - half))
            left = int(round(samp_c - half))
            top = min(max(top, 0), int(h_full - self.crop_size))
            left = min(max(left, 0), int(w_full - self.crop_size))
            windows.append((top, left, self.crop_size, self.crop_size))

            # debug probe: crop 实际中心在物方的分布
            line_real = torch.tensor([float(top) + half], dtype=torch.double, device=rpc.device)
            samp_real = torch.tensor([float(left) + half], dtype=torch.double, device=rpc.device)
            xy = linesamp_to_raw_xy(rpc, line=line_real, samp=samp_real, h=hh)[0].detach().cpu()
            center_xy_world.append(xy)

        center_stack = torch.stack(center_xy_world, dim=0)  # [V,2]
        d = torch.cdist(center_stack, center_stack, p=2)
        max_center_dist = float(d.max().item()) if d.numel() > 0 else 0.0

        debug = {
            "crop_anchor_x": float(x_anchor),
            "crop_anchor_y": float(y_anchor),
            "crop_anchor_h": float(h_anchor),
            "max_center_distance_m": max_center_dist,
        }
        return windows, debug

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_record = self._scene_from_index(index)
        scene_id = scene_record.scene_id
        rng = make_deterministic_rng(self.base_seed, self.epoch, scene_id, index)

        selected_views, ref_view_idx = self._select_views(scene_record, rng)

        # 1) 先读取 full-view rpc 与 full_hw（不读整图像素）
        rpc_gt_full_views = [self._load_rpc_full(v.rpc_path) for v in selected_views]
        full_hws = [tuple(v.full_hw) for v in selected_views]

        if self.strict_same_hw:
            h0, w0 = full_hws[0]
            for i, (hi, wi) in enumerate(full_hws[1:], start=1):
                if hi != h0 or wi != w0:
                    raise RuntimeError(
                        f"Scene {scene_id} selected views have different HW: "
                        f"view0=({h0},{w0}), view{i}=({hi},{wi})"
                    )

        # 2) 用 full-view rpc 先确定共同物方 anchor 与所有窗口
        h_refs_full = self._infer_height_ref(rpc_gt_full_views)
        h_anchor = float(h_refs_full.mean().item()) if h_refs_full.numel() > 0 else 0.0
        crop_windows, crop_dbg = self._select_joint_crop_windows(
            rpc_full_views=rpc_gt_full_views,
            full_hws=full_hws,
            h_anchor=h_anchor,
            rng=rng,
        )

        # 3) 按窗口读取 image/height，并把 full-view rpc 修正到 crop 坐标系
        images, heights, height_masks = [], [], []
        rpc_gt_views = []
        crop_tops, crop_lefts = [], []
        for vr, rpc_full, win in zip(selected_views, rpc_gt_full_views, crop_windows):
            top, left, ch, cw = win
            img = read_image_tif(vr.image_path, window=win)
            hgt, hmask = read_height_tif(vr.height_path, window=win)
            rpc_crop = self._crop_rpc_offsets(rpc_full, top=top, left=left)

            if tuple(img.shape[-2:]) != (ch, cw):
                raise RuntimeError(f"Cropped image shape mismatch: got={tuple(img.shape[-2:])}, expect={(ch,cw)}")
            if tuple(hgt.shape[-2:]) != (ch, cw):
                raise RuntimeError(f"Cropped height shape mismatch: got={tuple(hgt.shape[-2:])}, expect={(ch,cw)}")

            images.append(img)
            heights.append(hgt)
            height_masks.append(hmask)
            rpc_gt_views.append(rpc_crop)
            crop_tops.append(int(top))
            crop_lefts.append(int(left))

        images_t = torch.stack(images, dim=0).to(torch.float32)  # [V,3,512,512]
        height_gt_t = torch.stack(heights, dim=0).to(torch.float32)  # [V,1,512,512]
        height_mask_t = torch.stack(height_masks, dim=0).to(torch.float32)  # [V,1,512,512]

        # 4) 后续流程全部基于 crop 后 rpc / crop 尺寸
        height_ref = self._infer_height_ref(rpc_gt_views)
        image_shapes_crop = [(self.crop_size, self.crop_size) for _ in selected_views]
        scene_xy_center, scene_xy_scale = estimate_scene_xy_center_scale(
            selected_views_rpc_gt=rpc_gt_views,
            selected_image_shapes=image_shapes_crop,
            selected_height_ref=height_ref,
        )

        do_perturb = (self.mode == "train" and self.apply_perturbation) or (
            self.mode in {"val", "test"} and self.synthetic_perturbation_in_eval
        )

        if do_perturb:
            rpc_init_views, aff_fwd, aff_corr = build_synthetic_rpc_inputs(
                rpc_gt_views=rpc_gt_views,
                ref_view_idx=ref_view_idx,
                rng=rng,
                perturb_cfg=self.perturb_cfg,
                dtype=torch.float32,
                device=images_t.device,
            )
        else:
            rpc_init_views = [copy.deepcopy(r) for r in rpc_gt_views]
            eye = identity_affine_2x3(device=images_t.device, dtype=torch.float32)
            v = len(rpc_gt_views)
            aff_fwd = eye.unsqueeze(0).repeat(v, 1, 1)
            aff_corr = eye.unsqueeze(0).repeat(v, 1, 1)

        sample = {
            "images": images_t,
            "height_gt": height_gt_t,
            "height_valid_mask": height_mask_t,
            "rpc_gt": rpc_gt_views,
            "rpc_init": rpc_init_views,
            "affine_gt_forward": aff_fwd.to(torch.float32),
            "affine_gt_correction": aff_corr.to(torch.float32),
            "height_ref": height_ref.to(torch.float32),
            "scene_xy_center": scene_xy_center.to(torch.float32),
            "scene_xy_scale": scene_xy_scale.to(torch.float32),
            "ref_view_idx": int(ref_view_idx),
            "scene_id": int(scene_id),
            "view_ids": torch.tensor([v.view_id for v in selected_views], dtype=torch.long),
            "image_paths": [v.image_path for v in selected_views],
            "max_view_num": int(self.max_view_num) if self.max_view_num is not None else int(len(selected_views)),
            "min_view_num": int(self.min_view_num),
            # 附加调试字段（不影响下游主接口）
            "crop_tops": torch.tensor(crop_tops, dtype=torch.long),
            "crop_lefts": torch.tensor(crop_lefts, dtype=torch.long),
            "crop_anchor_xy": torch.tensor([crop_dbg["crop_anchor_x"], crop_dbg["crop_anchor_y"]], dtype=torch.float32),
            "crop_anchor_height": torch.tensor(crop_dbg["crop_anchor_h"], dtype=torch.float32),
            "max_center_distance_m": torch.tensor(crop_dbg["max_center_distance_m"], dtype=torch.float32),
        }
        if self.anchors_per_view_per_sample > 0:
            anchor_ls, anchor_h = [], []
            for vi in range(len(selected_views)):
                ls_i, h_i = self._sample_view_anchors(height_gt_t[vi], height_mask_t[vi])
                anchor_ls.append(ls_i)
                anchor_h.append(h_i)
            sample["anchor_line_samp_true"] = torch.stack(anchor_ls, dim=0).to(torch.float32)  # [V,K,2]
            sample["anchor_height_true"] = torch.stack(anchor_h, dim=0).to(torch.float32)  # [V,K]
        return sample


def rpc_scene_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch")

    h0 = batch[0]["images"].shape[-2]
    w0 = batch[0]["images"].shape[-1]
    for i, sample in enumerate(batch[1:], start=1):
        hi = sample["images"].shape[-2]
        wi = sample["images"].shape[-1]
        if hi != h0 or wi != w0:
            raise RuntimeError(f"Batch has inconsistent HW: sample0=({h0},{w0}), sample{i}=({hi},{wi})")

    # k 的上限必须受“样本真实视图数”约束，避免 k > v 导致后续字段维度错配。
    k_upper = int(
        min(
            min(int(x["images"].shape[0]), int(x.get("max_view_num", x["images"].shape[0])))
            for x in batch
        )
    )
    if k_upper <= 0:
        raise RuntimeError("effective view upper bound must be > 0 for collate")

    k_lower = int(min(max(int(x.get("min_view_num", 1)), 1) for x in batch))
    k_lower = min(k_lower, k_upper)
    k = int(torch.randint(low=k_lower, high=k_upper + 1, size=(1,)).item())

    def _pick_indices(v: int) -> torch.Tensor:
        if k == 1:
            return torch.tensor([0], dtype=torch.long)
        if v == 1:
            return torch.zeros((k,), dtype=torch.long)
        candidate = torch.arange(1, v, dtype=torch.long)
        need = k - 1
        if candidate.numel() >= need:
            perm = torch.randperm(candidate.numel())[:need]
            others = candidate[perm]
        else:
            rep = torch.randint(low=0, high=candidate.numel(), size=(need,), dtype=torch.long)
            others = candidate[rep]
        return torch.cat([torch.tensor([0], dtype=torch.long), others], dim=0)

    images_b, height_b, mask_b = [], [], []
    aff_fwd_b, aff_cor_b, href_b = [], [], []
    rpc_gt_b, rpc_init_b = [], []
    view_ids_b, image_paths_b = [], []
    scene_center_b, scene_scale_b, ref_idx_b, scene_id_b = [], [], [], []
    crop_tops_b, crop_lefts_b = [], []
    crop_anchor_xy_b, crop_anchor_h_b, max_center_dist_b = [], [], []
    anchor_ls_b, anchor_h_b = [], []

    for sample in batch:
        v = sample["images"].shape[0]
        idx = _pick_indices(v)

        images_b.append(sample["images"][idx])
        height_b.append(sample["height_gt"][idx])
        mask_b.append(sample["height_valid_mask"][idx])
        aff_fwd_b.append(sample["affine_gt_forward"][idx])
        aff_cor_b.append(sample["affine_gt_correction"][idx])
        href_b.append(sample["height_ref"][idx])

        rpc_gt_b.append([sample["rpc_gt"][int(i)] for i in idx.tolist()])
        rpc_init_b.append([sample["rpc_init"][int(i)] for i in idx.tolist()])
        view_ids_b.append(sample["view_ids"][idx])
        image_paths_b.append([sample["image_paths"][int(i)] for i in idx.tolist()])

        scene_center_b.append(sample["scene_xy_center"])
        scene_scale_b.append(sample["scene_xy_scale"])
        ref_idx_b.append(0)
        scene_id_b.append(int(sample["scene_id"]))

        if "crop_tops" in sample:
            crop_tops_b.append(sample["crop_tops"][idx])
        if "crop_lefts" in sample:
            crop_lefts_b.append(sample["crop_lefts"][idx])
        if "crop_anchor_xy" in sample:
            crop_anchor_xy_b.append(sample["crop_anchor_xy"])
        if "crop_anchor_height" in sample:
            crop_anchor_h_b.append(sample["crop_anchor_height"])
        if "max_center_distance_m" in sample:
            max_center_dist_b.append(sample["max_center_distance_m"])
        if "anchor_line_samp_true" in sample:
            anchor_ls_b.append(sample["anchor_line_samp_true"][idx])
        if "anchor_height_true" in sample:
            anchor_h_b.append(sample["anchor_height_true"][idx])

    out = {
        "images": torch.stack(images_b, dim=0).to(torch.float32),
        "height_gt": torch.stack(height_b, dim=0).to(torch.float32),
        "height_valid_mask": torch.stack(mask_b, dim=0).to(torch.float32),
        "affine_gt_forward": torch.stack(aff_fwd_b, dim=0).to(torch.float32),
        "affine_gt_correction": torch.stack(aff_cor_b, dim=0).to(torch.float32),
        "height_ref": torch.stack(href_b, dim=0).to(torch.float32),
        "scene_xy_center": torch.stack(scene_center_b, dim=0).to(torch.float32),
        "scene_xy_scale": torch.stack(scene_scale_b, dim=0).to(torch.float32),
        "rpc_gt": rpc_gt_b,
        "rpc_init": rpc_init_b,
        "ref_view_idx": torch.tensor(ref_idx_b, dtype=torch.long),
        "scene_id": torch.tensor(scene_id_b, dtype=torch.long),
        "view_ids": torch.stack(view_ids_b, dim=0).to(torch.long),
        "image_paths": image_paths_b,
    }
    if len(crop_tops_b) > 0:
        out["crop_tops"] = torch.stack(crop_tops_b, dim=0).to(torch.long)
    if len(crop_lefts_b) > 0:
        out["crop_lefts"] = torch.stack(crop_lefts_b, dim=0).to(torch.long)
    if len(crop_anchor_xy_b) > 0:
        out["crop_anchor_xy"] = torch.stack(crop_anchor_xy_b, dim=0).to(torch.float32)
    if len(crop_anchor_h_b) > 0:
        out["crop_anchor_height"] = torch.stack(crop_anchor_h_b, dim=0).to(torch.float32)
    if len(max_center_dist_b) > 0:
        out["max_center_distance_m"] = torch.stack(max_center_dist_b, dim=0).to(torch.float32)
    if len(anchor_ls_b) > 0:
        out["anchor_line_samp_true"] = torch.stack(anchor_ls_b, dim=0).to(torch.float32)
    if len(anchor_h_b) > 0:
        out["anchor_height_true"] = torch.stack(anchor_h_b, dim=0).to(torch.float32)

    v_out = int(out["images"].shape[1])
    if int(out["affine_gt_forward"].shape[1]) != v_out:
        raise RuntimeError(
            f"collate view mismatch: images V={v_out}, affine_gt_forward V={int(out['affine_gt_forward'].shape[1])}"
        )
    if "anchor_line_samp_true" in out and int(out["anchor_line_samp_true"].shape[1]) != v_out:
        raise RuntimeError(
            f"collate view mismatch: images V={v_out}, anchor_line_samp_true V={int(out['anchor_line_samp_true'].shape[1])}"
        )
    return out


def build_dataset(mode: str, **kwargs) -> RPCSceneDataset:
    return RPCSceneDataset(mode=mode, **kwargs)

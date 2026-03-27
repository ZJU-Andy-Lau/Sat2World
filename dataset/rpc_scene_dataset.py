"""dataset.rpc_scene_dataset

本文件实现 Sat2World 的主 Dataset、collate_fn 与构建便捷函数。

设计目标：
- 输出字段与 Sat2World.forward 约定严格兼容；
- 训练模式支持“在线视图采样 + 在线 RPC 仿射扰动”；
- 验证/测试默认不扰动，支持可选确定性合成扰动评测；
- rpc_init/rpc_gt 保持为嵌套 list[RPCModelParameterTorch]，不做 tensor 化。
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from geometry.rpc import RPCModelParameterTorch

from .io import (
    SceneRecord,
    ViewRecord,
    estimate_scene_xy_center_scale,
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
    1) 磁盘上的 RPC 是 GT 已平差 RPC（正确模型）。
    2) rpc_init 由 GT RPC 注入 forward 仿射扰动得到。
    3) affine_gt_forward: true pixel -> observed pixel。
    4) affine_gt_correction: observed pixel -> true pixel（forward 的逆）。
    5) 参考视图（ref_view_idx=0）永不扰动，forward/correction 均为单位阵。

    成员变量说明:
    - root/mode: 数据根目录与模式。
    - scenes: 扫描并过滤后的场景记录。
    - num_views: 每个样本选择视图数（test 可为 None 表示全视图）。
    - samples_per_scene: train 模式下每场景每 epoch 的采样次数。
    - apply_perturbation/synthetic_perturbation_in_eval: 扰动控制开关。
    - perturb_cfg: 扰动范围配置。
    - cache_rpc/cache_image/cache_height: 缓存开关与缓存字典。
    - base_seed/epoch: 确定性随机采样控制。
    - strict_same_hw: 是否严格要求样本内所有视图 H/W 一致。
    """

    def __init__(
        self,
        root: str,
        mode: str = "train",
        num_views: Optional[int] = None,
        max_view_num: Optional[int] = None,
        samples_per_scene: int = 1,
        min_views: int = 2,
        apply_perturbation: bool = True,
        synthetic_perturbation_in_eval: bool = False,
        perturb_cfg: Optional[PerturbationConfig] = None,
        cache_rpc: bool = True,
        cache_image: bool = False,
        cache_height: bool = False,
        base_seed: int = 0,
        reference_policy: str = "first",
        strict_same_hw: bool = True,
    ) -> None:
        """初始化 RPCSceneDataset。"""
        super().__init__()
        mode = mode.lower()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"mode must be one of train/val/test, got {mode}")

        self.root = root
        self.mode = mode
        # 兼容参数：优先使用 max_view_num；若未提供则退回 num_views。
        self.max_view_num = max_view_num if max_view_num is not None else num_views
        self.samples_per_scene = int(samples_per_scene)
        self.min_views = int(min_views)
        self.apply_perturbation = bool(apply_perturbation)
        self.synthetic_perturbation_in_eval = bool(synthetic_perturbation_in_eval)
        self.perturb_cfg = perturb_cfg or PerturbationConfig()

        self.cache_rpc = bool(cache_rpc)
        self.cache_image = bool(cache_image)
        self.cache_height = bool(cache_height)

        self.base_seed = int(base_seed)
        self.epoch = 0

        self.reference_policy = reference_policy
        if self.reference_policy != "first":
            raise ValueError("Current stage only supports reference_policy='first'")

        self.strict_same_hw = bool(strict_same_hw)

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

        # 简单缓存结构
        self.rpc_cache: dict[str, "RPCModelParameterTorch"] = {}
        self.image_cache: dict[str, torch.Tensor] = {}
        self.height_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于训练时改变确定性采样序列。"""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """返回数据集长度。

        规则:
        - train: len(scenes) * samples_per_scene
        - val/test: len(scenes)
        """
        if self.mode == "train":
            return len(self.scenes) * self.samples_per_scene
        return len(self.scenes)

    def _scene_from_index(self, index: int) -> SceneRecord:
        """根据 index 映射到 scene_record。"""
        if self.mode == "train":
            scene_idx = index // self.samples_per_scene
        else:
            scene_idx = index
        return self.scenes[scene_idx]

    def _select_views(self, scene_record: SceneRecord, rng) -> tuple[list[ViewRecord], int]:
        """按模式选择视图并返回参考视图索引。

        规则:
        - train: 返回该场景全部视图（排序后）；batch 内再统一随机抽取 K（1~max_view_num）。
        - val/test:
            - num_views is None -> 全视图排序后使用；
            - 否则取排序后的前 num_views；
          ref_view_idx=0。
        """
        views = list(scene_record.views)
        views.sort(key=lambda v: v.view_id)

        if self.mode == "train":
            return views, 0

        # val/test
        # eval 阶段也直接返回可用视图全集；batch 统一抽样逻辑在 collate_fn 中处理，
        # 若需要超过场景自身视图数，会通过重复采样补齐。
        selected = views if self.max_view_num is None else views
        return selected, 0

    def _load_view(self, view_record: ViewRecord) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, "RPCModelParameterTorch"]:
        """读取单视图 image/height/mask/rpc（带可选缓存）。"""
        if self.cache_image and view_record.image_path in self.image_cache:
            image = self.image_cache[view_record.image_path]
        else:
            image = read_image_tif(view_record.image_path)
            if self.cache_image:
                self.image_cache[view_record.image_path] = image

        if self.cache_height and view_record.height_path in self.height_cache:
            height_gt, height_mask = self.height_cache[view_record.height_path]
        else:
            height_gt, height_mask = read_height_tif(view_record.height_path)
            if self.cache_height:
                self.height_cache[view_record.height_path] = (height_gt, height_mask)

        if self.cache_rpc and view_record.rpc_path in self.rpc_cache:
            rpc_gt = copy.deepcopy(self.rpc_cache[view_record.rpc_path])
        else:
            rpc_gt = read_rpc_file(view_record.rpc_path)
            if self.cache_rpc:
                self.rpc_cache[view_record.rpc_path] = copy.deepcopy(rpc_gt)

        return image, height_gt, height_mask, rpc_gt

    def _infer_height_ref(self, rpc_views: Sequence["RPCModelParameterTorch"]) -> torch.Tensor:
        """从选中 RPC 列表推断 [V] 的 height_ref。"""
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        """生成单个 sample。

        主流程:
        1) index -> scene_record
        2) 构建确定性 rng
        3) 选视图
        4) 读取每视图数据
        5) 严格检查 H/W
        6) 组装 images/height_gt/mask
        7) 推断 height_ref
        8) 估计 scene_xy_center/scene_xy_scale
        9) 构造 rpc_init + affine 标签（按模式决定是否扰动）
        10) 返回可直接用于模型与训练的字段字典
        """
        scene_record = self._scene_from_index(index)
        scene_id = scene_record.scene_id

        rng = make_deterministic_rng(self.base_seed, self.epoch, scene_id, index)

        selected_views, ref_view_idx = self._select_views(scene_record, rng)

        images = []
        heights = []
        height_masks = []
        rpc_gt_views = []
        image_shapes = []

        for vr in selected_views:
            img, hgt, hmask, rpc_gt = self._load_view(vr)
            images.append(img)
            heights.append(hgt)
            height_masks.append(hmask)
            rpc_gt_views.append(rpc_gt)
            image_shapes.append((int(img.shape[-2]), int(img.shape[-1])))

        if self.strict_same_hw:
            h0, w0 = image_shapes[0]
            for i, (hi, wi) in enumerate(image_shapes[1:], start=1):
                if hi != h0 or wi != w0:
                    raise RuntimeError(
                        f"Scene {scene_id} selected views have different HW: "
                        f"view0=({h0},{w0}), view{i}=({hi},{wi})"
                    )

        images_t = torch.stack(images, dim=0).to(torch.float32)  # [V,3,H,W]
        height_gt_t = torch.stack(heights, dim=0).to(torch.float32)  # [V,1,H,W]
        height_mask_t = torch.stack(height_masks, dim=0).to(torch.float32)  # [V,1,H,W]

        height_ref = self._infer_height_ref(rpc_gt_views)  # [V]

        scene_xy_center, scene_xy_scale = estimate_scene_xy_center_scale(
            selected_views_rpc_gt=rpc_gt_views,
            selected_image_shapes=image_shapes,
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
            "images": images_t,  # [V,3,H,W]
            "height_gt": height_gt_t,  # [V,1,H,W]
            "height_valid_mask": height_mask_t,  # [V,1,H,W]
            "rpc_gt": rpc_gt_views,  # list[V]
            "rpc_init": rpc_init_views,  # list[V]
            "affine_gt_forward": aff_fwd.to(torch.float32),  # [V,2,3]
            "affine_gt_correction": aff_corr.to(torch.float32),  # [V,2,3]
            "height_ref": height_ref.to(torch.float32),  # [V]
            "scene_xy_center": scene_xy_center.to(torch.float32),  # [2], (y,x)
            "scene_xy_scale": scene_xy_scale.to(torch.float32),  # [2], (y,x)
            "ref_view_idx": int(ref_view_idx),  # 标量0
            "scene_id": int(scene_id),
            "view_ids": torch.tensor([v.view_id for v in selected_views], dtype=torch.long),
            "image_paths": [v.image_path for v in selected_views],
            "max_view_num": int(self.max_view_num) if self.max_view_num is not None else int(len(selected_views)),
        }
        return sample


def rpc_scene_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """自定义 collate_fn。

    功能:
    - stack 所有 tensor 字段；
    - 保留 rpc_gt/rpc_init 为嵌套对象列表；
    - 强制 batch 内 V、H、W 一致（否则报错）。
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    h0 = batch[0]["images"].shape[-2]
    w0 = batch[0]["images"].shape[-1]
    for i, sample in enumerate(batch[1:], start=1):
        hi = sample["images"].shape[-2]
        wi = sample["images"].shape[-1]
        if hi != h0 or wi != w0:
            raise RuntimeError(f"Batch has inconsistent HW: sample0=({h0},{w0}), sample{i}=({hi},{wi})")

    max_view_num = int(min(x.get("max_view_num", x["images"].shape[0]) for x in batch))
    if max_view_num <= 0:
        raise RuntimeError("max_view_num must be > 0 for collate")

    # 一个 batch 内统一随机选 K；不同 batch 会随机得到不同 K。
    k = int(torch.randint(low=1, high=max_view_num + 1, size=(1,)).item())

    def _pick_indices(v: int) -> torch.Tensor:
        # 参考视图固定为第 0 个；其余视图用于补足 K-1。
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

    out = {
        "images": torch.stack(images_b, dim=0).to(torch.float32),  # [B,K,3,H,W]
        "height_gt": torch.stack(height_b, dim=0).to(torch.float32),  # [B,K,1,H,W]
        "height_valid_mask": torch.stack(mask_b, dim=0).to(torch.float32),  # [B,K,1,H,W]
        "affine_gt_forward": torch.stack(aff_fwd_b, dim=0).to(torch.float32),  # [B,K,2,3]
        "affine_gt_correction": torch.stack(aff_cor_b, dim=0).to(torch.float32),  # [B,K,2,3]
        "height_ref": torch.stack(href_b, dim=0).to(torch.float32),  # [B,K]
        "scene_xy_center": torch.stack(scene_center_b, dim=0).to(torch.float32),  # [B,2]
        "scene_xy_scale": torch.stack(scene_scale_b, dim=0).to(torch.float32),  # [B,2]
        "rpc_gt": rpc_gt_b,  # list[B][K]
        "rpc_init": rpc_init_b,  # list[B][K]
        "ref_view_idx": torch.tensor(ref_idx_b, dtype=torch.long),  # [B]
        "scene_id": torch.tensor(scene_id_b, dtype=torch.long),  # [B]
        "view_ids": torch.stack(view_ids_b, dim=0).to(torch.long),  # [B,K]
        "image_paths": image_paths_b,
    }
    return out


def build_dataset(mode: str, **kwargs) -> RPCSceneDataset:
    """便捷构建函数。

    参数:
        mode: train / val / test。
        **kwargs: 透传给 RPCSceneDataset 初始化参数。

    返回:
        RPCSceneDataset 实例。
    """
    return RPCSceneDataset(mode=mode, **kwargs)

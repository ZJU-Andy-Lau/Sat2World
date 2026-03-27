"""dataset.io

本文件负责数据集的磁盘扫描、TIFF/RPC 读取与场景级几何归一化参数估计。

关键约定（必须与训练/模型方向一致）：
1. 磁盘上的 `rpc_{scene_id}_{view_id}.txt` 是 GT 已平差 RPC（正确 RPC）。
2. dataset 只负责读取与组织输入/标签，不在此处计算 point_gt 或渲染监督。
3. scene_xy_center / scene_xy_scale 坐标顺序固定为 (y, x)。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from geometry.rpc import RPCModelParameterTorch


@dataclass(frozen=True)
class ViewRecord:
    """单视角数据记录。

    字段说明：
    - scene_id / view_id: 场景与视图整数 ID。
    - image_path: RGB 影像路径。
    - height_path: 高程真值路径。
    - rpc_path: 该视图 GT 已平差 RPC 路径。
    """

    scene_id: int
    view_id: int
    image_path: str
    height_path: str
    rpc_path: str


@dataclass(frozen=True)
class SceneRecord:
    """单场景数据记录。

    字段说明：
    - scene_id: 场景 ID。
    - views: 按 view_id 升序排列的视图记录列表。
    """

    scene_id: int
    views: list[ViewRecord]


def _match_dir_name(name: str, prefix: str) -> Optional[int]:
    """匹配 `prefix_{id}` 目录命名并返回 id；不匹配返回 None。"""
    m = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", name)
    if m is None:
        return None
    return int(m.group(1))


def _select_unique_file(view_dir: Path, pattern: re.Pattern[str], desc: str) -> Path:
    """在目录中根据正则匹配唯一文件，否则抛出清晰错误。"""
    matches = [p for p in view_dir.iterdir() if p.is_file() and pattern.fullmatch(p.name)]
    if len(matches) == 0:
        raise FileNotFoundError(f"[{desc}] not found in {view_dir}")
    if len(matches) > 1:
        names = ", ".join(sorted(p.name for p in matches))
        raise RuntimeError(f"[{desc}] expected unique match in {view_dir}, but got: {names}")
    return matches[0]


def scan_dataset_root(root_dir: str | os.PathLike[str]) -> list[SceneRecord]:
    """扫描数据根目录并构建结构化元信息。

    目录规则（严格校验）：
    - 根目录下为 `scene_{scene_id}`。
    - 场景目录下为 `view_{view_id}`。
    - 每个视图目录中必须存在并命名匹配：
      - `image_{scene_id}_{view_id}.tif`
      - `height_{scene_id}_{view_id}.tif`
      - `rpc_{scene_id}_{view_id}.txt`

    参数:
        root_dir: 数据集根目录（例如 train_data 或 test_data）。

    返回:
        scenes: SceneRecord 列表，按 scene_id 升序；每个场景内 views 按 view_id 升序。
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found or not a directory: {root}")

    scenes: list[SceneRecord] = []
    scene_dirs: list[tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        sid = _match_dir_name(p.name, "scene")
        if sid is not None:
            scene_dirs.append((sid, p))

    scene_dirs.sort(key=lambda x: x[0])
    if len(scene_dirs) == 0:
        raise RuntimeError(f"No scene_* directories found under: {root}")

    for scene_id, scene_dir in scene_dirs:
        view_dirs: list[tuple[int, Path]] = []
        for p in scene_dir.iterdir():
            if not p.is_dir():
                continue
            vid = _match_dir_name(p.name, "view")
            if vid is not None:
                view_dirs.append((vid, p))
        view_dirs.sort(key=lambda x: x[0])

        if len(view_dirs) == 0:
            raise RuntimeError(f"Scene {scene_id} has no valid view_* directories: {scene_dir}")

        views: list[ViewRecord] = []
        for view_id, view_dir in view_dirs:
            image_pat = re.compile(rf"image_{scene_id}_{view_id}\.tif")
            height_pat = re.compile(rf"height_{scene_id}_{view_id}\.tif")
            rpc_pat = re.compile(rf"rpc_{scene_id}_{view_id}\.txt")

            image_path = _select_unique_file(view_dir, image_pat, "image tif")
            height_path = _select_unique_file(view_dir, height_pat, "height tif")
            rpc_path = _select_unique_file(view_dir, rpc_pat, "rpc txt")

            views.append(
                ViewRecord(
                    scene_id=scene_id,
                    view_id=view_id,
                    image_path=str(image_path),
                    height_path=str(height_path),
                    rpc_path=str(rpc_path),
                )
            )

        scenes.append(SceneRecord(scene_id=scene_id, views=views))

    return scenes


def _read_tif_with_rasterio(path: str) -> tuple[np.ndarray, Optional[float]]:
    """用 rasterio 读取 tif，返回数组与 nodata。"""
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read()  # [C,H,W]
        nodata = ds.nodata
    return arr, nodata


def _read_tif_with_tifffile(path: str) -> tuple[np.ndarray, Optional[float]]:
    """用 tifffile 读取 tif，返回数组与 nodata(None)。"""
    import tifffile

    arr = tifffile.imread(path)
    return arr, None


def _read_tif(path: str) -> tuple[np.ndarray, Optional[float]]:
    """按优先级读取 tif：rasterio -> tifffile。"""
    try:
        return _read_tif_with_rasterio(path)
    except Exception:
        return _read_tif_with_tifffile(path)


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """将输入数组转换为 CHW 形式。"""
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim={arr.ndim}, expect 2 or 3")

    # 常见情况：rasterio -> CHW；tifffile -> HWC
    if arr.shape[0] <= 8 and arr.shape[1] > 8 and arr.shape[2] > 8:
        return arr
    return np.transpose(arr, (2, 0, 1))


def read_image_tif(path: str, image_scale_mode: str = "dtype") -> torch.Tensor:
    """读取 RGB 影像 tif 并输出 [3,H,W] float32。

    参数:
        path: tif 路径。
        image_scale_mode: 默认 "dtype"。
            - "dtype": 整型按 dtype 最大值缩放到 [0,1]；浮点保持原值。

    返回:
        image: torch.float32, [3,H,W]。
    """
    arr, _ = _read_tif(path)
    arr = _to_chw(arr)

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] >= 3:
        arr = arr[:3]
    else:
        raise ValueError(f"Invalid channel number in image: {arr.shape[0]} at {path}")

    if image_scale_mode == "dtype":
        if np.issubdtype(arr.dtype, np.integer):
            maxv = np.iinfo(arr.dtype).max
            if maxv <= 0:
                raise ValueError(f"Invalid integer dtype max value: {arr.dtype}")
            arr = arr.astype(np.float32) / float(maxv)
        else:
            arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image_scale_mode: {image_scale_mode}")

    if not np.isfinite(arr).all():
        raise ValueError(f"Image contains NaN/Inf: {path}")

    return torch.from_numpy(arr).to(torch.float32)


def read_height_tif(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """读取高程 tif，返回高程与有效掩码。

    参数:
        path: 高程 tif 路径。

    返回:
        height: float32, [1,H,W]，保留绝对高程值，不做归一化；
                对 NaN/Inf/nodata 无效位置使用“有效像素平均高程”填充。
        valid_mask: float32, [1,H,W]，有效为1，无效为0（后续 loss 可据此屏蔽）。
    """
    arr, nodata = _read_tif(path)
    arr = _to_chw(arr)

    # 若多通道高程，按约定使用第一个通道。
    h = arr[0].astype(np.float32)

    finite_mask = np.isfinite(h)
    if nodata is None:
        valid = finite_mask
    else:
        if np.isnan(nodata):
            valid = finite_mask & (~np.isnan(h))
        else:
            valid = finite_mask & (h != float(nodata))

    if valid.any():
        mean_h = float(h[valid].mean())
    else:
        mean_h = 0.0
    h = np.where(valid, h, mean_h).astype(np.float32)

    height = torch.from_numpy(h[None, ...]).to(torch.float32)
    valid_mask = torch.from_numpy(valid.astype(np.float32)[None, ...]).to(torch.float32)
    return height, valid_mask


def read_rpc_file(path: str) -> "RPCModelParameterTorch":
    """读取 RPC 文本并返回 RPCModelParameterTorch。

    说明:
        该函数直接复用 geometry.rpc 的 load_rpc，不重复实现 RPC 文本解析逻辑。
    """
    from geometry.rpc import load_rpc

    return load_rpc(path, to_gpu=False)


def estimate_scene_xy_center_scale(
    selected_views_rpc_gt: Sequence["RPCModelParameterTorch"],
    selected_image_shapes: Sequence[tuple[int, int]],
    selected_height_ref: Sequence[float] | torch.Tensor,
    *,
    safety_factor: float = 1.05,
    min_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """估计当前 sample 的统一 scene_xy_center 与 scene_xy_scale（顺序 y,x）。

    算法:
    - 对每个视图采样 5 个像素点：中心 + 4角；
    - 在该视图 h_ref 高度上，调用 RPC_LINESAMP2XY 得到 x,y；
    - 统一转为 y,x 后聚合；
    - center = 均值；
    - scale = 相对 center 的最大绝对偏移 * safety_factor，并下限到 min_scale。

    参数:
        selected_views_rpc_gt: 当前 sample 选中的 GT RPC 列表。
        selected_image_shapes: 每个视图的 (H,W)。
        selected_height_ref: 每个视图的 h_ref。
        safety_factor: 缩放安全系数。
        min_scale: scale 下限。

    返回:
        scene_xy_center: float32, [2]，顺序 (y,x)
        scene_xy_scale: float32, [2]，顺序 (y,x)
    """
    if len(selected_views_rpc_gt) != len(selected_image_shapes):
        raise ValueError("selected_views_rpc_gt and selected_image_shapes length mismatch")

    if torch.is_tensor(selected_height_ref):
        h_refs = selected_height_ref.detach().cpu().tolist()
    else:
        h_refs = list(selected_height_ref)

    if len(h_refs) != len(selected_views_rpc_gt):
        raise ValueError("selected_height_ref length mismatch")

    all_yx: list[torch.Tensor] = []
    for rpc_obj, (h, w), h_ref in zip(selected_views_rpc_gt, selected_image_shapes, h_refs):
        # line,samp: center + 4 corners
        pts = torch.tensor(
            [
                [0.5 * (h - 1), 0.5 * (w - 1)],
                [0.0, 0.0],
                [0.0, float(w - 1)],
                [float(h - 1), 0.0],
                [float(h - 1), float(w - 1)],
            ],
            dtype=torch.double,
            device=rpc_obj.device,
        )
        line = pts[:, 0]
        samp = pts[:, 1]
        h_tensor = torch.full((pts.shape[0],), float(h_ref), dtype=torch.double, device=rpc_obj.device)

        x, y = rpc_obj.RPC_LINESAMP2XY(line_in=line, samp_in=samp, h_in=h_tensor, output_type="tensor")
        yx = torch.stack([y, x], dim=-1).to(torch.float32).cpu()
        all_yx.append(yx)

    yx_all = torch.cat(all_yx, dim=0)  # [M,2]
    center = yx_all.mean(dim=0)
    offset = (yx_all - center[None, :]).abs().amax(dim=0)
    scale = torch.clamp(offset * float(safety_factor), min=float(min_scale))
    return center.to(torch.float32), scale.to(torch.float32)

"""geometry 包导出。

本模块只做轻量导出，避免复杂业务逻辑混入。
"""

from .rpc_geometry import RPCGeometryOps
from .scene_geometry import (
    enforce_reference_affine_identity,
    expand_height_ref_map,
    infer_height_ref,
    infer_height_ref_batch,
    make_image_grid,
    make_patch_centers,
    reshape_bv_to_bvchw,
    reshape_bv_to_bvn,
)

__all__ = [
    "RPCGeometryOps",
    "make_image_grid",
    "make_patch_centers",
    "infer_height_ref",
    "infer_height_ref_batch",
    "expand_height_ref_map",
    "enforce_reference_affine_identity",
    "reshape_bv_to_bvchw",
    "reshape_bv_to_bvn",
]

"""轻量全链路自检脚本。

目标：在无真实数据/无DINO权重时，验证工程链路是否闭环：
config -> model -> renderer -> objective -> backward。
"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _DummyDINO(nn.Module):
    def __init__(self, patch_size: int = 16, embed_dim: int = 1024) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = self.proj(x)
        b, c, gh, gw = y.shape
        tokens = y.flatten(2).transpose(1, 2).contiguous()
        return {"x_norm_patchtokens": tokens}


def _build_real_batch_from_val(cfg: dict[str, Any], scene_id: int | None, val_index: int) -> dict[str, Any]:
    from dataset import build_dataset, rpc_scene_collate_fn

    data_cfg = cfg.get("data", {})
    val_cfg = data_cfg.get("val", {})
    if not isinstance(val_cfg, dict) or len(val_cfg) == 0:
        raise RuntimeError("Missing data.val config; cannot load real validation scene.")

    val_dataset = build_dataset(mode="val", **val_cfg)
    if len(val_dataset) <= 0:
        raise RuntimeError("Validation dataset is empty.")

    if scene_id is None:
        idx = int(max(min(val_index, len(val_dataset) - 1), 0))
    else:
        idx = -1
        for i, rec in enumerate(getattr(val_dataset, "scenes", [])):
            if int(getattr(rec, "scene_id", -1)) == int(scene_id):
                idx = i
                break
        if idx < 0:
            raise RuntimeError(f"Requested scene_id={scene_id} not found in val dataset.")

    sample = val_dataset[idx]
    batch = rpc_scene_collate_fn([sample])
    return batch


def _pick_valid_pixel(mask_2d: torch.Tensor) -> tuple[int, int]:
    h, w = int(mask_2d.shape[-2]), int(mask_2d.shape[-1])
    cy, cx = h // 2, w // 2
    if bool(mask_2d[cy, cx] > 0.5):
        return cy, cx
    valid = (mask_2d > 0.5).nonzero(as_tuple=False)
    if valid.numel() == 0:
        return cy, cx
    return int(valid[0, 0].item()), int(valid[0, 1].item())


def _sample_map_bilinear(map_2d: torch.Tensor, line: torch.Tensor, samp: torch.Tensor) -> torch.Tensor:
    """从 [H,W] 地图按 (line,samp) 双线性采样，返回 [N]。"""
    h, w = int(map_2d.shape[-2]), int(map_2d.shape[-1])
    x = (samp / max(w - 1, 1)) * 2.0 - 1.0
    y = (line / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    src = map_2d.view(1, 1, h, w).to(dtype=torch.float32)
    out = torch.nn.functional.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.view(-1)


def main() -> None:
    parser = argparse.ArgumentParser("Sat2World sanity check")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--scene-id", type=int, default=None, help="验证集 scene_id；不传则按 val-index 取样。")
    parser.add_argument("--val-index", type=int, default=0, help="当未指定 scene-id 时，使用 val dataset 的样本索引。")
    parser.add_argument("--use-dummy-backbone", action="store_true", help="启用 dummy backbone，避免依赖真实 DINO 权重。")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = str(cfg.get("train", {}).get("amp_dtype", "fp16")).lower()
    if device.type == "cuda" and amp_dtype == "bf16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    elif device.type == "cuda" and amp_dtype == "fp16":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    else:
        autocast_ctx = contextlib.nullcontext()

    use_scaler = bool(cfg.get("train", {}).get("enable_grad_scaler", True)) and amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    old_hub_load = torch.hub.load
    if args.use_dummy_backbone:
        # monkeypatch torch.hub.load，避免依赖真实 DINO 权重。
        torch.hub.load = lambda *a, **k: _DummyDINO(patch_size=16, embed_dim=1024)

    try:
        from engine.distributed import move_batch_to_device
        from scripts.train import build_model, build_renderer, build_objective
        from render.rpc_gaussian_renderer import project_gaussians_to_view

        model = build_model(cfg).to(device)
        renderer = build_renderer(cfg, model.rpc_ops)
        objective = build_objective(cfg, model.rpc_ops, model.patch_matcher)
        model.train()

        batch = _build_real_batch_from_val(cfg, scene_id=args.scene_id, val_index=args.val_index)
        batch = move_batch_to_device(batch, device)
        with autocast_ctx:
            outputs = model(batch)
            if "affine_coarse" in outputs:
                raise RuntimeError("Single-stage model should not output affine_coarse.")
            required_keys = [
                "affine_pred",
                "rpc_corrected",
                "height_abs",
                "gaussian_centers_rpc",
                "gaussian_centers_point",
            ]
            missing = [k for k in required_keys if k not in outputs]
            if missing:
                raise RuntimeError(f"Missing required single-stage outputs: {missing}")
            render_outputs = renderer.render_paths(outputs, batch, mode="train")
            total_loss, scalar_dict, aux_dict = objective(outputs, batch, global_step=0, epoch=0, render_outputs=render_outputs, mode="train")

        # ------------------------------------------------------------------
        # 额外 3DGS 渲染几何正确性检查：
        # 1) 选取 source 视图一个像素，用 rpc_gt + h_gt 构造物方点；
        # 2) 投影到 target 视图，检查投影像素误差；
        # 3) 用 0.5m 高斯尺度估计投影椭圆主轴像素长度，检查是否在合理范围。
        # ------------------------------------------------------------------
        bi = 0
        v_total = int(batch["images"].shape[1])
        if v_total < 2:
            raise RuntimeError(f"Need at least 2 views for sanity projection check, got V={v_total}")
        vi_src, vi_tgt = 0, 1
        h_img = int(batch["images"].shape[-2])
        w_img = int(batch["images"].shape[-1])
        src_mask = batch["height_valid_mask"][bi, vi_src, 0]
        py, px = _pick_valid_pixel(src_mask)
        line0 = torch.tensor([float(py)], dtype=torch.float32, device=device)
        samp0 = torch.tensor([float(px)], dtype=torch.float32, device=device)
        h0 = batch["height_gt"][bi, vi_src, 0, py, px].view(1).to(torch.float32)

        rpc_src = batch["rpc_gt"][bi][vi_src]
        rpc_tgt = batch["rpc_gt"][bi][vi_tgt]
        scene_center = batch["scene_xy_center"][bi]
        # 3DGS 渲染语义：局部米制坐标，投影时只做 offset，不做 scale。
        scene_scale_render = torch.ones_like(scene_center)

        with torch.no_grad():
            x_obj, y_obj = rpc_src.RPC_LINESAMP2XY(
                line_in=line0.to(dtype=torch.double),
                samp_in=samp0.to(dtype=torch.double),
                h_in=h0.to(dtype=torch.double),
                output_type="tensor",
                xy_center=scene_center.to(dtype=torch.double),
                xy_scale=scene_scale_render.to(dtype=torch.double),
            )
            line1, samp1 = rpc_tgt.RPC_XY2LINESAMP(
                x_in=x_obj,
                y_in=y_obj,
                h_in=h0.to(dtype=torch.double),
                output_type="tensor",
                xy_center=scene_center.to(dtype=torch.double),
                xy_scale=scene_scale_render.to(dtype=torch.double),
            )
            # 关键修复：不能将 target 投影像素与 source 像素直接比较。
            # 改为：
            # 1) src(h_gt) -> 3D -> tgt 得到预测像素 (line1,samp1)
            # 2) 在 tgt 视图该预测位置双线性采样 h_gt
            # 3) tgt(h_gt) 反投影回 3D，再投回 src，比较 src 回投误差
            # 这样同时使用 rpc_gt 与双视图 h_gt 进行一致性校验。
            h_tgt = _sample_map_bilinear(batch["height_gt"][bi, vi_tgt, 0], line1.to(torch.float32), samp1.to(torch.float32)).to(torch.double)
            x_tgt, y_tgt = rpc_tgt.RPC_LINESAMP2XY(
                line_in=line1,
                samp_in=samp1,
                h_in=h_tgt,
                output_type="tensor",
                xy_center=scene_center.to(dtype=torch.double),
                xy_scale=scene_scale_render.to(dtype=torch.double),
            )
            line_back, samp_back = rpc_src.RPC_XY2LINESAMP(
                x_in=x_tgt,
                y_in=y_tgt,
                h_in=h_tgt,
                output_type="tensor",
                xy_center=scene_center.to(dtype=torch.double),
                xy_scale=scene_scale_render.to(dtype=torch.double),
            )
            proj_err = torch.sqrt((line_back - line0.to(dtype=torch.double)).square() + (samp_back - samp0.to(dtype=torch.double)).square())
            proj_err_px = float(proj_err.detach().cpu().item())
            if not math.isfinite(proj_err_px) or proj_err_px > 2.0:
                raise RuntimeError(f"3DGS projection sanity failed: src->tgt->src consistency error too large ({proj_err_px:.6f}px)")

            centers_world = torch.stack(
                [
                    x_obj.to(dtype=torch.float32),
                    y_obj.to(dtype=torch.float32),
                    h0.to(dtype=torch.float32),
                ],
                dim=-1,
            ).view(1, 3)
            scale_05m = torch.full((1, 3), 0.5, dtype=torch.float32, device=device)
            rot_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
            mean_2d, cov_2d = project_gaussians_to_view(
                model.rpc_ops,
                centers_world,
                scale_05m,
                rot_identity,
                rpc_tgt,
                scene_center,
                scene_scale_render,
                eps_xy_fd=float(cfg.get("renderer", {}).get("eps_xy_fd", 1e-2)),
                eps_h_fd=float(cfg.get("renderer", {}).get("eps_h_fd", 1e-1)),
                eps_cov=float(cfg.get("renderer", {}).get("eps_cov", 1e-4)),
            )
            eig = torch.linalg.eigvalsh(cov_2d[0].to(torch.float64))
            sigma_major_px = float(torch.sqrt(eig.max().clamp_min(0.0)).detach().cpu().item())
            if not math.isfinite(sigma_major_px):
                raise RuntimeError("3DGS Gaussian size sanity failed: projected sigma is not finite.")

        if use_scaler:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        print("[sanity_check] success")
        print("loss_total:", float(total_loss.detach().cpu().item()))
        print("outputs keys:", sorted(list(outputs.keys()))[:12], "...")
        print("render rpc targets:", int(render_outputs["rpc"].get("num_targets", 0)))
        print("render point targets:", int(render_outputs["point"].get("num_targets", 0)))
        print("scene_id:", int(batch["scene_id"][bi].detach().cpu().item()))
        print("view_ids:", [int(x) for x in batch["view_ids"][bi].detach().cpu().tolist()])
        print("sampled source pixel(line,samp):", (float(line0.item()), float(samp0.item())))
        print("rpc reprojection error(px):", proj_err_px)
        print("projected gaussian major sigma(px) for 0.5m sphere:", sigma_major_px)
        print("scalar sample:", {k: scalar_dict[k] for k in list(scalar_dict.keys())[:8]})
        print("aux keys:", list(aux_dict.keys()))
    finally:
        if args.use_dummy_backbone:
            torch.hub.load = old_hub_load


if __name__ == "__main__":
    main()

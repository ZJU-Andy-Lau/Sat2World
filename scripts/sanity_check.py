"""轻量全链路自检脚本。

目标：在无真实数据/无DINO权重时，验证工程链路是否闭环：
config -> model -> renderer -> objective -> backward。
"""

from __future__ import annotations

import argparse
import copy
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


def _make_rpc_identity_like(device: torch.device) -> Any:
    from geometry.rpc import RPCModelParameterTorch

    data = torch.zeros(170, dtype=torch.double, device=device)
    # offsets / scales
    data[5:10] = 1.0  # LINE/SAMP/LAT/LON/HEIGHT scales
    # direct denominators
    data[30] = 1.0
    data[70] = 1.0
    # inverse denominators
    data[110] = 1.0
    data[150] = 1.0
    # simple linear numerators
    data[11] = 1.0  # L coefficient to line
    data[52] = 1.0  # P coefficient to samp
    data[91] = 1.0  # line to lat
    data[132] = 1.0  # samp to lon
    return RPCModelParameterTorch(data)


def build_synthetic_batch(device: torch.device) -> dict[str, Any]:
    b, v, h, w = 1, 2, 32, 32
    images = torch.rand(b, v, 3, h, w, device=device)
    height_gt = torch.rand(b, v, 1, h, w, device=device) * 20.0
    mask = torch.ones_like(height_gt)
    eye = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 2, 3).repeat(b, v, 1, 1)

    rpc_gt: list[list[Any]] = []
    rpc_init: list[list[Any]] = []
    for _ in range(b):
        g, i = [], []
        for _ in range(v):
            r = _make_rpc_identity_like(device=device)
            g.append(r)
            i.append(copy.deepcopy(r))
        rpc_gt.append(g)
        rpc_init.append(i)

    batch = {
        "images": images,
        "height_gt": height_gt,
        "height_valid_mask": mask,
        "rpc_gt": rpc_gt,
        "rpc_init": rpc_init,
        "affine_gt_forward": eye,
        "affine_gt_correction": eye,
        "height_ref": torch.zeros(b, v, device=device),
        "scene_xy_center": torch.zeros(b, 2, device=device),
        "scene_xy_scale": torch.ones(b, 2, device=device),
        "ref_view_idx": torch.zeros(b, dtype=torch.long, device=device),
        "scene_id": torch.zeros(b, dtype=torch.long, device=device),
        "view_ids": torch.tensor([[0, 1]], dtype=torch.long, device=device),
    }
    return batch


def main() -> None:
    parser = argparse.ArgumentParser("Sat2World sanity check")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # monkeypatch torch.hub.load，避免依赖真实 DINO 权重。
    old_hub_load = torch.hub.load
    torch.hub.load = lambda *a, **k: _DummyDINO(patch_size=16, embed_dim=1024)

    try:
        from scripts.train import build_model, build_renderer, build_objective

        model = build_model(cfg).to(device)
        renderer = build_renderer(cfg, model.rpc_ops)
        objective = build_objective(cfg, model.rpc_ops)

        batch = build_synthetic_batch(device)
        outputs = model(batch)
        if "affine_coarse" in outputs:
            raise RuntimeError("Single-stage model should not output affine_coarse.")
        required_keys = [
            "affine_pred",
            "rpc_corrected",
            "height_abs",
            "point_abs",
            "gaussian_centers_rpc",
            "gaussian_centers_point",
        ]
        missing = [k for k in required_keys if k not in outputs]
        if missing:
            raise RuntimeError(f"Missing required single-stage outputs: {missing}")
        render_outputs = renderer.render_paths(outputs, batch, mode="train")
        total_loss, scalar_dict, aux_dict = objective(outputs, batch, global_step=0, epoch=0, render_outputs=render_outputs, mode="train")

        total_loss.backward()

        print("[sanity_check] success")
        print("loss_total:", float(total_loss.detach().cpu().item()))
        print("outputs keys:", sorted(list(outputs.keys()))[:12], "...")
        print("render rpc targets:", int(render_outputs["rpc"].get("num_targets", 0)))
        print("render point targets:", int(render_outputs["point"].get("num_targets", 0)))
        print("scalar sample:", {k: scalar_dict[k] for k in list(scalar_dict.keys())[:8]})
        print("aux keys:", list(aux_dict.keys()))
    finally:
        torch.hub.load = old_hub_load


if __name__ == "__main__":
    main()

"""Sat2World 新渲染链路检查脚本（RPC->虚拟pinhole->CUDA栅格）。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import build_dataset, rpc_scene_collate_fn
from render.rpc_gaussian_renderer import RPCGaussianRenderer, RPCGaussianRendererCfg


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


class ProgressLogger:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.step = 0

    def log(self, message: str) -> None:
        self.step += 1
        now = time.perf_counter()
        print(
            f"[render_check][step={self.step:02d}] {message} | step={now - self.t_last:.2f}s total={now - self.t0:.2f}s",
            flush=True,
        )
        self.t_last = now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World virtual-pinhole render checker")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--view-i", type=int, default=0)
    p.add_argument("--view-j", type=int, default=1)
    p.add_argument("--fit-p95-threshold", type=float, default=1.0)
    p.add_argument("--fit-max-threshold", type=float, default=3.0)
    p.add_argument("--render-time-threshold-sec", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_val_batch(cfg: dict[str, Any], scene_index: int) -> dict[str, Any]:
    ds = build_dataset(mode="val", **cfg.get("data", {}).get("val", {}))
    if len(ds) == 0:
        raise RuntimeError("val dataset empty")
    if not (0 <= scene_index < len(ds)):
        raise ValueError(f"scene_index out of range: {scene_index}/{len(ds)}")
    sample = ds[scene_index]
    return rpc_scene_collate_fn([sample])


def to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from engine.distributed import move_batch_to_device

    return move_batch_to_device(batch, device)


def make_renderer(cfg: dict[str, Any]) -> RPCGaussianRenderer:
    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)
    # 检查时采用较密拟合
    rcfg.fit_grid_nx = max(int(getattr(rcfg, "fit_grid_nx", 24)), 24)
    rcfg.fit_grid_ny = max(int(getattr(rcfg, "fit_grid_ny", 24)), 24)
    rcfg.fit_grid_nz = max(int(getattr(rcfg, "fit_grid_nz", 7)), 7)
    from geometry import RPCGeometryOps

    return RPCGaussianRenderer(RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32), rcfg)


def build_synthetic_outputs(batch: dict[str, Any], vi: int, sh_dim: int = 48) -> dict[str, Any]:
    b, v, _, h, w = batch["height_gt"].shape
    dev = batch["images"].device

    centers_rpc = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)
    centers_point = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)

    # 构造一个规则局部米制网格作为测试中心
    ys = torch.linspace(-100.0, 100.0, h, device=dev)
    xs = torch.linspace(-100.0, 100.0, w, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gz = batch["height_gt"][0, vi, 0]
    centers_rpc[0, vi, 0] = gx
    centers_rpc[0, vi, 1] = gy
    centers_rpc[0, vi, 2] = gz
    centers_point.copy_(centers_rpc)

    opacity = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    opacity[0, vi] = 1.0

    scale = torch.full((b, v, 3, h, w), 0.5, device=dev, dtype=torch.float32)

    rotation = torch.zeros((b, v, 4, h, w), device=dev, dtype=torch.float32)
    rotation[:, :, 0] = 1.0

    sh = torch.zeros((b, v, sh_dim, h, w), device=dev, dtype=torch.float32)
    sh[0, vi, 0:3] = 1.0

    conf = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    conf[0, vi] = 1.0

    return {
        "gaussian_centers_rpc": centers_rpc,
        "gaussian_centers_point": centers_point,
        "gaussian_opacity": opacity,
        "gaussian_scale": scale,
        "gaussian_rotation": rotation,
        "gaussian_sh": sh,
        "gaussian_confidence_rpc": conf,
        "gaussian_confidence_point": conf,
        "rpc_corrected": batch["rpc_gt"],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = load_cfg(args.config)
    prog = ProgressLogger()

    prog.log("build batch")
    batch = to_device_batch(build_val_batch(cfg, args.scene_index), device)
    b, v, _, h, w = batch["images"].shape
    if b != 1:
        raise RuntimeError("render_check currently supports batch_size=1")
    if not (0 <= args.view_i < v and 0 <= args.view_j < v):
        raise ValueError(f"invalid view ids: i={args.view_i}, j={args.view_j}, V={v}")

    prog.log("build renderer")
    renderer = make_renderer(cfg)

    prog.log("fit virtual camera")
    cam = renderer.fit_virtual_camera_for_target(batch, 0, args.view_j)

    checks: list[CheckResult] = [
        CheckResult("fit_p95_px", cam.fit_p95, args.fit_p95_threshold, cam.fit_p95 <= args.fit_p95_threshold),
        CheckResult("fit_max_px", cam.fit_max, args.fit_max_threshold, cam.fit_max <= args.fit_max_threshold),
    ]

    prog.log("prepare synthetic outputs")
    outputs = build_synthetic_outputs(batch, args.view_i)

    prog.log("render paths")
    t0 = time.perf_counter()
    render_out = renderer.render_paths(outputs, batch, mode="val", global_step=0, epoch=0)
    dt = time.perf_counter() - t0
    checks.append(CheckResult("render_time_sec", dt, args.render_time_threshold_sec, dt <= args.render_time_threshold_sec))

    rpc_path = render_out["rpc"]
    point_path = render_out["point"]
    if int(rpc_path.get("num_targets", 0)) <= 0 or int(point_path.get("num_targets", 0)) <= 0:
        checks.append(CheckResult("num_targets", 0.0, 1.0, False, "render output has no targets"))

    print("\n================ Render Check Summary ================")
    print(f"scene_index={args.scene_index} view_i={args.view_i} view_j={args.view_j}")
    print(f"image_hw={h}x{w}")
    print(f"fit: p50={cam.fit_p50:.4f}px p95={cam.fit_p95:.4f}px max={cam.fit_max:.4f}px")
    print(f"render_time={dt:.4f}s")

    all_pass = True
    for r in checks:
        status = "PASS" if r.passed else "FAIL"
        extra = f" ({r.note})" if r.note else ""
        print(f"[{status}] {r.name}: value={r.value:.6f}, threshold={r.threshold:.6f}{extra}")
        all_pass = all_pass and r.passed

    print("------------------------------------------------------")
    print("RESULT:", "PASS" if all_pass else "FAIL")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

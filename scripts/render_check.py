"""Sat2World 3DGS 全链路检查脚本。

目标：
1) 在 val 集单场景上构造“可控 synthetic 高斯”；
2) 直接调用现有 RPCGaussianRenderer 渲染链路；
3) 检查中心投影、像面尺度、可见性(深度排序)三类关键正确性。

说明：
- 本脚本不依赖模型权重，不调用 Sat2World 主模型前向；
- 重点测试 render 链路是否与当前坐标/几何约定自洽；
- 默认采用“全像素 anchor”策略（view_i 的每个像素都作为高斯中心来源）。
"""

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
from dataset.io import raw_xy_to_linesamp
from render.rpc_gaussian_renderer import (
    RPCGaussianRenderer,
    RPCGaussianRendererCfg,
    compute_depth_proxy,
    estimate_local_view_direction,
    project_gaussians_to_view,
)


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


class ProgressLogger:
    """轻量进度日志器，打印步骤编号与耗时。"""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.t_last = self.t0
        self.step = 0

    def log(self, message: str) -> None:
        self.step += 1
        now = time.perf_counter()
        step_sec = now - self.t_last
        total_sec = now - self.t0
        print(f"[render_check][step={self.step:02d}] {message} | step={step_sec:.2f}s total={total_sec:.2f}s", flush=True)
        self.t_last = now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sat2World 3DGS render chain checker")
    p.add_argument("--config", type=str, default="config/default.yaml", help="配置文件路径")
    p.add_argument("--scene-index", type=int, default=0, help="val dataset 索引")
    p.add_argument("--view-i", type=int, default=0, help="source 视图索引")
    p.add_argument("--view-j", type=int, default=1, help="target 视图索引")
    p.add_argument(
        "--point-center-mode",
        type=str,
        default="normalized",
        choices=["normalized", "local_meter"],
        help=(
            "point path 输入中心语义："
            "normalized=按(point_abs)归一化语义直接喂入；"
            "local_meter=按局部米制语义喂入"
        ),
    )
    p.add_argument("--radius-meter", type=float, default=0.5, help="高斯球体半径(米)，用于scale三轴")
    p.add_argument("--center-threshold-p50", type=float, default=0.5, help="中心误差P50阈值(px)")
    p.add_argument("--center-threshold-p95", type=float, default=1.5, help="中心误差P95阈值(px)")
    p.add_argument("--scale-expected-pix", type=float, default=1.0, help="期望像面sigma(像素)")
    p.add_argument("--scale-tolerance", type=float, default=0.6, help="尺度允许偏差(像素)")
    p.add_argument("--visibility-dh", type=float, default=20.0, help="可见性测试近远点高程差(米)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_val_batch(cfg: dict[str, Any], scene_index: int) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    val_ds = build_dataset(mode="val", **data_cfg.get("val", {}))
    if len(val_ds) == 0:
        raise RuntimeError("val dataset 为空")
    if not (0 <= int(scene_index) < len(val_ds)):
        raise ValueError(f"scene_index 越界: {scene_index}, len={len(val_ds)}")
    sample = val_ds[int(scene_index)]
    return rpc_scene_collate_fn([sample])


def _to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _make_renderer(cfg: dict[str, Any]) -> RPCGaussianRenderer:
    # 这里显式覆盖若干配置，确保测试时尽量“不过滤、不下采样、不随机”。
    rcfg = RPCGaussianRendererCfg()
    for k, v in cfg.get("renderer", {}).items():
        if hasattr(rcfg, k):
            setattr(rcfg, k, v)

    rcfg.train_num_target_views = 1
    rcfg.val_num_target_views = 1
    rcfg.use_all_targets_in_val = False
    rcfg.exclude_self_source = True
    rcfg.allow_self_source_if_single_view = True

    rcfg.source_stride = 1
    rcfg.confidence_threshold = 0.0
    rcfg.topk_per_target = None
    rcfg.enable_voxelization = False

    rcfg.render_downsample_factor_train = 1
    rcfg.render_downsample_factor_val = 1

    rcfg.deterministic_target_selection = True
    rcfg.target_selection_mode = "ordered"
    rcfg.alpha_clamp_max = 1.0

    from geometry import RPCGeometryOps

    gops = RPCGeometryOps(rpc_dtype=torch.double, net_dtype=torch.float32)
    return RPCGaussianRenderer(gops, rcfg)


def _build_anchor_world_from_view_i(batch: dict[str, Any], vi: int) -> dict[str, torch.Tensor]:
    """构建 view_i 全像素 anchor 的物方点及其 normalized/local-meter 两种表示。"""
    b, v, _, h, w = batch["height_gt"].shape
    if b != 1:
        raise RuntimeError("本脚本当前仅支持 batch_size=1")
    if not (0 <= vi < v):
        raise ValueError(f"view_i 越界: {vi}, V={v}")

    rpc_i = batch["rpc_gt"][0][vi]
    h_map = batch["height_gt"][0, vi, 0].to(dtype=torch.double)

    # 像素网格（line, samp）
    line = torch.arange(h, device=h_map.device, dtype=torch.double).view(h, 1).expand(h, w).reshape(-1)
    samp = torch.arange(w, device=h_map.device, dtype=torch.double).view(1, w).expand(h, w).reshape(-1)
    h_flat = h_map.reshape(-1)

    # raw world (x,y,h)
    x_raw, y_raw = rpc_i.RPC_LINESAMP2XY(
        line_in=line.to(device=rpc_i.device),
        samp_in=samp.to(device=rpc_i.device),
        h_in=h_flat.to(device=rpc_i.device),
        output_type="tensor",
    )
    x_raw = x_raw.to(device=h_map.device, dtype=torch.float32)
    y_raw = y_raw.to(device=h_map.device, dtype=torch.float32)
    z = h_flat.to(dtype=torch.float32)

    center_yx = batch["scene_xy_center"][0].to(dtype=torch.float32)
    scale_yx = batch["scene_xy_scale"][0].to(dtype=torch.float32)
    cy, cx = center_yx[0], center_yx[1]
    sy, sx = scale_yx[0], scale_yx[1]

    x_norm = (x_raw - cx) / sx.clamp_min(1e-8)
    y_norm = (y_raw - cy) / sy.clamp_min(1e-8)

    # 局部米制（renderer 当前约定）: raw-center（不再做 /scale）
    x_local = x_raw - cx
    y_local = y_raw - cy

    raw_world = torch.stack([x_raw, y_raw, z], dim=-1).view(h, w, 3)
    norm_world = torch.stack([x_norm, y_norm, z], dim=-1).view(h, w, 3)
    local_world = torch.stack([x_local, y_local, z], dim=-1).view(h, w, 3)

    return {
        "raw_world": raw_world,
        "norm_world": norm_world,
        "local_world": local_world,
        "line": line.view(h, w),
        "samp": samp.view(h, w),
    }


def _build_synthetic_outputs(
    batch: dict[str, Any],
    vi: int,
    centers_point_hw3: torch.Tensor,
    centers_rpc_hw3: torch.Tensor,
    radius_meter: float,
    sh_dim: int,
) -> dict[str, Any]:
    """组装 renderer.render_paths 所需的 outputs 字典。"""
    b, v, _, h, w = batch["height_gt"].shape
    dev = batch["height_gt"].device

    centers_rpc = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)
    centers_point = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)
    centers_rpc[0, vi] = centers_rpc_hw3.permute(2, 0, 1)
    centers_point[0, vi] = centers_point_hw3.permute(2, 0, 1)

    opacity = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    opacity[0, vi] = 1.0

    scale = torch.zeros((b, v, 3, h, w), device=dev, dtype=torch.float32)
    scale[0, vi] = float(radius_meter)

    rotation = torch.zeros((b, v, 4, h, w), device=dev, dtype=torch.float32)
    rotation[:, :, 0:1] = 1.0  # 单位四元数 (w=1)

    # 用固定伪随机颜色，保证可复现。
    gen = torch.Generator(device="cpu")
    gen.manual_seed(123)
    sh = torch.zeros((b, v, sh_dim, h, w), device=dev, dtype=torch.float32)
    sh_rand = torch.rand((sh_dim, h, w), generator=gen, dtype=torch.float32)
    sh[0, vi] = sh_rand.to(device=dev)

    conf_rpc = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    conf_point = torch.zeros((b, v, 1, h, w), device=dev, dtype=torch.float32)
    conf_rpc[0, vi] = 1.0
    conf_point[0, vi] = 1.0

    return {
        "gaussian_centers_rpc": centers_rpc,
        "gaussian_centers_point": centers_point,
        "gaussian_opacity": opacity,
        "gaussian_scale": scale,
        "gaussian_rotation": rotation,
        "gaussian_sh": sh,
        "gaussian_confidence_rpc": conf_rpc,
        "gaussian_confidence_point": conf_point,
        "rpc_corrected": batch["rpc_gt"],  # synthetic: 直接使用 GT RPC
    }


def _project_anchor_to_view_j(raw_world_hw3: torch.Tensor, rpc_j: Any) -> torch.Tensor:
    """用 view_j 的 RPC_GT 把 raw world anchor 投影到像面，得到理论 anchor_i2j。"""
    h, w, _ = raw_world_hw3.shape
    xyz = raw_world_hw3.view(-1, 3)
    ls = raw_xy_to_linesamp(
        rpc_j,
        x=xyz[:, 0].to(device=rpc_j.device, dtype=torch.double),
        y=xyz[:, 1].to(device=rpc_j.device, dtype=torch.double),
        h=xyz[:, 2].to(device=rpc_j.device, dtype=torch.double),
    )
    return ls.to(dtype=torch.float32, device=raw_world_hw3.device).view(h, w, 2)


def _center_error_metrics(pred_ls: torch.Tensor, gt_ls: torch.Tensor) -> dict[str, float]:
    err = torch.linalg.norm(pred_ls - gt_ls, dim=-1).reshape(-1)
    return {
        "mean": float(err.mean().item()),
        "p50": float(torch.quantile(err, 0.5).item()),
        "p95": float(torch.quantile(err, 0.95).item()),
        "max": float(err.max().item()),
    }


def _scale_metrics_from_cov(cov_2d: torch.Tensor) -> dict[str, float]:
    """输出像面尺度统计（sigma级）。"""
    if cov_2d.numel() == 0:
        return {"sigma_mean": 0.0, "sigma_std": 0.0}
    s_line = cov_2d[:, 0, 0].clamp_min(1e-12).sqrt()
    s_samp = cov_2d[:, 1, 1].clamp_min(1e-12).sqrt()
    sigma = 0.5 * (s_line + s_samp)
    return {
        "sigma_mean": float(sigma.mean().item()),
        "sigma_std": float(sigma.std().item()),
    }


def _run_visibility_probe(
    renderer: RPCGaussianRenderer,
    batch: dict[str, Any],
    vj: int,
    base_center_local: torch.Tensor,
    base_height: float,
    dh: float,
    radius_meter: float,
) -> dict[str, float]:
    """构造两个重叠高斯(近/远)检查排序可见性。"""
    dev = batch["images"].device

    # two gaussians: same (x,y), different z, distinct colors
    c_near = base_center_local.clone()
    c_far = base_center_local.clone()
    c_near[2] = float(base_height - abs(dh) * 0.5)
    c_far[2] = float(base_height + abs(dh) * 0.5)

    centers = torch.stack([c_near, c_far], dim=0).to(device=dev, dtype=torch.float32)
    opacity = torch.ones((2, 1), device=dev, dtype=torch.float32)
    scale = torch.full((2, 3), float(radius_meter), device=dev, dtype=torch.float32)
    rotation = torch.zeros((2, 4), device=dev, dtype=torch.float32)
    rotation[:, 0] = 1.0

    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=dev, dtype=torch.float32)  # near=red, far=green
    conf = torch.ones((2, 1), device=dev, dtype=torch.float32)

    target_rpc = batch["rpc_gt"][0][vj]

    pack = {
        "centers_world": centers,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
        "rgb": rgb,
        "confidence": conf,
        "source_view_ids": torch.zeros((2,), device=dev, dtype=torch.long),
        "batch_index": 0,
        "target_view_index": int(vj),
        "target_rpc": target_rpc,
        "target_height_ref": float(batch["height_ref"][0, vj].item()),
        "raw_num_gaussians": 2,
        "after_stride_num_gaussians": 2,
        "after_conf_filter_num_gaussians": 2,
        "after_voxel_num_gaussians": 2,
    }

    out = renderer.render_single_path_single_target(pack, batch, mode="val")
    img = out["rendered_rgb"]  # [3,H,W]

    # 在理论中心像素附近读取颜色
    scene_ci = batch["scene_xy_center"][0]
    scene_si = torch.ones_like(scene_ci)
    mean_2d = project_gaussians_to_view(
        renderer.geometry_ops,
        centers,
        scale,
        rotation,
        target_rpc,
        scene_ci,
        scene_si,
        renderer.cfg.eps_xy_fd,
        renderer.cfg.eps_h_fd,
        renderer.cfg.eps_cov,
    )[0]

    # 取 near 的像素位置
    l = int(torch.round(mean_2d[0, 0]).clamp(0, img.shape[1] - 1).item())
    s = int(torch.round(mean_2d[0, 1]).clamp(0, img.shape[2] - 1).item())
    pix = img[:, l, s]

    # 计算 near/far 的颜色相似度
    sim_red = float((pix * torch.tensor([1.0, 0.0, 0.0], device=dev)).sum().item())
    sim_green = float((pix * torch.tensor([0.0, 1.0, 0.0], device=dev)).sum().item())

    # 按 renderer 当前排序逻辑推导“先画的是谁”
    view_dir = estimate_local_view_direction(
        renderer.geometry_ops,
        target_rpc,
        image_h=img.shape[1],
        image_w=img.shape[2],
        h_ref=float(batch["height_ref"][0, vj].item()),
        scene_xy_center=scene_ci,
        scene_xy_scale=scene_si,
        delta_h_dir=renderer.cfg.depth_dir_delta_h,
    ).to(device=dev, dtype=torch.float32)
    depth = compute_depth_proxy(centers, view_dir)
    order = torch.argsort(depth, descending=renderer.cfg.depth_sort_descending)
    first_idx = int(order[0].item())

    expected = "red" if first_idx == 0 else "green"
    got = "red" if sim_red >= sim_green else "green"

    return {
        "sim_red": sim_red,
        "sim_green": sim_green,
        "expected_front_color": 0.0 if expected == "red" else 1.0,
        "pred_front_color": 0.0 if got == "red" else 1.0,
        "visibility_ok": 1.0 if expected == got else 0.0,
    }


def main() -> None:
    prog = ProgressLogger()
    prog.log("startup")

    args = parse_args()
    prog.log("arguments parsed")
    cfg = load_cfg(args.config)
    prog.log(f"config loaded: {args.config}")

    device = torch.device(args.device)
    batch_cpu = build_val_batch(cfg, scene_index=int(args.scene_index))
    prog.log(f"val scene loaded and collated: scene_index={args.scene_index}")
    batch = _to_device_batch(batch_cpu, device)
    prog.log(f"batch moved to device: {device}")

    b, v, _, h, w = batch["images"].shape
    if b != 1:
        raise RuntimeError("当前脚本仅支持 B=1")
    if v < 2:
        raise RuntimeError(f"需要至少2个视图用于跨视图渲染检查，当前V={v}")

    vi = int(args.view_i)
    vj = int(args.view_j)
    if not (0 <= vi < v and 0 <= vj < v):
        raise ValueError(f"view_i/view_j越界: vi={vi}, vj={vj}, V={v}")
    if vi == vj:
        raise ValueError("view_i 与 view_j 不能相同")

    renderer = _make_renderer(cfg)
    prog.log("renderer created with deterministic/no-filter overrides")

    # ---------- 1) 构建 synthetic anchors/world ----------
    anchor = _build_anchor_world_from_view_i(batch, vi)
    prog.log(f"anchor world built from view_i={vi} with full pixels ({h}x{w})")
    raw_world = anchor["raw_world"]

    # point path 输入中心语义由参数控制：
    # normalized: 模拟 point_abs 归一化语义（黑盒链路常用于暴露坐标衔接问题）
    # local_meter: 局部米制语义（用于对照）
    centers_point = anchor["norm_world"] if args.point_center_mode == "normalized" else anchor["local_world"]

    # rpc path 这里使用局部米制（raw-center）构造，便于做对照基准。
    centers_rpc = anchor["local_world"]

    model_cfg = cfg.get("model", {})
    sh_dim = int(model_cfg.get("sh_dim", 48))

    outputs = _build_synthetic_outputs(
        batch=batch,
        vi=vi,
        centers_point_hw3=centers_point,
        centers_rpc_hw3=centers_rpc,
        radius_meter=float(args.radius_meter),
        sh_dim=sh_dim,
    )
    prog.log("synthetic outputs assembled")

    # ---------- 2) 调 renderer 全链路 ----------
    prog.log("calling renderer.render_paths (this may take long for full-pixel anchors)")
    render_out = renderer.render_paths(outputs, batch, mode="val", global_step=0, epoch=0)
    prog.log("renderer.render_paths finished")

    # ---------- 3) 中心投影正确性（投影层） ----------
    rpc_j = batch["rpc_gt"][0][vj]
    prog.log(f"projecting GT anchors from view_i={vi} to view_j={vj} using rpc_gt")
    anchor_i2j = _project_anchor_to_view_j(raw_world, rpc_j)
    prog.log("anchor_i2j projection finished")

    # 取与 point/rpc path 输入一致的一组中心做投影。
    # 注意：renderer 的投影统一使用 scene_center + scene_scale=ones。
    scene_ci = batch["scene_xy_center"][0]
    scene_si = torch.ones_like(scene_ci)

    point_centers_flat = centers_point.view(-1, 3)
    rpc_centers_flat = centers_rpc.view(-1, 3)

    scale_flat = torch.full((h * w, 3), float(args.radius_meter), device=device, dtype=torch.float32)
    rot_flat = torch.zeros((h * w, 4), device=device, dtype=torch.float32)
    rot_flat[:, 0] = 1.0

    point_mean_2d, point_cov_2d = project_gaussians_to_view(
        renderer.geometry_ops,
        point_centers_flat,
        scale_flat,
        rot_flat,
        rpc_j,
        scene_ci,
        scene_si,
        renderer.cfg.eps_xy_fd,
        renderer.cfg.eps_h_fd,
        renderer.cfg.eps_cov,
    )
    prog.log("point path projection and covariance finished")
    rpc_mean_2d, rpc_cov_2d = project_gaussians_to_view(
        renderer.geometry_ops,
        rpc_centers_flat,
        scale_flat,
        rot_flat,
        rpc_j,
        scene_ci,
        scene_si,
        renderer.cfg.eps_xy_fd,
        renderer.cfg.eps_h_fd,
        renderer.cfg.eps_cov,
    )
    prog.log("rpc path projection and covariance finished")

    point_mean_hw2 = point_mean_2d.view(h, w, 2)
    rpc_mean_hw2 = rpc_mean_2d.view(h, w, 2)

    metric_point_center = _center_error_metrics(point_mean_hw2, anchor_i2j)
    metric_rpc_center = _center_error_metrics(rpc_mean_hw2, anchor_i2j)

    # ---------- 4) 尺度正确性 ----------
    metric_point_scale = _scale_metrics_from_cov(point_cov_2d)
    metric_rpc_scale = _scale_metrics_from_cov(rpc_cov_2d)
    prog.log("center/scale metrics computed")

    # ---------- 5) 可见性(深度排序)检查 ----------
    c0 = centers_rpc[h // 2, w // 2].clone()  # 用中心像素附近的局部米制中心
    vis_metric = _run_visibility_probe(
        renderer=renderer,
        batch=batch,
        vj=vj,
        base_center_local=c0,
        base_height=float(raw_world[h // 2, w // 2, 2].item()),
        dh=float(args.visibility_dh),
        radius_meter=float(args.radius_meter),
    )
    prog.log("visibility probe finished")

    # ---------- 6) 输出报告 ----------
    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            name="point_center_p50_px",
            value=metric_point_center["p50"],
            threshold=float(args.center_threshold_p50),
            passed=metric_point_center["p50"] <= float(args.center_threshold_p50),
            note="point path中心误差P50",
        )
    )
    checks.append(
        CheckResult(
            name="point_center_p95_px",
            value=metric_point_center["p95"],
            threshold=float(args.center_threshold_p95),
            passed=metric_point_center["p95"] <= float(args.center_threshold_p95),
            note="point path中心误差P95",
        )
    )
    checks.append(
        CheckResult(
            name="rpc_center_p50_px",
            value=metric_rpc_center["p50"],
            threshold=float(args.center_threshold_p50),
            passed=metric_rpc_center["p50"] <= float(args.center_threshold_p50),
            note="rpc path中心误差P50(对照)",
        )
    )
    checks.append(
        CheckResult(
            name="rpc_center_p95_px",
            value=metric_rpc_center["p95"],
            threshold=float(args.center_threshold_p95),
            passed=metric_rpc_center["p95"] <= float(args.center_threshold_p95),
            note="rpc path中心误差P95(对照)",
        )
    )

    exp_sigma = float(args.scale_expected_pix)
    tol = float(args.scale_tolerance)
    checks.append(
        CheckResult(
            name="point_sigma_mean_px",
            value=metric_point_scale["sigma_mean"],
            threshold=tol,
            passed=abs(metric_point_scale["sigma_mean"] - exp_sigma) <= tol,
            note=f"point path尺度均值应接近{exp_sigma}±{tol}",
        )
    )
    checks.append(
        CheckResult(
            name="rpc_sigma_mean_px",
            value=metric_rpc_scale["sigma_mean"],
            threshold=tol,
            passed=abs(metric_rpc_scale["sigma_mean"] - exp_sigma) <= tol,
            note=f"rpc path尺度均值应接近{exp_sigma}±{tol}",
        )
    )

    checks.append(
        CheckResult(
            name="visibility_order",
            value=vis_metric["visibility_ok"],
            threshold=1.0,
            passed=vis_metric["visibility_ok"] >= 0.5,
            note="可见性排序是否符合当前depth_proxy+sort配置",
        )
    )

    print("\n================ Sat2World 3DGS Render Check ================")
    print(f"scene_index={args.scene_index}, view_i={vi}, view_j={vj}, HxW={h}x{w}")
    print(f"point_center_mode={args.point_center_mode}, radius_meter={args.radius_meter}")
    print("--------------------------------------------------------------")

    print("[center metrics]")
    print(" point:", metric_point_center)
    print(" rpc  :", metric_rpc_center)

    print("[scale metrics (sigma pix)]")
    print(" point:", metric_point_scale)
    print(" rpc  :", metric_rpc_scale)

    print("[visibility metrics]")
    print(vis_metric)

    print("[renderer stats]")
    print(" point path:", render_out["point"].get("stats", {}))
    print(" rpc path  :", render_out["rpc"].get("stats", {}))

    print("--------------------------------------------------------------")
    num_pass = 0
    for c in checks:
        ok = "PASS" if c.passed else "FAIL"
        print(f"[{ok}] {c.name}: value={c.value:.6f}, threshold={c.threshold:.6f} | {c.note}")
        if c.passed:
            num_pass += 1

    print("--------------------------------------------------------------")
    print(f"Summary: {num_pass}/{len(checks)} checks passed")
    if num_pass < len(checks):
        print("WARNING: 检查未全部通过，请优先关注 center/scale/visibility 的失败项。")
    print("==============================================================\n")
    prog.log("done")


if __name__ == "__main__":
    main()

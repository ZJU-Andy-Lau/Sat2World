"""engine.trainer

Sat2World 训练系统主入口。

核心语义：
- global_step 按“optimizer 更新步”计数，而非 micro-step；
- 支持 DDP + AMP + 梯度累积 + 中断恢复到 epoch 中间；
- 训练链路：batch -> model -> renderer(optional) -> objective -> backward/update。
"""

from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.distributed as dist

from engine.checkpoint import save_checkpoint
from engine.distributed import (
    all_reduce_mean,
    assert_rpc_tree_device,
    assert_tensor_tree_device,
    barrier,
    get_cuda_memory_stats,
    is_main_process,
    maybe_no_sync,
    move_batch_to_device,
)


class Trainer:
    """统一 Trainer。

    成员变量：
    - 训练状态：epoch/global_step/step_in_epoch/best_metric/start_epoch/skip_steps_in_current_epoch。
    - 固定监控样本：fixed_train_monitor_batch/fixed_val_monitor_batch。
    - 外部组件：model/renderer/objective/optimizer/scheduler/monitor。
    """

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        model: nn.Module,
        renderer: Any,
        objective: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: Any,
        val_loader: Any,
        device: torch.device,
        monitor: Any | None,
        scaler: Any | None,
        distributed_state: dict[str, Any],
        work_dir: str,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        """初始化 Trainer。"""
        self.cfg = cfg
        self.model = model
        self.renderer = renderer
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = monitor
        self.scaler = scaler
        self.distributed_state = distributed_state
        self.work_dir = Path(work_dir)
        ckpt_dir_cfg = str(cfg.get("checkpoints_dir", "")).strip()
        self.ckpt_dir = Path(ckpt_dir_cfg) if ckpt_dir_cfg else (self.work_dir / "checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.max_epochs = int(cfg.get("max_epochs", 1))
        self.accumulate_steps = int(cfg.get("accumulate_steps", 1))
        self.log_interval = int(cfg.get("log_interval", 20))
        self.hist_interval = int(cfg.get("hist_interval", 200))
        self.vis_interval = int(cfg.get("vis_interval", 500))
        self.ckpt_interval = int(cfg.get("ckpt_interval", 500))
        self.val_interval = int(cfg.get("val_interval", 1000))
        self.validate_each_epoch = bool(cfg.get("validate_each_epoch", True))
        self.validate_first = bool(cfg.get("validate_first", False))
        self.best_metric_name = str(cfg.get("best_metric", "loss_total"))
        self.best_mode = str(cfg.get("best_mode", "min"))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 0.0))
        self.grad_clip_mode = str(cfg.get("grad_clip_mode", "global")).strip().lower()
        if self.grad_clip_mode not in {"global", "grouped"}:
            self.grad_clip_mode = "global"
        self.grad_clip_group_norm = float(cfg.get("grad_clip_group_norm", self.grad_clip_norm))
        self.amp_dtype = str(cfg.get("amp_dtype", "fp16"))
        self.enable_render_train = bool(cfg.get("enable_render_train", True))
        self.enable_render_val = bool(cfg.get("enable_render_val", True))
        self.skip_nan_batch = bool(cfg.get("skip_nan_batch", True))
        self.max_train_steps_per_epoch = int(cfg.get("max_train_steps_per_epoch", -1))
        self.max_val_steps = int(cfg.get("max_val_steps", -1))
        self.device_sanity_check = bool(cfg.get("device_sanity_check", False))
        self.enable_fixed_monitor_cache = bool(cfg.get("enable_fixed_monitor_cache", True))
        self.monitor_use_current_batch = bool(cfg.get("monitor_use_current_batch", True))
        self.nan_probe_enable = bool(cfg.get("nan_probe_enable", True))
        self.nan_probe_mode = str(cfg.get("nan_probe_mode", "skip")).strip().lower()
        if self.nan_probe_mode not in {"skip", "raise"}:
            self.nan_probe_mode = "skip"
        self.nan_probe_sync_across_ranks = bool(cfg.get("nan_probe_sync_across_ranks", True))
        self.nan_probe_dump_rank_mode = str(cfg.get("nan_probe_dump_rank_mode", "all")).strip().lower()
        if self.nan_probe_dump_rank_mode not in {"all", "rank0"}:
            self.nan_probe_dump_rank_mode = "all"
        self.nan_probe_dump_max_events = int(cfg.get("nan_probe_dump_max_events", 200))
        self.nan_probe_events_written = 0
        self.nan_probe_dump_dir = self.work_dir / "nan_probe"
        if self.nan_probe_enable:
            self.nan_probe_dump_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.global_step = 0
        self.step_in_epoch = 0
        self.best_metric = float("inf") if self.best_mode == "min" else -float("inf")
        self.start_epoch = 0
        self.skip_steps_in_current_epoch = 0

        if resume_state is not None:
            saved_epoch = int(resume_state.get("epoch", 0))
            saved_step = int(resume_state.get("step_in_epoch", 0))
            saved_gstep = int(resume_state.get("global_step", 0))
            extra_state = resume_state.get("extra_state", {}) or {}
            if not isinstance(extra_state, dict):
                extra_state = {}
            save_reason = str(extra_state.get("save_reason", "")).strip().lower()
            resume_path = str(resume_state.get("resume_path", "")).strip()

            # 语义修复：
            # - epoch_end 保存点表示“该 epoch 已完成”，恢复应从下一 epoch 开始；
            # - 兼容旧 checkpoint（无 save_reason）：若从 last.pt 恢复且 step_in_epoch==0，按 epoch_end 处理。
            is_epoch_end_ckpt = (saved_step == 0) and (
                save_reason == "epoch_end"
                or (save_reason == "" and saved_gstep > 0 and Path(resume_path).name == "last.pt")
            )

            self.start_epoch = saved_epoch + 1 if is_epoch_end_ckpt else saved_epoch
            self.epoch = self.start_epoch
            self.skip_steps_in_current_epoch = 0 if is_epoch_end_ckpt else saved_step
            self.global_step = int(resume_state.get("global_step", 0))
            self.best_metric = float(resume_state.get("best_metric", self.best_metric))

        self.fixed_train_monitor_batch = None
        self.fixed_val_monitor_batch = None

    @staticmethod
    def _grad_group_name(param_name: str) -> str:
        n = param_name.lower()
        if ("backbone" in n) or ("encoder" in n) or ("geom_mlp" in n) or ("fuser" in n):
            return "backbone_encoder"
        if "dense_decoder" in n:
            return "dense_decoder"
        if ("height_adapter" in n) or ("height_anchor_head" in n) or ("height_local_head" in n):
            return "height"
        if ("point_adapter" in n) or ("point_xy_head" in n) or ("point_z_local_head" in n):
            return "point"
        if "affine_head" in n:
            return "affine"
        if "nce_projector" in n:
            return "nce_projector"
        if "patch_matcher" in n:
            return "patch_matcher"
        if ("gaussian_adapter" in n) or ("gaussian_head" in n):
            return "gaussian"
        return "other"

    def _clip_grad_norm(self) -> tuple[float, dict[str, float]]:
        groups = [
            "backbone_encoder",
            "dense_decoder",
            "height",
            "point",
            "affine",
            "nce_projector",
            "patch_matcher",
            "gaussian",
            "other",
        ]
        norms = {f"optim/grad_norm_{g}": 0.0 for g in groups}
        norms["optim/grad_norm_total_grouped_preclip"] = 0.0

        if self.grad_clip_mode != "grouped":
            if self.grad_clip_norm <= 0:
                return 0.0, norms
            g = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item())
            return g, norms

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        grouped_params: dict[str, list[torch.nn.Parameter]] = {k: [] for k in groups}
        for name, p in model_ref.named_parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue
            grouped_params[self._grad_group_name(name)].append(p)

        sq_sum = 0.0
        clip_norm = self.grad_clip_group_norm
        if clip_norm <= 0:
            return 0.0, norms
        for g in groups:
            ps = grouped_params[g]
            if len(ps) == 0:
                continue
            gn = float(torch.nn.utils.clip_grad_norm_(ps, clip_norm).item())
            norms[f"optim/grad_norm_{g}"] = gn
            sq_sum += gn * gn
        total_preclip = math.sqrt(max(sq_sum, 0.0))
        norms["optim/grad_norm_total_grouped_preclip"] = total_preclip
        return total_preclip, norms

    def _clone_batch_cpu(self, batch: dict[str, Any], keep_samples: int | None = 1) -> dict[str, Any]:
        """把 batch 截取并拷贝到 CPU，作为固定监控样本。"""
        out: dict[str, Any] = {}
        full_batch = keep_samples is None or int(keep_samples) <= 0
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = (v.detach().cpu().clone() if full_batch else v[:keep_samples].detach().cpu().clone())
            elif isinstance(v, list):
                out[k] = copy.deepcopy(v if full_batch else v[:keep_samples])
            else:
                out[k] = copy.deepcopy(v)
        return out

    def _set_epoch_for_samplers(self, epoch: int) -> None:
        """同步 sampler 与 dataset epoch。"""
        for loader in (self.train_loader, self.val_loader):
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            dataset = getattr(loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def _autocast_context(self):
        """返回 autocast 上下文。"""
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        if self.amp_dtype == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if self.amp_dtype == "fp16":
            return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        return torch.autocast(device_type="cuda", enabled=False)

    def _format_hhmmss(self, sec: float) -> str:
        sec_i = max(0, int(sec))
        h = sec_i // 3600
        m = (sec_i % 3600) // 60
        s = sec_i % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _forward_batch(self, batch_dev: dict[str, Any], mode: str):
        """统一前向：model -> renderer(optional) -> objective。"""
        if self.device_sanity_check:
            assert_tensor_tree_device(batch_dev, self.device, prefix="batch")
            for k in ("rpc_gt", "rpc_init", "rpc_corrected"):
                if k in batch_dev:
                    assert_rpc_tree_device(batch_dev[k], self.device, prefix=f"batch.{k}")
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, "set_runtime_context"):
            model_ref.set_runtime_context(global_step=self.global_step, mode=mode)
        outputs = self.model(batch_dev)
        if self.device_sanity_check:
            assert_tensor_tree_device(outputs, self.device, prefix="outputs")
            if "rpc_corrected" in outputs:
                assert_rpc_tree_device(outputs["rpc_corrected"], self.device, prefix="outputs.rpc_corrected")
        outputs_for_render = outputs
        if (
            "affine_pred" in outputs
            and "rpc_init" in batch_dev
            and hasattr(self.objective, "_replace_ref_affine_with_identity")
            and hasattr(model_ref, "rpc_ops")
        ):
            # 渲染路径沿用相对几何语义：ref 视图固定为 identity，再据此重算 corrected RPC。
            aff_for_render = self.objective._replace_ref_affine_with_identity(outputs["affine_pred"], batch_dev.get("ref_view_idx", None))
            outputs_for_render = dict(outputs)
            outputs_for_render["rpc_corrected"] = model_ref.rpc_ops.apply_affine_correction_batch(batch_dev["rpc_init"], aff_for_render)
        use_render = (mode == "train" and self.enable_render_train) or (mode != "train" and self.enable_render_val)
        render_outputs = (
            self.renderer.render_paths(
                outputs_for_render,
                batch_dev,
                mode=mode,
                global_step=self.global_step,
                epoch=self.epoch,
            )
            if (self.renderer is not None and use_render)
            else None
        )
        if self.device_sanity_check and render_outputs is not None:
            assert_tensor_tree_device(render_outputs, self.device, prefix="render_outputs")
        total_loss, scalar_dict, aux_dict = self.objective(outputs, batch_dev, self.global_step, self.epoch, render_outputs, mode)
        if self.device_sanity_check and torch.is_tensor(total_loss) and total_loss.device != self.device:
            raise RuntimeError(f"total_loss device mismatch: got {total_loss.device}, expect {self.device}")
        return outputs, render_outputs, total_loss, scalar_dict, aux_dict

    def _sanity_probe(self, outputs: dict[str, Any], batch_dev: dict[str, Any]) -> dict[str, float]:
        """训练稳定性探针。"""
        p: dict[str, float] = {}
        if "affine_pred" in outputs:
            ref = batch_dev.get("ref_view_idx", None)
            if ref is not None:
                a = outputs["affine_pred"]
                bsz = a.shape[0]
                eye = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=a.dtype, device=a.device)
                errs = []
                for bi in range(bsz):
                    rid = int(ref[bi].item()) if torch.is_tensor(ref) else int(ref)
                    errs.append((a[bi, rid] - eye).abs().mean())
                p["probe_reference_affine_identity_error"] = float(torch.stack(errs).mean().detach().item())
            a = outputs["affine_pred"]
            tx_ty_abs = a[..., :, 2].abs().mean()
            linear = a[..., :, :2]
            eye2 = torch.eye(2, dtype=linear.dtype, device=linear.device).view(1, 1, 2, 2)
            linear_dev = (linear - eye2).abs().mean()
            p["probe_affine_pred_translation_abs_mean"] = float(tx_ty_abs.detach().item())
            p["probe_affine_pred_linear_deviation_mean"] = float(linear_dev.detach().item())
        for name, key in [
            ("probe_nan_ratio_height", "height_abs"),
            ("probe_nan_ratio_point", "point_abs"),
            ("probe_nan_ratio_centers", "gaussian_centers_rpc"),
        ]:
            if key in outputs:
                t = outputs[key]
                p[name] = float((~torch.isfinite(t)).float().mean().detach().item())
        return p

    def _build_hist_dict(self, outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """构建直方图输入。"""
        d: dict[str, torch.Tensor] = {}
        if "affine_pred" in outputs:
            d["hist/affine_pred"] = outputs["affine_pred"].detach()
        if "height_abs" in outputs:
            d["hist/height_abs"] = outputs["height_abs"].detach()
        if "point_delta_xy_fine" in outputs:
            d["hist/point_delta_xy_fine"] = outputs["point_delta_xy_fine"].detach()
        if "point_z_local_offset" in outputs:
            d["hist/point_z_local_offset"] = outputs["point_z_local_offset"].detach()
        if "gaussian_opacity" in outputs:
            d["hist/gaussian_opacity"] = outputs["gaussian_opacity"].detach()
        if "gaussian_scale" in outputs:
            d["hist/gaussian_scale"] = outputs["gaussian_scale"].detach()
        if "gaussian_confidence_rpc" in outputs:
            d["hist/gaussian_conf_rpc"] = outputs["gaussian_confidence_rpc"].detach()
        if "gaussian_confidence_point" in outputs:
            d["hist/gaussian_conf_point"] = outputs["gaussian_confidence_point"].detach()
        return d

    def _compute_gt_point_map(self, batch_dev: dict[str, Any]) -> torch.Tensor | None:
        """为可视化生成 GT 点云图。"""
        if not hasattr(self.objective, "geometry_ops"):
            return None
        if "height_gt" not in batch_dev or "rpc_gt" not in batch_dev:
            return None
        b, v, _, h, w = batch_dev["height_gt"].shape
        from geometry.scene_geometry import make_image_grid

        grid = make_image_grid(h, w, device=batch_dev["height_gt"].device, dtype=batch_dev["height_gt"].dtype)
        return self.objective.geometry_ops.centers_from_rpc_and_height_batch(
            corrected_rpc_batch=batch_dev["rpc_gt"],
            pixel_grid=grid,
            height_abs=batch_dev["height_gt"],
            scene_xy_center=batch_dev.get("scene_xy_center", None),
            scene_xy_scale=batch_dev.get("scene_xy_scale", None),
        )

    def _train_one_step(self, batch: dict[str, Any], step_idx: int, data_time: float) -> dict[str, float]:
        """执行一个 train step（支持 AMP/DDP/梯度累积）。"""
        t0 = time.time()
        batch_dev = move_batch_to_device(batch, self.device)

        micro_step = (step_idx + 1) % self.accumulate_steps
        is_update_step = micro_step == 0

        with maybe_no_sync(self.model, enabled=not is_update_step):
            with self._autocast_context():
                outputs, render_outputs, loss, scalar, aux = self._forward_batch(batch_dev, mode="train")
                sanity = self._sanity_probe(outputs, batch_dev)
                scalar.update(sanity)
                loss_to_backward = loss / float(self.accumulate_steps)

            if self.nan_probe_enable:
                nonfinite_info = self._build_nonfinite_info(
                    loss=loss,
                    scalar=scalar,
                    aux=aux,
                    batch_dev=batch_dev,
                    outputs=outputs,
                    step_idx=step_idx,
                    is_update_step=is_update_step,
                )
                local_nonfinite = bool(nonfinite_info.get("local_nonfinite", False))
                global_nonfinite = self._sync_nonfinite_flag(local_nonfinite)
                if global_nonfinite:
                    self.optimizer.zero_grad(set_to_none=True)
                    self._maybe_dump_nan_report(nonfinite_info, local_nonfinite=local_nonfinite)
                    if self.nan_probe_mode == "raise":
                        bad = nonfinite_info.get("nonfinite_losses", [])
                        raise FloatingPointError(f"NaN probe triggered at global_step={self.global_step}, bad_losses={bad}")
                    return {"skip_non_finite": 1.0, "nan_probe_triggered": 1.0}

            if not torch.isfinite(loss):
                if self.skip_nan_batch:
                    self.optimizer.zero_grad(set_to_none=True)
                    return {"skip_non_finite": 1.0}
                raise FloatingPointError("Non-finite total_loss encountered in training step.")

            if self.scaler is not None:
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        grad_norm = 0.0
        grad_group_scalars = {
            "optim/grad_norm_backbone_encoder": 0.0,
            "optim/grad_norm_dense_decoder": 0.0,
            "optim/grad_norm_height": 0.0,
            "optim/grad_norm_point": 0.0,
            "optim/grad_norm_affine": 0.0,
            "optim/grad_norm_nce_projector": 0.0,
            "optim/grad_norm_patch_matcher": 0.0,
            "optim/grad_norm_gaussian": 0.0,
            "optim/grad_norm_other": 0.0,
            "optim/grad_norm_total_grouped_preclip": 0.0,
        }
        if is_update_step:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            grad_norm, grad_group_scalars = self._clip_grad_norm()

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1

        iter_time = time.time() - t0
        bsz = float(batch_dev["images"].shape[0] * batch_dev["images"].shape[1])
        imgps = bsz / max(iter_time, 1e-6)
        lr = float(self.optimizer.param_groups[0]["lr"])
        loss_scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
        mem = get_cuda_memory_stats(self.device)

        if render_outputs is not None:
            for path_name in ("rpc", "point"):
                stats = render_outputs.get(path_name, {}).get("stats", {})
                for k, v in stats.items():
                    scalar[f"render_stats_{path_name}_{k}"] = v

        scalar.update(
            {
                "optim/lr": lr,
                "optim/grad_norm": grad_norm,
                "optim/loss_scale": loss_scale,
                "time/data_time": data_time,
                "time/iter_time": iter_time,
                "time/img_per_sec": imgps,
                "system/gpu_mem_allocated_mb": mem["allocated_mb"],
            }
        )
        scalar.update(grad_group_scalars)
        scalar = all_reduce_mean(scalar)

        # 日志/可视化
        if is_main_process() and self.monitor is not None:
            if self.global_step % self.log_interval == 0 and is_update_step:
                self.monitor.log_scalars("train", scalar, self.global_step)
            if self.global_step % self.hist_interval == 0 and is_update_step:
                self.monitor.log_histograms(self._build_hist_dict(outputs), self.global_step)
            if self.global_step % self.vis_interval == 0 and is_update_step:
                self._run_monitor_visual(split="train", batch_override=batch)
                self._run_monitor_visual(split="val")

        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in scalar.items()}

    def _sync_nonfinite_flag(self, local_nonfinite: bool) -> bool:
        """跨 rank 同步 non-finite 触发标记。"""
        if not self.nan_probe_sync_across_ranks:
            return bool(local_nonfinite)
        if (not dist.is_available()) or (not dist.is_initialized()):
            return bool(local_nonfinite)
        t = torch.tensor(1 if local_nonfinite else 0, device=self.device, dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(int(t.item()) > 0)

    @staticmethod
    def _is_finite_value(x: Any) -> bool:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return True
            return bool(torch.isfinite(x).all().item())
        if isinstance(x, (int, float)):
            return math.isfinite(float(x))
        return True

    @staticmethod
    def _to_jsonable(x: Any) -> Any:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return "empty_tensor"
            if x.ndim == 0:
                v = float(x.detach().cpu().item())
                if math.isnan(v):
                    return "nan"
                if math.isinf(v):
                    return "inf" if v > 0 else "-inf"
                return v
            return {"type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype)}
        if isinstance(x, (int, float)):
            v = float(x)
            if math.isnan(v):
                return "nan"
            if math.isinf(v):
                return "inf" if v > 0 else "-inf"
            return v
        if isinstance(x, torch.device):
            return str(x)
        if isinstance(x, (list, tuple)):
            return [Trainer._to_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): Trainer._to_jsonable(v) for k, v in x.items()}
        return x

    def _build_nonfinite_info(
        self,
        *,
        loss: torch.Tensor,
        scalar: dict[str, Any],
        aux: dict[str, Any],
        batch_dev: dict[str, Any],
        outputs: dict[str, Any],
        step_idx: int,
        is_update_step: bool,
    ) -> dict[str, Any]:
        nonfinite_losses: list[str] = []
        nonfinite_metrics: list[str] = []

        for k, v in scalar.items():
            if k.startswith("loss_") and (not self._is_finite_value(v)):
                nonfinite_losses.append(k)
            if k.startswith("metric_") and (not self._is_finite_value(v)):
                nonfinite_metrics.append(k)

        aux_losses = aux.get("nan_probe_nonfinite_losses", [])
        aux_metrics = aux.get("nan_probe_nonfinite_metrics", [])
        if isinstance(aux_losses, list):
            nonfinite_losses.extend([str(x) for x in aux_losses])
        if isinstance(aux_metrics, list):
            nonfinite_metrics.extend([str(x) for x in aux_metrics])

        if not self._is_finite_value(loss):
            nonfinite_losses.append("loss_total")

        # 去重保序
        nonfinite_losses = list(dict.fromkeys(nonfinite_losses))
        nonfinite_metrics = list(dict.fromkeys(nonfinite_metrics))

        scene_id = batch_dev.get("scene_id", None)
        view_ids = batch_dev.get("view_ids", None)
        hmask = batch_dev.get("height_valid_mask", None)
        hpred = outputs.get("height_abs", None) if isinstance(outputs, dict) else None
        height_stats: dict[str, Any] = {}
        if torch.is_tensor(hmask):
            m = (hmask > 0.5)
            m_f = m.to(dtype=torch.float32)
            height_stats["height_valid_ratio"] = float(m_f.mean().item())
            if torch.is_tensor(hpred):
                fin = torch.isfinite(hpred)
                height_stats["height_pred_nonfinite_ratio_all"] = float((~fin).to(dtype=torch.float32).mean().item())
                valid_cnt = int(m.sum().item())
                invalid_cnt = int((~m).sum().item())
                if valid_cnt > 0:
                    height_stats["height_pred_nonfinite_ratio_valid"] = float((~fin[m]).to(dtype=torch.float32).mean().item())
                else:
                    height_stats["height_pred_nonfinite_ratio_valid"] = 0.0
                if invalid_cnt > 0:
                    height_stats["height_pred_nonfinite_ratio_invalid"] = float((~fin[~m]).to(dtype=torch.float32).mean().item())
                else:
                    height_stats["height_pred_nonfinite_ratio_invalid"] = 0.0

        info = {
            "local_nonfinite": (len(nonfinite_losses) > 0) or (len(nonfinite_metrics) > 0),
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "step_idx": int(step_idx),
            "is_update_step": bool(is_update_step),
            "rank": int(self.distributed_state.get("rank", 0)),
            "world_size": int(self.distributed_state.get("world_size", 1)),
            "loss_total": self._to_jsonable(loss),
            "nonfinite_losses": nonfinite_losses,
            "nonfinite_metrics": nonfinite_metrics,
            "first_bad_loss": nonfinite_losses[0] if len(nonfinite_losses) > 0 else str(aux.get("nan_probe_first_bad_loss", "")),
            "nan_probe_loss_snapshot": self._to_jsonable(aux.get("nan_probe_loss_snapshot", {})),
            "scene_id": self._to_jsonable(scene_id),
            "view_ids": self._to_jsonable(view_ids),
            "image_paths": self._to_jsonable(batch_dev.get("image_paths", [])),
            "height_stats": self._to_jsonable(height_stats),
            "loss_scale": float(self.scaler.get_scale()) if self.scaler is not None else 1.0,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        return info

    def _maybe_dump_nan_report(self, info: dict[str, Any], *, local_nonfinite: bool) -> None:
        """落盘 NaN 诊断报告。"""
        if self.nan_probe_events_written >= self.nan_probe_dump_max_events:
            return
        if self.nan_probe_dump_rank_mode == "rank0" and (not is_main_process()):
            return
        if (not local_nonfinite) and self.nan_probe_dump_rank_mode == "all":
            return

        self.nan_probe_events_written += 1
        rank = int(self.distributed_state.get("rank", 0))
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        p = self.nan_probe_dump_dir / f"nan_report_rank{rank}_g{int(self.global_step)}_{ts}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(info), f, ensure_ascii=False, indent=2)

        if is_main_process():
            print(
                "[nan-probe] trigger "
                f"global_step={info.get('global_step')} "
                f"rank={info.get('rank')} "
                f"first_bad_loss={info.get('first_bad_loss')} "
                f"report={p}",
                flush=True,
            )

    def _run_monitor_visual(self, split: str, batch_override: dict[str, Any] | None = None) -> None:
        """在固定监控样本上执行可视化推理。"""
        if self.monitor is None:
            return
        if split == "train" and self.monitor_use_current_batch and batch_override is not None:
            fixed = self._clone_batch_cpu(batch_override, keep_samples=None)
        else:
            fixed = self.fixed_train_monitor_batch if split == "train" else self.fixed_val_monitor_batch
        if fixed is None:
            return
        self.model.eval()
        with torch.no_grad():
            batch_dev = move_batch_to_device(copy.deepcopy(fixed), self.device)
            with self._autocast_context():
                outputs, render_outputs, _, _, aux = self._forward_batch(batch_dev, mode="val")
            gt_point = self._compute_gt_point_map(batch_dev)
            if gt_point is not None:
                aux["gt_point_map"] = gt_point
            self.monitor.log_visual_panels(batch_dev, outputs, render_outputs, aux, self.global_step, split)
            self.monitor.log_pointclouds(batch_dev, outputs, aux, self.global_step, split)
        self.model.train()

    def _save_last_checkpoint(self, step_in_epoch: int, *, save_reason: str | None = None) -> None:
        """保存 last checkpoint。"""
        t0 = time.time()
        reason = str(save_reason).strip() if save_reason is not None else ("epoch_end" if int(step_in_epoch) == 0 else "in_epoch")
        try:
            success = save_checkpoint(
                self.ckpt_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=self.epoch,
                step_in_epoch=step_in_epoch,
                global_step=self.global_step,
                best_metric=self.best_metric,
                cfg=self.cfg,
                extra_state={"save_reason": reason},
                min_disk_gb=float(self.cfg.get("min_disk_gb", 2.0)),
            )
            if success:
                dt = time.time() - t0
                if is_main_process() and self.monitor is not None:
                    self.monitor.log_scalars("ckpt", {"save_time_sec": dt}, self.global_step)
                    self.monitor.log_text("events/checkpoint", f"saved last checkpoint at step={self.global_step}", self.global_step)
        except Exception as e:
            print(f"[ERROR] Trainer failed to save last checkpoint: {e}")

    def _save_best_checkpoint(self) -> None:
        """保存 best checkpoint。"""
        t0 = time.time()
        try:
            success = save_checkpoint(
                self.ckpt_dir / "best.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=self.epoch,
                step_in_epoch=self.step_in_epoch,
                global_step=self.global_step,
                best_metric=self.best_metric,
                cfg=self.cfg,
                extra_state={"is_best": True},
                min_disk_gb=float(self.cfg.get("min_disk_gb", 2.0)),
            )
            if is_main_process():
                dt = time.time() - t0
                if self.monitor is not None:
                    self.monitor.log_scalars("ckpt", {"best_save_time_sec": dt}, self.global_step)
                if success:
                    if self.monitor is not None:
                        self.monitor.log_text("events/checkpoint", f"saved best checkpoint at step={self.global_step}", self.global_step)
                else:
                    print(f"[WARN] Skip saving best checkpoint at step={self.global_step} (save_checkpoint returned False)")
                    if self.monitor is not None:
                        self.monitor.log_text("events/checkpoint", f"failed to save best checkpoint at step={self.global_step}", self.global_step)
        except Exception as e:
            print(f"[ERROR] Trainer failed to save best checkpoint: {e}")

    def _run_train_epoch(self) -> None:
        """执行一个训练 epoch。"""
        self.model.train()
        epoch_t0 = time.time()
        steps_target = len(self.train_loader)
        if self.max_train_steps_per_epoch > 0:
            steps_target = min(steps_target, self.max_train_steps_per_epoch)
        prev_time = time.time()
        loss_sums = {
            "loss_total": 0.0,
            "loss_affine_grid": 0.0,
            "loss_affine_pair": 0.0,
            "loss_height": 0.0,
            "loss_height_reproj": 0.0,
            "loss_point": 0.0,
        }
        n_loss_steps = 0
        for step_idx, batch in enumerate(self.train_loader):
            if self.max_train_steps_per_epoch > 0 and step_idx >= self.max_train_steps_per_epoch:
                break
            if self.enable_fixed_monitor_cache and self.fixed_train_monitor_batch is None:
                self.fixed_train_monitor_batch = self._clone_batch_cpu(batch, keep_samples=1)
            if self.skip_steps_in_current_epoch > 0 and step_idx < self.skip_steps_in_current_epoch:
                continue
            if self.skip_steps_in_current_epoch > 0 and step_idx >= self.skip_steps_in_current_epoch:
                self.skip_steps_in_current_epoch = 0

            data_time = time.time() - prev_time
            logs = self._train_one_step(batch, step_idx, data_time)
            self.step_in_epoch = step_idx + 1
            prev_time = time.time()

            lt = logs.get("loss_total", None)
            lag = logs.get("loss_affine_grid", None)
            lap = logs.get("loss_affine_pair", None)
            lh = logs.get("loss_height", None)
            lhrep = logs.get("loss_height_reproj", None)
            lp = logs.get("loss_point", None)
            lpp = logs.get("loss_point_pair", None)
            lpr = logs.get("loss_point_reproj", None)
            lnp = logs.get("loss_normal_point", None)
            lnce = logs.get("loss_feature_nce", None)
            lssim = logs.get("loss_ssim", None)
            if lt is not None and lag is not None and lap is not None and lh is not None and lhrep is not None and lp is not None:
                loss_sums["loss_total"] += float(lt)
                loss_sums["loss_affine_grid"] += float(lag)
                loss_sums["loss_affine_pair"] += float(lap)
                loss_sums["loss_height"] += float(lh)
                loss_sums["loss_height_reproj"] += float(lhrep)
                loss_sums["loss_point"] += float(lp)
                n_loss_steps += 1

            if is_main_process():
                lr = float(logs.get("optim/lr", 0.0))
                elapsed = time.time() - epoch_t0
                done = max(step_idx + 1, 1)
                eta = (elapsed / done) * max(steps_target - done, 0)
                print(
                    "[train] "
                    f"ep={self.epoch} st={step_idx} gst={self.global_step} "
                    f"l_tot={float(lt) if lt is not None else float('nan'):.2f} "
                    f"l_ag={float(lag) if lag is not None else float('nan'):.2f} "
                    f"l_ap={float(lap) if lap is not None else float('nan'):.2f} "
                    f"l_h={float(lh) if lh is not None else float('nan'):.2f} "
                    f"l_hr={float(lhrep) if lhrep is not None else float('nan'):.2f} "
                    f"l_pt={float(lp) if lp is not None else float('nan'):.2f} "
                    f"l_pp={float(lpp) if lpp is not None else float('nan'):.2f} "
                    f"l_pr={float(lpr) if lpr is not None else float('nan'):.2f} "
                    f"l_np={float(lnp) if lnp is not None else float('nan'):.2f} "
                    f"l_nce={float(lnce) if lnce is not None else float('nan'):.2f} "
                    f"l_ssim={float(lssim) if lssim is not None else float('nan'):.2f} "
                    f"lr={lr:.2e} "
                    f"time={self._format_hhmmss(elapsed)} "
                    f"eta={self._format_hhmmss(eta)}"
                )

            if self.global_step > 0 and self.global_step % self.ckpt_interval == 0:
                self._save_last_checkpoint(step_in_epoch=step_idx + 1)

            if self.global_step > 0 and self.global_step % self.val_interval == 0:
                self.validate()
                self.model.train()

        self.step_in_epoch = 0
        if is_main_process() and n_loss_steps > 0:
            avg_total = loss_sums["loss_total"] / n_loss_steps
            avg_ag = loss_sums["loss_affine_grid"] / n_loss_steps
            avg_ap = loss_sums["loss_affine_pair"] / n_loss_steps
            avg_h_abs = loss_sums["loss_height"] / n_loss_steps
            avg_h_rel = loss_sums["loss_height_reproj"] / n_loss_steps
            avg_pt = loss_sums["loss_point"] / n_loss_steps
            print(
                "[train][epoch_summary] "
                f"epoch={self.epoch} "
                f"l_tot={avg_total:.2f} "
                f"l_ag={avg_ag:.2f} "
                f"l_ap={avg_ap:.2f} "
                f"l_h_abs={avg_h_abs:.2f} "
                f"l_h_rep={avg_h_rel:.2f} "
                f"l_pt={avg_pt:.2f}"
            )

    def validate(self) -> dict[str, float]:
        """执行完整验证流程。"""
        t0 = time.time()
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for step_idx, batch in enumerate(self.val_loader):
                if self.max_val_steps > 0 and step_idx >= self.max_val_steps:
                    break
                if self.enable_fixed_monitor_cache and self.fixed_val_monitor_batch is None:
                    self.fixed_val_monitor_batch = self._clone_batch_cpu(batch, keep_samples=1)
                batch_dev = move_batch_to_device(batch, self.device)
                with self._autocast_context():
                    _, _, _, scalar, _ = self._forward_batch(batch_dev, mode="val")
                scalar = all_reduce_mean(scalar)
                for k, v in scalar.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                n += 1
        if n > 0:
            agg = {k: v / n for k, v in agg.items()}

        if is_main_process() and self.monitor is not None:
            self.monitor.log_scalars("val", agg, self.global_step)
            self.monitor.log_text("events/validate", f"validation finished at step={self.global_step}", self.global_step)
            self.monitor.log_scalars("time", {"validation_sec": time.time() - t0}, self.global_step)

        cur = float(agg.get(self.best_metric_name, agg.get("loss_total", float("inf"))))
        better = (cur < self.best_metric) if self.best_mode == "min" else (cur > self.best_metric)
        if is_main_process():
            lt = float(agg.get("loss_total", float("nan")))
            lag = float(agg.get("loss_affine_grid", float("nan")))
            lap = float(agg.get("loss_affine_pair", float("nan")))
            lh = float(agg.get("loss_height", float("nan")))
            lhrep = float(agg.get("loss_height_reproj", float("nan")))
            lp = float(agg.get("loss_point", float("nan")))
            lpp = float(agg.get("loss_point_pair", float("nan")))
            lnp = float(agg.get("loss_normal_point", float("nan")))
            lnce = float(agg.get("loss_feature_nce", float("nan")))
            lssim = float(agg.get("loss_ssim", float("nan")))
            tag = " [best]" if better else ""
            print(
                "[val] "
                f"epoch={self.epoch} gstep={self.global_step} "
                f"l_tot={lt:.2f} "
                f"l_ag={lag:.2f} "
                f"l_ap={lap:.2f} "
                f"l_h_abs={lh:.2f} "
                f"l_h_rep={lhrep:.2f} "
                f"l_pt={lp:.2f} "
                f"l_pp={lpp:.2f}"
                f"l_np={lnp:.2f} "
                f"l_nce={lnce:.2f} "
                f"l_ssim={lssim:.2f}"
                f"{tag}"
            )
        if better:
            self.best_metric = cur
            self._save_best_checkpoint()
        barrier()
        return agg

    def fit(self) -> None:
        """训练主循环。"""
        try:
            if self.validate_first:
                self.validate()
            for ep in range(self.start_epoch, self.max_epochs):
                self.epoch = ep
                self._set_epoch_for_samplers(ep)
                self._run_train_epoch()
                if self.validate_each_epoch:
                    self.validate()
                self._save_last_checkpoint(step_in_epoch=0, save_reason="epoch_end")
        except KeyboardInterrupt:
            if is_main_process():
                print("KeyboardInterrupt: saving emergency checkpoint...")
            self._save_last_checkpoint(step_in_epoch=self.step_in_epoch, save_reason="emergency")
        finally:
            if is_main_process() and self.monitor is not None:
                self.monitor.flush()

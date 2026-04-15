"""engine.trainer

Sat2World 训练系统主入口。

核心语义：
- global_step 按“optimizer 更新步”计数，而非 micro-step；
- 支持 DDP + AMP + 梯度累积 + 中断恢复到 epoch 中间；
- 训练链路：batch -> model -> renderer(optional) -> objective -> backward/update。
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
        self.amp_dtype = str(cfg.get("amp_dtype", "fp16"))
        self.enable_render_train = bool(cfg.get("enable_render_train", True))
        self.enable_render_val = bool(cfg.get("enable_render_val", True))
        self.skip_nan_batch = bool(cfg.get("skip_nan_batch", True))
        self.max_train_steps_per_epoch = int(cfg.get("max_train_steps_per_epoch", -1))
        self.max_val_steps = int(cfg.get("max_val_steps", -1))
        self.device_sanity_check = bool(cfg.get("device_sanity_check", False))
        self.enable_fixed_monitor_cache = bool(cfg.get("enable_fixed_monitor_cache", True))
        self.monitor_use_current_batch = bool(cfg.get("monitor_use_current_batch", True))

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
        if "point_delta_fine" in outputs:
            d["hist/point_delta_fine"] = outputs["point_delta_fine"].detach()
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
        if is_update_step:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.grad_clip_norm > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm).item())

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
                outputs, render_outputs, _, scalar, aux = self._forward_batch(batch_dev, mode="val")
            gt_point = self._compute_gt_point_map(batch_dev)
            if gt_point is not None:
                aux["gt_point_map"] = gt_point
            self.monitor.log_visual_panels(batch_dev, outputs, render_outputs, aux, self.global_step, split)
            self.monitor.log_pointclouds(batch_dev, outputs, aux, self.global_step, split)
            self.monitor.log_scalars(split, scalar, self.global_step)
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
            "loss_height_rel": 0.0,
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
            lhrel = logs.get("loss_height_rel", None)
            lp = logs.get("loss_point", None)
            lnp = logs.get("loss_normal_point", None)
            lnce = logs.get("loss_feature_nce", None)
            lssim = logs.get("loss_ssim", None)
            if lt is not None and lag is not None and lap is not None and lh is not None and lhrel is not None and lp is not None:
                loss_sums["loss_total"] += float(lt)
                loss_sums["loss_affine_grid"] += float(lag)
                loss_sums["loss_affine_pair"] += float(lap)
                loss_sums["loss_height"] += float(lh)
                loss_sums["loss_height_rel"] += float(lhrel)
                loss_sums["loss_point"] += float(lp)
                n_loss_steps += 1

            if is_main_process():
                lr = float(logs.get("optim/lr", 0.0))
                elapsed = time.time() - epoch_t0
                done = max(step_idx + 1, 1)
                eta = (elapsed / done) * max(steps_target - done, 0)
                print(
                    "[train] "
                    f"epoch={self.epoch} step={step_idx} gstep={self.global_step} "
                    f"l_tot={float(lt) if lt is not None else float('nan'):.6f} "
                    f"l_ag={float(lag) if lag is not None else float('nan'):.6f} "
                    f"l_ap={float(lap) if lap is not None else float('nan'):.6f} "
                    f"l_h_abs={float(lh) if lh is not None else float('nan'):.6f} "
                    f"l_h_rel={float(lhrel) if lhrel is not None else float('nan'):.6f} "
                    f"l_pt={float(lp) if lp is not None else float('nan'):.6f} "
                    f"l_np={float(lnp) if lnp is not None else float('nan'):.6f} "
                    f"l_nce={float(lnce) if lnce is not None else float('nan'):.6f} "
                    f"l_ssim={float(lssim) if lssim is not None else float('nan'):.6f} "
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
            avg_h_rel = loss_sums["loss_height_rel"] / n_loss_steps
            avg_pt = loss_sums["loss_point"] / n_loss_steps
            print(
                "[train][epoch_summary] "
                f"epoch={self.epoch} "
                f"l_tot={avg_total:.6f} "
                f"l_ag={avg_ag:.6f} "
                f"l_ap={avg_ap:.6f} "
                f"l_h_abs={avg_h_abs:.6f} "
                f"l_h_rel={avg_h_rel:.6f} "
                f"l_pt={avg_pt:.6f}"
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
            lhrel = float(agg.get("loss_height_rel", float("nan")))
            lp = float(agg.get("loss_point", float("nan")))
            lnp = float(agg.get("loss_normal_point", float("nan")))
            lnce = float(agg.get("loss_feature_nce", float("nan")))
            lssim = float(agg.get("loss_ssim", float("nan")))
            tag = " [best]" if better else ""
            print(
                "[val] "
                f"epoch={self.epoch} gstep={self.global_step} "
                f"l_tot={lt:.6f} "
                f"l_ag={lag:.6f} "
                f"l_ap={lap:.6f} "
                f"l_h_abs={lh:.6f} "
                f"l_h_rel={lhrel:.6f} "
                f"l_pt={lp:.6f} "
                f"l_np={lnp:.6f} "
                f"l_nce={lnce:.6f} "
                f"l_ssim={lssim:.6f}"
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

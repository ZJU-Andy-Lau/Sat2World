"""engine.trainer

本文件实现 Sat2World 的训练引擎（Trainer）。

核心职责：
1) 组织训练循环（前向、渲染、损失、反向、优化）；
2) 管理 AMP、梯度裁剪、学习率调度；
3) 提供 checkpoint 保存/恢复；
4) 兼容 DDP 训练（含 module 包装与状态同步语义）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from loss import Sat2WorldLoss
from render import RPCGaussianRenderer


@dataclass
class TrainerCfg:
    """训练器配置。

    成员变量：
        epochs:
            训练总 epoch 数。
        grad_clip_norm:
            梯度裁剪阈值（global norm）。
        amp:
            是否启用自动混合精度（仅 CUDA 生效）。
        log_every:
            每隔多少个 iteration 打印一次日志。
        ckpt_every:
            每隔多少个 epoch 保存一次 checkpoint。
        out_dir:
            checkpoint 输出目录。
    """

    epochs: int = 1
    grad_clip_norm: float = 1.0
    amp: bool = True
    log_every: int = 10
    ckpt_every: int = 1
    out_dir: str = "outputs"


class Trainer:
    """Sat2World 训练器。

    功能概述：
    - 把 model、renderer、loss 串成训练闭环；
    - 管理 device/DDP/AMP 的工程细节；
    - 维护 epoch/global_step 与可恢复状态。

    成员变量：
        model:
            训练模型；world_size>1 时会被 DDP 包装。
        optimizer:
            优化器。
        train_loader:
            训练 dataloader。
        lr_scheduler:
            学习率调度器，可选。
        loss_fn:
            组合损失模块。
        renderer:
            渲染器模块。
        cfg:
            训练配置。
        device:
            训练设备。
        rank/world_size:
            分布式环境信息。
        scaler:
            AMP 梯度缩放器。
        epoch/global_step:
            训练进度状态，可写入 checkpoint。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Any,
        *,
        lr_scheduler: Any | None = None,
        loss_fn: Sat2WorldLoss | None = None,
        renderer: RPCGaussianRenderer | None = None,
        cfg: TrainerCfg | None = None,
        device: torch.device | str = "cuda",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """初始化训练器。

        输入：
            model:
                需要训练的模型实例。
            optimizer:
                优化器实例。
            train_loader:
                训练 dataloader。
            lr_scheduler:
                学习率调度器（可选）。
            loss_fn:
                损失模块（可选，缺省使用 Sat2WorldLoss）。
            renderer:
                渲染器（可选，缺省使用 RPCGaussianRenderer）。
            cfg:
                训练配置（可选）。
            device:
                设备字符串或 torch.device。
            rank/world_size:
                分布式 rank 与总进程数。

        输出：
            无显式返回值；完成所有训练状态初始化。
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn or Sat2WorldLoss()
        self.renderer = renderer or RPCGaussianRenderer()
        self.cfg = cfg or TrainerCfg()
        self.device = torch.device(device)
        self.rank = int(rank)
        self.world_size = int(world_size)

        self.model.to(self.device)
        if self.world_size > 1 and not isinstance(self.model, DDP):
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.cfg.amp and self.device.type == "cuda"))
        self.epoch = 0
        self.global_step = 0

        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)

    @property
    def _raw_model(self) -> torch.nn.Module:
        """返回未包装的原始模型。

        输入：
            无。

        输出：
            若当前 model 为 DDP，则返回 model.module；否则返回 model 本身。

        功能说明：
            统一用于参数访问与 state_dict 存取，避免 DDP 包装差异影响逻辑。
        """
        return self.model.module if isinstance(self.model, DDP) else self.model

    def state_dict(self) -> dict[str, Any]:
        """导出训练器状态字典。

        输入：
            无。

        输出：
            包含 model/optimizer/scaler/epoch/global_step/lr_scheduler 的状态字典。

        功能说明：
            该状态可直接用于 torch.save，后续通过 load_state_dict 完整恢复。
        """
        return {
            "model": self._raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "lr_scheduler": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """从状态字典恢复训练器状态。

        输入：
            state:
                与 state_dict() 输出兼容的字典。

        输出：
            无显式返回值；内部状态被就地更新。

        功能说明：
            用于恢复训练中断现场，包含参数、优化器、缩放器和进度状态。
        """
        self._raw_model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        if self.lr_scheduler is not None and state.get("lr_scheduler", None) is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])

    def save_checkpoint(self, name: str | None = None) -> Path:
        """保存 checkpoint 到输出目录。

        输入：
            name:
                checkpoint 文件名；为 None 时自动使用 epoch 命名。

        输出：
            Path：保存后的 checkpoint 路径。
        """
        if name is None:
            name = f"epoch_{self.epoch:04d}.pt"
        path = Path(self.cfg.out_dir) / name
        torch.save(self.state_dict(), path)
        return path

    def load_checkpoint(self, ckpt_path: str) -> None:
        """从磁盘加载 checkpoint 并恢复。

        输入：
            ckpt_path:
                checkpoint 文件路径。

        输出：
            无显式返回值；训练状态恢复到 checkpoint 对应时刻。
        """
        state = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """将 batch 中张量移动到训练设备。

        输入：
            batch:
                原始 batch 字典。

        输出：
            新字典：
                - 张量值迁移到 self.device；
                - 非张量对象保持原样（如 RPC 对象列表）。
        """
        out: dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """执行单个 iteration 训练步骤。

        输入：
            batch:
                dataloader 提供的单批数据。

        输出：
            dict[str, float]：
                当前 step 的各损失标量（已转 CPU float，便于日志记录）。

        功能说明：
            步骤顺序：
            1) batch 迁移到设备；
            2) 前向 + 渲染 + 损失；
            3) AMP 反向；
            4) 梯度裁剪与优化器更新；
            5) 可选调度器步进。
        """
        batch = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device.type, enabled=(self.cfg.amp and self.device.type == "cuda")):
            model_out = self.model(batch)
            render_out = self.renderer(model_out, batch)
            losses = self.loss_fn(model_out, batch, render_out)
            loss = losses["total"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self._raw_model.parameters(), self.cfg.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1
        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def train(self) -> None:
        """执行完整训练循环。

        输入：
            无（使用初始化时注入的 train_loader/cfg）。

        输出：
            无显式返回值；训练过程通过日志和 checkpoint 落盘体现。

        功能说明：
            - 按 epoch 外层循环迭代；
            - 若 sampler/dataset 支持 set_epoch，会同步更新；
            - 按 log_every 打印损失；
            - 按 ckpt_every 保存模型。
        """
        for ep in range(self.epoch, self.cfg.epochs):
            self.epoch = ep
            self.model.train()

            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(ep)
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(ep)

            for it, batch in enumerate(self.train_loader):
                logs = self._train_step(batch)
                if self.rank == 0 and (it % self.cfg.log_every == 0):
                    msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                    print(f"[epoch {ep} iter {it}] {msg}")

            if self.rank == 0 and (((ep + 1) % self.cfg.ckpt_every) == 0):
                self.save_checkpoint()

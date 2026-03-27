"""scripts.train_sat2world

Sat2World 训练入口脚本。

职责：
1) 解析训练参数；
2) 构建训练数据集与 DataLoader；
3) 构建模型、优化器与 Trainer；
4) 支持 checkpoint 恢复并启动训练。
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import RPCSceneDataset, rpc_scene_collate_fn
from engine import Trainer, TrainerCfg
from model import Sat2World, Sat2WorldCfg


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    输入：
        无。

    输出：
        argparse.ArgumentParser：
            已注册训练常用参数，包括数据路径、epoch、batch size、学习率、
            输出目录、恢复路径与每样本视图上限。

    功能说明：
        将训练关键超参数全部显式化，便于脚本化调用与实验复现。
    """
    p = argparse.ArgumentParser(description="Train Sat2World")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--max-view-num", type=int, default=4)
    return p


def main() -> None:
    """训练主函数。

    输入：
        无（参数来自命令行）。

    输出：
        无显式返回值；执行训练并按配置落盘 checkpoint。

    功能说明：
        处理完整训练启动流程：
        - 设备选择（CUDA/CPU）；
        - 数据集与 DataLoader 构建；
        - 模型与优化器初始化；
        - Trainer 创建；
        - 可选从 checkpoint 恢复；
        - 启动 trainer.train()。
    """
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RPCSceneDataset(
        root=args.data_root,
        mode="train",
        max_view_num=args.max_view_num,
        samples_per_scene=1,
        apply_perturbation=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=rpc_scene_collate_fn,
        drop_last=False,
    )

    model = Sat2World(Sat2WorldCfg())
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=loader,
        cfg=TrainerCfg(epochs=args.epochs, out_dir=args.out_dir),
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()

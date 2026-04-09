from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import moviepy.editor as mpy
import torch, copy
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn, optim
import numpy as np
import json, os, cv2
from ..dataset.data_module import get_data_shim
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, calculate_mae, calculate_rmse, calculate_less_thre_ratio
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.step_tracker import StepTracker
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import save_ply
from PIL import Image
import tifffile as tiff

def concat_context_target(batch):
    out = {}

    ctx = batch["context"]
    tgt = batch["target"]

    # 1. ref_filename：list，直接拼
    out["ref_filename"] = ctx["ref_filename"] + tgt["ref_filename"]

    # 2. tensor：在 view 维度(dim=1) 拼接
    out["image"]   = torch.cat([ctx["image"],   tgt["image"]],   dim=1)  # (1, 4, 3, 256, 256)
    out["cam2img"] = torch.cat([ctx["cam2img"], tgt["cam2img"]], dim=1)  # (1, 4, 4, 4)
    out["cam2enu"] = torch.cat([ctx["cam2enu"], tgt["cam2enu"]], dim=1)  # (1, 4, 4, 4)
    # out["gt_cls"]  = torch.cat([ctx["gt_cls"],  tgt["gt_cls"]],  dim=1)  # (1, 4, 256, 256)

    return out


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            self.val_step_outputs = {}
        self.val_step_cnt = 0

    def training_step(self, batch, batch_idx):
        filename = batch["context"]['ref_filename'][0][0].split("/0/")[-1]
        print("Filename: ", filename)
        # Only for test。复制 batch 给 target 和 context
        _, _, _, h, w = batch["context"]["image"].shape
        target_gt = batch["context"]["image"]  # 暂时用训练视角，而不用测试集合中：推理（未见过）的视角
        # For three resolutions, render them
        total_loss = 0
        loss_items = {}
        # Run the model.
        gaussian_dict,result_dict = self.encoder(batch["context"], self.global_step, False)
        for i in range(len(gaussian_dict)):
            gaussians = gaussian_dict[f"stage{i}"]["gaussians"]
            # pred_depths = result_dict[f"stage2"]["depths"]
            output = self.decoder.forward(gaussians, batch["context"], (h, w), mode = "train")

            ### save
            # per_filename = filename.split(".tif")[0]
            # test_save_path = f"/data/huangxuejun/project/SkySplat++_Lightning/Ablation/SkySplat++_guidedHeight_hiera_RES/work_dirs/train/stage{i}/train_render_3DGS_{per_filename}.png"
            # img = (output["render_color"][0][0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
            # Image.fromarray(img).save(test_save_path)  # 或 "output.jpg"
            ### save

            # Compute metrics.
            psnr_probabilistic = compute_psnr(rearrange(target_gt, "b v c h w -> (b v) c h w"),rearrange(output["render_color"], "b v c h w -> (b v) c h w"),)
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

            # Compute and log loss.
            for loss_fn in self.losses:
                # if ("mse" in loss_fn.name) or ("lpips" in loss_fn.name):
                loss = loss_fn.forward(output, batch, gaussians, self.global_step, self.trainer.max_steps - 1)
                self.log(f"loss/{loss_fn.name}", loss)
                loss_items[loss_fn.name] = loss.item()  # <--- 保存以便打印
                total_loss = total_loss + loss

        # if 'relheight' in loss_fn.name:
        #     loss = loss_fn.forward(batch, gaussians, self.global_step, i)
        #     self.log(f"loss/{loss_fn.name}", loss)
        #     loss_items[loss_fn.name] = loss.item()  # <--- 保存以便打印
        #     total_loss = total_loss + loss

        self.log("loss/total", total_loss)
        # Logging to console
        if (
                self.global_rank == 0
                and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            detail = " | ".join([f"{n}: {v:.6f}" for n, v in loss_items.items()])
            print(f"train step {self.global_step}; total: {total_loss:.6f} | {detail}")

        self.log("info/global_step", self.global_step)

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        return total_loss

    def test_step(self, batch, batch_idx):
        filename = batch["context"]['ref_filename'][0][0].split("/0/")[-1]
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        # Render Gaussians.
        gaussian_dict,result_dict = self.encoder(batch["context"], self.global_step, False)
        gaussians = gaussian_dict[f"stage2"]["gaussians"]
        ### 先渲染context的,再渲染target的!cat在一起
        # batch["all"] = concat_context_target(batch)
        # output = self.decoder.forward(gaussians, batch["all"], (h, w), mode="test")

        output = self.decoder.forward(gaussians, batch["target"], (h, w), mode="test")

        scene = batch["context"]['ref_filename'][0][0].split("/0/")[-1].split(".tif")[0].split("_RGB_")[0]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        ### RGB_save
        # per_filename = filename.split(".tif")[0]
        # test_save_path = f"/data/huangxuejun/project/SkySplat++_Lightning/Ablation/SkySplat++_guidedHeight_hiera_RES/work_dirs/test/target/test_render_3DGS_{per_filename}.png"
        # img = (output["render_color"][0][0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        # Image.fromarray(img).save(test_save_path)  # 或 "output.jpg"
        ### RGB_save

        ### 3DGS可视化
        # gaussians.harmonics = output['render_color'].squeeze(dim=0).permute(0, 2, 3, 1).reshape(1, 3, 256 * 256, 3)  ### 请注意要先进行permute交换维度，否则会发生tensor存储维度错误！！！
        ### 可视化的高度是ENU坐标系下的高度
        # save_ply(copy.deepcopy(gaussians),f"/data/huangxuejun/project/SkySplat-OV-Lightning/SkySplat_OV_Fusion_e3nn_wo_cosine_lr/work_dirs/{scene}.ply")
        ### 3DGS可视化

        ### RGB和高度值
        rgb_gt = batch["target"]["image"] # 暂时用训练视角，而不用测试集合中：推理（未见过）的视角
        rgb_gt = rearrange(rgb_gt, "b v c h w -> (b v) c h w")
        render_rgb = rearrange(output["render_color"], "b v c h w -> (b v) c h w")
        height_gt = batch["context"]['gt_height'] # 监督来源DAM_V2
        mask_height = ~torch.isnan(height_gt)  # 得到非 NaN 的掩膜
        pred_height = gaussians.hei.reshape(b, v, h, w)

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = render_rgb
            # save_txt = '/data/huangxuejun/project/SkySplat++_Lightning/Ablation/SkySplat++_guidedHeight_hiera_RES/work_dirs/test/test.txt'
            # save_psnr_value = compute_psnr(rgb_gt, rgb)
            # for save_psnr_index in range(len(save_psnr_value)):
            #     per_save_psnr_value = save_psnr_value[save_psnr_index].item()
            #     image_name = batch["target"]['ref_filename'][save_psnr_index][0].split('/')[-1].split('.tif')[0]
            #     with open(save_txt, "a") as f:
            #         line = [image_name, f"PSNR: {per_save_psnr_value:.6f}"]
            #         f.write(",".join(line) + "\n")


            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []
            if f"mae" not in self.test_step_outputs:
                self.test_step_outputs[f"mae"] = []
            if f"rmse" not in self.test_step_outputs:
                self.test_step_outputs[f"rmse"] = []
            if f"pag10" not in self.test_step_outputs:
                self.test_step_outputs[f"pag10"] = []
            if f"pag25" not in self.test_step_outputs:
                self.test_step_outputs[f"pag25"] = []
            if f"pag75" not in self.test_step_outputs:
                self.test_step_outputs[f"pag75"] = []
            self.test_step_outputs[f"psnr"].append(compute_psnr(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"ssim"].append(compute_ssim(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"lpips"].append(compute_lpips(rgb_gt, rgb).mean().item())
            self.test_step_outputs[f"mae"].append(calculate_mae(height_gt, pred_height, mask_height))
            self.test_step_outputs[f"rmse"].append(calculate_rmse(height_gt, pred_height, mask_height))
            self.test_step_outputs[f"pag10"].append(calculate_less_thre_ratio(height_gt, pred_height, mask_height, 1.0))
            self.test_step_outputs[f"pag25"].append(calculate_less_thre_ratio(height_gt, pred_height, mask_height, 2.5))
            self.test_step_outputs[f"pag75"].append(calculate_less_thre_ratio(height_gt, pred_height, mask_height, 7.5))
            # for i in range(3):
            #     ### 可视化height map！
            #     save_height_path = "/data/huangxuejun/project/SkySplat++_Lightning/Ablation/SkySplat++_guidedHeight_hiera_RES/work_dirs/test/context_height"
            #     save_txt = '/data/huangxuejun/project/SkySplat++_Lightning/Ablation/SkySplat++_guidedHeight_hiera_RES/work_dirs/test/test_height.txt'
            #     image_name = batch["context"]['ref_filename'][i][0].split('/')[-1].split('.tif')[0]
            #     per_output_path = os.path.join(save_height_path, f"hei_{image_name}.tif")
            #     height_map = pred_height[0][i].reshape(256, 256).detach().cpu().numpy()
            #     # 归一化处理到 0~255，用于可视化（处理负值）
            #     min_val = np.nanmin(height_map)
            #     max_val = np.nanmax(height_map)
            #     norm = (height_map - min_val) / (max_val - min_val + 1e-6)  # 避免除以0
            #     norm_uint8 = (norm * 255).astype(np.uint8)
            #     # 应用伪彩色（使用 Jet 色图）
            #     color_map = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
            #     cv2.imwrite(per_output_path.replace(".tif", "_jet.png"), color_map)
            #     tiff.imwrite(per_output_path, height_map.astype(np.float32))
            #     mae_val = calculate_mae(height_gt[0][i], pred_height[0][i], mask_height[0][i])
            #     with open(save_txt, "a") as f:
            #         line = [image_name, f"MAE: {mae_val:.6f}"]
            #         f.write(",".join(line) + "\n")
                ### 可视化height map！


    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                metric_scores = [float(x) for x in metric_scores]  # 转为原生 float
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]):]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()


    def validation_step(self, batch, batch_idx):
        filename = batch["context"]['ref_filename'][0][0].split("/0/")[-1]
        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        # Render Gaussians.
        gaussian_dict,result_dict = self.encoder(batch["context"], self.global_step, False)
        gaussians = gaussian_dict[f"stage2"]["gaussians"]
        output = self.decoder.forward(gaussians,batch["target"],(h, w), mode = "val")

        ### RGB_save
        per_filename = filename.split(".tif")[0]
        test_save_path = f"/data/huangxuejun/project/SkySplat++_Lightning/Ablation/SkySplat++_guidedHeight_hiera_RES/work_dirs/val/target/test_render_3DGS_{per_filename}.png"
        img = (output["render_color"][0][0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(test_save_path)  # 或 "output.jpg"
        ### RGB_save

        rgb_gt = batch["target"]["image"] # 暂时用训练视角，而不用测试集合中：推理（未见过）的视角
        rgb_gt = rearrange(rgb_gt, "b v c h w -> (b v) c h w")
        render_rgb = rearrange(output["render_color"], "b v c h w -> (b v) c h w")
        height_gt = batch["context"]['gt_height'] # 监督来源DAM_V2
        mask_height = ~torch.isnan(height_gt)  # 得到非 NaN 的掩膜
        pred_height = gaussians.hei.reshape(b, v, h, w)
        # compute scores
        rgb = render_rgb
        if f"psnr" not in self.val_step_outputs:
            self.val_step_outputs[f"psnr"] = []
        if f"ssim" not in self.val_step_outputs:
            self.val_step_outputs[f"ssim"] = []
        if f"lpips" not in self.val_step_outputs:
            self.val_step_outputs[f"lpips"] = []
        if f"mae" not in self.val_step_outputs:
            self.val_step_outputs[f"mae"] = []
        if f"rmse" not in self.val_step_outputs:
            self.val_step_outputs[f"rmse"] = []
        if f"pag10" not in self.val_step_outputs:
            self.val_step_outputs[f"pag10"] = []
        if f"pag25" not in self.val_step_outputs:
            self.val_step_outputs[f"pag25"] = []
        if f"pag75" not in self.val_step_outputs:
            self.val_step_outputs[f"pag75"] = []
        psnr_val, ssim_val, lpips_val = compute_psnr(rgb_gt, rgb).mean().item(), compute_ssim(rgb_gt, rgb).mean().item(), compute_lpips(rgb_gt, rgb).mean().item()
        mae_val, rmse_val = calculate_mae(height_gt, pred_height, mask_height), calculate_rmse(height_gt, pred_height, mask_height)
        pag10_val, pag25_val, pag75_val = (calculate_less_thre_ratio(height_gt, pred_height, mask_height, 1.0), calculate_less_thre_ratio(height_gt, pred_height, mask_height, 2.5),
                                           calculate_less_thre_ratio(height_gt, pred_height, mask_height, 7.5))
        self.val_step_outputs[f"psnr"].append(psnr_val)
        self.val_step_outputs[f"ssim"].append(ssim_val)
        self.val_step_outputs[f"lpips"].append(lpips_val)
        self.val_step_outputs[f"mae"].append(mae_val)
        self.val_step_outputs[f"rmse"].append(rmse_val)
        self.val_step_outputs[f"pag10"].append(pag10_val)
        self.val_step_outputs[f"pag25"].append(pag25_val)
        self.val_step_outputs[f"pag75"].append(pag75_val)
        ### 打印指标
        print(f"val step {int(self.val_step_cnt)}; "f"psnr: {psnr_val:.6f} | "f"ssim: {ssim_val:.6f} | "f"lpips: {lpips_val:.6f} "f"mae: {mae_val:.6f}"
              f" | "f"rmse: {rmse_val:.6f} | "f"pag10: {pag10_val:.6f} | "f"pag25: {pag25_val:.6f} | "f"pag75: {pag75_val:.6f}")
        self.val_step_cnt = self.val_step_cnt + 1

    def on_validation_epoch_end(self):
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        out_dir = Path(str(out_dir).replace("/test/", "/val/"))
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_scores = {}

        # ----- 计算平均得分并 log -----
        for metric_name, metric_scores in self.val_step_outputs.items():
            avg_scores = sum(metric_scores) / len(metric_scores)
            saved_scores[metric_name] = avg_scores
            print(f"val_avg_{metric_name}", avg_scores)
            # log 到 lightning
            self.log(f"val/{metric_name}", avg_scores, on_epoch=True, prog_bar=True)

        # ----- 构造当前 epoch 的条目 -----
        epoch_record = {"epoch": self.current_epoch}
        epoch_record.update(saved_scores)
        json_file = out_dir / f"scores_all_avg.json"
        # ----- 若文件已存在，则读取并 append -----
        if json_file.exists():
            with json_file.open("r") as f:
                history = json.load(f)
                # 防止格式不一致
                if not isinstance(history, list):
                    history = [history]
        else:
            history = []
        # append 当前 epoch 记录
        history.append(epoch_record)
        # ----- 保存回文件（不会覆盖历史） -----
        with json_file.open("w") as f:
            json.dump(history, f, indent=2)

        # ----- 清理 -----
        self.benchmarker.clear_history()
        self.val_step_outputs.clear()
        self.val_step_cnt = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }


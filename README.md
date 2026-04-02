# Sat2World

Sat2World 是一个面向**多视图遥感影像 + RPC 成像模型**的工程化训练框架。当前仓库已经实现了从数据读取、RPC 几何、模型前向、双路径 Gaussian 渲染、损失计算到 DDP 训练/验证/可视化/断点恢复的完整闭环。

---

## 1. 项目简介

### 1.1 项目目标

Sat2World 的目标是：在多视图遥感场景中，给定每视图影像和初始 RPC（含误差），联合学习并输出：

1. RPC 仿射校正（affine RPC correction）；
2. 绝对高程图（absolute height）；
3. 绝对点云图（absolute 3D point map）；
4. 双路径 Gaussian 场景表示（RPC+Height 与 Point 两条中心路径）；
5. 在 corrected RPC 条件下的可微渲染监督闭环。

### 1.2 核心思想

本项目借鉴了 AnySplat 的“多视图联合编码 + Gaussian 场景表示 + 渲染监督”思想，但将其迁移到遥感 RPC 成像范式：

- 不再使用针孔相机内外参，而是使用 RPC；
- 训练目标不只包含 NVS 外观，还包含几何校正（affine）与绝对高度/点云；
- 显式维护两条 Gaussian 中心路径并行训练。

### 1.3 与原始 AnySplat 的关键区别

- 几何模型：针孔相机 → **RPC**；
- 任务场景：自然图像 NVS → **遥感多视图几何恢复 + RPC 平差**；
- 输出设计：新增 **affine correction、absolute height、independent point head、dual-path Gaussian centers**；
- 训练监督：在 corrected RPC 下进行渲染约束，并与几何损失联合优化。

### 1.4 当前工程状态

当前仓库已具备可运行模块：

- dataset（扫描、读取、扰动、collate）
- geometry（RPC 核心运算、工程封装、场景几何工具）
- model（backbone/encoder/heads/coder/Sat2World）
- render（RPC Gaussian renderer）
- loss（多任务损失与调度）
- engine（DDP、Trainer、Checkpoint、TensorBoard）
- scripts（train / validate / sanity_check）
- config（`config/default.yaml`）
- env（`env.yaml`）

> 说明：仓库根目录当前**没有** `rpc.py` 文件，RPC 实现位于 `geometry/rpc.py`。

---

## 2. 核心特性

- RPC-aware 多视图编码与 patch 几何特征构建
- 在线构造初始 RPC 扰动（train）与可选评估扰动（val/test）
- 仿射校正预测（observed→true）
- 绝对高程恢复（height head + coder）
- 独立绝对点云恢复（point head + coder）
- 双路径 Gaussian 中心构造（RPC+Height / Point）
- corrected RPC 条件下可微渲染监督（rpc path + point path）
- DDP 多卡训练（`torch.distributed`）
- TensorBoard 标量/图像/点云可视化
- checkpoint 保存、自动恢复、best/last 管理

---

## 3. 项目结构

```text
Sat2World/
├── PROJECT_COGNITION.md          # 工程设计与约定说明
├── README.md                     # 本文档
├── env.yaml                      # Conda 环境定义（Sat2World, Python 3.10）
├── config/
│   └── default.yaml              # 主配置文件
├── dataset/                      # 数据扫描、TIFF/RPC读取、扰动、Dataset、collate
├── geometry/                     # RPC核心算子与几何工具
├── model/                        # Sat2World模型（backbone/encoder/heads/coder）
├── render/                       # RPC Gaussian renderer
├── loss/                         # 各分支损失与总目标
├── engine/                       # DDP、Trainer、Checkpoint、TensorBoard
├── scripts/                      # train.py / validate.py / sanity_check.py
└── third_party/
    ├── dinov3/                   # 本地 DINOv3 代码与权重加载入口（torch.hub local）
    └── AnySplat/                 # 参考实现（本项目当前主流程未直接调用）
```

模块协同主链路：

`dataset -> model -> renderer(optional) -> objective(loss) -> trainer(engine)`

---

## 4. 环境配置

### 4.1 推荐平台

- OS：Linux x86_64
- GPU：NVIDIA（建议 CUDA 12.1 对应驱动）
- Python：3.10

### 4.2 使用 `env.yaml` 创建环境

在仓库根目录执行：

```bash
conda env create -f env.yaml
conda activate Sat2World
```

`env.yaml` 已包含关键依赖：

- `pytorch=2.3.*`, `torchvision=0.18.*`, `pytorch-cuda=12.1`
- `numpy`, `pyyaml`, `rasterio`, `tifffile`, `opencv`, `tensorboard`, `pillow`

### 4.3 环境可用性验证

建议先运行：

```bash
python scripts/sanity_check.py --config config/default.yaml
```

如果脚本打印 `[sanity_check] success`，说明主流程（model + renderer + loss + backward）可正常运行。

> 说明：`sanity_check.py` 会 monkeypatch DINO backbone，因此不依赖真实 DINO 权重即可验证全链路。

### 4.4 几何预训练（仅 affine/height/point）

如果你希望先训练几何主线（不训练 3DGS 渲染相关能力），可使用：

```bash
python scripts/pretrain.py --config config/pretrain.yaml
```

该入口会复用现有数据/模型/DDP 训练框架，但使用几何预训练 objective，且不启用 renderer，并在模型前向中跳过 Gaussian 分支计算。

---

## 5. 数据准备

本项目 `dataset.scan_dataset_root()` 使用严格目录规范。

### 5.1 根目录概念

`config/default.yaml` 中典型配置：

- `data.train.root: train_data`
- `data.val.root: test_data`
- `data.test.root: test_data`

你可以改成绝对路径，例如：

- `/path/to/train_data`
- `/path/to/val_data`

### 5.2 目录命名规范（必须）

```text
<root>/
├── scene_0/
│   ├── view_0/
│   │   ├── image_0_0.tif
│   │   ├── height_0_0.tif
│   │   └── rpc_0_0.txt
│   ├── view_1/
│   │   ├── image_0_1.tif
│   │   ├── height_0_1.tif
│   │   └── rpc_0_1.txt
│   └── ...
├── scene_1/
│   └── view_*/...
└── ...
```

每个 `view_{view_id}` 目录必须唯一匹配：

- `image_{scene_id}_{view_id}.tif`
- `height_{scene_id}_{view_id}.tif`
- `rpc_{scene_id}_{view_id}.txt`

### 5.3 文件语义

- `image_*.tif`：多视图遥感影像（读取后变为 `[3,H,W]` float）
- `height_*.tif`：高程 GT（读取后 `[1,H,W]`）
- `rpc_*.txt`：**GT 已平差 RPC（真值 RPC）**

### 5.4 扰动与 `rpc_init` 生成逻辑

训练时（`mode=train` 且 `apply_perturbation=true`）：

- dataset 对**非参考视图**在线施加 forward affine 扰动（true→observed）；
- 由 `rpc_gt` 生成 `rpc_init`；
- 参考视图固定为第 0 视图（`reference_policy=first`），其 affine 恒为单位阵。

验证/测试默认不扰动（可通过 `synthetic_perturbation_in_eval=true` 开启确定性评测扰动）。

### 5.5 视图数与尺寸约束

- 场景视图数小于 `min_views` 会被过滤；全部过滤后会直接报错。
- 若 `strict_same_hw=true`，样本内各视图分辨率必须一致，否则报错。
- collate 阶段要求 batch 内 `H/W` 一致。

### 5.6 高程无效值处理

`read_height_tif()` 会输出：

- `height_gt`：无效像素以有效区域均值填充；
- `height_valid_mask`：有效为 1，无效为 0。

loss 会使用 mask 屏蔽无效区域。

---

## 6. 配置文件说明

主配置文件：`config/default.yaml`

主要分区：

- `system`：工程路径、随机种子、DDP backend、恢复策略
- `data`：train/val/test 根目录、视图采样、扰动、DataLoader
- `geometry`：RPC/net dtype（当前主要用于配置语义）
- `model`：DINO 权重路径、encoder/head/coder 参数
- `renderer`：目标视图采样、top-k、downsample、栅格化控制
- `loss`：各项损失权重、ramp/warmup、几何采样
- `optim`：AdamW 与 warmup-cosine 调度
- `train`：epoch、AMP、梯度累积、日志/可视化/验证/ckpt 间隔
- `logging`：TensorBoard 开关与可视化参数
- `checkpoint`：best/last 策略与指标名

### 6.1 首次运行最小必改项

至少修改以下字段：

1. `model.dino_weight_path`：真实 DINOv3 权重路径（训练/验证需要）
2. `data.train.root` / `data.val.root`：数据根目录
3. `system.work_dir`：实验输出目录
4. （可选）`data.loader.batch_size`、`data.train.max_view_num`

### 6.2 命令行覆盖优先级

`train.py` 支持通过 CLI 覆盖部分 `system` 字段：

- `--work-dir` 覆盖 `system.work_dir`
- `--seed` 覆盖 `system.seed`
- `--eval-only` 设置 `system.eval_only=true`
- `--resume` 覆盖 `system.resume_path`
- `--checkpoint` 覆盖 `system.checkpoint_path`

恢复路径选择逻辑（train）：

1. `--checkpoint`
2. `--resume`
3. `system.resume_path`
4. 若以上为空且 `system.auto_resume=true`，自动查找 `<work_dir>/checkpoints/last.pt` 或最新 `epoch_*.pt`

---

## 7. 快速开始（从零到跑通）

### Step 1) 创建环境

```bash
conda env create -f env.yaml
conda activate Sat2World
```

### Step 2) 准备数据

按第 5 节组织 `train_data` / `val_data` 目录。

### Step 3) 修改配置

编辑 `config/default.yaml`，至少修改：

- `model.dino_weight_path`
- `data.train.root`
- `data.val.root`
- `system.work_dir`

### Step 4) 运行全链路 sanity check

```bash
python scripts/sanity_check.py --config config/default.yaml
```

### Step 5) 启动单卡训练

```bash
python scripts/train.py --config config/default.yaml
```

### Step 6) 启动多卡训练（DDP）

```bash
torchrun --nproc_per_node=<NUM_GPUS> scripts/train.py --config config/default.yaml
```

例如 4 卡：

```bash
torchrun --nproc_per_node=4 scripts/train.py --config config/default.yaml
```

---

## 8. 训练指南

### 8.1 常用训练命令

#### 单卡

```bash
python scripts/train.py --config config/default.yaml
```

#### 指定输出目录

```bash
python scripts/train.py --config config/default.yaml --work-dir /path/to/work_dir
```

#### 从指定 checkpoint 恢复

```bash
python scripts/train.py --config config/default.yaml --resume /path/to/checkpoints/last.pt
```

或：

```bash
python scripts/train.py --config config/default.yaml --checkpoint /path/to/checkpoints/best.pt
```

#### 仅做 train 脚本内部一轮前向 sanity（不反向）

```bash
python scripts/train.py --config config/default.yaml --sanity-only
```

#### eval-only（走 train.py 入口）

```bash
python scripts/train.py --config config/default.yaml --eval-only --checkpoint /path/to/checkpoints/best.pt
```

### 8.2 训练过程中默认行为

- 可选渲染监督：由 `train.enable_render_train` / `train.enable_render_val` 控制
- 损失调度：`loss.warmup_steps_geom_only` + `loss.render_ramp_steps`
- AMP：`train.amp_dtype` + `train.enable_grad_scaler`
- 日志频率：`train.log_interval`, `hist_interval`, `vis_interval`
- 验证频率：`train.val_interval`、`train.validate_each_epoch`
- checkpoint 频率：`train.ckpt_interval` 与 epoch 末保存

### 8.3 输出目录与文件

默认以 `system.work_dir` 为根，例如 `work_dirs/sat2world_default/`：

- `checkpoints/last.pt`：最近状态
- `checkpoints/best.pt`：最佳指标状态
- TensorBoard 日志：默认 `logging.tensorboard.log_dir`（例如 `work_dirs/sat2world_default/tb`）

### 8.4 checkpoint 恢复语义

checkpoint 内保存：

- model / optimizer / scheduler / scaler
- `epoch`, `step_in_epoch`, `global_step`
- `best_metric`
- RNG state（python/numpy/torch）

恢复后 Trainer 会按 `step_in_epoch` 跳过当轮已完成 step，尽量无缝续训。

---

## 9. 验证 / 测试

使用独立脚本：`scripts/validate.py`

### 9.1 基本命令

```bash
python scripts/validate.py \
  --config config/default.yaml \
  --checkpoint /path/to/checkpoints/best.pt
```

可选：

```bash
python scripts/validate.py \
  --config config/default.yaml \
  --checkpoint /path/to/checkpoints/best.pt \
  --work-dir /path/to/val_work_dir \
  --seed 123
```

### 9.2 验证行为说明

- 使用 `data.val` 配置构建数据集；
- 加载 checkpoint（仅模型权重，优化器等不恢复）；
- 使用与训练一致的 objective/renderer 做前向评估；
- 调用 `Trainer.validate()` 聚合标量（跨卡 all-reduce mean）；
- 若 TensorBoard 开启，会写入验证曲线（默认 `tb_val` 或 `logging.tensorboard.log_dir`）。

---

## 10. 推理与模型输出使用

当前仓库**没有单独 `inference.py`**。可行方式：

1. 用 `scripts/validate.py` 跑前向评估；
2. 或在 Python 中直接调用 `model = build_model(cfg); outputs = model(batch)`。

### 10.1 模型主要输出字段

`Sat2World.forward()` 返回（节选）：

- 几何校正：`affine_pred`, `rpc_corrected`
- 高程：`height_abs`, `height_coarse`, `height_fine`, `height_logits`
- 点云：`point_anchor`, `point_abs`, `point_delta_coarse`, `point_delta_fine`
- Gaussian 属性：`gaussian_opacity`, `gaussian_scale`, `gaussian_rotation`, `gaussian_sh`
- 双路径中心：`gaussian_centers_rpc`, `gaussian_centers_point`
- 其他：`patch_valid_mask`, `patch_tokens_final`, `view_tokens_final`, `scene_token_final`

### 10.2 渲染输出

通过 `renderer.render_paths(outputs, batch, mode=...)` 可得到 `rpc` 与 `point` 两条渲染路径结果（RGB/alpha/height/target 等）。

> 目前未提供官方导出脚本（如点云文件/校正 RPC 文件导出）。如需导出，请在 Python 侧读取上述张量并自行保存。

---

## 11. 训练输出与可视化说明

本项目 `TensorBoardMonitor` 会记录三类信息：标量、图像、点云。

### 11.1 关键标量（示例）

- 总损失与子损失：`loss_total`, `loss_height`, `loss_point`, `loss_render_*` 等
- 优化器状态：`optim/lr`, `optim/grad_norm`, `optim/loss_scale`
- 时间/性能：`time/data_time`, `time/iter_time`, `time/img_per_sec`
- 显存：`system/gpu_mem_allocated_mb`
- 训练探针：如参考视图 affine 单位阵误差、NaN 比例

### 11.2 关键图像面板

- 输入与几何：
  - `vis/*/input_rgb`
  - `vis/*/height_gt`, `height_pred`, `height_error`
  - `vis/*/point_z_pred`, `point_z_gt`, `point_z_error`
- 渲染路径：
  - `vis/*/render_rpc_rgb`, `render_target_rgb`, `render_rpc_alpha`, `render_rpc_height`
  - `vis/*/render_point_rgb`, `render_point_alpha`, `render_point_height`
- Gaussian 诊断：
  - `vis/*/gaussian_conf_rpc`, `gaussian_conf_point`
  - `vis/*/gaussian_opacity`, `gaussian_scale_mag`
  - `vis/*/center_disagreement`
- 仿射几何：
  - `vis/*/affine_residual`
  - `vis/*/pairwise_error_matrix`（若可用）

### 11.3 点云可视化

- `vis/*/pc_gt`
- `vis/*/pc_pred`
- `vis/*/pc_rpc_center`

优先 `add_mesh`，失败时自动回退为 BEV 图。

### 11.4 如何判断训练是否正常

建议重点关注：

1. `loss_total`、`loss_height`、`loss_point` 是否整体下降；
2. `center_disagreement` 是否逐步减小（两条中心路径一致性提升）；
3. `affine_residual` 是否向低误差区域收敛；
4. `render_rpc_rgb` 与 `render_target_rgb` 差异是否收敛；
5. `gaussian_opacity`、`gaussian_scale_mag` 是否稳定无塌陷。

---

## 12. 关键约定与易混淆点（务必阅读）

1. **磁盘 RPC 是 GT 已平差 RPC。**
2. `rpc_init` 是 dataset 对 GT RPC 注入 forward 扰动后的“初始误差 RPC”。
3. `affine_gt_forward` 表示 `true pixel -> observed pixel`。
4. 模型预测 `affine_pred` 表示 `observed pixel -> true pixel`。
5. 第一张参考视图（索引 0）不施加扰动，affine 恒为单位阵。
6. renderer 使用的是 `outputs["rpc_corrected"]`，不是 `rpc_gt` 或 `rpc_init`。
7. `height_abs` 与 `point_abs` 是绝对量，不是相对偏移。
8. 高斯中心有两条路径：
   - `gaussian_centers_rpc`（corrected RPC + height）
   - `gaussian_centers_point`（point_abs）

---

## 13. 常见问题排查（FAQ / Troubleshooting）

### Q1. 启动即报找不到数据或文件命名不匹配

- 检查目录是否满足 `scene_{id}/view_{id}`；
- 检查每个 view 是否存在且仅存在：
  - `image_{sid}_{vid}.tif`
  - `height_{sid}_{vid}.tif`
  - `rpc_{sid}_{vid}.txt`

### Q2. 报错 “No valid scenes after filtering min_views”

- 增加每个场景可用视图数；或调小 `data.*.min_views`。

### Q3. 报错样本内分辨率不一致

- 当前默认 `strict_same_hw=true`，请保证同一 scene 视图分辨率一致；
- 或在配置中关闭严格模式（需确认后续流程可接受）。

### Q4. TensorBoard 没有内容

- 确认 `logging.tensorboard.enable=true`；
- 确认查看目录与 `logging.tensorboard.log_dir` 一致；
- 训练步数是否达到 `log_interval/hist_interval/vis_interval`。

### Q5. DDP 启动失败

- 请使用 `torchrun --nproc_per_node=<NUM_GPUS> ...`；
- 检查 `NCCL` 环境与 GPU 可见性；
- `system.ddp_backend` 默认是 `nccl`（仅 GPU 环境建议）。

### Q6. CUDA / PyTorch 不匹配

- 优先用仓库 `env.yaml` 重建环境；
- 确认驱动支持 CUDA 12.1。

### Q7. checkpoint 恢复后形状不匹配

- 核对 `config/default.yaml` 中模型结构相关参数是否与训练时一致（如 bins、embed_dim、sh_dim 等）；
- 核对 DINO 权重与模型版本。

### Q8. 渲染过慢/显存压力大

优先调整 `renderer` 配置：

- `topk_per_target`
- `chunk_size`
- `render_downsample_factor_train/val`
- `train_num_target_views` / `val_num_target_views`
- `source_stride`
- `enable_voxelization`（及 `voxel_xy`, `voxel_z`）

### Q9. loss 出现 NaN

- 检查输入影像/高程中是否异常值；
- 检查 RPC 文件是否可正确解析；
- 先降低学习率，观察 `grad_norm`；
- 确认 `height_valid_mask` 是否合理；
- 训练配置可开启 `train.skip_nan_batch=true` 避免进程直接中断。

### Q10. 参考视图 affine 不是单位阵

- 检查 dataset 是否被错误改动；
- 检查 `reference_policy` 是否仍为 `first`；
- 检查扰动构造是否错误地作用在 ref view。

---

## 14. 致谢 / 参考

- 本项目受 **AnySplat** 多视图编码与 Gaussian 表示思路启发，并针对遥感 RPC 场景进行工程改造。
- 视觉 backbone 依赖 `third_party/dinov3`（DINOv3 本地代码/权重加载）。
- 同时参考了相关多视图几何与高斯渲染实践。

---

## 15. 许可证

仓库根目录当前未提供统一 LICENSE 文件时，请以各子模块（尤其 `third_party/`）内的许可证与上游协议为准；在对外发布或商用前请先完成许可证合规核查。

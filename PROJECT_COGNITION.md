# Sat2World 项目认知基线（Phase-0）

> 目的：在开始实现前，统一项目目标、几何约定、模块边界与实现约束。

## 1. 项目目标（明确为新系统）

Sat2World 不是对 AnySplat 的针孔相机方案做小改动，而是一个面向遥感影像与 RPC 成像模型的多视图三维重建/平差系统。核心目标为：

- 输入：多张重叠遥感 RGB 影像 + 每张影像的初始 RPC（含未平差误差）。
- 训练设定：第一张视图为参考视图（不扰动、仿射固定为单位阵）；其余视图在线施加仿射扰动模拟初始误差。
- 网络联合输出：
  - 每视图 RPC 仿射校正参数；
  - 每视图绝对高程图；
  - 每视图独立绝对点云；
  - Gaussian 属性（不含中心）：opacity / scale / rotation / color(SH) / confidence。
- 双路径 Gaussian 中心必须同时存在并参与损失：
  - Path-A：预测仿射纠正后的 RPC + 预测绝对高程；
  - Path-B：预测独立绝对点云。
- 渲染闭环：渲染必须基于“预测仿射纠正后的 RPC”成像，不以针孔投影作为主渲染路径。

## 2. 对 `rpc.py` 的理解与不可破坏约束

`rpc.py` 是项目几何核心，应作为可微几何内核使用，不重写其底层数学，仅做工程封装与批处理扩展。

关键能力（当前已具备）：

- Direct/Inverse RPC：`RPC_OBJ2PHOTO` 与 `RPC_PHOTO2OBJ`。
- 仿射校正链路：`adjust_params` / `adjust_params_inv`、`Update_Adjust`、`Merge_Adjust`。
- 统一坐标辅助：`latlon <-> yx(Web Mercator)`，以及 `RPC_XY2LINESAMP` / `RPC_LINESAMP2XY`。
- 可微几何特征：`compute_geometry_features` 输出 20 维（高度导数 + 仿射导数 + 像素导数）。

工程约束：

1. 保持现有 RPC 数学实现与方向约定，禁止“重写版 RPC”。
2. 新增功能通过 wrapper/adapter 方式提供：批处理、并行、缓存、稳定性检查。
3. 所有后续几何模块应复用该文件接口，避免重复实现投影与反投影。

## 3. 对 AnySplat 的吸收结论（可借鉴 vs. 必须替换）

### 可借鉴

- 多视图联合编码主线：
  - VGGT `Aggregator` 采用 frame/global 交替注意力，有利于显式建模“视图内 + 视图间”信息流。
- 高斯参数解耦思想：
  - Gaussian 属性头输出 opacity/scale/rotation/SH，再由 adapter 组装 Gaussians。
- voxelization + 置信度聚合：
  - 基于置信度加权的体素融合（scatter）可用于减冗余与稳定训练。
- “几何主线 + 可微渲染闭环”训练范式：
  - 编码器几何预测与渲染监督一致化。

### 必须替换

- 相机模型：AnySplat 默认针孔（extrinsic/intrinsic + gsplat 光栅）；Sat2World 必须替换为 RPC 成像链。
- 中心构造：AnySplat 通常以 depth/point 与针孔解算中心；Sat2World 必须双路径中心（RPC+高程、独立点云）。
- 输出语义：Sat2World 需要显式仿射校正头、绝对高程恢复头、独立点云头。

## 4. 特征组织与编码原则（必须执行）

- 禁止将几何特征直接拼接原图通道。
- 正确流程：
  1) patch grid 上计算 20D geometry feature；
  2) 小 MLP 编码到 256D；
  3) 与 DINOv3 1024D token 拼接；
  4) 线性层压缩回 1024D；
  5) 再送入多视图交替注意力编码器。

## 5. 数值解码原则（必须执行）

- 高程与点云不能裸回归绝对值。
- 必须采用“物理锚点 + 有界残差解码”：
  - 网络输出落在小范围残差域；
  - 解码后恢复真实绝对高程/绝对坐标（非相对量、非比例量）。
- 点云分支与高程分支独立建模，属于过完备训练设计，不可降级成附庸。
- 匹配头（matching head）当前阶段不实现。

## 6. 训练目标与损失框架（阶段目标）

数据单元：`512x512 RGB + 512x512 高程GT + 已平差RPC`。

训练时多视图输入，参考视图固定，其余视图施加仿射扰动；网络恢复扰动并联合优化：

- Affine loss：
  1) 均匀网格点组合误差；
  2) 两两视图 anchor 点在“预测仿射+预测高程”作用下的一致性误差；
  3) 线性部分近单位阵正则。
- Elevation loss：Huber。
- Point cloud loss：每视图独立，用该图高程GT + GT RPC 构造物方真值点云监督预测点云。
- Render loss：双路径均参与；早期权重低，待中心更稳定后提升。

## 7. 目录职责边界（实施蓝图）

- `configs/`：全部超参数配置化（数据/模型/阶段/渲染/默认）。
- `data/`：资源与缓存，不放核心逻辑。
- `dataset/`：多视图组织、参考视图、仿射扰动、collate、mask、scene 元信息装配。
- `geometry/`：rpc.py 封装层、投影反投影接口、patch 几何特征、锚点与重叠工具。
- `model/`：DINOv3 封装、几何嵌入、token 融合、交替注意力编码器、多头预测与解码器。
- `gaussian/`：双路径 Gaussian 构造、属性结构、体素聚合工具。
- `render/`：RPC 投影渲染（含 3D->2D 协方差投影、tile rasterizer、统一接口）。
- `loss/`：仿射/高程/点云/双路径渲染/正则/总损失。
- `engine/`：trainer/evaluator/checkpoint/scheduler/阶段切换。
- `scripts/`：训练、验证、推理、导出入口。
- `utils/`：通用工具。
- `tests/`：重点覆盖 affine 方向、RPC wrapper、geometry feature、decoder 范围、模型/渲染最小前向。

## 8. Phase-0 结论

当前阶段结论：

1. 项目主线已统一为“多视图联合编码 + RPC 几何主线 + 双路径 Gaussian + 可微渲染闭环”。
2. AnySplat 仅作为组织思想参考，不沿用其针孔成像假设。
3. 后续实现将严格遵循目录职责与边界，不在本阶段动手写功能代码。


## 9. RPC 到健康针孔拟合（工程替换原则）

- 现有 `render/rpc_gaussian_renderer.py` 中的虚拟相机拟合将替换为“归一化 DLT 初始化 + 分阶段非线性优化（A/B/C）”。
- 目标模型固定为无畸变、零 skew 的 pinhole：`K=[[fx,0,cx],[0,fy,cy],[0,0,1]]`。
- `R` 使用 axis-angle 参数化并保证 `R∈SO(3)`，焦距使用 log 参数化保证正值。
- 主点采用有界参数化（tanh/sigmoid 等）并加入中心正则，禁止主点漂移到病态位置。
- 分阶段策略：
  - A：`fx=fy` 且 `cx,cy` 固定图像中心；
  - B：释放 `cx,cy`（有界）；
  - C：释放 `fx,fy`，并对 `log(fx/fy)` 正则。
- 目标函数采用 robust least-squares（Huber/soft_l1）并叠加：主点中心正则、焦距比正则、正深度 soft barrier、弱相机中心先验。
- 拟合诊断需要输出 train/val 的 RMSE、P50、P95、max、正深度比例、健康性检查结果与失败原因。
- 验收优先级：健康性优先于单纯最小误差；若 C 阶段仍不达标，仅告警，不通过放开 skew 或极端主点“强行压误差”。

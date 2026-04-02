# Sat2World 架构阅读笔记（Agent）

该文档由代码阅读过程自动整理，概述 Sat2World 的工程主链路：

1. `dataset/`：按 scene/view 组织多视角样本，在线构造 `rpc_init` 与 affine 标签；
2. `model/`：DINOv3 + 几何特征融合 + 交替多视图编码，输出 affine/height/point/gaussian；
3. `geometry/`：封装 RPC 正反投影、patch 几何特征与中心构造；
4. `render/`：在 `rpc_corrected` 下进行双路径高斯渲染；
5. `loss/`：仿射、高程、点云、中心一致性、渲染及正则的联合目标；
6. `engine/`：DDP + AMP 训练、验证、可视化、checkpoint。

> 说明：本文件仅用于仓库内阅读辅助，不参与训练逻辑。

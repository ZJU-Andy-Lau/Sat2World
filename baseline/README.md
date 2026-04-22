# Baseline Free Network BA

该 baseline 实现了：
- 按 `dataset/perturbation.py` 构造 synthetic `rpc_init`；
- 全图 pairwise matching（SIFT/LoFTR 可插拔）+ track 构建；
- 纯自由网（无 GCP、无 DEM 先验）BBA：固定第一个视图仿射为单位阵；
- 输出每视图 correction affine（observed->true）和每条 track 的 `(x,y,h)`；
- 评估 affine 误差（含 grid 误差）和 track 高程误差。

## 运行

```bash
python scripts/run_baseline_free_ba.py \
  --scene-dir /path/to/scene_0001 \
  --save-dir outputs/baseline_scene1 \
  --view-num 4 \
  --matcher sift \
  --apply-random-init-error
```

LoFTR:

```bash
python scripts/run_baseline_free_ba.py ... --matcher loftr --matcher-weights /local/loftr.ckpt
```

# Sat2World training stages and geometry coordinate conventions

This repository currently exposes three training entrypoints:

- `scripts/early_pretrain.py`: encoder-side early pretraining. This mode forces two-view batches and skips dense decoder, affine correction, dense height, point, Gaussian, and render branches. It trains the geometry-token/fuser/encoder/NCE/patch-match/early matching/projection/height heads only.
- `scripts/geometry_train.py`: geometry training for affine, height, normalized-lat/lon point plane, correspondence, and normal-height losses. It disables Gaussian/render training paths by default.
- `scripts/train.py`: full/default training with geometry, Gaussian attributes, and RPC/point rendering paths.

## Geometry features

`RPCGeometryOps.compute_patch_geometry_features_batch` now emits a derivative-free 30-D feature vector:

1. 9 height layers of raw geodetic `lat/lon` normalized by `scene_latlon_center/scale` (18 dimensions).
2. 9 corresponding RPC-normalized heights (9 dimensions).
3. RPC-normalized `line/samp` coordinates (2 dimensions).
4. Per-view `HEIGHT_OFF / 1000` (1 dimension).

## Point branch semantics

The point branch predicts only `point_latlon_norm` with channels `[lat_norm, lon_norm]`; it does not predict height. `point_latlon_anchor` is built from `rpc_init` at `height_ref` and normalized with scene-level raw-lat/lon parameters. Height supervision is handled independently by the height heads/losses.

`scene_xy_center/scale` are legacy local/WebMercator `(y, x)` normalization fields retained for renderer local-meter paths. `scene_latlon_center/scale` are raw-degree `(lat, lon)` fields used by the new geometry features and point losses.

For Gaussian rendering, point centers are constructed by combining `point_latlon_norm` with `height_abs` and converting lat/lon degree offsets around `scene_latlon_center` to approximate local meters. Normalized lat/lon is not passed directly to the renderer as meters.

from __future__ import annotations

import numpy as np
import torch


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return inter / (area_a + area_b - inter + 1e-12)


def estimate_candidate_pairs(rpc_views, image_shapes, xy_center, xy_scale, iou_thresh: float = 0.02):
    bboxes = []
    for rpc, (h, w) in zip(rpc_views, image_shapes):
        hh = float(rpc.HEIGHT_OFF.detach().cpu().item() if torch.is_tensor(rpc.HEIGHT_OFF) else rpc.HEIGHT_OFF)
        lines = torch.tensor([0.0, 0.0, h - 1.0, h - 1.0], dtype=torch.double, device=rpc.device)
        samps = torch.tensor([0.0, w - 1.0, 0.0, w - 1.0], dtype=torch.double, device=rpc.device)
        hs = torch.full_like(lines, hh)
        x, y = rpc.RPC_LINESAMP2XY(lines, samps, hs, xy_center=xy_center, xy_scale=xy_scale)
        x = x.detach().cpu().numpy(); y = y.detach().cpu().numpy()
        bboxes.append(np.array([x.min(), y.min(), x.max(), y.max()], dtype=np.float64))

    pairs = []
    for i in range(len(rpc_views)):
        for j in range(i + 1, len(rpc_views)):
            if _bbox_iou_xyxy(bboxes[i], bboxes[j]) >= iou_thresh:
                pairs.append((i, j))
    return pairs

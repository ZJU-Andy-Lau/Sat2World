from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .base import BaseMatcher


class SIFTMatcher(BaseMatcher):
    def __init__(self, ratio_test: float = 0.75, max_keypoints: int = 8000) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV(cv2) is required for SIFT matcher")
        self.ratio_test = float(ratio_test)
        self.sift = cv2.SIFT_create(nfeatures=int(max_keypoints))

    def _to_gray_u8(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[-1] == 3:
            gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        if gray.max() <= 1.0:
            gray = gray * 255.0
        return np.clip(gray, 0, 255).astype(np.uint8)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        g0 = self._to_gray_u8(image0)
        g1 = self._to_gray_u8(image1)
        k0, d0 = self.sift.detectAndCompute(g0, None)
        k1, d1 = self.sift.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 4 or len(k1) < 4:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        m01 = bf.knnMatch(d0, d1, k=2)
        m10 = bf.knnMatch(d1, d0, k=2)

        fwd = {}
        for m in m01:
            if len(m) == 2 and m[0].distance < self.ratio_test * m[1].distance:
                fwd[m[0].queryIdx] = m[0]
        bwd = {}
        for m in m10:
            if len(m) == 2 and m[0].distance < self.ratio_test * m[1].distance:
                bwd[m[0].queryIdx] = m[0]

        rows = []
        scores = []
        for qi, m in fwd.items():
            tj = m.trainIdx
            if tj not in bwd:
                continue
            mb = bwd[tj]
            if mb.trainIdx != qi:
                continue
            p0 = k0[qi].pt
            p1 = k1[tj].pt
            rows.append([p0[1], p0[0], p1[1], p1[0]])
            scores.append(1.0 / max(m.distance, 1e-6))

        if len(rows) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32), np.asarray(scores, dtype=np.float32)

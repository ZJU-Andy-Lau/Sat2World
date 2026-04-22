from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .types import PairMatch


def _ransac_filter(matches: np.ndarray, thresh: float = 3.0) -> np.ndarray:
    if matches.shape[0] < 8:
        return np.ones((matches.shape[0],), dtype=bool)
    if cv2 is None:
        return np.ones((matches.shape[0],), dtype=bool)
    p0 = matches[:, [1, 0]].astype(np.float32)
    p1 = matches[:, [3, 2]].astype(np.float32)
    H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, ransacReprojThreshold=float(thresh))
    if H is None or inl is None:
        return np.zeros((matches.shape[0],), dtype=bool)
    return inl.reshape(-1).astype(bool)


def run_pairwise_matching(images, matcher, pairs, max_matches_per_pair: int = 2000, ransac_thresh: float = 3.0):
    out: list[PairMatch] = []
    for i, j in pairs:
        m, s = matcher.match(images[i], images[j])
        raw_n = int(m.shape[0])
        if raw_n == 0:
            out.append(PairMatch(i, j, m, s, raw_count=0))
            continue
        if s is not None and m.shape[0] > max_matches_per_pair:
            idx = np.argsort(-s)[:max_matches_per_pair]
            m = m[idx]
            s = s[idx]
        inl = _ransac_filter(m, thresh=ransac_thresh)
        m2 = m[inl]
        s2 = s[inl] if s is not None else None
        out.append(PairMatch(i, j, m2.astype(np.float32), None if s2 is None else s2.astype(np.float32), raw_count=raw_n))
    return out

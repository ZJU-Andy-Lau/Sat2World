from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def affine_2x3_to_3x3(a: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=a.dtype)
    out[:2, :] = a
    return out


def invert_affine_2x3(a: np.ndarray) -> np.ndarray:
    ah = affine_2x3_to_3x3(a)
    inv = np.linalg.inv(ah)
    return inv[:2, :]


def compose_affine_2x3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C = b o a, matching geometry.rpc Update_Adjust semantics."""
    ah = affine_2x3_to_3x3(a)
    bh = affine_2x3_to_3x3(b)
    ch = bh @ ah
    return ch[:2, :]


def pair_indices(n: int):
    return itertools.combinations(range(n), 2)


def stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "var": 0.0}
    v = values.reshape(-1).astype(np.float64)
    return {
        "count": float(v.size),
        "mean": float(v.mean()),
        "median": float(np.quantile(v, 0.5)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(v.max()),
        "var": float(v.var()),
    }


def save_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

from __future__ import annotations

from .base import BaseMatcher
from .loftr_matcher import LoFTRMatcher
from .sift_matcher import SIFTMatcher


def build_matcher(name: str, *, weights_path: str | None = None, device: str = "cuda") -> BaseMatcher:
    n = name.lower()
    if n == "sift":
        return SIFTMatcher()
    if n == "loftr":
        if not weights_path:
            raise ValueError("--matcher-weights is required when matcher=loftr")
        return LoFTRMatcher(weights_path=weights_path, device=device)
    raise ValueError(f"Unknown matcher: {name}")

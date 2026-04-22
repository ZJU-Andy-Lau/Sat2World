from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseMatcher(ABC):
    @abstractmethod
    def match(self, image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (matches[N,4], scores[N] or None) in (line0, samp0, line1, samp1)."""
        raise NotImplementedError

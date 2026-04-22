from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .base import BaseMatcher


class LoFTRMatcher(BaseMatcher):
    def __init__(self, weights_path: str, device: str = "cuda") -> None:
        try:
            from kornia.feature import LoFTR
        except Exception as e:  # pragma: no cover
            raise RuntimeError("kornia.feature.LoFTR is required for loftr matcher") from e

        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"LoFTR weights not found: {weights_path}")
        self.device = torch.device(device)
        self.model = LoFTR(pretrained=None)
        ckpt = torch.load(str(p), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    def _to_gray_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 3 and image.shape[-1] == 3:
            gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
        else:
            gray = image
        gray = gray.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0
        return torch.from_numpy(gray)[None, None].to(self.device)

    @torch.no_grad()
    def match(self, image0: np.ndarray, image1: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        t0 = self._to_gray_tensor(image0)
        t1 = self._to_gray_tensor(image1)
        out = self.model({"image0": t0, "image1": t1})
        mk0 = out["keypoints0"].detach().cpu().numpy()
        mk1 = out["keypoints1"].detach().cpu().numpy()
        conf = out.get("confidence", None)
        if mk0.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        m = np.stack([mk0[:, 1], mk0[:, 0], mk1[:, 1], mk1[:, 0]], axis=1).astype(np.float32)
        scores = conf.detach().cpu().numpy().astype(np.float32) if conf is not None else None
        return m, scores

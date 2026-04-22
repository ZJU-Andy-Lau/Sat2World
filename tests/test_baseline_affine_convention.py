from __future__ import annotations

import numpy as np
import torch

from dataset.perturbation import identity_affine_2x3, invert_affine_2x3
from loss.common import apply_affine_to_points


def test_affine_forward_correction_inverse_roundtrip():
    fwd = torch.tensor([[1.001, 0.0002, 3.0], [-0.0003, 0.999, -2.0]], dtype=torch.float64)
    corr = invert_affine_2x3(fwd)
    p_true = torch.tensor([[100.0, 200.0], [20.0, 30.0]], dtype=torch.float64)
    p_obs = apply_affine_to_points(p_true, fwd)
    p_rec = apply_affine_to_points(p_obs, corr)
    assert torch.allclose(p_true, p_rec, atol=1e-6)


def test_identity_affine_shape_and_values():
    eye = identity_affine_2x3(dtype=torch.float64)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(eye.numpy(), exp)

"""Fixed 244-parameter offline selector and its distinct emitted prediction.

The training objective is expected *individual-expert* endpoint20 angular
loss.  It is deliberately not the angular error of the blended prediction.
Only the latter is reported as head-orientation accuracy.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import torch
from torch import nn

from .geometry import exp_so3, multiply_wxyz, normalize_wxyz


VARIANTS = ("h1_a0", "h20_a0", "h20_a20", "complete_horizon", "motion_only", "h1_a19")
SEEDS = (20261026, 20261027)
ARCHITECTURE = (10, 16, 4)
PARAMETERS = 244
DENSE_MACS = 224


class SelectorMLP(nn.Module):
    """Ten normalized causal features ->16 ReLU ->4 simplex weights.

    Initialization uses a local CPU RNG scope, so constructing this module does
    not change a caller's RNG stream. Every variant with the same seed starts
    from exactly the same tensors. No inference-time fitting is implemented.
    """

    def __init__(self, seed=SEEDS[0]):
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("selector initialization seed must be an integer")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.hidden = nn.Linear(10, 16, dtype=torch.float32, device="cpu")
            self.output = nn.Linear(16, 4, dtype=torch.float32, device="cpu")
        # Match the earlier NumPy selector prototype's orientation and draws.
        rng = np.random.default_rng(int(seed))
        with torch.no_grad():
            self.hidden.weight.copy_(torch.from_numpy(rng.normal(0, .02, (10, 16)).astype(np.float32).T))
            self.hidden.bias.zero_()
            self.output.weight.copy_(torch.from_numpy(rng.normal(0, .02, (16, 4)).astype(np.float32).T))
            self.output.bias.zero_()
        if sum(p.numel() for p in self.parameters()) != PARAMETERS:
            raise AssertionError("selector parameter contract changed")

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 10 or not features.is_floating_point():
            raise ValueError("selector requires floating [B,10] normalized features")
        return torch.softmax(self.output(torch.relu(self.hidden(features))), dim=1)


def normalise_features(raw_features, scaler):
    """Apply an existing train-only population scaler, then round to float32.

    Float64 subtraction/division is intentional and part of the experiment
    contract. This helper never estimates, updates or fits scaler statistics.
    """
    raw = np.asarray(raw_features, dtype=np.float64)
    mean = np.asarray(scaler.get("mean"), dtype=np.float64)
    scale = np.asarray(scaler.get("scale"), dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != 10 or mean.shape != (10,) or scale.shape != (10,)
            or not np.isfinite(raw).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale < 1e-8)
            or scaler.get("fit_partition") != "train" or scaler.get("ddof") != 0
            or scaler.get("epsilon") != 1e-8):
        raise ValueError("normalization requires ten finite train-only scaler columns")
    result = ((raw - mean) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalized selector features overflow float32")
    return np.ascontiguousarray(result)


def expected_expert_loss(weights, individual_expert_loss):
    """Differentiable batch mean of sum_m w_m * endpoint20_loss_m(degrees)."""
    if (weights.ndim != 2 or weights.shape[1] != 4 or not len(weights)
            or individual_expert_loss.shape != weights.shape
            or weights.device != individual_expert_loss.device
            or not weights.is_floating_point() or not individual_expert_loss.is_floating_point()):
        raise ValueError("surrogate requires aligned floating [B,4] weights and future labels")
    return torch.sum(weights * individual_expert_loss, dim=1).mean()


def predict_blended_quaternions(weights, velocities, current_q):
    """Emit all 20 float32 wxyz outputs from the blended *world* velocity.

    This is Exp(k * sum_m w_m v_m) * q_t, for k=1..20; it does not average
    quaternions or individual expert errors. Geometry is evaluated in float64
    and predictions are quantized to the graph's float32 interface.
    """
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    q = np.asarray(current_q)
    if (w.ndim != 2 or w.shape[1] != 4 or not len(w)
            or v.shape != (len(w), 4, 3) or q.shape != (len(w), 4)
            or not np.isfinite(w).all() or not np.isfinite(v).all()
            or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("prediction requires aligned simplex weights, four velocities and current quaternions")
    q = normalize_wxyz(q)
    velocity = np.sum(v * w[:, :, None], axis=1)
    rotations = exp_so3(np.arange(1, 21, dtype=np.float64)[None, :, None] * velocity[:, None, :])
    return normalize_wxyz(multiply_wxyz(rotations, q[:, None, :])).astype(np.float32)

"""Ordered-motion SO(3) model prototype; architecture only, no data access.

The geometry follows the 2026-10-29 reviewed architecture proposal. Quaternion
convention is normalized active-Hamilton wxyz. All sample widths are fixed
sample counts; this module makes no Hz/ms conversion and preserves the official
20-output ambiguity. The implementation is stateless and contains no fit,
optimizer, I/O, or online adaptation code.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import nn


WIDTHS: Final[tuple[int, ...]] = (8, 16, 32, 64, 128, 256, 399)
MEAN_EPS: Final[float] = 1.0e-6
AREA_EPS: Final[float] = 1.0e-10
HIDDEN_SEED: Final[int] = 20261029


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize wxyz quaternions with finite zero subgradient.

    Adapted from the reviewed ARC pilot geometry in
    ``ABC_IUEH/src/abc_iueh/arc_model.py``; the source file is not modified.
    Contract inputs are finite, nonzero quaternions. The scaling step avoids
    overflowing the squared norm on moderately nonunit intermediates.
    """
    scale = q.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-30)
    scaled = q / scale
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True).clamp_min(1.0e-30)


def multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product with broadcastable leading dimensions."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def cross3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3D cross product expanded into basic Mul/Sub/Concat operations."""
    ax, ay, az = a.unbind(dim=-1)
    bx, by, bz = b.unbind(dim=-1)
    return torch.stack((ay * bz - az * by,
                        az * bx - ax * bz,
                        ax * by - ay * bx), dim=-1)


def log_so3(q: torch.Tensor) -> torch.Tensor:
    """Principal SO(3) logarithm in radians with sign-invariant quaternions.

    Uses the reviewed ARC pilot canonical-sign rule, including deterministic
    tie handling at an exact half turn. PyTorch vector_norm has a finite zero
    subgradient; the division is protected even in the unselected branch.
    """
    q = normalize_quaternion(q)
    scalar, vector = q[..., :1], q[..., 1:]
    first = torch.where(vector[..., :1] != 0, vector[..., :1],
                        torch.where(vector[..., 1:2] != 0, vector[..., 1:2], vector[..., 2:3]))
    negative = (scalar < 0) | ((scalar == 0) & (first < 0))
    q = q * torch.where(negative, -1.0, 1.0)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, q[..., :1].abs())
    factor = torch.where(sine > 0, angle / sine.clamp_min(1.0e-30), 2.0)
    return q[..., 1:] * factor


def exp_so3(v: torch.Tensor) -> torch.Tensor:
    """SO(3) exponential with a safe small-angle branch and denominator."""
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle2 = angle * angle
    small_factor = 0.5 - angle2 / 48.0 + angle2 * angle2 / 3840.0
    factor = torch.where(angle > 1.0e-4,
                         torch.sin(angle * 0.5) / angle.clamp_min(1.0e-12),
                         small_factor)
    quat = torch.cat((torch.cos(angle * 0.5), v * factor), dim=-1)
    return normalize_quaternion(quat)


def geodesic_radians(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SO(3) angular distance; sign-invariant and stable near coincidence."""
    delta = multiply_wxyz(normalize_quaternion(a), conjugate_wxyz(normalize_quaternion(b)))
    sine = torch.linalg.vector_norm(delta[..., 1:], dim=-1)
    return 2.0 * torch.atan2(sine, delta[..., 0].abs())


def endpoint20_geodesic_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean endpoint-20 geodesic loss in radians for [B,20,4] targets."""
    if prediction.shape[-2:] != (20, 4) or target.shape != prediction.shape:
        raise ValueError("prediction and target must both have shape [B,20,4]")
    return geodesic_radians(prediction[:, -1], target[:, -1]).mean()


def ordered_area_balanced(increments: torch.Tensor, padded_to: int | None = None) -> torch.Tensor:
    """Return W^-2 sum_(i<j) u_i×u_j via a fixed balanced pair reduction.

    ``increments`` is chronological [B,W,3]. Optional right zero-padding only
    supplies identity leaves, so it cannot alter the result. Pairwise levels
    are Python-unrolled for static export: no ONNX Loop/Scan or data-dependent
    control flow is required. The merge law is (P,A)*(Q,B)=(P+Q,A+B+P×Q).
    """
    if increments.ndim != 3 or increments.shape[-1] != 3:
        raise ValueError("increments must have shape [B,W,3]")
    width = increments.shape[1]
    if width < 1:
        raise ValueError("increments must contain at least one vector")
    target = padded_to if padded_to is not None else 1 << (width - 1).bit_length()
    if target < width or target & (target - 1):
        raise ValueError("padded_to must be a power of two at least W")
    if target != width:
        zeros = increments.new_zeros((increments.shape[0], target - width, 3))
        increments = torch.cat((increments, zeros), dim=1)
    p = increments
    area = torch.zeros_like(increments)
    count = target
    while count > 1:
        p_left, p_right = p[:, 0::2], p[:, 1::2]
        a_left, a_right = area[:, 0::2], area[:, 1::2]
        area = a_left + a_right + cross3(p_left, p_right)
        p = p_left + p_right
        count //= 2
    return area[:, 0] / float(width * width)


def _smooth_norm(x: torch.Tensor, eps: float, scale: float) -> torch.Tensor:
    # Positive epsilon keeps both forward values and input derivatives finite.
    return scale * (torch.sqrt((x * x).sum(dim=-1) + eps * eps) - eps)


def _smooth_unit(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps * eps)


class OrderedMotionFeatures(nn.Module):
    """Fixed 399-increment geometry and 30 rotation-invariant scalars."""

    def __init__(self, erase_internal_order: bool = False):
        super().__init__()
        self.erase_internal_order = erase_internal_order

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing():
            if context.ndim != 3 or context.shape[1:] != (400, 4):
                raise ValueError("context must have shape [B,400,4]")
            if context.dtype != torch.float32:
                raise TypeError("context must be float32")

        q = normalize_quaternion(context)
        # q_i q_(i-1)^-1 is a world increment; conjugation moves each one to
        # the current body's frame. q_t is the newest observed orientation.
        world_increments = multiply_wxyz(q[:, 1:], conjugate_wxyz(q[:, :-1]))
        q_current = q[:, -1]
        body_increments = log_so3(multiply_wxyz(
            multiply_wxyz(conjugate_wxyz(q_current[:, None]), world_increments),
            q_current[:, None]))

        latest = body_increments[:, -1]
        means = [body_increments[:, -width:].mean(dim=1) for width in WIDTHS]
        mean_channels = torch.stack((latest, *means), dim=1)

        ordered_areas = []
        for width in WIDTHS:
            tree_size = 1 << (width - 1).bit_length()
            ordered_areas.append(ordered_area_balanced(
                body_increments[:, -width:], padded_to=tree_size))
        areas = torch.stack(ordered_areas, dim=1)
        if self.erase_internal_order:
            # Active order-erased control: the same seven vector slots use
            # latest×suffix-mean substitutes and all their scalar features
            # are recomputed. This preserves shapes and parameter count.
            means_tensor = torch.stack(means, dim=1)
            latest_per_width = latest[:, None, :].expand_as(means_tensor)
            areas = cross3(latest_per_width, means_tensor)

        mean_magnitudes = _smooth_norm(mean_channels, MEAN_EPS, 100.0)
        area_magnitudes = _smooth_norm(areas, AREA_EPS, 10000.0)
        latest_unit = _smooth_unit(latest, MEAN_EPS)
        mean_units = _smooth_unit(torch.stack(means, dim=1), MEAN_EPS)
        area_units = _smooth_unit(areas, AREA_EPS)
        latest_mean_dots = (latest_unit[:, None, :] * mean_units).sum(dim=-1)
        latest_mean_area_triples = (latest_unit[:, None, :] * cross3(mean_units, area_units)).sum(dim=-1)
        recent_mean = body_increments[:, -32:].mean(dim=1, keepdim=True)
        recent_dispersion = torch.sqrt(
            ((body_increments[:, -32:] - recent_mean) ** 2).sum(dim=-1).mean(dim=1, keepdim=True)
            + MEAN_EPS * MEAN_EPS) - MEAN_EPS
        recent_dispersion = 100.0 * recent_dispersion
        scalars = torch.cat((mean_magnitudes, area_magnitudes, latest_mean_dots,
                             latest_mean_area_triples, recent_dispersion), dim=1)
        normalized_vectors = torch.cat((_smooth_unit(mean_channels, MEAN_EPS),
                                        _smooth_unit(areas, AREA_EPS)), dim=1)
        return q_current, latest, normalized_vectors, scalars


def _deterministic_mlp(in_features: int, hidden: int, out_features: int,
                       seed: int = HIDDEN_SEED) -> nn.Sequential:
    """Create deterministic nonzero hidden weights and a zero output layer."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        first = nn.Linear(in_features, hidden)
        final = nn.Linear(hidden, out_features)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return nn.Sequential(first, nn.ReLU(), final)


class _OrderedMotionBase(nn.Module):
    parameter_count_expected: int = 2477

    def __init__(self, erase_internal_order: bool = False, single_exponential: bool = False,
                 seed: int = HIDDEN_SEED):
        super().__init__()
        self.features = OrderedMotionFeatures(erase_internal_order=erase_internal_order)
        self.head = _deterministic_mlp(30, 32, 45, seed=seed)
        self.single_exponential = single_exponential
        self.register_buffer("s", torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1) / 20.0)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _decode(self, q_current: torch.Tensor, latest: torch.Tensor, vectors: torch.Tensor,
                output: torch.Tensor) -> torch.Tensor:
        coefficients = 0.1 * torch.tanh(output.view(-1, 3, 15))
        corrections = torch.bmm(coefficients, vectors)
        b1 = 20.0 * latest + corrections[:, 0]
        b2, b3 = corrections[:, 1], corrections[:, 2]
        s = self.s.to(dtype=q_current.dtype, device=q_current.device)
        if self.single_exponential:
            tangent = s * b1[:, None, :] + (s * s) * b2[:, None, :] + (s * s * s) * b3[:, None, :]
            prediction = multiply_wxyz(q_current[:, None], exp_so3(tangent))
        else:
            factor1 = exp_so3(s * b1[:, None, :])
            factor2 = exp_so3((s * s) * b2[:, None, :])
            factor3 = exp_so3((s * s * s) * b3[:, None, :])
            prediction = multiply_wxyz(
                multiply_wxyz(multiply_wxyz(q_current[:, None], factor1), factor2), factor3)
        return normalize_quaternion(prediction)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        q_current, latest, vectors, scalars = self.features(context)
        return self._decode(q_current, latest, vectors, self.head(scalars))


class OrderedMotionProduct(_OrderedMotionBase):
    """Primary architecture: ordered areas and three-factor Exp decoder."""


class MeansOnlyControl(_OrderedMotionBase):
    """2,477-param order-erased control using latest×suffix-mean areas."""

    def __init__(self, seed: int = HIDDEN_SEED):
        super().__init__(erase_internal_order=True, single_exponential=False, seed=seed)


class SingleExponentialControl(_OrderedMotionBase):
    """2,477-param same-head control with one Exp of the summed tangent."""

    def __init__(self):
        super().__init__(erase_internal_order=False, single_exponential=True)


class DirectSameFeatureControl(nn.Module):
    """2,474-param free-vector control over the same 30 scalars + 45 vectors.

    The 75→29→9 MLP emits three bounded free 3-vectors. B1 adds its correction
    to the same 20*m_latest CAV base; B2/B3 are free vectors. It keeps the
    three-factor product decoder but intentionally gives up right equivariance.
    """

    parameter_count_expected = 2474

    def __init__(self):
        super().__init__()
        self.features = OrderedMotionFeatures()
        self.head = _deterministic_mlp(75, 29, 9, seed=HIDDEN_SEED + 1)
        self.register_buffer("s", torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1) / 20.0)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        q_current, latest, vectors, scalars = self.features(context)
        free = 0.1 * torch.tanh(self.head(torch.cat((scalars, vectors.flatten(start_dim=1)), dim=1)))
        b1 = 20.0 * latest + free[:, 0:3]
        b2, b3 = free[:, 3:6], free[:, 6:9]
        s = self.s.to(dtype=q_current.dtype, device=q_current.device)
        factor1 = exp_so3(s * b1[:, None, :])
        factor2 = exp_so3((s * s) * b2[:, None, :])
        factor3 = exp_so3((s * s * s) * b3[:, None, :])
        prediction = multiply_wxyz(
            multiply_wxyz(multiply_wxyz(q_current[:, None], factor1), factor2), factor3)
        return normalize_quaternion(prediction)


PARAMETER_COUNTS: Final[dict[str, int]] = {
    "ordered_motion_product": 2477,
    "means_only_order_erased": 2477,
    "single_exponential_same_head": 2477,
    "direct_same_feature": 2474,
}


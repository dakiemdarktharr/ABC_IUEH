"""Past-only expert evidence and motion features for one continuous segment.

All returned rows use the same 400-observation context and endpoint20 label.
Error features replay only forecasts whose targets have already arrived. Future
endpoint errors are returned separately as labels and never enter a feature.
No feature scaling is fitted here. Speeds are radians/sample; errors are degrees.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    exp_so3, geodesic_deg, inverse_wxyz, log_so3, multiply_wxyz, normalize_wxyz,
)


EXPERT_NAMES = ("ZV", "CAV_W1", "CAV_W5", "CAV_W20")
EVIDENCE_VARIANTS = {
    "h1_a0": (1, 0),
    "h20_a0": (20, 0),
    "h1_a19": (1, 19),
    "h20_a20": (20, 20),
}
_BANDWIDTHS = (1, 5, 20)
_PAIRS = ((0, 1), (0, 2), (1, 2))
_FIRST_VALID_EXPERT_ORIGIN = 20
_CONTEXT = 400
_HORIZON = 20


def _prefix_sum(errors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prefix only defined errors: unavailable early origins never add NaNs."""
    prefix = np.empty((len(errors) + 1, 4), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(errors, axis=0, dtype=np.float64, out=prefix[1:])
    return prefix


def _past_mean(
    prefix: NDArray[np.float64], current: NDArray[np.int64],
    origin_age: int, length: int,
) -> NDArray[np.float64]:
    """Average R forecast origins ending at t-origin_age, inclusively."""
    stop = current - origin_age - _FIRST_VALID_EXPERT_ORIGIN + 1
    start = stop - length
    if np.any(start < 0) or np.any(stop >= len(prefix)):
        raise ValueError("evidence history is unavailable inside the supplied context")
    return (prefix[stop] - prefix[start]) / length


def _motion_columns(
    q: NDArray[np.float64], origins: NDArray[np.int64],
    current_velocity: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    moving = current_velocity[:, 1:, :]
    speed = np.linalg.norm(moving, axis=-1)
    cosines = np.column_stack([
        np.clip(
            np.sum(moving[:, i] * moving[:, j], axis=-1)
            / np.maximum(speed[:, i] * speed[:, j], 1e-12),
            -1.0, 1.0,
        )
        for i, j in _PAIRS
    ])
    differences = np.column_stack([
        np.linalg.norm(moving[:, i] - moving[:, j], axis=-1)
        for i, j in _PAIRS
    ])
    net_rotation = multiply_wxyz(q[origins], inverse_wxyz(q[origins - 100]))
    net_speed = np.linalg.norm(log_so3(net_rotation), axis=-1, keepdims=True) / 100.0
    common = np.concatenate((speed, cosines), axis=1)
    motion_only = np.concatenate((common, differences, net_speed), axis=1)
    return common, motion_only


def build_segment_features(
    q: NDArray[np.float32], evidence_length: int = 80,
    include_complete_horizon: bool = True,
) -> dict:
    """Build exact replay evidence in O(N) time per requested horizon.

    ``q`` must be a finite, nonzero float32 ``(N,4)`` wxyz segment. Rotations
    are normalized in float64 for arithmetic. Every emitted prediction is
    normalized and quantized to float32 *before* its float64 geodesic error,
    matching the fixed-expert inference interface.

    ``origin_index`` is399..N-21 inclusive. Each evidence feature has columns
    [errors ZV/W1/W5/W20, speeds W1/W5/W20, cosines W1-W5/W1-W20/W5-W20].
    Motion-only columns are [the same speeds/cosines, three pairwise velocity
    difference norms, ||Log(q[t]*inverse(q[t-100]))||/100].

    Complete-Horizon averages k1..20 at *the same origins* t-20-r, then averages
    R origins. It shares the h1/h20 error passes with individual-horizon scores.
    Only one horizon's (N,4) errors is materialized at once; no (N,20,4) cache or
    overlapping-context table is constructed. The returned arrays themselves
    occupy O(N) space. R<=340 keeps h20,a20,W20 inside400 past observations.
    """
    if isinstance(evidence_length, bool) or not isinstance(evidence_length, Integral):
        raise ValueError("evidence_length must be an integer")
    evidence_length = int(evidence_length)
    if not 1 <= evidence_length <= 340:
        raise ValueError("evidence_length must be1..340 to fit all variants in400 past samples")
    source = np.asarray(q)
    if source.ndim != 2 or source.shape[1] != 4 or source.dtype != np.float32:
        raise ValueError("q must be float32 with shape (N,4)")
    unit = normalize_wxyz(source)
    origins = np.arange(_CONTEXT - 1, max(_CONTEXT - 1, len(source) - _HORIZON), dtype=np.int64)
    names = list(EVIDENCE_VARIANTS)
    if include_complete_horizon:
        names.append("complete_horizon")
    names.append("motion_only")
    if not len(origins):
        return {
            "origin_index": origins,
            "features": {name: np.empty((0, 10), dtype=np.float32) for name in names},
            "velocities": np.empty((0, 4, 3), dtype=np.float64),
            "current_q": np.empty((0, 4), dtype=np.float32),
            "target_q": np.empty((0, 4), dtype=np.float32),
            "expert_loss": np.empty((0, 4), dtype=np.float64),
        }

    # Every cached expert origin is >=20, so all three bandwidths are defined.
    valid_velocity = np.zeros((len(unit) - 20, 4, 3), dtype=np.float64)
    for expert, bandwidth in enumerate(_BANDWIDTHS, start=1):
        relative = multiply_wxyz(unit[20:], inverse_wxyz(unit[20 - bandwidth:len(unit) - bandwidth]))
        valid_velocity[:, expert] = log_so3(relative) / bandwidth
    current_velocity = valid_velocity[origins - 20].copy()
    common_motion, motion_only = _motion_columns(unit, origins, current_velocity)
    features = {"motion_only": motion_only.astype(np.float32)}
    complete_errors = np.zeros((len(unit) - 40, 4), dtype=np.float64) if include_complete_horizon else None
    expert_loss = None
    for horizon in (range(1, 21) if include_complete_horizon else (1, 20)):
        count = len(unit) - 20 - horizon
        prediction = normalize_wxyz(multiply_wxyz(
            exp_so3(horizon * valid_velocity[:count]), unit[20:len(unit) - horizon, None, :],
        )).astype(np.float32)
        errors = geodesic_deg(prediction, source[20 + horizon:, None, :])
        if complete_errors is not None:
            complete_errors += errors[:len(complete_errors)]
        if horizon in (1, 20):
            prefix = _prefix_sum(errors)
            for name, (evidence_horizon, target_age) in EVIDENCE_VARIANTS.items():
                if evidence_horizon == horizon:
                    score = _past_mean(prefix, origins, horizon + target_age, evidence_length)
                    features[name] = np.concatenate((score, common_motion), axis=1).astype(np.float32)
            if horizon == 20:
                expert_loss = errors[origins - 20].copy()
        # Do not keep a20-horizon history: the next pass can reuse this memory.
        del prediction, errors
    if complete_errors is not None:
        complete_errors /= 20.0
        score = _past_mean(_prefix_sum(complete_errors), origins, 20, evidence_length)
        features["complete_horizon"] = np.concatenate((score, common_motion), axis=1).astype(np.float32)
    return {
        "origin_index": origins,
        "features": {name: features[name] for name in names},
        "velocities": current_velocity,
        "current_q": source[origins].copy(),
        "target_q": source[origins + 20].copy(),
        "expert_loss": expert_loss,
    }

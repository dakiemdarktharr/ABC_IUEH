"""Rotation geometry using active Hamilton quaternions in scalar-first order.

All public functions use ``(..., 4)`` wxyz arrays; rotation vectors are radians.
SciPy receives an explicit wxyz-to-xyzw permutation at its boundary. SLERP is
offline interpolation between bracketing native observations, not a claim of
zero-latency native-sensor processing.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp


def _components(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalize_wxyz(q: ArrayLike) -> NDArray[np.float64]:
    """Normalize rotations, rejecting zero/nonfinite quaternions.

    Scaling before computing the norm avoids overflow/underflow for finite
    nonzero input. The preprocessing layer applies its separate norm epsilon.
    """
    array = _components(q, 4, "quaternion")
    scale = np.max(np.abs(array), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("quaternion norm must be nonzero")
    scaled = array / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def align_hemisphere(q: ArrayLike) -> NDArray[np.float64]:
    """Normalize an ``(N, 4)`` sequence and choose adjacent short-arc signs."""
    array = normalize_wxyz(q)
    if array.ndim != 2:
        raise ValueError("hemisphere alignment requires shape (N, 4)")
    if len(array) < 2:
        return array.copy()
    adjacent = np.sum(array[1:] * array[:-1], axis=-1)
    signs = np.r_[1.0, np.cumprod(np.where(adjacent < 0, -1.0, 1.0))]
    return array * signs[:, None]


def multiply_wxyz(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``a ⊗ b`` with NumPy leading-dimension broadcasting.

    For active rotations, ``a ⊗ b`` applies ``b`` first, then ``a``. Inputs are
    checked but not normalized, preserving the algebraic quaternion product.
    """
    left = _components(a, 4, "left quaternion")
    right = _components(b, 4, "right quaternion")
    left, right = np.broadcast_arrays(left, right)
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True
    )
    vector = (
        left[..., :1] * right[..., 1:]
        + right[..., :1] * left[..., 1:]
        + np.cross(left[..., 1:], right[..., 1:])
    )
    return np.concatenate((scalar, vector), axis=-1)


def inverse_wxyz(q: ArrayLike) -> NDArray[np.float64]:
    """Return the inverse *rotation*, normalizing a nonunit input first."""
    result = normalize_wxyz(q).copy()
    result[..., 1:] *= -1
    return result


def log_so3(q: ArrayLike) -> NDArray[np.float64]:
    """Principal SO(3) logarithm, a rotation vector with norm at most pi.

    At an exact half turn the sign is intrinsically ambiguous; the first
    nonzero vector component is chosen positive for deterministic q/-q parity.
    """
    array = normalize_wxyz(q)
    sign = np.where(array[..., 0] < 0, -1.0, 1.0)
    first_nonzero = np.argmax(array[..., 1:] != 0, axis=-1)
    first_value = np.take_along_axis(
        array[..., 1:], first_nonzero[..., None], axis=-1
    )[..., 0]
    sign = np.where((array[..., 0] == 0) & (first_value < 0), -1.0, sign)
    array = array * sign[..., None]
    sine = np.linalg.norm(array[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(sine, array[..., 0])
    factor = np.divide(angle, sine, out=np.full_like(sine, 2.0), where=sine > 0)
    return array[..., 1:] * factor[..., None]


def exp_so3(v: ArrayLike) -> NDArray[np.float64]:
    """SO(3) exponential of radians, using the analytic limit at zero."""
    vector = _components(v, 3, "rotation vector")
    scale = np.max(np.abs(vector), axis=-1, keepdims=True)
    scaled = np.divide(vector, scale, out=np.zeros_like(vector), where=scale > 0)
    angle = scale[..., 0] * np.linalg.norm(scaled, axis=-1)
    if not np.all(np.isfinite(angle)):
        raise ValueError("rotation-vector magnitude is too large")
    factor = 0.5 * np.sinc(angle / (2.0 * np.pi))
    return normalize_wxyz(
        np.concatenate((np.cos(angle / 2.0)[..., None], vector * factor[..., None]), axis=-1)
    )


def geodesic_deg(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Shortest SO(3) angle in degrees, invariant to either quaternion sign.

    The atan2 form agrees with ``2*acos(abs(dot(a,b)))`` and remains accurate
    when rotations are almost identical, where acos loses precision.
    """
    delta = multiply_wxyz(normalize_wxyz(a), inverse_wxyz(b))
    return np.degrees(
        2.0 * np.arctan2(np.linalg.norm(delta[..., 1:], axis=-1), np.abs(delta[..., 0]))
    )


def resample_slerp(
    t: ArrayLike, q: ArrayLike, target_rate_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Uniform grid anchored at the first timestamp, without extrapolation.

    Call this separately for each gap-free segment. Timestamps are seconds;
    sorting, clock-reset handling and native-rate eligibility belong upstream.
    """
    timestamps = np.asarray(t, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("SLERP requires at least two one-dimensional timestamps")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("SLERP timestamps must be finite and strictly increasing")
    quaternion = align_hemisphere(q)
    if quaternion.shape != (len(timestamps), 4):
        raise ValueError("SLERP quaternion shape must be (len(timestamps), 4)")
    if not np.isfinite(target_rate_hz) or target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be finite and positive")
    relative = timestamps - timestamps[0]
    steps = relative[-1] * target_rate_hz
    # Include an endpoint represented one ULP below an exact grid boundary.
    count = int(np.floor(np.nextafter(steps, np.inf))) + 1
    offsets = np.arange(count, dtype=np.float64) / target_rate_hz
    query = np.minimum(offsets, relative[-1])
    rotations = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
    xyzw = Slerp(relative, rotations)(query).as_quat()
    result = align_hemisphere(xyzw[:, [3, 0, 1, 2]])
    return timestamps[0] + offsets, result.astype(np.float32)

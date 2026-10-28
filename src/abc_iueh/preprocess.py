"""Auditable session preprocessing; team choices stay explicit in metadata.

Filtering uses the entire valid trimmed session before the duration cap. Euler
angles follow SciPy's active, extrinsic ``xyz`` convention. Sweep is the raw
90th-minus-10th percentile without unwrapping; only velocity uses wrapped
shortest differences. These convention/tail/gap choices are not organizer
clarifications. Offline SLERP uses bracketing observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .config import PipelineConfig
from .geometry import align_hemisphere, resample_slerp


@dataclass
class PreparedSegment:
    timestamps: NDArray[np.float64]
    quaternion: NDArray[np.float32]


@dataclass
class PreparedSession:
    accepted: bool
    reason: str
    stats: dict
    segments: list[PreparedSegment] = field(default_factory=list)


def _clock_tolerance(timestamps: NDArray[np.float64]) -> float:
    return max(1e-12, 8.0 * float(np.spacing(np.max(np.abs(timestamps)))))


def _validate_config(config: PipelineConfig) -> None:
    positive = (
        "target_rate_hz", "max_gap_seconds", "norm_epsilon",
        "filter_window_seconds", "session_cap_seconds",
    )
    for name in positive:
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(config.trim_seconds) or config.trim_seconds < 0:
        raise ValueError("trim_seconds must be finite and nonnegative")
    for name in ("sweep_threshold_deg", "velocity_threshold_deg_s"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config.min_filter_window_samples < 2:
        raise ValueError("min_filter_window_samples must be at least two")


def _motion_filter(
    timestamps: NDArray[np.float64], quaternion: NDArray[np.float64],
    trim_origin: float, recording_end: float, config: PipelineConfig,
    tolerance: float,
) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        euler = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
    window_seconds = config.filter_window_seconds
    window_indices = np.floor((timestamps - trim_origin) / window_seconds).astype(np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(window_indices)) + 1, len(timestamps)]
    sweeps, velocities = [], []
    short_windows = partial_windows = dropped_partial = masked_pairs = 0
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        nominal_start = trim_origin + int(window_indices[begin]) * window_seconds
        full = nominal_start + window_seconds <= recording_end + tolerance
        if not full:
            partial_windows += 1
            if config.tail_window_policy == "drop":
                dropped_partial += 1
                continue
        if end - begin < config.min_filter_window_samples:
            short_windows += 1
            continue
        duration = window_seconds if full else float(timestamps[end - 1] - timestamps[begin])
        if duration <= 0:
            short_windows += 1
            continue
        angles = euler[begin:end]
        sweeps.append(np.percentile(angles, 90, axis=0) - np.percentile(angles, 10, axis=0))
        difference = np.abs(np.diff(angles, axis=0))
        difference = np.minimum(difference, 360.0 - difference)
        contiguous = np.diff(timestamps[begin:end]) <= config.max_gap_seconds + tolerance
        masked_pairs += int(np.count_nonzero(~contiguous))
        velocities.append(difference[contiguous].sum(axis=0) / duration)
    result = {
        "euler_convention": "scipy_active_extrinsic_xyz_degrees",
        "sweep_unwrap": False,
        "velocity_convention": "shortest_wrapped_axis_differences_no_gap_crossing",
        "full_window_denominator": "nominal_window_seconds",
        "partial_window_denominator": "last_valid_minus_first_valid_timestamp",
        "window_origin_seconds": float(trim_origin),
        "window_seconds": float(window_seconds),
        "tail_window_policy": config.tail_window_policy,
        "valid_windows": len(sweeps),
        "short_windows": short_windows,
        "partial_windows": partial_windows,
        "dropped_partial_windows": dropped_partial,
        "masked_gap_velocity_pairs": masked_pairs,
        "gimbal_lock_warnings": sum("Gimbal lock" in str(item.message) for item in caught),
        "threshold_operator": config.threshold_operator,
        "sweep_threshold_deg": float(config.sweep_threshold_deg),
        "velocity_threshold_deg_s": float(config.velocity_threshold_deg_s),
    }
    if not sweeps:
        result.update({"passed": False, "failure": "no_valid_filter_windows"})
        return result
    median_sweep = np.median(sweeps, axis=0)
    median_velocity = np.median(velocities, axis=0)
    sweep, velocity = float(np.max(median_sweep)), float(np.max(median_velocity))
    gt = sweep > config.sweep_threshold_deg and velocity > config.velocity_threshold_deg_s
    ge = sweep >= config.sweep_threshold_deg and velocity >= config.velocity_threshold_deg_s
    result.update({
        "median_sweep_per_axis_deg": median_sweep.tolist(),
        "median_velocity_per_axis_deg_s": median_velocity.tolist(),
        "sweep_deg": sweep,
        "velocity_deg_s": velocity,
        "sweep_equals_threshold": sweep == config.sweep_threshold_deg,
        "velocity_equals_threshold": velocity == config.velocity_threshold_deg_s,
        "passes_gt": bool(gt),
        "passes_ge": bool(ge),
        "passed": bool(gt if config.threshold_operator == "gt" else ge),
    })
    return result


def prepare_session(
    timestamps_seconds: NDArray, quaternion_wxyz: NDArray, config: PipelineConfig
) -> PreparedSession:
    """Validate, native-rate gate, trim, filter, cap, split and SLERP a session.

    Invalid clocks reject rather than being sorted or repaired. Nonfinite or
    near-zero quaternion rows are dropped, exposing gaps before interpolation.
    Native rate is measured on original timestamps, before any quaternion row
    removal. Trimming and the cap are anchored to the original first timestamp.
    """
    _validate_config(config)
    stats: dict = {
        "native_rate_basis": "median_original_timestamp_differences_before_row_removal",
        "trim_seconds": float(config.trim_seconds),
        "session_cap_seconds": float(config.session_cap_seconds),
        "max_gap_seconds": float(config.max_gap_seconds),
        "norm_epsilon": float(config.norm_epsilon),
        "target_rate_hz": float(config.target_rate_hz),
        "offline_resampling": "SLERP_bracketing_native_samples_no_sensor_latency_claim",
        "filter_before_duration_cap": True,
    }

    def reject(reason: str) -> PreparedSession:
        stats["reason"] = reason
        return PreparedSession(False, reason, stats)

    try:
        timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    except (TypeError, ValueError):
        return reject("invalid_numeric_input")
    if timestamps.ndim != 1:
        return reject("invalid_timestamp_shape")
    stats["input_samples"] = len(timestamps)
    if quaternion.shape != (len(timestamps), 4):
        return reject("invalid_quaternion_shape")
    if len(timestamps) < 2:
        return reject("insufficient_original_samples")
    if not np.all(np.isfinite(timestamps)):
        stats["nonfinite_timestamp_rows"] = int(np.count_nonzero(~np.isfinite(timestamps)))
        return reject("nonfinite_timestamps")
    differences = np.diff(timestamps)
    if np.any(differences < 0):
        return reject("backward_timestamps")
    if np.any(differences == 0):
        return reject("duplicate_timestamps")
    native_rate = float(1.0 / np.median(differences))
    stats["native_rate_hz"] = native_rate
    stats["original_start_seconds"] = float(timestamps[0])
    stats["original_end_seconds"] = float(timestamps[-1])
    if not config.native_rate_min_hz <= native_rate <= config.native_rate_max_hz:
        return reject("native_rate_out_of_range")

    finite = np.all(np.isfinite(quaternion), axis=1)
    valid = np.zeros(len(quaternion), dtype=bool)
    finite_q = quaternion[finite]
    scale = np.max(np.abs(finite_q), axis=1)
    nonzero = scale > 0
    norm_scaled = np.linalg.norm(finite_q[nonzero] / scale[nonzero, None], axis=1)
    above_epsilon = np.zeros(len(finite_q), dtype=bool)
    above_epsilon[nonzero] = norm_scaled > config.norm_epsilon / scale[nonzero]
    valid[finite] = above_epsilon
    stats["nonfinite_quaternion_rows"] = int(np.count_nonzero(~finite))
    stats["nearzero_quaternion_rows"] = int(np.count_nonzero(finite) - np.count_nonzero(valid))
    stats["dropped_quaternion_rows"] = int(np.count_nonzero(~valid))
    trim_origin = float(timestamps[0] + config.trim_seconds)
    keep = valid & ((timestamps - timestamps[0]) >= config.trim_seconds)
    stats["trimmed_valid_samples"] = int(np.count_nonzero(keep))
    if np.count_nonzero(keep) < 20:
        return reject("insufficient_trimmed_samples")
    trimmed_t = timestamps[keep]
    trimmed_q = align_hemisphere(quaternion[keep])
    tolerance = _clock_tolerance(timestamps)
    filtering = _motion_filter(trimmed_t, trimmed_q, trim_origin, float(timestamps[-1]), config, tolerance)
    stats["filter"] = filtering
    if not filtering["passed"]:
        return reject(filtering.get("failure", "motion_filter_failed"))

    capped = trimmed_t - trim_origin <= config.session_cap_seconds
    capped_t, capped_q = trimmed_t[capped], trimmed_q[capped]
    stats["samples_removed_by_cap"] = int(np.count_nonzero(~capped))
    stats["capped_valid_samples"] = len(capped_t)
    if len(capped_t) < 2:
        return reject("insufficient_capped_samples")
    gap_indices = np.flatnonzero(np.diff(capped_t) > config.max_gap_seconds + tolerance) + 1
    bounds = np.r_[0, gap_indices, len(capped_t)]
    stats["split_gap_count"] = len(gap_indices)
    stats["candidate_segments"] = len(bounds) - 1
    stats["discarded_short_segments"] = 0
    segments = []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        if end - begin < 2:
            stats["discarded_short_segments"] += 1
            continue
        uniform_t, uniform_q = resample_slerp(capped_t[begin:end], capped_q[begin:end], config.target_rate_hz)
        if len(uniform_t) < config.min_segment_samples:
            stats["discarded_short_segments"] += 1
            continue
        segments.append(PreparedSegment(uniform_t, uniform_q))
    stats["accepted_segments"] = len(segments)
    stats["output_samples"] = sum(len(segment.timestamps) for segment in segments)
    if not segments:
        return reject("segments_too_short")
    stats["reason"] = "accepted"
    return PreparedSession(True, "accepted", stats, segments)

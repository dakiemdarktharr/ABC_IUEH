from dataclasses import replace
import json

import numpy as np
import pytest

from abc_iueh.config import PipelineConfig
from abc_iueh.geometry import exp_so3, geodesic_deg
from abc_iueh.preprocess import prepare_session


def rotation_session(duration=35.0, rate=90.0, speed_deg_s=30.0):
    t = np.arange(int(round(duration * rate)) + 1, dtype=np.float64) / rate
    v = np.zeros((len(t), 3))
    v[:, 2] = np.radians(speed_deg_s) * t
    return t, exp_so3(v)


def test_analytic_native_90_hz_session_passes_and_retains_correct_time_index():
    t, q = rotation_session()
    q[::3] *= -2
    result = prepare_session(t, q, PipelineConfig())
    assert result.accepted, result.reason
    assert result.stats["native_rate_hz"] == pytest.approx(90)
    assert result.stats["filter"]["velocity_deg_s"] == pytest.approx(30, abs=.1)
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.timestamps.dtype == np.float64
    assert segment.quaternion.dtype == np.float32
    assert segment.timestamps[0] == 5
    assert segment.timestamps[100] == pytest.approx(6)
    expected = exp_so3(np.column_stack((np.zeros((len(segment.timestamps), 2)), np.radians(30) * segment.timestamps)))
    assert np.max(geodesic_deg(segment.quaternion, expected)) < 1e-5
    np.testing.assert_allclose(np.linalg.norm(segment.quaternion, axis=1), 1, atol=1e-7)
    assert result.stats["filter"]["sweep_unwrap"] is False
    json.dumps(result.stats, allow_nan=False)


@pytest.mark.parametrize("kind,reason", [
    ("nan", "nonfinite_timestamps"),
    ("backward", "backward_timestamps"),
    ("duplicate", "duplicate_timestamps"),
])
def test_invalid_clocks_are_rejected_before_quaternion_removal(kind, reason):
    t, q = rotation_session()
    if kind == "nan":
        t[50] = np.nan
    elif kind == "backward":
        t[50] = t[49] - 1
    else:
        t[50] = t[49]
    q[50] = 0
    result = prepare_session(t, q, PipelineConfig())
    assert not result.accepted
    assert result.reason == reason
    assert not result.segments


def test_native_rate_gate_cannot_be_repaired_by_resampling():
    t, q = rotation_session(rate=60)
    result = prepare_session(t, q, PipelineConfig())
    assert not result.accepted
    assert result.reason == "native_rate_out_of_range"
    assert result.stats["native_rate_hz"] == pytest.approx(60)


def test_bad_quaternion_rows_do_not_change_measured_native_rate():
    t, q = rotation_session()
    q[::2] = 0
    q[31] = np.nan
    result = prepare_session(t, q, PipelineConfig())
    assert result.accepted, result.reason
    assert result.stats["native_rate_hz"] == pytest.approx(90)
    assert result.stats["nonfinite_quaternion_rows"] == 1
    assert result.stats["nearzero_quaternion_rows"] == (len(t) + 1) // 2
    assert result.stats["split_gap_count"] == 0


def test_removed_bad_rows_split_large_interpolation_gaps():
    t, q = rotation_session(duration=32)
    bad = (t >= 16) & (t <= 16.1)
    q[bad] = 1e-12
    result = prepare_session(t, q, PipelineConfig())
    assert result.accepted, result.reason
    assert result.stats["split_gap_count"] == 1
    assert result.stats["filter"]["masked_gap_velocity_pairs"] == 1
    assert len(result.segments) == 2
    assert result.segments[0].timestamps[-1] < 16
    assert result.segments[1].timestamps[0] > 16.1
    for segment in result.segments:
        np.testing.assert_allclose(np.diff(segment.timestamps), .01, atol=1e-14)


def test_velocity_never_spans_a_gap_even_when_orientation_jumps():
    t, q = rotation_session(duration=32, speed_deg_s=0)
    q[t > 16.1] = exp_so3([0, 0, np.pi / 2])
    q[(t >= 16) & (t <= 16.1)] = np.nan
    config = replace(PipelineConfig(), sweep_threshold_deg=0, velocity_threshold_deg_s=0)
    result = prepare_session(t, q, config)
    assert not result.accepted
    assert result.stats["filter"]["masked_gap_velocity_pairs"] == 1
    assert result.stats["filter"]["velocity_deg_s"] == 0


def test_motion_filter_runs_on_entire_trimmed_session_before_cap():
    t, _ = rotation_session(duration=40)
    v = np.zeros((len(t), 3))
    v[:, 2] = np.radians(30) * np.maximum(t - 15, 0)
    q = exp_so3(v)
    result = prepare_session(t, q, replace(PipelineConfig(), session_cap_seconds=10))
    assert result.accepted, result.reason
    assert result.stats["filter"]["velocity_deg_s"] > 15
    assert result.stats["samples_removed_by_cap"] > 0
    segment = result.segments[0]
    assert segment.timestamps[-1] <= 15
    np.testing.assert_allclose(segment.quaternion, np.tile([1, 0, 0, 0], (len(segment.timestamps), 1)))


def test_threshold_boundary_gt_and_ge_are_explicit():
    t, q = rotation_session(speed_deg_s=0)
    config = replace(PipelineConfig(), sweep_threshold_deg=0, velocity_threshold_deg_s=0)
    strict = prepare_session(t, q, config)
    inclusive = prepare_session(t, q, replace(config, threshold_operator="ge"))
    assert not strict.accepted
    assert inclusive.accepted
    assert strict.stats["filter"]["passes_gt"] is False
    assert strict.stats["filter"]["passes_ge"] is True
    assert strict.stats["filter"]["sweep_equals_threshold"] is True
    assert strict.stats["filter"]["velocity_equals_threshold"] is True


def test_partial_tail_elapsed_policy_and_drop_policy_are_observable():
    t, q = rotation_session(duration=9.5)
    config = replace(PipelineConfig(), min_segment_samples=420)
    partial = prepare_session(t, q, config)
    dropped = prepare_session(t, q, replace(config, tail_window_policy="drop"))
    assert partial.accepted, partial.reason
    assert partial.stats["filter"]["velocity_deg_s"] == pytest.approx(30)
    assert partial.stats["filter"]["partial_window_denominator"] == "last_valid_minus_first_valid_timestamp"
    assert not dropped.accepted
    assert dropped.reason == "no_valid_filter_windows"
    assert dropped.stats["filter"]["dropped_partial_windows"] == 1


def test_trim_uses_original_start_even_if_initial_orientation_rows_are_bad():
    t, q = rotation_session()
    q[t < 1] = 0
    result = prepare_session(t, q, PipelineConfig())
    assert result.accepted
    assert result.segments[0].timestamps[0] == 5


def test_gap_segments_that_cannot_supply_minimum_samples_are_rejected():
    t, q = rotation_session(duration=13)
    result = prepare_session(t, q, PipelineConfig())
    assert not result.accepted
    assert result.reason == "segments_too_short"

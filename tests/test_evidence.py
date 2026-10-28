import numpy as np
import pytest

from abc_iueh.baselines import predict_constant_velocity
from abc_iueh.evidence import EXPERT_NAMES, build_segment_features
from abc_iueh.geometry import (
    exp_so3, geodesic_deg, inverse_wxyz, log_so3, multiply_wxyz, normalize_wxyz,
)


def random_rotations(length=680, seed=782):
    """Changing axes give noncommuting rotations, rather than an easy z sweep."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0, .018, (length - 1, 3))
    q = np.empty((length, 4))
    q[0] = normalize_wxyz(rng.normal(size=4))
    for i, increment in enumerate(increments, start=1):
        q[i] = normalize_wxyz(multiply_wxyz(exp_so3(increment), q[i - 1]))
    return q.astype(np.float32)


def replay_prediction(unit, origin, horizon):
    velocity = [np.zeros(3)]
    for bandwidth in (1, 5, 20):
        velocity.append(log_so3(multiply_wxyz(unit[origin], inverse_wxyz(unit[origin - bandwidth]))) / bandwidth)
    return normalize_wxyz(multiply_wxyz(exp_so3(horizon * np.array(velocity)), unit[origin])).astype(np.float32)


def replay_evidence(q, current, horizon, age, length):
    unit = normalize_wxyz(q)
    errors = []
    for r in range(length):
        origin = current - age - horizon - r
        errors.append(geodesic_deg(replay_prediction(unit, origin, horizon), q[origin + horizon]))
    return np.mean(errors, axis=0)


def replay_complete_horizon(q, current, length):
    unit = normalize_wxyz(q)
    errors = []
    for r in range(length):
        origin = current - 20 - r
        for horizon in range(1, 21):
            errors.append(geodesic_deg(replay_prediction(unit, origin, horizon), q[origin + horizon]))
    return np.mean(errors, axis=0)


def test_prefix_evidence_matches_explicit_noncommuting_past_only_replay():
    q = random_rotations()
    result = build_segment_features(q)
    assert EXPERT_NAMES == ("ZV", "CAV_W1", "CAV_W5", "CAV_W20")
    np.testing.assert_array_equal(result["origin_index"], np.arange(399, len(q) - 20))
    assert result["origin_index"].dtype == np.int64
    for current in (399, 560):
        row = current - 399
        for name, (horizon, age) in {"h1_a0": (1, 0), "h20_a0": (20, 0), "h1_a19": (1, 19), "h20_a20": (20, 20)}.items():
            expected = replay_evidence(q, current, horizon, age, 80)
            np.testing.assert_allclose(result["features"][name][row, :4], expected, rtol=6e-8, atol=2e-6)
    expected_complete = replay_complete_horizon(q, 465, 80)
    np.testing.assert_allclose(result["features"]["complete_horizon"][465 - 399, :4], expected_complete, rtol=6e-8, atol=2e-6)
    for feature in result["features"].values():
        assert feature.shape == (len(q) - 419, 10)
        assert feature.dtype == np.float32
        assert np.all(np.isfinite(feature))
    assert result["velocities"].dtype == np.float64
    assert result["expert_loss"].dtype == np.float64
    assert result["current_q"].dtype == result["target_q"].dtype == np.float32


def test_matched_issue_age_has_same_origins_and_distinct_mature_targets():
    q = random_rotations()
    current = 550
    result = build_segment_features(q, evidence_length=1, include_complete_horizon=False)
    unit = normalize_wxyz(q)
    shared_origin = current - 20
    h1 = geodesic_deg(replay_prediction(unit, shared_origin, 1), q[current - 19])
    h20 = geodesic_deg(replay_prediction(unit, shared_origin, 20), q[current])
    np.testing.assert_allclose(result["features"]["h1_a19"][current - 399, :4], h1, atol=2e-6)
    np.testing.assert_allclose(result["features"]["h20_a0"][current - 399, :4], h20, atol=2e-6)
    assert not np.allclose(h1, h20)
    assert "complete_horizon" not in result["features"]


def test_future_suffix_perturbation_cannot_change_any_available_feature():
    q = random_rotations(length=850)
    current = 615
    changed = q.copy()
    rng = np.random.default_rng(8)
    changed[current + 1:] = normalize_wxyz(rng.normal(size=changed[current + 1:].shape)).astype(np.float32)
    original = build_segment_features(q)
    perturbed = build_segment_features(changed)
    prefix = original["origin_index"] <= current
    for name in original["features"]:
        np.testing.assert_array_equal(original["features"][name][prefix], perturbed["features"][name][prefix])
    np.testing.assert_array_equal(original["velocities"][prefix], perturbed["velocities"][prefix])
    np.testing.assert_array_equal(original["current_q"][prefix], perturbed["current_q"][prefix])
    # Future supervision is expected to change; it is intentionally separate.
    assert not np.array_equal(original["expert_loss"][prefix], perturbed["expert_loss"][prefix])


def test_quaternion_sign_and_world_left_rotation_preserve_scalar_features():
    q = random_rotations()
    signed = q.copy()
    signed[::3] *= -1
    world = exp_so3([.8, -.4, .3])
    rotated = normalize_wxyz(multiply_wxyz(world, q)).astype(np.float32)
    reference = build_segment_features(q)
    sign_result = build_segment_features(signed)
    rotation_result = build_segment_features(rotated)
    for name, features in reference["features"].items():
        np.testing.assert_array_equal(features, sign_result["features"][name])
        np.testing.assert_allclose(features, rotation_result["features"][name], atol=8e-6, rtol=2e-5)
    np.testing.assert_array_equal(reference["expert_loss"], sign_result["expert_loss"])
    np.testing.assert_allclose(reference["expert_loss"], rotation_result["expert_loss"], atol=1e-5, rtol=2e-5)


def test_actual_float32_endpoint_labels_match_fixed_expert_inference():
    q = random_rotations()
    result = build_segment_features(q, include_complete_horizon=False)
    for current in (399, 550, len(q) - 21):
        expected = []
        context = q[current - 399:current + 1][None]
        for bandwidth in (0, 1, 5, 20):
            prediction = predict_constant_velocity(context, bandwidth)[0, 19]
            expected.append(geodesic_deg(prediction, q[current + 20]))
        np.testing.assert_array_equal(result["expert_loss"][current - 399], expected)
    np.testing.assert_array_equal(result["current_q"], q[result["origin_index"]])
    np.testing.assert_array_equal(result["target_q"], q[result["origin_index"] + 20])


def test_motion_columns_are_exact_and_stationary_cosines_are_finite():
    q = random_rotations()
    result = build_segment_features(q, include_complete_horizon=False)
    current = 500
    row = current - 399
    velocity = result["velocities"][row, 1:]
    speed = np.linalg.norm(velocity, axis=1)
    pairs = ((0, 1), (0, 2), (1, 2))
    cosines = [np.clip(np.dot(velocity[i], velocity[j]) / max(speed[i] * speed[j], 1e-12), -1, 1) for i, j in pairs]
    differences = [np.linalg.norm(velocity[i] - velocity[j]) for i, j in pairs]
    unit = normalize_wxyz(q)
    net = np.linalg.norm(log_so3(multiply_wxyz(unit[current], inverse_wxyz(unit[current - 100])))) / 100
    np.testing.assert_allclose(result["features"]["motion_only"][row], [*speed, *cosines, *differences, net], rtol=6e-8, atol=1e-8)
    for feature in result["features"].values():
        np.testing.assert_array_equal(feature[:, 4:10] if feature is not result["features"]["motion_only"] else feature[:, :6], result["features"]["motion_only"][:, :6])
    stationary = build_segment_features(np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (420, 1)))
    for feature in stationary["features"].values():
        np.testing.assert_array_equal(feature, np.zeros((1, 10), dtype=np.float32))


def test_maximum_context_supported_history_has_no_early_nan_poisoning():
    q = random_rotations(length=425)
    result = build_segment_features(q, evidence_length=340)
    expected = replay_evidence(q, 399, 20, 20, 340)
    np.testing.assert_allclose(result["features"]["h20_a20"][0, :4], expected, atol=2e-6)
    assert all(np.isfinite(feature).all() for feature in result["features"].values())


@pytest.mark.parametrize("length", [0, 1, 419])
def test_short_segments_return_consistent_empty_arrays(length):
    q = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (length, 1))
    result = build_segment_features(q)
    assert result["origin_index"].shape == (0,)
    assert result["velocities"].shape == (0, 4, 3)
    assert result["expert_loss"].shape == (0, 4)
    assert all(feature.shape == (0, 10) for feature in result["features"].values())


@pytest.mark.parametrize("length", [0, -1, 341, 1.5, True])
def test_evidence_length_must_fit400_past_samples(length):
    with pytest.raises(ValueError, match="evidence_length"):
        build_segment_features(np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (420, 1)), evidence_length=length)


@pytest.mark.parametrize("bad", [np.zeros((420, 4), dtype=np.float32), np.ones((420, 3), dtype=np.float32), np.ones((420, 4), dtype=np.float64), np.full((420, 4), np.nan, dtype=np.float32)])
def test_invalid_quaternion_input_is_rejected(bad):
    with pytest.raises(ValueError):
        build_segment_features(bad)

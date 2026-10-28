import numpy as np
import pytest

from abc_iueh.geometry import (
    align_hemisphere, exp_so3, geodesic_deg, inverse_wxyz, log_so3,
    multiply_wxyz, normalize_wxyz, resample_slerp,
)


def test_normalization_is_stable_and_hemisphere_alignment_is_sequential():
    q = np.array([[1e300, 1e300, 0, 0], [-2, -2, 0, 0], [3, 3, 0, 0]])
    aligned = align_hemisphere(q)
    np.testing.assert_allclose(np.linalg.norm(aligned, axis=1), 1, atol=2e-16)
    np.testing.assert_allclose(aligned, np.tile([2**-0.5, 2**-0.5, 0, 0], (3, 1)))
    np.testing.assert_allclose(normalize_wxyz([1e-300, 0, 0, 0]), [1, 0, 0, 0])


def test_hamilton_product_active_rotation_order_and_inverse():
    half = 2**-0.5
    around_x = np.array([half, half, 0, 0])
    around_z = np.array([half, 0, 0, half])
    product = multiply_wxyz(around_z, around_x)
    np.testing.assert_allclose(product, [.5, .5, .5, .5], atol=2e-16)
    np.testing.assert_allclose(multiply_wxyz(product, inverse_wxyz(product)), [1, 0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(inverse_wxyz(5 * product), inverse_wxyz(product))


def test_log_exp_round_trip_near_zero_and_pi_and_sign_parity():
    vectors = np.array([[0, 0, 0], [1e-12, -2e-12, 3e-12], [.2, -.4, .1], [np.pi - 1e-10, 0, 0]])
    q = exp_so3(vectors)
    np.testing.assert_allclose(log_so3(q), vectors, atol=5e-15)
    np.testing.assert_allclose(log_so3(-q), log_so3(q), atol=5e-15)
    np.testing.assert_allclose(log_so3([0, -1, 0, 0]), [np.pi, 0, 0], atol=1e-15)
    np.testing.assert_allclose(exp_so3(log_so3(q)), q, atol=5e-15)


def test_geodesic_reports_degrees_and_resolves_small_angles():
    identity = [1, 0, 0, 0]
    quarter_turn = exp_so3([0, 0, np.pi / 2])
    assert geodesic_deg(identity, quarter_turn) == pytest.approx(90)
    assert geodesic_deg(quarter_turn, -quarter_turn) == pytest.approx(0, abs=1e-12)
    tiny = exp_so3([1e-10, 0, 0])
    assert geodesic_deg(identity, tiny) == pytest.approx(np.degrees(1e-10), rel=1e-12)
    assert geodesic_deg(identity, [0, 1, 0, 0]) == pytest.approx(180)


def test_slerp_90_to_100_hz_matches_analytic_rotation_and_sample_index():
    native_t = 50 + np.arange(181) / 90
    q = exp_so3(np.column_stack((np.zeros((181, 2)), .5 * (native_t - 50))))
    q[::2] *= -3  # Source representation/sign must not change interpolation.
    output_t, output_q = resample_slerp(native_t, q, 100)
    assert output_t.dtype == np.float64
    assert output_q.dtype == np.float32
    assert len(output_t) == 201
    np.testing.assert_allclose(np.diff(output_t), .01, atol=1e-14)
    assert output_t[100] == pytest.approx(native_t[90])
    expected = exp_so3(np.column_stack((np.zeros((201, 2)), .5 * (output_t - 50))))
    assert np.max(geodesic_deg(output_q, expected)) < 5e-6
    np.testing.assert_allclose(np.linalg.norm(output_q, axis=1), 1, atol=1e-7)
    assert np.all(np.sum(output_q[:-1] * output_q[1:], axis=1) >= 0)


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [1, np.nan, 0, 0], [1, 0, 0], [np.inf, 0, 0, 0]])
def test_geometry_rejects_invalid_quaternions(bad):
    with pytest.raises(ValueError):
        normalize_wxyz(bad)


@pytest.mark.parametrize("bad_t", [[0], [0, 0], [1, 0], [0, np.nan]])
def test_slerp_rejects_invalid_clocks(bad_t):
    with pytest.raises(ValueError):
        resample_slerp(bad_t, np.tile([1, 0, 0, 0], (len(bad_t), 1)), 100)

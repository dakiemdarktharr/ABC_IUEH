import numpy as np
from abc_iueh.baselines import predict_constant_velocity
from abc_iueh.geometry import exp_so3, geodesic_deg


def test_cav_sample20_matches_analytic_motion_and_q_sign():
    velocity = np.array([0.001, -0.002, 0.0005])
    context = exp_so3(np.arange(400)[:, None] * velocity).astype(np.float32)[None]
    prediction = predict_constant_velocity(context, bandwidth=20)
    target = exp_so3(419 * velocity)
    assert prediction.shape == (1, 20, 4) and prediction.dtype == np.float32
    assert float(geodesic_deg(prediction[0, -1], target)) < 1e-4
    flipped = context.copy()
    flipped[:, ::3] *= -1
    sign_prediction = predict_constant_velocity(flipped, bandwidth=20)
    assert float(np.max(geodesic_deg(prediction, sign_prediction))) < 1e-4

"""Synthetic-only mechanism and numerical tests for the bounded SO(3) IMM."""
import math

import pytest
import torch

from abc_iueh.boxrr_models import (
    conjugate_wxyz, exp_so3, geodesic_radians, log_so3, multiply_wxyz,
)
from abc_iueh.boxrr_v5_imm import IMMResidual, _so3_matrices


torch.set_num_threads(2)


def cv_motion(batch=2):
    rates = torch.tensor([[.7, -.25, .4], [-.3, .5, .6]])[:batch]
    start = exp_so3(torch.tensor([[.3, .2, -.4], [.1, -.6, .2]])[:batch])
    times = torch.arange(420).float()[None, :, None]*.01
    q = multiply_wxyz(start[:, None], exp_so3(times*rates[:, None]))
    return q[:, :400], q[:, 400:], rates


def varied_motion(batch=3):
    generator = torch.Generator().manual_seed(501)
    increments = torch.randn(batch, 420, 3, generator=generator)*.004
    increments += torch.tensor([.006, -.003, .002])
    return exp_so3(increments.cumsum(1))


@pytest.mark.parametrize("interaction", [True, False])
def test_constant_cv_physical_motion_and_pure_filter(interaction):
    net = IMMResidual(width=12, interaction=interaction)
    x, expected, rates = cv_motion()
    state = net.filter_state(x)
    torch.testing.assert_close(state["velocity"], rates[:, None].expand(-1, 2, -1), atol=2e-4, rtol=0)
    assert state["acceleration"].abs().max() < .004
    prediction = net.pure_filter(x)
    assert geodesic_radians(prediction, expected).max() < 1e-4
    torch.testing.assert_close(net(x), prediction, atol=3e-7, rtol=0)


def test_ca_prediction_uses_radians_seconds_and_half_acceleration():
    net = IMMResidual(width=8, interaction=False)
    q = torch.zeros(1, 2, 4); q[..., 0] = 1
    velocity = torch.tensor([[[0., 0., 1.], [0., 0., 1.]]])
    acceleration = torch.tensor([[[0., 0., 2.], [0., 0., 2.]]])
    covariance = torch.eye(9)[None, None].expand(1, 2, -1, -1).clone()
    for _ in range(20):
        q, velocity, acceleration, covariance = net._propagate(q, velocity, acceleration, covariance)
    # CV: .2 rad. CA: 1*.2 + .5*2*.2**2 = .24 rad, final w=1.4 rad/s.
    torch.testing.assert_close(log_so3(q)[..., 2], torch.tensor([[.2, .24]]), atol=2e-6, rtol=0)
    torch.testing.assert_close(velocity[..., 2], torch.tensor([[1., 1.4]]), atol=2e-6, rtol=0)
    assert acceleration[0, 0].abs().sum() == 0


def test_ca_filter_recovers_acceleration_from_past_only():
    net = IMMResidual(width=8, interaction=False, measurement_sigma=.0002,
                      cv_accel_noise=.1, ca_jerk_noise=.3)
    # Generate physical constant angular acceleration in one axis with an
    # independent scalar angle equation; initialize acceleration remains zero.
    t = (torch.arange(420).float()-399)*.01
    angles = .8*t+.5*1.5*t.square()
    axis = torch.tensor([0., 0., 1.])
    q = exp_so3(angles[None, :, None]*axis)
    state = net.filter_state(q[:, :400])
    assert state["acceleration"][0, 1, 2].item() == pytest.approx(1.5, abs=.03)
    forecast = net.forecast_states(q[:, :400])
    error = geodesic_radians(forecast["q"][:, :, 1],
                             multiply_wxyz(conjugate_wxyz(q[:, 399:400]), q[:, 400:]))
    assert error.max() < .004


def test_posterior_probabilities_covariance_and_forecast_uncertainty():
    net = IMMResidual(width=8)
    forecast = net.forecast_states(varied_motion()[:, :400])
    state = forecast["posterior"]
    torch.testing.assert_close(state["probabilities"].sum(-1), torch.ones(3), atol=1e-6, rtol=0)
    assert (state["probabilities"] >= 0).all()
    assert (state["probabilities"] <= 1).all()
    for cov in (state["covariance"], forecast["origin_covariance"], forecast["covariance"]):
        torch.testing.assert_close(cov, cov.transpose(-1, -2), atol=2e-7, rtol=1e-5)
        assert torch.linalg.cholesky_ex(cov[..., 0, :6, :6]).info.max() == 0
        assert torch.linalg.cholesky_ex(cov[..., 1, :, :]).info.max() == 0
        assert cov[..., 0, 6:, :].abs().sum() == 0
    # Forecast covariance grows from a nonzero reanchored posterior, not reset to zero.
    assert forecast["origin_covariance"][..., :3, :3].diagonal(dim1=-2, dim2=-1).min() > 0
    assert forecast["covariance"][:, -1, :, :3, :3].diagonal(dim1=-2, dim2=-1).sum() > forecast["origin_covariance"][..., :3, :3].diagonal(dim1=-2, dim2=-1).sum()
    assert not any(value.requires_grad for value in state.values())


def test_manifold_mixing_includes_between_state_dispersion():
    net = IMMResidual(width=8, dwell_cv=1e9, dwell_ca=1e9, covariance_floor=1e-9)
    # Override only this synthetic transition fixture to force equal mixing.
    net.transition.fill_(.5)
    q = exp_so3(torch.tensor([[[0., 0., -.2], [0., 0., .2]]]))
    velocity = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]])
    acceleration = torch.zeros_like(velocity)
    cov = torch.diag_embed(.1*net.active)[None]
    mq, mv, _, mixed, prior = net._interact(q, velocity, acceleration, cov, torch.tensor([[.5, .5]]))
    assert log_so3(mq).abs().max() < 1e-6
    torch.testing.assert_close(mv[..., 0], torch.ones(1, 2))
    torch.testing.assert_close(mixed[..., 3, 3], torch.full((1, 2), 1.1), atol=2e-6, rtol=0)
    torch.testing.assert_close(prior, torch.tensor([[.5, .5]]))
    # The orientation perturbation transport is analytic, not quaternion averaging.
    delta = torch.tensor([[.2, -.1, .4]])
    jacobian = _so3_matrices(delta)[2]
    epsilon = torch.eye(3)*2e-4
    moved = log_so3(multiply_wxyz(exp_so3(epsilon), exp_so3(delta)))
    finite_difference = ((moved-delta)/2e-4).T
    torch.testing.assert_close(jacobian[0], finite_difference, atol=5e-4, rtol=1e-3)


def test_innovation_evidence_includes_gaussian_normalizer():
    net = IMMResidual(width=8, measurement_sigma=.01)
    q = torch.zeros(1, 2, 4); q[..., 0] = 1
    velocity = torch.zeros(1, 2, 3)
    covariance = torch.diag_embed(net.active)[None]
    covariance[:, 0, :3, :3] *= .01
    covariance[:, 1, :3, :3] *= .04
    observation = exp_so3(torch.tensor([[.1, 0., 0.]]))
    prior = torch.tensor([[.4, .6]])
    result = net._measure(q, velocity, velocity.clone(), covariance, observation, prior)
    variance = torch.tensor([[.0101, .0401]])
    expected = -.5*(.1**2/variance+3*variance.log()+3*math.log(2*math.pi))
    torch.testing.assert_close(result[-1], expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(result[-3], torch.softmax(prior.log()+expected, -1), atol=2e-6, rtol=0)


def test_caller_autocast_cannot_reduce_filter_covariance_precision():
    net = IMMResidual(width=8)
    x, _, _ = cv_motion()
    expected = net.forecast_states(x)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = net.forecast_states(x)
        prediction = net(x)
    assert actual["covariance"].dtype == torch.float32
    assert prediction.dtype == torch.float32 and torch.isfinite(prediction).all()
    torch.testing.assert_close(actual["covariance"], expected["covariance"], atol=0, rtol=0)
    torch.testing.assert_close(actual["q"], expected["q"], atol=0, rtol=0)


def test_raw_last_anchor_is_retained_and_no_hidden_cross_call_state():
    net = IMMResidual(width=8)
    x = varied_motion()[:2, :400]
    x[:, -1] = multiply_wxyz(x[:, -1], exp_so3(torch.tensor([[.1, -.06, .02]])))
    forecast = net.forecast_states(x)
    identity = torch.zeros_like(forecast["origin_q"]); identity[..., 0] = 1
    torch.testing.assert_close(forecast["origin_q"], identity, atol=0, rtol=0)
    torch.testing.assert_close(forecast["posterior"]["anchor"], x[:, -1], atol=2e-7, rtol=0)
    assert log_so3(forecast["posterior"]["q"]).abs().max() > .001
    before = net(x)
    net(varied_motion(1)[:, :400])
    torch.testing.assert_close(before, net(x), atol=0, rtol=0)


def test_antipodes_global_rotation_basis_consistency_and_batch_independence():
    net = IMMResidual(width=8).eval()
    x = varied_motion()[:, :400]
    prediction = net(x)
    signs = torch.where(torch.arange(400) % 3 == 0, -1., 1.)[None, :, None]
    assert geodesic_radians(net(x*signs), prediction).max() < 2e-6
    rotation = exp_so3(torch.tensor([[.3, -.6, .8]]))
    moved = multiply_wxyz(rotation[:, None], x)
    expected = multiply_wxyz(rotation[:, None], prediction)
    assert geodesic_radians(net(moved), expected).max() < 2e-5
    # Changing the sensor/body basis rotates every rate/covariance coordinate;
    # isotropic filter physics still gives the equivalent physical orientation.
    moved_basis = multiply_wxyz(x, rotation[:, None])
    expected_basis = multiply_wxyz(net.pure_filter(x), rotation[:, None])
    assert geodesic_radians(net.pure_filter(moved_basis), expected_basis).max() < 3e-5
    alone = torch.cat([net(row[None]) for row in x], 0)
    torch.testing.assert_close(prediction, alone, atol=3e-7, rtol=1e-6)


@pytest.mark.parametrize("decoder", ["mean", "argmax", "risk"])
def test_decoder_controls_are_finite_unit_float32_and_sign_continuous(decoder):
    net = IMMResidual(width=8, decoder=decoder)
    x, _, _ = cv_motion()
    prediction = net.pure_filter(x)
    assert prediction.shape == (2, 20, 4) and prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction.norm(dim=-1), torch.ones(2, 20), atol=3e-6, rtol=0)
    previous = torch.cat((x[:, -1:], prediction[:, :-1]), 1)
    assert ((previous*prediction).sum(-1) >= 0).all()


def test_neural_gradient_paths_filter_detachment_and_unrestricted_correction():
    net = IMMResidual(width=12)
    q = varied_motion(2)
    x, y = q[:, :400].clone().requires_grad_(), q[:, 400:]
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    nonzero = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(x)
        angle = geodesic_radians(prediction, y)
        loss = angle[:, -1].mean()+.1*angle.mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in net.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if parameter.grad.abs().sum() > 0:
                nonzero.add(name)
        assert torch.isfinite(x.grad).all()
        optimizer.step()
    assert any(name.startswith("raw_encoder") for name in nonzero)
    assert any(name.startswith("state_encoder") for name in nonzero)
    assert any(name.startswith("fusion") for name in nonzero)
    assert any(name.startswith("residual") for name in nonzero)
    assert not any("noise" in name or "dwell" in name for name, _ in net.named_parameters())
    # A >.5-radian signed correction survives unchanged; there is no tanh/cap.
    with torch.no_grad():
        net.residual.weight.zero_()
        net.residual.bias.copy_(torch.tensor([.8, -.7, .6]).repeat(20))
    baseline = net.pure_filter(x)
    base_relative = log_so3(multiply_wxyz(conjugate_wxyz(x[:, -1:, :]), baseline))
    expected = multiply_wxyz(x[:, -1:, :], exp_so3(base_relative+torch.tensor([.8, -.7, .6])))
    assert geodesic_radians(net(x), expected).max() < 3e-6


def test_bounded_observation_support_and_matched_independent_control_config():
    net = IMMResidual(width=8)
    independent = IMMResidual(**{**net.get_config(), "interaction": False})
    independent.load_state_dict(net.state_dict())
    assert [(k, tuple(v.shape)) for k, v in net.named_parameters()] == [(k, tuple(v.shape)) for k, v in independent.named_parameters()]
    assert IMMResidual(**net.get_config()).get_config() == net.get_config()
    x = varied_motion(2)[:, :400]
    changed = x.clone()
    changed[:, :-40] = exp_so3(torch.randn_like(changed[:, :-40, :3]))
    torch.testing.assert_close(net(x), net(changed), atol=0, rtol=0)
    # Last-pose perturbations do affect predictions, so the support check is substantive.
    changed[:, -1] = exp_so3(torch.tensor([[.3, .2, .4]]))
    assert geodesic_radians(net(x), net(changed)).max() > .01
    interacting_state = net.filter_state(x)
    independent_state = independent.filter_state(x)
    assert (interacting_state["velocity"]-independent_state["velocity"]).abs().max() > 1e-4


@pytest.mark.parametrize("stationary", [True, False])
def test_stationary_and_near_pi_have_finite_forward_and_backward(stationary):
    net = IMMResidual(width=8)
    x = torch.zeros(2, 400, 4); x[..., 0] = 1
    if not stationary:
        x[:, -40:] = exp_so3(torch.tensor([[math.pi-1e-5, 0., 0.]]))
        x[:, -1] = exp_so3(torch.tensor([[-math.pi+1e-5, 0., 0.]]))
    prediction = net(x)
    target = exp_so3(torch.tensor([[[.01, -.02, .03]]])).expand(2, 20, -1)
    loss = geodesic_radians(prediction, target).mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


@pytest.mark.parametrize("config", [{"dt": 0.}, {"dt": .1}, {"updates": 399}, {"measurement_sigma": float("nan")},
                                     {"decoder": "oracle"}, {"width": 0}])
def test_invalid_configs_fail_before_filtering(config):
    with pytest.raises(ValueError):
        IMMResidual(**config)


def test_invalid_literal_zero_pose_fallback_stays_unit_and_backward_finite():
    net = IMMResidual(width=8)
    x = torch.zeros(2, 400, 4, requires_grad=True)
    prediction = net(x)
    expected = torch.zeros_like(prediction); expected[..., 0] = 1
    torch.testing.assert_close(prediction, expected, atol=0, rtol=0)
    target = exp_so3(torch.full((2, 20, 3), .01))
    geodesic_radians(prediction, target).mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

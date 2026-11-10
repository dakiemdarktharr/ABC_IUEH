"""Software-only fixtures for causal SG control and neural CDE integration.

No human data, pretrained artifacts, upstream execution, provider or Drive I/O.
"""
import pytest
import torch

from abc_iueh.boxrr_models import exp_so3, geodesic_radians, multiply_wxyz
from abc_iueh.boxrr_v5_cde import PolynomialControl, SGNeuralCDE


torch.set_num_threads(2)


def _times():
    return torch.arange(-399, 21, dtype=torch.float32)*.01


def _axis_motion(velocity=.7, acceleration=0., batch=2):
    t = _times()
    angle = .4+velocity*t+.5*acceleration*t.square()
    rotvec = torch.zeros(batch, len(t), 3)
    rotvec[..., 2] = angle
    base = exp_so3(torch.tensor([[.2, -.3, .1]]))
    return multiply_wxyz(base[:, None], exp_so3(rotvec))


def _motion(batch=3):
    t = _times()
    axes = torch.stack((.12*torch.sin(4*t)+.2*t,
                        .09*torch.cos(7*t)-.1*t,
                        .2*torch.sin(3*t)), -1)
    generator = torch.Generator().manual_seed(790)
    axes = axes[None].expand(batch, -1, -1).clone()
    axes += .003*torch.randn(axes.shape, generator=generator)
    axes += .03*torch.randn(batch, 1, 3, generator=generator)
    return exp_so3(axes)


def _model(**kwargs):
    torch.manual_seed(813)
    return SGNeuralCDE(width=12, **kwargs)


@pytest.mark.parametrize('solver', ['rk2', 'rk4'])
def test_shapes_float32_unit_quaternions_reload_and_statelessness(solver):
    net = _model(solver=solver).eval()
    context = _motion()[:, :400]
    prediction = net(context)
    assert prediction.shape == (3, 20, 4)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction.norm(dim=-1), torch.ones(3, 20), atol=2e-6, rtol=0)
    torch.testing.assert_close(net(context), prediction, atol=0, rtol=0)
    net(_motion(1)[:, :400])
    torch.testing.assert_close(net(context), prediction, atol=0, rtol=0)
    restored = SGNeuralCDE(**net.get_config()).eval()
    restored.load_state_dict(net.state_dict())
    torch.testing.assert_close(restored(context), prediction, atol=0, rtol=0)


@pytest.mark.parametrize('velocity,acceleration', [(0., 0.), (.7, 0.), (.7, .9)])
def test_sg_reconstructs_constant_cv_ca_in_physical_seconds(velocity, acceleration):
    recording = _axis_motion(velocity, acceleration)
    net = _model()
    path = net.fit_control(recording[:, :400])
    t = torch.tensor([-.31, -.17, 0., .01, .1, .2])
    expected = torch.zeros(2, len(t), 3)
    expected[..., 2] = velocity*t+.5*acceleration*t.square()
    torch.testing.assert_close(path.evaluate(t), expected, atol=5e-6, rtol=1e-4)
    expected[..., 2] = velocity+acceleration*t
    torch.testing.assert_close(path.evaluate(t, 1), expected, atol=6e-5, rtol=2e-4)
    expected[..., 2] = acceleration
    torch.testing.assert_close(path.evaluate(t, 2), expected, atol=4e-4, rtol=3e-4)
    prediction = net.polynomial_forecast(recording[:, :400])
    assert geodesic_radians(prediction, recording[:, 400:]).max() < 8e-6
    torch.testing.assert_close(path.evaluate(torch.tensor([0.])), torch.zeros(2, 1, 3), atol=0, rtol=0)


def test_cached_coefficients_really_extrapolate_and_analytical_derivatives_match():
    # Construct a known cubic directly in u; no future observations exist here.
    coefficients = torch.tensor([[[.2, 0., 0.], [.1, 0., 0.],
                                  [.03, 0., 0.], [-.02, 0., 0.]]])
    path = PolynomialControl(coefficients, torch.tensor([[1., 0., 0., 0.]]), .31)
    t = torch.tensor([.01, .1, .2])
    u = 1+2*t/.31
    expected = .1*(u-1)+.03*(u.square()-1)-.02*(u.pow(3)-1)
    torch.testing.assert_close(path.evaluate(t)[0, :, 0], expected)
    first = (.1+.06*u-.06*u.square())*(2/.31)
    second = (.06-.12*u)*(2/.31)**2
    torch.testing.assert_close(path.evaluate(t, 1)[0, :, 0], first)
    torch.testing.assert_close(path.evaluate(t, 2)[0, :, 0], second)
    # A fixed constant-velocity continuation differs from the actual cubic.
    linear_extension = t*path.evaluate(torch.zeros(1), 1)[0, 0, 0]
    assert (path.evaluate(t)[0, :, 0]-linear_extension).abs().max() > .02


def test_actual_prefix_control_cannot_see_post_origin_poses_or_early_unsupported_history():
    net = _model().eval()
    recording = _motion().requires_grad_(True)
    prediction = net(recording[:, :400])
    coefficients = net.fit_control(recording[:, :400]).coefficients.detach().clone()
    changed = recording.detach().clone()
    changed[:, 400:] = exp_so3(torch.randn_like(changed[:, 400:, :3])*2)
    changed[:, :368] = exp_so3(torch.randn_like(changed[:, :368, :3])*2)
    torch.testing.assert_close(net(changed[:, :400]), prediction, atol=0, rtol=0)
    torch.testing.assert_close(net.fit_control(changed[:, :400]).coefficients, coefficients, atol=0, rtol=0)
    prediction[..., 1:].square().sum().backward()
    assert recording.grad[:, 400:].count_nonzero() == 0
    assert recording.grad[:, :368].count_nonzero() == 0
    assert recording.grad[:, 368:400].abs().sum() > 0
    assert torch.isfinite(recording.grad).all()
    # This check prevents a vacuous disconnected-control implementation.
    changed[:, 390] = multiply_wxyz(changed[:, 390], exp_so3(torch.tensor([.02, -.01, .03])))
    assert (net.fit_control(changed[:, :400]).coefficients-coefficients).abs().max() > .001
    assert geodesic_radians(net(changed[:, :400]), prediction).max() > .001


def test_antipodes_and_common_world_frame_equivariance():
    net = _model().eval()
    context = _motion()[:, :400]
    prediction = net(context)
    signs = torch.where(torch.arange(400) % 3 == 0, -1., 1.)[None, :, None]
    assert geodesic_radians(net(context*signs), prediction).max() < 1e-6
    # Changing world/reference coordinates left-multiplies every orientation.
    frame = exp_so3(torch.tensor([[.7, -.4, .2]]))
    changed = multiply_wxyz(frame[:, None], context)
    expected = multiply_wxyz(frame[:, None], prediction)
    assert geodesic_radians(net(changed), expected).max() < 8e-5


def test_batch_members_are_independent_including_gradients():
    net = _model().eval()
    context = _motion()[:, :400].requires_grad_(True)
    joined = net(context)
    separate = torch.cat([net(context[i:i+1]) for i in range(3)])
    assert geodesic_radians(joined, separate).max() < 2e-6
    joined[0, -1, 1:].sum().backward()
    assert context.grad[1:].count_nonzero() == 0
    assert context.grad[0].abs().sum() > 0


def test_positive_weights_are_bounded_and_actively_change_the_sg_fit():
    net = _model()
    context = _motion()[:, :400]
    before = net.fit_control(context).coefficients.detach()
    with torch.no_grad():
        net.sg.weight_logits.copy_(torch.linspace(-50, 50, net.window))
    weights = net.sg.observation_weights()
    assert torch.isfinite(weights).all() and (weights > 0).all()
    assert weights.max()/weights.min() <= 20.0001
    assert (net.fit_control(context).coefficients-before).abs().max() > .0001


def test_training_gradients_reach_all_learned_components_and_both_control_maps():
    net = _model()
    recording = _motion()
    target = recording[:, 400:]
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    observed_nonzero = set()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(recording[:, :400])
        errors = geodesic_radians(prediction, target)
        loss = errors[:, -1].mean()+.1*errors.mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in net.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            if parameter.grad.abs().sum() > 0:
                observed_nonzero.add(name)
        projection_gradient = net.field.control_map.weight.grad.reshape(net.width, 7, net.width)
        assert projection_gradient[:, 0].abs().sum() > 0  # time
        assert projection_gradient[:, 1:4].abs().sum() > 0  # velocity
        assert projection_gradient[:, 4:7].abs().sum() > 0  # acceleration
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    assert observed_nonzero == set(dict(net.named_parameters()))


@pytest.mark.parametrize('literal_zero', [False, True])
def test_float32_stationary_zero_controls_and_input_backward_are_finite(literal_zero):
    net = _model()
    context = torch.zeros(2, 400, 4)
    if not literal_zero:
        context[..., 0] = 1
    context.requires_grad_(True)
    path = net.fit_control(context)
    assert path.evaluate(net.future_times, 1).count_nonzero() == 0
    assert path.evaluate(net.future_times, 2).count_nonzero() == 0
    prediction = net(context)
    target = exp_so3(torch.full((2, 20, 3), .01))
    geodesic_radians(prediction, target).mean().backward()
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(context.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_time_channel_advances_latent_even_when_velocity_and_acceleration_are_zero():
    net = _model()
    context = _axis_motion(0., 0.)[:, :400]
    path = net.fit_control(context)
    with torch.no_grad():
        # A deliberately isolated autonomous field makes omission of time fail.
        net.field.control_map.weight.zero_()
        net.field.control_map.bias.zero_()
        net.field.control_map.bias.reshape(net.width, 7)[:, 0] = .5
    origin, future = net.integrate_control(path)
    expected = net.future_times[:, None]*torch.tanh(torch.tensor(.5))
    torch.testing.assert_close(future-origin[:, None], expected[None].expand(2, -1, net.width), atol=2e-6, rtol=1e-5)


def test_fixed_step_convergence_includes_neural_field_and_decoded_forecast():
    # Strengthen nonlinear state dependence so RK2 errors exceed float32 noise.
    net = _model()
    with torch.no_grad():
        net.field.control_map.weight.mul_(5.)
        net.field.trunk[0].weight.mul_(3.)
        net.field.trunk[2].weight.mul_(3.)
        net.readout[-1].weight.mul_(8.)
    recording = _axis_motion(2., 7.)
    path = net.fit_control(recording[:, :400])
    states, outputs = [], []
    for substeps in (1, 2, 4, 8):
        variant = SGNeuralCDE(**{**net.get_config(), 'substeps': substeps})
        variant.load_state_dict(net.state_dict())
        states.append(variant.integrate_control(path)[1].detach())
        outputs.append(variant(recording[:, :400]).detach())
    coarse = (states[0]-states[3]).abs().max()
    medium = (states[1]-states[3]).abs().max()
    fine = (states[2]-states[3]).abs().max()
    assert coarse > 1e-5
    assert medium < .45*coarse
    assert fine < .45*medium
    output_errors = [geodesic_radians(prediction, outputs[-1]).max() for prediction in outputs[:-1]]
    assert output_errors[0] > 1e-6
    assert output_errors[1] < .6*output_errors[0]
    assert output_errors[2] < .6*output_errors[1]
    # Dynamics must contribute; an SG-only path would vacuously converge.
    assert geodesic_radians(outputs[0], net.polynomial_forecast(recording[:, :400])).max() > .001
    rk4 = SGNeuralCDE(**{**net.get_config(), 'solver': 'rk4'})
    rk4.load_state_dict(net.state_dict())
    assert (rk4.integrate_control(path)[1]-states[-1]).abs().max() < coarse*.1


def test_default_knot_and_function_evaluation_bound():
    net = _model()
    called = []
    hook = net.field.register_forward_hook(lambda *_: called.append(1))
    net(_motion(1)[:, :400])
    hook.remove()
    assert len(called) == 2*(32-1+20) == 102


@pytest.mark.parametrize('kwargs', [dict(window=33), dict(window=3), dict(polynomial_order=4),
                                  dict(dt=.1), dict(substeps=0), dict(substeps=9),
                                  dict(solver='adaptive'), dict(acceleration_scale=0)])
def test_contract_rejects_unsupported_configurations(kwargs):
    with pytest.raises(ValueError):
        _model(**kwargs)


def test_input_contract_is_preserved():
    net = _model()
    with pytest.raises(ValueError, match='400'):
        net(_motion(1))
    with pytest.raises(ValueError, match='400'):
        net(torch.ones(1, 400, 3))

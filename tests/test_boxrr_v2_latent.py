"""Synthetic software tests only; these fixtures are never experiment data."""
import copy
import inspect
import math

import pytest
import torch

from abc_iueh.boxrr_models import exp_so3, geodesic_radians, multiply_wxyz
from abc_iueh.boxrr_v2_deterministic import build_model as build_deterministic
from abc_iueh.boxrr_v2_latent import (
    after_optimizer_step, build_latent_model, mixture_log_prob, mixture_sample,
    parameter_counts, training_loss, variance_covariance, weighted_rotation_median,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def synthetic(batch=6):
    generator = torch.Generator().manual_seed(100)
    increments = torch.randn(batch, 420, 3, generator=generator)*.002
    increments += torch.randn(batch, 1, 3, generator=generator)*.003
    values = exp_so3(increments.cumsum(1))
    return values[:, :400], values[:, 400:]


def small(name, **config):
    return build_latent_model(name, seed=14, width=8, latent_dim=32, **config)


@pytest.mark.parametrize('name', ['energy_v2', 'jepa_v2'])
def test_output_is_normalized_and_context_only(name):
    model = small(name).eval()
    x, _ = synthetic()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (6, 20, 4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(6, 20), atol=2e-6, rtol=0)
    assert list(inspect.signature(model.forward).parameters) == ['context']


@pytest.mark.parametrize('name', ['energy_v2', 'jepa_v2'])
def test_reproducible_constructor_and_forward(name):
    torch.manual_seed(888)
    before = torch.random.get_rng_state()
    a, b = small(name).eval(), small(name).eval()
    assert torch.equal(before, torch.random.get_rng_state())
    x, _ = synthetic(2)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)
        first = a(x)
        torch.manual_seed(31)
        torch.randn(100)
        torch.testing.assert_close(first, a(x), atol=0, rtol=0)


@pytest.mark.parametrize('name', ['energy_v2', 'jepa_v2'])
def test_hemisphere_sign_invariance(name):
    model = small(name).eval()
    x, _ = synthetic(3)
    signed = x.clone()
    signed[:, ::2] *= -1
    with torch.no_grad():
        error = geodesic_radians(model(x), model(signed))
    assert error.max() < 3e-6


@pytest.mark.parametrize('name', ['energy_v2', 'jepa_v2'])
def test_relative_only_global_rotation_equivariance(name):
    model = small(name, use_gravity=False).eval()
    x, _ = synthetic(3)
    rotation = exp_so3(torch.tensor([.3, -.2, .4]))
    changed = multiply_wxyz(rotation, x)
    with torch.no_grad():
        expected = multiply_wxyz(rotation, model(x))
        actual = model(changed)
    # Float32 conditional geometry plus iterative median can amplify rounding
    # at nearly tied risks; 5e-5 rad is below .003 degrees.
    assert geodesic_radians(expected, actual).max() < 5e-5


def test_exact_gaussian_mixture_log_probability():
    torch.manual_seed(9)
    logits = torch.randn(2, 3)
    means = torch.randn(2, 3, 3)
    scales = torch.rand(2, 3, 3)+.01
    values = torch.randn(2, 7, 3)
    expected = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means, scales), 1),
    ).log_prob(values.transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(mixture_log_prob(values, logits, means, scales), expected)


def test_ancestral_sampling_has_correct_mixture_mean_and_no_gradient():
    logits = torch.tensor([[math.log(.8), math.log(.2)]], requires_grad=True)
    means = torch.tensor([[[1., 0., 0.], [-3., 0., 0.]]], requires_grad=True)
    scales = torch.full((1, 2, 3), .05, requires_grad=True)
    torch.manual_seed(44)
    samples = mixture_sample(logits, means, scales, 30000)
    assert not samples.requires_grad
    assert abs(samples[:, :, 0].mean().item()-.2) < .04
    assert samples[:, :, 1:].mean().abs() < .002


def test_corrected_nce_logit_cancels_known_reference_density():
    model = small('energy_v2', energy_warmup=0)
    x, y = synthetic(3)
    z, bundle = model.encode(x)
    point = model.decoder.tangent(z, bundle)[:, -1]
    distribution = model.distribution(z, point)
    values = torch.randn(3, 7, 3)*.1
    corrected = -model.energy(z, values, point, distribution)-mixture_log_prob(values, *distribution)
    residual = model.residual_energy(z, values, point)
    torch.testing.assert_close(corrected, -residual, atol=1e-6, rtol=1e-6)
    # Zero-initialized residual makes the initial density exactly q.
    assert corrected.abs().max() == 0


def test_quadrature_masses_include_component_probability():
    model = small('energy_v2', components=2, quadrature_per_component=8)
    logits = torch.tensor([[math.log(.9), math.log(.1)]])
    means = torch.tensor([[[.1, 0., 0.], [-.4, 0., 0.]]])
    scales = torch.ones_like(means)*.01
    points, log_mass = model.quadrature((logits, means, scales))
    masses = log_mass.exp()
    torch.testing.assert_close(masses.sum(-1), torch.ones(1))
    torch.testing.assert_close(masses[:, :8].sum(-1), torch.tensor([.9]))
    torch.testing.assert_close((points*masses[:, :, None]).sum(1),
                               .9*means[:, 0]+.1*means[:, 1], atol=1e-6, rtol=1e-6)


def test_continuous_median_never_worsens_finite_risk_and_can_leave_grid():
    rotations = exp_so3(torch.tensor([[[.1, .0, 0.], [-.04, .08, 0.], [-.04, -.08, 0.]]]))
    weights = torch.full((1, 3), 1/3)
    initial = exp_so3(torch.tensor([[.15, .1, 0.]]))
    chosen, risk = weighted_rotation_median(rotations, weights, initial, iterations=8)
    initial_risk = (geodesic_radians(initial[:, None], rotations)*weights).sum(-1)
    medoid_risk = (geodesic_radians(rotations[:, :, None], rotations[:, None])*weights[:, None]).sum(-1).min(-1).values
    assert torch.all(risk <= torch.minimum(initial_risk, medoid_risk)+1e-7)
    assert geodesic_radians(chosen[:, None], rotations).min() > 1e-3
    torch.testing.assert_close(chosen.norm(dim=-1), torch.ones(1), atol=1e-6, rtol=0)


def test_coincident_rotation_median_is_finite():
    rotations = exp_so3(torch.zeros(2, 12, 3))
    result, risk = weighted_rotation_median(rotations, torch.ones(2, 12), rotations[:, 0])
    assert torch.isfinite(result).all()
    assert risk.max() < 1e-6


def test_density_gradients_do_not_modify_supervised_forecaster():
    model = small('energy_v2', energy_warmup=0)
    x, y = synthetic(4)
    loss, info = training_loss(model, x, y, step=3)
    parameters = list(model.encoder.parameters())+list(model.decoder.parameters())
    full_grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    z, bundle = model.encode(x)
    prediction = model.decode(z, bundle)
    angles = geodesic_radians(prediction, y)
    supervised = angles[:, -1].mean()+.1*angles.mean()
    direct_grad = torch.autograd.grad(supervised, parameters, allow_unused=True)
    for full, direct in zip(full_grad, direct_grad):
        assert (full is None) == (direct is None)
        if full is not None:
            torch.testing.assert_close(full, direct, atol=1e-7, rtol=1e-6)
    loss.backward()
    assert model.density_head[-1].weight.grad.abs().sum() > 0
    assert model.energy_head[-1].weight.grad.abs().sum() > 0
    assert info['energy_active'] == 1.


def test_energy_warmup_delays_only_energy_gradient():
    model = small('energy_v2', energy_warmup=10)
    x, y = synthetic(3)
    loss, info = training_loss(model, x, y, step=1)
    loss.backward()
    assert info['energy_active'] == 0.
    assert model.energy_head[-1].weight.grad is None
    assert model.density_head[-1].weight.grad.abs().sum() > 0
    assert model.decoder.residual.weight.grad.abs().sum() > 0


def test_energy_forward_diagnostics_and_endpoint_correction():
    model = small('energy_v2').eval()
    x, _ = synthetic(3)
    with torch.no_grad():
        prediction, info = model.forward_with_diagnostics(x)
        proposal = model.proposal_prediction(x)
    assert all(torch.isfinite(value).all() for value in info.values())
    assert torch.all(info['energy_ess'] >= 1-1e-6)
    assert torch.all(info['energy_ess'] <= 48+1e-5)
    actual = geodesic_radians(proposal[:, -1], prediction[:, -1])
    torch.testing.assert_close(actual, info['energy_endpoint_correction_radians'], atol=2e-6, rtol=1e-4)


def test_jepa_future_teacher_has_exactly_twenty_tokens_and_four_targets():
    model = small('jepa_v2')
    x, y = synthetic(4)
    tokens = model.future_tokens(x[:, -1], y)
    assert tokens.shape == (4, 12, 20)
    assert model.target_embeddings(tokens).shape == (4, 4, 64)
    altered = y.clone()
    altered[:, 10:] = exp_so3(torch.full((4, 10, 3), .2))
    changed = model.future_tokens(x[:, -1], altered)
    torch.testing.assert_close(tokens[:, :, :10], changed[:, :, :10])
    with torch.no_grad():
        original_target = model.target_embeddings(tokens)
        altered_target = model.target_embeddings(changed)
    torch.testing.assert_close(original_target[:, :2], altered_target[:, :2])
    assert not torch.allclose(original_target[:, 2:], altered_target[:, 2:])


def test_jepa_pretrain_trains_context_predictor_and_online_future_not_decoder():
    model = small('jepa_v2')
    x, y = synthetic(6)
    loss, info = training_loss(model, x, y, step=1, phase='pretrain')
    assert 'endpoint_radians' not in info
    loss.backward()
    assert model.encoder.output[0].weight.grad.abs().sum() > 0
    assert model.predictor[-1].weight.grad.abs().sum() > 0
    assert model.patch_projection[-1].weight.grad.abs().sum() > 0
    assert model.descriptor_head.weight.grad.abs().sum() > 0
    assert model.decoder.residual.weight.grad is None
    assert all(p.grad is None and not p.requires_grad for m in model.teacher_modules() for p in m.parameters())
    assert all(torch.isfinite(value) for value in info.values())


def test_jepa_first_future_change_uses_observed_previous_increment():
    model = small('jepa_v2')
    angle = torch.arange(22).view(1, 22, 1)*torch.tensor([[[.001, 0., 0.]]])
    poses = exp_so3(angle)
    tokens = model.future_tokens(poses[:, 1], poses[:, 2:], previous=poses[:, 0])
    assert tokens.shape == (1, 12, 20)
    assert tokens[:, 3:6].abs().max() < 1e-5


def test_jepa_supervised_phase_trains_decoder():
    model = small('jepa_v2')
    x, y = synthetic(4)
    loss, info = training_loss(model, x, y, step=1, phase='finetune')
    loss.backward()
    assert model.decoder.residual.weight.grad.abs().sum() > 0
    assert info['endpoint_radians'] >= 0


def test_ema_is_exact_and_teacher_stays_in_eval():
    model = small('jepa_v2', ema_decay=.9)
    before = [p.clone() for p in model.target_stem.parameters()]
    with torch.no_grad():
        for parameter in model.encoder.stem.parameters():
            parameter.add_(.1)
    after_optimizer_step(model)
    for initial, target, source in zip(before, model.target_stem.parameters(), model.encoder.stem.parameters()):
        torch.testing.assert_close(target, .9*initial+.1*source)
    model.train()
    assert not any(m.training for m in model.teacher_modules())


def test_variance_diagnostics_detect_position_only_collapse():
    # Different constants per patch must not masquerade as batch diversity.
    values = torch.randn(1, 4, 16).expand(8, -1, -1)
    variance, covariance, std, _ = variance_covariance(values)
    assert variance > .98
    assert std < .011
    assert covariance < 1e-8


def test_jepa_rank_logging_cadence_and_exact_collapse(monkeypatch):
    from abc_iueh import boxrr_v2_latent as module
    model=small('jepa_v2');x,y=synthetic(4)
    _,quiet=training_loss(model,x,y,step=198,phase='pretrain')
    assert 'jepa_effective_rank' not in quiet
    _,logged=training_loss(model,x,y,step=199,phase='pretrain')
    assert 1<=logged['jepa_effective_rank']<=64.001
    original=module.variance_covariance
    def collapsed(values):
        var,cov,std,matrix=original(values)
        return var,cov,std,torch.zeros_like(matrix)
    monkeypatch.setattr(module,'variance_covariance',collapsed)
    _,zero=training_loss(model,x,y,step=199,phase='pretrain')
    assert zero['jepa_effective_rank'].item()==0


def test_jepa_inference_parameter_count_matches_fresh_temporal_control():
    model = small('jepa_v2')
    control = build_deterministic('tcn_v2', seed=14, width=8, latent_dim=32)
    counts = parameter_counts(model)
    assert counts['inference'] == sum(p.numel() for p in control.parameters())
    assert counts['total'] > counts['trainable'] > counts['inference']
    x, _ = synthetic(2)
    with torch.no_grad():
        torch.testing.assert_close(model(x), control(x), atol=0, rtol=0)


@pytest.mark.parametrize('name', ['energy_v2', 'jepa_v2'])
def test_checkpoint_round_trip(name, tmp_path):
    model = small(name).eval()
    x, _ = synthetic(2)
    path = tmp_path/'synthetic_state.pt'
    torch.save(model.state_dict(), path)
    replica = small(name).eval()
    replica.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(model(x), replica(x), atol=0, rtol=0)


def test_input_and_phase_guards():
    x, y = synthetic(2)
    with pytest.raises(ValueError):
        training_loss(small('energy_v2'), x, y, phase='pretrain')
    with pytest.raises(ValueError):
        training_loss(small('jepa_v2'), x, y[:, :19])
    with pytest.raises(ValueError):
        training_loss(small('jepa_v2'), x, y, phase='test_time_adapt')
    with pytest.raises(ValueError):
        build_latent_model('other')


"""Independent synthetic geometry, chronology and trainability checks for v2.

No human recordings or fitted artifacts are used by these tests.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from abc_iueh import boxrr_v2_deterministic as m

torch.set_num_threads(2)


def sequence(batch=4):
    generator = torch.Generator().manual_seed(175)
    t = torch.arange(420).float()/100
    signals = torch.stack((.2*torch.sin(2.1*t), .3*torch.sin(3.3*t+.4), .15*torch.cos(1.7*t)), -1)
    axes = torch.randn(batch, 1, 3, generator=generator)*.15
    q = m.exp_so3(signals[None]+axes+.0005*torch.randn(batch, 420, 3, generator=generator))
    return q[:, :400], q[:, 400:]


def identity(batch, length):
    q = torch.zeros(batch, length, 4)
    q[..., 0] = 1
    return q


def activate_neural_heads(net):
    """Do not let zero output initialization make invariance checks vacuous."""
    with torch.no_grad():
        net.decoder.gate.weight.normal_(0, .01)
        net.decoder.residual.weight.normal_(0, .01)
        if hasattr(net, 'horizon_head'):
            net.horizon_head.weight.normal_(0, .01)
        if hasattr(net, 'linear_skip'):
            net.linear_skip.weight.normal_(0, .01)


def test_rotation_features_match_independent_scipy():
    rotation = Rotation.random(73, random_state=np.random.default_rng(91))
    q = torch.tensor(rotation.as_quat()[:, [3, 0, 1, 2]], dtype=torch.float64)
    matrix = rotation.as_matrix()
    expected = np.concatenate((matrix[:, :, 0], matrix[:, :, 1]), -1)
    np.testing.assert_allclose(m.rotation_6d(q), expected, atol=2e-15)
    np.testing.assert_allclose(m.current_gravity(q), matrix[:, 1], atol=2e-15)
    torch.testing.assert_close(m.rotation_6d(-q), m.rotation_6d(q))
    torch.testing.assert_close(m.current_gravity(-q), m.current_gravity(q))


def test_gravity_is_yaw_invariant_and_tilt_sensitive():
    q = m.exp_so3(torch.tensor([[.2, -.3, .4]]))
    yaw = m.exp_so3(torch.tensor([[0., .7, 0.]]))
    torch.testing.assert_close(m.current_gravity(m.multiply_wxyz(yaw, q)), m.current_gravity(q), atol=2e-7, rtol=0)
    pitch = m.exp_so3(torch.tensor([[.7, 0., 0.]]))
    assert (m.current_gravity(m.multiply_wxyz(pitch, q))-m.current_gravity(q)).norm() > .5


def test_feature_dimensions_and_future_teacher_anchor_contract():
    x, y = sequence()
    features = m.MotionFeatures()
    bundle = features(x)
    assert bundle.tokens.shape == (4, 12, 400)
    assert bundle.summary.shape == (4, features.summary_dim)
    assert features.summary_dim == 243
    assert bundle.experts.shape == (4, 15, 20, 3)
    teacher_tokens = features.build_tokens(torch.cat((x[:, -1:], y), 1), reference=x[:, -1])[:, :, 1:]
    rel = m.multiply_wxyz(m.conjugate_wxyz(x[:, -1]), y[:, 0])
    torch.testing.assert_close(teacher_tokens[:, :3, 0]/100, m.log_so3(rel), atol=2e-7, rtol=0)
    assert teacher_tokens.shape == (4, 12, 20)
    assert m.MotionFeatures(False)(x).summary.shape == (4, 240)


def test_constant_angular_velocity_experts_and_stationary_case():
    velocity = torch.tensor([.001, -.002, .0015])
    t = torch.arange(400).float()-399
    q = m.exp_so3(t[None, :, None]*velocity)
    bundle = m.MotionFeatures()(q)
    expected = torch.arange(1, 21)[None, :, None]*velocity
    # CV estimators1..6 and constant-acceleration fits9..11 preserve constant
    # rotation, independent of neural gate damping or zero output initialization.
    for i in (1, 2, 3, 4, 5, 6, 9, 10, 11):
        torch.testing.assert_close(bundle.experts[:, i], expected, atol=3e-6, rtol=0)
    assert m.MotionFeatures()(identity(2, 400)).experts.abs().max() == 0


def test_causal_derivative_fit_predicts_acceleration_better_than_cv():
    # Analytic single-axis constant angular acceleration. A small ridge penalty
    # deliberately shrinks the estimate, so compare error rather than equality.
    t = torch.arange(420).float()-399
    angle = .001*t+.000002*t*t
    q = m.exp_so3(torch.stack((torch.zeros_like(t), angle, torch.zeros_like(t)), -1)[None])
    b = m.MotionFeatures()(q[:, :400])
    target = m.log_so3(q[:, 400:])
    cv_error = (b.experts[:, 3]-target).norm(dim=-1).mean()
    ca_error = (b.experts[:, 11]-target).norm(dim=-1).mean()
    assert ca_error < cv_error*.15


def test_desp_weights_match_independent_recursive_filter():
    rng = np.random.default_rng(214)
    observed = rng.normal(0, .03, (64, 3))
    f = m.MotionFeatures()
    for i, alpha in enumerate((.15, .35, .65)):
        single = observed[0].copy()
        double = observed[0].copy()
        for value in observed[1:]:
            single = alpha*value+(1-alpha)*single
            double = alpha*single+(1-alpha)*double
        expected = alpha/(1-alpha)*(single-double)
        actual = f.desp_slope_weights[i].double().numpy() @ observed
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_predictions_unit_sign_invariance_and_no_cross_batch_dependence(name):
    net = m.build_model(name, seed=9, width=16).eval()
    activate_neural_heads(net)
    x, _ = sequence()
    pred = net(x)
    assert pred.shape == (4, 20, 4) and torch.isfinite(pred).all()
    torch.testing.assert_close(pred.norm(dim=-1), torch.ones(4, 20), atol=3e-6, rtol=0)
    signs = torch.where(torch.arange(400)%3 == 0, -1., 1.).view(1, 400, 1)
    assert m.geodesic_radians(net(x*signs), pred).max() < 2e-6
    torch.testing.assert_close(net(x[:1]), pred[:1], atol=2e-6, rtol=0)
    torch.testing.assert_close(net(x), pred, atol=0, rtol=0)


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_yaw_equivariance_with_gravity_and_all_rotation_without_gravity(name):
    x, _ = sequence()
    for gravity, vector in ((True, [0., .8, 0.]), (False, [.4, -.2, .6])):
        net = m.build_model(name, seed=9, width=16, use_gravity=gravity).eval()
        activate_neural_heads(net)
        rotation = m.exp_so3(torch.tensor([vector]))
        expected = m.multiply_wxyz(rotation[:, None], net(x))
        transformed = net(m.multiply_wxyz(rotation[:, None], x))
        assert m.geodesic_radians(transformed, expected).max() < 4e-6


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_finite_optimization_reaches_backbone_and_physics_gate(name):
    net = m.build_model(name, seed=93, width=16)
    x, y = sequence()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    seen = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        error = m.geodesic_radians(net(x), y)
        loss = error[:, -1].mean()+.1*error.mean()
        assert torch.isfinite(loss)
        loss.backward()
        for key, p in net.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), key
                if p.grad.abs().sum() > 0:
                    seen.add(key)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    assert any('encoder' in key for key in seen)
    assert any('decoder.gate' in key for key in seen)
    assert any('decoder.residual' in key for key in seen)
    if name == 'mlp_v2':
        assert 'linear_skip.weight' in seen
    if name == 'tcn_v2':
        assert 'horizon_projection.weight' in seen


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_stationary_input_gradients_and_future_independence(name):
    net = m.build_model(name, width=16)
    x = identity(3, 400)
    predicted = net(x)
    torch.testing.assert_close(predicted, identity(3, 20), atol=0, rtol=0)
    target = m.exp_so3(torch.full((3, 20, 3), .001))
    loss = m.geodesic_radians(predicted, target).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    # Computing teacher-only target features cannot mutate prediction caches.
    net.features.build_tokens(torch.cat((x[:, -1:], target), 1), reference=x[:, -1])
    torch.testing.assert_close(net(x), predicted, atol=0, rtol=0)


@pytest.mark.parametrize('name', ['mlp_v2', 'tcn_v2'])
def test_full400_learned_temporal_path_not_only_summary(name):
    net = m.build_model(name, width=16)
    x, _ = sequence()
    bundle = net.features(x)
    tokens = bundle.tokens.detach().requires_grad_(True)
    fixed_summary = torch.zeros_like(bundle.summary)
    latent = net.encoder(fixed_summary, tokens)
    weights = torch.arange(latent.shape[-1]).float()
    (latent*weights).sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad[:, :, :128].abs().sum() > 1e-6
    changed = tokens.detach().clone()
    changed[:, :3, :128] += .5
    assert (net.encoder(fixed_summary, changed)-latent.detach()).abs().max() > 1e-5


def test_temporal_stack_prefix_causality():
    encoder = m.TemporalEncoder(243, width=16)
    tokens = torch.randn(2, 12, 80)
    changed = tokens.clone()
    changed[:, :, 43:] += 20
    torch.testing.assert_close(encoder.encode_tokens(tokens)[:, :, :43],
                               encoder.encode_tokens(changed)[:, :, :43], atol=0, rtol=0)


def test_ordered_decoder_is_noncommutative_product():
    bundle = m.MotionFeatures()(identity(1, 400))
    decoder = m.OrderedDecoder(8, 15)
    with torch.no_grad():
        decoder.residual.bias.copy_(torch.tensor([3., 0., 0., 0., 4., 0., 0., 0., 0.]))
    predicted = decoder(torch.zeros(1, 8), bundle)[0, -1]
    expected = Rotation.from_rotvec([.3, 0, 0])*Rotation.from_rotvec([0, .4, 0])
    np.testing.assert_allclose(predicted.detach().numpy()[[1, 2, 3, 0]], expected.as_quat(), atol=2e-7)
    single = m.exp_so3(torch.tensor([.3, .4, 0.]))
    assert m.geodesic_radians(predicted, single) > .05


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_config_seed_and_checkpoint_roundtrip(name, tmp_path):
    net = m.build_model(name, seed=18, width=64, latent_dim=128).eval()
    same = m.build_model(name, seed=18, width=64, latent_dim=128).eval()
    for a, b in zip(net.parameters(), same.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    activate_neural_heads(net)
    x, _ = sequence(2)
    file = tmp_path/'weights.pt'
    torch.save(net.state_dict(), file)
    same.load_state_dict(torch.load(file, weights_only=True))
    torch.testing.assert_close(net(x), same(x), atol=0, rtol=0)


def test_invalid_contract_and_unknown_config_fail_closed():
    with pytest.raises(ValueError, match='400'):
        m.build_model('tcn_v2')(identity(2, 399))
    with pytest.raises(ValueError, match='Unknown deterministic'):
        m.build_model('typo')
    with pytest.raises(ValueError, match='Unknown model options'):
        m.build_model('tcn_v2', future_information=True)
    with pytest.raises(ValueError, match='Unknown model options'):
        m.build_model('ordered_v2', kernel=31)


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_torchscript_trace_deployment_accepts_new_batch_size(name):
    net = m.build_model(name, width=16).eval()
    activate_neural_heads(net)
    x, _ = sequence(2)
    with pytest.warns(torch.jit.TracerWarning):
        traced = torch.jit.trace(net, (x,), check_trace=False)
    different, _ = sequence(3)
    torch.testing.assert_close(traced(different), net(different), atol=3e-6, rtol=0)

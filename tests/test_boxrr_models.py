"""Synthetic geometry, leakage and objective tests for the five research arms."""
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from abc_iueh import boxrr_models as m
from abc_iueh.boxrr_train import model_module

torch.set_num_threads(2)


def sequence(batch=5):
    generator = torch.Generator().manual_seed(807)
    u = torch.randn(batch, 420, 3, generator=generator)*.003
    u = u+torch.tensor([.002, -.001, .003]).view(1, 1, 3)
    q = m.exp_so3(u.cumsum(1))
    return q[:, :400], q[:, 400:]


def baseline():
    path = Path(__file__).parents[1] / 'src/abc_iueh/ordered_motion_model.py'
    return model_module(path)


def test_geometry_matches_independent_scipy_and_signs():
    rng = np.random.default_rng(72)
    a = Rotation.random(150, random_state=rng)
    b = Rotation.random(150, random_state=rng)
    aq = torch.tensor(a.as_quat()[:, [3, 0, 1, 2]])
    bq = torch.tensor(b.as_quat()[:, [3, 0, 1, 2]])
    expected = (a.inv()*b).magnitude()
    np.testing.assert_allclose(m.geodesic_radians(aq, bq), expected, atol=1e-12)
    torch.testing.assert_close(m.geodesic_radians(-aq, bq), m.geodesic_radians(aq, bq))
    reconstructed = m.exp_so3(m.log_so3(aq))
    assert m.geodesic_radians(aq, reconstructed).max() < 1e-12
    assert m.geodesic_radians(torch.tensor([[1., 0., 0., 0.]]), torch.tensor([[0., 1., 0., 0.]])).item() == pytest.approx(np.pi)


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_shapes_units_determinism_and_global_rotation(name):
    net = m.build_model(name, baseline_module=baseline(), seed=33).eval()
    x, _ = sequence()
    pred = net(x)
    assert pred.shape == (5, 20, 4)
    assert torch.isfinite(pred).all()
    torch.testing.assert_close(pred.norm(dim=-1), torch.ones(5, 20), atol=3e-6, rtol=0)
    torch.testing.assert_close(net(x), pred, atol=0, rtol=0)
    signs = torch.where(torch.arange(400) % 3 == 0, -1., 1.).view(1, 400, 1)
    assert m.geodesic_radians(net(x*signs), pred).max() < 1e-5
    rotation = m.exp_so3(torch.tensor([[.3, -.4, .8]]))
    moved = m.multiply_wxyz(rotation[:, None], x)
    expected = m.multiply_wxyz(rotation[:, None], pred)
    assert m.geodesic_radians(net(moved), expected).max() < 2e-5


@pytest.mark.parametrize('name', m.MODEL_NAMES)
def test_optimization_finite_and_reaches_learned_components(name):
    net = m.build_model(name, baseline_module=baseline(), seed=14)
    x, y = sequence()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    saw_nonzero = set()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostic = m.training_loss(net, x, y, step, 100)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert torch.isfinite(diagnostic['endpoint_radians'])
        loss.backward()
        for key, parameter in net.named_parameters():
            if not parameter.requires_grad:
                assert parameter.grad is None
            elif parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), key
                if parameter.grad.abs().sum() > 0:
                    saw_nonzero.add(key)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        m.after_optimizer_step(net)
    assert saw_nonzero
    if name != 'ordered':
        assert any('decoder.residual' in p for p in saw_nonzero)
        assert any('encoder' in p for p in saw_nonzero)
    if name == 'energy':
        assert any('energy_head' in p for p in saw_nonzero)
    if name == 'jepa':
        assert any('predictor' in p for p in saw_nonzero)
        assert all(not p.requires_grad for p in net.target_encoder.parameters())


@pytest.mark.parametrize('name', ['residual_mlp', 'tcn', 'energy', 'jepa'])
def test_stationary_input_has_finite_loss_and_gradients(name):
    net = m.build_model(name)
    x = torch.zeros(4, 400, 4); x[..., 0] = 1
    y = torch.zeros(4, 20, 4); y[..., 0] = 1
    loss, _ = m.training_loss(net, x, y, 1000, 20000)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_causal_conv_does_not_read_later_tokens():
    net = m.CausalBlock(8, 4).eval()
    x = torch.randn(2, 8, 40)
    changed = x.clone(); changed[:, :, 25:] += 100
    torch.testing.assert_close(net(x)[:, :, :25], net(changed)[:, :, :25])


@pytest.mark.parametrize('name', ['energy', 'jepa'])
def test_future_targets_cannot_change_inference_without_parameter_update(name):
    net = m.build_model(name).eval()
    x, y = sequence()
    before = net(x).detach()
    m.training_loss(net, x, y, 10, 100)
    m.training_loss(net, x, -y.roll(1, 0), 10, 100)
    after = net(x).detach()
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_energy_inference_actually_uses_energy_distribution():
    net = m.build_model('energy').eval()
    x, _ = sequence()
    z, last, velocity = net.encode(x)
    tangent = net.decoder.tangent(z, velocity)
    candidates = net.candidates(tangent[:, -1], velocity)
    def choose(self, z, candidate, proposal):
        energy = torch.full(candidate.shape[:2], 100., device=candidate.device)
        energy[:, 3] = -100.
        return energy
    net.energy = types.MethodType(choose, net)
    result = net.decode_energy(z, tangent, velocity)
    torch.testing.assert_close(result[:, -1], candidates[:, 3])
    assert (result[:, -1]-tangent[:, -1]).abs().max() > .02


def test_ema_teacher_update_is_exact_and_stays_frozen():
    net = m.build_model('jepa')
    teacher = next(net.target_encoder.parameters())
    online = next(net.token_encoder.parameters())
    initial = teacher.detach().clone()
    with torch.no_grad(): online.add_(.02)
    expected = initial*.998+online.detach()*.002
    m.after_optimizer_step(net)
    torch.testing.assert_close(teacher, expected)
    assert teacher.requires_grad is False
    net.train()
    assert not net.target_encoder.training


def test_checkpoint_roundtrip_retains_ema_and_energy(tmp_path):
    x, y = sequence()
    for name in ['energy', 'jepa']:
        net = m.build_model(name).eval()
        output = net(x)
        file = tmp_path / (name+'.pt')
        torch.save(net.state_dict(), file)
        restored = m.build_model(name, seed=88).eval()
        restored.load_state_dict(torch.load(file, weights_only=True))
        torch.testing.assert_close(restored(x), output, atol=0, rtol=0)


def test_invalid_model_and_shapes_fail_explicitly():
    with pytest.raises(ValueError): m.build_model('not_a_model')
    with pytest.raises(ValueError): m.build_model('ordered')
    with pytest.raises(ValueError): m.build_model('tcn')(torch.ones(2, 399, 4))

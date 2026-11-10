"""Synthetic invariance, context reach, gradient and export tests; no recordings."""
import pytest
import torch

from abc_iueh.boxrr_v2_extra import build_extra_model
from abc_iueh.boxrr_v2_deterministic import (
    exp_so3, geodesic_radians, multiply_wxyz)

torch.set_num_threads(2)


def motion(batch=3):
    g = torch.Generator().manual_seed(427)
    delta = .002*torch.randn(batch, 420, 3, generator=g)
    delta += torch.tensor([.003, -.001, .002])
    poses = exp_so3(delta.cumsum(1))
    return poses[:, :400], poses[:, 400:]


@pytest.mark.parametrize('use_gravity', [False, True])
def test_output_signs_frame_and_no_state(use_gravity):
    model = build_extra_model(seed=17, use_gravity=use_gravity).eval()
    # Exercise the learned recurrent branch, not only the initial physics prior.
    with torch.no_grad():
        generator = torch.Generator().manual_seed(921)
        for head in (model.decoder.gate, model.decoder.residual):
            head.weight.copy_(.01*torch.randn(head.weight.shape, generator=generator))
    x, _ = motion()
    y = model(x)
    assert y.shape == (3, 20, 4)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.norm(dim=-1), torch.ones(3, 20), atol=3e-6, rtol=0)
    signs = torch.where(torch.arange(400) % 3 == 0, -1., 1.)[None, :, None]
    assert geodesic_radians(model(x*signs), y).max() < 1e-5
    # Optional world-up feature permits global yaw invariance; arbitrary global
    # rotation is required only for the explicitly relative-only configuration.
    rot = [.0, .8, .0] if use_gravity else [.3, -.4, .8]
    r = exp_so3(torch.tensor([rot]))
    moved = multiply_wxyz(r[:, None], x)
    expected = multiply_wxyz(r[:, None], y)
    assert geodesic_radians(model(moved), expected).max() < 2e-5
    model(x.flip(0))
    torch.testing.assert_close(model(x), y, atol=0, rtol=0)


def test_all400_samples_have_encoder_paths_and_patch_order_is_preserved():
    model = build_extra_model(seed=9)
    x, _ = motion(batch=2)
    bundle = model.features(x)
    tokens = bundle.tokens.detach().requires_grad_(True)
    summary = bundle.summary.detach()
    encoded = model.encoder(summary, tokens)
    weights = torch.linspace(-1, 1, encoded.shape[-1])
    (encoded*weights).sum().backward()
    # Every temporal position reaches the recurrent branch, including sample0;
    # this does not assert the trained model will use every sample equally.
    assert torch.all(tokens.grad.abs().sum((0, 1)) > 0)
    patches = model.encoder.patch_tokens(tokens)
    assert patches.shape == (2, 50, 8*12)
    torch.testing.assert_close(patches[:, 0], tokens[:, :, :8].transpose(1, 2).flatten(1))
    swapped = tokens.detach().clone()
    swapped[:, :, [0, 1]] = swapped[:, :, [1, 0]]
    assert not torch.allclose(model.encoder(summary, swapped), encoded, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('stationary', [False, True])
def test_finite_training_gradients_reach_recurrent_and_recent_branches(stationary):
    model = build_extra_model(seed=3, width=16, latent_dim=64)
    x, y = motion(batch=4)
    if stationary:
        x = torch.zeros_like(x); x[..., 0] = 1
        y = torch.zeros_like(y); y[..., 0] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    nonzero = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        angle = geodesic_radians(model(x), y)
        loss = angle[:, -1].mean()+.2*angle.mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), name
                if param.grad.abs().sum() > 0:
                    nonzero.add(name)
        optimizer.step()
    if not stationary:
        for branch in ('encoder.recurrent', 'encoder.recent', 'encoder.patch_projection', 'decoder.residual'):
            assert any(name.startswith(branch) for name in nonzero), branch


def test_seed_is_local_and_interface_rejects_non400_context():
    state = torch.random.get_rng_state().clone()
    a = build_extra_model(seed=44)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    b = build_extra_model(seed=44)
    for first, second in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    x, _ = motion(batch=1)
    with pytest.raises(ValueError, match='400'):
        a(x[:, :-1])
    with pytest.raises(ValueError):
        build_extra_model('unrecognized')
    with pytest.raises(ValueError):
        build_extra_model(unknown_option=1)


def test_static_torchscript_export_preserves400_to20_predictions():
    model = build_extra_model(seed=31, width=16, latent_dim=64).eval()
    x, _ = motion(batch=1)
    with torch.no_grad():
        model.decoder.residual.weight.fill_(.001)
        traced = torch.jit.trace(model, x, check_trace=True)
        torch.testing.assert_close(traced(x), model(x), atol=1e-6, rtol=1e-6)

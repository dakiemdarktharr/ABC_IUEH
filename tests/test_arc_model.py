"""Synthetic training geometry and controlled-arm tests; no human data reads."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abc_iueh.arc_model import (
    ARMS, PARAMETER_COUNTS, ArcModel, conjugate_wxyz, exp_so3,
    geodesic_radians, log_so3, multiply_wxyz, nearest_probabilities,
    normalize_quaternion,
)
from abc_iueh.geometry import geodesic_deg as reference_degrees


def context_batch(batch=3, seed=20261028):
    rng = np.random.default_rng(seed)
    q = [exp_so3(torch.tensor(rng.normal(0, .3, (batch, 3)), dtype=torch.float32))]
    for _ in range(399):
        delta = exp_so3(torch.tensor(rng.normal(0, .007, (batch, 3)), dtype=torch.float32))
        q.append(normalize_quaternion(multiply_wxyz(delta, q[-1])))
    return torch.stack(q, dim=1)


@pytest.mark.parametrize("arm", ARMS)
def test_counts_shapes_units_and_losses(arm):
    model = ArcModel(arm, 20261026)
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNTS[arm]
    assert sum(p.numel() for p in model.encoder.parameters()) == 1360
    context = context_batch()
    target = multiply_wxyz(exp_so3(torch.tensor([[.02, -.03, .01]])), context[:, -1])
    prediction = model(context)
    assert prediction.shape == (3, 20, 4)
    assert prediction.dtype == torch.float32
    torch.testing.assert_close(torch.linalg.vector_norm(prediction, dim=-1), torch.ones(3, 20), atol=2e-7, rtol=0)
    assert model.features(context).shape == (3, 3, 341)
    costs = model.candidate_costs(context, target)
    independent = reference_degrees(model.endpoint_candidates(context).detach().numpy(), target[:, None].numpy())
    np.testing.assert_allclose(costs.detach().numpy() * 180 / np.pi, independent, atol=2e-5, rtol=2e-5)
    per_example = model.training_loss(context, target, reduction="none")
    assert per_example.shape == (3,) and torch.isfinite(per_example).all()
    torch.testing.assert_close(model.training_loss(context, target), per_example.mean())
    model.training_loss(context, target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_common_encoder_initialization_and_global_rng_preservation():
    torch.manual_seed(341)
    before = torch.random.get_rng_state().clone()
    models = [ArcModel(arm, 20261026) for arm in ARMS]
    assert torch.equal(before, torch.random.get_rng_state())
    assert len({m.encoder_sha256() for m in models}) == 1
    assert models[0].encoder_sha256() != ArcModel("nll21", 20261027).encoder_sha256()
    for m in models:
        for name, p in m.named_parameters():
            if name.endswith("bias"):
                assert torch.count_nonzero(p) == 0


@pytest.mark.parametrize("arm", ARMS)
def test_ignored_prefix_and_future_suffix_causality(arm):
    model = ArcModel(arm)
    context = context_batch(batch=1)
    changed_prefix = context.clone()
    changed_prefix[:, :58] = context_batch(batch=1, seed=99)[:, :58]
    torch.testing.assert_close(model(context), model(changed_prefix), atol=0, rtol=0)
    # Extending a recording never changes an explicitly selected past window.
    longer = torch.cat((context, context_batch(batch=1, seed=76)[:, :20]), dim=1)
    changed_future = longer.clone()
    changed_future[:, 400:] = context_batch(batch=1, seed=77)[:, :20]
    torch.testing.assert_close(model(longer[:, :400]), model(changed_future[:, :400]), atol=0, rtol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_sign_and_world_frame_consistency(arm):
    model = ArcModel(arm)
    context = context_batch(batch=2)
    signs = torch.tensor(np.random.default_rng(22).choice([-1., 1.], (2, 400, 1)), dtype=torch.float32)
    original = model(context)
    assert geodesic_radians(original, model(context * signs)).max() < 2e-6
    global_q = exp_so3(torch.tensor([[[.6, -.3, .8]]]))
    changed = model(multiply_wxyz(global_q, context))
    expected = multiply_wxyz(global_q, original)
    assert geodesic_radians(changed, expected).max() < 3e-6


def test_geometry_at_zero_has_finite_gradients_and_correct_exp_jacobian():
    zero = torch.zeros(2, 3, requires_grad=True)
    q = exp_so3(zero)
    q[..., 1:].sum().backward()
    torch.testing.assert_close(zero.grad, torch.full((2, 3), .5))
    v = torch.zeros(2, 3, requires_grad=True)
    geodesic_radians(exp_so3(v), torch.tensor([[1., 0, 0, 0]])).sum().backward()
    assert torch.isfinite(v.grad).all()
    torch.testing.assert_close(v.grad, torch.zeros_like(v))
    near = torch.tensor([[1e-7, -2e-7, 3e-7]], requires_grad=True)
    log_so3(exp_so3(near)).sum().backward()
    torch.testing.assert_close(near.grad, torch.ones_like(near), atol=1e-5, rtol=1e-5)


def test_noncommuting_angle_gradient_agrees_with_finite_difference():
    vector = torch.tensor([[.04, -.03, .02]], dtype=torch.float64, requires_grad=True)
    target = exp_so3(torch.tensor([[-.02, .05, .03]], dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda x: geodesic_radians(exp_so3(x), target),
                                    (vector,), eps=1e-6, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("arm", ARMS)
def test_stationary_training_and_long_arc_fallback(arm):
    model = ArcModel(arm)
    stationary = torch.tensor([1., 0, 0, 0]).view(1, 1, 4).repeat(2, 400, 1)
    target = exp_so3(torch.tensor([[0., 0, 0], [.02, -.01, .03]]))
    loss = model.training_loss(stationary, target)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if arm != "vector_angle":
        torch.testing.assert_close(model(stationary), stationary[:, -1:].expand(-1, 20, -1), atol=0, rtol=0)
    # Every arm has the same deterministic CAV fallback at20||v||>=pi.
    indices = torch.arange(400, dtype=torch.float32).view(1, 400, 1)
    fast = exp_so3(indices * torch.tensor([[[.2, 0, 0]]]))
    expected = multiply_wxyz(exp_so3(model.horizons * log_so3(multiply_wxyz(fast[:, -1:], conjugate_wxyz(fast[:, -2:-1])))), fast[:, -1:])
    assert geodesic_radians(model(fast), expected).max() < 2e-6


def test_vector_zero_residual_matches_cav_and_noncommuting_body_composition():
    model = ArcModel("vector_angle")
    context = context_batch(batch=2)
    output = torch.zeros(2, 3, requires_grad=True)
    current, velocity = model._current_velocity(context)
    expected = multiply_wxyz(exp_so3(model.horizons * velocity), current)
    prediction = model._prediction(context, output, "median")
    assert geodesic_radians(prediction, expected).max() < 2e-6
    residual = torch.tensor([[.03, -.02, .04], [-.02, .01, .05]], requires_grad=True)
    actual = model._prediction(context, residual, "median")
    pure_v = torch.cat((torch.zeros_like(velocity[..., :1]), velocity), dim=-1)
    body_v = multiply_wxyz(multiply_wxyz(conjugate_wxyz(current), pure_v), current)[..., 1:]
    expected = multiply_wxyz(current, exp_so3(model.horizons * body_v + model.horizons / 20 * residual[:, None]))
    assert geodesic_radians(actual, expected).max() < 2e-6
    geodesic_radians(actual[:, -1], context[:, -1]).sum().backward()
    assert torch.isfinite(residual.grad).all() and residual.grad.abs().sum() > 0


def test_decoder_decisions_and_original_off_arc_counterexample():
    model = ArcModel("nll21")
    logits = torch.full((1, 21), -100.)
    logits[0, 0], logits[0, 20] = np.log(.6), np.log(.4)
    assert model.exposure(logits, "median").item() == 0
    torch.testing.assert_close(model.exposure(logits, "mean"), torch.tensor([[[8.]]]))
    assert model.exposure(logits, "map").item() == 0
    indices = torch.arange(21, dtype=torch.float32).view(21, 1)
    candidates = exp_so3(indices * torch.tensor([[0., 0., np.pi / 180]]))
    target_x = exp_so3(torch.tensor([[np.pi / 18, 0., 0.]]))
    target_z = candidates[20:21]
    costs_x, costs_z = geodesic_radians(candidates, target_x), geodesic_radians(candidates, target_z)
    assert costs_x.argmin() == 0 and costs_z.argmin() == 20
    actual_risk = (2 * costs_x + costs_z) / 3
    assert actual_risk[1] < actual_risk[0]
    risk_model = ArcModel("risk21")
    assert risk_model.exposure(actual_risk[None]).item() == actual_risk.argmin().item()
    assert risk_model.exposure(actual_risk[None]).item() > 0
    # Uniform exact ties are preserved, not collapsed to a mean label.
    torch.testing.assert_close(nearest_probabilities(torch.tensor([[0., 1., 0.]])), torch.tensor([[.5, 0., .5]]))


def test_scalar_risk_and_nll_loss_contracts():
    context = context_batch(batch=2)
    target = context[:, -1]
    for arm in ("scalar_angle", "scalar_pinball", "risk21", "nll21"):
        model = ArcModel(arm)
        output = model.logits(context)
        costs = model.candidate_costs(context, target).detach()
        probabilities = nearest_probabilities(costs)
        if arm == "scalar_angle":
            expected = geodesic_radians(model(context)[:, -1], target)
        elif arm == "scalar_pinball":
            expected = (probabilities * .5 * (20 * torch.sigmoid(output) - model.candidate_indices).abs() / 20).sum(-1)
        elif arm == "risk21":
            expected = ((output - costs) ** 2).mean(-1)
        else:
            expected = -(probabilities * torch.log_softmax(output, -1)).sum(-1)
        torch.testing.assert_close(model.training_loss(context, target, "none"), expected)
    scalar = ArcModel("scalar_angle")
    assert 0 < scalar.exposure(torch.tensor([[.1]])).item() < 20


def test_invalid_arm_context_decoder_and_reduction():
    with pytest.raises(ValueError):
        ArcModel("unknown")
    model = ArcModel("nll21")
    with pytest.raises(ValueError):
        model(torch.zeros(400, 4))
    with pytest.raises(ValueError):
        model.exposure(torch.zeros(1, 21), "unsupported")
    with pytest.raises(ValueError):
        model.training_loss(context_batch(1), torch.tensor([[1., 0, 0, 0]]), "invalid")


@pytest.mark.parametrize("arm", ARMS)
def test_cached_costs_preserve_loss_and_parameter_gradients(arm):
    model = ArcModel(arm)
    context = context_batch(batch=2)
    target = multiply_wxyz(exp_so3(torch.tensor([[.03, .01, -.02]])), context[:, -1])
    costs = model.candidate_costs(context, target).detach()
    ordinary = model.training_loss(context, target)
    ordinary.backward()
    ordinary_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    cached = model.training_loss(context, target, candidate_costs=costs)
    cached.backward()
    torch.testing.assert_close(cached, ordinary, rtol=0, atol=0)
    for p, previous in zip(model.parameters(), ordinary_grads):
        torch.testing.assert_close(p.grad, previous, rtol=0, atol=0)


@pytest.mark.parametrize("decode", ["median", "mean", "map"])
def test_shared_probability_readout_equals_forward(decode):
    model = ArcModel("nll21")
    context = context_batch(batch=2)
    probabilities = torch.softmax(model.logits(context), dim=-1)
    torch.testing.assert_close(model(context, decode), model.decode_probabilities(context, probabilities, decode), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optional actual CUDA synthetic backward")
@pytest.mark.parametrize("arm", ARMS)
def test_actual_cuda_finite_backward_for_stationary_and_noncommuting(arm):
    model = ArcModel(arm).cuda()
    moving = context_batch(batch=2).cuda()
    stationary = torch.tensor([1., 0, 0, 0], device="cuda").view(1, 1, 4).repeat(2, 400, 1)
    context = torch.cat((stationary, moving))
    target = multiply_wxyz(exp_so3(torch.tensor([[.02, -.03, .01]], device="cuda")), context[:, -1])
    loss = model.training_loss(context, target)
    loss.backward()
    assert loss.device.type == "cuda" and torch.isfinite(loss)
    assert all(p.grad is not None and p.grad.device.type == "cuda" and torch.isfinite(p.grad).all() for p in model.parameters())

"""Synthetic selector math; these tests never read human recordings."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abc_iueh.evidence import build_segment_features
from abc_iueh.evaluation import predict_endpoint
from abc_iueh.geometry import exp_so3, geodesic_deg, multiply_wxyz
from abc_iueh.selector import (
    DENSE_MACS, PARAMETERS, VARIANTS, SelectorMLP, expected_expert_loss,
    normalise_features, predict_blended_quaternions,
)


def scaler():
    return {"fit_partition": "train", "ddof": 0, "epsilon": 1e-8,
            "mean": np.arange(10).tolist(), "scale": np.linspace(.5, 2, 10).tolist()}


def test_exact_architecture_initialization_and_local_rng_scope():
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    model = SelectorMLP(seed=20261026)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(value.numel() for value in model.parameters()) == PARAMETERS == 244
    assert DENSE_MACS == 10 * 16 + 16 * 4 == 224
    rng = np.random.default_rng(20261026)
    np.testing.assert_array_equal(model.hidden.weight.detach(), rng.normal(0, .02, (10, 16)).astype("f4").T)
    np.testing.assert_array_equal(model.output.weight.detach(), rng.normal(0, .02, (16, 4)).astype("f4").T)
    assert torch.count_nonzero(model.hidden.bias) == torch.count_nonzero(model.output.bias) == 0
    other = SelectorMLP(seed=20261026)
    assert all(torch.equal(value, other.state_dict()[name]) for name, value in model.state_dict().items())
    different = SelectorMLP(seed=20261027)
    assert not torch.equal(model.hidden.weight, different.hidden.weight)
    result = model(torch.zeros((7, 10)))
    assert result.shape == (7, 4) and result.dtype == torch.float32
    torch.testing.assert_close(result.sum(dim=1), torch.ones(7), rtol=0, atol=1e-7)
    assert bool((result >= 0).all())
    with pytest.raises(ValueError):
        model(torch.zeros((7, 14)))  # Future labels cannot be appended as inputs.


def test_existing_scaler_float64_arithmetic_then_float32_without_fit():
    raw = np.random.default_rng(12).normal(size=(13, 10)).astype("f4")
    state = scaler()
    expected = ((raw.astype("f8") - state["mean"]) / state["scale"]).astype("f4")
    actual = normalise_features(raw, state)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert state == scaler()
    for changed in (dict(state, fit_partition="development"), dict(state, scale=[0] * 10),
                    dict(state, ddof=1), dict(state, mean=[float("nan")] * 10)):
        with pytest.raises(ValueError):
            normalise_features(raw, changed)


def test_manual_numpy_sgd_update_and_autograd_parity():
    model = SelectorMLP(20261026).double()
    rng = np.random.default_rng(53)
    x = rng.normal(size=(9, 10))
    cost = rng.uniform(0, 45, size=(9, 4))
    w1 = model.hidden.weight.detach().numpy().T.copy()
    b1 = model.hidden.bias.detach().numpy().copy()
    w2 = model.output.weight.detach().numpy().T.copy()
    b2 = model.output.bias.detach().numpy().copy()
    z = x @ w1 + b1
    hidden = np.maximum(z, 0)
    logits = hidden @ w2 + b2
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights = exponent / exponent.sum(axis=1, keepdims=True)
    expected = (weights * cost).sum(axis=1, keepdims=True)
    dlogit = weights * (cost - expected) / len(x)
    gw2, gb2 = hidden.T @ dlogit, dlogit.sum(axis=0)
    dz = (dlogit @ w2.T) * (z > 0)
    gw1, gb1 = x.T @ dz, dz.sum(axis=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=0, weight_decay=0)
    loss = expected_expert_loss(model(torch.tensor(x)), torch.tensor(cost))
    loss.backward()
    assert loss.item() == pytest.approx(expected.mean(), abs=1e-12)
    for actual, wanted in ((model.hidden.weight.grad.T, gw1), (model.hidden.bias.grad, gb1),
                           (model.output.weight.grad.T, gw2), (model.output.bias.grad, gb2)):
        np.testing.assert_allclose(actual, wanted, atol=1e-13, rtol=1e-11)
    analytic = model.output.bias.grad[1].item()
    epsilon = 1e-5
    with torch.no_grad():
        model.output.bias[1] += epsilon
    plus = expected_expert_loss(model(torch.tensor(x)), torch.tensor(cost)).item()
    with torch.no_grad():
        model.output.bias[1] -= 2 * epsilon
    minus = expected_expert_loss(model(torch.tensor(x)), torch.tensor(cost)).item()
    with torch.no_grad():
        model.output.bias[1] += epsilon
    assert analytic == pytest.approx((plus - minus) / (2 * epsilon), abs=3e-10)
    optimizer.step()
    for actual, wanted in ((model.hidden.weight.T, w1 - .001 * gw1), (model.hidden.bias, b1 - .001 * gb1),
                           (model.output.weight.T, w2 - .001 * gw2), (model.output.bias, b2 - .001 * gb2)):
        np.testing.assert_allclose(actual.detach(), wanted, atol=1e-15, rtol=1e-12)


def test_world_velocity_blend_full_interface_and_surrogate_gap():
    weights = np.array([[.5, .5, 0, 0]], dtype=np.float32)
    velocities = np.array([[[.01, 0, 0], [-.01, 0, 0], [0, .1, 0], [0, 0, .1]]])
    current = exp_so3([[.2, .3, -.1]]).astype("f4")
    prediction = predict_blended_quaternions(weights, velocities, current)
    assert prediction.shape == (1, 20, 4) and prediction.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(prediction, axis=-1), 1, atol=1e-7)
    actual = geodesic_deg(prediction[:, 19], current)[0]
    cost = torch.tensor([[np.degrees(.2), np.degrees(.2), 70, 80]], dtype=torch.float32)
    surrogate = expected_expert_loss(torch.tensor(weights), cost).item()
    assert actual < 1e-8 and surrogate > 11
    one_hot = np.array([[0, 1, 0, 0]], dtype=np.float32)
    emitted = predict_blended_quaternions(one_hot, velocities, current)
    wanted = multiply_wxyz(exp_so3(np.arange(1, 21)[:, None] * velocities[0, 1]), current[0]).astype("f4")
    assert np.max(geodesic_deg(emitted[0], wanted)) < 3e-6


def test_first_usable_origin_has_no_suffix_label_or_feature_leakage():
    t = np.arange(470)
    q = exp_so3(np.column_stack((.002 * t, .000001 * t**2, .01 * np.sin(t / 9)))).astype("f4")
    changed = q.copy()
    changed[400:] = multiply_wxyz(exp_so3([.7, -.2, .3]), changed[400:]).astype("f4")
    original = build_segment_features(q)
    perturbed = build_segment_features(changed)
    assert original["origin_index"][0] == 399
    assert not np.array_equal(original["expert_loss"][0], perturbed["expert_loss"][0])
    model = SelectorMLP()
    identity = {"fit_partition": "train", "ddof": 0, "epsilon": 1e-8, "mean": [0] * 10, "scale": [1] * 10}
    for variant in VARIANTS:
        np.testing.assert_array_equal(original["features"][variant][0], perturbed["features"][variant][0])
        left = torch.tensor(normalise_features(original["features"][variant][:1], identity))
        right = torch.tensor(normalise_features(perturbed["features"][variant][:1], identity))
        torch.testing.assert_close(model(left), model(right), rtol=0, atol=0)


def test_endpoint_only_evaluation_exact_full_emitted_trajectory_parity():
    rng = np.random.default_rng(103)
    current = exp_so3(rng.normal(size=(513, 3))).astype("f4")
    velocities = rng.normal(0, .08, size=(513, 4, 3))
    # Every row uses distinct non-collinear axes and distinct simplex weights.
    raw = rng.uniform(.01, 1, size=(513, 4))
    weights = (raw / raw.sum(axis=1, keepdims=True)).astype("f4")
    full = predict_blended_quaternions(weights, velocities, current)
    endpoint = predict_endpoint(current, velocities, weights)
    np.testing.assert_array_equal(endpoint, full[:, 19])
    assert np.max(geodesic_deg(endpoint, full[:, 19])) < 1e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="synthetic CUDA test requires CUDA PyTorch")
def test_synthetic_cuda_forward_backward_and_parameter_update():
    model = SelectorMLP().cuda()
    features = torch.ones((128, 10), device="cuda")
    labels = torch.arange(4, device="cuda", dtype=torch.float32)[None].expand(128, -1)
    before = model.output.bias.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=0, weight_decay=0)
    weights = model(features)
    loss = expected_expert_loss(weights, labels)
    loss.backward()
    assert weights.is_cuda and loss.is_cuda and all(p.grad.is_cuda for p in model.parameters())
    optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(before, model.output.bias)
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="synthetic CUDA parity requires CUDA PyTorch")
def test_synthetic_cuda_cpu_one_update_parity():
    cpu, gpu = SelectorMLP(), SelectorMLP().cuda()
    rng = np.random.default_rng(65)
    features = rng.normal(size=(128, 10)).astype("f4")
    labels = rng.uniform(0, 30, size=(128, 4)).astype("f4")
    weights = []
    for model, device in ((cpu, "cpu"), (gpu, "cuda")):
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=0, weight_decay=0)
        prediction = model(torch.tensor(features, device=device))
        weights.append(prediction.detach().cpu())
        expected_expert_loss(prediction, torch.tensor(labels, device=device)).backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.testing.assert_close(weights[0], weights[1], rtol=1e-6, atol=1e-7)
    for name, value in cpu.state_dict().items():
        torch.testing.assert_close(value, gpu.state_dict()[name].cpu(), rtol=1e-6, atol=1e-8)

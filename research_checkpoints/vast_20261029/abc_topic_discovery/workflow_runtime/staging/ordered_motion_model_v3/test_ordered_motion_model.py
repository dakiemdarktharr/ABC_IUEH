"""Synthetic-only tests; intentionally not run before coordinator RAM gate."""
from __future__ import annotations

import math

import torch

from ordered_motion_model import (
    AREA_EPS,
    DirectSameFeatureControl,
    MeansOnlyControl,
    OrderedMotionFeatures,
    OrderedMotionProduct,
    SingleExponentialControl,
    cross3,
    exp_so3,
    endpoint20_geodesic_loss,
    geodesic_radians,
    log_so3,
    multiply_wxyz,
    ordered_area_balanced,
)


def _explicit_area(increments: torch.Tensor) -> torch.Tensor:
    width = increments.shape[1]
    terms = [cross3(increments[:, i], increments[:, j])
             for i in range(width) for j in range(i + 1, width)]
    return torch.stack(terms, dim=0).sum(dim=0) / float(width * width)


def _explicit_area_high_precision(increments: torch.Tensor) -> torch.Tensor:
    return _explicit_area(increments.to(torch.float64))


def _identity_context(batch: int = 2) -> torch.Tensor:
    context = torch.zeros(batch, 400, 4, dtype=torch.float32)
    context[..., 0] = 1.0
    return context


def _synthetic_context(batch: int = 2) -> torch.Tensor:
    """Small generated rotations, with no human or external arrays."""
    torch.manual_seed(1947)
    rates = torch.randn(batch, 399, 3, dtype=torch.float32) * 0.008
    rates[..., 0] += 0.002
    current = torch.zeros(batch, 4, dtype=torch.float32)
    current[:, 0] = 1.0
    frames = [current]
    for index in range(399):
        current = multiply_wxyz(exp_so3(rates[:, index]), current)
        frames.append(current)
    return torch.stack(frames, dim=1)


def _left(context: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return multiply_wxyz(g.view(1, 1, 4).expand_as(context), context)


def _right(context: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return multiply_wxyz(context, h.view(1, 1, 4).expand_as(context))


def _assert_same_rotation(a: torch.Tensor, b: torch.Tensor, atol: float = 3.0e-5) -> None:
    assert torch.all(geodesic_radians(a, b) < atol)


def test_balanced_area_matches_pair_sum_and_reversal() -> None:
    torch.manual_seed(19)
    path = torch.randn(3, 13, 3, dtype=torch.float32) * 0.03
    actual = ordered_area_balanced(path)
    expected = _explicit_area(path)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-7)
    reversed_area = ordered_area_balanced(path.flip(dims=(1,)))
    torch.testing.assert_close(reversed_area, -actual, rtol=2.0e-5, atol=2.0e-7)


def test_near_pi_log_branch_and_quaternion_sign_are_explicit() -> None:
    axis = torch.tensor([0.8, -0.3, 0.51961523], dtype=torch.float32)
    axis = axis / torch.linalg.vector_norm(axis)
    delta = 2.0e-4
    below = log_so3(exp_so3((math.pi - delta) * axis))
    above = log_so3(exp_so3((math.pi + delta) * axis))
    torch.testing.assert_close(torch.linalg.vector_norm(below),
                               torch.tensor(math.pi - delta), rtol=1.0e-5, atol=2.0e-6)
    torch.testing.assert_close(torch.linalg.vector_norm(above),
                               torch.tensor(math.pi - delta), rtol=1.0e-5, atol=2.0e-6)
    torch.testing.assert_close(below, (math.pi - delta) * axis, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(above, -(math.pi - delta) * axis, rtol=2.0e-5, atol=2.0e-5)

    exact_half_turn = torch.cat((torch.zeros(1), axis)).view(1, 4)
    positive_sign = log_so3(exact_half_turn)
    negative_sign = log_so3(-exact_half_turn)
    torch.testing.assert_close(positive_sign, negative_sign, rtol=0.0, atol=1.0e-6)
    assert positive_sign[0, 0] > 0


def test_cancellation_and_nearly_collinear_areas_match_float64_reference() -> None:
    # An exact out-and-back path has zero signed area. A tiny final transverse
    # perturbation leaves a small residual after cancellation of larger terms.
    a = torch.tensor([0.05, 0.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0, 0.05, 0.0], dtype=torch.float64)
    c = torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64)
    loop = torch.stack((a, b, c, -c, -b, -a)).unsqueeze(0)
    loop[0, -1, 1] += 1.0e-5
    loop32 = loop.to(torch.float32).requires_grad_(True)
    loop_area = ordered_area_balanced(loop32)
    loop_reference = _explicit_area_high_precision(loop)
    torch.testing.assert_close(loop_area.to(torch.float64), loop_reference,
                               rtol=2.0e-2, atol=1.0e-11)

    # Nearly collinear vectors have a very small area. Compare the O(n) tree
    # with a float64 explicit pair sum, then stress the area normalization's
    # epsilon and backward path around that cancellation-sensitive value.
    x = torch.linspace(-0.025, 0.025, 64, dtype=torch.float64)
    y = 2.0e-9 * torch.sin(torch.linspace(0.0, 9.0, 64, dtype=torch.float64))
    nearly_collinear = torch.stack((x, y, torch.zeros_like(x)), dim=-1).unsqueeze(0)
    nearly32 = nearly_collinear.to(torch.float32).requires_grad_(True)
    near_area = ordered_area_balanced(nearly32)
    near_reference = _explicit_area_high_precision(nearly_collinear)
    torch.testing.assert_close(near_area.to(torch.float64), near_reference,
                               rtol=5.0e-2, atol=1.0e-14)
    normalized_area = near_area / torch.sqrt((near_area * near_area).sum(dim=-1, keepdim=True)
                                               + AREA_EPS * AREA_EPS)
    (loop_area.sum() + normalized_area.sum()).backward()
    assert torch.isfinite(loop_area).all()
    assert torch.isfinite(loop32.grad).all()
    assert torch.isfinite(normalized_area).all()
    assert torch.isfinite(nearly32.grad).all()


def test_exact_collinear_area_is_zero() -> None:
    scalar = torch.linspace(-0.03, 0.03, 399, dtype=torch.float32)
    path = torch.zeros(1, 399, 3, dtype=torch.float32)
    path[0, :, 0] = scalar
    torch.testing.assert_close(ordered_area_balanced(path), torch.zeros(1, 3),
                               rtol=0.0, atol=0.0)


def test_same_mean_different_internal_order_changes_area() -> None:
    first = torch.zeros(1, 8, 3, dtype=torch.float32)
    first[0, 0, 0], first[0, 1, 1] = 0.2, 0.3
    swapped = first.clone()
    swapped[:, 0], swapped[:, 1] = first[:, 1], first[:, 0]
    torch.testing.assert_close(first.mean(dim=1), swapped.mean(dim=1))
    area_a = ordered_area_balanced(first)
    area_b = ordered_area_balanced(swapped)
    torch.testing.assert_close(area_a, -area_b)
    assert torch.linalg.vector_norm(area_a) > 0


def test_zero_log_exp_and_stationary_model_have_finite_gradients() -> None:
    zero = torch.zeros(4, 3, dtype=torch.float32, requires_grad=True)
    exp_so3(zero).sum().backward()
    assert torch.isfinite(zero.grad).all()

    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    log_so3(identity).sum().backward()
    assert torch.isfinite(identity.grad).all()

    model = OrderedMotionProduct()
    context = _identity_context(1)
    target = context[:, :20].clone()
    target[:, :, 0] = torch.cos(torch.tensor(0.025))
    target[:, :, 1] = torch.sin(torch.tensor(0.025))
    prediction = model(context)
    loss = geodesic_radians(prediction[:, -1], target[:, -1]).mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head[2].weight.grad is not None


def test_output_head_can_learn_from_nonzero_motion() -> None:
    model = OrderedMotionProduct()
    context = _synthetic_context(batch=1)
    prediction = model(context)
    _, latest, _, _ = model.features(context)
    direction = latest / torch.linalg.vector_norm(latest, dim=-1, keepdim=True).clamp_min(1.0e-8)
    target_end = multiply_wxyz(prediction.detach()[:, -1], exp_so3(0.02 * direction.detach()))
    target = torch.cat((prediction.detach()[:, :-1], target_end[:, None]), dim=1)
    model.zero_grad(set_to_none=True)
    endpoint20_geodesic_loss(model(context), target).backward()
    assert torch.isfinite(model.head[2].weight.grad).all()
    assert torch.linalg.vector_norm(model.head[2].weight.grad) > 0

    # Move only the initially-zero output layer once. The hidden layer is
    # expected to start receiving gradient only after this update.
    with torch.no_grad():
        model.head[2].weight -= 1.0e-3 * model.head[2].weight.grad
        model.head[2].bias -= 1.0e-3 * model.head[2].bias.grad
    model.zero_grad(set_to_none=True)
    endpoint20_geodesic_loss(model(context), target).backward()
    assert torch.isfinite(model.head[0].weight.grad).all()
    assert torch.isfinite(model.head[0].bias.grad).all()
    assert torch.linalg.vector_norm(model.head[0].weight.grad) > 0


def test_individual_quaternion_sign_changes_preserve_prediction_rotation() -> None:
    model = OrderedMotionProduct().eval()
    context = _synthetic_context()
    signs = torch.where(torch.arange(400) % 2 == 0, 1.0, -1.0).view(1, 400, 1)
    signed = context * signs
    _assert_same_rotation(model(context), model(signed))


def test_global_left_and_right_equivariance() -> None:
    model = OrderedMotionProduct().eval()
    context = _synthetic_context()
    prediction = model(context)

    g = exp_so3(torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.float32))[0]
    left_prediction = model(_left(context, g))
    _assert_same_rotation(left_prediction, multiply_wxyz(g.view(1, 1, 4), prediction))

    h = exp_so3(torch.tensor([[-0.17, 0.11, 0.23]], dtype=torch.float32))[0]
    right_prediction = model(_right(context, h))
    _assert_same_rotation(right_prediction, multiply_wxyz(prediction, h.view(1, 1, 4)))


def test_shapes_norms_counts_and_zero_head_initialization() -> None:
    models = (OrderedMotionProduct(), MeansOnlyControl(), SingleExponentialControl(),
              DirectSameFeatureControl())
    expected_counts = (2477, 2477, 2477, 2474)
    context = _synthetic_context(batch=1)
    for model, expected in zip(models, expected_counts, strict=True):
        assert model.trainable_parameter_count == expected
        assert torch.count_nonzero(model.head[0].weight) > 0
        assert torch.count_nonzero(model.head[2].weight) == 0
        assert torch.count_nonzero(model.head[2].bias) == 0
        prediction = model(context)
        assert prediction.shape == (1, 20, 4)
        torch.testing.assert_close(torch.linalg.vector_norm(prediction, dim=-1),
                                   torch.ones(1, 20), rtol=2.0e-6, atol=2.0e-6)
    repeated = OrderedMotionProduct()
    assert torch.equal(models[0].head[0].weight, repeated.head[0].weight)
    assert torch.equal(models[0].head[0].bias, repeated.head[0].bias)
    current, latest, vectors, scalars = OrderedMotionFeatures()(context)
    assert current.shape == (1, 4)
    assert latest.shape == (1, 3)
    assert vectors.shape == (1, 15, 3)
    assert scalars.shape == (1, 30)
    s = torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1) / 20.0
    cav = multiply_wxyz(current[:, None], exp_so3(s * (20.0 * latest[:, None, :])))
    _assert_same_rotation(models[0](context), cav)


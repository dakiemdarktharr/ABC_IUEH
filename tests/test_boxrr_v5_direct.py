"""Synthetic software checks only; these tests establish no BOXRR accuracy."""
import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from abc_iueh.boxrr_models import (
    exp_so3, geodesic_radians, multiply_wxyz,
)
from abc_iueh.boxrr_v5_direct import (
    CausalDepthwiseBlock, DirectSO3Conv, RelativeSO3Features,
)

torch.set_num_threads(2)


def sequence(batch=4):
    generator = torch.Generator().manual_seed(851)
    vectors = torch.randn(batch, 420, 3, generator=generator)*.003
    vectors += torch.tensor([.002, -.001, .003])
    poses = exp_so3(vectors.cumsum(1))
    return poses[:, :400], poses[:, 400:]


def common_loss(prediction, target):
    errors = geodesic_radians(prediction, target)
    return errors[:, -1].mean()+.1*errors.mean()


def zero_nonlinear(model):
    with torch.no_grad():
        for head in (model.early_head, model.endpoint_head):
            head.weight.zero_()
            head.bias.zero_()


def test_features_match_independent_noncommuting_scipy_rotations():
    time = np.arange(400)/100
    rotations = Rotation.from_euler(
        'xyz', np.stack((.3*np.sin(time), .2*time, .15*time*time), -1))
    q = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]])[None]
    last, tokens, increments = RelativeSO3Features()(q)
    relative = rotations[-1].inv()*rotations
    matrices = relative.as_matrix()
    expected6d = np.concatenate((matrices[:, :, 0], matrices[:, :, 1]), -1)
    expected_increment = np.zeros((400, 3))
    expected_increment[1:] = (relative[1:]*relative[:-1].inv()).as_rotvec()
    np.testing.assert_allclose(tokens[0, :6].T, expected6d, atol=2e-15)
    np.testing.assert_allclose(increments[0], expected_increment, atol=2e-15)
    np.testing.assert_allclose(tokens[0, 6:].T, 100*expected_increment, atol=2e-13)
    torch.testing.assert_close(last, q[:, -1])
    # A body-frame difference would be wrong for this noncommuting motion.
    body_increment = (rotations[:-1].inv()*rotations[1:]).as_rotvec()
    assert np.max(np.abs(expected_increment[1:]-body_increment)) > .001


@pytest.mark.parametrize('width, expected_parameters', [(32, 29882), (64, 68314)])
def test_shape_units_parameter_count_and_serializable_config(width, expected_parameters):
    torch.manual_seed(3)
    model = DirectSO3Conv(width)
    x, _ = sequence()
    prediction = model(x)
    assert prediction.shape == (4, 20, 4)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction.norm(dim=-1), torch.ones(4, 20), atol=2e-6, rtol=0)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
    config = json.loads(json.dumps(model.get_config()))
    assert config['width'] == width
    assert config['early_heads_detached'] is False
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_arbitrary_antipodes_preserve_features_and_prediction_rotations():
    torch.manual_seed(7)
    model = DirectSO3Conv(32)
    x, _ = sequence()
    signs = torch.where(torch.arange(400).remainder(3) == 0, -1., 1.)
    changed = x*signs[None, :, None]
    _, tokens, increments = model.features(x)
    _, changed_tokens, changed_increments = model.features(changed)
    torch.testing.assert_close(tokens, changed_tokens, atol=0, rtol=0)
    torch.testing.assert_close(increments, changed_increments, atol=0, rtol=0)
    assert geodesic_radians(model(x), model(changed)).max() < 2e-6


def test_change_of_world_frame_commutes_with_forecast():
    torch.manual_seed(8)
    model = DirectSO3Conv(32).double()
    x, _ = sequence()
    x = x.double()
    world = exp_so3(torch.tensor([[.6, -.3, .2]], dtype=torch.float64))
    transformed = multiply_wxyz(world[:, None], x)
    _, tokens, increments = model.features(x)
    _, changed_tokens, changed_increments = model.features(transformed)
    torch.testing.assert_close(tokens, changed_tokens, atol=4e-13, rtol=2e-12)
    torch.testing.assert_close(increments, changed_increments, atol=4e-15, rtol=2e-12)
    expected = multiply_wxyz(world[:, None], model(x))
    assert geodesic_radians(model(transformed), expected).max() < 1e-12


@pytest.mark.parametrize('stationary', [False, True])
def test_initialized_linear_branch_matches_exact_known_motion(stationary):
    model = DirectSO3Conv(32).double()
    zero_nonlinear(model)
    base = exp_so3(torch.tensor([[.4, -.2, .3]], dtype=torch.float64))
    velocity = torch.tensor([.002, -.003, .004], dtype=torch.float64)
    if stationary:
        velocity.zero_()
    past_time = torch.arange(-399., 1., dtype=torch.float64)
    future_time = torch.arange(1., 21., dtype=torch.float64)
    context = multiply_wxyz(base[:, None], exp_so3(past_time[:, None]*velocity)[None])
    expected = multiply_wxyz(base[:, None], exp_so3(future_time[:, None]*velocity)[None])
    assert geodesic_radians(model(context), expected).max() < 5e-15


def test_linear_branch_is_a_direct_trainable_lag_map_with_physical_units():
    model = DirectSO3Conv(32)
    zero_nonlinear(model)
    x, _ = sequence()
    last, _, increments = model.features(x)
    with torch.no_grad():
        model.linear_branch.weight.zero_()
        model.linear_branch.weight[:, 235] = torch.arange(1., 21.)*.4
    displacement = increments[:, 235:236]*(torch.arange(1., 21.)*.4)[None, :, None]
    expected = multiply_wxyz(last[:, None], exp_so3(displacement))
    assert geodesic_radians(model(x), expected).max() < 2e-6


@pytest.mark.parametrize('dilation', [1, 2, 4])
def test_depthwise_block_cannot_read_later_feature_tokens(dilation):
    block = CausalDepthwiseBlock(32, dilation)
    tokens = torch.randn(2, 32, 100)
    changed = tokens.clone()
    changed[:, :, 65:] += 100
    torch.testing.assert_close(block(tokens)[:, :, :65], block(changed)[:, :, :65], atol=0, rtol=0)


@pytest.mark.parametrize('width', [32, 64])
def test_joint_loss_updates_trunk_both_heads_and_independent_linear_branch(width):
    torch.manual_seed(12)
    model = DirectSO3Conv(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, target = sequence()
    x.requires_grad_()
    before_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    before_prediction = model(x).detach()
    loss = common_loss(model(x), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter.detach(), before_parameters[name]), name
    after = model(x).detach()
    assert geodesic_radians(before_prediction[:, -1], after[:, -1]).max() > 1e-4
    assert geodesic_radians(before_prediction[:, :19], after[:, :19]).max() > 1e-4


def test_early_horizon_loss_alone_reaches_shared_trunk():
    torch.manual_seed(13)
    model = DirectSO3Conv(32)
    x, target = sequence()
    geodesic_radians(model(x)[:, :19], target[:, :19]).mean().backward()
    assert model.stem.weight.grad.abs().sum() > 0
    assert model.early_head.weight.grad.abs().sum() > 0
    assert model.time_head.weight.grad[:19].abs().sum() > 0
    assert model.time_head.weight.grad[19:].abs().sum() == 0
    assert model.endpoint_head.weight.grad.abs().sum() == 0


@pytest.mark.parametrize('zero_input', [False, True])
def test_stationary_input_and_identity_target_have_finite_backward(zero_input):
    model = DirectSO3Conv(32)
    x = torch.zeros(2, 400, 4)
    if not zero_input:
        x[..., 0] = 1.
    target = torch.zeros(2, 20, 4)
    target[..., 0] = 1.
    prediction = model(x)
    torch.testing.assert_close(prediction.norm(dim=-1), torch.ones(2, 20), atol=2e-6, rtol=0)
    loss = common_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_batch_independence_and_calls_do_not_mutate_state():
    model = DirectSO3Conv(32).train()
    x, _ = sequence()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    together = model(x)
    separate = torch.cat([model(row[None]) for row in x])
    torch.testing.assert_close(together, separate, atol=3e-6, rtol=2e-6)
    model(-x.roll(1, 0))
    torch.testing.assert_close(model(x), together, atol=0, rtol=0)
    model.eval()
    torch.testing.assert_close(model(x), together, atol=0, rtol=0)
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())


def test_tiny_synthetic_fit_checks_software_learnability_only():
    torch.manual_seed(20)
    model = DirectSO3Conv(32)
    x, _ = sequence(batch=3)
    displacement = torch.tensor([.06, -.03, .04])[None, None].expand(3, 20, 3)
    target = multiply_wxyz(x[:, -1:], exp_so3(displacement))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    def squared_error():
        errors = geodesic_radians(model(x), target).square()
        return errors[:, -1].mean()+.1*errors.mean()
    initial = squared_error().item()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = squared_error()
        loss.backward()
        optimizer.step()
    assert squared_error().item() < initial*.15


def test_invalid_shapes_and_width_fail():
    with pytest.raises(ValueError, match='width'):
        DirectSO3Conv(16)
    with pytest.raises(ValueError, match='shape'):
        DirectSO3Conv()(torch.ones(2, 399, 4))

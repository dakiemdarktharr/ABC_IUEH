import numpy as np
import pytest

from abc_iueh.geometry import exp_so3, multiply_wxyz
from abc_iueh.neural import RelativeMLP, adjacent_features


def synthetic_motion(batch=8, seed=104):
    rng = np.random.default_rng(seed)
    velocity = rng.uniform(-.008, .008, (batch, 1, 3))
    time = np.arange(420)[None, :, None]
    anchor = exp_so3(rng.normal(0, .5, (batch, 1, 3)))
    motion = multiply_wxyz(anchor, exp_so3(time * velocity)).astype(np.float32)
    return motion[:, :400].copy(), motion[:, 400:].copy()


def test_full_interface_parameter_count_and_input_immutability():
    context, target = synthetic_motion(3)
    before_context, before_target = context.copy(), target.copy()
    context.flags.writeable = False
    target.flags.writeable = False
    model = RelativeMLP(seed=81)
    prediction = model.predict(context)
    assert prediction.shape == (3, 20, 4)
    assert prediction.dtype == np.float32 and np.isfinite(prediction).all()
    np.testing.assert_allclose(np.linalg.norm(prediction, axis=-1), 1, atol=2e-7)
    assert model.parameter_count == sum(getattr(model, n).size for n in ("w1", "b1", "w2", "b2")) == 80572
    assert np.isfinite(model.train_step(context, target))
    np.testing.assert_array_equal(context, before_context)
    np.testing.assert_array_equal(target, before_target)


@pytest.mark.parametrize("bad", [
    np.zeros((1, 399, 4), dtype=np.float32),
    np.zeros((400, 4), dtype=np.float32),
    np.zeros((0, 400, 4), dtype=np.float32),
    np.ones((1, 400, 4), dtype=np.float64),
    np.zeros((1, 400, 4), dtype=np.float32),
    np.full((1, 400, 4), np.nan, dtype=np.float32),
    np.full((1, 400, 4), np.inf, dtype=np.float32),
])
def test_predict_rejects_invalid_context(bad):
    with pytest.raises(ValueError):
        RelativeMLP().predict(bad)


def test_features_are_short_arc_body_increments_with_halfturn_tie():
    context = np.tile(np.array([1, 0, 0, 0], np.float32), (1, 400, 1))
    context[:, 200:] = np.array([0, -1, 0, 0], np.float32)
    feature = adjacent_features(context)
    assert feature.shape == (1, 399, 3) and feature.dtype == np.float32
    np.testing.assert_array_equal(feature[0, 199], [2, 0, 0])
    flipped = context.copy()
    flipped[:, ::2] *= -1
    np.testing.assert_array_equal(adjacent_features(flipped), feature)
    # This checks component semantics independently of the model's dense layers.
    angle = .4
    context[:, 200:] = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0], np.float32)
    np.testing.assert_allclose(adjacent_features(context)[0, 199],
                               [0, 2 * np.sin(angle / 2), 0], atol=1e-7)
    assert adjacent_features(context[:, :2]).shape == (1, 1, 3)
    assert adjacent_features(np.tile(context, (1, 3, 1))).shape == (1, 1199, 3)
    with pytest.raises(ValueError):
        adjacent_features(context[:, :1])


def test_quaternion_sign_and_left_world_frame_equivariance():
    context, target = synthetic_motion(4)
    model = RelativeMLP(seed=9, input_mean=[.001, -.002, .0004], input_std=[.01, .02, .03])
    prediction = model.predict(context)
    np.testing.assert_allclose(model.predict(-context), -prediction, atol=2e-7)
    signs = np.ones((4, 400, 1), dtype=np.float32)
    signs[:, ::3] = -1
    np.testing.assert_allclose(model.predict(context * signs), prediction * signs[:, -1:], atol=3e-7)
    rotation = exp_so3([.6, -.4, .3])[None, None]
    rotated_context = multiply_wxyz(rotation, context).astype(np.float32)
    expected = multiply_wxyz(rotation, prediction).astype(np.float32)
    np.testing.assert_allclose(model.predict(rotated_context), expected, atol=8e-7)
    loss = model.loss_and_gradients(context, target)[0]
    flipped_target = target.copy()
    flipped_target[:, ::2] *= -1
    assert model.loss_and_gradients(context, flipped_target)[0] == loss


def test_analytic_gradient_through_both_normalizations_and_composition():
    velocity = np.array([.003, -.004, .002])
    rotation = exp_so3([.3, -.2, .1])
    sequence = multiply_wxyz(rotation, exp_so3(np.arange(420)[:, None] * velocity)).astype(np.float32)
    context, target = sequence[None, :400], sequence[None, 400:]
    model = RelativeMLP(seed=61, input_mean=[.0001, -.0002, .0003], input_std=[.004, .006, .008])
    model.b1[2] = 3
    model.w2[2, 7] = .13
    model.b2[7] = .17
    before = {name: getattr(model, name).copy() for name in ("w1", "b1", "w2", "b2")}
    _, gradients = model.loss_and_gradients(context, target)
    for name, index in (("w1", (4, 2)), ("b1", (2,)), ("w2", (2, 7)), ("b2", (7,))):
        parameter = getattr(model, name)
        original = parameter[index].copy()
        epsilon = np.float32(.001)
        parameter[index] = original + epsilon
        plus = model.loss_and_gradients(context, target)[0]
        parameter[index] = original - epsilon
        minus = model.loss_and_gradients(context, target)[0]
        parameter[index] = original
        numeric = (plus - minus) / (2 * float(epsilon))
        analytic = float(gradients[name][index])
        assert abs(analytic) > 1e-6
        np.testing.assert_allclose(analytic, numeric, rtol=.008, atol=5e-7)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(model, name), value)
    assert model._step == 0


def test_adam_learns_synthetic_velocity_on_unseen_synthetic_examples():
    context, target = synthetic_motion(24, seed=504)
    held_context, held_target = synthetic_motion(12, seed=505)
    model = RelativeMLP(seed=24, input_std=[.004, .004, .004])
    before = model.loss_and_gradients(held_context, held_target)[0]
    for _ in range(120):
        model.train_step(context, target, learning_rate=.003)
    after = model.loss_and_gradients(held_context, held_target)[0]
    assert after < before * .15


def test_checkpoint_prediction_and_adam_continuation_parity(tmp_path):
    context, target = synthetic_motion(3)
    mean, std = np.array([.001, -.001, .002]), np.array([.01, .02, .03])
    model = RelativeMLP(seed=72, input_mean=mean, input_std=std)
    mean[:] = 9
    std[:] = 9
    for _ in range(3):
        model.train_step(context, target)
    path = tmp_path / "checkpoint.npz"
    model.save(path)
    loaded = RelativeMLP.load(path)
    np.testing.assert_array_equal(loaded.predict(context), model.predict(context))
    assert loaded.train_step(context, target) == model.train_step(context, target)
    for name in ("w1", "b1", "w2", "b2", "input_mean", "input_std"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(model, name))
    with np.load(path, allow_pickle=False) as archive:
        assert all(value.dtype.kind != "O" for value in archive.values())
        assert "context" not in archive and "target" not in archive


def test_scaler_validation_and_train_rejection_do_not_update_state():
    for kwargs in ({"input_mean": [1, 2]}, {"input_std": [-1, 1, 1]},
                   {"input_std": [1, np.inf, 1]}):
        with pytest.raises(ValueError):
            RelativeMLP(**kwargs)
    floored = RelativeMLP(input_std=[0, 0, 0])
    np.testing.assert_array_equal(floored.input_std, np.full(3, 1e-8, np.float32))
    context, target = synthetic_motion(2)
    model = RelativeMLP()
    before = model.predict(context)
    for learning_rate in (-.1, 0, np.nan, np.inf):
        with pytest.raises(ValueError):
            model.train_step(context, target, learning_rate=learning_rate)
    with pytest.raises(ValueError, match="batch"):
        model.train_step(context, target[:1])
    with pytest.raises(ValueError):
        model.train_step(context, target.astype(np.float64))
    np.testing.assert_array_equal(model.predict(context), before)
    assert model._step == 0

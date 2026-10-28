"""NumPy reference MLP for the full 400-to-20 quaternion forecasting interface.

The 1197->64 ReLU->60 network has 80,572 trainable parameters. Its inputs are
short-arc adjacent body-frame increments, ``2*vec(inv(q[i])*q[i+1])``, scaled by
three axis statistics supplied by the caller. This module never fits a scaler
or reads a dataset. Fit those statistics on training users only.

Outputs are Cayley-style local rotations ``normalize([1, .5*r])`` composed on
the right of the last observed rotation. All 20 outputs contribute to the
sign-aligned quaternion MSE training loss; this is not the endpoint20 geodesic
reporting metric. Training is offline Adam, with no adaptation during predict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

_F32 = np.float32
_PARAMETER_SHAPES = {"w1": (1197, 64), "b1": (64,), "w2": (64, 60), "b2": (60,)}
_CONJUGATE = np.array([1, -1, -1, -1], dtype=np.float32)
_STD_FLOOR = np.float32(1e-8)


def _multiply(a: NDArray, b: NDArray) -> NDArray[np.float32]:
    """Hamilton product, retaining float32 and broadcasting batch dimensions."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw), axis=-1,
    )


def _unit(q: NDArray[np.float32]) -> NDArray[np.float32]:
    # Scale first so finite float32 inputs cannot overflow the squared norm.
    scale = np.max(np.abs(q), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("quaternion norm must be nonzero")
    scaled = q / scale
    return scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)


def _quaternions(value: ArrayLike, samples: int | None, name: str) -> NDArray[np.float32]:
    array = np.asarray(value)
    if (array.dtype != np.float32 or array.ndim != 3
            or array.shape[0] < 1 or array.shape[-1] != 4
            or (array.shape[1] < 2 if samples is None else array.shape[1] != samples)):
        length = "T" if samples is None else str(samples)
        raise ValueError(f"{name} must be float32 [B,{length},4] with B >= 1 and T >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return _unit(array)


def _adjacent(q: NDArray[np.float32]) -> NDArray[np.float32]:
    relative = _multiply(q[:, :-1] * _CONJUGATE, q[:, 1:])
    sign = np.where(relative[..., :1] < 0, _F32(-1), _F32(1))
    # At an exact half-turn, choose the first nonzero vector component positive.
    # This removes the remaining q/-q ambiguity without changing other arcs.
    first = np.argmax(relative[..., 1:] != 0, axis=-1)
    first_value = np.take_along_axis(relative[..., 1:], first[..., None], axis=-1)
    sign = np.where((relative[..., :1] == 0) & (first_value < 0), _F32(-1), sign)
    return _F32(2) * sign * relative[..., 1:]


def adjacent_features(context: ArrayLike) -> NDArray[np.float32]:
    """Return unscaled ``[B,T-1,3]`` increments for train-only axis statistics.

    Inputs must be finite float32 ``[B,T,4]`` with T >= 2. This permits fitting
    statistics over whole training segments without overlapping-window bias.
    Nonzero quaternions are
    normalized on a copy; caller buffers and their hemisphere signs are intact.
    This small-angle representation is not the exact SO(3) logarithm.
    """
    return _adjacent(_quaternions(context, None, "context"))


def _axis_statistic(value: ArrayLike | None, default: float, name: str) -> NDArray:
    result = np.full(3, default, dtype=np.float32) if value is None else np.array(
        value, dtype=np.float32, copy=True,
    )
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-axis vector")
    return result


class RelativeMLP:
    """Stateless quaternion predictor with caller-supplied train-only scaling.

    W1 uses He-normal initialization (std=sqrt(2/1197)); W2 uses normal std=.001
    so initial predictions are close to the last observation. Biases are zero.
    A local Generator makes initialization repeatable without global RNG state.
    Checkpoints include Adam moments and step for exact training continuation.
    """

    parameter_count = 80572

    def __init__(
        self, seed: int = 20261026, input_mean: ArrayLike | None = None,
        input_std: ArrayLike | None = None,
    ) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.input_mean = _axis_statistic(input_mean, 0.0, "input_mean")
        std = _axis_statistic(input_std, 1.0, "input_std")
        if np.any(std < 0):
            raise ValueError("input_std must be nonnegative")
        self.input_std = np.maximum(std, _STD_FLOOR)
        self.w1 = (rng.normal(size=(1197, 64)) * np.sqrt(2.0 / 1197)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.w2 = (rng.normal(size=(64, 60)) * .001).astype(np.float32)
        self.b2 = np.zeros(60, dtype=np.float32)
        self._m = {name: np.zeros(shape, dtype=np.float32)
                   for name, shape in _PARAMETER_SHAPES.items()}
        self._v = {name: np.zeros(shape, dtype=np.float32)
                   for name, shape in _PARAMETER_SHAPES.items()}
        self._step = 0

    def _forward(self, context: ArrayLike) -> tuple[NDArray, tuple]:
        q = _quaternions(context, 400, "context")
        f = ((_adjacent(q) - self.input_mean) / self.input_std).reshape(len(q), 1197)
        z = f @ self.w1 + self.b1
        h = np.maximum(z, _F32(0))
        r = (h @ self.w2 + self.b2).reshape(len(q), 20, 3)
        raw_local = np.concatenate(
            (np.ones((len(q), 20, 1), dtype=np.float32), _F32(.5) * r), axis=-1,
        )
        local_norm = np.linalg.norm(raw_local, axis=-1, keepdims=True)
        local = raw_local / local_norm
        anchor = q[:, -1:, :]
        raw_world = _multiply(anchor, local)
        world_norm = np.linalg.norm(raw_world, axis=-1, keepdims=True)
        prediction = raw_world / world_norm
        if not np.all(np.isfinite(prediction)):
            raise ValueError("model produced nonfinite predictions")
        return prediction, (f, z, h, anchor, local, local_norm, world_norm)

    def predict(self, context: ArrayLike) -> NDArray[np.float32]:
        """Predict unit wxyz rotations, accepting only float32 ``[B,400,4]``."""
        return self._forward(context)[0]

    def loss_and_gradients(
        self, context: ArrayLike, target: ArrayLike,
    ) -> tuple[float, dict[str, NDArray[np.float32]]]:
        """Return pre-update MSE and analytic gradients without changing state.

        The target hemisphere is selected per sample/output by prediction dot
        target. The loss is nondifferentiable at exactly zero dot; that tie uses
        the supplied target sign. Gradients include both normalization stages
        and the Hamilton product, treating context/target as fixed observations.
        """
        target_unit = _quaternions(target, 20, "target")
        prediction, cache = self._forward(context)
        if target_unit.shape != prediction.shape:
            raise ValueError("context and target batch sizes must match")
        f, z, h, anchor, local, local_norm, world_norm = cache
        aligned = np.where(
            np.sum(prediction * target_unit, axis=-1, keepdims=True) < 0,
            -target_unit, target_unit,
        )
        diff = prediction - aligned
        loss = float(np.mean(np.square(diff, dtype=np.float64)))
        gradient = _F32(2.0 / diff.size) * diff
        # For u=x/||x||: J^T g = (g-u*(u.g))/||x||.
        gradient = (gradient - prediction * np.sum(
            prediction * gradient, axis=-1, keepdims=True,
        )) / world_norm
        # The transpose of left multiplication by q is multiplication by conj(q).
        gradient = _multiply(anchor * _CONJUGATE, gradient)
        gradient = (gradient - local * np.sum(
            local * gradient, axis=-1, keepdims=True,
        )) / local_norm
        dr = (_F32(.5) * gradient[..., 1:]).reshape(len(prediction), 60)
        dz = (dr @ self.w2.T) * (z > 0)
        gradients = {"w1": f.T @ dz, "b1": dz.sum(axis=0),
                     "w2": h.T @ dr, "b2": dr.sum(axis=0)}
        if not np.isfinite(loss) or any(not np.all(np.isfinite(g)) for g in gradients.values()):
            raise ValueError("model produced nonfinite loss or gradients")
        return loss, gradients

    def train_step(
        self, context: ArrayLike, target: ArrayLike, learning_rate: float = .001,
    ) -> float:
        """Take one offline Adam step (betas=.9,.999, epsilon=1e-8).

        Returns the loss before the update. No scaler or observation buffer is
        modified, and ``predict`` never advances optimizer state.
        """
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        loss, gradients = self.loss_and_gradients(context, target)
        self._step += 1
        beta1, beta2 = _F32(.9), _F32(.999)
        correction1 = _F32(1.0 - .9 ** self._step)
        correction2 = _F32(1.0 - .999 ** self._step)
        for name, gradient in gradients.items():
            moment, variance = self._m[name], self._v[name]
            moment *= beta1
            moment += _F32(.1) * gradient
            variance *= beta2
            variance += _F32(.001) * gradient * gradient
            update = (moment / correction1) / (np.sqrt(variance / correction2) + _F32(1e-8))
            getattr(self, name)[:] -= _F32(learning_rate) * update
        return loss

    def save(self, path: str | Path) -> None:
        """Write weights/scaler/Adam state only; never serialize observations."""
        state = {name: getattr(self, name) for name in _PARAMETER_SHAPES}
        state.update({f"adam_m_{name}": value for name, value in self._m.items()})
        state.update({f"adam_v_{name}": value for name, value in self._v.items()})
        state.update(input_mean=self.input_mean, input_std=self.input_std,
                     format_version=np.array(1, dtype=np.int64),
                     adam_step=np.array(self._step, dtype=np.int64),
                     seed=np.array(self.seed, dtype=np.int64))
        # A file handle preserves the exact requested name, including no suffix.
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **state)

    @classmethod
    def load(cls, path: str | Path) -> RelativeMLP:
        """Load a versioned numeric checkpoint; pickle is never enabled."""
        with np.load(path, allow_pickle=False) as archive:
            expected = {"input_mean", "input_std", "format_version", "adam_step", "seed"}
            expected.update(_PARAMETER_SHAPES)
            expected.update(f"adam_{part}_{name}" for part in ("m", "v")
                            for name in _PARAMETER_SHAPES)
            if set(archive.files) != expected:
                raise ValueError("checkpoint fields do not match RelativeMLP format")
            integers = {}
            for name in ("format_version", "adam_step", "seed"):
                value = archive[name]
                if value.shape != () or value.dtype.kind not in "iu":
                    raise ValueError(f"checkpoint {name} must be an integer scalar")
                integers[name] = int(value)
            if integers["format_version"] != 1 or integers["adam_step"] < 0:
                raise ValueError("unsupported checkpoint version or invalid Adam step")
            arrays = {}
            shapes = dict(_PARAMETER_SHAPES, input_mean=(3,), input_std=(3,))
            for part in ("m", "v"):
                shapes.update({f"adam_{part}_{name}": shape
                               for name, shape in _PARAMETER_SHAPES.items()})
            for name, shape in shapes.items():
                value = archive[name]
                if value.shape != shape or value.dtype != np.float32 or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid checkpoint array {name}")
                arrays[name] = value.copy()
            if np.any(arrays["input_std"] < _STD_FLOOR) or any(
                np.any(arrays[f"adam_v_{name}"] < 0) for name in _PARAMETER_SHAPES
            ):
                raise ValueError("checkpoint has invalid scaler or Adam variance")
        model = cls(integers["seed"], arrays["input_mean"], arrays["input_std"])
        for name in _PARAMETER_SHAPES:
            getattr(model, name)[:] = arrays[name]
            model._m[name][:] = arrays[f"adam_m_{name}"]
            model._v[name][:] = arrays[f"adam_v_{name}"]
        model._step = integers["adam_step"]
        return model

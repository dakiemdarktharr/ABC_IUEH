"""Self-contained, stateless float32 400 -> 20 selector inference.

Only construction, the explicit offline reference, and ONNX inspection use
Python/NumPy host arithmetic. ``forward`` is tensor arithmetic on the input's
device, with constant gather indices and no recurrent or data-dependent loop.
The float64 experiment is the numerical reference, not a bit-identity claim.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evidence import EVIDENCE_VARIANTS, build_segment_features
from .selector import (
    DENSE_MACS, PARAMETERS, VARIANTS, SelectorMLP, normalise_features,
    predict_blended_quaternions,
)


_STATE_SHAPES = {"hidden.weight": (16, 10), "hidden.bias": (16,),
                 "output.weight": (4, 16), "output.bias": (4,)}
_FORBIDDEN_OPS = {"Loop", "If", "Scan", "NonZero", "TopK"}


def _norm(vector):
    return torch.sqrt(torch.sum(vector * vector, dim=-1, keepdim=True))


def _unit(quaternion):
    # The graph contract supplies finite unit inputs. Scaling also avoids
    # overflow during normalization of finite, moderately nonunit intermediates.
    scale = torch.clamp(torch.amax(torch.abs(quaternion), dim=-1, keepdim=True), min=1e-30)
    scaled = quaternion / scale
    return scaled / torch.clamp(_norm(scaled), min=1e-30)


def _multiply(left, right):
    aw, ax, ay, az = (left[..., i:i + 1] for i in range(4))
    bw, bx, by, bz = (right[..., i:i + 1] for i in range(4))
    return torch.cat((
        aw * bw - (ax * bx + ay * by + az * bz),
        (aw * bx + ax * bw) + (ay * bz - az * by),
        (aw * by + ay * bw) + (az * bx - ax * bz),
        (aw * bz + az * bw) + (ax * by - ay * bx),
    ), dim=-1)


def _conjugate(quaternion):
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _log_so3(quaternion):
    quaternion = _unit(quaternion)
    scalar, vector = quaternion[..., :1], quaternion[..., 1:]
    # An exact half turn uses the first nonzero vector component's sign. This
    # expresses the reference's tie convention without ArgMax or NonZero.
    first = torch.where(vector[..., :1] != 0, vector[..., :1],
                        torch.where(vector[..., 1:2] != 0, vector[..., 1:2], vector[..., 2:3]))
    negative = (scalar < 0) | ((scalar == 0) & (first < 0))
    signed = quaternion * torch.where(negative, -1.0, 1.0)
    sine = _norm(signed[..., 1:])
    # Canonical scalar is nonnegative. Abs additionally turns -0 into +0:
    # legacy ONNX atan2 lowering uses y/x and otherwise mishandles signed zero.
    angle = 2.0 * torch.atan2(sine, torch.abs(signed[..., :1]))
    factor = torch.where(sine > 0, angle / torch.clamp(sine, min=1e-30), 2.0)
    return signed[..., 1:] * factor


def _exp_so3(vector):
    angle = _norm(vector)
    half_angle = angle * 0.5
    factor = torch.where(angle > 0, torch.sin(half_angle) / torch.clamp(angle, min=1e-30), 0.5)
    return _unit(torch.cat((torch.cos(half_angle), vector * factor), dim=-1))


def _geodesic_deg(left, right):
    delta = _multiply(_unit(left), _conjugate(_unit(right)))
    return 114.59155902616465 * torch.atan2(_norm(delta[..., 1:]), torch.abs(delta[..., :1]))


def _validate_variant(variant, evidence_length):
    if variant not in VARIANTS:
        raise ValueError(f"unknown selector feature variant: {variant!r}")
    if (isinstance(evidence_length, bool) or not isinstance(evidence_length, Integral)
            or not 1 <= evidence_length <= 340):
        raise ValueError("evidence_length must be an integer1..340 inside400 observations")
    return int(evidence_length)


def _validate_context(context):
    # Shape/type are interface validation only. Input values must satisfy the
    # published finite, nonzero, normalized wxyz precondition; no host sync or
    # data-to-Python conversion belongs in the exported forward path.
    if not torch.jit.is_tracing():
        if context.shape != (1, 400, 4) or context.dtype != torch.float32:
            raise ValueError("context must be float32 [1,400,4]")


class FullSelectorInference(nn.Module):
    """Frozen 244-parameter selector plus evidence, experts and SO(3) output.

    Pass one checkpoint's extracted ``state_dict`` and the matching variant's
    already fitted train scaler. Construction copies tensors and statistics;
    it never loads a file or fits a statistic. R128/R256 h1/h20 use the same API.
    All inference arithmetic, including scaler constants, is float32.
    """

    def __init__(self, selector_state_dict, scaler, variant="h20_a0", evidence_length=80):
        super().__init__()
        self.evidence_length = _validate_variant(variant, evidence_length)
        self.variant = variant
        if not isinstance(selector_state_dict, Mapping) or set(selector_state_dict) != set(_STATE_SHAPES):
            raise ValueError("selector_state_dict must contain the four244-parameter tensors")
        copied = {}
        for name, shape in _STATE_SHAPES.items():
            value = selector_state_dict[name]
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all())):
                raise ValueError(f"invalid finite float32 selector tensor: {name}")
            copied[name] = value.detach().cpu().clone()
        # Reuse the experiment's validation contract without fitting anything.
        normalise_features(np.zeros((1, 10), dtype=np.float32), scaler)
        self.selector = SelectorMLP()
        self.selector.load_state_dict(copied, strict=True)
        self.selector.requires_grad_(False)
        self.selector.eval()
        self.register_buffer("feature_mean", torch.tensor(scaler["mean"], dtype=torch.float32).reshape(1, 10))
        self.register_buffer("feature_scale", torch.tensor(scaler["scale"], dtype=torch.float32).reshape(1, 10))
        if not bool(torch.isfinite(self.feature_mean).all() & torch.isfinite(self.feature_scale).all()):
            raise ValueError("train scaler constants must be representable in float32")
        self.register_buffer("bandwidth", torch.tensor([1.0, 5.0, 20.0], dtype=torch.float32).reshape(1, 1, 3, 1))
        self.register_buffer("pair_left", torch.tensor([0, 0, 1], dtype=torch.int64))
        self.register_buffer("pair_right", torch.tensor([1, 2, 2], dtype=torch.int64))
        self.register_buffer("output_horizons", torch.arange(1, 21, dtype=torch.float32).reshape(1, 20, 1))
        if variant == "motion_only":
            history_origins = torch.empty(0, dtype=torch.int64)
            horizons = torch.empty(0, dtype=torch.int64)
        else:
            horizon, age = (20, 0) if variant == "complete_horizon" else EVIDENCE_VARIANTS[variant]
            history_origins = 399 - age - horizon - torch.arange(self.evidence_length, dtype=torch.int64)
            horizons = torch.arange(1, 21, dtype=torch.int64) if variant == "complete_horizon" else torch.tensor([horizon])
        origins = torch.cat((history_origins, torch.tensor([399], dtype=torch.int64)))
        self._velocity_rows = len(origins)
        self._replay_horizons = len(horizons)
        self.register_buffer("velocity_origins", origins)
        self.register_buffer("velocity_past", (origins[:, None] - torch.tensor([1, 5, 20])).reshape(-1))
        self.register_buffer("replay_horizons", horizons.to(torch.float32).reshape(1, 1, -1, 1, 1))
        self.register_buffer("replay_targets", (history_origins[:, None] + horizons[None, :]).reshape(-1))
        minimum = min(int(self.velocity_past.min()), 299 if variant == "motion_only" else 399)
        self.minimum_context_index = minimum

    def _velocities(self, unit):
        current = torch.index_select(unit, 1, self.velocity_origins)
        past = torch.index_select(unit, 1, self.velocity_past).reshape(1, self._velocity_rows, 3, 4)
        relative = _multiply(current[:, :, None, :], _conjugate(past))
        moving = _log_so3(relative) / self.bandwidth
        return torch.cat((torch.zeros_like(moving[:, :, :1, :]), moving), dim=2), current

    def _features_and_velocity(self, context):
        unit = _unit(context)
        velocities, origins = self._velocities(unit)
        current_velocity = velocities[:, -1]
        moving = current_velocity[:, 1:]
        speed = _norm(moving)[..., 0]
        left = torch.index_select(moving, 1, self.pair_left)
        right = torch.index_select(moving, 1, self.pair_right)
        denominator = torch.index_select(speed, 1, self.pair_left) * torch.index_select(speed, 1, self.pair_right)
        cosine = torch.clamp(torch.sum(left * right, dim=-1) / torch.clamp(denominator, min=1e-12), -1.0, 1.0)
        common = torch.cat((speed, cosine), dim=1)
        if self.variant == "motion_only":
            differences = _norm(left - right)[..., 0]
            relative100 = _multiply(unit[:, 399], _conjugate(unit[:, 299]))
            net_speed = _norm(_log_so3(relative100)) / 100.0
            raw = torch.cat((common, differences, net_speed), dim=1)
        else:
            rotation = _exp_so3(velocities[:, :-1, None, :, :] * self.replay_horizons)
            replay = _unit(_multiply(rotation, origins[:, :-1, None, None, :]))
            target = torch.index_select(unit, 1, self.replay_targets).reshape(
                1, self.evidence_length, self._replay_horizons, 1, 4)
            errors = _geodesic_deg(replay, target)[..., 0]
            score = torch.mean(errors, dim=(1, 2))
            raw = torch.cat((score, common), dim=1)
        return raw, current_velocity, unit[:, 399]

    def raw_features(self, context):
        """Diagnostic raw10 columns, computed by the same float32 graph path."""
        _validate_context(context)
        return self._features_and_velocity(context)[0]

    def selector_weights(self, context):
        """Diagnostic simplex weights; no inference-time updates."""
        raw = self.raw_features(context)
        return self._weights(raw)

    def _weights(self, raw):
        normalized = (raw - self.feature_mean) / self.feature_scale
        # The existing selector's eager shape guard is data-independent but
        # emits a tracing warning. Its two frozen layers are the same model.
        return torch.softmax(self.selector.output(torch.relu(self.selector.hidden(normalized))), dim=1)

    def forward(self, context):
        _validate_context(context)
        raw, velocities, current = self._features_and_velocity(context)
        weights = self._weights(raw)
        velocity = torch.sum(velocities * weights[:, :, None], dim=1)
        rotation = _exp_so3(self.output_horizons * velocity[:, None, :])
        return _unit(_multiply(rotation, current[:, None, :]))

    def workload(self):
        """Algorithm dimensions, without calling nonlinear work 'dense MACs'."""
        replay = 0 if self.variant == "motion_only" else 4 * self.evidence_length * self._replay_horizons
        return {
            "parameters": PARAMETERS, "selector_dense_macs": DENSE_MACS,
            "mac_definition": "one dense weight multiplication and accumulation; biases/nonlinear/geometry excluded",
            "stateless_context_samples": 400, "earliest_context_index_used": self.minimum_context_index,
            "velocity_relative_quaternions": 3 * self._velocity_rows,
            "replayed_expert_quaternions": replay, "replayed_geodesic_errors": replay,
            "output_quaternions": 20, "replay_horizons": self._replay_horizons,
        }


def reference_from_segment(segment, selector_state_dict, scaler, *, variant="h20_a0", evidence_length=80, origin=399):
    """Offline float64 experiment reference for an allowed or synthetic segment.

    ``segment`` needs the origin's20 targets only because the existing builder
    also returns labels. They never enter features, weights or predictions.
    This helper does not open any file and does not authorize any data access.
    """
    evidence_length = _validate_variant(variant, evidence_length)
    if isinstance(origin, bool) or not isinstance(origin, Integral) or not 399 <= origin < len(segment) - 20:
        raise ValueError("reference origin must have400 context samples and20 targets")
    built = build_segment_features(segment, evidence_length=evidence_length,
                                   include_complete_horizon=variant == "complete_horizon")
    row = int(origin) - 399
    raw = built["features"][variant][row:row + 1]
    normalized = normalise_features(raw, scaler)
    model = SelectorMLP()
    model.load_state_dict(selector_state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        weights = model(torch.from_numpy(normalized)).numpy()
    prediction = predict_blended_quaternions(weights, built["velocities"][row:row + 1], built["current_q"][row:row + 1])
    return {"raw_features": raw, "normalized_features": normalized, "weights": weights, "prediction": prediction}


def inspect_onnx(path):
    """Check the actual graph and count static operator-output elements.

    Counts describe ONNX mathematical work before provider optimizations, not
    latency, hardware instructions, or a total full-pipeline MAC estimate.
    Reduction additions are input-elements minus output-elements; ReduceMean
    also has one division/output. Dense MACs stay a separate explicit figure.
    """
    import onnx

    graph_model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph_model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(graph_model, check_type=True, strict_mode=True)
    graph = inferred.graph
    values = [*graph.input, *graph.output, *graph.value_info]
    shapes, types, sequences = {}, Counter(), []
    for value in values:
        if value.type.HasField("sequence_type"):
            sequences.append(value.name)
        if value.type.HasField("tensor_type"):
            tensor = value.type.tensor_type
            types[onnx.TensorProto.DataType.Name(tensor.elem_type)] += 1
            shapes[value.name] = [int(d.dim_value) if d.HasField("dim_value") else None for d in tensor.shape.dim]
    for initializer in graph.initializer:
        types[onnx.TensorProto.DataType.Name(initializer.data_type)] += 1
        shapes[initializer.name] = list(initializer.dims)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                types[onnx.TensorProto.DataType.Name(attribute.t.data_type)] += 1
    operators = Counter(node.op_type for node in graph.node)
    domains = sorted(set(node.domain for node in graph.node))
    if _FORBIDDEN_OPS.intersection(operators) or any(domain not in ("", "ai.onnx") for domain in domains):
        raise ValueError("export contains a forbidden control-flow/selection/custom operator")
    other_floats = [kind for kind in types if kind == "DOUBLE" or
                    ((kind.startswith("FLOAT") or kind.startswith("BFLOAT")) and kind != "FLOAT")]
    if other_floats or sequences:
        raise ValueError("export contains non-float32 floating or sequence types")
    signature = {"inputs": {v.name: shapes[v.name] for v in graph.input},
                 "outputs": {v.name: shapes[v.name] for v in graph.output}}
    if signature != {"inputs": {"context": [1, 400, 4]}, "outputs": {"prediction": [1, 20, 4]}}:
        raise ValueError("export does not preserve the static context/prediction interface")
    for value in (*graph.input, *graph.output):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("export input/output must be float32")

    def elements(name):
        shape = shapes.get(name)
        return None if shape is None or any(d is None for d in shape) else int(np.prod(shape, dtype=np.int64))

    outputs, unknown = Counter(), []
    reduction_additions, reduction_divisions, reduction_max_comparisons = 0, 0, 0
    for node in graph.node:
        counts = [elements(name) for name in node.output]
        if any(count is None for count in counts):
            unknown.append(node.name or node.op_type)
        else:
            outputs[node.op_type] += sum(counts)
        if node.op_type in ("ReduceSum", "ReduceMean"):
            n_in, n_out = elements(node.input[0]), elements(node.output[0])
            if n_in is not None and n_out is not None:
                reduction_additions += n_in - n_out
                reduction_divisions += n_out if node.op_type == "ReduceMean" else 0
        if node.op_type == "ReduceMax":
            n_in, n_out = elements(node.input[0]), elements(node.output[0])
            if n_in is not None and n_out is not None:
                reduction_max_comparisons += n_in - n_out
    scalar_ops = ("Add", "Sub", "Mul", "Div", "Sqrt", "Sin", "Cos", "Atan", "Abs", "Neg", "Relu")
    return {
        "opset_imports": {item.domain or "ai.onnx": item.version for item in graph_model.opset_import},
        "signature": signature, "operators": dict(sorted(operators.items())),
        "tensor_types": dict(sorted(types.items())), "domains": domains,
        "node_count": len(graph.node), "file_bytes": Path(path).stat().st_size,
        "operator_output_elements": dict(sorted(outputs.items())),
        "elementwise_scalar_evaluations": {name: outputs[name] for name in scalar_ops if name in outputs},
        "reduction_scalar_additions": reduction_additions, "reduction_scalar_divisions": reduction_divisions,
        "reduction_max_comparisons": reduction_max_comparisons,
        "gemm_bias_additions": outputs["Gemm"],
        "unknown_output_shape_nodes": unknown, "selector_dense_macs": DENSE_MACS,
        "cost_scope": "static ONNX operator output elements plus reduction arithmetic; not total MACs or latency",
    }


def export_full_selector_onnx(model, path, *, example_context=None):
    """Legacy-export a CPU model at opset17 and inspect the resulting graph.

    Existing artifacts are never overwritten. No dynamic axes or external
    weight files are requested. Export does not measure target-device latency.
    """
    import onnx  # Fail before writing a partial export if inspection is absent.

    del onnx
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not isinstance(model, FullSelectorInference) or next(model.parameters()).device.type != "cpu":
        raise ValueError("export requires a FullSelectorInference model on CPU")
    if example_context is None:
        example_context = torch.zeros((1, 400, 4), dtype=torch.float32)
        example_context[..., 0] = 1.0
    _validate_context(example_context)
    if example_context.device.type != "cpu":
        raise ValueError("export example must be on CPU")
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, (example_context,), str(path), export_params=True,
                          opset_version=17, do_constant_folding=True, dynamo=False,
                          external_data=False, input_names=["context"], output_names=["prediction"])
    inventory = inspect_onnx(path)
    inventory["algorithm_workload"] = model.workload()
    return inventory

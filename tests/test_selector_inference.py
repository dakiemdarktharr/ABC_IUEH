"""Synthetic full-graph checks; no recording, cache or checkpoint reads."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abc_iueh.geometry import exp_so3, geodesic_deg, multiply_wxyz, normalize_wxyz
from abc_iueh.selector import SelectorMLP, VARIANTS
from abc_iueh.selector_inference import (
    FullSelectorInference, _exp_so3, _log_so3, export_full_selector_onnx,
    reference_from_segment,
)


def synthetic_segment(length=450, seed=118, scale=.018):
    rng = np.random.default_rng(seed)
    q = np.empty((length, 4))
    q[0] = normalize_wxyz(rng.normal(size=4))
    for i in range(1, length):
        q[i] = normalize_wxyz(multiply_wxyz(exp_so3(rng.normal(0, scale, 3)), q[i - 1]))
    return q.astype(np.float32)


@pytest.fixture(scope="module")
def segment():
    return synthetic_segment()


@pytest.fixture(scope="module")
def state():
    result = {name: value.detach().clone() for name, value in SelectorMLP().state_dict().items()}
    # A stronger deterministic synthetic selector exposes feature/scaler drift;
    # these tensors are generated here, not a real-data fit or checkpoint.
    result["hidden.weight"] *= 10
    result["output.weight"] *= 10
    result["output.bias"][:] = torch.tensor([.1, -.2, .3, -.1])
    return result


@pytest.fixture(scope="module")
def scaler():
    return {"fit_partition": "train", "ddof": 0, "epsilon": 1e-8,
            "mean": [4., 4., 4., 4., .02, .02, .02, .1, .1, .1],
            "scale": [5., 5., 5., 5., .05, .05, .05, 1., 1., 1.]}


@pytest.mark.parametrize("variant", VARIANTS)
def test_full_float32_path_matches_float64_offline_reference(segment, state, scaler, variant):
    model = FullSelectorInference(state, scaler, variant)
    reference = reference_from_segment(segment, state, scaler, variant=variant)
    context = torch.from_numpy(segment[:400][None])
    with torch.no_grad():
        raw = model.raw_features(context).numpy()
        weights = model.selector_weights(context).numpy()
        prediction = model(context).numpy()
    assert raw.shape == (1, 10) and raw.dtype == np.float32
    np.testing.assert_allclose(raw, reference["raw_features"], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(weights, reference["weights"], atol=5e-6, rtol=5e-5)
    assert prediction.shape == (1, 20, 4) and prediction.dtype == np.float32
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(np.linalg.norm(prediction, axis=-1), 1, atol=2e-7, rtol=0)
    # A numerical portability gate fixed before real-data export evaluation.
    assert np.max(geodesic_deg(prediction, reference["prediction"])) <= .02
    assert sum(p.numel() for p in model.parameters()) == 244
    assert all(not p.requires_grad for p in model.parameters())
    assert all(value.dtype != torch.float64 for value in model.state_dict().values())


@pytest.mark.parametrize("evidence_length", [128, 256, 340])
@pytest.mark.parametrize("variant", ["h1_a0", "h20_a0"])
def test_prospective_longer_histories_stay_inside_context(segment, state, scaler, variant, evidence_length):
    model = FullSelectorInference(state, scaler, variant, evidence_length)
    reference = reference_from_segment(segment, state, scaler, variant=variant, evidence_length=evidence_length)
    context = torch.from_numpy(segment[:400][None])
    with torch.no_grad():
        np.testing.assert_allclose(model.raw_features(context).numpy(), reference["raw_features"], atol=5e-5, rtol=5e-5)
        assert np.max(geodesic_deg(model(context).numpy(), reference["prediction"])) <= .02
    assert model.minimum_context_index >= 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_stateless_prefix_and_future_suffix_causality(segment, state, scaler, variant):
    model = FullSelectorInference(state, scaler, variant)
    original = torch.from_numpy(segment[:400][None].copy())
    changed_prefix = original.clone()
    stop = model.minimum_context_index
    changed_prefix[:, :stop] = torch.from_numpy(synthetic_segment(stop, seed=602))[None]
    with torch.no_grad():
        before = model(original)
        assert torch.equal(before, model(changed_prefix))
        # A call on a different stream cannot leave Python/session state.
        model(torch.from_numpy(synthetic_segment(400, seed=77))[None])
        assert torch.equal(before, model(original))
    changed_suffix = segment.copy()
    changed_suffix[400:] = synthetic_segment(len(segment) - 400, seed=774)
    reference = reference_from_segment(segment, state, scaler, variant=variant)
    perturbed = reference_from_segment(changed_suffix, state, scaler, variant=variant)
    np.testing.assert_array_equal(reference["raw_features"], perturbed["raw_features"])
    np.testing.assert_array_equal(reference["prediction"], perturbed["prediction"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_sign_invariance_and_world_frame_equivariance(segment, state, scaler, variant):
    model = FullSelectorInference(state, scaler, variant)
    q = segment[:400].copy()
    signed = q.copy()
    signed[::3] *= -1
    world = exp_so3([.8, -.4, .3])
    rotated = normalize_wxyz(multiply_wxyz(world, q)).astype(np.float32)
    with torch.no_grad():
        raw = model.raw_features(torch.from_numpy(q[None])).numpy()
        signed_raw = model.raw_features(torch.from_numpy(signed[None])).numpy()
        rotated_raw = model.raw_features(torch.from_numpy(rotated[None])).numpy()
        prediction = model(torch.from_numpy(q[None])).numpy()
        signed_prediction = model(torch.from_numpy(signed[None])).numpy()
        rotated_prediction = model(torch.from_numpy(rotated[None])).numpy()
    np.testing.assert_array_equal(raw, signed_raw)
    assert np.max(geodesic_deg(prediction, signed_prediction)) < 1e-8
    np.testing.assert_allclose(raw, rotated_raw, atol=8e-5, rtol=1e-4)
    expected_rotated = multiply_wxyz(world, prediction)
    assert np.max(geodesic_deg(rotated_prediction, expected_rotated)) < .002


@pytest.mark.parametrize("scale", [0., 1e-10, 1e-7, 1.8])
@pytest.mark.parametrize("variant", ["h20_a0", "complete_horizon", "motion_only"])
def test_stationary_tiny_and_wide_rotations_remain_finite(state, scaler, scale, variant):
    q = synthetic_segment(400, seed=67, scale=scale)
    model = FullSelectorInference(state, scaler, variant)
    with torch.no_grad():
        context = torch.from_numpy(q[None])
        prediction = model(context).numpy()
        features = model.raw_features(context).numpy()
    assert np.isfinite(features).all() and np.isfinite(prediction).all()
    np.testing.assert_allclose(np.linalg.norm(prediction, axis=-1), 1, atol=2e-7, rtol=0)
    if scale == 0:
        np.testing.assert_allclose(features, 0, atol=1e-7, rtol=0)
        assert np.max(geodesic_deg(prediction, q[-1])) < 1e-5


def test_exact_half_turn_tie_and_zero_log_exp_are_deterministic():
    half_turns = torch.tensor([[0., -1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., -1.]])
    torch.testing.assert_close(_log_so3(half_turns), _log_so3(-half_turns), rtol=0, atol=0)
    np.testing.assert_allclose(_log_so3(half_turns).numpy(), np.eye(3) * np.pi, atol=2e-7, rtol=0)
    zero = torch.zeros((5, 3), dtype=torch.float32)
    expected = torch.tensor([[1., 0., 0., 0.]]).expand(5, 4)
    torch.testing.assert_close(_exp_so3(zero), expected, rtol=0, atol=0)
    torch.testing.assert_close(_log_so3(expected), zero, rtol=0, atol=0)


def test_maximum_delayed_history_reaches_index0_and_default_dtype_cannot_add_float64(segment, state, scaler):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = FullSelectorInference(state, scaler, "h20_a20", 340)
    finally:
        torch.set_default_dtype(previous)
    assert model.minimum_context_index == 0
    assert all(value.dtype != torch.float64 for value in model.state_dict().values())
    reference = reference_from_segment(segment, state, scaler, variant="h20_a20", evidence_length=340)
    with torch.no_grad():
        actual = model(torch.from_numpy(segment[:400][None])).numpy()
    assert actual.dtype == np.float32
    assert np.max(geodesic_deg(actual, reference["prediction"])) <= .02


@pytest.mark.parametrize("bad_length", [0, 341, 1.5, True])
def test_invalid_history_is_rejected(state, scaler, bad_length):
    with pytest.raises(ValueError, match="evidence_length"):
        FullSelectorInference(state, scaler, evidence_length=bad_length)


def test_invalid_constructor_and_interface_are_rejected(state, scaler):
    with pytest.raises(ValueError, match="variant"):
        FullSelectorInference(state, scaler, variant="future_labels")
    for bad in ({}, dict(state, **{"hidden.weight": torch.zeros((10, 16))}),
                dict(state, **{"hidden.weight": state["hidden.weight"].double()}),
                dict(state, **{"output.bias": torch.full((4,), float("nan"))})):
        with pytest.raises(ValueError, match="selector"):
            FullSelectorInference(bad, scaler)
    for bad in (dict(scaler, fit_partition="development"), dict(scaler, ddof=1),
                dict(scaler, mean=[float("nan")] * 10), dict(scaler, scale=[0.] * 10)):
        with pytest.raises(ValueError):
            FullSelectorInference(state, bad)
    model = FullSelectorInference(state, scaler)
    for context in (torch.ones((400, 4)), torch.ones((2, 400, 4)), torch.ones((1, 400, 4), dtype=torch.float64)):
        with pytest.raises(ValueError, match="context"):
            model(context)


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_static_onnx_inventory_and_cpu_runtime_parity(tmp_path, segment, state, scaler, variant):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = FullSelectorInference(state, scaler, variant)
    path = Path(tmp_path) / f"{variant}.onnx"
    context = segment[:400][None].copy()
    inventory = export_full_selector_onnx(model, path, example_context=torch.from_numpy(context))
    assert inventory["opset_imports"] == {"ai.onnx": 17}
    assert inventory["signature"] == {"inputs": {"context": [1, 400, 4]}, "outputs": {"prediction": [1, 20, 4]}}
    assert not {"Loop", "If", "Scan", "NonZero", "TopK", "ArgMax"}.intersection(inventory["operators"])
    assert set(inventory["tensor_types"]) <= {"FLOAT", "INT64", "BOOL"}
    assert inventory["unknown_output_shape_nodes"] == []
    assert inventory["operators"]["Gemm"] == 2 and inventory["operators"]["Softmax"] == 1
    assert inventory["selector_dense_macs"] == 224
    assert inventory["operator_output_elements"]["Mul"] > 224
    assert inventory["reduction_scalar_additions"] > 0
    assert inventory["algorithm_workload"]["replayed_geodesic_errors"] == (
        0 if variant == "motion_only" else 4 * 80 * (20 if variant == "complete_horizon" else 1))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    # A random context differs from the export example and prevents a trace
    # accidentally specialized to stationary orientations from passing.
    actual = session.run(["prediction"], {"context": context})[0]
    reference = reference_from_segment(segment, state, scaler, variant=variant)
    with torch.no_grad():
        eager = model(torch.from_numpy(context)).numpy()
    assert actual.dtype == np.float32 and actual.shape == (1, 20, 4)
    assert np.isfinite(actual).all()
    assert np.max(geodesic_deg(actual, eager)) < .002
    assert np.max(geodesic_deg(actual, reference["prediction"])) <= .02
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1, atol=3e-7, rtol=0)
    # Ninety cases over all six parametrized variants: ordinary motion, tiny
    # changes, stationarity and large noncommuting per-sample rotations. Keep
    # the same predeclared output-angle gate for every regime.
    for seed in (118, 228, 918):
        for scale in (0., 1e-10, 1e-7, .018, 1.8):
            alternative_segment = synthetic_segment(seed=seed, scale=scale)
            alternative = alternative_segment[:400][None].copy()
            output = session.run(["prediction"], {"context": alternative})[0]
            alternative_reference = reference_from_segment(alternative_segment, state, scaler, variant=variant)
            assert np.isfinite(output).all()
            assert np.max(geodesic_deg(output, alternative_reference["prediction"])) <= .02
            np.testing.assert_allclose(np.linalg.norm(output, axis=-1), 1, atol=3e-7, rtol=0)
    half_turn_segment = np.tile(np.array([[1, 0, 0, 0], [0, -1, 0, 0]], dtype=np.float32), (225, 1))
    half_turn_output = session.run(["prediction"], {"context": half_turn_segment[:400][None].copy()})[0]
    half_turn_reference = reference_from_segment(half_turn_segment, state, scaler, variant=variant)
    assert np.isfinite(half_turn_output).all()
    assert np.max(geodesic_deg(half_turn_output, half_turn_reference["prediction"])) <= .02
    with pytest.raises(FileExistsError):
        export_full_selector_onnx(model, path)

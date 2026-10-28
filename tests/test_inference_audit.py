"""Synthetic measurement-recorder checks; no human recordings are opened."""
import copy
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abc_iueh.geometry import exp_so3
from abc_iueh.inference_audit import (
    _FROZEN_CONFIG, _artifact_hashes, _latency_summary, _load_checkpoint,
    _evaluate_variant, _parity_summary, _prediction, collect_sampled_segments, run_inference_audit,
    sample_user_origins, validate_inference_config,
)
from abc_iueh.selector import SelectorMLP
from abc_iueh.sources import sha256_file


@pytest.fixture
def sampled_cache(tmp_path):
    path = tmp_path / "synthetic_origins.h5"
    with h5py.File(path, "w") as cache:
        for partition, user, session, count in (
                ("train", "long_private", "a", 80), ("train", "long_private", "b", 120),
                ("train", "short_private", "c", 30), ("development", "dev_private", "d", 45)):
            cache[f"{partition}/{user}/{session}/0/origin_index"] = np.arange(count) + 399
        partitions = {"train": ("long_private", "short_private"), "development": ("dev_private",),
                      "final": ("final_sentinel",)}
        yield cache, partitions


def test_balanced_sampling_keeps_26_per_user_despite_unequal_recording_lengths(sampled_cache):
    cache, partitions = sampled_cache
    samples, public, digest = sample_user_origins(cache, partitions)
    assert len(samples) == 78
    assert len(digest) == 64
    for user in ("long_private", "short_private", "dev_private"):
        selected = [sample for sample in samples if sample["user"] == user]
        assert len(selected) == 26
        assert len({(sample["session"], sample["origin_index"]) for sample in selected}) == 26
    assert public["train"] == {"users": 2, "contexts": 52, "contexts_per_user": 26, "population_windows": 230}
    assert public["development"]["contexts"] == 26
    assert sample_user_origins(cache, partitions) == (samples, public, digest)
    assert sample_user_origins(cache, partitions, seed=20261029)[2] != digest


def test_uniform_within_user_global_origin_mapping_does_not_overweight_short_segments(sampled_cache):
    cache, partitions = sampled_cache
    samples, _, _ = sample_user_origins(cache, partitions)
    selected = sorted(np.random.default_rng(20261028).choice(200, size=26, replace=False))
    expected = [("a", int(index) + 399) if index < 80 else ("b", int(index) - 80 + 399) for index in selected]
    actual = [(sample["session"], sample["origin_index"]) for sample in samples if sample["user"] == "long_private"]
    assert actual == expected


def test_sampling_rejects_insufficient_user_windows_and_wrong_cache_index(sampled_cache):
    cache, partitions = sampled_cache
    with pytest.raises(ValueError, match="fixed number of complete origins"):
        sample_user_origins(cache, partitions, contexts_per_user=31)
    # Every origin for this user will be invalid regardless of the random draw.
    origin = cache["train/long_private/a/0/origin_index"]
    origin[:] = origin[:] + 1
    with pytest.raises(ValueError, match="stride1 indexing"):
        sample_user_origins(cache, partitions)


def test_guarded_collection_never_requests_final_and_retains_context_target_alignment(sampled_cache):
    cache, partitions = sampled_cache
    samples, _, _ = sample_user_origins(cache, partitions)
    arrays = {}
    for partition in ("train", "development"):
        for user, session, count in (("long_private", "a", 80), ("long_private", "b", 120),
                                     ("short_private", "c", 30), ("dev_private", "d", 45)):
            if user in partitions[partition]:
                angles = np.arange(count + 419)[:, None] * np.array([[0., 0., .001]])
                arrays[(partition, user, session, "0")] = exp_so3(angles).astype(np.float32)

    class GuardedStage:
        def __init__(self):
            self.partitions = partitions
            self.calls = []

        def iter_segments(self, partition):
            assert partition != "final", "final value read sentinel"
            self.calls.append(partition)
            for key, values in arrays.items():
                if key[0] == partition:
                    yield *key[1:], values

    stage = GuardedStage()
    selected = collect_sampled_segments(stage, samples)
    assert stage.calls == ["train", "development"]
    for sample, segment in zip(samples, selected):
        q = arrays[(sample["partition"], sample["user"], sample["session"], sample["segment"])]
        origin = sample["origin_index"]
        assert segment.shape == (420, 4)
        np.testing.assert_array_equal(segment[:400], q[origin - 399:origin + 1])
        np.testing.assert_array_equal(segment[419], q[origin + 20])
    forbidden = [dict(samples[0], partition="final", user="final_sentinel")]
    with pytest.raises(PermissionError, match="forbidden user/partition"):
        collect_sampled_segments(stage, forbidden)


@pytest.mark.parametrize("field,value", [("seed", 20261027), ("final_access_allowed", True)])
def test_frozen_config_refuses_seed_or_holdout_changes(field, value):
    config = copy.deepcopy(_FROZEN_CONFIG)
    config[field] = value
    with pytest.raises(ValueError, match="frozen inference"):
        validate_inference_config(config)


@pytest.mark.parametrize("section,field,value", [
    ("numerical_gates", "max_sampled_real_trajectory_geodesic_drift_deg", .03),
    ("numerical_gates", "absolute_sampled_mean_endpoint20_error_change_deg", .02),
    ("parity_sampling", "contexts_per_user", 25), ("latency", "repetitions", 100),
    ("graph", "float64_allowed", True),
])
def test_frozen_config_does_not_rescue_a_failed_gate_or_reduce_measurement(section, field, value):
    config = copy.deepcopy(_FROZEN_CONFIG)
    config[section][field] = value
    with pytest.raises(ValueError, match="frozen inference"):
        validate_inference_config(config)
    assert validate_inference_config(copy.deepcopy(_FROZEN_CONFIG)) == _FROZEN_CONFIG


def _checkpoint(path, *, seed=20261026, state=None):
    value = {"state_dict": SelectorMLP().state_dict() if state is None else state,
             "architecture": [10, 16, 4], "parameters": 244, "dense_MACs": 224,
             "variant": "h20_a0", "seed": seed, "optimizer_steps": 2000,
             "checkpoint_selection": "fixed_final_step_no_development_selection",
             "loss": "expected_individual_expert_endpoint20_angular_loss", "bindings": {"safe_hash": "f" * 64}}
    torch.save(value, path)
    return value


def test_checkpoint_load_requires_saved_hash_first_seed_and_safe_tensor_only_loader(tmp_path, monkeypatch):
    path = tmp_path / "synthetic_selector.pt"
    value = _checkpoint(path)
    original = torch.load
    calls = []

    def guarded(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["weights_only"] is True
        assert kwargs["map_location"] == "cpu"
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    loaded = _load_checkpoint(path, "h20_a0", value["bindings"], sha256_file(path))
    assert set(loaded) == set(SelectorMLP().state_dict())
    assert len(calls) == 1
    with pytest.raises(ValueError, match="hash changed"):
        _load_checkpoint(path, "h20_a0", value["bindings"], "0" * 64)
    assert len(calls) == 1
    _checkpoint(path, seed=20261027)
    with pytest.raises(ValueError, match="fixed first-seed"):
        _load_checkpoint(path, "h20_a0", value["bindings"], sha256_file(path))


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "float64", "extra_key"])
def test_checkpoint_hash_alone_cannot_authorize_invalid_learned_state(tmp_path, kind):
    path = tmp_path / "synthetic_selector.pt"
    state = SelectorMLP().state_dict()
    if kind == "shape":
        state["hidden.weight"] = torch.zeros((15, 10))
    elif kind == "nonfinite":
        state["hidden.weight"][0, 0] = torch.nan
    elif kind == "float64":
        state["hidden.weight"] = state["hidden.weight"].double()
    else:
        state["unknown"] = torch.zeros(1)
    value = _checkpoint(path, state=state)
    with pytest.raises(ValueError):
        _load_checkpoint(path, "h20_a0", value["bindings"], sha256_file(path))


def test_receipt_hash_refuses_changed_artifact_and_path_traversal(tmp_path):
    path = tmp_path / "aggregate.json"
    path.write_text('{"synthetic":true}', encoding="utf-8")
    expected = {path.name: sha256_file(path)}
    assert _artifact_hashes(tmp_path, expected) == expected
    path.write_text('{"synthetic":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its saved receipt"):
        _artifact_hashes(tmp_path, expected)
    with pytest.raises(ValueError, match="artifact identity"):
        _artifact_hashes(tmp_path, {"../aggregate.json": "a" * 64})


def test_parity_checks_entire_trajectory_and_signed_endpoint_metric_change():
    reference = np.zeros((2, 20, 4), dtype=np.float32)
    reference[..., 0] = 1
    predicted = reference.copy()
    # Intermediate output diverges0.03deg, endpoint diverges0.02deg. Both gates
    # must fail even though all quaternions are finite and correctly shaped.
    predicted[0, 3] = exp_so3([0, 0, np.deg2rad(.03)]).astype(np.float32)
    predicted[:, -1] = exp_so3([0, 0, np.deg2rad(.02)]).astype(np.float32)
    targets = reference[:, -1]
    result = _parity_summary(predicted, reference, targets, _FROZEN_CONFIG)
    assert result["max_trajectory_geodesic_drift_deg"] == pytest.approx(.03, abs=1e-6)
    assert result["sampled_mean_endpoint20_error_change_deg"] == pytest.approx(.02, abs=1e-6)
    assert not any(result["numerical_gate_checks"].values())
    assert _FROZEN_CONFIG["numerical_gates"]["max_sampled_real_trajectory_geodesic_drift_deg"] == .02
    with pytest.raises(ValueError, match="exactly the same fixed contexts"):
        _parity_summary(predicted[:1], reference, targets, _FROZEN_CONFIG)


def test_invalid_prediction_and_latency_cannot_be_published_as_measurements():
    with pytest.raises(ValueError, match="finite float32"):
        _prediction(np.zeros((1, 20, 4), dtype=np.float64))
    invalid = np.zeros((1, 20, 4), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite float32"):
        _prediction(invalid)
    with pytest.raises(ValueError, match="nonnegative milliseconds"):
        _latency_summary([1, -1], clock="synthetic", warmup=50)
    result = _latency_summary([1, 2, 3], clock="synthetic", warmup=50)
    assert result["median_ms"] == 2
    assert result["p95_ms"] == pytest.approx(2.9)
    assert result["p99_ms"] == pytest.approx(2.98)
    assert "not_organizer_XR_target" in result["scope"]


def test_existing_run_is_refused_before_any_recording_access(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    existing = tmp_path / "existing_private_run"
    existing.mkdir()
    sentinel = existing / "valuable.json"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        run_inference_audit(None, "unused", "unused", existing, copy.deepcopy(_FROZEN_CONFIG))
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_synthetic_real_export_parity_and_measurement_integration(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import abc_iueh.inference_audit as audit

    # This explicitly substitutes a two-repetition synthetic-test timer. The
    # returned record says two repetitions, not the production protocol's500.
    def small_truthful_timer(function, warmup, repetitions):
        measured = []
        function()
        for _ in range(2):
            started = time.perf_counter_ns()
            function()
            measured.append((time.perf_counter_ns() - started) / 1e6)
        return _latency_summary(measured, clock="synthetic_test_CPU_wall", warmup=1), measured

    monkeypatch.setattr(audit, "_time_cpu", small_truthful_timer)
    segments = []
    for speed in (.001, .002, .003):
        segments.append(exp_so3(np.arange(420)[:, None] * np.array([[0., 0., speed]])).astype(np.float32))
    samples = [{"partition": "train", "user": "private_a"},
               {"partition": "train", "user": "private_b"},
               {"partition": "development", "user": "private_c"}]
    scaler = {"fit_partition": "train", "ddof": 0, "epsilon": 1e-8,
              "mean": [0.] * 10, "scale": [1.] * 10}
    result, differences, timings = _evaluate_variant(
        "h20_a0", SelectorMLP().state_dict(), scaler, samples, segments,
        tmp_path / "synthetic.onnx", _FROZEN_CONFIG, use_cuda=False)
    assert result["all_fixed_numerical_gates_passed"]
    assert result["trainable_parameters"] == 244
    assert result["inventory"]["selector_dense_macs"] == 224
    assert result["inventory"]["signature"]["outputs"] == {"prediction": [1, 20, 4]}
    assert result["parity"]["train"]["ORT_CPU"]["contexts"] == 2
    assert result["parity"]["development"]["eager_cpu"]["contexts"] == 1
    assert result["latency"]["ORT_CPU"]["repetitions"] == 2
    assert len(differences) == 3
    assert len(timings["ORT_CPU"]) == 2
    serialized = json.dumps(result, allow_nan=False)
    for private in ("private_a", "private_b", "private_c", str(tmp_path), "state_dict", "feature_mean"):
        assert private not in serialized

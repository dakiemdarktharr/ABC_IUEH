"""Fixed train/dev numerical portability and local full-graph cost audit.

This recorder never selects a model and never offers final-partition access.
Graphs, identities and traces stay in an exclusive private run directory.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import h5py
import numpy as np
import torch

from .geometry import exp_so3, geodesic_deg, multiply_wxyz, normalize_wxyz
from .selector import PARAMETERS, VARIANTS
from .selector_inference import (FullSelectorInference, export_full_selector_onnx,
                                  reference_from_segment)
from .sources import sha256_file, write_json
from .stage import cache_groups, validate_feature_bundle


_FROZEN_CONFIG = {
    "stage": "numerical_portability_and_local_cost_not_accuracy_selection",
    "main_selector_study": "data/selector_runs/20261028_r80_gpu",
    "variants": list(VARIANTS), "seed": 20261026,
    "checkpoint_choice": "first_predeclared_seed_for_every_variant_no_development_selection",
    "graph": {"input_name": "context", "input_shape": [1, 400, 4], "output_name": "prediction",
              "output_shape": [1, 20, 4], "dtype": "float32", "opset": 17,
              "dynamic_axes": False, "float64_allowed": False, "state": "stateless"},
    "parity_sampling": {"partitions": ["train", "development"], "contexts_per_user": 26,
                        "seed": 20261028,
                        "sampling": "uniform_complete_origins_within_each_user_without_replacement"},
    "numerical_gates": {"max_synthetic_trajectory_geodesic_drift_deg": .02,
                        "max_sampled_real_trajectory_geodesic_drift_deg": .02,
                        "absolute_sampled_mean_endpoint20_error_change_deg": .01,
                        "interpretation": "numerical_portability_only_not_predictive_improvement"},
    "latency": {"warmup": 50, "repetitions": 500, "CPU_ORT_threads": 1,
                "report": ["median_ms", "p95_ms", "p99_ms"],
                "scope": "local_batch1_forward_excludes_file_IO_not_organizer_XR_target",
                "CUDA_timing_if_measured": "synchronized_resident_input_kernel_time_separate_from_CPU_ORT"},
    "parameter_count": {"trainable": 244, "selector_dense_MACs": 224,
                        "whole_graph_cost": "operator_and_scalar_arithmetic_inventory_with_nonlinear_operations_separate"},
    "final_access_allowed": False, "artifact_public": False,
}
_STATE_SHAPES = {"hidden.weight": (16, 10), "hidden.bias": (16,),
                 "output.weight": (4, 16), "output.bias": (4,)}
_PARTITIONS = ("train", "development")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("an experiment JSON artifact must be an object")
    return value


def validate_inference_config(config):
    """Refuse altered sampling, gates, checkpoint choice or latency protocol."""
    def check(actual, expected):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise ValueError("unsupported frozen inference configuration")
            for key in expected:
                check(actual[key], expected[key])
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise ValueError("unsupported frozen inference configuration")
            for one, two in zip(actual, expected):
                check(one, two)
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError("unsupported frozen inference configuration")
    check(config, _FROZEN_CONFIG)
    return config


def _source_hashes():
    directory = Path(__file__).parent
    return {name: sha256_file(directory / name) for name in (
        "inference_audit.py", "selector_inference.py", "selector.py", "evidence.py",
        "geometry.py", "stage.py", "sources.py",
    )}


def _artifact_hashes(directory, expected):
    """Hash only receipt-declared direct children; never interpret data values."""
    if not isinstance(expected, dict) or not expected:
        raise ValueError("missing trusted artifact hashes")
    actual = {}
    for name, digest in expected.items():
        if (not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
                or not isinstance(digest, str) or len(digest) != 64):
            raise ValueError("invalid trusted artifact identity")
        actual[name] = sha256_file(Path(directory) / name)
        if actual[name] != digest:
            raise ValueError("main study artifact differs from its saved receipt")
    return actual


def _validate_main(stage, baseline_dir, main_dir):
    receipt, plan, summary = (_json(Path(main_dir) / name) for name in
                              ("receipt.json", "experiment_plan.json", "summary.json"))
    required_artifacts = {"experiment_plan.json", "errors.h5", "diagnostics.h5", "private_user_stats.json",
                          "training_receipts.json", "summary.json", "sampled_train_origins_seed_20261026.npy",
                          "sampled_train_origins_seed_20261027.npy"}
    if set(receipt.get("artifact_sha256", {})) != required_artifacts:
        raise ValueError("main study receipt has an unexpected artifact inventory")
    artifacts = _artifact_hashes(main_dir, receipt["artifact_sha256"])
    bindings = receipt.get("bindings")
    if not isinstance(bindings, dict) or plan.get("bindings") != bindings or summary.get("bindings") != bindings:
        raise ValueError("main study provenance bindings disagree")
    if (bindings.get("source_hdf5_sha256") != stage.hdf5_sha256
            or bindings.get("split_sha256") != stage.split_sha256
            or summary.get("final_holdout_evaluated") is not False
            or summary.get("parameters") != PARAMETERS or summary.get("evidence_length") != 80
            or summary.get("variants") != list(VARIANTS) or summary.get("seeds") != [20261026, 20261027]
            or summary.get("windows") != {"train": 3330312, "development": 1189027}):
        raise ValueError("main study differs from the locked train/development selector")
    feature_receipt = validate_feature_bundle(stage, baseline_dir)
    if (feature_receipt["cache_sha256"] != bindings.get("baseline_feature_cache_sha256")
            or feature_receipt["scalers_sha256"] != bindings.get("baseline_scalers_sha256")
            or sha256_file(Path(baseline_dir) / "feature_receipt.json") != bindings.get("feature_receipt_sha256")
            or sha256_file(Path(baseline_dir) / "experiment_plan.json") != bindings.get("baseline_experiment_plan_sha256")):
        raise ValueError("baseline artifacts differ from the main study binding")
    sources = receipt.get("implementation_sources_sha256")
    if (not isinstance(sources, dict) or plan.get("implementation_sources_sha256") != sources
            or summary.get("implementation_sources_sha256") != sources):
        raise ValueError("main study source manifests disagree")
    for name, digest in sources.items():
        if Path(name).name != name or sha256_file(Path(__file__).parent / name) != digest:
            raise ValueError("a main study implementation source changed")
    expected_methods = {f"Selector_{variant}_seed_{seed}" for variant in VARIANTS for seed in (20261026, 20261027)}
    if set(receipt.get("checkpoint_sha256", {})) != expected_methods:
        raise ValueError("main checkpoint inventory differs from the two fixed seeds")
    checkpoint_hashes = {}
    for method, digest in receipt["checkpoint_sha256"].items():
        actual = sha256_file(Path(main_dir) / "checkpoints" / f"{method}.pt")
        if actual != digest:
            raise ValueError("a main checkpoint differs from its saved receipt")
        checkpoint_hashes[method] = actual
    for audit in (receipt.get("audit", {}), summary.get("audit", {})):
        if (audit.get("data_and_split_unchanged") is not True or audit.get("final_quaternion_datasets_read") != 0
                or audit.get("source_hdf5_sha256") != stage.hdf5_sha256
                or audit.get("split_sha256") != stage.split_sha256):
            raise ValueError("main study does not attest to guarded train/development access")
    return {"main_receipt_sha256": sha256_file(Path(main_dir) / "receipt.json"),
            "main_artifact_sha256": artifacts, "main_checkpoint_sha256": checkpoint_hashes,
            "bindings": bindings, "main_implementation_sources_sha256": sources,
            "inference_implementation_sources_sha256": _source_hashes()}


def _load_checkpoint(path, variant, bindings, expected_hash):
    if sha256_file(path) != expected_hash:
        raise ValueError("checkpoint hash changed before trusted local loading")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "architecture", "parameters", "dense_MACs", "variant", "seed", "optimizer_steps",
                "checkpoint_selection", "loss", "bindings"}
    if (not isinstance(checkpoint, dict) or set(checkpoint) != required
            or checkpoint["architecture"] != [10, 16, 4] or checkpoint["parameters"] != 244
            or checkpoint["dense_MACs"] != 224 or checkpoint["variant"] != variant
            or checkpoint["seed"] != 20261026 or checkpoint["optimizer_steps"] != 2000
            or checkpoint["checkpoint_selection"] != "fixed_final_step_no_development_selection"
            or checkpoint["loss"] != "expected_individual_expert_endpoint20_angular_loss"
            or checkpoint["bindings"] != bindings):
        raise ValueError("checkpoint metadata violates the fixed first-seed audit contract")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or set(state) != set(_STATE_SHAPES):
        raise ValueError("checkpoint state has unexpected learned tensor keys")
    for name, shape in _STATE_SHAPES.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError("checkpoint learned tensors must be finite float32 with trusted shapes")
    if sum(value.numel() for value in state.values()) != 244:
        raise ValueError("checkpoint must contain exactly244 learned parameters")
    return state


def sample_user_origins(cache, partitions, *, contexts_per_user=26, seed=20261028):
    """Balanced fixed user sampling over complete segment-local origins.

    ``partitions`` is the stage's private user mapping. No final value/group is
    traversed. Returned identities belong only in the private sample receipt.
    """
    if (type(contexts_per_user) is not int or contexts_per_user < 1
            or type(seed) is not int or seed < 0):
        raise ValueError("invalid fixed parity-sampling budget or seed")
    rng, samples, public = np.random.default_rng(seed), [], {}
    for partition in _PARTITIONS:
        users = partitions.get(partition)
        if not isinstance(users, (list, tuple)) or not users or len(users) != len(set(users)):
            raise ValueError("sampling requires unique locked train/development users")
        groups = list(cache_groups(cache, partition))
        if {row[0] for row in groups} != set(users):
            raise ValueError("cache and locked sampling users differ")
        total_windows = 0
        for user in sorted(users):
            segments = sorted([row for row in groups if row[0] == user], key=lambda row: (row[1], row[2]))
            counts = np.asarray([len(row[3]["origin_index"]) for row in segments], dtype=np.int64)
            if np.any(counts <= 0) or int(counts.sum()) < contexts_per_user:
                raise ValueError("each allowed user must have the fixed number of complete origins")
            cumulative = np.cumsum(counts)
            selected = np.sort(rng.choice(int(cumulative[-1]), size=contexts_per_user, replace=False))
            total_windows += int(cumulative[-1])
            for index in selected:
                segment_index = int(np.searchsorted(cumulative, index, side="right"))
                _, session, segment, group = segments[segment_index]
                local = int(index - (cumulative[segment_index - 1] if segment_index else 0))
                origin = int(group["origin_index"][local])
                if origin != 399 + local:
                    raise ValueError("cache origin row does not match complete stride1 indexing")
                samples.append({"partition": partition, "user": user, "session": session,
                                "segment": segment, "origin_index": origin})
        public[partition] = {"users": len(users), "contexts": len(users) * contexts_per_user,
                             "contexts_per_user": contexts_per_user, "population_windows": total_windows}
    serialized = json.dumps(samples, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return samples, public, hashlib.sha256(serialized).hexdigest()


def collect_sampled_segments(stage, samples):
    """Read values exclusively through the locked stage's train/dev iterator."""
    grouped = {}
    for index, sample in enumerate(samples):
        partition = sample.get("partition")
        if partition not in _PARTITIONS or sample.get("user") not in stage.partitions[partition]:
            raise PermissionError("an inference audit sample reached a forbidden user/partition")
        key = (partition, sample["user"], sample["session"], sample["segment"])
        grouped.setdefault(key, []).append((index, sample["origin_index"]))
    result = [None] * len(samples)
    for partition in _PARTITIONS:
        seen = set()
        for user, session, segment, quaternion in stage.iter_segments(partition):
            key = (partition, user, session, segment)
            if key in seen or user not in stage.partitions[partition]:
                raise ValueError("duplicate or unexpected guarded source segment")
            seen.add(key)
            for index, origin in grouped.get(key, []):
                if type(origin) is not int or origin < 399 or origin + 20 >= len(quaternion):
                    raise ValueError("sample origin does not have400 context and20 target observations")
                selected = np.array(quaternion[origin - 399:origin + 21], dtype=np.float32, copy=True)
                if selected.shape != (420, 4) or not np.isfinite(selected).all():
                    raise ValueError("invalid finite sampled source quaternion segment")
                result[index] = selected
        if any(key[0] == partition and key not in seen for key in grouped):
            raise ValueError("a sampled segment is absent from the guarded source")
    if not result or any(value is None for value in result):
        raise ValueError("guarded sampling did not cover all fixed contexts")
    return result


def _prediction(value):
    array = np.asarray(value)
    if array.shape != (1, 20, 4) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("inference output must be finite float32 [1,20,4]")
    norms = np.linalg.norm(array.astype(np.float64), axis=-1)
    if np.any(norms == 0):
        raise ValueError("inference output contains a zero-norm quaternion")
    norm_error = float(np.max(np.abs(norms - 1)))
    return array, norm_error


def _parity_summary(predictions, references, targets, config):
    predicted, reference, target = (np.asarray(values) for values in (predictions, references, targets))
    if len(predicted) != len(reference) or len(predicted) != len(target) or not len(predicted):
        raise ValueError("parity vectors must cover exactly the same fixed contexts")
    if (predicted.shape != (len(predicted), 20, 4) or reference.shape != predicted.shape
            or target.shape != (len(predicted), 4) or predicted.dtype != np.float32
            or reference.dtype != np.float32 or not np.isfinite(predicted).all()
            or not np.isfinite(reference).all() or not np.isfinite(target).all()):
        raise ValueError("parity requires aligned finite float32 trajectories and endpoint targets")
    drift = geodesic_deg(predicted, reference)
    actual = geodesic_deg(predicted[:, -1], target)
    reference_error = geodesic_deg(reference[:, -1], target)
    score_change = float(actual.mean() - reference_error.mean())
    maximum = float(drift.max())
    gates = config["numerical_gates"]
    return {"contexts": len(predicted), "output_quaternions": int(drift.size),
            "reference_sampled_mean_endpoint20_error_deg": float(reference_error.mean()),
            "emitted_sampled_mean_endpoint20_error_deg": float(actual.mean()),
            "sampled_mean_endpoint20_error_change_deg": score_change,
            "absolute_sampled_mean_endpoint20_error_change_deg": abs(score_change),
            "max_trajectory_geodesic_drift_deg": maximum,
            "mean_trajectory_geodesic_drift_deg": float(drift.mean()),
            "max_output_norm_deviation": float(np.abs(np.linalg.norm(predicted.astype(np.float64), axis=-1) - 1).max()),
            "all_outputs_finite": bool(np.isfinite(predicted).all()),
            "numerical_gate_checks": {
                "trajectory_drift_within_fixed_0_02_deg": maximum <= gates["max_sampled_real_trajectory_geodesic_drift_deg"],
                "absolute_partition_mean_endpoint_error_change_within_fixed_0_01_deg":
                    abs(score_change) <= gates["absolute_sampled_mean_endpoint20_error_change_deg"]}}


def _latency_summary(milliseconds, *, clock, warmup):
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("latency samples must be finite nonnegative milliseconds")
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)), "repetitions": len(values), "warmup": warmup,
            "clock": clock, "batch_size": 1, "resident_input": True, "file_IO_included": False,
            "scope": "local_full_stateless_400_to_20_forward_not_organizer_XR_target"}


def _time_cpu(function, warmup, repetitions):
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        function()
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return _latency_summary(timings, clock="CPU_perf_counter_ns_wall", warmup=warmup), timings


def _time_cuda(model, context, warmup, repetitions):
    for _ in range(warmup):
        model(context)
    torch.cuda.synchronize(context.device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    event_times, wall_times = [], []
    for _ in range(repetitions):
        torch.cuda.synchronize(context.device)
        started = time.perf_counter_ns()
        begin.record()
        model(context)
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter_ns() - started) / 1e6)
        event_times.append(begin.elapsed_time(end))
    event = _latency_summary(event_times, clock="synchronized_CUDA_events_resident_input", warmup=warmup)
    event["synchronized_wall"] = _latency_summary(wall_times, clock="CPU_perf_counter_ns_with_CUDA_synchronization", warmup=warmup)
    return event, {"CUDA_event_ms": event_times, "synchronized_wall_ms": wall_times}


def _synthetic_segments():
    cases = []
    for scale in (0., .018, 1.8):
        rng = np.random.default_rng(20261028)
        q = np.empty((420, 4), dtype=np.float64)
        q[0] = [1, 0, 0, 0]
        for index in range(1, len(q)):
            q[index] = normalize_wxyz(multiply_wxyz(exp_so3(rng.normal(0, scale, 3)), q[index - 1]))
            if np.dot(q[index], q[index - 1]) < 0:
                q[index] *= -1
        cases.append(q.astype(np.float32))
    q = np.zeros((420, 4), dtype=np.float32)
    q[::2, 0], q[1::2, 1] = 1, 1
    cases.append(q)
    return cases


def _make_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("the audit requires explicitly recorded CPU-only ONNX execution")
    return session


def _evaluate_variant(variant, state, scaler, samples, segments, path, config, *, use_cuda):
    model = FullSelectorInference(state, scaler, variant=variant, evidence_length=80).eval()
    example = torch.from_numpy(segments[0][:400][None].copy())
    inventory = export_full_selector_onnx(model, path, example_context=example)
    session = _make_session(path)
    cuda_model = copy.deepcopy(model).to("cuda:0").eval() if use_cuda else None
    providers = ("eager_cpu", "ORT_CPU") + (("eager_CUDA",) if use_cuda else ())
    values = {partition: {provider: [] for provider in providers} for partition in _PARTITIONS}
    references = {partition: [] for partition in _PARTITIONS}
    targets = {partition: [] for partition in _PARTITIONS}
    private_differences = []
    synthetic = {provider: 0.0 for provider in providers}

    def evaluate(segment):
        context = np.ascontiguousarray(segment[:400][None])
        reference, _ = _prediction(reference_from_segment(segment, state, scaler, variant=variant)["prediction"])
        output = {"eager_cpu": _prediction(model(torch.from_numpy(context)).numpy())[0],
                  "ORT_CPU": _prediction(session.run(["prediction"], {"context": context})[0])[0]}
        if use_cuda:
            tensor = torch.from_numpy(context).to("cuda:0")
            output["eager_CUDA"] = _prediction(cuda_model(tensor).cpu().numpy())[0]
        return reference, output

    with torch.inference_mode():
        for segment in _synthetic_segments():
            reference, output = evaluate(segment)
            for provider in providers:
                synthetic[provider] = max(synthetic[provider], float(geodesic_deg(output[provider], reference).max()))
        for sample_index, (sample, segment) in enumerate(zip(samples, segments)):
            partition = sample["partition"]
            reference, output = evaluate(segment)
            references[partition].append(reference[0])
            targets[partition].append(segment[419])
            item = {"sample_index": sample_index, "providers": {}}
            for provider in providers:
                values[partition][provider].append(output[provider][0])
                item["providers"][provider] = {
                    "max_trajectory_geodesic_drift_deg": float(geodesic_deg(output[provider], reference).max()),
                    "endpoint20_error_deg": float(geodesic_deg(output[provider][0, 19], segment[419]))}
            private_differences.append(item)
        latency, private_timings = {}, {}
        settings = config["latency"]
        resident_CPU = example.numpy()
        calls = {"eager_cpu": lambda: model(example),
                 "ORT_CPU": lambda: session.run(["prediction"], {"context": resident_CPU})}
        for provider, function in calls.items():
            latency[provider], private_timings[provider] = _time_cpu(function, settings["warmup"], settings["repetitions"])
        if use_cuda:
            resident = example.to("cuda:0")
            latency["eager_CUDA"], private_timings["eager_CUDA"] = _time_cuda(
                cuda_model, resident, settings["warmup"], settings["repetitions"])
    parity = {}
    for partition in _PARTITIONS:
        parity[partition] = {provider: _parity_summary(values[partition][provider], references[partition],
                                                     targets[partition], config) for provider in providers}
        parity[partition]["ORT_vs_eager_CPU_max_trajectory_drift_deg"] = float(geodesic_deg(
            np.asarray(values[partition]["ORT_CPU"]), np.asarray(values[partition]["eager_cpu"])).max())
    synthetic_limit = config["numerical_gates"]["max_synthetic_trajectory_geodesic_drift_deg"]
    passed = (all(value <= synthetic_limit for value in synthetic.values())
              and all(all(parity[p][provider]["numerical_gate_checks"].values())
                      for p in _PARTITIONS for provider in providers))
    return {"variant": variant, "seed": 20261026, "evidence_length": 80,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "graph_bytes": path.stat().st_size, "graph_sha256": sha256_file(path),
            "inventory": inventory, "parity": parity, "latency": latency,
            "synthetic_actual_checkpoint_probe": {"cases": 4, "synthetic_inputs_only": True,
                                                   "max_trajectory_drift_deg": synthetic,
                                                   "fixed_gate_deg": synthetic_limit},
            "all_fixed_numerical_gates_passed": passed}, private_differences, private_timings


def run_inference_audit(stage, baseline_dir, main_dir, output_dir, config, *, config_path="configs/inference_audit.json"):
    """Run the frozen real-context audit once, preserving every result and gate.

    The caller must wait until other real selector studies have completed before
    timing. The standard sensitivity-completion receipt is required here too.
    Returned/public summary has no private sample identities, paths or tensors.
    """
    import onnx
    import onnxruntime as ort

    validate_inference_config(config)
    config_path, baseline_dir, main_dir, output_dir = map(Path, (config_path, baseline_dir, main_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("inference audit refuses to overwrite an existing run")
    if _json(config_path) != config:
        raise ValueError("inference audit arguments differ from the saved frozen config")
    if main_dir.resolve() != Path(config["main_selector_study"]).resolve():
        raise ValueError("the audit must use the recorded main selector study")
    sensitivity = main_dir.parent / "20261028_R_sensitivity_gpu" / "summary.json"
    if not sensitivity.is_file():
        raise RuntimeError("wait for the R128/R256 study completion summary before inference timing")
    bindings = _validate_main(stage, baseline_dir, main_dir)
    before_audit = stage.verify_unchanged()
    config_hash = sha256_file(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(output_dir / "experiment_plan.json", {"created_utc": _utc(), "config": config,
                                                    "config_sha256": config_hash, "bindings": bindings,
                                                    "sensitivity_completion_summary_sha256": sha256_file(sensitivity),
                                                    "synthetic_actual_checkpoint_probe": {
                                                        "seed": 20261028, "random_walk_scales_rad": [0, .018, 1.8],
                                                        "alternating_exact_halfturn": True}})
    try:
        with h5py.File(baseline_dir / "features.h5", "r") as cache:
            samples, sample_counts, sample_hash = sample_user_origins(cache, stage.partitions)
        if sample_counts != {"train": {"users": 15, "contexts": 390, "contexts_per_user": 26, "population_windows": 3330312},
                             "development": {"users": 5, "contexts": 130, "contexts_per_user": 26, "population_windows": 1189027}}:
            raise ValueError("real inference sampling differs from the fixed520-context train/dev cohort")
        write_json(output_dir / "private_samples.json", {"sample_sha256": sample_hash, "samples": samples})
        segments = collect_sampled_segments(stage, samples)
        scalers = _json(baseline_dir / "scalers.json")["variants"]
        torch.set_num_threads(1)
        if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
            raise ValueError("the fixed float32 inference audit requires autocast disabled")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
        runtime = {"python_version": platform.python_version(), "operating_system": platform.system(),
                   "numpy_version": np.__version__, "torch_version": str(torch.__version__),
                   "torch_cuda_version": torch.version.cuda, "onnx_version": onnx.__version__,
                   "onnxruntime_version": ort.__version__, "ORT_available_providers": ort.get_available_providers(),
                   "ORT_measured_provider": "CPUExecutionProvider", "ORT_intra_op_threads": 1, "ORT_inter_op_threads": 1,
                   "torch_CPU_threads": torch.get_num_threads(), "cuda_measured": use_cuda,
                   "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
                   "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
                   "device_name": torch.cuda.get_device_name(0) if use_cuda else None,
                   "timing_scope": "local_desktop_no_competing_study_local_batch1_not_organizer_XR_target"}
        graphs = output_dir / "graphs"
        graphs.mkdir()
        results = {}
        for variant in VARIANTS:
            method = f"Selector_{variant}_seed_20261026"
            state = _load_checkpoint(main_dir / "checkpoints" / f"{method}.pt", variant, bindings["bindings"],
                                     bindings["main_checkpoint_sha256"][method])
            result, differences, timings = _evaluate_variant(variant, state, scalers[variant], samples, segments,
                                                              graphs / f"{variant}.onnx", config, use_cuda=use_cuda)
            results[variant] = result
            write_json(output_dir / f"private_differences_{variant}.json", {"samples": differences})
            write_json(output_dir / f"private_timing_samples_{variant}.json", timings)
            print(json.dumps({"event": "inference_variant_done", "variant": variant,
                              "fixed_numerical_gates_passed": result["all_fixed_numerical_gates_passed"]}), flush=True)
        if use_cuda:
            torch.cuda.synchronize(0)
            runtime["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated(0)
            runtime["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved(0)
        after_audit = stage.verify_unchanged()
        if (_validate_main(stage, baseline_dir, main_dir) != bindings or sha256_file(config_path) != config_hash
                or _source_hashes() != bindings["inference_implementation_sources_sha256"]):
            raise ValueError("source, configuration or main artifacts changed during inference audit")
        passed = all(item["all_fixed_numerical_gates_passed"] for item in results.values())
        summary = {"created_utc": _utc(), "stage": config["stage"], "status": "PASS" if passed else "FAIL_numerical_portability",
                   "all_fixed_numerical_gates_passed": passed, "results": results, "sampling": sample_counts,
                   "sample_schedule_sha256": sample_hash, "config_sha256": config_hash,
                   "bindings": bindings, "runtime": runtime, "audit_before": before_audit, "audit_after": after_audit,
                   "checkpoint_choice": config["checkpoint_choice"], "seed": 20261026,
                   "sampled_score_scope": "balanced26_contexts_per_allowed_user_portability_diagnostic_not_all_window_accuracy",
                   "parameter_scope": "244_trainable_parameters_224_selector_dense_MACs_whole_graph_inventory_separate",
                   "latency_scope": config["latency"]["scope"], "final_holdout_evaluated": False,
                   "model_selection_performed": False, "artifact_public": False,
                   "elapsed_seconds": time.perf_counter() - started}
        write_json(output_dir / "summary.json", summary)
        artifacts = sorted(path for path in output_dir.rglob("*") if path.is_file())
        receipt = {"created_utc": _utc(), "status": summary["status"], "bindings": bindings,
                   "sample_schedule_sha256": sample_hash, "config_sha256": config_hash, "audit": after_audit,
                   "artifact_sha256": {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in artifacts}}
        write_json(output_dir / "receipt.json", receipt)
        return summary
    except BaseException as error:
        write_json(output_dir / "failure.json", {"created_utc": _utc(), "exception_type": type(error).__name__,
                                                "private_error_trace": traceback.format_exc(),
                                                "config_sha256": config_hash, "bindings": bindings,
                                                "final_holdout_evaluated": False})
        raise

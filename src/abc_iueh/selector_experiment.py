"""Matched fixed-step CUDA selector experiment; final access is unavailable.

All model fitting reads only the sealed training features and separate future
expert-loss labels. Development arrays enter only after both seeds have fitted.
The six variants share initialization, sampled global origins and SGD budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import platform
import re
import time

import h5py
import numpy as np
import torch

from .evaluation import predict_endpoint
from .geometry import exp_so3, geodesic_deg, inverse_wxyz, log_so3, multiply_wxyz, normalize_wxyz
from .selector import (
    ARCHITECTURE, DENSE_MACS, PARAMETERS, SEEDS, VARIANTS, SelectorMLP,
    expected_expert_loss, normalise_features, predict_blended_quaternions,
)
from .sources import sha256_file, utc_now, write_json
from .stage import cache_groups, validate_cache_binding, validate_feature_bundle


_PARTITIONS = ("train", "development")
_TRANSFER_CHUNK = 65536
_EVALUATION_BATCH = 8192
_SENSITIVITY_VARIANTS = ("h1_a0", "h20_a0")

_FROZEN_CONFIG = {
    "stage": "train_development_selector_only", "partitions": ["train", "development"],
    "context_samples": 400, "horizon_samples": 20, "stride": 1, "evidence_length": 80,
    "variants": list(VARIANTS),
    "selector": {
        "architecture": list(ARCHITECTURE), "parameters": PARAMETERS, "dense_MACs": DENSE_MACS,
        "steps": 2000, "batch_size": 128, "optimizer": "SGD", "learning_rate": .001,
        "momentum": 0.0, "weight_decay": 0.0, "seeds": list(SEEDS),
        "loss": "expected_individual_expert_endpoint20_angular_loss", "early_stopping": False,
        "initialization": "normal_std_0.02_zero_bias", "same_initial_state_per_seed_across_variants": True,
        "sampling": "uniform_global_train_origins_with_replacement",
        "same_origin_batches_per_seed_across_variants": True,
    },
    "feature_standardization": "existing_train_only_window_weighted_population_mean_std_epsilon_1e-8_float64_then_float32",
    "baseline_config": "configs/baseline_features.json",
    "uncertainty": {"cluster": "user", "bootstrap_resamples": 2000, "seed": 20261026,
                    "estimand": "window_weighted_ratio", "development_inference": "exploratory_only_five_users"},
    "primary_contrast": "h20_a0_minus_h1_a0_two_seed_mean_errors",
    "success_thresholds_unchanged": {"relative_improvement_minimum": .03,
                                    "absolute_improvement_degrees_minimum": .05,
                                    "paired_user_bootstrap_CI_upper_delta_below_zero": True,
                                    "users_improved_fraction_minimum": .60,
                                    "confirmatory_partition": "final_not_accessed_in_this_study"},
    "device_required": "cuda", "precision": "float32_no_autocast_no_TF32", "deterministic_algorithms": True,
    "evaluation_batch_size": 65536, "diagnostics": ["expected_loss_deg", "hard_error_deg"],
    "prospective_sensitivity": {"R": [128, 256], "variants": ["h1_a0", "h20_a0"], "same_training_budget": True},
    "paper_scope": "exploratory_train_development_study_with_sealed_final_holdout",
    "final_access_allowed": False, "selection_after_final_results_allowed": False,
    "model_or_feature_artifacts_public": False, "cloud_approved_cost": 0,
}
_HASH_FIELDS = ("baseline_feature_cache_sha256", "baseline_scalers_sha256", "source_hdf5_sha256", "split_sha256")


def validate_selector_config(config):
    """Reject unsupported or ignored changes to the prospective R80 recipe."""
    def check(actual, expected, path):
        if isinstance(expected, dict):
            valid = isinstance(actual, dict) and set(actual) == set(expected)
        elif isinstance(expected, list):
            valid = isinstance(actual, list) and len(actual) == len(expected)
        elif isinstance(expected, bool):
            valid = type(actual) is bool and actual == expected
        elif isinstance(expected, int):
            valid = type(actual) is int and actual == expected
        elif isinstance(expected, float):
            valid = type(actual) in (int, float) and actual == expected and np.isfinite(actual)
        else:
            valid = type(actual) is type(expected) and actual == expected
        if not valid:
            raise ValueError(f"unsupported frozen selector config at {path}")
        if isinstance(expected, dict):
            for key, value in expected.items():
                check(actual[key], value, f"{path}.{key}")
        elif isinstance(expected, list):
            for index, value in enumerate(expected):
                check(actual[index], value, f"{path}[{index}]")
    if (not isinstance(config, dict)
            or set(config) != set(_FROZEN_CONFIG) | set(_HASH_FIELDS) | {"baseline_code_commit"}):
        raise ValueError("unsupported frozen selector config keys")
    check({key: config[key] for key in _FROZEN_CONFIG}, _FROZEN_CONFIG, "config")
    for name, width in [(field, 64) for field in _HASH_FIELDS] + [("baseline_code_commit", 40)]:
        if not isinstance(config[name], str) or re.fullmatch(f"[0-9a-f]{{{width}}}", config[name]) is None:
            raise ValueError(f"invalid frozen selector config binding {name}")
    return config


def _groups(cache, partition):
    groups = list(cache_groups(cache, partition))
    keys = [tuple(row[:3]) for row in groups]
    if not groups or keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("selector cache groups must be nonempty unique lexical segment rows")
    return groups


def _variant_selection(variants):
    """Only the main set or the prospectively declared h1/h20 subset."""
    selected = tuple(variants)
    if selected not in (VARIANTS, _SENSITIVITY_VARIANTS):
        raise ValueError("selector helpers require all six main variants or the declared h1/h20 sensitivity pair")
    return selected


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_runtime(device):
    """Require a working CUDA computation, not just a driver/device listing."""
    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("the real selector study requires CUDA; CPU is available only to lower-level synthetic tests")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable; selector study cannot fall back to CPU")
    torch.cuda.set_device(torch.cuda.current_device() if requested.index is None else requested.index)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if torch.is_autocast_enabled("cuda"):
        raise ValueError("selector study requires float32 without a surrounding CUDA autocast context")
    left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    product = left @ left.T
    end.record()
    _synchronize(device)
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    if not product.is_cuda or not np.array_equal(product.cpu().numpy(), expected @ expected.T):
        raise RuntimeError("actual CUDA matrix computation failed its numerical check")
    properties = torch.cuda.get_device_properties(device)
    runtime = {
        "requested_device": str(requested), "actual_device": str(device),
        "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "h5py_version": h5py.__version__, "operating_system": platform.system(),
        "cuda_available": True, "device_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_device_memory_bytes": properties.total_memory,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "actual_cuda_probe": {"operation": "float32_4x4_matrix_multiply",
                              "input_is_cuda": left.is_cuda, "output_is_cuda": product.is_cuda,
                              "numerical_check_passed": True,
                              "synchronized_cuda_event_elapsed_ms": begin.elapsed_time(end)},
    }
    del left, product
    torch.cuda.reset_peak_memory_stats(device)
    return device, runtime


def _validate_source_partition(cache, stage, partition, *, chunk_size=_TRANSFER_CHUNK):
    """Independent guarded source check of full current/endpoint/label coverage.

    Source quaternion values are obtained solely through iter_segments. There
    is no path, flag or API allowing a final-partition source read here.
    """
    if partition not in _PARTITIONS or chunk_size < 1:
        raise PermissionError("selector source checks permit train/development only")
    groups = {tuple(row[:3]): row[3] for row in _groups(cache, partition)}
    seen, windows = set(), 0
    for user, session, segment, q in stage.iter_segments(partition):
        key = user, session, segment
        if (user not in stage.partitions[partition] or key in seen or q.flags.writeable
                or q.dtype != np.float32 or q.ndim != 2 or q.shape[1] != 4):
            raise ValueError("invalid or duplicate guarded selector source segment")
        seen.add(key)
        count = max(0, len(q) - 419)
        if not count:
            if key in groups:
                raise ValueError("short source segment unexpectedly has cached windows")
            continue
        if key not in groups:
            raise ValueError("selector cache omits a complete guarded source segment")
        group, unit = groups[key], normalize_wxyz(q)
        if len(group["origin_index"]) != count:
            raise ValueError("selector cache/source window count mismatch")
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            origins = np.arange(399 + start, 399 + stop, dtype=np.int64)
            if (not np.array_equal(group["origin_index"][start:stop], origins)
                    or not np.array_equal(group["current_q"][start:stop], q[origins])
                    or not np.array_equal(group["target_q"][start:stop], q[origins + 20])):
                raise ValueError("cached current/endpoint20 targets or origins differ from guarded source")
            velocities = np.zeros((stop - start, 4, 3), dtype=np.float64)
            for expert, bandwidth in enumerate((1, 5, 20), 1):
                velocities[:, expert] = log_so3(multiply_wxyz(
                    unit[origins], inverse_wxyz(unit[origins - bandwidth]))) / bandwidth
            if not np.array_equal(group["velocities"][start:stop], velocities):
                raise ValueError("cached world velocities differ from guarded source")
            prediction = normalize_wxyz(multiply_wxyz(
                exp_so3(20 * velocities), unit[origins, None, :])).astype(np.float32)
            expert_loss = geodesic_deg(prediction, q[origins + 20, None, :])
            if not np.allclose(group["expert_loss"][start:stop], expert_loss, rtol=0, atol=1e-12):
                raise ValueError("cached future expert losses differ from guarded source endpoint20 labels")
        windows += count
    if set(groups) != {key for key in seen if key in groups}:
        raise ValueError("selector cache contains a segment absent from guarded source")
    if not windows:
        raise ValueError("guarded selector partition has no complete windows")
    return {"partition": partition, "source_segments": len(seen), "windows_verified": windows,
            "current_target_velocity_and_expert_loss_checked": True}


def _load_training_arrays(cache, scalers, device, *, chunk_size=_TRANSFER_CHUNK, variants=VARIANTS):
    """Contiguous reads, float64 scaling ->float32, then direct device transfer."""
    if chunk_size < 1:
        raise ValueError("training transfer chunk must be positive")
    variants = _variant_selection(variants)
    groups = _groups(cache, "train")
    total = sum(len(group["origin_index"]) for *_, group in groups)
    features = {variant: torch.empty((total, 10), dtype=torch.float32, device=device) for variant in variants}
    labels = torch.empty((total, 4), dtype=torch.float32, device=device)
    position = 0
    for *_, group in groups:
        count = len(group["origin_index"])
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            destination = slice(position + start, position + stop)
            for variant in variants:
                normalized = normalise_features(group[f"features/{variant}"][start:stop], scalers[variant])
                features[variant][destination] = torch.as_tensor(normalized, device=device)
            future_loss = group["expert_loss"][start:stop]
            if (future_loss.shape != (stop - start, 4) or not np.isfinite(future_loss).all()
                    or np.any(future_loss < 0) or np.any(future_loss > 180 + 1e-6)):
                raise ValueError("invalid separate endpoint20 expert-loss training labels")
            labels[destination] = torch.as_tensor(future_loss.astype(np.float32), device=device)
        position += count
    if position != total:
        raise ValueError("training preload failed full cache coverage")
    _synchronize(device)
    return features, labels


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _train_matched_seed(features, labels, *, seed, steps, batch_size, learning_rate, device, variants=VARIANTS):
    """Fixed SGD for all variants using the identical global origin per step."""
    variants = _variant_selection(variants)
    if (set(features) != set(variants) or labels.ndim != 2 or labels.shape[1] != 4 or not len(labels)
            or steps < 1 or batch_size < 1 or learning_rate != .001
            or any(x.shape != (len(labels), 10) or x.device != labels.device for x in features.values())):
        raise ValueError("invalid matched selector training arrays or fixed SGD settings")
    from .selector_metrics import method_name
    global_indexes = np.random.default_rng(seed).integers(0, len(labels), size=(steps, batch_size), dtype=np.int64)
    index_hash = hashlib.sha256(global_indexes.astype("<i8", copy=False).tobytes()).hexdigest()
    device_indexes = torch.as_tensor(global_indexes, device=device)
    initial = SelectorMLP(seed=seed).to(device)
    initial_hash = _state_sha256(initial)
    models = {method_name(variant, seed): copy.deepcopy(initial) for variant in variants}
    optimizers = {name: torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
                  for name, model in models.items()}
    losses = torch.empty((len(variants), steps), dtype=torch.float32, device=device)
    _synchronize(device)
    started = time.perf_counter()
    actual_device_flags = []
    for step in range(steps):
        indexes = device_indexes[step]
        batch_labels = labels[indexes]
        for variant_index, variant in enumerate(variants):
            name = method_name(variant, seed)
            model, optimizer = models[name], optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            batch_features = features[variant][indexes]
            weights = model(batch_features)
            loss = expected_expert_loss(weights, batch_labels)
            loss.backward()
            if step == 0:
                actual_device_flags.append({
                    "method": name, "features_device": str(batch_features.device),
                    "labels_device": str(batch_labels.device), "weights_device": str(weights.device),
                    "loss_device": str(loss.device),
                    "parameters_and_gradients_on_device": all(
                        p.device == labels.device and p.grad is not None and p.grad.device == labels.device
                        for p in model.parameters()),
                })
            optimizer.step()
            losses[variant_index, step] = loss.detach()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            _synchronize(device)
            values = losses[:, step].cpu().tolist()
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError("selector training produced a nonfinite/negative expected angular loss")
            print(json.dumps({"event": "selector_training", "seed": seed,
                              "step": step + 1, "steps": steps, "losses": dict(zip(variants, values))}), flush=True)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    all_losses = losses.cpu().numpy().astype(np.float64)
    if not np.isfinite(all_losses).all() or np.any(all_losses < 0):
        raise ValueError("selector training losses must all be finite angular expectations")
    fits = []
    for variant_index, variant in enumerate(variants):
        name = method_name(variant, seed)
        model, values = models[name], all_losses[variant_index]
        if sum(p.numel() for p in model.parameters()) != PARAMETERS or not all(
                torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("selector checkpoint violates finite 244-parameter contract")
        fits.append({
            "method": name, "variant": variant, "seed": seed,
            "first_step": 1, "last_step": steps, "optimizer_steps": steps,
            "optimizer": "SGD", "learning_rate": learning_rate, "momentum": 0, "weight_decay": 0,
            "batch_size": batch_size, "sampled_training_windows": steps * batch_size,
            "population_training_windows": len(labels), "sampling_seed": seed,
            "sampling": "uniform_global_train_origins_with_replacement",
            "global_origin_batches_sha256": index_hash, "initial_state_sha256": initial_hash,
            "final_state_sha256": _state_sha256(model), "first_step_loss": float(values[0]),
            "last_step_loss": float(values[-1]), "mean_step_loss": float(values.mean()),
            "first_100_step_mean_loss": float(values[:100].mean()),
            "last_100_step_mean_loss": float(values[-100:].mean()), "step_losses": values.tolist(),
            "loss": "expected_individual_expert_endpoint20_angular_loss_degrees",
            "checkpoint_selection": "fixed_final_step_no_development_selection",
            "matched_seed_training_elapsed_seconds": elapsed, "actual_tensor_execution": actual_device_flags[variant_index],
        })
    return models, fits, global_indexes


def _evaluate_partition(cache, partition, models, scalers, error_group, diagnostic_group, *, device,
                        batch_size=_EVALUATION_BATCH, variants=VARIANTS):
    """All windows, actual emitted float32 endpoint20, in sealed cache order."""
    from .selector_metrics import cache_row_order_sha256, method_name
    if partition not in _PARTITIONS or batch_size < 1:
        raise PermissionError("selector evaluation permits train/development only")
    variants = _variant_selection(variants)
    expected_methods = {method_name(variant, seed) for variant in variants for seed in SEEDS}
    if set(models) != expected_methods:
        raise ValueError("selector evaluation requires every fixed variant/seed checkpoint")
    groups = _groups(cache, partition)
    total = sum(len(group["origin_index"]) for *_, group in groups)
    row_hash = cache_row_order_sha256(cache, partition)
    error_group.attrs["row_order_sha256"] = diagnostic_group.attrs["row_order_sha256"] = row_hash
    all_methods = sorted(expected_methods | {method_name(variant) for variant in variants})
    for name in all_methods:
        error_group.create_dataset(name, shape=(total,), dtype="f8", chunks=True, compression="lzf")
    for name in sorted(expected_methods):
        diagnostics = diagnostic_group.create_group(name)
        for field in ("expected_loss_deg", "hard_error_deg"):
            diagnostics.create_dataset(field, shape=(total,), dtype="f8", chunks=True, compression="lzf")
        diagnostics.create_dataset("chosen_expert", shape=(total,), dtype="u1", chunks=True, compression="lzf")
    sums, counts, position = {name: {} for name in all_methods}, {}, 0
    for user, _, _, group in groups:
        count = len(group["origin_index"])
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            destination = slice(position + start, position + stop)
            current, target = group["current_q"][start:stop], group["target_q"][start:stop]
            velocities, expert_loss = group["velocities"][start:stop], group["expert_loss"][start:stop]
            for variant in variants:
                normalized = normalise_features(group[f"features/{variant}"][start:stop], scalers[variant])
                tensor = torch.as_tensor(normalized, device=device)
                seed_errors = []
                for seed in SEEDS:
                    name = method_name(variant, seed)
                    model = models[name]
                    model.eval()
                    with torch.inference_mode():
                        weights = model(tensor).cpu().numpy()
                    # Compute precisely the emitted endpoint without allocating
                    # the other 19 outputs during this endpoint-only metric.
                    # Full-interface parity is tested against the core wrapper.
                    endpoint = predict_endpoint(current, velocities, weights)
                    if endpoint.dtype != np.float32 or endpoint.shape != (stop - start, 4):
                        raise ValueError("selector metric endpoint must be float32 [B,4]")
                    angles = geodesic_deg(endpoint, target)
                    if not np.isfinite(angles).all() or np.any(angles < 0) or np.any(angles > 180 + 1e-6):
                        raise ValueError("invalid actual emitted selector endpoint20 angular errors")
                    error_group[name][destination] = angles
                    sums[name][user] = sums[name].get(user, 0.0) + float(angles.sum(dtype=np.float64))
                    seed_errors.append(angles)
                    chosen = np.argmax(weights, axis=1)
                    diagnostic_group[name]["expected_loss_deg"][destination] = np.sum(
                        weights.astype(np.float64) * expert_loss, axis=1)
                    diagnostic_group[name]["hard_error_deg"][destination] = expert_loss[np.arange(len(chosen)), chosen]
                    diagnostic_group[name]["chosen_expert"][destination] = chosen.astype(np.uint8)
                mean_name = method_name(variant)
                mean_angles = np.mean(seed_errors, axis=0)
                error_group[mean_name][destination] = mean_angles
                sums[mean_name][user] = sums[mean_name].get(user, 0.0) + float(mean_angles.sum(dtype=np.float64))
        counts[user] = counts.get(user, 0) + count
        position += count
        print(json.dumps({"event": "selector_evaluation", "partition": partition,
                          "windows_done": position, "windows_total": total}), flush=True)
    if position != total or sum(counts.values()) != total:
        raise ValueError("selector errors failed full stride1 cache coverage")
    _synchronize(device)
    return sums, counts


def _source_bindings():
    directory = Path(__file__).parent
    return {name: sha256_file(directory / name) for name in (
        "selector.py", "selector_experiment.py", "selector_metrics.py", "stage.py",
        "evidence.py", "geometry.py", "evaluation.py",
    )}


def _check_bundle_arguments(stage, baseline_dir, config, config_path):
    config_path, baseline_dir = Path(config_path), Path(baseline_dir)
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    validate_selector_config(on_disk)
    if on_disk != config:
        raise ValueError("selector arguments differ from the saved prospective config")
    receipt = validate_feature_bundle(stage, baseline_dir, config_path=config["baseline_config"])
    bindings = {
        "baseline_feature_cache_sha256": receipt["cache_sha256"],
        "baseline_scalers_sha256": receipt["scalers_sha256"],
        "source_hdf5_sha256": stage.hdf5_sha256, "split_sha256": stage.split_sha256,
    }
    if any(config[key] != value for key, value in bindings.items()):
        raise ValueError("selector config does not match the sealed source/split/cache/scaler bindings")
    # Static-mixture errors and private sufficient statistics remain immutable
    # references. Their complete alignment is independently checked by metrics.
    for filename in ("analytic_errors.h5", "analytic_user_statistics.json"):
        if not (baseline_dir / filename).is_file():
            raise FileNotFoundError(f"selector study requires the original {filename}")
    return receipt, {
        **bindings, "selector_config_sha256": sha256_file(config_path),
        "feature_receipt_sha256": sha256_file(baseline_dir / "feature_receipt.json"),
        "baseline_experiment_plan_sha256": sha256_file(baseline_dir / "experiment_plan.json"),
        "baseline_analytic_errors_sha256": sha256_file(baseline_dir / "analytic_errors.h5"),
        "baseline_analytic_user_statistics_sha256": sha256_file(baseline_dir / "analytic_user_statistics.json"),
    }


def _save_checkpoint(model, path, *, variant, seed, steps, bindings):
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    expected_shapes = {"hidden.weight": (16, 10), "hidden.bias": (16,),
                       "output.weight": (4, 16), "output.bias": (4,)}
    if (set(state) != set(expected_shapes) or sum(value.numel() for value in state.values()) != PARAMETERS
            or any(tuple(value.shape) != expected_shapes[name] or value.dtype != torch.float32
                   or not torch.isfinite(value).all().item() for name, value in state.items())):
        raise ValueError("cannot save an invalid/nonfinite selector state_dict")
    torch.save({"state_dict": state, "architecture": list(ARCHITECTURE), "parameters": PARAMETERS,
                "dense_MACs": DENSE_MACS, "variant": variant, "seed": seed,
                "optimizer_steps": steps, "checkpoint_selection": "fixed_final_step_no_development_selection",
                "loss": "expected_individual_expert_endpoint20_angular_loss", "bindings": bindings}, path)
    return sha256_file(path)


def run_selector_study(stage, baseline_dir, output_dir, config, *, device="cuda",
                       config_path="configs/selector_study.json"):
    """Fit the frozen six-variant/two-seed CUDA study and return aggregates.

    ``output_dir`` is an exclusive private experiment directory, normally under
    ignored ``data/selector_runs``. It contains all checkpoints, origin draws,
    per-window errors and participant statistics. Only the returned summary is
    suitable for aggregate public review. No final-partition option exists.
    """
    from .selector_metrics import method_name, summarize_selector_partition
    validate_selector_config(config)
    baseline_dir, output_dir = Path(baseline_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("selector study already exists; choose a new private run directory")
    feature_receipt, bindings = _check_bundle_arguments(stage, baseline_dir, config, config_path)
    device, runtime = _cuda_runtime(device)
    sources = _source_bindings()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    write_json(output_dir / "experiment_plan.json", {"config": config, "bindings": bindings,
                                                    "implementation_sources_sha256": sources,
                                                    "created_utc": utc_now()})
    started = time.perf_counter()
    settings, uncertainty = config["selector"], config["uncertainty"]
    scalers = json.loads((baseline_dir / "scalers.json").read_text(encoding="utf-8"))["variants"]
    baseline_private = json.loads((baseline_dir / "analytic_user_statistics.json").read_text(encoding="utf-8"))
    fits, checkpoints, public, private, source_checks = [], {}, {}, {}, []
    error_path, diagnostic_path = output_dir / "errors.h5", output_dir / "diagnostics.h5"
    with h5py.File(baseline_dir / "features.h5", "r") as cache:
        counts = validate_cache_binding(cache, stage)
        source_checks.append(_validate_source_partition(cache, stage, "train"))
        _synchronize(device)
        transfer_started = time.perf_counter()
        features, labels = _load_training_arrays(cache, scalers, device)
        preload_seconds = time.perf_counter() - transfer_started
        models = {}
        for seed in settings["seeds"]:
            seed_models, seed_fits, indexes = _train_matched_seed(
                features, labels, seed=seed, steps=settings["steps"], batch_size=settings["batch_size"],
                learning_rate=settings["learning_rate"], device=device)
            np.save(output_dir / f"sampled_train_origins_seed_{seed}.npy", indexes, allow_pickle=False)
            models.update(seed_models)
            fits.extend(seed_fits)
            for variant in VARIANTS:
                name = method_name(variant, seed)
                checkpoints[name] = _save_checkpoint(seed_models[name], checkpoint_dir / f"{name}.pt",
                                                     variant=variant, seed=seed, steps=settings["steps"], bindings=bindings)
        del features, labels
        _synchronize(device)
        # The only development source read happens after every checkpoint is
        # already fixed and saved; there is no development fitting/selection.
        source_checks.append(_validate_source_partition(cache, stage, "development"))
        with (h5py.File(error_path, "x") as errors, h5py.File(diagnostic_path, "x") as diagnostics,
              h5py.File(baseline_dir / "analytic_errors.h5", "r") as baseline_errors):
            for store in (errors, diagnostics):
                for key, value in bindings.items():
                    store.attrs[key] = value
                store.attrs["final_access_allowed"] = False
                store.attrs["row_order"] = "lexical_cache_groups_then_complete_stride1_origins"
            errors.attrs["metric"] = "actual_emitted_float32_endpoint20_geodesic_degrees"
            errors.attrs["two_seed_mean_interpretation"] = "arithmetic_mean_of_errors_not_prediction_ensemble"
            diagnostics.attrs["future_expert_losses_are_features"] = False
            for partition in _PARTITIONS:
                error_group, diagnostic_group = errors.create_group(partition), diagnostics.create_group(partition)
                sums, user_counts = _evaluate_partition(
                    cache, partition, models, scalers, error_group, diagnostic_group,
                    device=device, batch_size=config["evaluation_batch_size"])
                private[partition] = {"user_sums": sums, "user_counts": user_counts}
                reference = baseline_private[partition]
                public[partition] = summarize_selector_partition(
                    error_group, sums, user_counts, diagnostic_group, partition=partition,
                    resamples=uncertainty["bootstrap_resamples"], seed=uncertainty["seed"], cache=cache,
                    baseline_error_group=baseline_errors[partition],
                    baseline_user_sums=reference["user_sums"], baseline_user_counts=reference["user_counts"])
    _synchronize(device)
    runtime["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated(device)
    runtime["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved(device)
    runtime["peak_statistics_reset_before_training_preload"] = True
    audit = stage.verify_unchanged()
    _, final_bindings = _check_bundle_arguments(stage, baseline_dir, config, config_path)
    if final_bindings != bindings or _source_bindings() != sources:
        raise ValueError("selector config, baseline artifacts or implementation source changed during the run")
    summary = {
        "created_utc": utc_now(), "stage": "train_development_selector_only", "results": public,
        "architecture": list(ARCHITECTURE), "parameters": PARAMETERS, "dense_MACs": DENSE_MACS,
        "dense_MACs_scope": "selector_dense_layers_only_excludes_evidence_experts_geometry",
        "whole_graph_MACs": {"status": "unmeasured", "value": None},
        "evidence_length": config["evidence_length"], "baseline_code_commit": config["baseline_code_commit"],
        "variants": list(VARIANTS), "seeds": list(SEEDS), "windows": counts,
        "training": [{key: value for key, value in fit.items() if key != "step_losses"} for fit in fits],
        "input_scaling": {"fit_partition": "train", "existing_scalers_reused": True,
                          "window_count": counts["train"], "features": 10, "ddof": 0, "epsilon": 1e-8,
                          "arithmetic": "float64_subtraction_and_division_then_float32"},
        "evaluation_batch_size": config["evaluation_batch_size"],
        "source_checks": source_checks, "runtime": runtime, "preload_elapsed_seconds": preload_seconds,
        "two_seed_mean_interpretation": "arithmetic_mean_of_endpoint20_errors_not_prediction_ensemble",
        "checkpoint_selection": "fixed_final_step_no_development_selection",
        "development_interpretation": "exploratory_only_five_users_no_confirmatory_final_claim",
        "final_holdout_evaluated": False, "bindings": bindings, "implementation_sources_sha256": sources,
        "feature_receipt_schema_version": feature_receipt["schema_version"],
        "elapsed_seconds": time.perf_counter() - started, "audit": audit,
    }
    write_json(output_dir / "private_user_stats.json", private)
    write_json(output_dir / "training_receipts.json", {"fits": fits})
    write_json(output_dir / "summary.json", summary)
    receipt = {"created_utc": utc_now(), "bindings": bindings, "implementation_sources_sha256": sources,
               "checkpoint_sha256": checkpoints, "audit": audit,
               "artifact_sha256": {name: sha256_file(output_dir / name) for name in (
                   "experiment_plan.json", "errors.h5", "diagnostics.h5", "private_user_stats.json",
                   "training_receipts.json", "summary.json",
                   *(f"sampled_train_origins_seed_{seed}.npy" for seed in SEEDS))}}
    write_json(output_dir / "receipt.json", receipt)
    return summary

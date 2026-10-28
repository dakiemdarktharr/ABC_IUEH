"""Prospective R128/R256 children of the immutable R80 selector study.

Only guarded train/development reads are available. Each child has independent
train-only scalers, local datasets and a sealed receipt; the six-variant R80
cache validator is deliberately never applied to a two-variant child cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
from numbers import Real
from pathlib import Path
import re
import time

import h5py
import numpy as np
import torch

from .evidence import EXPERT_NAMES, build_segment_features
from . import selector_experiment as experiment
from .selector import ARCHITECTURE, DENSE_MACS, PARAMETERS, SEEDS
from .selector_metrics import cache_row_order_sha256, method_name, summarize_selector_partition
from .sources import sha256_file, utc_now, write_json
from .stage import PopulationMoments, cache_groups


VARIANTS = ("h1_a0", "h20_a0")
EVIDENCE_LENGTHS = (128, 256)
_PARTITIONS = ("train", "development")
_CHUNK = 65536
_ROOT = Path(__file__).resolve().parents[2]
_PLAN = _ROOT / "docs/selector_sensitivity_plan.md"
_SHARED = {"origin_index": ((), "int64"), "current_q": ((4,), "float32"),
           "target_q": ((4,), "float32"), "velocities": ((4, 3), "float64"),
           "expert_loss": ((4,), "float64")}
_FROZEN = {
    "schema_version": 1, "stage": "train_development_selector_R_sensitivity_only",
    "partitions": list(_PARTITIONS), "evidence_lengths": list(EVIDENCE_LENGTHS),
    "variants": list(VARIANTS), "context_samples": 400, "horizon_samples": 20, "stride": 1,
    "selector": copy.deepcopy(experiment._FROZEN_CONFIG["selector"]),
    "feature_standardization": "independent_train_only_window_weighted_population_mean_std_epsilon_1e-8_float64_then_float32",
    "uncertainty": copy.deepcopy(experiment._FROZEN_CONFIG["uncertainty"]),
    "success_thresholds_unchanged": copy.deepcopy(experiment._FROZEN_CONFIG["success_thresholds_unchanged"]),
    "primary_contrast": "h20_a0_minus_h1_a0_two_seed_mean_errors",
    "evaluation_batch_size": 65536, "device_required": "cuda",
    "precision": "float32_no_autocast_no_TF32", "deterministic_algorithms": True,
    "R_selection_allowed": False, "final_access_allowed": False,
    "model_or_feature_artifacts_public": False, "cloud_approved_cost": 0,
    "paper_scope": "exploratory_train_development_study_with_sealed_final_holdout",
}


def _read(path):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid sensitivity JSON: {Path(path).name}") from error
    if not isinstance(result, dict):
        raise ValueError("sensitivity JSON must be an object")
    return result


def _same(actual, expected):
    """Compare frozen JSON recursively, rejecting booleans posing as numbers."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_same(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, float):
        return type(actual) in (int, float) and np.isfinite(actual) and actual == expected
    return type(actual) is type(expected) and actual == expected


def validate_sensitivity_config(config):
    """Freeze both R values, the complete compute budget and sealed-final scope."""
    if (not isinstance(config, dict) or set(config) != set(_FROZEN) | {"parent_main_config_sha256"}
            or not _same({key: config[key] for key in _FROZEN}, _FROZEN)
            or not isinstance(config["parent_main_config_sha256"], str)
            or re.fullmatch("[0-9a-f]{64}", config["parent_main_config_sha256"]) is None):
        raise ValueError("unsupported frozen sensitivity config")
    return config


def _sources():
    return {**experiment._source_bindings(), "selector_sensitivity.py": sha256_file(__file__)}


def _context(stage, baseline_dir, config, config_path, main_config_path):
    validate_sensitivity_config(config)
    if not _same(_read(config_path), config):
        raise ValueError("saved sensitivity config differs from supplied configuration")
    if sha256_file(main_config_path) != config["parent_main_config_sha256"]:
        raise ValueError("parent main selector config SHA-256 changed")
    main_config = experiment.validate_selector_config(_read(main_config_path))
    for key in ("selector", "uncertainty", "success_thresholds_unchanged", "evaluation_batch_size"):
        if not _same(config[key], main_config[key]):
            raise ValueError("sensitivity budget/thresholds differ from the parent main study")
    parent_receipt, parent = experiment._check_bundle_arguments(stage, baseline_dir, main_config, main_config_path)
    return main_config, parent_receipt, {
        **parent, "sensitivity_config_sha256": sha256_file(config_path),
        "sensitivity_plan_sha256": sha256_file(_PLAN),
    }


def _R(value):
    if type(value) is not int or value not in EVIDENCE_LENGTHS:
        raise ValueError("sensitivity evidence length must be prospectively fixed R128 or R256")


def _local_group(parent, name):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink) or not isinstance(parent[name], h5py.Group):
        raise ValueError("sensitivity groups must be local hard links")
    return parent[name]


def _local_dataset(parent, name, shape, dtype):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
        raise ValueError("sensitivity datasets must be local hard links")
    data = parent[name]
    if (not isinstance(data, h5py.Dataset) or data.shape != shape or data.dtype != np.dtype(dtype)
            or data.is_virtual or data.external):
        raise ValueError("sensitivity dataset shape/dtype/storage differs from the frozen schema")
    return data


def validate_sensitivity_cache_binding(cache, parent_cache, stage, evidence_length):
    """Validate exact child coverage/schema/finite values against the sealed R80.

    Parent validation is a caller prerequisite. Source quaternion arrays are
    never indexed here. Every shared value and lexical row order must equal the
    already sealed parent; feature matrices alone may depend on R.
    """
    _R(evidence_length)
    if (set(cache) != set(_PARTITIONS) or cache.attrs.get("source_hdf5_sha256") != stage.hdf5_sha256
            or cache.attrs.get("split_sha256") != stage.split_sha256):
        raise ValueError("sensitivity cache is not bound to train/development source/split")
    for name, expected in (("evidence_length", evidence_length), ("context_samples", 400), ("horizon_samples", 20)):
        actual = cache.attrs.get(name)
        if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, np.integer)) or actual != expected:
            raise ValueError(f"invalid sensitivity cache {name}")
    if not isinstance(cache.attrs.get("final_access_allowed"), (bool, np.bool_)) or cache.attrs["final_access_allowed"]:
        raise ValueError("sensitivity cache must forbid final access")
    try:
        experts = json.loads(cache.attrs["expert_names_json"])
        recorded_counts = json.loads(cache.attrs["window_counts_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid sensitivity cache metadata") from error
    if experts != list(EXPERT_NAMES):
        raise ValueError("sensitivity expert order differs from parent")
    counts = {partition: 0 for partition in _PARTITIONS}
    for partition in _PARTITIONS:
        child_partition = _local_group(cache, partition)
        if set(child_partition) != set(stage.partitions[partition]) or set(child_partition) != set(parent_cache[partition]):
            raise ValueError("sensitivity users differ from the locked partition")
        for user in child_partition:
            user_group = _local_group(child_partition, user)
            if set(user_group) != set(parent_cache[partition][user]):
                raise ValueError("sensitivity sessions differ from the sealed parent")
            for session in user_group:
                session_group = _local_group(user_group, session)
                if set(session_group) != set(parent_cache[partition][user][session]):
                    raise ValueError("sensitivity segments differ from the sealed parent")
                for segment in session_group:
                    group = _local_group(session_group, segment)
                    parent = parent_cache[partition][user][session][segment]
                    count = len(parent["origin_index"])
                    if not count or set(group) != set(_SHARED) | {"features"}:
                        raise ValueError("unexpected sensitivity cache segment schema")
                    datasets = {name: _local_dataset(group, name, (count, *tail), dtype)
                                for name, (tail, dtype) in _SHARED.items()}
                    features = _local_group(group, "features")
                    if set(features) != set(VARIANTS):
                        raise ValueError("sensitivity features must contain only the declared pair")
                    for variant in VARIANTS:
                        _local_dataset(features, variant, (count, 10), "float32")
                    for start in range(0, count, _CHUNK):
                        stop = min(count, start + _CHUNK)
                        origins = np.arange(399 + start, 399 + stop, dtype=np.int64)
                        if not np.array_equal(datasets["origin_index"][start:stop], origins):
                            raise ValueError("sensitivity origins must preserve every stride1 parent origin")
                        for name, data in datasets.items():
                            values = data[start:stop]
                            if not np.isfinite(values).all() or not np.array_equal(values, parent[name][start:stop]):
                                raise ValueError("sensitivity shared values differ from the sealed parent")
                        for variant in VARIANTS:
                            if not np.isfinite(features[variant][start:stop]).all():
                                raise ValueError("sensitivity features must be finite")
                    counts[partition] += count
        if cache_row_order_sha256(cache, partition) != cache_row_order_sha256(parent_cache, partition):
            raise ValueError("sensitivity row order differs from the R80 parent")
    if (not isinstance(recorded_counts, dict) or set(recorded_counts) != set(_PARTITIONS)
            or any(type(v) is not int or v < 1 for v in recorded_counts.values()) or counts != recorded_counts):
        raise ValueError("sensitivity window counts differ from datasets")
    return counts


def _validate_scalers(scalers, stage, counts, evidence_length, cache):
    if (set(scalers) != {"created_utc", "fit_partition", "source_hdf5_sha256", "split_sha256", "evidence_length", "variants"}
            or scalers["fit_partition"] != "train" or scalers["source_hdf5_sha256"] != stage.hdf5_sha256
            or scalers["split_sha256"] != stage.split_sha256 or type(scalers["evidence_length"]) is not int
            or scalers["evidence_length"] != evidence_length or not isinstance(scalers["variants"], dict)
            or set(scalers["variants"]) != set(VARIANTS)):
        raise ValueError("sensitivity scalers differ from independent train-only fitting")
    for variant, state in scalers["variants"].items():
        moments = PopulationMoments()
        for *_, group in cache_groups(cache, "train"):
            moments.update(group[f"features/{variant}"][:])
        expected = moments.as_dict()
        if expected["count"] != counts["train"] or not _same(state, expected):
            raise ValueError("sensitivity scaler does not equal the train-only population moments")


def build_sensitivity_feature_cache(stage, baseline_dir, output_dir, evidence_length, config, *,
                                    config_path="configs/selector_sensitivity.json",
                                    main_config_path="configs/selector_study.json"):
    """Create an exclusive, separately sealed two-variant child feature bundle.

    This may run before main fitting completes. Actual child fitting is a
    separate operation requiring the completed parent-main receipt.
    """
    _R(evidence_length)
    _, _, bindings = _context(stage, baseline_dir, config, config_path, main_config_path)
    output_dir, baseline_dir = Path(output_dir), Path(baseline_dir)
    if output_dir.exists():
        raise FileExistsError("sensitivity feature directory already exists; never overwrite/reseal")
    sources = _sources()
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = {"schema_version": 1, "created_utc": utc_now(), "config": config, "bindings": bindings,
            "evidence_length": evidence_length, "implementation_sources_sha256": sources,
            "parent_evidence_length": 80, "final_access_allowed": False}
    write_json(output_dir / "experiment_plan.json", plan)
    temporary, output = output_dir / "features.h5.part", output_dir / "features.h5"
    counts, moments = {p: 0 for p in _PARTITIONS}, {v: PopulationMoments() for v in VARIANTS}
    started = time.perf_counter()
    try:
        with h5py.File(baseline_dir / "features.h5", "r") as parent, h5py.File(temporary, "x") as cache:
            for key, value in (("source_hdf5_sha256", stage.hdf5_sha256), ("split_sha256", stage.split_sha256),
                               ("evidence_length", evidence_length), ("context_samples", 400), ("horizon_samples", 20),
                               ("expert_names_json", json.dumps(EXPERT_NAMES)), ("final_access_allowed", False)):
                cache.attrs[key] = value
            for partition in _PARTITIONS:
                seen = set()
                for user, session, segment, q in stage.iter_segments(partition):
                    key = user, session, segment
                    if (user not in stage.partitions[partition] or user in stage.partitions["final"] or key in seen
                            or q.flags.writeable or q.dtype != np.float32 or q.ndim != 2 or q.shape[1] != 4):
                        raise ValueError("invalid guarded sensitivity source segment")
                    seen.add(key)
                    data = build_segment_features(q, evidence_length=evidence_length, include_complete_horizon=False)
                    count = len(data["origin_index"])
                    if not count:
                        continue
                    path = f"{partition}/{user}/{session}/{segment}"
                    if path not in parent:
                        raise ValueError("guarded sensitivity segment is absent from parent cache")
                    group = cache.create_group(path)
                    for name in _SHARED:
                        if not np.array_equal(data[name], parent[path][name][:]):
                            raise ValueError("new sensitivity arrays differ from shared R80 origins/labels/predictions")
                        group.create_dataset(name, data=data[name], chunks=True, compression="lzf")
                    for variant in VARIANTS:
                        x = data["features"][variant]
                        if x.shape != (count, 10) or x.dtype != np.float32 or not np.isfinite(x).all():
                            raise ValueError("invalid sensitivity causal feature matrix")
                        group.create_dataset(f"features/{variant}", data=x, chunks=True, compression="lzf")
                    counts[partition] += count
                    print(json.dumps({"event": "sensitivity_features_segment", "R": evidence_length,
                                      "partition": partition, "windows": count,
                                      "cumulative_windows": counts[partition]}), flush=True)
                # Match scaler accumulation to the canonical lexical cache order.
                if partition == "train":
                    for *_, group in cache_groups(cache, "train"):
                        for variant in VARIANTS:
                            moments[variant].update(group[f"features/{variant}"][:])
            cache.attrs["window_counts_json"] = json.dumps(counts)
            validate_sensitivity_cache_binding(cache, parent, stage, evidence_length)
        audit = stage.verify_unchanged()
        _, _, after = _context(stage, baseline_dir, config, config_path, main_config_path)
        if bindings != after or sources != _sources():
            raise ValueError("sensitivity parent/config/source/plan changed during feature construction")
        scalers = {"created_utc": utc_now(), "fit_partition": "train", "source_hdf5_sha256": stage.hdf5_sha256,
                   "split_sha256": stage.split_sha256, "evidence_length": evidence_length,
                   "variants": {v: moments[v].as_dict() for v in VARIANTS}}
        temporary.replace(output)
        write_json(output_dir / "scalers.json", scalers)
        receipt = {"schema_version": 1, "sealed_utc": utc_now(), "evidence_length": evidence_length,
                   "variants": list(VARIANTS), "windows": counts, "bindings": bindings,
                   "implementation_sources_sha256": sources, "cache_bytes": output.stat().st_size,
                   "elapsed_seconds": time.perf_counter() - started, "audit": audit,
                   "artifact_sha256": {name: sha256_file(output_dir / name)
                                       for name in ("features.h5", "scalers.json", "experiment_plan.json")}}
        write_json(output_dir / "feature_receipt.json", receipt)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return validate_sensitivity_feature_bundle(stage, baseline_dir, output_dir, evidence_length, config,
                                                config_path=config_path, main_config_path=main_config_path)


def validate_sensitivity_feature_bundle(stage, baseline_dir, output_dir, evidence_length, config, *,
                                         config_path="configs/selector_sensitivity.json",
                                         main_config_path="configs/selector_study.json"):
    """Revalidate hashes, schema, coverage, bindings and train-only moments."""
    _R(evidence_length)
    _, _, bindings = _context(stage, baseline_dir, config, config_path, main_config_path)
    directory = Path(output_dir)
    receipt, plan = _read(directory / "feature_receipt.json"), _read(directory / "experiment_plan.json")
    expected_keys = {"schema_version", "sealed_utc", "evidence_length", "variants", "windows", "bindings",
                     "implementation_sources_sha256", "cache_bytes", "elapsed_seconds", "audit", "artifact_sha256"}
    expected_artifacts = {"features.h5", "scalers.json", "experiment_plan.json"}
    if (set(receipt) != expected_keys or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or type(receipt["evidence_length"]) is not int or receipt["evidence_length"] != evidence_length
            or receipt["variants"] != list(VARIANTS) or receipt["bindings"] != bindings
            or receipt["implementation_sources_sha256"] != _sources()
            or not isinstance(receipt["sealed_utc"], str) or not receipt["sealed_utc"]
            or type(receipt["cache_bytes"]) is not int or receipt["cache_bytes"] != (directory / "features.h5").stat().st_size
            or isinstance(receipt["elapsed_seconds"], bool) or not isinstance(receipt["elapsed_seconds"], Real)
            or not np.isfinite(receipt["elapsed_seconds"]) or receipt["elapsed_seconds"] < 0
            or not isinstance(receipt["artifact_sha256"], dict) or set(receipt["artifact_sha256"]) != expected_artifacts):
        raise ValueError("invalid sealed sensitivity feature receipt schema/bindings")
    for name, expected in receipt["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"sealed sensitivity feature SHA-256 mismatch: {name}")
    if (set(plan) != {"schema_version", "created_utc", "config", "bindings", "evidence_length",
                      "implementation_sources_sha256", "parent_evidence_length", "final_access_allowed"}
            or plan["schema_version"] != 1 or not _same(plan["config"], config) or plan["bindings"] != bindings
            or type(plan["evidence_length"]) is not int or plan["evidence_length"] != evidence_length
            or plan["implementation_sources_sha256"] != _sources() or type(plan["parent_evidence_length"]) is not int
            or plan["parent_evidence_length"] != 80 or plan["final_access_allowed"] is not False):
        raise ValueError("sensitivity feature plan differs from the frozen child contract")
    with h5py.File(directory / "features.h5", "r") as cache, h5py.File(Path(baseline_dir) / "features.h5", "r") as parent:
        counts = validate_sensitivity_cache_binding(cache, parent, stage, evidence_length)
        _validate_scalers(_read(directory / "scalers.json"), stage, counts, evidence_length, cache)
        groups = {p: sum(1 for _ in cache_groups(cache, p)) for p in _PARTITIONS}
    audit = receipt["audit"]
    if (counts != receipt["windows"] or not isinstance(audit, dict)
            or audit.get("source_hdf5_sha256") != stage.hdf5_sha256 or audit.get("split_sha256") != stage.split_sha256
            or audit.get("data_and_split_unchanged") is not True or type(audit.get("final_quaternion_datasets_read")) is not int
            or audit["final_quaternion_datasets_read"] != 0 or type(audit.get("accessed_user_count")) is not int
            or audit["accessed_user_count"] != sum(len(stage.partitions[p]) for p in _PARTITIONS)
            or not isinstance(audit.get("segment_reads"), dict) or set(audit["segment_reads"]) != set(_PARTITIONS)
            or any(type(audit["segment_reads"][p]) is not int or audit["segment_reads"][p] < groups[p] for p in _PARTITIONS)):
        raise ValueError("invalid sensitivity feature counts/train-development audit")
    stage.verify_unchanged()
    return receipt


def _completed_main(stage, baseline_dir, main_dir, config, config_path, main_config_path):
    """Require immutable completed main artifacts before any child GPU fitting."""
    main_config, parent_features, bindings = _context(stage, baseline_dir, config, config_path, main_config_path)
    directory = Path(main_dir)
    receipt, summary, plan = (_read(directory / name) for name in ("receipt.json", "summary.json", "experiment_plan.json"))
    main_bindings = {key: value for key, value in bindings.items() if not key.startswith("sensitivity_")}
    main_sources = experiment._source_bindings()
    if (receipt.get("bindings") != main_bindings or receipt.get("implementation_sources_sha256") != main_sources
            or summary.get("bindings") != main_bindings or summary.get("implementation_sources_sha256") != main_sources
            or summary.get("final_holdout_evaluated") is not False or summary.get("evidence_length") != 80
            or summary.get("variants") != list(experiment.VARIANTS) or summary.get("seeds") != list(SEEDS)
            or not _same(summary.get("windows"), parent_features["windows"])
            or plan.get("config") != main_config or plan.get("bindings") != main_bindings
            or plan.get("implementation_sources_sha256") != main_sources):
        raise ValueError("completed R80 main receipt/config/source bindings differ")
    required = {"experiment_plan.json", "errors.h5", "diagnostics.h5", "private_user_stats.json",
                "training_receipts.json", "summary.json", *(f"sampled_train_origins_seed_{seed}.npy" for seed in SEEDS)}
    if not isinstance(receipt.get("artifact_sha256"), dict) or set(receipt["artifact_sha256"]) != required:
        raise ValueError("completed R80 receipt omits required artifacts")
    for name, digest in receipt["artifact_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError("completed R80 main artifact SHA-256 changed")
    expected_methods = {method_name(v, s) for v in experiment.VARIANTS for s in SEEDS}
    if not isinstance(receipt.get("checkpoint_sha256"), dict) or set(receipt["checkpoint_sha256"]) != expected_methods:
        raise ValueError("completed R80 receipt omits checkpoints")
    for name, digest in receipt["checkpoint_sha256"].items():
        if sha256_file(directory / "checkpoints" / f"{name}.pt") != digest:
            raise ValueError("completed R80 checkpoint SHA-256 changed")
    fits = _read(directory / "training_receipts.json").get("fits")
    if (not isinstance(fits, list) or len(fits) != len(expected_methods)
            or any(not isinstance(f, dict) for f in fits) or {f.get("method") for f in fits} != expected_methods):
        raise ValueError("completed R80 receipt omits matched fits")
    matches = {}
    for seed in SEEDS:
        seed_fits = [fit for fit in fits if fit.get("seed") == seed]
        if len(seed_fits) != len(experiment.VARIANTS):
            raise ValueError("completed R80 receipt has incomplete fixed seed coverage")
        arrays = np.load(directory / f"sampled_train_origins_seed_{seed}.npy", allow_pickle=False)
        expected = np.random.default_rng(seed).integers(0, summary["windows"]["train"], size=(2000, 128), dtype=np.int64)
        if arrays.dtype != np.int64 or not np.array_equal(arrays, expected):
            raise ValueError("completed R80 sampled origins differ from the fixed global schedule")
        digest = hashlib.sha256(expected.astype("<i8", copy=False).tobytes()).hexdigest()
        hashes = {fit.get("initial_state_sha256") for fit in seed_fits}
        if len(hashes) != 1:
            raise ValueError("completed R80 fits do not share seed initialization")
        for fit in seed_fits:
            _validate_fit(fit, seed, summary["windows"]["train"], digest, next(iter(hashes)))
        matches[seed] = {"global_origin_batches_sha256": digest, "initial_state_sha256": next(iter(hashes))}
    audit = receipt.get("audit", {})
    if audit.get("final_quaternion_datasets_read") != 0 or audit.get("data_and_split_unchanged") is not True:
        raise ValueError("completed R80 receipt has no sealed-final audit")
    return {"main_receipt_sha256": sha256_file(directory / "receipt.json"),
            "main_summary_sha256": sha256_file(directory / "summary.json"), **bindings}, matches


def _validate_fit(fit, seed, population, origin_hash, initial_hash):
    if not isinstance(fit, dict):
        raise ValueError("fit receipt must be an object")
    for digest in (origin_hash, initial_hash, fit.get("final_state_sha256")):
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("fit receipt requires valid state and origin SHA-256 bindings")
    expected = {"seed": seed, "first_step": 1, "last_step": 2000, "optimizer_steps": 2000,
                "batch_size": 128, "sampled_training_windows": 256000, "population_training_windows": population,
                "sampling_seed": seed, "optimizer": "SGD", "learning_rate": .001, "momentum": 0,
                "weight_decay": 0, "global_origin_batches_sha256": origin_hash, "initial_state_sha256": initial_hash,
                "sampling": "uniform_global_train_origins_with_replacement",
                "loss": "expected_individual_expert_endpoint20_angular_loss_degrees",
                "checkpoint_selection": "fixed_final_step_no_development_selection"}
    if any(not _same(fit.get(key), value) for key, value in expected.items()):
        raise ValueError("fit receipt does not truthfully match the fixed 2000-step R80 budget/origins/initialization")
    losses = np.asarray(fit.get("step_losses"), dtype=np.float64)
    if losses.shape != (2000,) or not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("fit receipt must retain all actual 2000 finite SGD losses")


def _run_child(stage, baseline_dir, main_dir, feature_dir, config, *, device, runtime,
               config_path, main_config_path, main_bindings, matches):
    """Run only the strict public recipe; synthetic short fits cannot be sealed."""
    current, current_matches = _completed_main(stage, baseline_dir, main_dir, config, config_path, main_config_path)
    if current != main_bindings or current_matches != matches:
        raise ValueError("immutable R80 main bindings changed before sensitivity GPU fitting")
    evidence_length = _read(feature_dir / "feature_receipt.json")["evidence_length"]
    feature_receipt = validate_sensitivity_feature_bundle(stage, baseline_dir, feature_dir, evidence_length, config,
                                                          config_path=config_path, main_config_path=main_config_path)
    bindings = {**main_bindings, "child_feature_receipt_sha256": sha256_file(feature_dir / "feature_receipt.json"),
                "child_feature_cache_sha256": feature_receipt["artifact_sha256"]["features.h5"],
                "child_scalers_sha256": feature_receipt["artifact_sha256"]["scalers.json"]}
    output = feature_dir / "study"
    output.mkdir(exist_ok=False)
    (output / "checkpoints").mkdir()
    sources = _sources()
    write_json(output / "experiment_plan.json", {"created_utc": utc_now(), "config": config, "bindings": bindings,
                                                 "evidence_length": evidence_length, "implementation_sources_sha256": sources})
    scalers = _read(feature_dir / "scalers.json")["variants"]
    baseline_stats = _read(Path(baseline_dir) / "analytic_user_statistics.json")
    fits, checkpoints, public, private, models = [], {}, {}, {}, {}
    settings, uncertainty = config["selector"], config["uncertainty"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with h5py.File(feature_dir / "features.h5", "r") as cache:
        counts = feature_receipt["windows"]
        transfer_started = time.perf_counter()
        features, labels = experiment._load_training_arrays(cache, scalers, device, variants=VARIANTS)
        preload_seconds = time.perf_counter() - transfer_started
        for seed in SEEDS:
            seed_models, seed_fits, indexes = experiment._train_matched_seed(
                features, labels, seed=seed, steps=settings["steps"], batch_size=settings["batch_size"],
                learning_rate=settings["learning_rate"], device=device, variants=VARIANTS)
            if len(seed_fits) != 2 or {f.get("method") for f in seed_fits} != {method_name(v, seed) for v in VARIANTS}:
                raise ValueError("sensitivity training did not return exactly the declared variant pair")
            digest = hashlib.sha256(indexes.astype("<i8", copy=False).tobytes()).hexdigest()
            if indexes.shape != (2000, 128) or indexes.dtype != np.int64 or digest != matches[seed]["global_origin_batches_sha256"]:
                raise ValueError("sensitivity global origins do not match the immutable R80 schedule")
            for fit in seed_fits:
                _validate_fit(fit, seed, counts["train"], digest, matches[seed]["initial_state_sha256"])
            np.save(output / f"sampled_train_origins_seed_{seed}.npy", indexes, allow_pickle=False)
            models.update(seed_models)
            fits.extend(seed_fits)
            for variant in VARIANTS:
                name = method_name(variant, seed)
                checkpoints[name] = experiment._save_checkpoint(seed_models[name], output / "checkpoints" / f"{name}.pt",
                    variant=variant, seed=seed, steps=settings["steps"], bindings=bindings)
        del features, labels
        with (h5py.File(output / "errors.h5", "x") as errors,
              h5py.File(output / "diagnostics.h5", "x") as diagnostics,
              h5py.File(Path(baseline_dir) / "analytic_errors.h5", "r") as baseline_errors):
            for store in (errors, diagnostics):
                for key, value in bindings.items():
                    store.attrs[key] = value
                store.attrs["evidence_length"] = evidence_length
                store.attrs["final_access_allowed"] = False
            errors.attrs["metric"] = "actual_emitted_float32_endpoint20_geodesic_degrees"
            diagnostics.attrs["future_expert_losses_are_features"] = False
            for partition in _PARTITIONS:
                eg, dg = errors.create_group(partition), diagnostics.create_group(partition)
                sums, user_counts = experiment._evaluate_partition(cache, partition, models, scalers, eg, dg,
                    device=device, batch_size=config["evaluation_batch_size"], variants=VARIANTS)
                private[partition] = {"user_sums": sums, "user_counts": user_counts}
                reference = baseline_stats[partition]
                public[partition] = summarize_selector_partition(eg, sums, user_counts, dg,
                    partition=partition, resamples=uncertainty["bootstrap_resamples"], seed=uncertainty["seed"],
                    cache=cache, baseline_error_group=baseline_errors[partition],
                    baseline_user_sums=reference["user_sums"], baseline_user_counts=reference["user_counts"], variants=VARIANTS)
    experiment._synchronize(device)
    runtime = {**runtime, "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
               "peak_statistics_reset_before_training_preload": True}
    audit = stage.verify_unchanged()
    after, after_matches = _completed_main(stage, baseline_dir, main_dir, config, config_path, main_config_path)
    validate_sensitivity_feature_bundle(stage, baseline_dir, feature_dir, evidence_length, config,
                                        config_path=config_path, main_config_path=main_config_path)
    if after != main_bindings or after_matches != matches or _sources() != sources:
        raise ValueError("sensitivity parent/config/source/plan changed during fitting")
    summary = {"created_utc": utc_now(), "stage": config["stage"], "evidence_length": evidence_length,
               "results": public, "variants": list(VARIANTS), "seeds": list(SEEDS), "windows": counts,
               "architecture": list(ARCHITECTURE), "parameters": PARAMETERS, "dense_MACs": DENSE_MACS,
               "dense_MACs_scope": "selector_dense_layers_only_excludes_evidence_experts_geometry",
               "training": [{k: v for k, v in fit.items() if k != "step_losses"} for fit in fits],
               "input_scaling": {"fit_partition": "train", "independently_fitted_for_this_R": True,
                                 "window_count": counts["train"], "features": 10, "ddof": 0, "epsilon": 1e-8},
               "runtime": runtime, "preload_elapsed_seconds": preload_seconds,
               "elapsed_seconds": time.perf_counter() - started, "audit": audit, "bindings": bindings,
               "implementation_sources_sha256": sources, "final_holdout_evaluated": False,
               "development_interpretation": "exploratory_only_five_users_no_confirmatory_final_claim",
               "R_selection_allowed": False, "checkpoint_selection": "fixed_final_step_no_development_selection",
               "two_seed_mean_interpretation": "arithmetic_mean_of_endpoint20_errors_not_prediction_ensemble"}
    write_json(output / "private_user_stats.json", private)
    write_json(output / "training_receipts.json", {"fits": fits})
    write_json(output / "summary.json", summary)
    write_json(output / "receipt.json", {"created_utc": utc_now(), "evidence_length": evidence_length,
        "bindings": bindings, "implementation_sources_sha256": sources, "checkpoint_sha256": checkpoints,
        "audit": audit, "artifact_sha256": {name: sha256_file(output / name) for name in (
            "experiment_plan.json", "errors.h5", "diagnostics.h5", "private_user_stats.json", "training_receipts.json",
            "summary.json", *(f"sampled_train_origins_seed_{seed}.npy" for seed in SEEDS))}})
    return summary


def run_sensitivity(stage, baseline_dir, main_study_dir, output_dir, config, *, device="cuda",
                    config_path="configs/selector_sensitivity.json", main_config_path="configs/selector_study.json"):
    """Run both predeclared R values, retaining every child and no best-R choice.

    All artifacts are private under an exclusive output directory. The return
    value contains aggregate metrics only. There is no final-partition option
    or CPU fallback. To prebuild caches, use the independent feature API; this
    entry point always creates a complete new two-child run.
    """
    main_bindings, matches = _completed_main(stage, baseline_dir, main_study_dir, config, config_path, main_config_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("sensitivity run already exists; choose an exclusive private path")
    device, runtime = experiment._cuda_runtime(device)
    sources = _sources()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "experiment_plan.json", {"created_utc": utc_now(), "config": config,
        "bindings": main_bindings, "implementation_sources_sha256": sources, "evidence_lengths": list(EVIDENCE_LENGTHS)})
    children = {}
    for evidence_length in EVIDENCE_LENGTHS:
        feature_dir = output_dir / f"R{evidence_length}"
        build_sensitivity_feature_cache(stage, baseline_dir, feature_dir, evidence_length, config,
                                       config_path=config_path, main_config_path=main_config_path)
        children[str(evidence_length)] = _run_child(stage, baseline_dir, main_study_dir, feature_dir, config,
            device=device, runtime=runtime, config_path=config_path, main_config_path=main_config_path,
            main_bindings=main_bindings, matches=matches)
    after, after_matches = _completed_main(stage, baseline_dir, main_study_dir, config, config_path, main_config_path)
    if after != main_bindings or after_matches != matches or sources != _sources():
        raise ValueError("sensitivity source/config/parent changed during the two-child run")
    summary = {"created_utc": utc_now(), "stage": config["stage"], "evidence_lengths": list(EVIDENCE_LENGTHS),
               "variants": list(VARIANTS), "results_by_R": children, "bindings": main_bindings,
               "implementation_sources_sha256": sources, "R_selection_allowed": False,
               "final_holdout_evaluated": False, "development_interpretation": "exploratory_only_five_users",
               "audit": stage.verify_unchanged()}
    write_json(output_dir / "summary.json", summary)
    artifacts = ["experiment_plan.json", "summary.json"]
    artifacts += [f"R{r}/{name}" for r in EVIDENCE_LENGTHS for name in ("feature_receipt.json", "study/receipt.json")]
    write_json(output_dir / "receipt.json", {"created_utc": utc_now(), "bindings": main_bindings,
        "implementation_sources_sha256": sources, "audit": summary["audit"],
        "artifact_sha256": {name: sha256_file(output_dir / name) for name in artifacts}})
    return summary

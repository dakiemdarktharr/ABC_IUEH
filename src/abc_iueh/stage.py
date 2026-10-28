"""Locked train/development access and private causal feature artifacts."""
from pathlib import Path
import hashlib
import json
import time
from numbers import Real

import h5py
import numpy as np

from .sources import sha256_file, utc_now, write_json


# This is an execution contract, not an organizer clarification. Changing the
# recorded plan requires an explicit implementation/review rather than silently
# recording options which the feature and baseline code does not implement.
_FROZEN_CONFIG = {
    "stage": "train_development_baselines_and_features_only",
    "seed": 20261026,
    "partitions": ["train", "development"],
    "context_samples": 400,
    "horizon_samples": 20,
    "stride": 1,
    "evidence_length": 80,
    "variants": {"h1_a0": [1, 0], "h20_a0": [20, 0], "h1_a19": [1, 19], "h20_a20": [20, 20]},
    "complete_horizon": True,
    "rank_temperature": 1.0,
    "feature_standardization": "train_only_window_weighted_population_mean_std_epsilon_1e-8",
    "static_mixture": {"simplex_step": 0.25, "max_calibration_windows": 65536, "fit_partition": "train",
                       "objective": "actual_endpoint20_mean_geodesic_degrees"},
    "uncertainty": {"cluster": "user", "bootstrap_resamples": 2000, "seed": 20261026,
                    "estimand": "window_weighted_ratio", "development_inference": "exploratory_only_five_users"},
    "neural_baseline": {"architecture": [1197, 64, 60], "parameters": 80572, "steps": 2000,
                        "batch_size": 128, "optimizer": "Adam", "learning_rate": 0.001,
                        "seeds": [20261026, 20261027],
                        "input_scaling": "three_axis_train_increment_population_mean_std",
                        "loss": "sign_aligned_quaternion_MSE_all_20_outputs", "early_stopping": False},
    "final_access_allowed": False,
    "selection_after_final_results_allowed": False,
    "model_or_feature_artifacts_public": False,
}
_FEATURE_VARIANTS = {"h1_a0", "h20_a0", "h1_a19", "h20_a20", "complete_horizon", "motion_only"}
_PARTITIONS = ("train", "development")
_RECEIPT_SCHEMA_VERSION = 1


def validate_experiment_config(config):
    """Reject unsupported/ignored options anywhere in the frozen stage plan."""
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
            raise ValueError(f"unsupported frozen experiment config at {path}")
        if isinstance(expected, dict):
            for key, value in expected.items():
                check(actual[key], value, f"{path}.{key}")
        elif isinstance(expected, list):
            for index, value in enumerate(expected):
                check(actual[index], value, f"{path}[{index}]")
    check(config, _FROZEN_CONFIG, "config")
    return config


class LockedDevelopmentStage:
    """This API has no final-partition opt-in."""

    def __init__(self, hdf5_path, split_path):
        self.hdf5_path, self.split_path = Path(hdf5_path), Path(split_path)
        split_bytes = self.split_path.read_bytes()
        self.split_sha256 = hashlib.sha256(split_bytes).hexdigest()
        manifest = json.loads(split_bytes)
        if (manifest.get("mode") != "locked_confirmatory" or
                manifest.get("unit") != "user_before_windowing" or
                not manifest.get("training_allowed") or manifest.get("holdout_pretraining_allowed")):
            raise PermissionError("a locked user split with no holdout pretraining is required")
        self.partitions = {k: tuple(v) for k, v in manifest["partitions"].items()}
        if set(self.partitions) != {"train", "development", "final"}:
            raise ValueError("unexpected split partitions")
        seen = set()
        for users in self.partitions.values():
            if not users or len(users) != len(set(users)) or seen.intersection(users):
                raise ValueError("empty, duplicate or overlapping user partition")
            seen.update(users)
        if len(seen) < 30 or len(self.partitions["development"]) != 5 or len(self.partitions["final"]) < 12:
            raise ValueError("locked cohort does not meet predeclared user counts")
        self.hdf5_sha256 = sha256_file(self.hdf5_path)
        if self.hdf5_sha256 != manifest["source_hdf5_sha256"]:
            raise ValueError("HDF5 no longer matches the locked split")
        with h5py.File(self.hdf5_path, "r") as source:
            if seen != set(source["users"].keys()):
                raise ValueError("source user inventory differs from split")
            if source.attrs["quaternion_order"] != "wxyz" or source.attrs["sample_rate_hz"] != 100:
                raise ValueError("expected wxyz quaternions on the primary 100 Hz clock")
        self.accessed_users = set()
        self.read_segments = {"train": 0, "development": 0}

    def iter_segments(self, partition):
        if partition not in {"train", "development"}:
            raise PermissionError("this stage never reads final quaternion datasets")
        with h5py.File(self.hdf5_path, "r") as source:
            for user in self.partitions[partition]:
                for session, group in source[f"users/{user}/sessions"].items():
                    for segment, data in group["segments"].items():
                        if user in self.partitions["final"]:
                            raise PermissionError("final user reached the train/development reader")
                        q = data["quaternion"][:]
                        if q.dtype != np.float32 or q.ndim != 2 or q.shape[1] != 4:
                            raise ValueError("unexpected stored quaternion format")
                        q.setflags(write=False)
                        self.accessed_users.add(user)
                        self.read_segments[partition] += 1
                        yield user, session, segment, q

    def verify_unchanged(self):
        if sha256_file(self.split_path) != self.split_sha256 or sha256_file(self.hdf5_path) != self.hdf5_sha256:
            raise ValueError("locked data or split changed during the experiment")
        if self.accessed_users.intersection(self.partitions["final"]):
            raise PermissionError("final user access detected")
        return {"source_hdf5_sha256": self.hdf5_sha256, "split_sha256": self.split_sha256,
                "data_and_split_unchanged": True, "final_quaternion_datasets_read": 0,
                "accessed_user_count": len(self.accessed_users), "segment_reads": self.read_segments.copy()}


class PopulationMoments:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, array):
        x = np.asarray(array, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("moments require nonempty finite rows")
        count = len(x)
        mean = x.mean(axis=0)
        m2 = np.sum((x - mean) ** 2, axis=0)
        if not self.count:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta**2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def as_dict(self, epsilon=1e-8):
        if not self.count:
            raise ValueError("cannot fit empty training statistics")
        std = np.sqrt(np.maximum(self.m2 / self.count, 0))
        return {"fit_partition": "train", "count": self.count, "mean": self.mean.tolist(),
                "std": std.tolist(), "scale": np.maximum(std, epsilon).tolist(), "epsilon": epsilon,
                "ddof": 0}


def build_feature_cache(stage, output, scalers_output, *, evidence_length=80, include_complete_horizon=True,
                        config_path="configs/baseline_features.json"):
    """Write and seal a new complete bundle under the frozen execution plan."""
    from .evidence import build_segment_features, EXPERT_NAMES
    output, scalers_output = Path(output), Path(scalers_output)
    config = validate_experiment_config(_read_json(config_path))
    if (type(evidence_length) is not int or evidence_length != config["evidence_length"] or
            type(include_complete_horizon) is not bool or include_complete_horizon != config["complete_horizon"]):
        raise ValueError("feature options differ from the frozen experiment config")
    directory = output.parent
    if output.name != "features.h5" or scalers_output.resolve() != (directory / "scalers.json").resolve():
        raise ValueError("feature bundle requires features.h5 and scalers.json in one run directory")
    plan_path, receipt_path = directory / "experiment_plan.json", directory / "feature_receipt.json"
    if any(path.exists() for path in (output, scalers_output, plan_path, receipt_path)):
        raise FileExistsError("feature bundle already exists; choose a new run path")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, {"config": config, "config_sha256": sha256_file(config_path),
                           "source_hdf5_sha256": stage.hdf5_sha256, "split_sha256": stage.split_sha256})
    temporary = output.with_name(output.name + ".part")
    moments, windows = {}, {"train": 0, "development": 0}
    started = time.perf_counter()
    try:
        with h5py.File(temporary, "w") as cache:
            cache.attrs["source_hdf5_sha256"] = stage.hdf5_sha256
            cache.attrs["split_sha256"] = stage.split_sha256
            cache.attrs["evidence_length"] = evidence_length
            cache.attrs["context_samples"] = 400
            cache.attrs["horizon_samples"] = 20
            cache.attrs["expert_names_json"] = json.dumps(EXPERT_NAMES)
            cache.attrs["final_access_allowed"] = False
            for partition in ("train", "development"):
                for user, session, segment, q in stage.iter_segments(partition):
                    data = build_segment_features(q, evidence_length=evidence_length,
                                                  include_complete_horizon=include_complete_horizon)
                    count = len(data["origin_index"])
                    if not count:
                        continue
                    group = cache.create_group(f"{partition}/{user}/{session}/{segment}")
                    for key in ("origin_index", "velocities", "current_q", "target_q", "expert_loss"):
                        group.create_dataset(key, data=data[key], compression="lzf", chunks=True)
                    for name, x in data["features"].items():
                        if x.shape != (count, 10) or not np.isfinite(x).all():
                            raise ValueError("invalid causal feature matrix")
                        group.create_dataset(f"features/{name}", data=x, dtype="f4", compression="lzf", chunks=True)
                        if partition == "train":
                            moments.setdefault(name, PopulationMoments()).update(x)
                    windows[partition] += count
                    print(json.dumps({"event": "features_segment", "partition": partition,
                                      "windows": count, "cumulative_windows": windows[partition]}), flush=True)
            cache.attrs["window_counts_json"] = json.dumps(windows)
        audit = stage.verify_unchanged()
        scalers = {"created_utc": utc_now(), "fit_partition": "train", "split_sha256": stage.split_sha256,
                   "source_hdf5_sha256": stage.hdf5_sha256,
                   "variants": {name: state.as_dict() for name, state in moments.items()}}
        if any(s["count"] != windows["train"] for s in scalers["variants"].values()):
            raise ValueError("scaler fit count does not match training windows")
        temporary.replace(output)
        write_json(scalers_output, scalers)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    receipt = {"windows": windows, "variants": sorted(moments), "elapsed_seconds": time.perf_counter() - started,
               "cache_bytes": output.stat().st_size, "cache_sha256": sha256_file(output), "audit": audit}
    write_json(receipt_path, receipt)
    return seal_feature_receipt(stage, directory, config_path=config_path)


def cache_groups(cache, partition):
    if partition not in {"train", "development"}:
        raise PermissionError("no final feature partition is permitted")
    if "final" in cache:
        raise ValueError("a train/development cache must not contain final features")
    for user, users in cache[partition].items():
        for session, sessions in users.items():
            for segment, group in sessions.items():
                yield user, session, segment, group


def validate_cache_binding(cache, stage):
    """Check cache metadata and origins against train/dev source shapes only.

    Quaternion/feature/label values are deliberately not read here. In particular,
    final source groups are never traversed, even to check their segment shapes.
    """
    from .evidence import EXPERT_NAMES
    if (cache.attrs.get("source_hdf5_sha256") != stage.hdf5_sha256 or
            cache.attrs.get("split_sha256") != stage.split_sha256 or set(cache) != set(_PARTITIONS)):
        raise ValueError("feature cache does not match locked train/development stage")
    for name, expected in (("context_samples", 400), ("horizon_samples", 20), ("evidence_length", 80)):
        actual = cache.attrs.get(name)
        if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, np.integer)) or actual != expected:
            raise ValueError(f"invalid feature-cache {name}")
    if not isinstance(cache.attrs.get("final_access_allowed"), (bool, np.bool_)) or cache.attrs["final_access_allowed"]:
        raise ValueError("feature cache must forbid final access")
    try:
        experts = json.loads(cache.attrs["expert_names_json"])
        recorded_counts = json.loads(cache.attrs["window_counts_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid feature-cache metadata") from error
    if experts != list(EXPERT_NAMES):
        raise ValueError("unexpected feature-cache expert order")
    _validate_counts(recorded_counts, "cache")
    counts = {partition: 0 for partition in _PARTITIONS}
    with h5py.File(stage.hdf5_path, "r") as source:
        for partition in _PARTITIONS:
            _require_hard_group(cache, partition)
            if set(cache[partition]) != set(stage.partitions[partition]):
                raise ValueError("feature-cache users differ from the locked partition")
            expected_groups = {}
            for user in stage.partitions[partition]:
                _require_hard_group(cache[partition], user)
                for session, session_group in source[f"users/{user}/sessions"].items():
                    for segment, segment_group in session_group["segments"].items():
                        # Dataset metadata only: do not index the source quaternions.
                        q = segment_group["quaternion"]
                        if q.dtype != np.dtype("float32") or q.ndim != 2 or q.shape[1] != 4:
                            raise ValueError("unexpected source train/development quaternion shape or dtype")
                        count = max(0, q.shape[0] - 419)
                        if count:
                            expected_groups[(user, session, segment)] = count
            actual_groups = set()
            for user in cache[partition]:
                for session in cache[partition][user]:
                    _require_hard_group(cache[partition][user], session)
                    for segment in cache[partition][user][session]:
                        _require_hard_group(cache[partition][user][session], segment)
                        key = (user, session, segment)
                        if key not in expected_groups:
                            raise ValueError("unexpected feature-cache segment")
                        actual_groups.add(key)
                        group = cache[partition][user][session][segment]
                        count = expected_groups[key]
                        expected_shapes = {
                            "origin_index": ((count,), "int64"), "velocities": ((count, 4, 3), "float64"),
                            "current_q": ((count, 4), "float32"), "target_q": ((count, 4), "float32"),
                            "expert_loss": ((count, 4), "float64"),
                        }
                        if set(group) != set(expected_shapes) | {"features"}:
                            raise ValueError("unexpected feature-cache datasets")
                        for name, (shape, dtype) in expected_shapes.items():
                            _require_dataset(group, name, shape, dtype)
                        _require_hard_group(group, "features")
                        if set(group["features"]) != _FEATURE_VARIANTS:
                            raise ValueError("feature-cache variants differ from frozen plan")
                        for name in _FEATURE_VARIANTS:
                            _require_dataset(group["features"], name, (count, 10), "float32")
                        for start in range(0, count, 65536):
                            stop = min(count, start + 65536)
                            if not np.array_equal(group["origin_index"][start:stop],
                                                  np.arange(399 + start, 399 + stop, dtype=np.int64)):
                                raise ValueError("feature-cache origins must cover every stride1 context exactly")
                        counts[partition] += count
            if actual_groups != set(expected_groups):
                raise ValueError("feature cache does not cover every train/development source segment")
    if counts != recorded_counts:
        raise ValueError("feature-cache window counts differ from its datasets")
    return counts


def _require_hard_group(parent, name):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink) or not isinstance(parent[name], h5py.Group):
        raise ValueError("feature cache requires local hard-linked groups")


def _require_dataset(parent, name, shape, dtype):
    if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
        raise ValueError("feature cache requires local hard-linked datasets")
    dataset = parent[name]
    if (not isinstance(dataset, h5py.Dataset) or dataset.shape != shape or dataset.dtype != np.dtype(dtype) or
            dataset.is_virtual or dataset.external):
        raise ValueError(f"unexpected feature-cache shape/dtype/storage for {name}")


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {Path(path).name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {Path(path).name}")
    return value


def _validate_counts(counts, label):
    if (not isinstance(counts, dict) or set(counts) != set(_PARTITIONS) or
            any(type(value) is not int or value < 1 for value in counts.values())):
        raise ValueError(f"invalid {label} window counts")


def _validate_scalers(scalers, stage, counts):
    if (scalers.get("fit_partition") != "train" or scalers.get("split_sha256") != stage.split_sha256 or
            scalers.get("source_hdf5_sha256") != stage.hdf5_sha256 or
            not isinstance(scalers.get("variants"), dict) or set(scalers["variants"]) != _FEATURE_VARIANTS):
        raise ValueError("scalers do not match locked train-only feature fitting")
    for state in scalers["variants"].values():
        if (not isinstance(state, dict) or state.get("fit_partition") != "train" or
                type(state.get("count")) is not int or state["count"] != counts["train"] or
                type(state.get("ddof")) is not int or state["ddof"] != 0 or
                type(state.get("epsilon")) not in (int, float) or state["epsilon"] != 1e-8):
            raise ValueError("invalid train-only scaler fit/count/statistics")
        arrays = {}
        for field in ("mean", "std", "scale"):
            values = state.get(field)
            if (not isinstance(values, list) or len(values) != 10 or
                    any(isinstance(v, bool) or not isinstance(v, Real) for v in values)):
                raise ValueError("scaler vectors must contain ten finite numeric values")
            arrays[field] = np.asarray(values, dtype=np.float64)
            if not np.isfinite(arrays[field]).all():
                raise ValueError("scaler vectors must contain ten finite numeric values")
        if np.any(arrays["std"] < 0) or not np.array_equal(arrays["scale"], np.maximum(arrays["std"], 1e-8)):
            raise ValueError("scaler scale must use the frozen population std floor")


def _inspect_feature_bundle(stage, run_dir, config_path, *, require_seal):
    directory = Path(run_dir)
    config = validate_experiment_config(_read_json(config_path))
    config_hash = sha256_file(config_path)
    plan_path, receipt_path = directory / "experiment_plan.json", directory / "feature_receipt.json"
    plan, receipt = _read_json(plan_path), _read_json(receipt_path)
    validate_experiment_config(plan.get("config"))
    if (plan["config"] != config or plan.get("config_sha256") != config_hash or
            plan.get("source_hdf5_sha256") != stage.hdf5_sha256 or plan.get("split_sha256") != stage.split_sha256):
        raise ValueError("experiment plan does not match config or locked data binding")
    cache_path, scaler_path = directory / "features.h5", directory / "scalers.json"
    actual_hashes = {"cache_sha256": sha256_file(cache_path), "scalers_sha256": sha256_file(scaler_path),
                     "experiment_plan_sha256": sha256_file(plan_path), "config_sha256": config_hash}
    if receipt.get("cache_sha256") != actual_hashes["cache_sha256"]:
        raise ValueError("feature cache SHA-256 differs from original feature receipt")
    if type(receipt.get("cache_bytes")) is not int or receipt["cache_bytes"] != cache_path.stat().st_size:
        raise ValueError("feature cache byte count differs from original feature receipt")
    _validate_counts(receipt.get("windows"), "receipt")
    if (not isinstance(receipt.get("variants"), list) or receipt["variants"] != sorted(_FEATURE_VARIANTS) or
            type(receipt.get("elapsed_seconds")) not in (int, float) or
            not np.isfinite(receipt["elapsed_seconds"]) or receipt["elapsed_seconds"] < 0):
        raise ValueError("invalid feature receipt schema")
    audit = receipt.get("audit")
    if (not isinstance(audit, dict) or audit.get("source_hdf5_sha256") != stage.hdf5_sha256 or
            audit.get("split_sha256") != stage.split_sha256 or audit.get("data_and_split_unchanged") is not True or
            type(audit.get("final_quaternion_datasets_read")) is not int or audit["final_quaternion_datasets_read"] != 0 or
            type(audit.get("accessed_user_count")) is not int or
            audit["accessed_user_count"] != sum(len(stage.partitions[p]) for p in _PARTITIONS)):
        raise ValueError("invalid feature receipt train/development access audit")
    reads = audit.get("segment_reads")
    if (not isinstance(reads, dict) or set(reads) != set(_PARTITIONS) or
            any(type(value) is not int or value < 1 for value in reads.values())):
        raise ValueError("invalid feature receipt segment-read audit")
    with h5py.File(cache_path, "r") as cache:
        counts = validate_cache_binding(cache, stage)
        if any(reads[partition] < sum(1 for _ in cache_groups(cache, partition)) for partition in _PARTITIONS):
            raise ValueError("feature receipt segment-read audit omits cached segments")
    if counts != receipt["windows"]:
        raise ValueError("feature receipt window counts differ from cache")
    _validate_scalers(_read_json(scaler_path), stage, counts)
    # Existing seals are always verified, including on an idempotent reseal.
    seal_fields = {"schema_version", "sealed_utc", "scalers_sha256", "experiment_plan_sha256", "config_sha256"}
    present_seal_fields = set(receipt).intersection(seal_fields)
    if present_seal_fields and present_seal_fields != seal_fields:
        raise ValueError("incomplete sealed feature receipt schema")
    sealed = bool(present_seal_fields)
    if require_seal and not sealed:
        raise ValueError("feature receipt is not sealed; run seal-features first")
    if sealed:
        if (type(receipt["schema_version"]) is not int or receipt["schema_version"] != _RECEIPT_SCHEMA_VERSION or
                not isinstance(receipt.get("sealed_utc"), str) or not receipt["sealed_utc"]):
            raise ValueError("unsupported sealed feature receipt schema")
        for key, actual in actual_hashes.items():
            if receipt.get(key) != actual:
                raise ValueError(f"sealed feature bundle {key} mismatch")
    stage.verify_unchanged()
    return receipt, actual_hashes


def validate_feature_bundle(stage, run_dir, config_path="configs/baseline_features.json"):
    """Verify all four private bundle files before any experiment consumes them."""
    receipt, _ = _inspect_feature_bundle(stage, run_dir, config_path, require_seal=True)
    return receipt


def seal_feature_receipt(stage, run_dir, config_path="configs/baseline_features.json"):
    """Bind existing scaler/plan/config bytes to an original cache receipt.

    This never regenerates data or replaces a disputed cache checksum. A sealed
    receipt can be checked again, but changed artifacts cannot be resealed.
    """
    receipt, hashes = _inspect_feature_bundle(stage, run_dir, config_path, require_seal=False)
    if "schema_version" not in receipt:
        receipt = {**receipt, "schema_version": _RECEIPT_SCHEMA_VERSION, "sealed_utc": utc_now(), **hashes}
        write_json(Path(run_dir) / "feature_receipt.json", receipt)
    return receipt

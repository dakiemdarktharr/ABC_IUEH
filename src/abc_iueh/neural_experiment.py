"""Fixed train/development MLP experiment with no final-partition reader.

Only a batch of overlapping windows is copied at a time. Training origins are
sampled uniformly over all eligible training windows, with replacement. The
reported two-seed mean is the arithmetic mean of errors, not predictions.
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .evaluation import summarize_methods
from .geometry import geodesic_deg
from .neural import RelativeMLP, adjacent_features
from .sources import utc_now, write_json
from .stage import (
    PopulationMoments, cache_groups, validate_cache_binding,
    validate_experiment_config, validate_feature_bundle,
)

_PARTITIONS = ("train", "development")
_CONTEXT, _HORIZON = 400, 20
_EVALUATION_BATCH = 512
_MEAN_ERROR_METHOD = "MLP_two_seed_mean_error"


def _load_segments(stage, partition):
    if partition not in _PARTITIONS:
        raise PermissionError("neural experiment never reads final quaternions")
    result = {}
    for user, session, segment, q in stage.iter_segments(partition):
        key = (user, session, segment)
        if user not in stage.partitions[partition] or key in result:
            raise ValueError("unexpected or duplicate segment in guarded reader")
        if q.dtype != np.float32 or q.ndim != 2 or q.shape[1] != 4 or len(q) < 2:
            raise ValueError("expected float32 quaternion segment with at least two samples")
        if q.flags.writeable:
            raise ValueError("guarded segment arrays must be read-only")
        result[key] = q
    if not result:
        raise ValueError(f"empty {partition} quaternion partition")
    return result


def _fit_input_statistics(training_segments, chunk_increments=65536):
    """Fit each adjacent train increment once, including chunk boundaries."""
    if chunk_increments < 1:
        raise ValueError("increment chunk must be positive")
    moments = PopulationMoments()
    for key in sorted(training_segments):
        q = training_segments[key]
        for start in range(0, len(q) - 1, chunk_increments):
            stop = min(start + chunk_increments + 1, len(q))
            increments = adjacent_features(q[None, start:stop, :])[0]
            moments.update(increments)
    return moments.as_dict(epsilon=1e-8)


def _window_view(q):
    """Read-only [N-419,420,4] view; this never allocates overlapping windows."""
    if len(q) < _CONTEXT + _HORIZON:
        raise ValueError("segment has no complete 400-to-20 window")
    return np.moveaxis(sliding_window_view(q, _CONTEXT + _HORIZON, axis=0), -1, 1)


class _TrainingOrigins:
    """Map a uniform global index to a segment-local context/target pair."""

    def __init__(self, training_segments):
        self.keys = tuple(key for key in sorted(training_segments)
                          if len(training_segments[key]) >= _CONTEXT + _HORIZON)
        self.views = tuple(_window_view(training_segments[key]) for key in self.keys)
        self.counts = np.array([len(view) for view in self.views], dtype=np.int64)
        self.cumulative = np.cumsum(self.counts, dtype=np.int64)
        if not len(self.cumulative):
            raise ValueError("training partition has no complete windows")
        self.total = int(self.cumulative[-1])

    def batch_at(self, global_indexes):
        indexes = np.asarray(global_indexes)
        if (indexes.ndim != 1 or not len(indexes) or indexes.dtype.kind not in "iu"
                or np.any(indexes < 0) or np.any(indexes >= self.total)):
            raise ValueError("training indexes must be nonempty valid global integers")
        indexes = indexes.astype(np.int64, copy=False)
        segments = np.searchsorted(self.cumulative, indexes, side="right")
        starts = np.r_[0, self.cumulative[:-1]]
        local = indexes - starts[segments]
        batch = np.empty((len(indexes), _CONTEXT + _HORIZON, 4), dtype=np.float32)
        for segment_index in np.unique(segments):
            selected = np.flatnonzero(segments == segment_index)
            batch[selected] = self.views[segment_index][local[selected]]
        return batch[:, :_CONTEXT], batch[:, _CONTEXT:]

    def sample(self, rng, batch_size):
        if batch_size < 1:
            raise ValueError("training batch size must be positive")
        return self.batch_at(rng.integers(0, self.total, size=batch_size))


def _train_model(model, origins, *, seed, steps, batch_size, learning_rate):
    """Fixed-step fitting; caller exposes no development data to this helper."""
    if steps < 1:
        raise ValueError("training steps must be positive")
    rng = np.random.default_rng(seed)
    losses = np.empty(steps, dtype=np.float64)
    started = time.perf_counter()
    for step in range(steps):
        context, target = origins.sample(rng, batch_size)
        loss = model.train_step(context, target, learning_rate=learning_rate)
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("training produced an invalid quaternion MSE")
        losses[step] = loss
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(json.dumps({"event": "neural_training", "seed": seed,
                              "step": step + 1, "steps": steps, "loss": float(loss)}), flush=True)
    tail = min(100, steps)
    return {
        "seed": seed, "first_step": 1, "last_step": steps, "optimizer_steps": steps,
        "optimizer": "Adam", "learning_rate": learning_rate, "batch_size": batch_size,
        "sampling": "uniform_global_train_origins_with_replacement",
        "sampling_seed": seed, "population_training_windows": origins.total,
        "sampled_training_windows": steps * batch_size,
        "first_step_loss": float(losses[0]), "last_step_loss": float(losses[-1]),
        "mean_step_loss": float(losses.mean()),
        "first_100_step_mean_loss": float(losses[:tail].mean()),
        "last_100_step_mean_loss": float(losses[-tail:].mean()),
        "loss_summary_window_steps": tail,
        "loss": "sign_aligned_quaternion_MSE_all_20_outputs",
        "checkpoint_selection": "fixed_final_step_no_development_selection",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_partition(cache, partition, segments, models, error_group, *, batch_size=512):
    """Emit endpoint errors in exactly the analytic cache's flattening order."""
    if partition not in _PARTITIONS or batch_size < 1 or len(models) != 2:
        raise ValueError("evaluation requires train/development, two seeds and a positive batch")
    groups = list(cache_groups(cache, partition))
    total = sum(len(group["origin_index"]) for *_, group in groups)
    if not total:
        raise ValueError(f"{partition} feature cache contains no windows")
    expected_keys = {key for key, q in segments.items() if len(q) >= _CONTEXT + _HORIZON}
    if {(u, s, g) for u, s, g, _ in groups} != expected_keys:
        raise ValueError("feature-cache segments differ from guarded quaternion windows")
    methods = (*models, _MEAN_ERROR_METHOD)
    for method in methods:
        error_group.create_dataset(method, shape=(total,), dtype="f8", chunks=True, compression="lzf")
    user_sums = {method: {} for method in methods}
    user_counts, position = {}, 0
    for user, session, segment, group in groups:
        q = segments[(user, session, segment)]
        windows = _window_view(q)
        count = len(windows)
        if (group["origin_index"].shape != (count,) or group["target_q"].shape != (count, 4)
                or not np.array_equal(group["origin_index"][:], np.arange(399, len(q) - 20))):
            raise ValueError("feature origins are not the full aligned stride1 window sequence")
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            batch = windows[start:stop]
            target = group["target_q"][start:stop]
            if not np.array_equal(target, batch[:, -1, :]):
                raise ValueError("cached endpoint20 targets do not match guarded source samples")
            errors = {}
            for method, model in models.items():
                prediction = model.predict(batch[:, :_CONTEXT])
                if prediction.shape != (stop - start, _HORIZON, 4) or prediction.dtype != np.float32:
                    raise ValueError("MLP prediction must be float32 [B,20,4]")
                errors[method] = geodesic_deg(prediction[:, -1], target)
            errors[_MEAN_ERROR_METHOD] = np.mean(np.stack(list(errors.values())), axis=0)
            for method, angles in errors.items():
                if not np.isfinite(angles).all() or np.any(angles < 0) or np.any(angles > 180 + 1e-6):
                    raise ValueError("invalid neural endpoint20 angular errors")
                error_group[method][position + start:position + stop] = angles
                method_users = user_sums[method]
                method_users[user] = method_users.get(user, 0.0) + float(angles.sum(dtype=np.float64))
        user_counts[user] = user_counts.get(user, 0) + count
        position += count
        print(json.dumps({"event": "neural_evaluation", "partition": partition,
                          "windows_done": position, "windows_total": total}), flush=True)
    if position != total:
        raise ValueError("neural errors do not cover the full cached origin sequence")
    return user_sums, user_counts


def run_neural(stage, cache_path, run_dir, config, *, config_path="configs/baseline_features.json"):
    """Fit fixed neural seeds and score all train/development windows privately.

    Creates an exclusive ``run_dir/neural`` directory. A successful run returns
    its aggregate summary and private flat error HDF5 path; weights, scaler and
    participant sufficient statistics stay beside that private error file.
    """
    validate_experiment_config(config)
    run_dir = Path(run_dir)
    if Path(cache_path).resolve() != (run_dir / "features.h5").resolve():
        raise ValueError("neural experiment must consume its run directory's sealed features.h5")
    validate_feature_bundle(stage, run_dir, config_path=config_path)
    directory = run_dir / "neural"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    error_path = directory / "errors.h5"
    started = time.perf_counter()
    settings, uncertainty = config["neural_baseline"], config["uncertainty"]
    with h5py.File(cache_path, "r") as cache:
        validate_cache_binding(cache, stage)
        training = _load_segments(stage, "train")
        scaler = _fit_input_statistics(training)
        np.savez_compressed(
            directory / "scaler.npz", mean=np.asarray(scaler["mean"], dtype=np.float64),
            std=np.asarray(scaler["std"], dtype=np.float64),
            scale=np.asarray(scaler["scale"], dtype=np.float64),
            count=np.int64(scaler["count"]), epsilon=np.float64(scaler["epsilon"]), ddof=np.int64(0),
        )
        origins = _TrainingOrigins(training)
        models, fits = {}, []
        for seed in settings["seeds"]:
            model = RelativeMLP(seed=seed, input_mean=scaler["mean"], input_std=scaler["scale"])
            fit = _train_model(model, origins, seed=seed, steps=settings["steps"],
                               batch_size=settings["batch_size"], learning_rate=settings["learning_rate"])
            method = f"MLP_seed_{seed}"
            model.save(directory / f"{method}.npz")
            models[method] = model
            fits.append(fit)
        # Development quaternions are loaded only after both fits are complete.
        development = _load_segments(stage, "development")
        private, public = {}, {}
        with h5py.File(error_path, "x") as errors:
            errors.attrs["source_hdf5_sha256"] = stage.hdf5_sha256
            errors.attrs["split_sha256"] = stage.split_sha256
            errors.attrs["final_access_allowed"] = False
            errors.attrs["metric"] = "actual_endpoint20_geodesic_degrees"
            errors.attrs["two_seed_mean_interpretation"] = "arithmetic_mean_of_errors_not_prediction_ensemble"
            for partition, segments in (("train", training), ("development", development)):
                error_group = errors.create_group(partition)
                sums, counts = _evaluate_partition(cache, partition, segments, models, error_group,
                                                  batch_size=_EVALUATION_BATCH)
                methods, contrasts = summarize_methods(
                    error_group, sums, counts, resamples=uncertainty["bootstrap_resamples"],
                    seed=uncertainty["seed"],
                )
                public[partition] = {"methods": methods, "contrasts": contrasts}
                private[partition] = {"user_sums": sums, "user_counts": counts}
    audit = stage.verify_unchanged()
    summary = {
        "created_utc": utc_now(), "stage": "train_development_only", "results": public,
        "architecture": settings["architecture"], "parameters": settings["parameters"],
        "training": fits,
        "input_scaling": {"fit_partition": "train", "increments": scaler["count"],
                          "axes": 3, "ddof": 0, "epsilon": scaler["epsilon"],
                          "weighting": "each_adjacent_training_segment_increment_once"},
        "evaluation_batch_size": _EVALUATION_BATCH,
        "two_seed_mean_interpretation": "arithmetic_mean_of_endpoint20_errors_not_prediction_ensemble",
        "development_interpretation": "exploratory_only_five_users_no_checkpoint_selection",
        "elapsed_seconds": time.perf_counter() - started, "audit": audit,
    }
    write_json(directory / "private_user_stats.json", private)
    write_json(directory / "summary.json", summary)
    return summary, error_path

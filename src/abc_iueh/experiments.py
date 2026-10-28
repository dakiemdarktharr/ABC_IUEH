"""Reproducible train/development analytic baselines; no final evaluator."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import h5py
import numpy as np

from .evaluation import analytic_errors, fit_static_mixture, summarize_methods
from .sources import sha256_file, utc_now, write_json
from .stage import cache_groups, validate_cache_binding


def sample_train_calibration(cache, maximum, seed):
    """Uniform sample of global training origins, without a window-sized array."""
    groups = list(cache_groups(cache, "train"))
    counts = np.array([len(group["origin_index"]) for *_, group in groups], dtype=np.int64)
    if not len(counts) or maximum < 1:
        raise ValueError("training calibration requires nonempty data and budget")
    total = int(counts.sum())
    selected = np.sort(np.random.default_rng(seed).choice(total, size=min(total, maximum), replace=False))
    chunks = {name: [] for name in ("current_q", "velocities", "target_q")}
    offset = 0
    for count, (*_, group) in zip(counts, groups):
        start, stop = np.searchsorted(selected, [offset, offset + count])
        local = selected[start:stop] - offset
        if len(local):
            for name in chunks:
                chunks[name].append(group[name][local])
        offset += count
    result = {name: np.concatenate(rows, axis=0) for name, rows in chunks.items()}
    result["sample_receipt"] = {"partition": "train", "seed": seed, "population_windows": total,
                                "sampled_windows": len(selected),
                                "global_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest()}
    return result


def read_feature_group(group):
    return {**{name: group[name][:] for name in ("current_q", "target_q", "velocities", "expert_loss")},
            "features": {name: value[:] for name, value in group["features"].items()}}


def run_analytic(stage, directory, config, *, config_path="configs/baseline_features.json"):
    """Consume only a verified sealed bundle; final is never an option."""
    from .stage import validate_experiment_config, validate_feature_bundle
    validate_experiment_config(config)
    directory = Path(directory)
    plan = json.loads((directory / "experiment_plan.json").read_text(encoding="utf-8"))
    if config != plan.get("config"):
        raise ValueError("analytic arguments differ from frozen run plan")
    validate_feature_bundle(stage, directory, config_path=config_path)
    output = directory / "analytic_errors.h5"
    if output.exists() or (directory / "analytic_summary.json").exists():
        raise FileExistsError("analytic run already exists")
    started = time.perf_counter()
    private_stats, public = {}, {}
    with h5py.File(directory / "features.h5", "r") as cache, h5py.File(output, "x") as errors:
        validate_cache_binding(cache, stage)
        calibration = sample_train_calibration(cache, config["static_mixture"]["max_calibration_windows"], config["seed"])
        weights, fit = fit_static_mixture(calibration["current_q"], calibration["velocities"], calibration["target_q"],
                                          step=config["static_mixture"]["simplex_step"])
        fit["sampling"] = calibration["sample_receipt"]
        write_json(directory / "static_mixture_fit.json", fit)
        del calibration
        for partition in config["partitions"]:
            total = sum(len(group["origin_index"]) for *_, group in cache_groups(cache, partition))
            error_group = errors.create_group(partition)
            user_sums, user_counts, position = {}, {}, 0
            for user, _, _, group in cache_groups(cache, partition):
                data = read_feature_group(group)
                values = analytic_errors(data, weights, config["rank_temperature"])
                count = len(data["current_q"])
                for method, angles in values.items():
                    if method not in error_group:
                        error_group.create_dataset(method, shape=(total,), dtype="f8", chunks=True, compression="lzf")
                    error_group[method][position:position + count] = angles
                    method_users = user_sums.setdefault(method, {})
                    method_users[user] = method_users.get(user, 0.0) + float(np.sum(angles, dtype=np.float64))
                user_counts[user] = user_counts.get(user, 0) + count
                position += count
                print(json.dumps({"event": "analytic_segment", "partition": partition, "windows_done": position,
                                  "windows_total": total}), flush=True)
            if position != total:
                raise ValueError("analytic output does not cover every cached origin")
            methods, contrasts = summarize_methods(error_group, user_sums, user_counts,
                                                    resamples=config["uncertainty"]["bootstrap_resamples"],
                                                    seed=config["uncertainty"]["seed"])
            public[partition] = {"methods": methods, "contrasts": contrasts}
            private_stats[partition] = {"user_sums": user_sums, "user_counts": user_counts}
        errors.attrs["source_hdf5_sha256"] = stage.hdf5_sha256
        errors.attrs["split_sha256"] = stage.split_sha256
        errors.attrs["final_access_allowed"] = False
    development_methods = public["development"]["methods"]
    best = min(("CAV_W1", "CAV_W5", "CAV_W20"), key=lambda name: development_methods[name]["mean_geodesic_deg"])
    summary = {"created_utc": utc_now(), "stage": "train_development_only", "results": public,
               "lowest_development_cav_descriptive": {"method": best, "comparison_partition": "development",
                                                      "interpretation": "observed_minimum_of_predeclared_references_no_final_model_selection"},
               "static_fit": {k: fit[k] for k in ("fit_partition", "grid_candidates", "calibration_windows", "objective")},
               "oracle_interpretation": "hindsight_min_expert_loss_not_causal_or_submittable",
               "elapsed_seconds": time.perf_counter() - started,
               "audit": stage.verify_unchanged()}
    write_json(directory / "analytic_user_statistics.json", private_stats)
    write_json(directory / "analytic_summary.json", summary)
    write_json(directory / "analytic_receipt.json", {"errors_sha256": sha256_file(output),
                                                    "summary_sha256": sha256_file(directory / "analytic_summary.json"),
                                                    "audit": summary["audit"]})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", default="data/processed/alyx.h5")
    parser.add_argument("--split", default="data/user_split.json")
    parser.add_argument("--config", default="configs/baseline_features.json")
    parser.add_argument("--run-dir", default="data/baseline_runs/20261028_r80")
    args = parser.parse_args(argv)
    from .stage import LockedDevelopmentStage, validate_feature_bundle
    stage, directory = LockedDevelopmentStage(args.hdf5, args.split), Path(args.run_dir)
    validate_feature_bundle(stage, directory, config_path=args.config)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_analytic(stage, directory, config, config_path=args.config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

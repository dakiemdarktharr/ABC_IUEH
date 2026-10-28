"""Recorded inner-user pilot metrics; no source-data or holdout access.

The API accepts private paired endpoint-error recordings, verifies each user's
coverage/order/sum, and returns aggregate-only exploratory summaries. Bootstrap
draws condition on the fixed cross-validation fits and two seeds; no retraining
or population coverage is implied.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

TRAINING_SEEDS = (20261026, 20261027)
FOLD_IDS = (0, 1, 2)
ARMS = ("nll21", "scalar_angle", "scalar_pinball", "risk21", "vector_angle")
METHODS = (
    "nll21_median", "nll21_mean", "nll21_map", "scalar_angle",
    "scalar_pinball", "risk21", "vector_angle", "static_mixture",
    "prior_median", "prior_mean", "prior_map", "ZV", "CAV_W1", "CAV_W5", "CAV_W20",
)
GATE_COMPARATORS = ("static_mixture", "scalar_angle", "risk21", "scalar_pinball")
PARAMETERS = {"nll21": 1717, "scalar_angle": 1649, "scalar_pinball": 1649,
              "risk21": 1717, "vector_angle": 1683}
CONV_DENSE_MACS = {"nll21": 425904, "scalar_angle": 425840,
                   "scalar_pinball": 425840, "risk21": 425904, "vector_angle": 425872}


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("row-order binding must be a lowercase SHA-256")
    return value


def row_order_digest(user_row_order_sha256: Mapping[str, str]) -> str:
    """Canonical private user/digest map binding; do not publish the map."""
    if not isinstance(user_row_order_sha256, Mapping) or not user_row_order_sha256:
        raise ValueError("nonempty user row-order map required")
    if any(not isinstance(u, str) or not u for u in user_row_order_sha256):
        raise ValueError("user row-order keys must be nonempty strings")
    canonical = {u: _digest(user_row_order_sha256[u]) for u in sorted(user_row_order_sha256)}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest()


def _coverage(user_counts, user_folds, orders):
    if not isinstance(user_counts, Mapping) or len(user_counts) != 15:
        raise ValueError("pilot requires exactly 15 outer-training validation users")
    if any(not isinstance(u, str) or not u for u in user_counts):
        raise ValueError("private user keys must be nonempty strings")
    users = sorted(user_counts)
    if not isinstance(user_folds, Mapping) or set(user_folds) != set(users):
        raise ValueError("fold map must cover exactly the validation users")
    if not isinstance(orders, Mapping) or set(orders) != set(users):
        raise ValueError("row-order map must cover exactly the validation users")
    counts = np.array([_positive_int(user_counts[u], "user count") for u in users], dtype=np.int64)
    fold_values = []
    for u in users:
        value = _nonnegative_int(user_folds[u], "fold ID")
        if value not in FOLD_IDS:
            raise ValueError("only three fixed inner fold IDs are allowed")
        fold_values.append(value)
        _digest(orders[u])
    folds = np.asarray(fold_values)
    if any(int(np.count_nonzero(folds == f)) != 5 for f in FOLD_IDS):
        raise ValueError("each fold must validate exactly five whole users")
    return users, counts, folds


def _keys(group, expected, message):
    if set(group.keys()) != set(expected):
        raise ValueError(message)


def _error_vector(dataset, expected_count, expected_order, expected_sum):
    if not hasattr(dataset, "attrs") or dataset.attrs.get("row_order_sha256") != expected_order:
        raise ValueError("endpoint errors must bind to the same per-user origin row order")
    if dataset.shape != (expected_count,):
        raise ValueError("endpoint error row counts differ from validation origin counts")
    if np.dtype(dataset.dtype).kind not in "fiu":
        raise ValueError("endpoint errors must be numeric")
    values = np.asarray(dataset[:], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 180)):
        raise ValueError("endpoint errors must be finite geodesic degrees in [0,180]")
    recorded = _number(expected_sum, "recorded user error sum")
    actual = float(values.sum(dtype=np.float64))
    if recorded < 0 or not math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError("direct recorded error sums disagree with private user sums")
    return values


def _summary(values, sums, counts, boot, draws):
    interval = np.percentile(boot, [2.5, 97.5])
    return {
        "windows": int(counts.sum()), "users": int(len(counts)),
        "mean_geodesic_deg": float(sums.sum() / counts.sum()),
        "p95_geodesic_deg": float(np.percentile(values, 95)),
        "user_macro_mean_deg": float(np.mean(sums / counts)),
        "conditional_user_cluster_95_CI_mean_deg": [float(x) for x in interval],
        "conditional_user_cluster_95_CI_user_macro_mean_deg": [float(x) for x in
            np.percentile((sums / counts)[draws].mean(axis=1), [2.5, 97.5])],
    }


def _resource_summary(resources):
    result = {"fixed_parameters": PARAMETERS.copy(), "fixed_conv_dense_MACs": CONV_DENSE_MACS.copy(),
              "MAC_scope": "convolution_and_dense_only_excludes_geometry_decisions_activations_bias_padding",
              "static_neural_compute_matched": False}
    if resources is None:
        result.update(status="not_supplied", strong_same_wall_compute_claim_allowed=False)
        return result
    if not isinstance(resources, Mapping) or set(resources) != set(ARMS):
        raise ValueError("resource records must cover exactly the five learned arms")
    permitted = {"fold", "seed", "training_seconds", "validation_seconds",
                 "peak_allocated_bytes", "peak_reserved_bytes"}
    total_times, cells, public_arms = {}, {}, {}
    for arm in ARMS:
        records = resources[arm]
        if not isinstance(records, (list, tuple)) or len(records) != 6:
            raise ValueError("each learned arm requires six fold/seed resource records")
        seen, sanitized = set(), []
        for record in records:
            if not isinstance(record, Mapping) or set(record) != permitted:
                raise ValueError("resource fields must use the public-safe whitelist")
            fold = _nonnegative_int(record["fold"], "resource fold")
            training_seed = _nonnegative_int(record["seed"], "resource seed")
            key = (fold, training_seed)
            if fold not in FOLD_IDS or training_seed not in TRAINING_SEEDS or key in seen:
                raise ValueError("resource fold/seed cells must be unique and fixed")
            seen.add(key)
            seconds = _number(record["training_seconds"], "training seconds")
            validation = _number(record["validation_seconds"], "validation seconds")
            allocated = _nonnegative_int(record["peak_allocated_bytes"], "peak allocated bytes")
            reserved = _nonnegative_int(record["peak_reserved_bytes"], "peak reserved bytes")
            if seconds <= 0 or validation < 0 or reserved < allocated:
                raise ValueError("invalid measured resource scope")
            cells.setdefault(key, []).append(seconds)
            sanitized.append({"fold": fold, "seed": training_seed, "training_seconds": seconds,
                              "validation_seconds": validation, "peak_allocated_bytes": allocated,
                              "peak_reserved_bytes": reserved})
        total_times[arm] = sum(r["training_seconds"] for r in sanitized)
        public_arms[arm] = {"parameters": PARAMETERS[arm], "conv_dense_MACs": CONV_DENSE_MACS[arm],
                            "total_training_seconds": total_times[arm],
                            "records": sorted(sanitized, key=lambda r: (r["fold"], r["seed"]))}
    ratio = max(total_times.values()) / min(total_times.values())
    max_cell_ratio = max(max(v) / min(v) for v in cells.values())
    result.update(status="measured_training_time_receipts_updates_require_runner_scope", arms=public_arms,
                  maximum_to_minimum_total_training_time_ratio=float(ratio),
                  maximum_per_fold_seed_training_time_ratio=float(max_cell_ratio),
                  strong_same_wall_compute_claim_allowed=bool(ratio <= 1.2 and max_cell_ratio <= 1.2),
                  unequal_time_action="qualify_claim_no_automatic_extra_fit_or_schedule_change")
    return result


def summarize_arc_pilot(error_group, user_sums, user_counts, user_folds, *,
                        row_order_sha256, resources=None, resamples=2000, seed=20261028):
    """Summarize all fixed inner-validation errors with private identity removal.

    error_group layout: method/str(training_seed)/user -> 1D HDF5 error dataset.
    Each dataset has attr row_order_sha256 matching the supplied private map.
    The root group has that attr set to row_order_digest(the supplied map).
    user_sums[method][str(seed)][user] records float64 sums independently.
    user_counts and user_folds are keyed by the same private 15 user strings.
    Baselines must be recorded identically at both seed keys; no mean-error
    dataset is accepted because the paired arithmetic mean is computed here.
    """
    resamples = _positive_int(resamples, "bootstrap resamples")
    seed = _nonnegative_int(seed, "bootstrap seed")
    users, counts, folds = _coverage(user_counts, user_folds, row_order_sha256)
    if not hasattr(error_group, "attrs") or error_group.attrs.get("row_order_sha256") != row_order_digest(row_order_sha256):
        raise ValueError("error recording root row-order binding differs")
    _keys(error_group, METHODS, "recording must contain exactly the fixed pilot methods")
    if not isinstance(user_sums, Mapping) or set(user_sums) != set(METHODS):
        raise ValueError("private sums must cover exactly the pilot methods")
    draws = np.random.default_rng(seed).integers(0, len(users), size=(resamples, len(users)))
    denominator = counts[draws].sum(axis=1)
    summaries, sums_by_method, bootstrap = {}, {}, {}
    seed_means, fold_means = {}, {}
    analytic = {"static_mixture", "prior_median", "prior_mean", "prior_map", "ZV", "CAV_W1", "CAV_W5", "CAV_W20"}
    for method in METHODS:
        _keys(error_group[method], map(str, TRAINING_SEEDS), "each method must contain both fixed seeds")
        if not isinstance(user_sums[method], Mapping) or set(user_sums[method]) != set(map(str, TRAINING_SEEDS)):
            raise ValueError("each private sum table must contain both fixed seeds")
        vectors = [[], []]
        sums = np.empty((2, len(users)), dtype=np.float64)
        for s, training_seed in enumerate(TRAINING_SEEDS):
            key = str(training_seed)
            _keys(error_group[method][key], users, "each seed must cover exactly the same validation users")
            if not isinstance(user_sums[method][key], Mapping) or set(user_sums[method][key]) != set(users):
                raise ValueError("private seed sums have different user coverage")
            for i, user in enumerate(users):
                value = _error_vector(error_group[method][key][user], int(counts[i]),
                                      row_order_sha256[user], user_sums[method][key][user])
                vectors[s].append(value)
                sums[s, i] = value.sum(dtype=np.float64)
        if method in analytic and any(not np.array_equal(a, b) for a, b in zip(*vectors)):
            raise ValueError("deterministic analytic baseline errors must be identical across seeds")
        mean_values = np.concatenate([(a + b) / 2 for a, b in zip(*vectors)])
        mean_sums = sums.mean(axis=0)
        sums_by_method[method] = mean_sums
        boot = mean_sums[draws].sum(axis=1) / denominator
        bootstrap[method] = boot
        item = _summary(mean_values, mean_sums, counts, boot, draws)
        item["per_seed"] = {}
        for s, training_seed in enumerate(TRAINING_SEEDS):
            seed_boot = sums[s, draws].sum(axis=1) / denominator
            item["per_seed"][str(training_seed)] = _summary(np.concatenate(vectors[s]), sums[s], counts, seed_boot, draws)
        seed_means[method] = [item["per_seed"][str(s)]["mean_geodesic_deg"] for s in TRAINING_SEEDS]
        item["seed_mean_range_deg"] = float(max(seed_means[method]) - min(seed_means[method]))
        item["folds"] = {}
        fold_means[method] = []
        for f in FOLD_IDS:
            selected = folds == f
            fold_mean = float(mean_sums[selected].sum() / counts[selected].sum())
            fold_means[method].append(fold_mean)
            item["folds"][str(f)] = {"users": int(selected.sum()), "windows": int(counts[selected].sum()),
                                     "mean_geodesic_deg": fold_mean,
                                     "per_seed_mean_geodesic_deg": [float(sums[s, selected].sum() / counts[selected].sum()) for s in range(2)]}
        summaries[method] = item
    primary, contrasts = "nll21_median", {}
    for comparator in METHODS:
        if comparator == primary:
            continue
        left, right = summaries[primary]["mean_geodesic_deg"], summaries[comparator]["mean_geodesic_deg"]
        delta = left - right
        relative = None if right == 0 else float((right - left) / right)
        delta_boot = bootstrap[primary] - bootstrap[comparator]
        right_boot = bootstrap[comparator]
        ratio_valid = right_boot > 0
        ratio_boot = (right_boot[ratio_valid] - bootstrap[primary][ratio_valid]) / right_boot[ratio_valid]
        fold_deltas = [a - b for a, b in zip(fold_means[primary], fold_means[comparator])]
        per_seed_deltas = [a - b for a, b in zip(seed_means[primary], seed_means[comparator])]
        improved_folds = sum(value < 0 for value in fold_deltas)
        positive_direction = all(value < 0 for value in per_seed_deltas)
        thresholds = bool(relative is not None and relative >= 0.03 and -delta >= 0.05
                          and positive_direction and improved_folds >= 2)
        contrasts[f"{primary}_minus_{comparator}"] = {
            "mean_delta_deg": float(delta), "absolute_improvement_deg": float(-delta),
            "relative_improvement_fraction": relative,
            "conditional_user_cluster_95_CI_delta_deg": [float(x) for x in np.percentile(delta_boot, [2.5, 97.5])],
            "conditional_user_cluster_95_CI_relative_improvement_fraction":
                ([float(x) for x in np.percentile(ratio_boot, [2.5, 97.5])] if ratio_boot.size else None),
            "relative_bootstrap_valid_draws": int(ratio_valid.sum()),
            "relative_bootstrap_zero_reference_draws": int((~ratio_valid).sum()),
            "per_seed_delta_deg": [float(x) for x in per_seed_deltas],
            "both_seed_directions_improve": positive_direction,
            "fold_delta_deg": [float(x) for x in fold_deltas], "folds_improved": improved_folds,
            "users_improved": int(np.sum(sums_by_method[primary] < sums_by_method[comparator])),
            "exploratory_pilot_thresholds_passed": thresholds,
        }
    passed = all(contrasts[f"{primary}_minus_{m}"]["exploratory_pilot_thresholds_passed"] for m in GATE_COMPARATORS)
    vector_better = summaries["vector_angle"]["mean_geodesic_deg"] < summaries[primary]["mean_geodesic_deg"]
    return {
        "schema_version": 1, "scope": "exploratory_outer_train_user_disjoint_inner_validation_only",
        "methods": summaries, "contrasts": contrasts,
        "primary_contrasts": [f"{primary}_minus_{m}" for m in ("static_mixture", "scalar_angle", "risk21")],
        "gate": {"conjunctive_comparators": list(GATE_COMPARATORS), "passed": passed,
                 "vector_control_has_lower_mean_error": bool(vector_better),
                 "primary_has_lowest_mean_among_recorded_methods": bool(all(
                     summaries[primary]["mean_geodesic_deg"] <= item["mean_geodesic_deg"]
                     for item in summaries.values())),
                 "gate_is_not_novelty_eligibility_or_acceptance": True,
                 "CI_is_reported_not_a_gate": True},
        "uncertainty": {"cluster": "whole_user_all_sessions_and_overlapping_origins", "resamples": resamples,
                        "seed": seed, "paired_draws_shared": True,
                        "estimand": "window_weighted_ratio_sum_errors_over_sum_counts",
                        "relative_ratio_recomputed_per_draw": True,
                        "CI_scope": "conditional_on_two_fixed_seeds_and_fixed_cross_validation_models_no_independent_retraining",
                        "two_seed_aggregation": "arithmetic_mean_of_paired_endpoint_errors_not_prediction_ensemble",
                        "p95_scope": "descriptive_percentile_of_paired_seed_mean_errors_no_window_independence"},
        "validation": {"users": 15, "folds": 3, "users_per_fold": 5,
                       "windows": int(counts.sum()), "row_order_binding_verified": True,
                       "direct_error_counts_sums_finite_range_verified": True,
                       "analytic_seed_identity_verified": True},
        "resources": _resource_summary(resources),
        "development": "not_read_or_evaluated_by_this_metrics_API",
        "final_holdout": "sealed_not_read_or_evaluated_by_this_metrics_API",
        "limits": ["single_conditional_Alyx_corpus", "selection_fold_results_not_independent_confirmation",
                   "different_training_objective_units_not_matched_loss_scaling",
                   "no_foundation_high_novelty_or_target_XR_claim", "official_sample_time_dispute_unresolved"],
    }

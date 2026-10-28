"""Private selector error validation and aggregate-only train/dev statistics.

The endpoint metric is the emitted quaternion's output20 geodesic angle.  The
two-seed row is an arithmetic mean of errors, never a prediction ensemble.
No function in this module accepts a final partition.
"""
import hashlib
import json
from numbers import Real

import h5py
import numpy as np

from .evaluation import cluster_intervals
from .stage import cache_groups


VARIANTS = ("h1_a0", "h20_a0", "h1_a19", "h20_a20", "complete_horizon", "motion_only")
TRAINING_SEEDS = (20261026, 20261027)
_CHUNK = 65536
_PARTITIONS = ("train", "development")


def method_name(variant, seed=None):
    """Return the stable private HDF/public aggregate method name."""
    if variant not in VARIANTS or (seed is not None and (type(seed) is not int or seed not in TRAINING_SEEDS)):
        raise ValueError("unsupported selector variant or training seed")
    suffix = "two_seed_mean_error" if seed is None else f"seed_{seed}"
    return f"Selector_{variant}_{suffix}"


def _partition(partition):
    if partition not in _PARTITIONS:
        raise PermissionError("selector metrics permit train/development only")


def _cache_blocks(cache, partition):
    _partition(partition)
    if cache is None:
        raise ValueError("a verified feature cache is required to bind flattened errors")
    blocks, position = [], 0
    for user, session, segment, group in cache_groups(cache, partition):
        origin = group["origin_index"]
        if origin.ndim != 1 or origin.dtype.kind not in "iu" or not len(origin):
            raise ValueError("invalid cache origin sequence")
        count = len(origin)
        blocks.append((user, session, segment, position, position + count, group))
        position += count
    if not position:
        raise ValueError("cannot summarize an empty feature partition")
    return blocks, position


def cache_row_order_sha256(cache, partition):
    """Fingerprint exact cache iteration order, private identities and origins.

    Store this digest on the private error and diagnostic partition groups. The
    digest is an integrity binding, not a substitute for the sealed cache hash.
    It reads origin indexes only, never quaternion/features/targets or final.
    """
    blocks, _ = _cache_blocks(cache, partition)
    digest = hashlib.sha256(b"abc_selector_row_order_v1\0")
    digest.update(partition.encode("utf-8") + b"\0")
    for user, session, segment, start, stop, group in blocks:
        identity = json.dumps([user, session, segment, stop - start], separators=(",", ":"),
                              ensure_ascii=True).encode("utf-8")
        digest.update(len(identity).to_bytes(8, "little") + identity)
        origin = group["origin_index"]
        for offset in range(0, len(origin), _CHUNK):
            values = np.asarray(origin[offset:offset + _CHUNK], dtype="<i8")
            if not np.array_equal(values, np.arange(399 + offset, 399 + offset + len(values), dtype=np.int64)):
                raise ValueError("cache origins must preserve every stride1 context in order")
            digest.update(values.tobytes())
    return digest.hexdigest()


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _counts_and_sums(user_sums, user_counts, methods, blocks):
    if not isinstance(user_counts, dict) or not user_counts:
        raise ValueError("private user counts must be a nonempty mapping")
    expected = {}
    for user, _, _, start, stop, _ in blocks:
        expected[user] = expected.get(user, 0) + stop - start
    for count in user_counts.values():
        _positive_integer(count, "private user count")
    if user_counts != expected:
        raise ValueError("private user counts differ from cache coverage")
    if not isinstance(user_sums, dict) or set(user_sums) != set(methods):
        raise ValueError("private user sums differ from the required method set")
    users = sorted(user_counts)
    counts = np.array([user_counts[user] for user in users], dtype=np.float64)
    columns = []
    for method in methods:
        stats = user_sums[method]
        if not isinstance(stats, dict) or set(stats) != set(users):
            raise ValueError("each method must cover the same private users")
        values = []
        for user, count in zip(users, counts):
            value = stats[user]
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
                    or value < 0 or value > (180 + 1e-5) * count):
                raise ValueError("invalid private user angular-error sum")
            values.append(value)
        columns.append(values)
    return users, counts, np.asarray(columns, dtype=np.float64).T


def _dataset(group, method, total):
    if not isinstance(group.get(method, getlink=True), h5py.HardLink):
        raise ValueError("error/diagnostic vectors must be local hard-linked datasets")
    data = group[method]
    if (not isinstance(data, h5py.Dataset) or data.shape != (total,) or data.dtype.kind != "f"
            or data.is_virtual or data.external):
        raise ValueError("angular-error vectors must have the exact flat cache row count")
    return data


def _angles(values):
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 180 + 1e-5):
        raise ValueError("angular errors must be finite degrees in [0,180]")
    return array


def _validated_vector(data, blocks, stats):
    values = _angles(data[:])
    actual = {}
    for user, _, _, start, stop, _ in blocks:
        actual[user] = actual.get(user, 0.0) + float(values[start:stop].sum(dtype=np.float64))
    if any(not np.isclose(actual[user], stats[user], rtol=2e-12, atol=1e-9) for user in actual):
        raise ValueError("private sums disagree with flat error rows in cache user order")
    return values


def _require_order(group, expected):
    if group.attrs.get("row_order_sha256") != expected:
        raise ValueError("flat errors/diagnostics are not bound to the same cache row order")


def _verify_seed_means(error_group, total, variants):
    for variant in variants:
        one, two = (_dataset(error_group, method_name(variant, seed), total) for seed in TRAINING_SEEDS)
        average = _dataset(error_group, method_name(variant), total)
        for start in range(0, total, _CHUNK):
            stop = min(total, start + _CHUNK)
            expected = (one[start:stop] + two[start:stop]) / 2.0
            if not np.array_equal(average[start:stop], expected):
                raise ValueError("two-seed row must be the exact paired arithmetic mean of errors")


def _method_summary(values, sums, counts, interval, macro_interval):
    return {
        "mean_geodesic_deg": float(sums.sum() / counts.sum()),
        "user_macro_mean_deg": float(np.mean(sums / counts)),
        "p95_geodesic_deg": float(np.percentile(values, 95)),
        "user_cluster_95_CI_mean_deg": interval.tolist(),
        "user_cluster_95_CI_user_macro_mean_deg": macro_interval.tolist(),
        "users": len(counts), "windows": int(counts.sum()),
    }


def _contrast(left, right, methods, sums, counts, bootstrap, partition):
    li, ri = methods.index(left), methods.index(right)
    delta = sums[:, li] - sums[:, ri]
    mean_delta = float(delta.sum() / counts.sum())
    comparator_mean = float(sums[:, ri].sum() / counts.sum())
    ci = np.percentile(bootstrap[:, li] - bootstrap[:, ri], [2.5, 97.5])
    absolute = -mean_delta
    relative = absolute / comparator_mean if comparator_mean else None
    improved = int(np.count_nonzero(delta < 0))
    checks = {"relative_improvement_at_least_3_percent": relative is not None and relative >= 0.03,
              "absolute_improvement_at_least_0_05_deg": absolute >= 0.05,
              "paired_CI_upper_delta_below_zero": bool(ci[1] < 0),
              "at_least_60_percent_users_improved": improved / len(counts) >= 0.60}
    margin = max(0.05, 0.03 * comparator_mean)
    equivalent = bool(ci[0] > -margin and ci[1] < margin)
    return {
        "mean_delta_deg": mean_delta,
        "user_cluster_95_CI_delta_deg": ci.tolist(),
        "absolute_improvement_deg": absolute, "relative_improvement_fraction": relative,
        "users_improved": improved, "users": len(counts),
        "users_improved_fraction": improved / len(counts),
        "practical_threshold_checks_descriptive": checks,
        "all_practical_thresholds_met_on_this_partition_descriptive": all(checks.values()),
        "final_success_criterion": "not_evaluated_final_untouched",
        "equivalence_margin_deg": margin,
        "conditional_development_equivalence_illustration": (
            "CI_inside_margin_conditional_development_only" if partition == "development" and equivalent
            else "inconclusive" if partition == "development" else "not_applicable_training_in_sample"),
        "interpretation": ("exploratory_development_not_confirmatory_H1_or_formal_final_equivalence"
                           if partition == "development" else "training_in_sample_descriptive_only"),
    }


def _diagnostics(diagnostic_group, error_group, total, expected_order, variants):
    if diagnostic_group is None:
        return {"status": "not_recorded", "interpretation": "surrogate_is_not_emitted_blend_loss"}
    _require_order(diagnostic_group, expected_order)
    names = {method_name(variant, seed) for variant in variants for seed in TRAINING_SEEDS}
    if set(diagnostic_group) != names:
        raise ValueError("diagnostics must cover the required variants and both fixed seeds")
    results = {}
    for variant in variants:
        diagnostics = []
        for seed in TRAINING_SEEDS:
            method = method_name(variant, seed)
            if not isinstance(diagnostic_group.get(method, getlink=True), h5py.HardLink):
                raise ValueError("diagnostic method groups must be local hard links")
            group = diagnostic_group[method]
            if not isinstance(group, h5py.Group) or not {"expected_loss_deg", "hard_error_deg"} <= set(group):
                raise ValueError("diagnostics require expected individual loss and hard-emitted error")
            expected = _dataset(group, "expected_loss_deg", total)
            hard = _dataset(group, "hard_error_deg", total)
            actual = _dataset(error_group, method, total)
            totals = np.zeros(6, dtype=np.float64)
            expected_sum, hard_sum, gaps, hard_gaps = 0.0, 0.0, [], []
            for start in range(0, total, _CHUNK):
                stop = min(total, start + _CHUNK)
                x, y, z = (_angles(data[start:stop]) for data in (expected, actual, hard))
                expected_sum += float(x.sum(dtype=np.float64))
                hard_sum += float(z.sum(dtype=np.float64))
                gap, hard_gap = x - y, z - y
                gaps.append(float(gap.sum(dtype=np.float64)))
                hard_gaps.append(float(hard_gap.sum(dtype=np.float64)))
                totals += [len(x), x.sum(), y.sum(), np.dot(x, x), np.dot(y, y), np.dot(x, y)]
            n, sx, sy, sxx, syy, sxy = totals
            variance_product = max(0.0, sxx - sx * sx / n) * max(0.0, syy - sy * sy / n)
            correlation = ((sxy - sx * sy / n) / np.sqrt(variance_product)) if variance_product > 0 else None
            diagnostics.append({
                "expected_individual_expert_loss_mean_deg": expected_sum / total,
                "actual_emitted_blend_error_mean_deg": float(sy / total),
                "surrogate_minus_emitted_blend_mean_deg": sum(gaps) / total,
                "hard_argmax_emitted_error_mean_deg": hard_sum / total,
                "hard_argmax_minus_emitted_blend_mean_deg": sum(hard_gaps) / total,
                "window_pooled_Pearson_expected_vs_emitted": (float(np.clip(correlation, -1, 1))
                                                            if correlation is not None else None),
            })
            results[method] = diagnostics[-1]
        # Means commute with the predeclared arithmetic average over fixed seeds.
        results[method_name(variant)] = {
            key: float(np.mean([item[key] for item in diagnostics]))
            for key in diagnostics[0] if not key.startswith("window_pooled_")
        }
    return {"status": "recorded", "methods": results,
            "interpretation": "paired_same_window_diagnostic_surrogate_not_actual_emitted_blend_loss",
            "gap_scope": "expected_individual_loss_is_not_a_bound_on_blend_error_gap_does_not_establish_causal_learnability",
            "correlation_scope": "post_hoc_descriptive_window_pooled_overlapping_windows_no_independent_window_inference",
            "two_seed_diagnostic": "arithmetic_mean_of_seed_diagnostic_means_not_prediction_ensemble"}


def summarize_selector_partition(error_group, user_sums, user_counts, diagnostic_group=None, *,
                                 partition="development", resamples=2000, seed=20261026, cache=None,
                                 baseline_error_group=None, baseline_user_sums=None, baseline_user_counts=None,
                                 variants=VARIANTS):
    """Validate private arrays/stats, then return aggregate metrics without IDs.

    Pass an already sealed/validated feature cache and the matching private
    analytic baseline group/statistics. Per-user identities and sums never occur
    in the returned object. Bootstrap uncertainty is conditional on these two
    fitted seeds; training randomness is recorded separately as seed spread.
    """
    _partition(partition)
    if (not isinstance(variants, (list, tuple)) or not variants or any(type(value) is not str for value in variants)
            or len(set(variants)) != len(variants)
            or any(variant not in VARIANTS for variant in variants) or not {"h1_a0", "h20_a0"} <= set(variants)):
        raise ValueError("variants must be a unique declared subset containing the fixed primary pair")
    _positive_integer(resamples, "bootstrap resamples")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    blocks, total = _cache_blocks(cache, partition)
    expected_order = cache_row_order_sha256(cache, partition)
    _require_order(error_group, expected_order)
    methods = [method_name(variant, training_seed) for variant in variants for training_seed in TRAINING_SEEDS]
    methods += [method_name(variant) for variant in variants]
    if set(error_group) != set(methods):
        raise ValueError("error group must contain the required variants, two seeds and their exact mean-error rows")
    users, counts, sums = _counts_and_sums(user_sums, user_counts, methods, blocks)
    _verify_seed_means(error_group, total, variants)
    baseline_arguments = (baseline_error_group, baseline_user_sums, baseline_user_counts)
    if any(value is None for value in baseline_arguments):
        raise ValueError("the train-fitted static baseline errors and matching private statistics are required")
    if baseline_user_counts != user_counts:
        raise ValueError("static baseline users/counts differ from selector coverage")
    if not isinstance(baseline_user_sums, dict) or "static_mixture" not in baseline_user_sums:
        raise ValueError("missing private static-mixture user statistics")
    static_stats = {"static_mixture": baseline_user_sums["static_mixture"]}
    _, _, static_sums = _counts_and_sums(static_stats, baseline_user_counts, ["static_mixture"], blocks)
    methods.append("static_mixture")
    sums = np.column_stack([sums, static_sums])
    intervals, bootstrap = cluster_intervals(sums, counts, resamples=resamples, seed=seed)
    indexes = np.random.default_rng(seed).integers(0, len(users), size=(resamples, len(users)))
    macro_intervals = np.percentile((sums / counts[:, None])[indexes].mean(axis=1), [2.5, 97.5], axis=0)
    result = {}
    for i, method in enumerate(methods):
        group, stats = ((baseline_error_group, static_stats) if method == "static_mixture" else (error_group, user_sums))
        values = _validated_vector(_dataset(group, method, total), blocks, stats[method])
        result[method] = _method_summary(values, sums[:, i], counts, intervals[:, i], macro_intervals[:, i])
    left = method_name("h20_a0")
    comparators = [method_name(variant) for variant in variants if variant != "h20_a0"] + ["static_mixture"]
    contrasts = {f"{left}_minus_{right}": _contrast(left, right, methods, sums, counts, bootstrap, partition)
                 for right in comparators}
    spread = {}
    for variant in variants:
        means = [result[method_name(variant, training_seed)]["mean_geodesic_deg"] for training_seed in TRAINING_SEEDS]
        spread[variant] = {"training_seeds": list(TRAINING_SEEDS), "per_seed_mean_geodesic_deg": means,
                           "min_seed_mean_deg": min(means), "max_seed_mean_deg": max(means),
                           "seed_mean_range_deg": max(means) - min(means),
                           "interpretation": "two_fixed_seed_spread_separate_from_conditional_user_cluster_CI"}
    return {
        "partition": partition, "feature_variants": list(variants),
        "methods": result, "contrasts": contrasts, "seed_spread": spread,
        "primary_contrast": f"{left}_minus_{method_name('h1_a0')}",
        "surrogate_diagnostics": _diagnostics(diagnostic_group, error_group, total, expected_order, variants),
        "two_seed_mean_interpretation": "arithmetic_mean_of_endpoint20_errors_not_prediction_ensemble",
        "uncertainty": {"cluster": "user_with_all_sessions_and_overlapping_windows", "resamples": resamples,
                        "seed": seed, "estimand": "window_weighted_ratio_sum_errors_over_sum_windows",
                        "paired_draws_shared_by_all_methods": True,
                        "CI_scope": "conditional_on_two_fixed_fitted_seeds_not_training_population_or_seed_randomness",
                        "p95_scope": "descriptive_pooled_window_percentile_no_independent_window_CI"},
        "interpretation": ("exploratory_development_only_no_confirmatory_H1_or_final_success_claim"
                           if partition == "development" else "training_in_sample_descriptive_only"),
        "final_holdout": "untouched_not_evaluated_by_this_API",
    }

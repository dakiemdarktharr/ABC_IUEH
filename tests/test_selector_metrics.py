"""Synthetic recording tests; no human arrays or final holdout are read."""
import json

import h5py
import numpy as np
import pytest

from abc_iueh.selector_metrics import (TRAINING_SEEDS, VARIANTS, cache_row_order_sha256,
                                       method_name, summarize_selector_partition)


def _fixture(tmp_path, *, diagnostics=True):
    cache = h5py.File(tmp_path / "synthetic_cache.h5", "w")
    for partition in ("train", "development"):
        # Several sessions of one user must remain one bootstrap cluster.
        for user, session, count in (("private_long", "session_a", 3),
                                     ("private_long", "session_b", 1),
                                     ("private_short", "session_c", 1)):
            cache.create_dataset(f"{partition}/{user}/{session}/0/origin_index", data=np.arange(count) + 399)
    store = h5py.File(tmp_path / "synthetic_errors.h5", "w")
    errors, baseline = store.create_group("errors"), store.create_group("baseline")
    diag = store.create_group("diagnostics") if diagnostics else None
    counts = {"private_long": 4, "private_short": 1}
    sums = {}
    for variant in VARIANTS:
        values = ([np.array([1., 1, 1, 1, 9]), np.array([3., 3, 3, 3, 7])]
                  if variant == "h20_a0" else [np.array([2., 2, 2, 2, 8]), np.array([4., 4, 4, 4, 6])])
        for seed, value in zip(TRAINING_SEEDS, values):
            method = method_name(variant, seed)
            errors[method] = value
            sums[method] = {"private_long": float(value[:4].sum()), "private_short": float(value[4])}
            if diag is not None:
                group = diag.create_group(method)
                group["expected_loss_deg"] = value + 1
                group["hard_error_deg"] = value + 0.5
        mean = (values[0] + values[1]) / 2
        errors[method_name(variant)] = mean
        sums[method_name(variant)] = {"private_long": float(mean[:4].sum()), "private_short": float(mean[4])}
    baseline["static_mixture"] = np.full(5, 2.0)
    baseline_sums = {"static_mixture": {"private_long": 8., "private_short": 2.}}
    order = cache_row_order_sha256(cache, "development")
    errors.attrs["row_order_sha256"] = order
    if diag is not None:
        diag.attrs["row_order_sha256"] = order
    return cache, store, errors, sums, counts, diag, baseline, baseline_sums


def _summary(fixture, **kwargs):
    cache, _, errors, sums, counts, diag, baseline, baseline_sums = fixture
    return summarize_selector_partition(errors, sums, counts, diag, cache=cache,
                                         baseline_error_group=baseline, baseline_user_sums=baseline_sums,
                                         baseline_user_counts=counts.copy(), **kwargs)


@pytest.fixture
def synthetic(tmp_path):
    value = _fixture(tmp_path)
    yield value
    value[1].close()
    value[0].close()


def test_unequal_user_lengths_use_ratio_and_whole_user_paired_CI(synthetic):
    result = _summary(synthetic)
    h1 = result["methods"][method_name("h1_a0")]
    h20 = result["methods"][method_name("h20_a0")]
    # Long user: h1=3, h20=2; short user: h1=7, h20=8.
    assert h1["mean_geodesic_deg"] == pytest.approx(19 / 5)
    assert h1["user_macro_mean_deg"] == pytest.approx(5)
    assert h1["p95_geodesic_deg"] == pytest.approx(6.2)
    assert h20["mean_geodesic_deg"] == pytest.approx(16 / 5)
    primary = result["contrasts"][result["primary_contrast"]]
    assert primary["mean_delta_deg"] == pytest.approx(-3 / 5)
    # With two complete-user draws the possible ratios are -1, -3/5, +1.
    # Window resampling or user-macro bootstrapping would estimate something else.
    assert primary["user_cluster_95_CI_delta_deg"] == pytest.approx([-1, 1])
    assert primary["users_improved"] == 1
    assert primary["users"] == 2
    assert primary["relative_improvement_fraction"] == pytest.approx(3 / 19)
    assert primary["absolute_improvement_deg"] == pytest.approx(0.6)
    assert not primary["all_practical_thresholds_met_on_this_partition_descriptive"]
    assert primary["conditional_development_equivalence_illustration"] == "inconclusive"
    assert "final_success_criterion" in primary


def test_two_seed_mean_averages_paired_errors_and_reports_seed_spread_separately(synthetic):
    result = _summary(synthetic)
    means = result["seed_spread"]["h20_a0"]
    assert means["training_seeds"] == list(TRAINING_SEEDS)
    assert means["per_seed_mean_geodesic_deg"] == pytest.approx([2.6, 3.8])
    assert means["seed_mean_range_deg"] == pytest.approx(1.2)
    assert result["methods"][method_name("h20_a0")]["mean_geodesic_deg"] == pytest.approx(3.2)
    assert "not_prediction_ensemble" in result["two_seed_mean_interpretation"]
    assert len(result["methods"]) == 19
    assert len(result["contrasts"]) == 6
    assert "static_mixture" in result["methods"]


def test_diagnostics_compare_paired_surrogate_and_actual_blend(synthetic):
    result = _summary(synthetic)
    diagnostic = result["surrogate_diagnostics"]
    for method in (method_name("h20_a0", TRAINING_SEEDS[0]), method_name("h20_a0")):
        item = diagnostic["methods"][method]
        assert item["surrogate_minus_emitted_blend_mean_deg"] == pytest.approx(1)
        assert item["hard_argmax_minus_emitted_blend_mean_deg"] == pytest.approx(0.5)
        assert item["expected_individual_expert_loss_mean_deg"] == pytest.approx(
            item["actual_emitted_blend_error_mean_deg"] + 1)
    assert diagnostic["methods"][method_name("h20_a0", TRAINING_SEEDS[0])][
        "window_pooled_Pearson_expected_vs_emitted"] == pytest.approx(1)
    assert "post_hoc_descriptive" in diagnostic["correlation_scope"]


def test_public_summary_has_no_private_identity_or_per_user_stats(synthetic):
    serialized = json.dumps(_summary(synthetic), allow_nan=False)
    for private in ("private_long", "private_short", "session_a", "user_sums", "user_counts", "synthetic_cache.h5"):
        assert private not in serialized
    assert "exploratory_development_only_no_confirmatory_H1" in serialized
    assert "untouched_not_evaluated" in serialized


def test_same_global_sum_with_wrong_user_row_order_is_rejected(synthetic):
    errors = synthetic[2]
    method = method_name("h1_a0", TRAINING_SEEDS[0])
    value = errors[method][:]
    value[[0, 4]] = value[[4, 0]]
    errors[method][:] = value
    # Update the rowwise two-seed mean: its arithmetic is still valid, but
    # coverage and global sums alone cannot detect the cross-user permutation.
    errors[method_name("h1_a0")][:] = (value + errors[method_name("h1_a0", TRAINING_SEEDS[1])][:]) / 2
    with pytest.raises(ValueError, match="private sums disagree"):
        _summary(synthetic)


def test_seed_mean_permutation_is_rejected_even_if_total_is_unchanged(synthetic):
    mean = synthetic[2][method_name("h20_a0")]
    values = mean[:]
    mean[:] = np.roll(values, 1)
    with pytest.raises(ValueError, match="exact paired arithmetic mean"):
        _summary(synthetic)


@pytest.mark.parametrize("target", ["errors", "diagnostics"])
def test_row_order_binding_is_required(synthetic, target):
    group = synthetic[2] if target == "errors" else synthetic[5]
    group.attrs["row_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="same cache row order"):
        _summary(synthetic)


def test_cache_origins_cannot_be_reordered(synthetic):
    origin = synthetic[0]["development/private_long/session_a/0/origin_index"]
    origin[:] = origin[:][::-1]
    with pytest.raises(ValueError, match="stride1 context in order"):
        _summary(synthetic)


def test_wrong_counts_fail_before_reporting(synthetic):
    synthetic[4]["private_long"] = 3
    with pytest.raises(ValueError, match="counts differ from cache"):
        _summary(synthetic)


def test_baseline_user_count_mismatch_fails(synthetic):
    cache, _, errors, sums, counts, diag, baseline, baseline_sums = synthetic
    with pytest.raises(ValueError, match="static baseline users/counts differ"):
        summarize_selector_partition(errors, sums, counts, diag, cache=cache,
                                     baseline_error_group=baseline, baseline_user_sums=baseline_sums,
                                     baseline_user_counts={"private_long": 3, "private_short": 2})


@pytest.mark.parametrize("bad", [np.nan, -1., 180.1])
def test_invalid_seed_error_cannot_be_hidden_by_sufficient_statistics(synthetic, bad):
    method = method_name("h1_a0", TRAINING_SEEDS[0])
    synthetic[2][method][0] = bad
    value = (synthetic[2][method][:] + synthetic[2][method_name("h1_a0", TRAINING_SEEDS[1])][:]) / 2
    synthetic[2][method_name("h1_a0")][:] = value
    with pytest.raises(ValueError):
        _summary(synthetic)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_bootstrap_resample_count_requires_positive_integer(synthetic, bad):
    with pytest.raises(ValueError, match="positive integer"):
        _summary(synthetic, resamples=bad)


def test_final_is_rejected_before_traversal(synthetic):
    with pytest.raises(PermissionError, match="train/development only"):
        _summary(synthetic, partition="final")
    with pytest.raises(PermissionError, match="train/development only"):
        cache_row_order_sha256(synthetic[0], "final")


def test_training_summary_cannot_be_called_development_or_final_evidence(synthetic):
    errors, diag = synthetic[2], synthetic[5]
    order = cache_row_order_sha256(synthetic[0], "train")
    errors.attrs["row_order_sha256"] = diag.attrs["row_order_sha256"] = order
    result = _summary(synthetic, partition="train")
    primary = result["contrasts"][result["primary_contrast"]]
    assert result["interpretation"] == "training_in_sample_descriptive_only"
    assert primary["conditional_development_equivalence_illustration"] == "not_applicable_training_in_sample"


def test_equivalence_is_only_conditional_development_illustration(synthetic):
    errors, sums, diag = synthetic[2], synthetic[3], synthetic[5]
    for training_seed in (*TRAINING_SEEDS, None):
        h1, h20 = method_name("h1_a0", training_seed), method_name("h20_a0", training_seed)
        errors[h20][:] = errors[h1][:]
        sums[h20] = sums[h1].copy()
        if training_seed is not None:
            diag[h20]["expected_loss_deg"][:] = errors[h20][:] + 1
            diag[h20]["hard_error_deg"][:] = errors[h20][:] + 0.5
    result = _summary(synthetic)
    primary = result["contrasts"][result["primary_contrast"]]
    assert primary["equivalence_margin_deg"] == pytest.approx(0.03 * 3.8)
    assert primary["conditional_development_equivalence_illustration"] == "CI_inside_margin_conditional_development_only"
    assert "not_confirmatory_H1_or_formal_final_equivalence" in primary["interpretation"]
    assert not primary["all_practical_thresholds_met_on_this_partition_descriptive"]


def test_even_all_numerical_development_thresholds_do_not_decide_final_success(synthetic):
    errors, sums, diag = synthetic[2], synthetic[3], synthetic[5]
    for training_seed in (*TRAINING_SEEDS, None):
        h1, h20 = method_name("h1_a0", training_seed), method_name("h20_a0", training_seed)
        errors[h20][:] = errors[h1][:] - 0.2
        sums[h20] = {"private_long": float(errors[h20][:4].sum()), "private_short": float(errors[h20][4])}
        if training_seed is not None:
            diag[h20]["expected_loss_deg"][:] = errors[h20][:] + 1
            diag[h20]["hard_error_deg"][:] = errors[h20][:] + 0.5
    # Compute the stored paired mean using the same arithmetic as the writer.
    errors[method_name("h20_a0")][:] = sum(errors[method_name("h20_a0", seed)][:] for seed in TRAINING_SEEDS) / 2
    sums[method_name("h20_a0")] = {"private_long": float(errors[method_name("h20_a0")][:4].sum()),
                                   "private_short": float(errors[method_name("h20_a0")][4])}
    result = _summary(synthetic)
    primary = result["contrasts"][result["primary_contrast"]]
    assert primary["all_practical_thresholds_met_on_this_partition_descriptive"]
    assert primary["final_success_criterion"] == "not_evaluated_final_untouched"
    assert "no_confirmatory_H1_or_final_success_claim" in result["interpretation"]


def test_declared_primary_pair_subset_reuses_same_recording_contract(synthetic):
    errors, sums, diag = synthetic[2], synthetic[3], synthetic[5]
    kept = ("h1_a0", "h20_a0")
    for variant in VARIANTS:
        if variant in kept:
            continue
        for training_seed in (*TRAINING_SEEDS, None):
            method = method_name(variant, training_seed)
            del errors[method]
            del sums[method]
            if training_seed is not None:
                del diag[method]
    result = _summary(synthetic, variants=kept)
    assert result["feature_variants"] == list(kept)
    assert len(result["methods"]) == 7
    assert len(result["contrasts"]) == 2
    assert result["primary_contrast"] in result["contrasts"]

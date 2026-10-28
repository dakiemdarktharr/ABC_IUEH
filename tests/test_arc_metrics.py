"""Synthetic paired-user recordings only; no human data or final access."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from abc_iueh.arc_metrics import (ARMS, METHODS, TRAINING_SEEDS, row_order_digest,
                                  summarize_arc_pilot)


@pytest.fixture
def recording(tmp_path):
    users = [f"private_user_{i:02}" for i in range(15)]
    counts = {u: (4 if i < 10 else 1) for i, u in enumerate(users)}
    folds = {u: i % 3 for i, u in enumerate(users)}
    orders = {u: hashlib.sha256(f"synthetic_origin_order_{i}".encode()).hexdigest()
              for i, u in enumerate(users)}
    with h5py.File(tmp_path / "synthetic_arc_errors.h5", "w") as store:
        root = store.create_group("errors")
        root.attrs["row_order_sha256"] = row_order_digest(orders)
        sums = {}
        for method in METHODS:
            sums[method] = {}
            for s, training_seed in enumerate(TRAINING_SEEDS):
                seed_key = str(training_seed)
                group = root.require_group(f"{method}/{seed_key}")
                sums[method][seed_key] = {}
                for i, user in enumerate(users):
                    # Primary: long user seed errors 1,3 => 2; short 9,7 => 8.
                    # Learned reference: long 2,4 => 3; short 8,6 => 7.
                    if method == "nll21_median":
                        error = ([1., 3.][s] if i < 10 else [9., 7.][s])
                    elif method in ("scalar_angle", "scalar_pinball", "risk21", "vector_angle"):
                        error = ([2., 4.][s] if i < 10 else [8., 6.][s])
                    else:
                        error = 3. if i < 10 else 7.
                    value = np.full(counts[user], error, dtype=np.float64)
                    dataset = group.create_dataset(user, data=value)
                    dataset.attrs["row_order_sha256"] = orders[user]
                    sums[method][seed_key][user] = float(value.sum())
        yield root, sums, counts, folds, orders


def summary(recording, **kwargs):
    root, sums, counts, folds, orders = recording
    return summarize_arc_pilot(root, sums, counts, folds, row_order_sha256=orders, **kwargs)


def test_whole_user_ratio_paired_seed_errors_and_relative_draw_ratios(recording):
    result = summary(recording)
    primary = result["methods"]["nll21_median"]
    reference = result["methods"]["scalar_angle"]
    assert primary["mean_geodesic_deg"] == pytest.approx(120 / 45)
    assert primary["user_macro_mean_deg"] == pytest.approx(4)
    assert primary["p95_geodesic_deg"] == pytest.approx(8)
    assert primary["per_seed"]["20261026"]["mean_geodesic_deg"] == pytest.approx(85 / 45)
    assert primary["per_seed"]["20261027"]["mean_geodesic_deg"] == pytest.approx(155 / 45)
    assert reference["mean_geodesic_deg"] == pytest.approx(155 / 45)
    contrast = result["contrasts"]["nll21_median_minus_scalar_angle"]
    assert contrast["mean_delta_deg"] == pytest.approx(-35 / 45)
    assert contrast["relative_improvement_fraction"] == pytest.approx(35 / 155)
    assert contrast["users_improved"] == 10
    assert contrast["folds_improved"] == 3
    # Independently recompute exactly paired user-cluster ratios, preserving unequal counts.
    counts = np.array(list(recording[2].values()))
    p_sums = counts * np.array([2.] * 10 + [8.] * 5)
    r_sums = counts * np.array([3.] * 10 + [7.] * 5)
    draws = np.random.default_rng(20261028).integers(0, 15, size=(2000, 15))
    denom = counts[draws].sum(axis=1)
    p_boot, r_boot = p_sums[draws].sum(axis=1) / denom, r_sums[draws].sum(axis=1) / denom
    assert contrast["conditional_user_cluster_95_CI_delta_deg"] == pytest.approx(
        np.percentile(p_boot - r_boot, [2.5, 97.5]))
    assert contrast["conditional_user_cluster_95_CI_relative_improvement_fraction"] == pytest.approx(
        np.percentile((r_boot - p_boot) / r_boot, [2.5, 97.5]))
    assert primary["conditional_user_cluster_95_CI_user_macro_mean_deg"] == pytest.approx(
        np.percentile(np.array([2.] * 10 + [8.] * 5)[draws].mean(axis=1), [2.5, 97.5]))
    assert "not_prediction_ensemble" in result["uncertainty"]["two_seed_aggregation"]


def test_gate_requires_all_comparators_and_both_seed_directions(recording):
    result = summary(recording)
    # NLL seed2 has the same pooled error as the fixed static baseline: strict direction fails.
    assert not result["contrasts"]["nll21_median_minus_static_mixture"]["both_seed_directions_improve"]
    assert not result["gate"]["passed"]
    assert result["gate"]["CI_is_reported_not_a_gate"]
    assert result["primary_contrasts"] == ["nll21_median_minus_static_mixture",
                                           "nll21_median_minus_scalar_angle",
                                           "nll21_median_minus_risk21"]


def test_passing_pilot_gate_does_not_hide_a_better_vector_control(recording):
    for seed in TRAINING_SEEDS:
        for user in recording[2]:
            for method, error in (("nll21_median", .5), ("vector_angle", .4)):
                recording[0][f"{method}/{seed}"][user][:] = error
                recording[1][method][str(seed)][user] = error * recording[2][user]
    result = summary(recording)
    assert result["gate"]["passed"]
    assert result["gate"]["vector_control_has_lower_mean_error"]
    assert not result["gate"]["primary_has_lowest_mean_among_recorded_methods"]
    assert result["gate"]["gate_is_not_novelty_eligibility_or_acceptance"]


def test_summary_removes_private_identity_and_per_user_statistics(recording):
    serialized = json.dumps(summary(recording), allow_nan=False)
    for forbidden in ("private_user", "synthetic_arc", "user_sums", "user_counts", "synthetic_origin"):
        assert forbidden not in serialized
    assert "no_independent_retraining" in serialized
    assert "sealed_not_read_or_evaluated" in serialized
    assert summary(recording)["resources"]["strong_same_wall_compute_claim_allowed"] is False


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 180.01])
def test_invalid_values_cannot_be_hidden_in_private_sums(recording, bad):
    root, sums, _, _, _ = recording
    user = sorted(recording[2])[0]
    root["nll21_median/20261026"][user][0] = bad
    sums["nll21_median"]["20261026"][user] = float(np.sum(root["nll21_median/20261026"][user][:]))
    with pytest.raises(ValueError, match="finite geodesic"):
        summary(recording)


def test_wrong_sums_and_counts_are_rejected(recording):
    recording[1]["nll21_median"]["20261026"][sorted(recording[2])[0]] += 1
    with pytest.raises(ValueError, match="sums disagree"):
        summary(recording)


def test_wrong_origin_coverage_count_is_rejected(recording):
    recording[2][sorted(recording[2])[0]] += 1
    with pytest.raises(ValueError, match="row counts differ"):
        summary(recording)


def test_matching_global_sum_but_cross_user_permutation_is_rejected(recording):
    root = recording[0]["nll21_median/20261026"]
    users = sorted(recording[2])
    first, last = float(root[users[0]][0]), float(root[users[-1]][0])
    root[users[0]][0], root[users[-1]][0] = last, first
    with pytest.raises(ValueError, match="sums disagree"):
        summary(recording)


@pytest.mark.parametrize("where", ["root", "dataset"])
def test_per_user_and_root_row_order_bindings_are_required(recording, where):
    target = recording[0] if where == "root" else recording[0]["nll21_median/20261026"][sorted(recording[2])[0]]
    target.attrs["row_order_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="row.order"):
        summary(recording)


def test_row_order_digest_is_order_independent_and_strict():
    assert row_order_digest({"x": "a" * 64, "y": "b" * 64}) == row_order_digest({"y": "b" * 64, "x": "a" * 64})
    with pytest.raises(ValueError):
        row_order_digest({"private": "not_sha256"})


def test_missing_method_seed_user_or_unbalanced_fold_is_rejected(recording):
    recording[3][sorted(recording[2])[0]] = 1
    with pytest.raises(ValueError, match="five whole users"):
        summary(recording)


@pytest.mark.parametrize("level", ["method", "seed", "user"])
def test_missing_recording_members_fail_closed(recording, level):
    if level == "method":
        del recording[0]["nll21_median"]
    elif level == "seed":
        del recording[0]["nll21_median/20261026"]
    else:
        del recording[0]["nll21_median/20261026"][sorted(recording[2])[0]]
    with pytest.raises(ValueError, match="exactly|both fixed seeds"):
        summary(recording)


def test_deterministic_baseline_cannot_have_seed_differences(recording):
    user = sorted(recording[2])[0]
    ds = recording[0]["static_mixture/20261027"][user]
    ds[:] = ds[:] + 1
    recording[1]["static_mixture"]["20261027"][user] = float(ds[:].sum())
    with pytest.raises(ValueError, match="identical across seeds"):
        summary(recording)


def test_zero_reference_ratios_report_undefined_without_nan(recording):
    for seed in TRAINING_SEEDS:
        for user in recording[2]:
            recording[0][f"risk21/{seed}"][user][:] = 0
            recording[1]["risk21"][str(seed)][user] = 0.
    result = summary(recording)
    contrast = result["contrasts"]["nll21_median_minus_risk21"]
    assert contrast["relative_improvement_fraction"] is None
    assert contrast["conditional_user_cluster_95_CI_relative_improvement_fraction"] is None
    assert contrast["relative_bootstrap_zero_reference_draws"] == 2000
    assert not result["gate"]["passed"]
    json.dumps(result, allow_nan=False)


def resource_fixture(multiplier=1.):
    return {arm: [{"fold": fold, "seed": seed, "training_seconds": multiplier if arm == "risk21" else 1.,
                   "validation_seconds": .1, "peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
                  for fold in range(3) for seed in TRAINING_SEEDS] for arm in ARMS}


def test_equal_exposure_does_not_imply_equal_wall_compute(recording):
    resources = resource_fixture(1.21)
    result = summary(recording, resources=resources)
    assert not result["resources"]["strong_same_wall_compute_claim_allowed"]
    assert result["resources"]["maximum_to_minimum_total_training_time_ratio"] == pytest.approx(1.21)
    assert not result["resources"]["static_neural_compute_matched"]
    resources["risk21"][0]["private_path"] = "forbidden"
    with pytest.raises(ValueError, match="public.safe whitelist"):
        summary(recording, resources=resources)


def test_resource_records_require_unique_fold_seed_cells(recording):
    records = resource_fixture()
    records["nll21"][0] = records["nll21"][1].copy()
    with pytest.raises(ValueError, match="unique and fixed"):
        summary(recording, resources=records)


def test_frozen_json_and_metrics_contract_agree_before_fitting():
    from abc_iueh.arc_metrics import CONV_DENSE_MACS, PARAMETERS
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/arc_pilot.json").read_text())
    assert config["evaluation"]["methods"] == list(METHODS)
    assert config["arms"] == list(ARMS)
    assert config["training"]["seeds"] == list(TRAINING_SEEDS)
    assert config["training"]["steps"] == 5000
    assert config["model"]["parameters"] == PARAMETERS
    assert config["model"]["full_stateless_conv_dense_MACs"] == CONV_DENSE_MACS
    assert config["equal_compute_secondary"]["selection_uses_losses_or_accuracy"] is False
    assert config["development_signal_access_allowed"] is config["final_access_allowed"] is False


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_bootstrap_draw_count_is_strict(recording, bad):
    with pytest.raises(ValueError, match="positive integer"):
        summary(recording, resamples=bad)

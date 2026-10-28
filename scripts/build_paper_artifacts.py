#!/usr/bin/env python3
"""Build public aggregate-only figures and tables for the exploratory paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


EXPECTED_WINDOWS = {"train": 3_330_312, "development": 1_189_027}
EXPECTED_USERS = {"train": 15, "development": 5}
EXPECTED_VARIANTS = (
    "h1_a0",
    "h20_a0",
    "h1_a19",
    "h20_a20",
    "complete_horizon",
    "motion_only",
)
PRIMARY_CONTRAST = (
    "Selector_h20_a0_two_seed_mean_error_minus_"
    "Selector_h1_a0_two_seed_mean_error"
)
DELTA_CI_KEY = "user_cluster_95_CI_delta_deg"


class ArtifactError(ValueError):
    """Raised when an aggregate does not match the recorded study contract."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"{name} must be finite")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ArtifactError(f"{label} input must be an aggregate JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not read {label} aggregate JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} aggregate root must be an object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{name} must be an object")
    return value


def _check_ci(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ArtifactError(f"{name} must contain exactly two CI endpoints")
    low = _finite_number(value[0], f"{name}[0]")
    high = _finite_number(value[1], f"{name}[1]")
    if low > high:
        raise ArtifactError(f"{name} lower endpoint exceeds upper endpoint")
    return low, high


def _metric(partition: dict[str, Any], method: str, split: str) -> dict[str, Any]:
    methods = _mapping(partition.get("methods"), f"{split}.methods")
    if method not in methods:
        raise ArtifactError(f"{split} is missing required method {method}")
    value = _mapping(methods[method], f"{split}.{method}")
    for field in ("mean_geodesic_deg", "p95_geodesic_deg"):
        _finite_number(value.get(field), f"{split}.{method}.{field}")
    if value["p95_geodesic_deg"] < value["mean_geodesic_deg"]:
        raise ArtifactError(f"{split}.{method} has P95 below its mean")
    return value


def _validate_partition(partition: dict[str, Any], split: str, *, expect_ci: bool = True) -> None:
    expected_n = EXPECTED_WINDOWS[split]
    expected_u = EXPECTED_USERS[split]
    methods = _mapping(partition.get("methods"), f"{split}.methods")
    if not methods:
        raise ArtifactError(f"{split}.methods must not be empty")
    for method, raw in methods.items():
        item = _mapping(raw, f"{split}.{method}")
        if item.get("users") != expected_u or item.get("windows") != expected_n:
            raise ArtifactError(
                f"{split}.{method} has unexpected users/windows; expected "
                f"{expected_u}/{expected_n}"
            )
        _metric(partition, method, split)
        if expect_ci and "user_cluster_95_CI_mean_deg" in item:
            _check_ci(item["user_cluster_95_CI_mean_deg"], f"{split}.{method}.mean_CI")


def _validate_contrasts(partition: dict[str, Any], split: str) -> None:
    contrasts = _mapping(partition.get("contrasts"), f"{split}.contrasts")
    if PRIMARY_CONTRAST not in contrasts:
        raise ArtifactError(f"{split} is missing its declared h20_a0 minus h1_a0 contrast")
    item = _mapping(contrasts[PRIMARY_CONTRAST], f"{split}.primary_contrast")
    mean_delta = _finite_number(item.get("mean_delta_deg"), f"{split}.primary.mean_delta_deg")
    del mean_delta
    _check_ci(item.get(DELTA_CI_KEY), f"{split}.primary.{DELTA_CI_KEY}")
    if item.get("users") != EXPECTED_USERS[split]:
        raise ArtifactError(f"{split}.primary contrast has an unexpected user count")
    if item.get("final_success_criterion") != "not_evaluated_final_untouched":
        raise ArtifactError(f"{split}.primary contrast does not preserve the sealed-final scope")


def _validate_uncertainty(partition: dict[str, Any], split: str) -> None:
    uncertainty = _mapping(partition.get("uncertainty"), f"{split}.uncertainty")
    if uncertainty.get("estimand") != "window_weighted_ratio_sum_errors_over_sum_windows":
        raise ArtifactError(f"{split} does not record the required window-weighted ratio estimand")
    if uncertainty.get("resamples") != 2000 or uncertainty.get("seed") != 20261026:
        raise ArtifactError(f"{split} does not match the recorded user-bootstrap settings")
    if uncertainty.get("paired_draws_shared_by_all_methods") is not True:
        raise ArtifactError(f"{split} does not record paired user-cluster bootstrap draws")


def _validate_main(main: dict[str, Any]) -> None:
    if main.get("stage") != "train_development_selector_only":
        raise ArtifactError("--main is not the R80 train/development selector aggregate")
    if main.get("evidence_length") != 80:
        raise ArtifactError("--main must have evidence_length=80")
    variants = main.get("variants")
    if not isinstance(variants, list) or len(variants) != len(EXPECTED_VARIANTS) or set(variants) != set(EXPECTED_VARIANTS):
        raise ArtifactError("--main variant list differs from the fixed six-variant study")
    if main.get("seeds") != [20261026, 20261027]:
        raise ArtifactError("--main seed list differs from the two fixed seeds")
    if main.get("windows") != EXPECTED_WINDOWS:
        raise ArtifactError("--main window counts do not match the locked train/development counts")
    if main.get("final_holdout_evaluated") is not False:
        raise ArtifactError("--main does not certify that final remained unevaluated")
    audit = _mapping(main.get("audit"), "main.audit")
    if audit.get("final_quaternion_datasets_read") != 0:
        raise ArtifactError("--main reports final-partition quaternion reads")

    results = _mapping(main.get("results"), "main.results")
    for split in ("train", "development"):
        partition = _mapping(results.get(split), f"main.results.{split}")
        _validate_partition(partition, split)
        _validate_contrasts(partition, split)
        _validate_uncertainty(partition, split)
        if partition.get("final_holdout") != "untouched_not_evaluated_by_this_API":
            raise ArtifactError(f"main {split} partition does not preserve the sealed-final scope")
        for variant in EXPECTED_VARIANTS:
            method = f"Selector_{variant}_two_seed_mean_error"
            item = _metric(partition, method, split)
            _check_ci(item.get("user_cluster_95_CI_mean_deg"), f"{split}.{method}.mean_CI")
        static = _metric(partition, "static_mixture", split)
        _check_ci(static.get("user_cluster_95_CI_mean_deg"), f"{split}.static_mixture.mean_CI")


def _validate_baselines(baselines: dict[str, Any]) -> None:
    if baselines.get("scope") != "WhoIsAlyx_train_development_pilot_not_official_ABC_score":
        raise ArtifactError("--baselines has an unknown scope")
    analytic = _mapping(baselines.get("analytic"), "baselines.analytic")
    neural = _mapping(baselines.get("neural"), "baselines.neural")
    for split in ("train", "development"):
        analytic_part = _mapping(analytic.get("results"), "baselines.analytic.results")
        neural_results = _mapping(neural.get("results"), "baselines.neural.results")
        a_partition = _mapping(analytic_part.get(split), f"baselines.analytic.{split}")
        n_partition = _mapping(neural_results.get(split), f"baselines.neural.{split}")
        a_methods = _mapping(a_partition.get("methods"), f"baselines.analytic.{split}.methods")
        n_methods = _mapping(n_partition.get("methods"), f"baselines.neural.{split}.methods")
        for method in ("ZV", "CAV_W1", "CAV_W5", "CAV_W20", "static_mixture"):
            item = _mapping(a_methods.get(method), f"baseline.{split}.{method}")
            if item.get("users") != EXPECTED_USERS[split] or item.get("windows") != EXPECTED_WINDOWS[split]:
                raise ArtifactError(f"baseline {split}.{method} has unexpected counts")
            _metric(a_partition, method, split)
        mlp = _mapping(n_methods.get("MLP_two_seed_mean_error"), f"baseline.{split}.MLP_two_seed_mean_error")
        if mlp.get("users") != EXPECTED_USERS[split] or mlp.get("windows") != EXPECTED_WINDOWS[split]:
            raise ArtifactError(f"baseline {split}.MLP has unexpected counts")
        _metric(n_partition, "MLP_two_seed_mean_error", split)
        if baselines.get("final_validation", {}).get("final_forecast_evaluation") is not False:
            raise ArtifactError("--baselines does not certify that final forecast evaluation was not run")


def _validate_sensitivity(sensitivity: dict[str, Any]) -> None:
    if sensitivity.get("stage") != "train_development_selector_R_sensitivity_only":
        raise ArtifactError("--sensitivity has an unknown stage")
    if sensitivity.get("final_holdout_evaluated") is not False:
        raise ArtifactError("--sensitivity does not certify that final remained unevaluated")
    results_by_r = _mapping(sensitivity.get("results_by_R"), "sensitivity.results_by_R")
    if set(results_by_r) != {"128", "256"}:
        raise ArtifactError("--sensitivity must contain exactly R=128 and R=256")
    for r_text in ("128", "256"):
        record = _mapping(results_by_r[r_text], f"sensitivity.R{r_text}")
        r = int(r_text)
        if record.get("evidence_length") != r:
            raise ArtifactError(f"sensitivity R={r} evidence length is inconsistent")
        if record.get("variants") != ["h1_a0", "h20_a0"]:
            raise ArtifactError(f"sensitivity R={r} has unexpected variants")
        if record.get("seeds") != [20261026, 20261027]:
            raise ArtifactError(f"sensitivity R={r} has unexpected seeds")
        if record.get("windows") != EXPECTED_WINDOWS:
            raise ArtifactError(f"sensitivity R={r} has unexpected train/development counts")
        if record.get("final_holdout_evaluated") is not False:
            raise ArtifactError(f"sensitivity R={r} does not preserve the sealed-final scope")
        audit = _mapping(record.get("audit"), f"sensitivity.R{r}.audit")
        if audit.get("final_quaternion_datasets_read") != 0:
            raise ArtifactError(f"sensitivity R={r} reports final-partition quaternion reads")
        results = _mapping(record.get("results"), f"sensitivity.R{r}.results")
        for split in ("train", "development"):
            partition = _mapping(results.get(split), f"sensitivity.R{r}.{split}")
            _validate_partition(partition, split)
            _validate_contrasts(partition, split)
            _validate_uncertainty(partition, split)
            if partition.get("final_holdout") != "untouched_not_evaluated_by_this_API":
                raise ArtifactError(f"sensitivity R={r} {split} lost the sealed-final scope")
            for variant in ("h1_a0", "h20_a0"):
                method = f"Selector_{variant}_two_seed_mean_error"
                item = _metric(partition, method, split)
                _check_ci(item.get("user_cluster_95_CI_mean_deg"), f"sensitivity.R{r}.{split}.{method}.mean_CI")
        methods = results["development"]["methods"]
        for variant in ("h1_a0", "h20_a0"):
            key = f"Selector_{variant}_two_seed_mean_error"
            _metric(results["development"], key, "development")
        if methods["static_mixture"].get("windows") != EXPECTED_WINDOWS["development"]:
            raise ArtifactError(f"sensitivity R={r} static-mixture coverage differs")


def _validate_cross_source_static(main: dict[str, Any], baselines: dict[str, Any]) -> None:
    for split in ("train", "development"):
        main_metric = _metric(main["results"][split], "static_mixture", split)
        baseline_metric = baselines["analytic"]["results"][split]["methods"]["static_mixture"]
        for field in ("mean_geodesic_deg", "p95_geodesic_deg"):
            if not math.isclose(main_metric[field], baseline_metric[field], rel_tol=0.0, abs_tol=1e-12):
                raise ArtifactError(f"R80 and baseline {split} static-mixture {field} disagree")


def _table_value(metric: dict[str, Any], field: str) -> str:
    return f"{_finite_number(metric[field], field):.4f}"


def _write_results_table(main: dict[str, Any], baselines: dict[str, Any], path: Path) -> None:
    analytic = baselines["analytic"]["results"]
    neural = baselines["neural"]["results"]
    row_specs = [
        ("Zero velocity", "baseline", "ZV"),
        ("CAV, $W=1$", "baseline", "CAV_W1"),
        ("CAV, $W=5$", "baseline", "CAV_W5"),
        ("CAV, $W=20$", "baseline", "CAV_W20"),
        ("Static mixture", "baseline", "static_mixture"),
        ("Relative MLP (80,572 params)", "mlp", "MLP_two_seed_mean_error"),
        ("Selector $h=1,a=0$", "selector", "h1_a0"),
        ("Selector $h=20,a=0$", "selector", "h20_a0"),
        ("Selector $h=1,a=19$", "selector", "h1_a19"),
        ("Selector $h=20,a=20$", "selector", "h20_a20"),
        ("Selector, complete horizon", "selector", "complete_horizon"),
        ("Selector, motion only", "selector", "motion_only"),
    ]
    train = main["results"]["train"]
    dev = main["results"]["development"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Endpoint-20 geodesic error (degrees). Selector and MLP rows use the arithmetic mean of two fixed-seed error vectors, not an ensemble. Train is in-sample ($n=15$ users); development has $n=5$ users. P95 is the pooled-window percentile.}",
        r"\label{tab:results}",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Train mean & Dev mean & Dev P95 \\",
        r"\midrule",
    ]
    for label, source, key in row_specs:
        values = []
        for split in ("train", "development"):
            if source == "baseline":
                metric = analytic[split]["methods"][key]
            elif source == "mlp":
                metric = neural[split]["methods"][key]
            else:
                metric = _metric(main["results"][split], f"Selector_{key}_two_seed_mean_error", split)
            values.append(metric)
        lines.append(
            f"{label} & {_table_value(values[0], 'mean_geodesic_deg')} & "
            f"{_table_value(values[1], 'mean_geodesic_deg')} & "
            f"{_table_value(values[1], 'p95_geodesic_deg')} \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sensitivity_table(main: dict[str, Any], sensitivity: dict[str, Any], path: Path) -> None:
    rows = []
    records = {"80": main, **sensitivity["results_by_R"]}
    for r_text in ("80", "128", "256"):
        record = records[r_text]
        if r_text == "80":
            split = record["results"]["development"]
        else:
            split = record["results"]["development"]
        methods = split["methods"]
        h1 = _metric(split, "Selector_h1_a0_two_seed_mean_error", "development")
        h20 = _metric(split, "Selector_h20_a0_two_seed_mean_error", "development")
        contrast = split["contrasts"][PRIMARY_CONTRAST]
        delta = _finite_number(contrast["mean_delta_deg"], f"R{r_text}.delta")
        low, high = _check_ci(contrast.get(DELTA_CI_KEY), f"R{r_text}.{DELTA_CI_KEY}")
        rows.append((r_text, h1, h20, delta, low, high))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{R sensitivity on development ($n=5$ users). The paired difference is h20 minus h1 in window-weighted mean endpoint-20 error; intervals are the paired user-cluster bootstrap 95\% CIs. Positive values indicate higher error for h20. These are exploratory descriptions, not formal hypothesis or equivalence tests.}",
        r"\label{tab:sensitivity}",
        r"\footnotesize",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Evidence length $R$ & $h=1,a=0$ & $h=20,a=0$ & $h20-h1$ difference [95\% CI] \\",
        r"\midrule",
    ]
    for r_text, h1, h20, delta, low, high in rows:
        lines.append(
            f"{r_text} & {_table_value(h1, 'mean_geodesic_deg')} & "
            f"{_table_value(h20, 'mean_geodesic_deg')} & "
            f"{delta:+.4f} [{low:+.4f}, {high:+.4f}] \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _save_figure(fig: plt.Figure, stem: str, directory: Path) -> None:
    fig.savefig(directory / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(directory / f"{stem}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_evidence_age(directory: Path) -> None:
    # Latest mature forecast pairings at one R80 origin; all positions are samples.
    rows = (
        (r"$h=1,\ a=0$", -1, 0),
        (r"$h=20,\ a=0$", -20, 0),
        (r"$h=1,\ a=19$", -20, -19),
        (r"$h=20,\ a=20$", -40, -20),
    )
    issue_color = "#365f91"
    target_color = "#b45f06"
    fig, ax = plt.subplots(figsize=(3.45, 2.12), constrained_layout=True)
    y_positions = list(range(len(rows) - 1, -1, -1))
    for (label, issue, target), y in zip(rows, y_positions):
        ax.plot([issue, target], [y, y], color="#9a9a9a", lw=1.15, zorder=1)
        if target - issue > 1:
            ax.add_patch(
                FancyArrowPatch(
                    (issue + 0.4, y),
                    (target - 0.4, y),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.8,
                    color="#555555",
                    zorder=2,
                )
            )
        else:
            ax.add_patch(
                FancyArrowPatch(
                    (issue + 0.08, y),
                    (target - 0.08, y),
                    arrowstyle="-|>",
                    mutation_scale=5,
                    linewidth=0.8,
                    color="#555555",
                    zorder=2,
                )
            )
        ax.scatter(issue, y, marker="s", s=22, color=issue_color, edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(target, y, marker="o", s=25, color=target_color, edgecolor="white", linewidth=0.35, zorder=4)

    ax.axvline(0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
    ax.text(0.25, 3.35, "current origin $t=0$", ha="left", va="bottom", fontsize=6.6)
    ax.set_xlim(-43.5, 5.4)
    ax.set_ylim(-0.52, 3.85)
    ax.set_yticks(y_positions, [row[0] for row in rows])
    ax.set_xticks([-40, -20, 0])
    ax.set_xlabel("Sample index relative to current origin (samples)")
    ax.set_ylabel(None)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5, linestyle=":")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", pad=2)
    ax.set_axisbelow(True)
    _save_figure(fig, "evidence_age", directory)


def _sensitivity_rows(main: dict[str, Any], sensitivity: dict[str, Any]) -> list[tuple[str, float, float, float, float]]:
    records = {"80": main, **sensitivity["results_by_R"]}
    rows = []
    for r_text in ("80", "128", "256"):
        partition = records[r_text]["results"]["development"]
        contrast = partition["contrasts"][PRIMARY_CONTRAST]
        mean = _finite_number(contrast["mean_delta_deg"], f"R{r_text}.mean_delta_deg")
        low, high = _check_ci(contrast.get(DELTA_CI_KEY), f"R{r_text}.{DELTA_CI_KEY}")
        rows.append((r_text, mean, low, high, _finite_number(
            partition["methods"]["Selector_h20_a0_two_seed_mean_error"]["mean_geodesic_deg"],
            f"R{r_text}.h20.mean_geodesic_deg",
        )))
    return rows


def _plot_sensitivity(main: dict[str, Any], sensitivity: dict[str, Any], directory: Path) -> None:
    rows = _sensitivity_rows(main, sensitivity)
    fig, ax = plt.subplots(figsize=(3.45, 1.85), constrained_layout=True)
    y_positions = [2, 1, 0]
    colors = ["#365f91", "#4f7d5c", "#b45f06"]
    for (r_text, mean, low, high, _), y, color in zip(rows, y_positions, colors):
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=4.2,
            linewidth=1.2,
            elinewidth=1.15,
            zorder=3,
        )
    ax.axvline(0, color="#333333", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    ax.set_yticks(y_positions, [f"R={row[0]}" for row in rows])
    ax.set_xlabel(r"$h20-h1$ paired mean endpoint error difference ($^\circ$)")
    ax.set_xlim(-0.0008, max(row[3] for row in rows) * 1.24)
    ax.set_ylim(-0.55, 2.55)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.5, linestyle=":")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_axisbelow(True)
    _save_figure(fig, "R_sensitivity", directory)


def _write_figure_tex(path: Path) -> None:
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/evidence_age.pdf}
\caption{Latest mature pairings at the current origin, with $R=80$ past origins averaged. Horizon $h$ is the issue-to-target gap in samples and $a$ is target age. Squares show issue samples; circles show targets. The $h=20,a=0$ and $h=1,a=19$ pairings share relative issue index $-20$, that is, issue age $20$.}
\label{fig:evidence_age}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/R_sensitivity.pdf}
\caption{Development paired differences (five users) in endpoint-20 mean error, $h20-h1$, with user-cluster bootstrap 95\% CIs at $R=80,128,256$. Positive values indicate higher error for $h20$. The intervals describe this exploratory development sample and do not constitute formal hypothesis or equivalence tests.}
\label{fig:R_sensitivity}
\end{figure}
"""
    path.write_text(text, encoding="utf-8")


def _validate_inference(inference: dict[str, Any]) -> list[dict[str, Any]]:
    if inference.get("stage") != "numerical_portability_and_local_cost_not_accuracy_selection":
        raise ArtifactError("--inference has an unknown stage")
    if inference.get("status") != "PASS" or inference.get("all_fixed_numerical_gates_passed") is not True:
        raise ArtifactError("--inference did not pass all fixed numerical portability gates")
    if inference.get("final_holdout_evaluated") is not False:
        raise ArtifactError("--inference does not certify that final remained unevaluated")
    runtime = _mapping(inference.get("runtime"), "inference.runtime")
    if runtime.get("ORT_measured_provider") != "CPUExecutionProvider":
        raise ArtifactError("--inference must identify measured CPU ONNX Runtime execution")
    sampling = _mapping(inference.get("sampling"), "inference.sampling")
    expected_contexts = {"train": 390, "development": 130}
    for split in ("train", "development"):
        partition = _mapping(sampling.get(split), f"inference.sampling.{split}")
        expected = {
            "users": EXPECTED_USERS[split],
            "contexts": expected_contexts[split],
            "contexts_per_user": 26,
            "population_windows": EXPECTED_WINDOWS[split],
        }
        if any(partition.get(key) != value for key, value in expected.items()):
            raise ArtifactError(f"--inference {split} sample counts differ from the fixed train/development audit")
    results = _mapping(inference.get("results"), "inference.results")
    if set(results) != set(EXPECTED_VARIANTS):
        raise ArtifactError("--inference does not contain all six fixed selector variants")
    records = []
    for variant in EXPECTED_VARIANTS:
        record = _mapping(results[variant], f"inference.{variant}")
        if record.get("all_fixed_numerical_gates_passed") is not True:
            raise ArtifactError(f"inference.{variant} did not pass all fixed numerical portability gates")
        if record.get("trainable_parameters") != 244:
            raise ArtifactError(f"inference.{variant} parameter count is not 244")
        inventory = _mapping(record.get("inventory"), f"inference.{variant}.inventory")
        node_count = inventory.get("node_count")
        if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count <= 0:
            raise ArtifactError(f"inference.{variant} inventory node_count must be positive")
        if inventory.get("selector_dense_macs") != 224:
            raise ArtifactError(f"inference.{variant} selector MAC count is not 224")
        graph_bytes = record.get("graph_bytes")
        if isinstance(graph_bytes, bool) or not isinstance(graph_bytes, int) or graph_bytes <= 0:
            raise ArtifactError(f"inference.{variant} graph byte count must be positive")
        if inventory.get("file_bytes") != graph_bytes:
            raise ArtifactError(f"inference.{variant} graph byte counts disagree")
        ort = _mapping(record.get("latency"), f"inference.{variant}.latency").get("ORT_CPU")
        ort = _mapping(ort, f"inference.{variant}.latency.ORT_CPU")
        median = _finite_number(ort.get("median_ms"), f"inference.{variant}.ORT_CPU.median_ms")
        p95 = _finite_number(ort.get("p95_ms"), f"inference.{variant}.ORT_CPU.p95_ms")
        if median <= 0 or p95 < median:
            raise ArtifactError(f"inference.{variant} CPU ORT latency statistics are inconsistent")
        scalar = _mapping(inventory.get("elementwise_scalar_evaluations"), f"inference.{variant}.scalar_ops")
        for op in ("Add", "Sub", "Mul", "Div", "Sqrt", "Sin", "Cos", "Atan", "Abs", "Neg", "Relu"):
            if op not in scalar:
                raise ArtifactError(f"inference.{variant} is missing scalar count for {op}")
            count = scalar[op]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ArtifactError(f"inference.{variant} scalar count {op} must be a nonnegative integer")
        for field in ("reduction_scalar_additions", "reduction_scalar_divisions", "reduction_max_comparisons", "gemm_bias_additions"):
            count = inventory.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ArtifactError(f"inference.{variant}.{field} must be a nonnegative integer")
        parity = _mapping(record.get("parity"), f"inference.{variant}.parity")
        max_real_drift = 0.0
        for split in ("train", "development"):
            providers = _mapping(parity.get(split), f"inference.{variant}.parity.{split}")
            for provider in ("eager_cpu", "ORT_CPU", "eager_CUDA"):
                stats = _mapping(providers.get(provider), f"inference.{variant}.{split}.{provider}")
                expected_context_count = expected_contexts[split]
                if stats.get("contexts") != expected_context_count or stats.get("output_quaternions") != expected_context_count * 20:
                    raise ArtifactError(f"inference.{variant}.{split}.{provider} has unexpected sampled output counts")
                if stats.get("all_outputs_finite") is not True:
                    raise ArtifactError(f"inference.{variant}.{split}.{provider} has nonfinite outputs")
                checks = _mapping(stats.get("numerical_gate_checks"), f"inference.{variant}.{split}.{provider}.gates")
                if not checks or any(value is not True for value in checks.values()):
                    raise ArtifactError(f"inference.{variant}.{split}.{provider} failed a numerical portability gate")
                drift = _finite_number(
                    stats.get("max_trajectory_geodesic_drift_deg"),
                    f"inference.{variant}.{split}.{provider}.max_drift",
                )
                if drift < 0:
                    raise ArtifactError(f"inference.{variant}.{split}.{provider} has negative trajectory drift")
                max_real_drift = max(max_real_drift, drift)
        records.append(
            {
                "variant": variant,
                "result": record,
                "inventory": inventory,
                "median": median,
                "p95": p95,
                "max_real_drift": max_real_drift,
            }
        )
    return records


def _write_inference_tables(inference: dict[str, Any], tables_dir: Path) -> None:
    records = _validate_inference(inference)
    individual = [
        next(item for item in records if item["variant"] == variant)
        for variant in ("h1_a0", "h20_a0", "h1_a19", "h20_a20")
    ]
    cost_fields = (
        "node_count",
        "file_bytes",
        "selector_dense_macs",
        "elementwise_scalar_evaluations",
        "reduction_scalar_additions",
        "reduction_scalar_divisions",
        "reduction_max_comparisons",
        "gemm_bias_additions",
    )
    reference = {field: individual[0]["inventory"][field] for field in cost_fields}
    if any({field: item["inventory"][field] for field in cost_fields} != reference for item in individual[1:]):
        raise ArtifactError("individual-horizon selector variants do not have identical static inventory counts")

    max_sampled_drift = max(item["max_real_drift"] for item in records)
    gate_note = (
        " All fixed portability gates passed; maximum sampled train/development trajectory drift across eager CPU, "
        "CPU ORT, and eager CUDA was "
        f"{max_sampled_drift:.6f} degrees. Eager CUDA is a separate tensor-parity backend, not CUDA ORT or XR latency."
    )
    latency_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        "\\caption{Local batch-1 ONNX graph size and CPU ONNX Runtime forward latency. Each row is the same 244-parameter selector (224 dense selector MACs per call); these counts exclude evidence processing and the full-graph operator work summarized separately. This laptop measurement is not target-XR latency."
        + gate_note
        + "}",
        r"\label{tab:resources}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Variant & Parameters & Selector MACs & Nodes & Graph bytes & CPU ORT median (ms) & CPU ORT P95 (ms) \\",
        r"\midrule",
    ]
    inventory_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full static ONNX scalar-cost inventory by variant. Elementwise counts sum inferred output elements; reduction counts use input minus output elements; Gemm bias additions are reported separately. These are graph arithmetic counts, not hardware instruction counts or latency.}",
        r"\label{tab:operator_inventory}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Variant & Mul & Add & Sub & Div & Sqrt & Sin & Cos & Atan & Abs & Neg & ReLU & Red. add & Red. div & Max cmp & Gemm bias \\",
        r"\midrule",
    ]
    compact_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Compact static R80 arithmetic inventory. The individual-horizon row represents the h20 graph; h1 has identical counts. Each row also has 224 dense selector MACs and 20 Gemm bias additions, reported separately and excluded from the grouped arithmetic columns. These groups are graph counts, not a whole-graph MAC estimate or hardware instruction count.}",
        r"\label{tab:operator_inventory_compact}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Graph & Mul & Add/Sub + reduction adds & Div + mean div & Sqrt & Sin + Cos + Atan \\",
        r"\midrule",
    ]
    labels = {
        "h1_a0": "$h=1,a=0$",
        "h20_a0": "$h=20,a=0$",
        "h1_a19": "$h=1,a=19$",
        "h20_a20": "$h=20,a=20$",
        "complete_horizon": "Complete horizon",
        "motion_only": "Motion only",
    }
    for item in records:
        result, inventory = item["result"], item["inventory"]
        latency_lines.append(
            f"{labels[item['variant']]} & 244 & 224 & {inventory['node_count']} & {result['graph_bytes']} & "
            f"{item['median']:.4f} & {item['p95']:.4f} \\\\")
        scalar = inventory["elementwise_scalar_evaluations"]
        counts = [scalar[name] for name in ("Mul", "Add", "Sub", "Div", "Sqrt", "Sin", "Cos", "Atan", "Abs", "Neg", "Relu")]
        counts.extend(
            inventory[name]
            for name in (
                "reduction_scalar_additions",
                "reduction_scalar_divisions",
                "reduction_max_comparisons",
                "gemm_bias_additions",
            )
        )
        inventory_lines.append(f"{labels[item['variant']]} & " + " & ".join(str(value) for value in counts) + r" \\")
    compact_specs = (
        ("Individual horizon ($h=20$; same for $h=1$)", "h20_a0"),
        ("Complete horizon", "complete_horizon"),
        ("Motion only", "motion_only"),
    )
    by_variant = {item["variant"]: item for item in records}
    for label, variant in compact_specs:
        inventory = by_variant[variant]["inventory"]
        scalar = inventory["elementwise_scalar_evaluations"]
        grouped = (
            scalar["Mul"],
            scalar["Add"] + scalar["Sub"] + inventory["reduction_scalar_additions"],
            scalar["Div"] + inventory["reduction_scalar_divisions"],
            scalar["Sqrt"],
            scalar["Sin"] + scalar["Cos"] + scalar["Atan"],
        )
        compact_lines.append(label + " & " + " & ".join(str(value) for value in grouped) + r" \\")
    latency_lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    inventory_lines.extend((r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"))
    compact_lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    (tables_dir / "resources.tex").write_text("\n".join(latency_lines) + "\n", encoding="utf-8")
    (tables_dir / "operator_inventory_full_supplement.tex").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )
    (tables_dir / "operator_inventory_compact.tex").write_text(
        "\n".join(compact_lines) + "\n", encoding="utf-8"
    )
    stale_paper_table = tables_dir / "operator_inventory.tex"
    if stale_paper_table.exists():
        stale_paper_table.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_manifest(args: argparse.Namespace, output: Path, input_paths: dict[str, Path], script_path: Path) -> None:
    relative_outputs = [
            "figures/evidence_age.pdf",
            "figures/evidence_age.png",
            "figures/R_sensitivity.pdf",
            "figures/R_sensitivity.png",
            "generated/figures.tex",
            "generated/tables/results.tex",
            "generated/tables/sensitivity.tex",
        ]
    if args.inference is not None:
        relative_outputs.extend(
            (
                "generated/tables/resources.tex",
                "generated/tables/operator_inventory_compact.tex",
                "generated/tables/operator_inventory_full_supplement.tex",
            )
        )
    output_files = sorted(path for relative in relative_outputs if (path := output / relative).is_file())
    manifest = {
        "schema_version": 1,
        "command_argv": sys.argv,
        "input_aggregates": {
            name: {"file_name": path.name, "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "generator_sha256": _sha256(script_path),
        "outputs": {
            path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in output_files
        },
        "scope": "public aggregate-only plots and tables; final holdout untouched",
    }
    (output / "figures" / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_artifacts(
    main_path: Path,
    sensitivity_path: Path,
    baselines_path: Path,
    output: Path,
    inference_path: Path | None = None,
) -> dict[str, Any]:
    main = _read_json(main_path, "main")
    sensitivity = _read_json(sensitivity_path, "sensitivity")
    baselines = _read_json(baselines_path, "baselines")
    _validate_main(main)
    _validate_sensitivity(sensitivity)
    _validate_baselines(baselines)
    _validate_cross_source_static(main, baselines)

    figures_dir = output / "figures"
    tables_dir = output / "generated" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _configure_plotting()
    _plot_evidence_age(figures_dir)
    _plot_sensitivity(main, sensitivity, figures_dir)
    _write_results_table(main, baselines, tables_dir / "results.tex")
    _write_sensitivity_table(main, sensitivity, tables_dir / "sensitivity.tex")
    _write_figure_tex(output / "generated" / "figures.tex")
    if inference_path is not None:
        inference = _read_json(inference_path, "inference")
        _write_inference_tables(inference, tables_dir)
    table_files = [tables_dir / "results.tex", tables_dir / "sensitivity.tex"]
    if inference_path is not None:
        table_files.extend(
            (
                tables_dir / "resources.tex",
                tables_dir / "operator_inventory_compact.tex",
                tables_dir / "operator_inventory_full_supplement.tex",
            )
        )
    return {
        "output": output,
        "figure_files": [
            figures_dir / "evidence_age.pdf",
            figures_dir / "evidence_age.png",
            figures_dir / "R_sensitivity.pdf",
            figures_dir / "R_sensitivity.png",
        ],
        "table_files": table_files,
        "figure_tex": output / "generated" / "figures.tex",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", required=True, type=Path, help="R80 aggregate summary JSON")
    parser.add_argument("--sensitivity", required=True, type=Path, help="R128/R256 aggregate summary JSON")
    parser.add_argument("--baselines", required=True, type=Path, help="public baseline aggregate JSON")
    parser.add_argument("--inference", type=Path, help="optional public inference aggregate JSON")
    parser.add_argument("--output", type=Path, help="artifact root; defaults to a fresh temporary directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="abc-paper-artifacts-"))
    try:
        built = build_artifacts(args.main, args.sensitivity, args.baselines, output, args.inference)
        if args.inference is None:
            warnings.warn(
                "No inference aggregate was supplied; no measured resource or latency table was generated.",
                stacklevel=1,
            )
        input_paths = {"main": args.main, "sensitivity": args.sensitivity, "baselines": args.baselines}
        if args.inference is not None:
            input_paths["inference"] = args.inference
        script_path = Path(__file__).resolve()
        _build_manifest(args, output, input_paths, script_path)
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote paper artifacts under {built['output']}")
    for key in ("figure_files", "table_files"):
        for path in built[key]:
            print(f"  {path.relative_to(output)}  sha256={_sha256(path)}")
    print(f"  {built['figure_tex'].relative_to(output)}  sha256={_sha256(built['figure_tex'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

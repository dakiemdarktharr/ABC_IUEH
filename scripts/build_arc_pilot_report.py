"""Build aggregate-only pilot tables; never read human vectors or checkpoints.

Run after the source-frozen GPU process has completed. This is a reporting
consumer, not an experiment runner or an automatic publication decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


METHODS = ("static_mixture", "CAV_W1", "CAV_W5", "CAV_W20", "ZV",
           "nll21_median", "nll21_mean", "nll21_map", "scalar_angle",
           "scalar_pinball", "risk21", "vector_angle", "prior_median", "prior_mean", "prior_map")
LABELS = {"static_mixture": "Static mixture", "nll21_median": "NLL21 median (primary)",
          "nll21_mean": "NLL21 mean", "nll21_map": "NLL21 MAP", "scalar_angle": "Direct scalar-angle",
          "scalar_pinball": "Scalar-pinball", "risk21": "Angular-risk21", "vector_angle": "Direct vector-angle"}
TOP_FIELDS = {"status", "matched_updates_primary", "matched_training_time_secondary", "runtime", "bindings",
              "development_signal_values_read", "final_array_values_read", "data_audit", "read_set_counters",
              "sources_unchanged", "config_unchanged"}
PRIVATE_KEYS = {"user_ids", "user_sums", "user_counts", "user_folds", "fit_users", "validation_users",
                "calibration_indexes", "prior_probabilities", "static_weights", "state_dict", "optimizer_states",
                "sampling_schedule", "process_id_private"}
MEASURED_COMMIT = "13c74071156b5b8dfc893f42107ad2cb82ea6921"


def check_public(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("private participant/model content in aggregate")
        for child in value.values():
            check_public(child)
    elif isinstance(value, list):
        for child in value:
            check_public(child)
    elif isinstance(value, str):
        if re.search(r"[A-Za-z]:[\\/]", value) or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
            raise ValueError("personal path or email in aggregate")


def build(source: Path, root: Path):
    raw = source.read_bytes()
    result = json.loads(raw)
    if result.get("status") != "complete_exploratory_outer_train_inner_cv_only":
        raise ValueError("only a completed actual production pilot can be reported")
    if set(result) != TOP_FIELDS:
        raise ValueError("unexpected pilot summary fields; review before exporting")
    if (result["development_signal_values_read"] != 0 or result["final_array_values_read"] != 0
            or not result["sources_unchanged"] or not result["config_unchanged"]
            or not result["data_audit"]["data_and_split_unchanged"]):
        raise ValueError("read-set or source-integrity gate failed")
    primary = result["matched_updates_primary"]["metrics"]
    timed = result["matched_training_time_secondary"]["metrics"]
    for metrics in (primary, timed):
        if set(metrics["methods"]) != set(METHODS):
            raise ValueError("method coverage differs from fixed pilot")
        if metrics["validation"]["users"] != 15 or metrics["validation"]["windows"] != 3330312:
            raise ValueError("unexpected inner-user coverage")
    ledger = [json.loads(p.read_text(encoding="utf-8")) for p in source.parent.glob("session_*.json")]
    session_wall = sum(r["cuda_context_wall_seconds"] for r in ledger)
    allocation = max(r["peak_allocated_bytes"] for arm in primary["resources"]["arms"].values() for r in arm["records"])
    reserved = max(r["peak_reserved_bytes"] for arm in primary["resources"]["arms"].values() for r in arm["records"])
    trims = source.parent / "memory_trim_events.jsonl"
    trim_count = len(trims.read_text(encoding="utf-8").splitlines()) if trims.exists() else 0
    public = {**result, "reporting": {
        "source_summary_sha256": hashlib.sha256(raw).hexdigest(), "measured_source_commit": MEASURED_COMMIT,
        "measured_worktree_line_ending_note": "runner worktree CRLF; committed LF bytes bound separately in private commit receipt",
        "session_count_including_resource_pauses": len(ledger), "GPU_context_wall_seconds_including_preparation_IO_and_validation": session_wall,
        "sum_synchronized_training_seconds_all30fits": sum(v["total_training_seconds"] for v in primary["resources"]["arms"].values()),
        "shared_worker_peak_allocated_bytes": allocation, "shared_worker_peak_reserved_bytes": reserved,
        "peak_scope": "shared labels/data/five resident models and validation; not isolated per-model VRAM",
        "own_process_working_set_trim_event_records": trim_count,
        "operational_timing_limit": "recorded runtime includes resumes/paging; time matching is backend-qualified synchronized wall time",
        "cloud_cost_authorized_or_used": 0,
    }}
    check_public(public)
    destination = root / "docs/results/arc_pilot_20261028.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Angular-exposure GPU pilot — actual results, 2026-10-28", "",
             "User-disjoint inner validation inside the existing 15 outer-training users only: 10 fit / 5 validation per fold, 3 folds, 2 seeds, 30 fits.",
             "The previously used 5 development and 15 final users were not evaluated. These numbers are not directly comparable to the old five-user development means.", "",
             "Primary: 5,000 Adam updates, batch size 128, same fold-fit sampling schedule and cloned encoder initialization per fold/seed.",
             "Secondary: timing-only selection of saved 100-step prefixes, no additional fits; training examples differ in that comparison.", "",
             "| Method | Fixed 5,000 mean (degrees) | Approximate time-matched mean (degrees) | Fixed 5,000 P95 (degrees) |",
             "|---|---:|---:|---:|"]
    for method in METHODS:
        left, right = primary["methods"][method], timed["methods"][method]
        lines.append(f"| {LABELS.get(method, method)} | {left['mean_geodesic_deg']:.6f} | {right['mean_geodesic_deg']:.6f} | {left['p95_geodesic_deg']:.6f} |")
    lines += ["", "## Predeclared NLL-median contrasts", "",
              "Negative delta means NLL-median has lower error. CIs are paired whole-user bootstrap intervals conditional on two seeds and fixed CV fits.", "",
             "| Comparison | Fixed 5,000 delta (degrees) [95% CI] | Users improved | Time-matched delta (degrees) [95% CI] |",
              "|---|---:|---:|---:|"]
    for method in ("static_mixture", "scalar_angle", "risk21", "scalar_pinball", "vector_angle"):
        key = "nll21_median_minus_" + method
        left, right = primary["contrasts"][key], timed["contrasts"][key]
        lc, rc = left["conditional_user_cluster_95_CI_delta_deg"], right["conditional_user_cluster_95_CI_delta_deg"]
        lines.append(f"| NLL median − {LABELS.get(method,method)} | {left['mean_delta_deg']:+.6f} [{lc[0]:+.6f},{lc[1]:+.6f}] | {left['users_improved']}/15 | {right['mean_delta_deg']:+.6f} [{rc[0]:+.6f},{rc[1]:+.6f}] |")
    lines += ["", f"Fixed-update conjunctive gate: **{'PASS' if primary['gate']['passed'] else 'FAIL'}**. Time-matched gate: **{'PASS' if timed['gate']['passed'] else 'FAIL'}**.",
              "The gate requires at least 3% and 0.05° improvement against each of static/scalar-angle/risk21/pinball, both seed directions and at least 2/3 folds. It is not a novelty or publication-acceptance certificate.", "",
              "## Measured resources", "",
              f"Device: {result['runtime']['device_name']}; Torch {result['runtime']['torch_version']}.",
              f"All 30 fits synchronized training: {public['reporting']['sum_synchronized_training_seconds_all30fits']:.2f} s. GPU-context session wall including preparation/validation/I/O/resource pauses: {session_wall:.2f} s ({session_wall/3600:.3f} h).",
              f"Shared worker peak CUDA allocated/reserved: {allocation/2**20:.2f}/{reserved/2**20:.2f} MiB. This includes caches and all resident arms.",
              f"Maximum/minimum fixed-update training-time ratio: {primary['resources']['maximum_to_minimum_total_training_time_ratio']:.3f}; per-cell maximum {primary['resources']['maximum_per_fold_seed_training_time_ratio']:.3f}.",
              f"Approximate time-matched ratios: {timed['resources']['maximum_to_minimum_total_training_time_ratio']:.6f} overall and {timed['resources']['maximum_per_fold_seed_training_time_ratio']:.6f} in the worst cell; checkpoints have 100-step granularity.",
              "Equal updates are not equal time; the separate timed-prefix analysis was fixed before any efficacy result. Static calibration is a separate cheap baseline, not a neural-compute-matched fit.", "",
              "## Interpretation limits", "",
              "A closest-label median is optimal for snapped short-arc labels, not arbitrary off-axis rotations. Direct scalar and actual angular-risk controls are mandatory for this reason.",
              "Single conditional Alyx corpus; inner folds are exploratory selection-fold evidence. No foundation breadth, final confirmation, target-XR latency or high-novelty architecture claim is established.",
              "BOXRR remains disabled; Alyx-only challenge eligibility and the 20 samples / 100 Hz / 100 ms specification dispute remain unresolved.",
              "Per-seed/fold estimates, full user-cluster uncertainty, compute selections and source bindings are in the aggregate JSON. Human vectors, participant maps and weights remain private.", "",
              f"Measured code commit: `{MEASURED_COMMIT}`; summary SHA-256 `{public['reporting']['source_summary_sha256']}`.",
              "See [fixed plan](arc_pilot_plan.md), [code review](arc_pilot_code_review.md), [independent result audit](arc_pilot_results_audit.md), and [aggregate](results/arc_pilot_20261028.json).", ""]
    report = root / "docs/arc_pilot_results_2026-10-28.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return destination, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/arc_pilot/20261028/summary.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    for path in build(args.source, args.root):
        print(path)


if __name__ == "__main__":
    main()

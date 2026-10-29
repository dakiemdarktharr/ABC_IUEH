#!/usr/bin/env python3
"""Offline all-in cost preflight for a manually priced Vast training lease.

This tool never contacts a provider and never authorizes a run. Its output is
budget arithmetic from operator-supplied prices and quantities only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PlanError(ValueError):
    pass


def _number(obj: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"{key} must be a numeric value copied from a current price/usage estimate")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise PlanError(f"{key} must be finite and at least {minimum}")
    return value


def evaluate(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise PlanError("plan must be a JSON object")
    cap = _number(plan, "budget_cap_usd", minimum=0.01)
    reserve = _number(plan, "reserve_usd", minimum=0.01)
    prior = _number(plan, "charges_already_incurred_usd")
    prior_final = _number(plan, "prior_final_storage_export_charges_usd")
    final_estimate = _number(plan, "final_storage_export_estimate_usd")
    if cap != 20.0 or reserve != 4.0:
        raise PlanError("this user-directed campaign is fixed at a $20 total cap with a $4 final-charge reserve")
    if prior_final > prior:
        raise PlanError("prior final storage/export charges cannot exceed all prior charges")
    gpu_count_value = plan.get("gpu_count")
    if isinstance(gpu_count_value, bool) or not isinstance(gpu_count_value, int) or gpu_count_value != 1:
        raise PlanError("gpu_count must be exactly 1 for this package")
    hours = _number(plan, "instance_hours_max", minimum=0.01)
    compute_rate = _number(plan, "compute_usd_per_gpu_hour", minimum=0.000001)
    storage = plan.get("storage")
    egress = plan.get("egress")
    if not isinstance(storage, dict) or not isinstance(egress, dict):
        raise PlanError("storage and egress must be explicit JSON objects")
    if storage.get("includes_stopped_instance_storage") is not True:
        raise PlanError("confirm that storage pricing includes attached/stopped-instance storage")
    if egress.get("includes_checkpoint_copy_out") is not True:
        raise PlanError("confirm that the egress estimate includes checkpoint copy-out")

    storage_gib = _number(storage, "gib")
    storage_hours = _number(storage, "billed_hours")
    storage_rate = _number(storage, "usd_per_gib_hour")
    egress_gib = _number(egress, "gib_max")
    egress_rate = _number(egress, "usd_per_gib")
    other = _number(plan, "fees_tax_and_other_usd")
    retries = _number(plan, "retry_failure_allowance_usd")

    price_source = plan.get("price_source")
    if not isinstance(price_source, dict):
        raise PlanError("price_source must identify when and where prices were checked")
    checked_at = price_source.get("checked_at_utc")
    source_label = price_source.get("source")
    if not isinstance(checked_at, str) or not checked_at.strip():
        raise PlanError("price_source.checked_at_utc is required")
    if not isinstance(source_label, str) or not source_label.strip():
        raise PlanError("price_source.source is required (for example, provider listing and storage page)")

    compute_cost = hours * compute_rate
    storage_cost = storage_gib * storage_hours * storage_rate
    egress_cost = egress_gib * egress_rate
    core_incremental = compute_cost + storage_cost + egress_cost + other + retries
    prior_core = prior - prior_final
    core_total = prior_core + core_incremental
    final_total = prior_final + final_estimate
    estimate = core_incremental + final_estimate
    all_in = prior + estimate
    soft_threshold = cap - reserve
    fixed_core = storage_cost + egress_cost + other + retries
    max_instance_hours = max(0.0, (soft_threshold - prior_core - fixed_core) / compute_rate)
    fits = core_total < soft_threshold and final_total <= reserve and all_in <= cap

    return {
        "schema": "vast-budget-preflight-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": str(plan.get("provider", "unspecified")),
        "plan_id": str(plan.get("plan_id", "unspecified")),
        "price_source": {"source": source_label, "checked_at_utc": checked_at},
        "budget_cap_usd": round(cap, 6),
        "reserve_usd": round(reserve, 6),
        "soft_pause_threshold_usd": round(soft_threshold, 6),
        "charges_already_incurred_usd": round(prior, 6),
        "prior_final_storage_export_charges_usd": round(prior_final, 6),
        "modeled_costs_usd": {
            "compute": round(compute_cost, 6),
            "stopped_or_attached_storage": round(storage_cost, 6),
            "egress": round(egress_cost, 6),
            "fees_tax_and_other": round(other, 6),
            "retry_failure_allowance": round(retries, 6),
            "planned_core_operating_costs": round(core_incremental, 6),
            "planned_final_storage_export_reserve_use": round(final_estimate, 6),
            "incremental_all_in": round(estimate, 6),
            "core_total_including_prior": round(core_total, 6),
            "final_storage_export_total_including_prior": round(final_total, 6),
            "including_prior_charges": round(all_in, 6),
        },
        "planned_instance_hours_max": hours,
        "compute_rate_usd_per_gpu_hour": compute_rate,
        "max_instance_hours_under_model_after_reserve": round(max_instance_hours, 6),
        "budget_math_fits_with_reserve": fits,
        "reserve_discretionary_training_allowed": False,
        "provider_spend_enforcement": "NOT_CHECKED; this offline tool cannot enforce or verify a provider/account cap",
        "run_authorized": False,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="operator-completed JSON price/usage plan")
    parser.add_argument("--write-new", type=Path, help="write report to a new file; refuses to overwrite")
    args = parser.parse_args(argv)
    try:
        raw = args.plan.read_bytes()
        plan = json.loads(raw)
        report = evaluate(plan)
        report["input_file_sha256"] = hashlib.sha256(raw).hexdigest()
        output = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.write_new:
            args.write_new.parent.mkdir(parents=True, exist_ok=True)
            with args.write_new.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
        print(output, end="")
        return 0 if report["budget_math_fits_with_reserve"] else 3
    except (OSError, json.JSONDecodeError, PlanError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

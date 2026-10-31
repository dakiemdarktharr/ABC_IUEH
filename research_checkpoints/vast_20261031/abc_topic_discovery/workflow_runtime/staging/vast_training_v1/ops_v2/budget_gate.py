#!/usr/bin/env python3
"""Fail-closed per-update USD forecast gate for a reviewed Vast lease.

No provider API or credentials are used. The trusted root controller supplies
fresh operator-verified offer data and an authenticated campaign ledger. This
is a conservative soft-stop forecast, not a provider-enforced spending cap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
BUDGET_POLICY = {
    "soft_training_stop_usd": 4.0,
    "final_charge_reserve_usd": 12.0,
    "campaign_total_usd": 16.0,
}
REQUIRED_PRICE_CATEGORIES = (
    "on_demand_compute",
    "minimum_billing",
    "stopped_storage",
    "egress_and_checkpoint_export",
    "fees_and_taxes",
)


class BudgetStop(RuntimeError):
    """Raised before an optimizer update when the current forecast fails closed."""


def _finite(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is missing or not numeric")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


@dataclass(frozen=True)
class CostDecision:
    allowed: bool
    reasons: tuple[str, ...]
    projected_core_usd: float
    projected_final_usd: float
    projected_total_usd: float
    projected_billed_compute_hours: float
    quote_age_seconds: float
    ledger_age_seconds: float
    hard_cap_enforced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "vast-before-update-cost-decision-v1",
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "projected_core_usd": round(self.projected_core_usd, 6),
            "projected_final_reserve_use_usd": round(self.projected_final_usd, 6),
            "projected_total_usd": round(self.projected_total_usd, 6),
            "projected_billed_compute_hours": round(self.projected_billed_compute_hours, 6),
            "quote_age_seconds": round(self.quote_age_seconds, 3),
            "ledger_age_seconds": round(self.ledger_age_seconds, 3),
            "hard_cap_enforced": False,
            "enforcement": "local forecast only; provider billing and delayed charges are not enforced",
        }


def evaluate_before_update(
    quote: dict[str, Any],
    ledger: dict[str, Any],
    *,
    remaining_updates: int,
    worst_case_seconds_per_update: float,
    now: datetime | None = None,
    max_quote_age_seconds: int = 3600,
    max_ledger_age_seconds: int = 60,
) -> CostDecision:
    """Project campaign totals before allowing the next update.

    Contract: `ledger.committed_core_usd` includes all campaign core charges to
    date except compute on the current live lease. `ledger.committed_final_usd`
    includes final storage/export charges already posted. Quote fields ending
    in `_remaining_usd` cover future commitments not already included by the
    ledger. The root controller must construct these from its frozen plan,
    monotonic lease clock, and reviewed price snapshot.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    reasons: list[str] = []
    if quote.get("schema") != "vast-verified-offer-v1":
        reasons.append("missing reviewed price-offer schema")
    if quote.get("provider") != "Vast.ai" or not str(quote.get("offer_id", "")).strip():
        reasons.append("provider offer identity is missing")
    if quote.get("currency") != "USD":
        reasons.append("quote currency is not USD")
    if quote.get("verified_by_root") is not True:
        reasons.append("root has not verified and bound the offer snapshot")
    source_sha = quote.get("source_sha256")
    if not isinstance(source_sha, str) or not SHA_RE.fullmatch(source_sha):
        reasons.append("offer source SHA-256 is missing")
    categories = quote.get("verified_price_categories")
    if not isinstance(categories, dict) or any(categories.get(name) is not True for name in REQUIRED_PRICE_CATEGORIES):
        reasons.append("one or more required billing categories are unverified")
    quote_at = _timestamp(quote.get("checked_at_utc"), "quote.checked_at_utc")
    quote_age = (current - quote_at).total_seconds()
    if quote_age < 0 or quote_age > max_quote_age_seconds:
        reasons.append("price offer is future-dated or stale for this lease")

    if ledger.get("schema") != "vast-campaign-ledger-v1" or ledger.get("root_authenticated") is not True:
        reasons.append("missing root-authenticated campaign ledger")
    ledger_at = _timestamp(ledger.get("as_of_utc"), "ledger.as_of_utc")
    ledger_age = (current - ledger_at).total_seconds()
    if ledger_age < 0 or ledger_age > max_ledger_age_seconds:
        reasons.append("campaign ledger is future-dated or stale")

    if isinstance(remaining_updates, bool) or not isinstance(remaining_updates, int) or remaining_updates < 1:
        reasons.append("remaining_updates must include the next update")
        remaining_updates = max(0, int(remaining_updates)) if isinstance(remaining_updates, int) else 0
    try:
        update_seconds = _finite(worst_case_seconds_per_update, "worst_case_seconds_per_update", 0.000001)
        rate = _finite(quote.get("compute_usd_per_gpu_hour"), "compute_usd_per_gpu_hour", 0.000001)
        billing_increment = _finite(quote.get("billing_increment_hours"), "billing_increment_hours", 0.000001)
        minimum_hours = _finite(quote.get("minimum_billable_hours"), "minimum_billable_hours", 0.000001)
        shutdown_seconds = _finite(quote.get("shutdown_and_export_compute_seconds_remaining"), "shutdown_and_export_compute_seconds_remaining")
        elapsed_seconds = _finite(ledger.get("current_lease_elapsed_seconds"), "ledger.current_lease_elapsed_seconds")
        committed_core = _finite(ledger.get("committed_core_usd"), "ledger.committed_core_usd")
        committed_final = _finite(ledger.get("committed_final_usd"), "ledger.committed_final_usd")
        noncompute_remaining = _finite(quote.get("noncompute_core_remaining_usd"), "quote.noncompute_core_remaining_usd")
        final_remaining = _finite(quote.get("final_storage_export_remaining_usd"), "quote.final_storage_export_remaining_usd")
    except ValueError as exc:
        reasons.append(str(exc))
        update_seconds = rate = billing_increment = minimum_hours = shutdown_seconds = 0.0
        elapsed_seconds = committed_core = committed_final = noncompute_remaining = final_remaining = 0.0

    raw_future_hours = (elapsed_seconds + remaining_updates * update_seconds + shutdown_seconds) / 3600.0
    if billing_increment > 0:
        billed_hours = math.ceil(max(minimum_hours, raw_future_hours) / billing_increment) * billing_increment
    else:
        billed_hours = minimum_hours
    compute_forecast = billed_hours * rate
    projected_core = committed_core + compute_forecast + noncompute_remaining
    projected_final = committed_final + final_remaining
    projected_total = projected_core + projected_final

    limits = quote.get("limits")
    if not isinstance(limits, dict):
        reasons.append("campaign limits are missing")
        soft_stop, final_reserve, total_cap = 4.0, 12.0, 16.0
    else:
        try:
            soft_stop = _finite(limits.get("soft_training_stop_usd"), "soft_training_stop_usd", 0.01)
            final_reserve = _finite(limits.get("final_charge_reserve_usd"), "final_charge_reserve_usd", 0.01)
            total_cap = _finite(limits.get("campaign_total_usd"), "campaign_total_usd", 0.01)
        except ValueError as exc:
            reasons.append(str(exc))
            soft_stop, final_reserve, total_cap = 4.0, 12.0, 16.0
    if (soft_stop, final_reserve, total_cap) != (4.0, 12.0, 16.0):
        reasons.append("offer is not bound to the $4 all-in soft stop and $16 hard campaign ceiling")
    # Include storage, egress, fees and already-paid final costs at the soft
    # boundary. The $12 difference is headroom, never automatic spend authority.
    if projected_total >= BUDGET_POLICY["soft_training_stop_usd"]:
        reasons.append("projected all-in spend reaches the $4 soft stop; user decision required")
    if projected_final > final_reserve:
        reasons.append("projected final charges exceed the $12 non-discretionary headroom")
    if projected_total > BUDGET_POLICY["campaign_total_usd"]:
        reasons.append("projected campaign total exceeds the $16 hard ceiling")

    return CostDecision(
        allowed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        projected_core_usd=projected_core,
        projected_final_usd=projected_final,
        projected_total_usd=projected_total,
        projected_billed_compute_hours=billed_hours,
        quote_age_seconds=quote_age,
        ledger_age_seconds=ledger_age,
    )


class BudgetGate:
    """Trainer callback adapter; ledger source is supplied by the trusted root."""

    def __init__(
        self,
        quote: dict[str, Any],
        ledger_supplier: Callable[[], dict[str, Any]],
        worst_case_seconds_per_update: float,
    ) -> None:
        self._quote = quote
        self._ledger_supplier = ledger_supplier
        self._seconds_per_update = worst_case_seconds_per_update
        self.last_decision: CostDecision | None = None

    @property
    def budget_policy(self) -> dict[str, float]:
        return dict(BUDGET_POLICY)

    @property
    def last_decision_json(self) -> dict[str, Any] | None:
        return self.last_decision.as_dict() if self.last_decision is not None else None

    def before_update(self, remaining_updates: int) -> bool:
        try:
            ledger = self._ledger_supplier()
        except Exception as exc:
            decision = CostDecision(False, (f"campaign ledger unavailable: {type(exc).__name__}",), 0, 0, 0, 0, 0, 0)
            self.last_decision = decision
            raise BudgetStop("budget gate denied update: campaign ledger unavailable") from exc
        try:
            if not isinstance(ledger, dict):
                raise ValueError("campaign ledger supplier returned a non-object")
            decision = evaluate_before_update(
                self._quote,
                ledger,
                remaining_updates=remaining_updates,
                worst_case_seconds_per_update=self._seconds_per_update,
            )
        except Exception as exc:
            decision = CostDecision(False, (f"budget inputs invalid: {type(exc).__name__}",), 0, 0, 0, 0, 0, 0)
            self.last_decision = decision
            raise BudgetStop("budget gate denied update: invalid cost inputs") from exc
        self.last_decision = decision
        if not decision.allowed:
            raise BudgetStop("budget gate denied update: " + "; ".join(decision.reasons))
        return True


BeforeUpdateGate = BudgetGate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quote", type=Path, help="root-verified fresh Vast offer snapshot JSON")
    parser.add_argument("ledger", type=Path, help="root-authenticated current campaign ledger JSON")
    parser.add_argument("--remaining-updates", type=int, required=True)
    parser.add_argument("--worst-case-seconds-per-update", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        decision = evaluate_before_update(
            _read_object(args.quote),
            _read_object(args.ledger),
            remaining_updates=args.remaining_updates,
            worst_case_seconds_per_update=args.worst_case_seconds_per_update,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"allowed": False, "reason": str(exc), "hard_cap_enforced": False}, sort_keys=True))
        return 2
    print(json.dumps(decision.as_dict(), indent=2, sort_keys=True))
    return 0 if decision.allowed else 3


if __name__ == "__main__":
    raise SystemExit(main())

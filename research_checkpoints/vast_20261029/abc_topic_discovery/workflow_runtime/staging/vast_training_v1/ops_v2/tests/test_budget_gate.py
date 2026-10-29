import unittest
from datetime import datetime, timedelta, timezone

from budget_gate import BudgetGate, BudgetStop, evaluate_before_update


NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def quote() -> dict:
    return {
        "schema": "vast-verified-offer-v1",
        "provider": "Vast.ai",
        "offer_id": "synthetic-offer",
        "currency": "USD",
        "verified_by_root": True,
        "source_sha256": "a" * 64,
        "verified_price_categories": {
            "on_demand_compute": True,
            "minimum_billing": True,
            "stopped_storage": True,
            "egress_and_checkpoint_export": True,
            "fees_and_taxes": True,
        },
        "checked_at_utc": NOW.isoformat(),
        "compute_usd_per_gpu_hour": 1.0,
        "billing_increment_hours": 1.0,
        "minimum_billable_hours": 1.0,
        "shutdown_and_export_compute_seconds_remaining": 90,
        "noncompute_core_remaining_usd": 2.0,
        "final_storage_export_remaining_usd": 1.0,
        "limits": {
            "soft_training_stop_usd": 16.0,
            "final_charge_reserve_usd": 4.0,
            "campaign_total_usd": 20.0,
        },
    }


def ledger() -> dict:
    return {
        "schema": "vast-campaign-ledger-v1",
        "root_authenticated": True,
        "as_of_utc": NOW.isoformat(),
        "committed_core_usd": 1.0,
        "committed_final_usd": 0.5,
        "current_lease_elapsed_seconds": 60.0,
    }


class BudgetGateTests(unittest.TestCase):
    def test_fresh_verified_offer_and_ledger_allow_with_margin(self) -> None:
        result = evaluate_before_update(
            quote(), ledger(), remaining_updates=100,
            worst_case_seconds_per_update=10.0, now=NOW,
        )
        self.assertTrue(result.allowed)
        self.assertLess(result.projected_core_usd, 16.0)
        self.assertLessEqual(result.projected_final_usd, 4.0)
        self.assertLessEqual(result.projected_total_usd, 20.0)
        self.assertFalse(result.hard_cap_enforced)

    def test_update_is_denied_before_soft_threshold_even_if_total_is_under_20(self) -> None:
        state = ledger()
        state["committed_core_usd"] = 14.5
        result = evaluate_before_update(
            quote(), state, remaining_updates=1,
            worst_case_seconds_per_update=1.0, now=NOW,
        )
        self.assertFalse(result.allowed)
        self.assertGreaterEqual(result.projected_core_usd, 16.0)
        self.assertLess(result.projected_total_usd, 20.0)

    def test_final_charge_reserve_is_not_available_for_training(self) -> None:
        state = ledger()
        state["committed_final_usd"] = 3.5
        result = evaluate_before_update(
            quote(), state, remaining_updates=1,
            worst_case_seconds_per_update=1.0, now=NOW,
        )
        self.assertFalse(result.allowed)
        self.assertGreater(result.projected_final_usd, 4.0)

    def test_stale_or_unverified_quote_and_ledger_fail_closed(self) -> None:
        stale = quote()
        stale["checked_at_utc"] = (NOW - timedelta(hours=2)).isoformat()
        result = evaluate_before_update(
            stale, ledger(), remaining_updates=1,
            worst_case_seconds_per_update=1.0, now=NOW,
        )
        self.assertFalse(result.allowed)
        incomplete = quote()
        incomplete["verified_price_categories"]["stopped_storage"] = False
        result = evaluate_before_update(
            incomplete, ledger(), remaining_updates=1,
            worst_case_seconds_per_update=1.0, now=NOW,
        )
        self.assertFalse(result.allowed)
        state = ledger()
        state["root_authenticated"] = False
        result = evaluate_before_update(
            quote(), state, remaining_updates=1,
            worst_case_seconds_per_update=1.0, now=NOW,
        )
        self.assertFalse(result.allowed)

    def test_callback_raises_budget_stop_when_ledger_unavailable(self) -> None:
        def unavailable():
            raise OSError("synthetic ledger missing")

        gate = BudgetGate(quote(), unavailable, 10.0)
        with self.assertRaises(BudgetStop):
            gate.before_update(remaining_updates=100)
        self.assertIsNotNone(gate.last_decision)
        self.assertFalse(gate.last_decision.allowed)
        self.assertEqual(gate.budget_policy, {
            "soft_training_stop_usd": 16.0,
            "final_charge_reserve_usd": 4.0,
            "campaign_total_usd": 20.0,
        })
        self.assertIsInstance(gate.last_decision_json, dict)


if __name__ == "__main__":
    unittest.main()

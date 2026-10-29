import unittest

from budget_guard import PlanError, evaluate


def plan() -> dict:
    return {
        "plan_id": "test-only",
        "provider": "synthetic",
        "budget_cap_usd": 20.0,
        "reserve_usd": 4.0,
        "charges_already_incurred_usd": 2.0,
        "prior_final_storage_export_charges_usd": 0.5,
        "final_storage_export_estimate_usd": 1.0,
        "gpu_count": 1,
        "compute_usd_per_gpu_hour": 1.0,
        "instance_hours_max": 8.0,
        "storage": {"gib": 10, "billed_hours": 1000, "usd_per_gib_hour": 0.0001, "includes_stopped_instance_storage": True},
        "egress": {"gib_max": 5, "usd_per_gib": 0.1, "includes_checkpoint_copy_out": True},
        "fees_tax_and_other_usd": 0.5,
        "retry_failure_allowance_usd": 0.5,
        "price_source": {"source": "synthetic test fixture", "checked_at_utc": "2026-01-01T00:00:00Z"},
    }


class BudgetGuardTests(unittest.TestCase):
    def test_all_categories_fit_under_soft_and_total_caps(self) -> None:
        result = evaluate(plan())
        self.assertTrue(result["budget_math_fits_with_reserve"])
        self.assertEqual(result["soft_pause_threshold_usd"], 16.0)
        self.assertEqual(result["modeled_costs_usd"]["including_prior_charges"], 13.5)
        self.assertFalse(result["reserve_discretionary_training_allowed"])
        self.assertIn("NOT_CHECKED", result["provider_spend_enforcement"])

    def test_crossing_soft_threshold_fails_even_below_total_cap(self) -> None:
        fixture = plan()
        fixture["instance_hours_max"] = 13.0
        result = evaluate(fixture)
        self.assertFalse(result["budget_math_fits_with_reserve"])
        self.assertLessEqual(result["modeled_costs_usd"]["including_prior_charges"], 20.0)
        self.assertGreater(result["modeled_costs_usd"]["core_total_including_prior"], 16.0)

    def test_reaching_soft_threshold_denies(self) -> None:
        fixture = plan()
        fixture["instance_hours_max"] = 12.0
        result = evaluate(fixture)
        self.assertEqual(result["modeled_costs_usd"]["core_total_including_prior"], 16.0)
        self.assertFalse(result["budget_math_fits_with_reserve"])

    def test_final_storage_export_cannot_consume_more_than_reserve(self) -> None:
        fixture = plan()
        fixture["final_storage_export_estimate_usd"] = 4.0
        result = evaluate(fixture)
        self.assertFalse(result["budget_math_fits_with_reserve"])

    def test_missing_or_wrong_reserve_values_fail_closed(self) -> None:
        fixture = plan()
        fixture["reserve_usd"] = 3.99
        with self.assertRaises(PlanError):
            evaluate(fixture)
        fixture = plan()
        fixture["compute_usd_per_gpu_hour"] = None
        with self.assertRaises(PlanError):
            evaluate(fixture)


if __name__ == "__main__":
    unittest.main()

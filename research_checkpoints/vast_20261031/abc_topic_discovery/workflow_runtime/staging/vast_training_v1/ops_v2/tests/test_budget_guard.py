import unittest
from datetime import datetime, timedelta, timezone
from budget_guard import PlanError, evaluate

NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)


def plan():
    return {
        'plan_id': 'synthetic', 'provider': 'synthetic',
        'budget_cap_usd': 16.0, 'reserve_usd': 12.0,
        'billing_increment_hours': 1 / 3600, 'minimum_billable_hours': 1 / 3600,
        'charges_already_incurred_usd': 0.5,
        'prior_final_storage_export_charges_usd': 0.1,
        'final_storage_export_estimate_usd': 0.4,
        'gpu_count': 1, 'compute_usd_per_gpu_hour': 1.0, 'instance_hours_max': 1.0,
        'storage': {'gib': 10, 'billed_hours': 1000, 'usd_per_gib_hour': 0.00001,
                    'includes_stopped_instance_storage': True},
        'egress': {'gib_max': 0.5, 'usd_per_gib': 0.1, 'includes_checkpoint_copy_out': True},
        'fees_tax_and_other_usd': 0.1, 'retry_failure_allowance_usd': 0.1,
        'price_source': {'source': 'synthetic fixture', 'checked_at_utc': NOW.isoformat()},
    }


class BudgetGuardTests(unittest.TestCase):
    def test_all_categories_count_at_all_in_soft_stop(self):
        result = evaluate(plan(), now=NOW)
        self.assertTrue(result['budget_math_fits_with_reserve'])
        self.assertEqual(result['soft_pause_threshold_usd'], 4)
        self.assertEqual(result['budget_cap_usd'], 16)
        self.assertEqual(result['modeled_costs_usd']['including_prior_charges'], 2.25)
        self.assertFalse(result['reserve_discretionary_training_allowed'])
        self.assertFalse(result['run_authorized'])

    def test_soft_boundary_includes_final_fees(self):
        fixture = plan(); fixture['instance_hours_max'] = 2.75
        result = evaluate(fixture, now=NOW)
        self.assertEqual(result['modeled_costs_usd']['including_prior_charges'], 4)
        self.assertLess(result['modeled_costs_usd']['core_total_including_prior'], 4)
        self.assertFalse(result['budget_math_fits_with_reserve'])

    def test_old_policy_cannot_raise_limits(self):
        fixture = plan(); fixture.update(budget_cap_usd=20, reserve_usd=4)
        with self.assertRaises(PlanError): evaluate(fixture, now=NOW)

    def test_minimum_and_increment_rounding_cannot_hide_cost(self):
        fixture = plan(); fixture.update(instance_hours_max=0.01, minimum_billable_hours=3,
                                        billing_increment_hours=1)
        result = evaluate(fixture, now=NOW)
        self.assertEqual(result['projected_billed_compute_hours'], 3)
        self.assertFalse(result['budget_math_fits_with_reserve'])

    def test_prior_retry_and_final_costs_are_not_reset(self):
        fixture = plan(); fixture['charges_already_incurred_usd'] = 15.9
        result = evaluate(fixture, now=NOW)
        self.assertGreater(result['modeled_costs_usd']['including_prior_charges'], 16)
        self.assertFalse(result['budget_math_fits_with_reserve'])

    def test_stale_future_naive_and_missing_prices_fail_closed(self):
        for stamp in ((NOW - timedelta(hours=2)).isoformat(),
                      (NOW + timedelta(seconds=1)).isoformat(), '2026-10-31', None):
            fixture = plan(); fixture['price_source']['checked_at_utc'] = stamp
            with self.subTest(stamp=stamp), self.assertRaises(PlanError): evaluate(fixture, now=NOW)
        fixture = plan(); fixture['compute_usd_per_gpu_hour'] = None
        with self.assertRaises(PlanError): evaluate(fixture, now=NOW)

    def test_storage_must_cover_instance_lifetime(self):
        fixture = plan(); fixture['storage']['billed_hours'] = 0
        with self.assertRaises(PlanError): evaluate(fixture, now=NOW)

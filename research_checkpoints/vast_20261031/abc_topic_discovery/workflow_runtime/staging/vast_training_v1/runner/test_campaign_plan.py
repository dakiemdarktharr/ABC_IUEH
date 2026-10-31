"""Generated command-plan tests; no command is executed."""
import unittest
from unittest.mock import patch
from .campaign_plan import build_plan, generate_run_ids
from .screen_plan import ARMS
from . import launcher


class PlanTests(unittest.TestCase):
    def config(self):
        return {"bundles": {str(i): {"directory": f"/bundle/{i}", "identity_sha256": "a"*64,
                "public_key": f"/keys/{i}", "key_id": f"key-{i}"} for i in range(3)},
            "model_sha256": {a: "b"*64 for a in ARMS}, "runner_sha256": "c"*64,
            "checkpoint_root": "/checkpoints", "run_ids": generate_run_ids(), "backup_dir": "/backup",
            "python_executable": "/opt/abc/.venv/bin/python",
            "storage_attestation": "/attestation", "quote": "/quote", "ledger": "/ledger",
            "worst_case_seconds_per_update": 0.01, "per_fit_overhead_seconds": 30,
            "time_ledger": "/ledger-dir/ledger.jsonl", "time_original": "/history/ledger.jsonl",
            "time_anchor": "/anchor-dir/anchor.jsonl", "reservation_seconds": 300,
            "shutdown_seconds": 30, "interrupt_grace_seconds": 15,
            "term_grace_seconds": 8, "kill_verify_seconds": 5}

    def test_inventory_and_each_selected_command_are_separate(self):
        plan = build_plan(self.config())
        self.assertEqual(plan["status"], "NOT_EXECUTED")
        self.assertEqual(len({row["fit"] for row in plan["fits"]}), 18)
        self.assertNotIn("selected_command", plan)
        for row in plan["fits"]:
            command = build_plan(self.config(), fit_id=row["fit"])["selected_command"]
            self.assertEqual(command["anchor_binding"], "read_current_at_launch_under_lock")
            argv = command["argv"]
            self.assertEqual(argv[:3], ["/opt/abc/.venv/bin/python", "-m", "vast_training_v1.runner.launcher"])
            self.assertRegex(argv[argv.index("--run-id") + 1], r"^[0-9a-f]{32}$")
            self.assertEqual(argv[argv.index("--time-anchor") + 1], "/anchor-dir/anchor.jsonl")
            self.assertNotIn("--expected-anchor", argv)
            with patch.object(launcher, "open_prepared_train_bundle") as bundle:
                with self.assertRaises(PermissionError):
                    launcher.main(argv[3:])
                bundle.assert_not_called()


    def test_rclone_plan_uses_exclusive_remote_option(self):
        config=self.config(); del config['backup_dir']; del config['storage_attestation']
        config['backup_rclone_remote']='drive:private/run'
        config['time_journal_remote']='drive:private/time-journal'
        config['time_journal_head']='/independent/time-journal-head'
        fit=build_plan(config)['fits'][0]['fit']
        argv=build_plan(config,fit_id=fit)['selected_command']['argv']
        self.assertEqual(argv[argv.index('--backup-rclone-remote')+1], 'drive:private/run')
        self.assertEqual(argv[argv.index('--time-journal-remote')+1], 'drive:private/time-journal')
        self.assertNotIn('--backup-dir',argv)
        config['storage_attestation']='/attestation'
        with self.assertRaises(ValueError): build_plan(config)

    def test_missing_bundle_fails(self):
        config = self.config()
        del config["bundles"]["2"]
        with self.assertRaises(ValueError):
            build_plan(config)

    def test_run_ids_must_be_complete_unique_and_valid(self):
        config = self.config()
        ids = config["run_ids"]
        self.assertEqual(len(set(ids.values())), 18)
        first, second = list(ids)[:2]
        ids[second] = ids[first]
        with self.assertRaises(ValueError): build_plan(config)
        ids[second] = "bad"
        with self.assertRaises(ValueError): build_plan(config)
        del ids[second]
        with self.assertRaises(ValueError): build_plan(config)

    def test_invalid_time_config_and_unknown_fit_fail(self):
        for field, value in [("reservation_seconds", True), ("reservation_seconds", 12451),
                             ("shutdown_seconds", 299), ("term_grace_seconds", 31),
                             ("kill_verify_seconds", float("nan")), ("time_anchor", "/ledger-dir/anchor"),
                             ("python_executable", "/usr/bin/python3.11")]:
            config = self.config(); config[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_plan(config)
        with self.assertRaises(ValueError): build_plan(self.config(), fit_id="unknown")

    def test_full_fit_and_campaign_must_fit_reservations_before_launch(self):
        for field, value in [('worst_case_seconds_per_update', 1),
                             ('per_fit_overhead_seconds', 250),
                             ('reservation_seconds', 700)]:
            config = self.config(); config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): build_plan(config)


if __name__ == "__main__":
    unittest.main()

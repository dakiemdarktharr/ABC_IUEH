"""Generated-only launcher gate tests without torch or recordings."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vast_training_v1.runner import launcher
from vast_training_v1.runner.fit_api import ScreenRecipe


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.identity = "a" * 64
        self.controller = launcher.TrainingController(recipe=ScreenRecipe(), identity=self.identity, run_id="d" * 32,
            expected_runner="b" * 64, expected_model="c" * 64, resume_digest=None,
            backup_dir=Path("unused"), storage_attestation=Path("unused"))

    def test_launcher_denies_before_bundle_access(self):
        args = ["--mode", "pilot", "--run-id", "d" * 32, "--fold-id", "0", "--seed", "20261026",
                "--arm", "ordered_product", "--bundle-dir", "unused", "--bundle-sha256", "unused",
                "--public-key", "unused", "--key-id", "unused", "--runner-sha256", "unused",
                "--model-sha256", "unused", "--checkpoint-dir", "unused", "--backup-dir", "unused",
                "--storage-attestation", "unused", "--quote", "unused", "--ledger", "unused",
                "--worst-case-seconds-per-update", "1", "--time-ledger", "unused-ledger",
                "--time-original", "unused-original", "--time-anchor", "unused-anchor",
                "--reservation-seconds", "20", "--shutdown-seconds", "5",
                "--interrupt-grace-seconds", "2", "--term-grace-seconds", "1",
                "--kill-verify-seconds", "1"]
        with patch.object(launcher, "open_prepared_train_bundle") as reader:
            with self.assertRaises(PermissionError):
                launcher.main(args)
            reader.assert_not_called()

    def test_purpose_and_recipe_are_bound(self):
        c = self.controller
        manifest = {"purpose": ScreenRecipe().purpose, "role": "outer_fit10", "fold_id": 0}
        lease = c.authorize_verified_bundle(manifest=manifest, bundle_identity_sha256=self.identity,
            expected_run_binding=c.lease, expected_identity_sha256=self.identity)
        self.assertIs(lease, c.lease)
        manifest["purpose"] = "ordered_motion_exploratory_fit_only_micro_pilot"
        with self.assertRaises(PermissionError):
            c.authorize_verified_bundle(manifest=manifest, bundle_identity_sha256=self.identity,
                expected_run_binding=c.lease, expected_identity_sha256=self.identity)

    def test_backup_failure_stops_callback(self):
        c = self.controller
        with tempfile.TemporaryDirectory() as tmp:
            payload = Path(tmp) / "checkpoint.pt"
            payload.write_bytes(b"generated")
            digest = hashlib.sha256(b"generated").hexdigest()
            meta = Path(tmp) / "metadata.json"
            meta.write_text(json.dumps({"payload_sha256": digest, "global_step": 100,
                "data_sha256": self.identity, "code_sha256": "b" * 64,
                "model_sha256": "c" * 64, "recipe_sha256": c.recipe_sha,
                "run_id": "d"*32, "plan_sha256": c.plan_sha, "fold_id": c.recipe["fold_id"],
                "training_seed": c.recipe["training_seed"], "arm": c.recipe["arm_id"]}))
            with patch.object(launcher, "backup_checkpoint", side_effect=OSError("backup failed")):
                with self.assertRaises(OSError):
                    c.checkpoint_committed(checkpoint_sha256=digest, completed_updates=100,
                        data_identity_sha256=self.identity, recipe_sha256=c.recipe_sha,
                        checkpoint_path=str(payload), metadata_path=str(meta))
            self.assertEqual(c.backups, [])


    def test_remote_backup_is_synchronous_and_pins_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=launcher.TrainingController(recipe=ScreenRecipe(),identity=self.identity,run_id='d'*32,
                expected_runner='b'*64,expected_model='c'*64,resume_digest=None,remote_prefix='drive:private/run')
            payload=Path(tmp)/'payload'; payload.write_bytes(b'generated'); digest=hashlib.sha256(b'generated').hexdigest()
            meta=Path(tmp)/'meta.json'
            values={'payload_sha256':digest,'global_step':100,'data_sha256':self.identity,'code_sha256':'b'*64,
                    'model_sha256':'c'*64,'recipe_sha256':c.recipe_sha,'run_id':'d'*32,'plan_sha256':c.plan_sha,
                    'fold_id':c.recipe['fold_id'],'training_seed':c.recipe['training_seed'],'arm':c.recipe['arm_id']}
            meta.write_text(json.dumps(values))
            args=dict(checkpoint_sha256=digest,completed_updates=100,data_identity_sha256=self.identity,
                      recipe_sha256=c.recipe_sha,checkpoint_path=str(payload),metadata_path=str(meta))
            result={'manifest_remote':'drive:private/run/manifest','manifest_sha256':'e'*64,'payload_sha256':digest}
            with patch.object(launcher.remote_backup,'backup_checkpoint',return_value=result): c.checkpoint_committed(**args)
            self.assertEqual(c.backups[0]['run_id'],'d'*32)
            values['run_id']='f'*32; meta.write_text(json.dumps(values))
            with patch.object(launcher.remote_backup,'backup_checkpoint') as upload, self.assertRaises(PermissionError):
                c.checkpoint_committed(**args)
            upload.assert_not_called()

    def test_remote_backup_failure_revokes_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=launcher.TrainingController(recipe=ScreenRecipe(),identity=self.identity,run_id='d'*32,
                expected_runner='b'*64,expected_model='c'*64,resume_digest=None,remote_prefix='drive:private/run')
            payload=Path(tmp)/'payload'; payload.write_bytes(b'x'); digest=hashlib.sha256(b'x').hexdigest()
            meta=Path(tmp)/'meta'; meta.write_text(json.dumps({'payload_sha256':digest,'global_step':100,
                'data_sha256':self.identity,'code_sha256':'b'*64,'model_sha256':'c'*64,'recipe_sha256':c.recipe_sha,
                'run_id':'d'*32,'plan_sha256':c.plan_sha,'fold_id':c.recipe['fold_id'],
                'training_seed':c.recipe['training_seed'],'arm':c.recipe['arm_id']}))
            with patch.object(launcher.remote_backup,'backup_checkpoint',side_effect=OSError('failure')):
                with self.assertRaises(OSError): c.checkpoint_committed(checkpoint_sha256=digest,completed_updates=100,
                    data_identity_sha256=self.identity,recipe_sha256=c.recipe_sha,checkpoint_path=str(payload),metadata_path=str(meta))
            self.assertFalse(c.lease.active); self.assertEqual(c.backups,[])

    def test_live_meter_advances_elapsed_without_claiming_provider_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quote, ledger = root / "quote.json", root / "ledger.json"
            quote.write_text("{}")
            state = {"schema": "vast-campaign-ledger-v1", "root_authenticated": True,
                "as_of_utc": datetime.now(timezone.utc).isoformat(), "committed_core_usd": 1.0,
                "committed_final_usd": 0.5, "current_lease_elapsed_seconds": 60.0}
            ledger.write_text(json.dumps(state))
            seen = []
            class FakeGate:
                last_decision_json = {"allowed": True}
                def __init__(self, _quote, supplier, _seconds):
                    seen.append(supplier())
                def before_update(self, **kwargs):
                    return True
            with patch.object(launcher.time, "monotonic", side_effect=[100.0, 175.0]), \
                 patch.object(launcher, "BudgetGate", FakeGate):
                meter = launcher.LiveBudgetGate(quote, ledger, 1.0)
                self.assertTrue(meter.before_update(remaining_updates=10))
            self.assertGreaterEqual(seen[0]["current_lease_elapsed_seconds"], 135.0)
            self.assertLess(seen[0]["current_lease_elapsed_seconds"], 136.0)
            self.assertEqual(seen[0]["committed_core_usd"], 1.0)
            state["committed_core_usd"] = 0.5
            ledger.write_text(json.dumps(state))
            with self.assertRaises(launcher.BudgetStop):
                meter.before_update(remaining_updates=10)

    def test_price_change_cannot_reprice_previously_elapsed_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); quote, ledger = root / 'quote', root / 'ledger'
            quote.write_text(json.dumps({'offer_id': 'first', 'compute_usd_per_gpu_hour': 1}))
            ledger.write_text(json.dumps({'schema': 'vast-campaign-ledger-v1', 'root_authenticated': True,
                'as_of_utc': datetime.now(timezone.utc).isoformat(), 'current_lease_elapsed_seconds': 10,
                'committed_core_usd': 0, 'committed_final_usd': 0}))
            gate = launcher.LiveBudgetGate(quote, ledger, 0.1)
            quote.write_text(json.dumps({'offer_id': 'first', 'compute_usd_per_gpu_hour': 0.1}))
            with self.assertRaises(launcher.BudgetStop): gate.before_update(remaining_updates=1)


if __name__ == "__main__":
    unittest.main()

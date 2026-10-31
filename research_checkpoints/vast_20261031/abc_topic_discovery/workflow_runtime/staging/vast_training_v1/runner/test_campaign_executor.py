"""Synthetic campaign execution; subprocess and checkpoints are generated fixtures."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import signal
import subprocess
import time
from contextlib import nullcontext
import tempfile
import unittest
from unittest.mock import patch

from . import campaign_executor as ex
from . import campaign_plan
from . import test_campaign_plan
check_python = ex._check_python


def sha(raw): return hashlib.sha256(raw).hexdigest()


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.backup = self.root / 'backup'; self.backup.mkdir()
        self.config = test_campaign_plan.PlanTests().config(); self.config['backup_dir'] = str(self.backup.resolve())
        ledger = self.root / 'usd.json'
        ledger.write_text(json.dumps({'schema':'vast-campaign-ledger-v1', 'root_authenticated':True,
            'as_of_utc':datetime.now(timezone.utc).isoformat(), 'current_lease_elapsed_seconds':10,
            'committed_core_usd':0, 'committed_final_usd':0}))
        self.config['ledger'] = str(ledger.resolve())
        from ..ops_v2.tests.test_budget_gate import quote
        offer = quote(); offer['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
        price = self.root / 'quote.json'; price.write_text(json.dumps(offer))
        self.config['quote'] = str(price.resolve())
        p = patch.object(campaign_plan, '_path', side_effect=lambda value: value)
        p.start(); self.addCleanup(p.stop)
        p = patch.object(ex, '_check_python'); p.start(); self.addCleanup(p.stop)
        p = patch.object(ex, 'check_capacity'); p.start(); self.addCleanup(p.stop)
        p = patch.object(ex, '_check_stop_client'); p.start(); self.addCleanup(p.stop)
        p = patch.object(ex, '_campaign_deadline', side_effect=lambda _: nullcontext())
        p.start(); self.addCleanup(p.stop)
        p = patch.object(ex.lease_clock, '_boot_id', return_value='0' * 32); p.start(); self.addCleanup(p.stop)
        self.calls = []
        self.fail_at = None
        self.wrong_fit = False
        self.remote_objects = {}
        self.omit_remote_pin = False
        self.wrong_metadata = False

    def fake_run(self, argv, *, stdout, stderr, **kwargs):
        def arg(flag): return argv[argv.index(flag) + 1]
        self.assertIn('--shared-lease-clock', argv)
        self.assertIn('--lease-clock-genesis', argv)
        fold, seed, arm = int(arg('--fold-id')), int(arg('--seed')), arg('--arm')
        name = f'{fold}-{seed}-{arm}'; self.calls.append(name)
        if self.fail_at == len(self.calls):
            stderr.write(b'synthetic budget denial\n')
            return type('Result', (), {'returncode': 3})()
        raw = ('synthetic-' + name).encode(); ph = sha(raw)
        remote = 'backup_rclone_remote' in self.config
        payload_name = ('checkpoint-' + ph + '.pt') if remote else (name + '.pt')
        meta_name = ('checkpoint-' + ph + '.meta.json') if remote else (name + '.json')
        run_id = self.config['run_ids'][f'fold-{fold}-seed-{seed}-{arm}']
        plan_sha = ex.THREE_ARM_DRAFT_SHA256
        payload = self.backup / payload_name; payload.write_bytes(raw)
        meta = {'schema': 'abc-training-checkpoint-v1', 'run_id': run_id, 'plan_sha256': plan_sha,
                'data_sha256': 'a'*64, 'code_sha256': 'c'*64, 'model_sha256': 'b'*64,
                'fold_id': fold, 'training_seed': seed, 'arm': arm, 'global_step': 5000,
                'total_steps': 5000, 'payload_sha256': ph, 'payload_size_bytes': len(raw),
                'resume_state_complete': True, 'state_components': {k: True for k in
                    ('model', 'optimizer', 'sampler', 'python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng')}}
        if self.wrong_metadata: meta['run_id'] = 'f' * 32
        mb = json.dumps(meta, sort_keys=True).encode(); meta_path = self.backup / meta_name; meta_path.write_bytes(mb)
        manifest = {'schema': 'abc-checkpoint-copy-v1', 'payload_file': payload.name,
            'payload_sha256': ph, 'payload_size_bytes': len(raw), 'metadata_file': meta_path.name,
            'metadata_sha256': sha(mb), 'metadata_size_bytes': len(mb)}
        manifest_path = self.backup / (name + '.manifest'); manifest_path.write_text(json.dumps(manifest))
        report = {'purpose': 'ordered_motion_three_arm_screen', 'role': 'outer_fit10',
            'fold_id': fold + int(self.wrong_fit), 'seed': seed, 'arm': arm, 'completed_updates': 5000,
            'data_identity_sha256': 'a'*64, 'model_sha256': 'b'*64, 'checkpoint_sha256': ph,
            'run_id': run_id}
        final = {'manifest_sha256': ex.sha256_file(manifest_path), 'payload_sha256': ph, 'step': 5000}
        if remote:
            prefix = self.config['backup_rclone_remote']
            final.update({'manifest_remote': f'{prefix}/checkpoint-{ph}.manifest.json', 'run_id': run_id,
                'plan_sha256': plan_sha, 'data_sha256': 'a'*64, 'code_sha256': 'c'*64,
                'model_sha256': 'b'*64, 'fold_id': fold, 'training_seed': seed, 'arm': arm})
            if self.omit_remote_pin: del report['run_id']
            self.remote_objects[f'{prefix}/{payload_name}'] = raw
            self.remote_objects[f'{prefix}/{meta_name}'] = mb
            self.remote_objects[final['manifest_remote']] = manifest_path.read_bytes()
        else:
            final['manifest_path'] = str(manifest_path.resolve())
        fit = {'fit': report, 'verified_backups': [final]}
        supervisor = {'time_supervisor': {'exit_code': 0, 'timed_out': False, 'orphaned': False,
            'external_stop_signal': None, 'same_process_group_empty_verified': True}}
        stdout.write((json.dumps({'checkpoint_backup_verified': final}) + '\n' + json.dumps(fit) + '\n' + json.dumps(supervisor) + '\n').encode())
        return type('Result', (), {'returncode': 0})()

    def test_full_schedule_and_verified_resume_then_tamper(self):
        state = self.root / 'state'
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run):
            result = ex.execute_campaign(self.config, state)
            self.assertEqual(result['fits_verified'], 18)
            self.assertEqual(len(self.calls), 18)
            self.assertEqual(ex.execute_campaign(self.config, state, continue_verified=True)['fits_verified'], 18)
            self.assertEqual(len(self.calls), 18)
            next(self.backup.glob('*.pt')).write_bytes(b'tampered')
            with self.assertRaises((ex.CampaignStop, ValueError)):
                ex.execute_campaign(self.config, state, continue_verified=True)
            self.assertEqual(len(self.calls), 18)

    def test_budget_failure_halts_and_does_not_retry(self):
        state = self.root / 'state'; self.fail_at = 3
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run):
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, state)
            self.assertEqual(len(self.calls), 3)
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, state, continue_verified=True)
            self.assertEqual(len(self.calls), 3)

    def test_success_code_with_wrong_report_stops_before_next_fit(self):
        self.wrong_fit = True
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run):
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, self.root / 'state')
            self.assertEqual(len(self.calls), 1)

    def test_remote_restores_and_verifies_each_fit_with_local_controller_pins(self):
        del self.config['backup_dir']; del self.config['storage_attestation']
        self.config['backup_rclone_remote'] = 'mock:checkpoints'
        self.config['time_journal_remote'] = 'mock:time-journal'; self.config['time_journal_head'] = '/independent/head'
        objects = self.remote_objects
        class Transport:
            def download_to(self, name, path): Path(path).write_bytes(objects[name])
            def read_digest(self, name): return sha(objects[name]), len(objects[name])
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run), \
             patch.object(ex.remote_backup, 'RcloneTransport', return_value=Transport()), \
             patch.object(ex.remote_backup.os, 'link', side_effect=lambda a, b: shutil.copyfile(a, b)):
            result = ex.execute_campaign(self.config, self.root / 'state')
            self.assertEqual(result['fits_verified'], 18)
            self.assertEqual(len(self.calls), 18)
            # The next pass must re-read remote bytes, not trust old receipts.
            obj = next(k for k in objects if k.endswith('.pt')); objects[obj] = b'tampered'
            with self.assertRaises((ex.CampaignStop, ValueError)):
                ex.execute_campaign(self.config, self.root / 'state', continue_verified=True)
            self.assertEqual(len(self.calls), 18)

    def test_remote_missing_local_run_pin_stops_before_next_fit(self):
        del self.config['backup_dir']; del self.config['storage_attestation']
        self.config['backup_rclone_remote'] = 'mock:checkpoints'; self.omit_remote_pin = True
        self.config['time_journal_remote'] = 'mock:time-journal'; self.config['time_journal_head'] = '/independent/head'
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run):
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, self.root / 'state')
            self.assertEqual(len(self.calls), 1)

    def test_crash_marker_never_retried(self):
        state = self.root / 'state'
        with patch.object(ex, '_run_launcher', side_effect=OSError('synthetic launch failure')):
            with self.assertRaises(OSError): ex.execute_campaign(self.config, state)
        with patch.object(ex, '_run_launcher') as call:
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, state, continue_verified=True)
            call.assert_not_called()

    def test_local_backup_rejects_another_run_even_when_bytes_and_sidecar_match(self):
        self.wrong_metadata = True
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run):
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, self.root / 'state')
        self.assertEqual(len(self.calls), 1)

    def test_campaign_forecast_rejects_total_schedule_before_any_launch(self):
        offer = json.loads(Path(self.config['quote']).read_text())
        offer['compute_usd_per_gpu_hour'] = 3
        Path(self.config['quote']).write_text(json.dumps(offer))
        with patch.object(ex, '_run_launcher') as launch, self.assertRaises(ex.CampaignStop):
            ex.execute_campaign(self.config, self.root / 'state')
        launch.assert_not_called()

    def test_sigterm_always_requests_provider_stop(self):
        config_file = self.root / 'config.json'; config_file.write_text(json.dumps(self.config))
        previous = signal.getsignal(signal.SIGTERM)
        with patch.object(ex, 'execute_campaign', side_effect=lambda *a, **k: signal.raise_signal(signal.SIGTERM)), \
             patch.object(ex, 'request_instance_stop', return_value={'cli_accepted': True}) as stop:
            self.assertEqual(ex.main([str(config_file), str(self.root / 'state'), '--stop-instance-id', '1234']), 130)
        stop.assert_called_once_with('1234')
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_launcher_timeout_requests_cooperative_containment_instead_of_killing_parent(self):
        with patch.object(ex.subprocess, 'Popen') as popen:
            child = popen.return_value
            child.wait.side_effect = [subprocess.TimeoutExpired('synthetic', 1), 0]
            child.poll.return_value = None
            with self.assertRaises(subprocess.TimeoutExpired):
                ex._run_launcher(['synthetic'], stdout=None, stderr=None, timeout=1, shutdown_seconds=2)
            child.send_signal.assert_called_once_with(signal.SIGINT)
            child.kill.assert_not_called()

    def test_gap_in_receipts_stops_before_a_new_fit(self):
        state = self.root / 'state'
        with patch.object(ex, '_run_launcher', side_effect=self.fake_run):
            ex.execute_campaign(self.config, state)
            rows = campaign_plan.build_plan(self.config)['fits']
            for suffix in ('.started.json', '.finished.json', '.stdout', '.stderr'):
                (state / (rows[0]['fit'] + suffix)).unlink()
            with self.assertRaises(ex.CampaignStop): ex.execute_campaign(self.config, state, continue_verified=True)
            self.assertEqual(len(self.calls), 18)

    def test_provider_stop_is_explicit_validated_and_never_claims_billing_stopped(self):
        with patch.object(ex.subprocess, 'run') as run:
            run.return_value.returncode = 0
            result = ex.request_instance_stop('1234')
            self.assertTrue(result['cli_accepted'])
            self.assertFalse(result['provider_stop_verified'])
            self.assertTrue(result['stopped_storage_may_still_be_billed'])
            self.assertEqual(run.call_args.args[0], ['vastai', 'stop', 'instance', '1234'])
            for bad in ('0', '01', '1;echo', '-1', '1 2', ''):
                with self.subTest(bad=bad), self.assertRaises(ValueError): ex.request_instance_stop(bad)
            self.assertEqual(run.call_count, 1)

    def test_main_requests_stop_on_success_and_failure_when_flagged(self):
        config_file = self.root / 'config.json'; config_file.write_text(json.dumps(self.config))
        with patch.object(ex, 'execute_campaign', return_value={'status': 'complete'}), \
             patch.object(ex, 'request_instance_stop', return_value={'cli_accepted': True}) as stop:
            self.assertEqual(ex.main([str(config_file), str(self.root / 'state'), '--stop-instance-id', '1234']), 0)
            stop.assert_called_once_with('1234')
        with patch.object(ex, 'execute_campaign', side_effect=ex.CampaignStop('synthetic')), \
             patch.object(ex, 'request_instance_stop', return_value={'cli_accepted': False}) as stop:
            self.assertEqual(ex.main([str(config_file), str(self.root / 'state'), '--stop-instance-id', '1234']), 2)
            stop.assert_called_once_with('1234')
        with patch.object(ex, 'execute_campaign', return_value={'status': 'complete'}), \
             patch.object(ex, 'request_instance_stop') as stop:
            with self.assertRaises(SystemExit): ex.main([str(config_file), str(self.root / 'state')])
            stop.assert_not_called()

class VenvTests(unittest.TestCase):
    def test_requires_executable_and_pyvenv_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / '.venv'; (root / 'bin').mkdir(parents=True)
            python = root / 'bin' / 'python'; python.write_text('synthetic')
            python.chmod(0o700)
            with self.assertRaises(ex.CampaignStop): check_python(python)
            (root / 'pyvenv.cfg').write_text('synthetic')
            check_python(python)
            with self.assertRaises(ex.CampaignStop): check_python(root / 'bin' / 'missing')


@unittest.skipUnless(hasattr(signal, 'setitimer'), 'Linux campaign deadline')
class DeadlineTests(unittest.TestCase):
    def test_whole_campaign_timer_interrupts_waiting_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGALRM)
        with patch.object(ex, 'build_plan', return_value={'full_campaign_reserved_seconds': 0.1}):
            with self.assertRaises(KeyboardInterrupt):
                with ex._campaign_deadline({}): time.sleep(2)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

if __name__ == '__main__': unittest.main()

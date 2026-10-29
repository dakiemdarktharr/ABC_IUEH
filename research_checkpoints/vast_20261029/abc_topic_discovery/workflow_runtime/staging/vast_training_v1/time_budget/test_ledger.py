import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import ledger
from . import bootstrap
from . import supervisor_linux as supervisor
from .guard import TimeBudgetGate
from .guard import TimeBudgetStop
from .combined import CombinedBudgetGate
from ..ops_v2 import budget_gate as usd
from ..runner import launcher
from .test_guard import HISTORY, rows

class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        raw, self.head = rows(HISTORY)
        self.original = root / 'original.jsonl'; self.original.write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()
        for name, value in [('HISTORY_SHA256', self.sha), ('HISTORY_HEAD', self.head)]:
            p = patch.object(ledger, name, value); p.start(); self.addCleanup(p.stop)
        self.path = root / 'new.jsonl'
        ledger.initialize_ledger(self.path, self.original)

    def authority(self, sha=None, head=None):
        return ledger.ReservationAuthority(self.path, self.original,
            expected_sha256=sha or self.sha, expected_head=head or self.head)

    def test_no_reset_and_permanent_debit(self):
        before = self.original.read_bytes(); a = self.authority()
        snap = a.reserve('a' * 16, 120)
        self.assertEqual(snap.nonexport_seconds, 270)
        with self.assertRaises((BlockingIOError, OSError)): self.authority()
        a.close(); self.assertEqual(self.original.read_bytes(), before)
        with self.assertRaises(FileExistsError): ledger.initialize_ledger(self.path, self.original)
        with self.assertRaises(ValueError): self.authority()
        b = self.authority(snap.file_sha256, snap.head_hash)
        with self.assertRaises(ValueError): b.reserve('a' * 16, 1)
        self.assertEqual(b.reserve('b' * 16, 5).nonexport_seconds, 275); b.close()

    def test_caps_and_invalid(self):
        a = self.authority()
        with self.assertRaises(ValueError): a.reserve('a' * 16, 12451)
        with self.assertRaises(ValueError): a.reserve('a' * 16, True)
        snap = a.reserve('a' * 16, 12450); a.close()
        b = self.authority(snap.file_sha256, snap.head_hash)
        with self.assertRaises(ValueError): b.reserve('b' * 16, 1)
        snap = b.reserve('b' * 16, 1800, kind='cuda_export'); b.close()
        c = self.authority(snap.file_sha256, snap.head_hash)
        with self.assertRaises(ValueError): c.reserve('c' * 16, 1, kind='cuda_export')
        c.close()

    def test_authority_one_use_and_gate(self):
        a = self.authority(); snap = a.reserve('a' * 16, 30)
        kw = dict(ledger_bytes=a.raw, anchored_head=snap.head_hash, reservation_id='a' * 16,
            authority=a, worst_update_seconds=1, shutdown_seconds=2, request_stop=lambda: None)
        gate = TimeBudgetGate(**kw)
        with self.assertRaises(PermissionError): TimeBudgetGate(**kw)
        gate.start_watchdog(); self.assertTrue(gate.before_update(remaining_updates=1))
        gate.close_after_process_tree_exits(); a.close()

    def test_tamper_and_partial_fail_closed(self):
        a = self.authority(); snap = a.reserve('a' * 16, 30); a.close()
        self.path.write_bytes(self.path.read_bytes() + b'{')
        with self.assertRaises(ValueError): self.authority(snap.file_sha256, snap.head_hash)
        self.original.write_bytes(b'tampered')
        with self.assertRaises(ValueError): self.authority(snap.file_sha256, snap.head_hash)

    def test_other_process_cannot_acquire_lock(self):
        a = self.authority()
        self.addCleanup(a.close)
        script = ('import sys; from vast_training_v1.time_budget.ledger import _acquire; '
                  '_acquire(sys.argv[1])')
        result = subprocess.run([sys.executable, '-B', '-c', script, str(self.path) + '.lock'],
                                capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        a.close()
        result = subprocess.run([sys.executable, '-B', '-c', script, str(self.path) + '.lock'],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_combined_denies_time_and_usd_and_lock_loss(self):
        a = self.authority(); snap = a.reserve('c' * 16, 30)
        clock = [10.0]
        tg = TimeBudgetGate(ledger_bytes=a.raw, anchored_head=snap.head_hash,
            reservation_id='c' * 16, authority=a, worst_update_seconds=1,
            shutdown_seconds=2, request_stop=lambda: None, clock=lambda: clock[0])
        cost = usd.BudgetGate({}, lambda: {}, 1)
        combined = CombinedBudgetGate(tg, cost)
        with self.assertRaises(TimeBudgetStop): combined.before_update(remaining_updates=1)
        tg.start_watchdog()
        allow = usd.CostDecision(True, (), 0, 0, 0, 0, 0, 0)
        deny = usd.CostDecision(False, ('synthetic denial',), 0, 0, 0, 0, 0, 0)
        with patch.object(usd, 'evaluate_before_update', return_value=allow):
            self.assertTrue(combined.before_update(remaining_updates=1))
        with patch.object(usd, 'evaluate_before_update', return_value=deny):
            with self.assertRaises(usd.BudgetStop): combined.before_update(remaining_updates=1)
        clock[0] = 37.0
        with self.assertRaises(TimeBudgetStop): combined.before_update(remaining_updates=1)
        a.close()
        with self.assertRaises(TimeBudgetStop): combined.before_update(remaining_updates=1)
        tg.close_after_process_tree_exits()

    def test_bootstrap_anchor_and_failed_anchor_are_fail_closed(self):
        anchor_dir = self.root / 'independent'; anchor_dir.mkdir()
        anchor = anchor_dir / 'anchor.jsonl'
        snap = ledger.verify_ledger(self.original.read_bytes(), expected_head=self.head)
        bootstrap.initialize_anchor(anchor, snap)
        session = bootstrap.AnchoredReservation(self.path, self.original, anchor, 'd' * 16, 10)
        self.assertEqual(session.snapshot.nonexport_seconds, 160)
        session.close()
        self.assertEqual(bootstrap._read(anchor)[1]['ledger_head'], session.snapshot.head_hash)
        with patch.object(bootstrap, '_write_all', side_effect=OSError('synthetic anchor failure')):
            with self.assertRaises(OSError):
                bootstrap.AnchoredReservation(self.path, self.original, anchor, 'e' * 16, 10)
        with self.assertRaises(ValueError):
            bootstrap.AnchoredReservation(self.path, self.original, anchor, 'f' * 16, 10)
        self.assertEqual(ledger.verify_ledger(self.path.read_bytes(),
            expected_head=__import__('json').loads(self.path.read_text().splitlines()[-1])['entry_hash']).nonexport_seconds, 170)

    def test_each_fit_reads_latest_anchor_and_adds_own_debit(self):
        anchor_dir = self.root / 'independent'; anchor_dir.mkdir()
        anchor = anchor_dir / 'anchor.jsonl'
        bootstrap.initialize_anchor(anchor, ledger.verify_ledger(self.original.read_bytes(), expected_head=self.head))
        first = bootstrap.AnchoredReservation(self.path, self.original, anchor, 'i' * 16, 20)
        first_head = first.snapshot.head_hash; first.close()
        second = bootstrap.AnchoredReservation(self.path, self.original, anchor, 'j' * 16, 30)
        self.assertEqual(second.snapshot.nonexport_seconds, 200)
        self.assertNotEqual(second.snapshot.head_hash, first_head)
        self.assertEqual(bootstrap._read(anchor)[1]['ledger_head'], second.snapshot.head_hash)
        second.close()

    def test_live_budget_gate_is_accepted_and_policy_checked(self):
        from datetime import datetime, timezone
        import json
        quote = self.root / 'quote.json'; quote.write_text('{}')
        state = self.root / 'usd.json'
        state.write_text(json.dumps({'schema': 'vast-campaign-ledger-v1', 'root_authenticated': True,
            'as_of_utc': datetime.now(timezone.utc).isoformat(), 'committed_core_usd': 0,
            'committed_final_usd': 0, 'current_lease_elapsed_seconds': 0}))
        live = launcher.LiveBudgetGate(quote, state, 1)
        a = self.authority(); snap = a.reserve('g' * 16, 30)
        tg = TimeBudgetGate(ledger_bytes=a.raw, anchored_head=snap.head_hash,
            reservation_id='g' * 16, authority=a, worst_update_seconds=1,
            shutdown_seconds=2, request_stop=lambda: None)
        combined = CombinedBudgetGate(tg, live)
        tg.start_watchdog()
        with patch.object(launcher, 'BudgetGate') as gate:
            gate.return_value.before_update.return_value = True
            self.assertTrue(combined.before_update(remaining_updates=1))
        a.close(); tg.close_after_process_tree_exits()

    def test_supervisor_parent_flow_with_mocked_fork(self):
        from unittest.mock import MagicMock
        anchor_dir = self.root / 'independent'; anchor_dir.mkdir()
        anchor = anchor_dir / 'anchor.jsonl'
        bootstrap.initialize_anchor(anchor, ledger.verify_ledger(self.original.read_bytes(), expected_head=self.head))
        session = bootstrap.AnchoredReservation(self.path, self.original, anchor, 'h' * 16, 30)
        self.addCleanup(session.close)
        cost = usd.BudgetGate({}, lambda: {}, 1)
        proc = MagicMock()
        proc.poll.side_effect = [None, 0, 0, 0]
        with patch.object(supervisor, '_linux_available', return_value=True), \
             patch.object(supervisor.os, 'fork', return_value=1234, create=True), \
             patch.object(supervisor.os, 'read', return_value=b'R'), \
             patch.object(supervisor.os, 'write', return_value=1), \
             patch.object(supervisor.select, 'select', return_value=([1], [], [])), \
             patch.object(supervisor.signal, 'SIGUSR1', 10, create=True), \
             patch.object(supervisor.signal, 'signal'), \
             patch.object(supervisor.signal, 'getsignal', return_value=None), \
             patch.object(supervisor, '_ForkChild', return_value=proc), \
             patch.object(supervisor.run_guard, 'wait_group_empty', return_value=True):
            result = supervisor.supervise(session, cost, worst_update_seconds=1, shutdown_seconds=5,
                interrupt_grace=1, term_grace=1, kill_verify=1, worker=lambda _: 0)
        self.assertTrue(result['same_process_group_empty_verified'])
        self.assertFalse(result['detached_sessions_checked'])
        self.assertEqual(result['exit_code'], 0)

if __name__ == '__main__': unittest.main()

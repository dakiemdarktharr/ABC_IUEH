"""Linux-only synthetic integration; no CUDA, Torch or human data."""
import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from . import bootstrap, ledger, supervisor_linux
from .guard import TimeBudgetStop
from .test_guard import HISTORY, rows

class FakeUSD:
    @property
    def budget_policy(self):
        return {'soft_training_stop_usd': 16.0, 'final_charge_reserve_usd': 4.0, 'campaign_total_usd': 20.0}
    def before_update(self, *, remaining_updates): return True

@unittest.skipUnless(os.name == 'posix' and Path('/proc').is_dir(), 'requires Linux process groups')
class LinuxSupervisorTests(unittest.TestCase):
    def test_timeout_requests_checkpoint_and_retains_reservation(self):
        raw, head = rows(HISTORY)
        with tempfile.TemporaryDirectory() as name, \
             patch.object(ledger, 'HISTORY_SHA256', hashlib.sha256(raw).hexdigest()), \
             patch.object(ledger, 'HISTORY_HEAD', head):
            root = Path(name); original = root / 'original'; original.write_bytes(raw)
            current = root / 'current'; snap = ledger.initialize_ledger(current, original)
            ext = root / 'ext'; ext.mkdir(); anchor = ext / 'anchor'
            bootstrap.initialize_anchor(anchor, snap)
            reservation = bootstrap.AnchoredReservation(current, original, anchor, 'z' * 16, 3)
            marker = root / 'checkpointed'
            def worker(gate):
                while True:
                    try: gate.before_update(remaining_updates=1)
                    except TimeBudgetStop:
                        marker.write_text('synthetic checkpoint'); return 0
                    time.sleep(.01)
            try:
                result = supervisor_linux.supervise(reservation, FakeUSD(), worst_update_seconds=.05,
                    shutdown_seconds=1, interrupt_grace=.4, term_grace=.2, kill_verify=.2, worker=worker)
                self.assertTrue(result['same_process_group_empty_verified'])
                self.assertTrue(marker.exists())
                self.assertEqual(reservation.snapshot.nonexport_seconds, 153)
            finally: reservation.close()

if __name__ == '__main__': unittest.main()

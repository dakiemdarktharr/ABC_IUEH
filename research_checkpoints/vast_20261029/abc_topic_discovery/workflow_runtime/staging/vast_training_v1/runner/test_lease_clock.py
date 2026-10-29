"""Synthetic multi-process USD lease accounting; no provider or GPU operations."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LeaseClockTests(unittest.TestCase):
    def invoke(self, code, *args):
        root = Path(__file__).resolve().parents[2]
        return subprocess.run([sys.executable, '-B', '-c', code, *map(str, args)], cwd=root,
            capture_output=True, text=True, timeout=10)

    def test_two_processes_preserve_clock_and_operator_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); clock = root / 'clock'; ledger = root / 'ledger.json'
            state = {'schema':'vast-campaign-ledger-v1','root_authenticated':True,
                     'as_of_utc':datetime.now(timezone.utc).isoformat(),
                     'current_lease_elapsed_seconds':10,'committed_core_usd':1,'committed_final_usd':0}
            ledger.write_text(json.dumps(state))
            prefix = "import json,sys; from pathlib import Path; from vast_training_v1.runner import lease_clock as c; "
            created = self.invoke(prefix + "c._boot_id=lambda:'boot-a'; c._now=lambda:100.; print(c.create(sys.argv[1],json.loads(Path(sys.argv[2]).read_text())))", clock, ledger)
            self.assertEqual(created.returncode, 0, created.stderr); genesis = created.stdout.strip()
            state['committed_core_usd'] = 3; ledger.write_text(json.dumps(state))
            evaluate = prefix + "c._boot_id=lambda:sys.argv[4]; c._now=lambda:float(sys.argv[5]); print(json.dumps(c.evaluate(sys.argv[1],sys.argv[3],json.loads(Path(sys.argv[2]).read_text()),mark=True)))"
            first = self.invoke(evaluate, clock, ledger, genesis, 'boot-a', 150)
            self.assertEqual(first.returncode, 0, first.stderr); self.assertGreaterEqual(json.loads(first.stdout)[0], 60)
            second = self.invoke(evaluate, clock, ledger, genesis, 'boot-a', 200)
            self.assertEqual(second.returncode, 0, second.stderr); self.assertGreaterEqual(json.loads(second.stdout)[0], 110)
            head = json.loads(clock.read_text().splitlines()[-1])['entry_hash']
            regressed = self.invoke(evaluate, clock, ledger, genesis, 'boot-a', 199)
            self.assertNotEqual(regressed.returncode, 0)
            rebooted = self.invoke(evaluate, clock, ledger, genesis, 'boot-b', 250)
            self.assertNotEqual(rebooted.returncode, 0)
            state['committed_core_usd'] = 2; ledger.write_text(json.dumps(state))
            lowered = self.invoke(evaluate, clock, ledger, genesis, 'boot-a', 250)
            self.assertNotEqual(lowered.returncode, 0)
            clock.write_bytes(clock.read_bytes().splitlines(keepends=True)[0])
            check = prefix + "c._boot_id=lambda:'boot-a'; c._now=lambda:250.; c.check(sys.argv[1],sys.argv[2],expected_head=sys.argv[3])"
            rollback = self.invoke(check, clock, genesis, head)
            self.assertNotEqual(rollback.returncode, 0)

    def test_live_budget_gate_elapsed_does_not_reset_between_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); clock = root / 'clock'; ledger = root / 'ledger.json'; quote = root / 'quote.json'
            quote.write_text('{}')
            ledger.write_text(json.dumps({'schema':'vast-campaign-ledger-v1','root_authenticated':True,
                'as_of_utc':datetime.now(timezone.utc).isoformat(),'current_lease_elapsed_seconds':15,
                'committed_core_usd':1,'committed_final_usd':0}))
            prefix = "import json,sys; from pathlib import Path; from vast_training_v1.runner import lease_clock as c; c._boot_id=lambda:'boot-a'; "
            init = self.invoke(prefix + "c._now=lambda:100.; print(c.create(sys.argv[1],json.loads(Path(sys.argv[2]).read_text())))", clock, ledger)
            self.assertEqual(init.returncode, 0, init.stderr); genesis = init.stdout.strip()
            script = prefix + """
from vast_training_v1.runner import launcher
c._now=lambda:float(sys.argv[5])
class Probe:
    last_decision_json={}
    def __init__(self,q,supplier,seconds): self.supplier=supplier
    def before_update(self,**kw): print(self.supplier()['current_lease_elapsed_seconds']); return True
launcher.BudgetGate=Probe
g=launcher.LiveBudgetGate(Path(sys.argv[3]),Path(sys.argv[2]),1,shared_lease_clock=Path(sys.argv[1]),lease_clock_genesis=sys.argv[4])
g.before_update(remaining_updates=1)
"""
            first = self.invoke(script, clock, ledger, quote, genesis, 160)
            second = self.invoke(script, clock, ledger, quote, genesis, 220)
            self.assertEqual(first.returncode, 0, first.stderr); self.assertEqual(second.returncode, 0, second.stderr)
            self.assertGreaterEqual(float(first.stdout.strip()), 75)
            self.assertGreaterEqual(float(second.stdout.strip()), 135)

    def test_wrong_genesis_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); clock = root / 'clock'; ledger = root / 'ledger.json'
            ledger.write_text(json.dumps({'schema':'vast-campaign-ledger-v1','root_authenticated':True,
                'as_of_utc':datetime.now(timezone.utc).isoformat(),'current_lease_elapsed_seconds':0,
                'committed_core_usd':0,'committed_final_usd':0}))
            prefix = "import json,sys; from pathlib import Path; from vast_training_v1.runner import lease_clock as c; c._boot_id=lambda:'boot-a'; c._now=lambda:100.; "
            result = self.invoke(prefix + "print(c.create(sys.argv[1],json.loads(Path(sys.argv[2]).read_text())))", clock, ledger)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.invoke(prefix + "c.check(sys.argv[1],'0'*64)", clock)
            self.assertNotEqual(result.returncode, 0)

if __name__ == '__main__': unittest.main()

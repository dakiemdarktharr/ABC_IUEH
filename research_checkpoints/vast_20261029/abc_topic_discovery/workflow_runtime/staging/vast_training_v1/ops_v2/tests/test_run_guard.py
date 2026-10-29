import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


RUN_GUARD = Path(__file__).resolve().parents[1] / "run_guard.py"


@unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "Linux process-group regression")
class RunGuardProcessGroupTests(unittest.TestCase):
    def test_external_sigterm_contains_group_and_reports_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ready = root / "ready"
            audit = root / "watchdog.jsonl"
            code = f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
            proc = subprocess.Popen([sys.executable, str(RUN_GUARD), "--max-seconds", "30",
                "--interrupt-grace-seconds", "1", "--terminate-grace-seconds", "1",
                "--kill-verify-seconds", "2", "--audit-new", str(audit), "--",
                sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline and proc.poll() is None:
                    time.sleep(0.05)
                self.assertTrue(ready.exists(), "child failed to start")
                proc.send_signal(signal.SIGTERM)
                out, err = proc.communicate(timeout=10)
                self.assertEqual(proc.returncode, 128 + signal.SIGTERM, err + out)
                record = json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])
                self.assertEqual(record["external_stop_signal"], "SIGTERM")
                self.assertTrue(record["same_process_group_empty_verified"])
                self.assertFalse(record["detached_sessions_outside_process_group_inspected"])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    def test_sigint_exited_leader_does_not_skip_child_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child_pid_file = root / "child.pid"
            audit = root / "watchdog.jsonl"
            child_code = (
                "import pathlib,signal,time,os; "
                "signal.signal(signal.SIGINT, signal.SIG_IGN); "
                f"pathlib.Path({str(child_pid_file)!r}).write_text(str(os.getpid())); "
                "time.sleep(30)"
            )
            leader_code = (
                "import signal,subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
                "time.sleep(30)"
            )
            proc = subprocess.run(
                [
                    sys.executable, str(RUN_GUARD),
                    "--max-seconds", "1",
                    "--interrupt-grace-seconds", "0.2",
                    "--terminate-grace-seconds", "1",
                    "--kill-verify-seconds", "2",
                    "--audit-new", str(audit),
                    "--", sys.executable, "-c", leader_code,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(proc.returncode, 124, proc.stderr + proc.stdout)
            self.assertTrue(child_pid_file.exists(), proc.stderr + proc.stdout)
            records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            finished = records[-1]
            self.assertTrue(finished["timed_out"])
            self.assertTrue(finished["leader_exited_while_group_members_remained"])
            self.assertIn("SIGTERM", finished["signals_sent"])
            self.assertTrue(finished["same_process_group_empty_verified"])

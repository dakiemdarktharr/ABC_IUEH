import hashlib
import json
import threading
import unittest

from .guard import HISTORY_RECEIPT, TimeBudgetGate, TimeBudgetStop, verify_ledger


def rows(*events):
    previous = None
    output = []
    for event in events:
        row = {**event, "prev_hash": previous}
        previous = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        output.append(json.dumps({**row, "entry_hash": previous}))
    return ("\n".join(output) + "\n").encode(), previous


HISTORY = {"resource_kind": "cuda_non_export", "seconds": 150.0, "receipt": HISTORY_RECEIPT}


def reserve(seconds=20.0, ident="synthetic"):
    return {"resource_kind": "cuda_non_export", "seconds": seconds,
            "receipt": {"kind": "q1_time_reservation_v1", "reservation_id": ident}}


class Authority:
    def __init__(self, ok=True): self.ok = ok
    def authorize_time_reservation(self, **kwargs): return self.ok


class TimeBudgetTests(unittest.TestCase):
    def test_verified_history_and_reservation_count_once(self):
        raw, head = rows(HISTORY, reserve())
        snap = verify_ledger(raw, expected_head=head, reservation_id="synthetic")
        self.assertEqual(snap.nonexport_seconds, 170)
        self.assertEqual(snap.reservation_seconds, 20)
        self.assertEqual(snap.rows, 2)

    def test_wrong_anchor_tamper_history_and_duplicate_fail(self):
        raw, head = rows(HISTORY, reserve())
        with self.assertRaises(ValueError): verify_ledger(raw, expected_head="0" * 64)
        with self.assertRaises(ValueError): verify_ledger(raw.replace(b'150.0', b'151.0'), expected_head=head)
        empty_history, empty_head = rows(reserve())
        with self.assertRaises(ValueError): verify_ledger(empty_history, expected_head=empty_head)
        repeated, repeated_head = rows(HISTORY, HISTORY, reserve())
        with self.assertRaises(ValueError): verify_ledger(repeated, expected_head=repeated_head)
        duplicated, dup_head = rows(HISTORY, reserve(), reserve())
        with self.assertRaises(ValueError): verify_ledger(duplicated, expected_head=dup_head, reservation_id="synthetic")

    def test_caps_and_export_count(self):
        raw, head = rows(HISTORY, reserve(12451))
        with self.assertRaises(ValueError): verify_ledger(raw, expected_head=head, reservation_id="synthetic")
        raw, head = rows(HISTORY, {"resource_kind": "cuda_export", "seconds": 14251}, reserve(1))
        with self.assertRaises(ValueError): verify_ledger(raw, expected_head=head, reservation_id="synthetic")

    def test_before_update_stops_before_shutdown_allowance(self):
        raw, head = rows(HISTORY, reserve())
        clock = [100.0]
        calls = []
        gate = TimeBudgetGate(ledger_bytes=raw, anchored_head=head, reservation_id="synthetic",
            authority=Authority(), worst_update_seconds=2, shutdown_seconds=5,
            request_stop=lambda: calls.append(1), clock=lambda: clock[0])
        with self.assertRaises(TimeBudgetStop): gate.before_update(remaining_updates=1)
        gate.start_watchdog()
        self.assertTrue(gate.before_update(remaining_updates=1))
        clock[0] = 113.0
        with self.assertRaises(TimeBudgetStop): gate.before_update(remaining_updates=1)
        self.assertEqual(calls, [], "trainer exception path handles synchronous checkpoint")
        clock[0] = 112.0
        with self.assertRaises(TimeBudgetStop): gate.elapsed()
        gate.close_after_process_tree_exits()

    def test_missing_authority_fails(self):
        raw, head = rows(HISTORY, reserve())
        with self.assertRaises(PermissionError):
            TimeBudgetGate(ledger_bytes=raw, anchored_head=head, reservation_id="synthetic",
                authority=Authority(False), worst_update_seconds=2, shutdown_seconds=5, request_stop=lambda: None)

    def test_watchdog_requests_stop_once_without_updates(self):
        raw, head = rows(HISTORY, reserve(0.2))
        stopped = threading.Event()
        calls = []
        def stop():
            calls.append(1)
            stopped.set()
        gate = TimeBudgetGate(ledger_bytes=raw, anchored_head=head, reservation_id="synthetic",
            authority=Authority(), worst_update_seconds=0.01, shutdown_seconds=0.1, request_stop=stop)
        gate.start_watchdog()
        self.assertTrue(stopped.wait(2))
        gate.request_stop()
        self.assertEqual(calls, [1])
        with self.assertRaises(TimeBudgetStop): gate.before_update(remaining_updates=1)
        gate.close_after_process_tree_exits()


if __name__ == "__main__": unittest.main()

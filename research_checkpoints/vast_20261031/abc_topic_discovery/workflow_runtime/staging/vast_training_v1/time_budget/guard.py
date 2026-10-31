"""Verify the original Q1 ledger and meter a root-reserved process wall time.

No ledger writes or CUDA operations occur here. A trusted root must reserve
the complete process ceiling in the canonical ledger under its ledger lock.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

TOTAL_CAP = 14400.0
NONEXPORT_CAP = 12600.0
HISTORY_RECEIPT = "run_manifests/core_integration_20261028/cuda_test_incident.json"
MAX_LEDGER_BYTES = 4 * 1024 * 1024


class TimeBudgetStop(RuntimeError):
    """A time control failed closed before the next optimizer update."""


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _constant(value):
    raise ValueError("non-finite JSON number")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _seconds(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


@dataclass(frozen=True)
class Snapshot:
    file_sha256: str
    head_hash: str
    rows: int
    nonexport_seconds: float
    export_seconds: float
    reservation_seconds: float | None


def verify_ledger(raw: bytes, *, expected_head: str, reservation_id: str | None = None) -> Snapshot:
    """Validate original workflow ledger, its externally anchored head, and debit.

    A matching chain is integrity evidence, not authentication by itself.
    """
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_LEDGER_BYTES:
        raise ValueError("ledger absent or too large")
    lines = raw.decode("utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("empty ledger or blank row")
    previous = None
    nonexport = export = 0.0
    historical = 0
    reservation = None
    for i, line in enumerate(lines):
        row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
        if type(row) is not dict or row.get("prev_hash") != previous:
            raise ValueError(f"ledger chain mismatch at row {i + 1}")
        recorded = row.get("entry_hash")
        unsigned = dict(row)
        unsigned.pop("entry_hash", None)
        if type(recorded) is not str or hashlib.sha256(_canonical(unsigned)).hexdigest() != recorded:
            raise ValueError(f"ledger hash mismatch at row {i + 1}")
        previous = recorded
        kind = row.get("resource_kind")
        if kind not in ("cuda_non_export", "cuda_export"):
            continue
        seconds = _seconds(row.get("seconds"), "ledger debit")
        if kind == "cuda_non_export":
            nonexport += seconds
            if row.get("receipt") == HISTORY_RECEIPT and seconds == 150:
                historical += 1
        else:
            export += seconds
        receipt = row.get("receipt")
        if type(receipt) is dict and receipt.get("kind") == "q1_time_reservation_v1" and receipt.get("reservation_id") == reservation_id:
            if reservation_id is None or reservation is not None or kind != "cuda_non_export":
                raise ValueError("invalid or duplicate reservation")
            reservation = seconds
    if previous != expected_head:
        raise ValueError("ledger head differs from independent root anchor")
    if historical != 1:
        raise ValueError("unique original 150-second historical debit missing")
    if nonexport > NONEXPORT_CAP or nonexport + export > TOTAL_CAP:
        raise ValueError("Q1 CUDA ledger already exceeds cap")
    if reservation_id is not None and (not isinstance(reservation_id, str) or not reservation_id.strip() or reservation is None or reservation <= 0):
        raise ValueError("pre-debited reservation missing")
    return Snapshot(hashlib.sha256(raw).hexdigest(), previous, len(lines), nonexport, export, reservation)


class TimeBudgetGate:
    """Per-process wall ceiling; instantiate before the CUDA process is launched.

    `authority` is a trusted root capability with `authorize_time_reservation`;
    it must verify exclusive reservation ownership and ledger locking. The stop
    callback must request checkpoint then stop via an independent supervisor.
    """
    def __init__(self, *, ledger_bytes: bytes, anchored_head: str, reservation_id: str,
                 authority, worst_update_seconds: float, shutdown_seconds: float,
                 request_stop: Callable[[], None], clock: Callable[[], float] = time.monotonic):
        snap = verify_ledger(ledger_bytes, expected_head=anchored_head, reservation_id=reservation_id)
        self.reserved = snap.reservation_seconds
        self.worst_update = _seconds(worst_update_seconds, "worst update", positive=True)
        self.shutdown = _seconds(shutdown_seconds, "shutdown allowance", positive=True)
        if self.worst_update + self.shutdown >= self.reserved:
            raise ValueError("reservation cannot fit one update plus shutdown")
        if not callable(request_stop) or not callable(clock):
            raise ValueError("stop callback and monotonic clock required")
        if authority.authorize_time_reservation(reservation_id=reservation_id, ledger_sha256=snap.file_sha256,
                                                head_sha256=snap.head_hash, seconds=self.reserved) is not True:
            raise PermissionError("root reservation ownership not authorized")
        self._authority = authority
        self._clock, self._stop = clock, request_stop
        self._started = _seconds(clock(), "monotonic clock")
        self._last_elapsed = 0.0
        self._lock = threading.Lock()
        self._stopped = False
        self._closed = False
        self._timer = None

    def elapsed(self):
        now = _seconds(self._clock(), "monotonic clock")
        elapsed = now - self._started
        with self._lock:
            if elapsed < self._last_elapsed:
                raise TimeBudgetStop("monotonic clock regressed")
            self._last_elapsed = elapsed
        return elapsed

    def before_update(self, *, remaining_updates: int) -> bool:
        if type(remaining_updates) is not int or remaining_updates < 1 or self._closed or self._stopped or self._timer is None:
            raise TimeBudgetStop("inactive time gate or invalid remaining updates")
        if self.elapsed() + self.worst_update + self.shutdown >= self.reserved:
            # The trainer's exception path checkpoints synchronously. Avoid
            # interrupting it before that handler starts; the watchdog still
            # requests stop at the shutdown boundary if it stalls.
            raise TimeBudgetStop("CUDA wall reservation requires checkpoint and stop")
        return True

    def request_stop(self):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop()

    def start_watchdog(self):
        """Request stop at the shutdown boundary, including when an update stalls."""
        with self._lock:
            if self._timer is not None or self._closed:
                raise TimeBudgetStop("watchdog already started or closed")
            remaining = self.reserved - self.shutdown - (self._clock() - self._started)
            if not math.isfinite(remaining) or remaining <= 0:
                raise TimeBudgetStop("watchdog start is already late")
            self._timer = threading.Timer(remaining, self.request_stop)
            self._timer.daemon = True
            self._timer.start()

    def close_after_process_tree_exits(self):
        """Root calls only after the supervised CUDA process tree is gone."""
        with self._lock:
            if self._closed:
                raise TimeBudgetStop("time gate already closed")
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()

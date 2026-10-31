"""Cooperative bootstrap for a separately stored append-only anchor.

No process or GPU is launched. A protected independent anchor location must be
provided by the operator; filesystem separation is not proof of authentication.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from .guard import _canonical, _object, _constant, NONEXPORT_CAP, TOTAL_CAP
from .ledger import ReservationAuthority, _acquire, _unlock, _write_all


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _row(previous, sha, head):
    row = {'schema': 'q1-anchor-v1', 'prev_hash': previous,
           'ledger_sha256': sha, 'ledger_head': head}
    row['entry_hash'] = _hash(_canonical(row))
    return _canonical(row) + b'\n'


def _read(path):
    raw = Path(path).read_bytes()
    if not raw or len(raw) > 4 * 1024 * 1024 or not raw.endswith(b'\n'):
        raise ValueError('missing, oversized or incomplete anchor')
    previous = None
    row = None
    for line in raw.splitlines():
        row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
        if type(row) is not dict or set(row) != {'schema', 'prev_hash', 'ledger_sha256', 'ledger_head', 'entry_hash'}:
            raise ValueError('invalid anchor row')
        if row['schema'] != 'q1-anchor-v1' or row['prev_hash'] != previous:
            raise ValueError('anchor chain mismatch')
        for key in ('entry_hash', 'ledger_sha256', 'ledger_head'):
            value = row[key]
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid anchor digest')
        unsigned = dict(row); del unsigned['entry_hash']
        if _hash(_canonical(unsigned)) != row['entry_hash']:
            raise ValueError('anchor digest mismatch')
        previous = row['entry_hash']
    return raw, row


def initialize_anchor(path, snapshot):
    """Create once using a separately verified initial ledger snapshot."""
    path = Path(path)
    lock = _acquire(str(path) + '.lock')
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb', buffering=0) as stream:
            _write_all(stream, _row(None, snapshot.file_sha256, snapshot.head_hash))
            os.fsync(stream.fileno())
        if os.name != 'nt':
            fd = os.open(str(path.parent), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        _unlock(lock); lock.close()


def check_capacity(ledger_path, original_path, anchor_path, seconds):
    """Verify remaining campaign capacity under both locks, without a debit."""
    if type(seconds) is not int or seconds <= 0:
        raise ValueError('campaign time estimate must be positive whole seconds')
    lock = _acquire(str(anchor_path) + '.lock')
    authority = None
    try:
        _, last = _read(anchor_path)
        authority = ReservationAuthority(ledger_path, original_path,
            expected_sha256=last['ledger_sha256'], expected_head=last['ledger_head'])
        snap = authority.snapshot
        if (snap.nonexport_seconds + seconds > NONEXPORT_CAP
                or snap.nonexport_seconds + snap.export_seconds + seconds > TOTAL_CAP):
            raise ValueError('remaining full campaign exceeds the reconciled time ledger')
        return snap
    finally:
        if authority is not None: authority.close()
        _unlock(lock); lock.close()


class AnchoredReservation:
    """Append reservation then durable anchor while holding both locks.

    Close only after process-group exit is verified. Anchor append failure leaves
    a permanent debit and stale anchor; no launch or automatic reset is permitted.
    """
    def __init__(self, ledger_path, original_path, anchor_path, reservation_id, seconds, *, started_monotonic=None):
        now = time.monotonic()
        if started_monotonic is not None and (type(started_monotonic) not in (int, float)
                                              or not 0 <= started_monotonic <= now):
            raise ValueError('invalid process start clock')
        self.started_monotonic = now if started_monotonic is None else started_monotonic
        self.seconds = seconds
        self.authority = None
        self._anchor_lock = None
        ledger_path, anchor_path = Path(ledger_path), Path(anchor_path)
        if ledger_path.resolve() == anchor_path.resolve() or ledger_path.parent.resolve() == anchor_path.parent.resolve():
            raise ValueError('anchor must be on a separately configured directory')
        try:
            self._anchor_lock = _acquire(str(anchor_path) + '.lock')
            anchor_raw, last = _read(anchor_path)
            self.authority = ReservationAuthority(ledger_path, original_path,
                expected_sha256=last['ledger_sha256'], expected_head=last['ledger_head'])
            snap = self.authority.reserve(reservation_id, seconds)
            if Path(anchor_path).read_bytes() != anchor_raw:
                raise ValueError('anchor changed while locked; reconcile')
            line = _row(last['entry_hash'], snap.file_sha256, snap.head_hash)
            with anchor_path.open('ab', buffering=0) as stream:
                _write_all(stream, line); os.fsync(stream.fileno())
            actual, appended = _read(anchor_path)
            if actual != anchor_raw + line or appended['ledger_sha256'] != snap.file_sha256:
                raise ValueError('anchor persistence verification failed')
            self.snapshot = snap
            self.ledger_bytes = self.authority.raw
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.authority is not None:
            self.authority.close(); self.authority = None
        if self._anchor_lock is not None:
            _unlock(self._anchor_lock); self._anchor_lock.close(); self._anchor_lock = None

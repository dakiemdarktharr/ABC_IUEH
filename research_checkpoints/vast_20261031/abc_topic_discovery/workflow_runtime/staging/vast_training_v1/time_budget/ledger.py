"""Cooperative, append-only Q1 reservation authority; no CUDA or cloud actions.

The caller must protect the directory, retain each returned anchor independently,
and supervise the entire process tree. This is not an authenticated service.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .guard import _canonical, verify_ledger, NONEXPORT_CAP, TOTAL_CAP

HISTORY_SHA256 = '1a38701d8e884cfa654f7165bef9b67bdbf97ca9a95a06799890aba0d420cb75'
HISTORY_HEAD = '726413b0ef42c3cc01a8f708ee65a6c55ff18cfcd970eb971ad367bf7fc45913'
_ID = re.compile(r'^[A-Za-z0-9_-]{16,128}$')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_all(stream, raw):
    view = memoryview(raw)
    while view:
        count = stream.write(view)
        if not count:
            raise OSError('incomplete ledger write; reconcile under lock')
        view = view[count:]


def _history(raw):
    if _digest(raw) != HISTORY_SHA256:
        raise ValueError('original history SHA differs from pinned snapshot')
    snap = verify_ledger(raw, expected_head=HISTORY_HEAD)
    if (snap.nonexport_seconds, snap.export_seconds) != (150.0, 0.0):
        raise ValueError('original history CUDA totals differ')
    if not raw.endswith(b'\n'):
        raise ValueError('history must end with newline')
    return snap


def _lock(stream):
    if os.name == 'nt':
        import msvcrt
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(stream):
    if os.name == 'nt':
        import msvcrt
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _acquire(path):
    # Persistent lock inode: never unlink, replace, or use another lock path.
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    stream = os.fdopen(fd, 'r+b', buffering=0)
    try:
        _lock(stream)
        return stream
    except BaseException:
        stream.close()
        raise


def _read_checked(path, history, expected_sha256, expected_head):
    raw = Path(path).read_bytes()
    if not raw.startswith(history) or _digest(raw) != expected_sha256:
        raise ValueError('ledger prefix or independent SHA anchor mismatch')
    suffix = raw[len(history):]
    ids = set()
    if suffix and not suffix.endswith(b'\n'):
        raise ValueError('incomplete append; reconcile without resetting')
    for line in suffix.splitlines():
        # verify_ledger checks the full hash chain; reject unrecognized records
        # so this new authority cannot silently ignore unaccounted operations.
        row = json.loads(line)
        if set(row) != {'entry_hash', 'prev_hash', 'record_type', 'resource_kind', 'seconds', 'receipt'}:
            raise ValueError('unexpected reservation row schema')
        receipt = row['receipt']
        if (row['record_type'] != 'budget_receipt' or row['resource_kind'] not in ('cuda_non_export', 'cuda_export')
            or type(row['seconds']) is not int or row['seconds'] <= 0
            or type(receipt) is not dict or set(receipt) != {'kind', 'reservation_id'}
            or receipt['kind'] != 'q1_time_reservation_v1'
            or type(receipt['reservation_id']) is not str or not _ID.fullmatch(receipt['reservation_id'])
            or receipt['reservation_id'] in ids):
            raise ValueError('invalid or repeated reservation')
        ids.add(receipt['reservation_id'])
    snap = verify_ledger(raw, expected_head=expected_head)
    return raw, snap, ids


def initialize_ledger(path, original_path):
    """Explicit one-time creation; never replaces an existing ledger."""
    path = Path(path)
    history = Path(original_path).read_bytes()
    snap = _history(history)
    lock = _acquire(str(path) + '.lock')
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb', buffering=0) as out:
            _write_all(out, history)
            os.fsync(out.fileno())
        if os.name != 'nt':
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return snap
    finally:
        _unlock(lock)
        lock.close()


class ReservationAuthority:
    """Owns the exclusive cooperative lock until the supervised process exits.

    A reservation is consumed permanently on append, even if launch or anchoring
    fails. Only the instance that made the append may authorize its gate once.
    """
    def __init__(self, path, original_path, *, expected_sha256, expected_head):
        self.path = Path(path)
        self.history = Path(original_path).read_bytes()
        _history(self.history)
        self._lock = _acquire(str(self.path) + '.lock')
        self._issued = None
        self._claimed = False
        try:
            self.raw, self.snapshot, self.ids = _read_checked(
                self.path, self.history, expected_sha256, expected_head)
        except BaseException:
            self.close()
            raise

    def reserve(self, reservation_id, seconds, *, kind='cuda_non_export'):
        if self._lock is None or self._issued is not None:
            raise RuntimeError('authority closed or already issued a reservation')
        if type(reservation_id) is not str or not _ID.fullmatch(reservation_id) or reservation_id in self.ids:
            raise ValueError('reservation ID invalid or already used')
        if type(seconds) is not int or seconds <= 0 or kind not in ('cuda_non_export', 'cuda_export'):
            raise ValueError('reservation requires positive whole seconds and valid kind')
        n = self.snapshot.nonexport_seconds + (seconds if kind == 'cuda_non_export' else 0)
        total = self.snapshot.nonexport_seconds + self.snapshot.export_seconds + seconds
        if n > NONEXPORT_CAP or total > TOTAL_CAP:
            raise ValueError('reservation exceeds CUDA wall cap')
        row = {'prev_hash': self.snapshot.head_hash, 'record_type': 'budget_receipt',
               'resource_kind': kind, 'seconds': seconds,
               'receipt': {'kind': 'q1_time_reservation_v1', 'reservation_id': reservation_id}}
        row['entry_hash'] = _digest(_canonical(row))
        line = _canonical(row) + b'\n'
        # Revalidate immediately before append under the same lock.
        _read_checked(self.path, self.history, self.snapshot.file_sha256, self.snapshot.head_hash)
        with self.path.open('ab', buffering=0) as out:
            _write_all(out, line)
            os.fsync(out.fileno())
        self.raw, self.snapshot, self.ids = _read_checked(
            self.path, self.history, _digest(self.raw + line), row['entry_hash'])
        self._issued = (reservation_id, seconds, kind)
        return self.snapshot

    def authorize_time_reservation(self, *, reservation_id, ledger_sha256, head_sha256, seconds):
        if self._lock is None or self._claimed or self._issued != (reservation_id, seconds, 'cuda_non_export'):
            return False
        try:
            _read_checked(self.path, self.history, ledger_sha256, head_sha256)
        except (OSError, ValueError):
            return False
        if (ledger_sha256, head_sha256) != (self.snapshot.file_sha256, self.snapshot.head_hash):
            return False
        self._claimed = True
        return True

    def close(self):
        if self._lock is not None:
            _unlock(self._lock)
            self._lock.close()
            self._lock = None

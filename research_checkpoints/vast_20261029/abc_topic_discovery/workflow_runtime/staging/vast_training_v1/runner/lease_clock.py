"""Boot-bound shared lease clock; prevents per-fit USD elapsed resets.

The first record is externally pinned by the campaign receipt. All writers use
one persistent lock and hash chain. This is accounting, not provider billing.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from ..time_budget.ledger import _acquire, _unlock, _write_all
from ..time_budget.guard import _canonical, _object, _constant

MAX_BYTES = 4 * 1024 * 1024


def _boot_id():
    value = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()
    return str(uuid.UUID(value))


def _now(): return time.monotonic()


def _ledger(ledger):
    if type(ledger) is not dict or ledger.get('schema') != 'vast-campaign-ledger-v1' or ledger.get('root_authenticated') is not True:
        raise ValueError('invalid operator USD ledger')
    stamp = datetime.fromisoformat(ledger['as_of_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None: raise ValueError('USD ledger timestamp lacks timezone')
    stamp = stamp.astimezone(timezone.utc)
    values = []
    for key in ('current_lease_elapsed_seconds', 'committed_core_usd', 'committed_final_usd'):
        x = ledger[key]
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError('invalid USD ledger value')
        values.append(float(x))
    return stamp, values


def _row(value):
    row = dict(value); row['entry_hash'] = hashlib.sha256(_canonical(value)).hexdigest()
    return row, _canonical(row) + b'\n'


def _load(path, genesis_hash):
    raw = Path(path).read_bytes()
    if not raw or len(raw) > MAX_BYTES or not raw.endswith(b'\n'):
        raise ValueError('shared lease clock absent, too large or truncated')
    previous = None; first = None; last = None; hashes = []
    for line in raw.splitlines():
        row = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
        if type(row) is not dict or row.get('prev_hash') != previous:
            raise ValueError('shared lease clock chain mismatch')
        unsigned = dict(row); recorded = unsigned.pop('entry_hash', None)
        if type(recorded) is not str or hashlib.sha256(_canonical(unsigned)).hexdigest() != recorded:
            raise ValueError('shared lease clock hash mismatch')
        if first is None:
            if set(unsigned) != {'schema','kind','boot_id','monotonic','elapsed','core','final','ledger_at','prev_hash'} or unsigned['kind'] != 'genesis' or unsigned['schema'] != 'abc-shared-lease-clock-v1':
                raise ValueError('invalid lease clock genesis')
            first = row
        else:
            if set(unsigned) != {'schema','kind','monotonic','elapsed','core','final','ledger_at','prev_hash'} or unsigned['kind'] != 'observation' or unsigned['schema'] != 'abc-shared-lease-clock-v1':
                raise ValueError('invalid lease clock observation')
            if (row['core'] < last['core'] or row['final'] < last['final'] or row['ledger_at'] < last['ledger_at']
                    or row['monotonic'] < last['monotonic'] or row['elapsed'] < last['elapsed']):
                raise ValueError('lease clock observation regressed')
        for key in ('monotonic', 'elapsed', 'core', 'final'):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError('invalid lease clock numeric value')
        previous = recorded; hashes.append(recorded); last = row
    if first['entry_hash'] != genesis_hash:
        raise ValueError('shared lease clock genesis differs from campaign anchor')
    if first['boot_id'] != _boot_id():
        raise ValueError('host boot changed; lease accounting requires reconciliation')
    if _now() < last['monotonic']:
        raise ValueError('shared monotonic clock regressed')
    return first, last, hashes


def create(path, ledger):
    """Create once from fresh operator ledger and conservative age allowance."""
    path = Path(path); stamp, (elapsed, core, final) = _ledger(ledger)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if not 0 <= age <= 60: raise ValueError('fresh ledger required to anchor lease clock')
    monotonic = _now()
    first, raw = _row({'schema':'abc-shared-lease-clock-v1','kind':'genesis',
        'boot_id':_boot_id(),'monotonic':monotonic,'elapsed':elapsed + age,
        'core':core,'final':final,'ledger_at':stamp.isoformat(),'prev_hash':None})
    lock = _acquire(str(path) + '.lock')
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb', buffering=0) as stream:
            _write_all(stream, raw); os.fsync(stream.fileno())
        if os.name == 'posix':
            fd = os.open(str(path.parent), os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    finally:
        _unlock(lock); lock.close()
    return first['entry_hash']


def check(path, genesis_hash, *, expected_head=None):
    lock = _acquire(str(path) + '.lock')
    try:
        first, last, hashes = _load(path, genesis_hash)
        if expected_head is not None and last['entry_hash'] != expected_head:
            raise ValueError('shared lease clock head differs from verified receipt')
        return last['entry_hash']
    finally:
        _unlock(lock); lock.close()


def evaluate(path, genesis_hash, ledger, *, mark=False, minimum_monotonic=None):
    """Merge operator debits without reset and return conservative elapsed."""
    path = Path(path); stamp, (elapsed, core, final) = _ledger(ledger)
    lock = _acquire(str(path) + '.lock')
    try:
        first, last, _ = _load(path, genesis_hash)
        now = _now()
        if now < last['monotonic'] or (minimum_monotonic is not None and now < minimum_monotonic):
            raise ValueError('shared monotonic clock regressed')
        at = stamp.isoformat()
        if core < last['core'] or final < last['final'] or at < last['ledger_at']:
            raise ValueError('operator USD ledger regressed from recorded observation')
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        if age < 0: raise ValueError('operator USD ledger is future dated')
        effective = max(elapsed + age, float(last['elapsed']) + (now - float(last['monotonic'])))
        if mark or core != last['core'] or final != last['final'] or at != last['ledger_at']:
            row, raw = _row({'schema':'abc-shared-lease-clock-v1','kind':'observation',
                'monotonic':now,'elapsed':effective,'core':core,'final':final,'ledger_at':at,'prev_hash':last['entry_hash']})
            if path.stat().st_size + len(raw) > MAX_BYTES: raise ValueError('shared lease clock is full')
            with path.open('ab', buffering=0) as stream:
                _write_all(stream, raw); os.fsync(stream.fileno())
        return effective, now
    finally:
        _unlock(lock); lock.close()

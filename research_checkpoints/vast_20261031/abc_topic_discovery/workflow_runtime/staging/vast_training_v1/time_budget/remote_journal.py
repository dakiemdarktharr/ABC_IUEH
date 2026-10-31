"""Content-addressed, read-back-verified off-host time ledger snapshots.

Requires exactly one cooperating writer for the remote prefix. Rclone/Drive
listing is not a linearizable lock or an independent latest-head authority.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from . import bootstrap, ledger
from .guard import _canonical, _object, _constant
from ..ops_v2.remote_backup import RcloneTransport, _remote_prefix

_NAME = re.compile(r'^journal-([0-9a-f]{64})\.json$')
_HEX = re.compile(r'^[0-9a-f]{64}$')
MAX_GENERATIONS = 256


def _sha(raw): return hashlib.sha256(raw).hexdigest()


def _pin(value, *, optional=False):
    if optional and value is None: return value
    if type(value) is not str or not _HEX.fullmatch(value): raise ValueError('invalid independent journal SHA pin')
    return value


class JournalTransport(RcloneTransport):
    def list_files(self, prefix):
        try:
            result = subprocess.run([self.executable, 'lsf', _remote_prefix(prefix), '--files-only', '--format', 'p'],
                check=True, timeout=self.timeout_seconds, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return result.stdout.decode('utf-8').splitlines()
        except (OSError, subprocess.SubprocessError, UnicodeError):
            raise ValueError('journal listing unavailable') from None


def _download(transport, path, target, expected):
    transport.download_to(path, target)
    raw = target.read_bytes()
    if len(raw) > 4 * 1024 * 1024 or _sha(raw) != expected:
        raise ValueError('journal object readback hash or size mismatch')
    return raw


def _read_generation(transport, prefix, digest, history):
    _pin(digest)
    with tempfile.TemporaryDirectory(prefix='abc-time-journal-read-') as tmp:
        root = Path(tmp)
        raw = _download(transport, f'{prefix}/journal-{digest}.json', root/'manifest', digest)
        obj = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
        if (type(obj) is not dict or set(obj) != {'schema','previous','ledger_sha256','ledger_head','anchor_sha256','anchor_head'}
            or obj['schema'] != 'q1-time-journal-v1' or raw != _canonical(obj) + b'\n'):
            raise ValueError('invalid time journal manifest')
        _pin(obj['previous'], optional=True)
        for k in ('ledger_sha256','ledger_head','anchor_sha256','anchor_head'): _pin(obj[k])
        lraw = _download(transport, f"{prefix}/ledger-{obj['ledger_sha256']}.jsonl", root/'ledger', obj['ledger_sha256'])
        araw = _download(transport, f"{prefix}/anchor-{obj['anchor_sha256']}.jsonl", root/'anchor', obj['anchor_sha256'])
        _, snap, _ = ledger._read_checked(root/'ledger', history, obj['ledger_sha256'], obj['ledger_head'])
        _, last = bootstrap._read(root/'anchor')
        if (last['entry_hash'] != obj['anchor_head'] or last['ledger_sha256'] != snap.file_sha256
            or last['ledger_head'] != snap.head_hash): raise ValueError('journal anchor and ledger disagree')
        return obj, lraw, araw


def _all_generations(transport, prefix, history):
    names = transport.list_files(prefix)
    if type(names) is not list: raise ValueError('journal listing malformed')
    pins = set()
    for name in names:
        if type(name) is not str: raise ValueError('journal listing malformed')
        if name.startswith('journal-'):
            match = _NAME.fullmatch(name)
            if not match: raise ValueError('malformed journal object name')
            pins.add(match.group(1))
    if len(pins) > MAX_GENERATIONS: raise ValueError('journal generation limit exceeded')
    entries = {pin: _read_generation(transport, prefix, pin, history) for pin in pins}
    roots = [pin for pin, (row, _, _) in entries.items() if row['previous'] is None]
    if not entries: return entries, None
    if len(roots) != 1: raise ValueError('ambiguous journal roots')
    successors = {}
    for pin, (row, current_l, current_a) in entries.items():
        prev = row['previous']
        if prev is None: continue
        if prev not in entries or prev in successors: raise ValueError('journal missing history or has multiple writers')
        _, old_l, old_a = entries[prev]
        if (not current_l.startswith(old_l) or len(current_l) <= len(old_l)
            or not current_a.startswith(old_a) or len(current_a) <= len(old_a)):
            raise ValueError('journal ledger or anchor was not append-only')
        successors[prev] = pin
    walked = set(); current = roots[0]
    while current is not None:
        if current in walked: raise ValueError('journal cycle')
        walked.add(current); nxt = successors.get(current)
        if nxt is None: tip = current
        current = nxt
    if walked != set(entries): raise ValueError('journal chain is disconnected')
    return entries, tip


def _head(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('independent journal head file unavailable')
    raw = path.read_bytes()
    if not raw or not raw.endswith(b'\n'): raise ValueError('incomplete journal head file')
    previous = None
    for line in raw.splitlines():
        obj = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
        if (type(obj) is not dict or set(obj) != {'schema','journal_sha256','prev_hash','entry_hash'}
                or obj['schema'] != 'q1-time-journal-head-v1' or obj['prev_hash'] != previous):
            raise ValueError('invalid journal head file')
        unsigned = dict(obj); recorded = unsigned.pop('entry_hash')
        if _sha(_canonical(unsigned)) != recorded: raise ValueError('journal head chain mismatch')
        _pin(obj['journal_sha256'], optional=True)
        previous = recorded
    return obj['journal_sha256']


def _head_row(pin, previous):
    row = {'schema':'q1-time-journal-head-v1','journal_sha256':pin,'prev_hash':previous}
    row['entry_hash'] = _sha(_canonical(row))
    return _canonical(row) + b'\n'


def initialize_head(path, *, pin=None):
    """Explicit setup once; caller must independently verify the initial pin."""
    path = Path(path); _pin(pin, optional=True)
    data = _head_row(pin, None)
    fd = os.open(str(path), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb', buffering=0) as stream:
        ledger._write_all(stream,data); os.fsync(stream.fileno())
    _fsync_dir(path.parent)


def _fsync_dir(path):
    if os.name == 'posix':
        fd = os.open(str(path), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def _advance_head(path, old, new):
    path = Path(path)
    if _head(path) != old: raise ValueError('journal head changed while locked')
    last = json.loads(path.read_bytes().splitlines()[-1])
    data = _head_row(new, last['entry_hash'])
    with path.open('ab', buffering=0) as stream:
        ledger._write_all(stream, data); os.fsync(stream.fileno())
    if _head(path) != new: raise ValueError('journal head append verification failed')


def publish(session, anchor_path, head_path, remote_prefix, *, transport=None):
    """Publish while reservation/anchor locks are held, before CUDA fork."""
    if type(session) is not bootstrap.AnchoredReservation or session.authority is None or session._anchor_lock is None:
        raise PermissionError('live reserved and anchored session required')
    prefix = _remote_prefix(remote_prefix)
    transport = transport if transport is not None else JournalTransport()
    lock = ledger._acquire(str(head_path) + '.lock')
    try:
        previous = _head(head_path)
        history = session.authority.history
        entries, tip = _all_generations(transport, prefix, history)
        if tip != previous: raise ValueError('journal latest differs from independent head; reconcile')
        lraw = session.authority.path.read_bytes()
        if lraw != session.ledger_bytes: raise ValueError('local ledger changed after reservation')
        _, snap, _ = ledger._read_checked(session.authority.path, history, session.snapshot.file_sha256, session.snapshot.head_hash)
        araw, last = bootstrap._read(anchor_path)
        if last['ledger_sha256'] != snap.file_sha256 or last['ledger_head'] != snap.head_hash:
            raise ValueError('local anchor differs from reserved ledger')
        if tip is not None:
            _, old_l, old_a = entries[tip]
            if not lraw.startswith(old_l) or len(lraw) <= len(old_l) or not araw.startswith(old_a) or len(araw) <= len(old_a):
                raise ValueError('local state does not extend remote journal')
        row = {'schema':'q1-time-journal-v1','previous':previous, 'ledger_sha256':_sha(lraw),
               'ledger_head':snap.head_hash,'anchor_sha256':_sha(araw),'anchor_head':last['entry_hash']}
        encoded = _canonical(row) + b'\n'; new = _sha(encoded)
        with tempfile.TemporaryDirectory(prefix='abc-time-journal-write-') as tmp:
            root = Path(tmp)
            for name, raw in ((f"ledger-{row['ledger_sha256']}.jsonl",lraw),
                              (f"anchor-{row['anchor_sha256']}.jsonl",araw),
                              (f'journal-{new}.json',encoded)):
                local = root/name; local.write_bytes(raw)
                transport.put_if_absent(local, f'{prefix}/{name}')
                if transport.read_digest(f'{prefix}/{name}') != (_sha(raw),len(raw)):
                    raise ValueError('journal upload readback failed')
        _, tip_after = _all_generations(transport, prefix, history)
        if tip_after != new: raise ValueError('journal publication not visible as unique head')
        _advance_head(head_path, previous, new)
        return new
    finally:
        ledger._unlock(lock); lock.close()


def recover(*, remote_prefix, expected_head, original_path, ledger_path, anchor_path, head_path, transport=None):
    """Restore only an explicitly pinned unique latest chain to new files."""
    prefix = _remote_prefix(remote_prefix); expected_head = _pin(expected_head)
    history = Path(original_path).read_bytes(); ledger._history(history)
    transport = transport if transport is not None else JournalTransport()
    entries, tip = _all_generations(transport, prefix, history)
    if tip != expected_head: raise ValueError('remote latest differs from independent recovery pin')
    _, lraw, araw = entries[tip]
    for path, raw in ((Path(ledger_path),lraw),(Path(anchor_path),araw)):
        fd = os.open(str(path), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,'wb',buffering=0) as stream:
            ledger._write_all(stream,raw); os.fsync(stream.fileno())
        _fsync_dir(path.parent)
    initialize_head(head_path,pin=tip)
    return tip


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    init = commands.add_parser('init-head', help='create a local head once after independent verification')
    init.add_argument('head', type=Path); init.add_argument('--pin')
    restore = commands.add_parser('recover', help='restore a pinned unique latest remote journal to new files')
    restore.add_argument('--remote', required=True); restore.add_argument('--expected-head', required=True)
    restore.add_argument('--original', type=Path, required=True); restore.add_argument('--ledger', type=Path, required=True)
    restore.add_argument('--anchor', type=Path, required=True); restore.add_argument('--head', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'init-head':
        initialize_head(args.head, pin=args.pin)
    else:
        pin = recover(remote_prefix=args.remote, expected_head=args.expected_head, original_path=args.original,
                      ledger_path=args.ledger, anchor_path=args.anchor, head_path=args.head)
        print(json.dumps({'recovered_journal_sha256': pin}, sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())

"""Linux fork supervisor prototype for an already anchored reservation.

The caller must fork before any CUDA, Torch or background threads. This module
controls only the child's original process group, not detached processes.
"""
from __future__ import annotations

import math
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path

from .bootstrap import AnchoredReservation
from .combined import CombinedBudgetGate
from .guard import TimeBudgetGate
from ..ops_v2 import run_guard


class ContainmentFailure(RuntimeError):
    """A group survived escalation; retain the lock and stop the host."""


def _linux_available():
    return os.name == 'posix' and Path('/proc').is_dir()


class _ForkChild:
    def __init__(self, pid):
        self.pid = pid
        self.code = None

    def poll(self):
        if self.code is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.code = os.waitstatus_to_exitcode(status)
        return self.code


def supervise(session, usd_gate, *, worst_update_seconds, shutdown_seconds,
              interrupt_grace, term_grace, kill_verify, worker):
    """Fork one gated worker. Caller keeps session locked until return.

    `worker(gate)` runs only in the child. On a containment failure the caller
    must retain the session lock and stop the host manually. No provider billing
    or detached-session enforcement is claimed. Must run in the main thread.
    """
    if not _linux_available():
        raise RuntimeError('Linux /proc and POSIX process groups are required')
    if threading.active_count() != 1 or 'torch' in sys.modules:
        raise RuntimeError('fork requires a single thread and no imported Torch')
    if type(session) is not AnchoredReservation or session.authority is None:
        raise PermissionError('live anchored reservation required')
    values = (shutdown_seconds, interrupt_grace, term_grace, kill_verify)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid shutdown or grace duration')
    if interrupt_grace + term_grace + kill_verify > shutdown_seconds:
        raise ValueError('all signal grace periods must fit shutdown allowance')
    deadline = session.started_monotonic + session.seconds - shutdown_seconds
    if time.monotonic() >= deadline:
        raise RuntimeError('reservation is too short after bootstrap')
    if not callable(worker):
        raise TypeError('worker callback required')
    # Construct before fork, but start no timer or thread until after fork.
    parent_pid = os.getpid()
    gate = TimeBudgetGate(ledger_bytes=session.ledger_bytes,
        anchored_head=session.snapshot.head_hash,
        reservation_id=session.authority._issued[0], authority=session.authority,
        worst_update_seconds=worst_update_seconds, shutdown_seconds=shutdown_seconds,
        request_stop=lambda: os.kill(parent_pid, signal.SIGUSR1))
    gate._started = session.started_monotonic
    # Validate USD protocol and policy before forking.
    CombinedBudgetGate(gate, usd_gate)
    stop = []
    def request_stop(sig, _frame):
        if not stop:
            stop.append(sig)
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    for sig in saved: signal.signal(sig, request_stop)
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    try:
        pid = os.fork()
    except BaseException:
        for fd in (ready_r, ready_w, go_r, go_w): os.close(fd)
        for sig, old in saved.items(): signal.signal(sig, old)
        raise
    if pid == 0:
        os.close(ready_r); os.close(go_w)
        try:
            os.setsid()
            requested = []
            signal.signal(signal.SIGINT, lambda *_: requested.append(True))
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            os.write(ready_w, b'R'); os.close(ready_w)
            if os.read(go_r, 1) != b'G':
                os._exit(125)
            os.close(go_r)
            gate.start_watchdog()
            combined = CombinedBudgetGate(gate, usd_gate, stop_requested=lambda: bool(requested))
            combined.before_cuda()
            code = worker(combined)
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(code if type(code) is int and 0 <= code <= 255 else 1)
        except BaseException as exc:
            print('supervised worker failed: ' + type(exc).__name__, file=sys.stderr, flush=True)
            os._exit(1)
    os.close(ready_w); os.close(go_r)
    proc = _ForkChild(pid)
    group_ready = False
    try:
        while time.monotonic() < deadline and not stop and proc.poll() is None:
            ready, _, _ = select.select([ready_r], [], [], min(0.05, max(0, deadline - time.monotonic())))
            if ready:
                group_ready = os.read(ready_r, 1) == b'R'
                break
        os.close(ready_r); ready_r = -1
        if not group_ready or stop or time.monotonic() >= deadline:
            os.close(go_w); go_w = -1
            if proc.poll() is None:
                os.kill(pid, signal.SIGKILL)
            while proc.poll() is None: time.sleep(0.01)
            raise RuntimeError('worker failed pre-launch supervisor handshake')
        os.write(go_w, b'G'); os.close(go_w); go_w = -1
        while proc.poll() is None and not stop and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        timed_out = not stop and proc.poll() is None
        signals = []
        orphaned = False
        if stop or timed_out:
            empty, signals, _ = run_guard.contain_group(pid, proc, first_signal=signal.SIGINT,
                first_grace=interrupt_grace, term_grace=term_grace, kill_verify=kill_verify)
        else:
            empty = run_guard.wait_group_empty(pid, proc, 0)
            if not empty:
                orphaned = True
                empty, signals, _ = run_guard.contain_group(pid, proc, first_signal=signal.SIGINT,
                    first_grace=interrupt_grace, term_grace=term_grace, kill_verify=kill_verify)
        if not empty:
            raise ContainmentFailure('process group survived; retain lock and stop host')
        return {'exit_code': proc.poll(), 'timed_out': timed_out, 'signals_sent': signals,
                'external_stop_signal': stop[0] if stop else None, 'orphaned': orphaned,
                'same_process_group_empty_verified': True, 'detached_sessions_checked': False,
                'elapsed_seconds': time.monotonic() - session.started_monotonic}
    except BaseException as error:
        # After the child creates its session, never abandon its process group
        # on an unexpected parent-side error.
        if not isinstance(error, ContainmentFailure) and group_ready and run_guard.process_group_exists(pid):
            empty, _, _ = run_guard.contain_group(pid, proc, first_signal=signal.SIGINT,
                first_grace=interrupt_grace, term_grace=term_grace, kill_verify=kill_verify)
            if not empty:
                raise ContainmentFailure('process group survived; retain lock and stop host')
        raise
    finally:
        if ready_r >= 0: os.close(ready_r)
        if go_w >= 0: os.close(go_w)
        for sig, old in saved.items(): signal.signal(sig, old)

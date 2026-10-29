#!/usr/bin/env python3
"""Linux wall-time watchdog that escalates signals to surviving process groups.

This only controls the command's original POSIX process group. Descendants that
daemonize into a different session/group require a cgroup-backed launcher and
are outside this watchdog's guarantee. It does not control provider billing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def audit_event(path: Path, record: dict[str, object], *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if create else "a"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def process_group_exists(pgid: int) -> bool:
    """Return whether any process remains in the group, even if its leader exited."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def wait_group_empty(pgid: int, proc: subprocess.Popen[bytes], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        proc.poll()  # Reap the leader without making it the group-liveness oracle.
        if not process_group_exists(pgid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def contain_group(
    pgid: int,
    proc: subprocess.Popen[bytes],
    *,
    first_signal: int,
    first_grace: float,
    term_grace: float,
    kill_verify: float,
) -> tuple[bool, list[str], bool]:
    """Signal the whole group and verify it no longer exists before success."""
    sent: list[str] = []
    if signal_group(pgid, first_signal):
        sent.append(signal.Signals(first_signal).name)
    if wait_group_empty(pgid, proc, first_grace):
        return True, sent, False
    leader_exited_while_members_remain = proc.poll() is not None
    if signal_group(pgid, signal.SIGTERM):
        sent.append("SIGTERM")
    if wait_group_empty(pgid, proc, term_grace):
        return True, sent, leader_exited_while_members_remain
    if signal_group(pgid, signal.SIGKILL):
        sent.append("SIGKILL")
    return wait_group_empty(pgid, proc, kill_verify), sent, leader_exited_while_members_remain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, required=True, help="per-run wall ceiling from the approved budget gate")
    parser.add_argument("--interrupt-grace-seconds", type=float, default=45.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=15.0)
    parser.add_argument("--kill-verify-seconds", type=float, default=10.0)
    parser.add_argument("--child-exit-grace-seconds", type=float, default=1.0)
    parser.add_argument("--audit-new", type=Path, required=True, help="new metadata-only audit JSONL path")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command and args after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if os.name != "posix" or not Path("/proc").is_dir():
        parser.error("this process-group watchdog requires Linux /proc and POSIX process groups")
    graces = (args.interrupt_grace_seconds, args.terminate_grace_seconds, args.kill_verify_seconds, args.child_exit_grace_seconds)
    if args.max_seconds <= 0 or any(not math.isfinite(x) or x < 0 for x in graces):
        parser.error("max-seconds must be positive and grace periods cannot be negative")
    if not command or not command[0]:
        parser.error("provide a command after --")
    if args.audit_new.exists():
        parser.error(f"audit path already exists: {args.audit_new}")

    started_at = datetime.now(timezone.utc).isoformat()
    audit_event(args.audit_new, {
        "schema": "vast-run-watchdog-v2",
        "event": "started",
        "started_at_utc": started_at,
        "max_seconds": args.max_seconds,
        "command_and_environment_recorded": False,
    }, create=True)
    started_mono = time.monotonic()
    received_signals: list[int] = []
    def request_stop(signum: int, _frame: object) -> None:
        if not received_signals:
            received_signals.append(signum)
    # Capture external termination rather than leaving the new-session child
    # running when the watchdog itself receives SIGINT or SIGTERM.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        proc = subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        audit_event(args.audit_new, {
            "schema": "vast-run-watchdog-v2",
            "event": "launch_failed",
            "error_type": type(exc).__name__,
        })
        print(f"BLOCKED: could not start command: {exc}", file=sys.stderr)
        return 2

    pgid = proc.pid
    timed_out = False
    orphaned = False
    leader_exited_with_group_survivors = False
    group_empty = False
    signals_sent: list[str] = []
    deadline = started_mono + args.max_seconds
    while proc.poll() is None and not received_signals and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    externally_stopped = bool(received_signals)
    timed_out = not externally_stopped and proc.poll() is None
    if timed_out or externally_stopped:
        group_empty, signals_sent, leader_exited_with_group_survivors = contain_group(
            pgid, proc,
            first_signal=signal.SIGINT,
            first_grace=args.interrupt_grace_seconds,
            term_grace=args.terminate_grace_seconds,
            kill_verify=args.kill_verify_seconds,
        )
    else:
        # A child/worker can outlive a successfully exited leader. Allow a
        # short natural shutdown, then contain and report it as a leak.
        group_empty = wait_group_empty(pgid, proc, args.child_exit_grace_seconds)
        if not group_empty:
            orphaned = True
            group_empty, signals_sent, leader_exited_with_group_survivors = contain_group(
                pgid, proc,
                first_signal=signal.SIGTERM,
                first_grace=args.terminate_grace_seconds,
                term_grace=0.0,
                kill_verify=args.kill_verify_seconds,
            )

    elapsed = time.monotonic() - started_mono
    leader_returncode = proc.poll()
    result = leader_returncode if leader_returncode is not None else 255
    containment_failed = not group_empty
    record = {
        "schema": "vast-run-watchdog-v2",
        "event": "finished",
        "started_at_utc": started_at,
        "elapsed_monotonic_seconds": round(elapsed, 3),
        "max_seconds": args.max_seconds,
        "timed_out": timed_out,
        "external_stop_signal": signal.Signals(received_signals[0]).name if received_signals else None,
        "leader_exit_code": result,
        "orphaned_descendants_detected": orphaned,
        "leader_exited_while_group_members_remained": leader_exited_with_group_survivors,
        "signals_sent": signals_sent,
        "same_process_group_empty_verified": group_empty,
        "detached_sessions_outside_process_group_inspected": False,
        "provider_stopped": False,
        "billing_enforced": False,
        "command_and_environment_recorded": False,
    }
    audit_event(args.audit_new, record)
    print(json.dumps(record, sort_keys=True))
    if containment_failed:
        print("STOP REQUIRED: process group survived SIGKILL verification; stop the provider instance in its control panel.", file=sys.stderr)
        return 125
    if timed_out:
        return 124
    if externally_stopped:
        return 128 + received_signals[0]
    if orphaned:
        return 125
    return result


if __name__ == "__main__":
    raise SystemExit(main())

"""Monitor a local fit-only export; never rent a VM or transfer data.

Polling detects low memory but cannot prevent an instantaneous drop between
samples. The child must also enforce its own preflight and per-segment checks.
Incomplete export directories must be treated as failed, never uploaded.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time

MIN_AVAILABLE = 2 * 1024**3


def available_bytes() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_phys", ctypes.c_ulonglong), ("available_phys", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Windows could not report available physical memory")
        return int(status.available_phys)
    from ..runner.trainer import available_physical_ram_bytes
    return available_physical_ram_bytes()


def monitor(command: list[str], *, read_available=available_bytes,
            popen=subprocess.Popen, interval: float = 0.25) -> int:
    if read_available() < MIN_AVAILABLE:
        raise MemoryError("fit export requires at least 2 GiB available before launch")
    env = dict(os.environ, ABC_FIT_EXPORT_MONITORED="1")
    child = popen(command, env=env)
    try:
        while True:
            result = child.poll()
            if read_available() < MIN_AVAILABLE:
                if result is None:
                    child.kill()
                    child.wait()
                raise MemoryError("fit export stopped because available RAM fell below 2 GiB; discard partial output")
            if result is not None:
                return int(result)
            time.sleep(interval)
    except BaseException:
        if child.poll() is None:
            child.kill()
            child.wait()
        raise


def main() -> int:
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        # argparse exits before the worker checks its monitoring marker or
        # imports the source reader; help is safe even when RAM is low.
        return subprocess.call([sys.executable, "-m", "vast_training_v1.data.export", *sys.argv[1:]])
    try:
        return monitor([sys.executable, "-m", "vast_training_v1.data.export", *sys.argv[1:]])
    except (MemoryError, OSError) as exc:
        print(f"EXPORT STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

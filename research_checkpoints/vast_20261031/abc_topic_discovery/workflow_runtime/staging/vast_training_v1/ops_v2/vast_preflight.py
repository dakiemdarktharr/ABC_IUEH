#!/usr/bin/env python3
"""Read-only Linux runtime preflight for one RTX 3090 and a pinned Python env."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path


def available_ram_bytes() -> int:
    # Share the trainer's host/cgroup hierarchy check, before numerical imports.
    staging = str(Path(__file__).resolve().parents[2])
    if staging not in sys.path:
        sys.path.insert(0, staging)
    from vast_training_v1.runner.trainer import available_physical_ram_bytes
    return available_physical_ram_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-peak-ram-gib", type=float, required=True, help="measured/bounded RSS for this run's loaded data, plus runtime overhead")
    parser.add_argument("--ram-headroom-gib", type=float, default=2.0)
    parser.add_argument("--min-free-vram-gib", type=float, required=True)
    args = parser.parse_args(argv)
    if platform.system() != "Linux":
        parser.error("run this on the target Linux instance")
    if sys.version_info[:2] != (3, 11):
        parser.error(f"Python 3.11 is pinned for this package; found {platform.python_version()}")
    if (not all(math.isfinite(value) for value in (args.expected_peak_ram_gib, args.ram_headroom_gib, args.min_free_vram_gib))
            or args.expected_peak_ram_gib <= 0 or args.ram_headroom_gib < 2 or args.min_free_vram_gib <= 0):
        parser.error("expected RAM must be positive, RAM headroom at least 2 GiB, and minimum VRAM positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        parser.error("set CUDA_VISIBLE_DEVICES to exactly one device, typically 0, before preflight")
    visible = [item.strip() for item in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if item.strip()]
    if len(visible) != 1:
        parser.error("CUDA_VISIBLE_DEVICES must name exactly one GPU")

    try:
        free_ram = available_ram_bytes()
    except Exception as exc:
        print(f"BLOCKED: cannot verify available RAM: {exc}", file=sys.stderr)
        return 2
    required_ram = (args.expected_peak_ram_gib + args.ram_headroom_gib) * (1024 ** 3)
    if free_ram < required_ram:
        print(f"BLOCKED: available RAM {free_ram / 1024**3:.2f} GiB < required {required_ram / 1024**3:.2f} GiB", file=sys.stderr)
        return 3

    try:
        import torch
    except Exception as exc:  # import diagnostics are useful on remote hosts
        print(f"BLOCKED: PyTorch import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if torch.__version__ != "2.9.1+cu128":
        print(f"BLOCKED: expected torch 2.9.1+cu128; found {torch.__version__}", file=sys.stderr)
        return 2
    if torch.version.cuda != "12.8":
        print(f"BLOCKED: expected CUDA runtime 12.8; found {torch.version.cuda}", file=sys.stderr)
        return 2
    try:
        import numpy
        import scipy
        import h5py
    except Exception as exc:
        print(f"BLOCKED: project dependency import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    expected_versions = {
        "numpy": (numpy.__version__, "2.4.6"),
        "scipy": (scipy.__version__, "1.17.1"),
        "h5py": (h5py.__version__, "3.16.0"),
    }
    mismatched = [f"{name}={actual} (expected {expected})" for name, (actual, expected) in expected_versions.items() if actual != expected]
    if mismatched:
        print("BLOCKED: project dependency pin mismatch: " + ", ".join(mismatched), file=sys.stderr)
        return 2
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        print("BLOCKED: expected exactly one visible CUDA GPU", file=sys.stderr)
        return 2
    props = torch.cuda.get_device_properties(0)
    if "3090" not in props.name or (props.major, props.minor) != (8, 6):
        print(f"BLOCKED: expected an RTX 3090 (compute capability 8.6); found {props.name} {(props.major, props.minor)}", file=sys.stderr)
        return 2
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    free_ram = available_ram_bytes()
    required_ram = (args.expected_peak_ram_gib + args.ram_headroom_gib) * (1024 ** 3)
    if free_ram < required_ram:
        print(f"BLOCKED: available RAM {free_ram / 1024**3:.2f} GiB < required {required_ram / 1024**3:.2f} GiB", file=sys.stderr)
        return 3
    if free_vram < args.min_free_vram_gib * (1024 ** 3):
        print(f"BLOCKED: free VRAM {free_vram / 1024**3:.2f} GiB < required {args.min_free_vram_gib:.2f} GiB", file=sys.stderr)
        return 3

    report = {
        "schema": "vast-runtime-preflight-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "free_vram_gib": round(free_vram / (1024 ** 3), 3),
        "total_vram_gib": round(total_vram / (1024 ** 3), 3),
        "available_ram_gib": round(free_ram / (1024 ** 3), 3),
        "required_ram_gib": round(required_ram / (1024 ** 3), 3),
        "dataset_accessed": False,
        "cuda_smoke_run": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

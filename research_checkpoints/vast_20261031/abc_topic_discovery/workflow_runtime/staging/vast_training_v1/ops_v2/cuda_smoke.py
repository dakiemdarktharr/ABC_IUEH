#!/usr/bin/env python3
"""Synthetic, no-dataset PyTorch/CUDA and checkpoint round-trip smoke check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-dir", type=Path, required=True, help="user-selected persistent/mounted directory to test write/read access")
    args = parser.parse_args()
    import torch

    if torch.__version__ != "2.9.1+cu128" or torch.version.cuda != "12.8" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("BLOCKED: expected pinned PyTorch with exactly one visible CUDA device")
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    if "3090" not in props.name or (props.major, props.minor) != (8, 6):
        raise SystemExit(f"BLOCKED: expected RTX 3090 CC 8.6, found {props.name} {(props.major, props.minor)}")

    # Synthetic arithmetic and a tiny one-step fit; no challenge or user data.
    model = torch.nn.Linear(4, 2, bias=True).to(dev)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.arange(32, dtype=torch.float32, device=dev).reshape(8, 4) / 31.0
    y = torch.zeros((8, 2), dtype=torch.float32, device=dev)
    optim.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optim.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise SystemExit("BLOCKED: synthetic CUDA loss is not finite")

    args.persistent_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(args.persistent_dir, os.W_OK | os.R_OK):
        raise SystemExit("BLOCKED: persistent destination is not readable/writable by this user")
    # Save only generated toy weights to the selected persistent mount and
    # reload them there. The receipt is unique, contains no user data, and is
    # left in place so the operator can inspect mount availability.
    smoke_dir = args.persistent_dir / f"abc-vast-smoke-{os.getpid()}"
    smoke_dir.mkdir(mode=0o700)
    path = smoke_dir / "toy.pt"
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(cpu_state, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    restored = torch.load(path, map_location=dev, weights_only=True)
    for key, value in model.state_dict().items():
        if not torch.equal(restored[key], value):
            raise SystemExit(f"BLOCKED: persistent synthetic checkpoint round trip failed at {key}")
    after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise SystemExit("BLOCKED: synthetic checkpoint bytes changed during readback")
    marker = smoke_dir / "receipt.json"
    payload = json.dumps({
        "schema": "abc-vast-storage-smoke-v1",
        "torch": torch.__version__,
        "gpu": props.name,
        "toy_checkpoint_sha256": before_hash,
        "toy_checkpoint_round_trip": True,
    }) + "\n"
    with marker.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if marker.read_text(encoding="utf-8") != payload:
        raise SystemExit("BLOCKED: persistent destination receipt did not read back exactly")

    torch.cuda.empty_cache()
    print(json.dumps({
        "schema": "abc-vast-cuda-smoke-v1",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "synthetic_step_loss": float(loss.detach().cpu()),
        "toy_checkpoint_round_trip": True,
        "toy_checkpoint_sha256": before_hash,
        "toy_checkpoint_path": str(path),
        "persistent_mount_marker": str(marker),
        "dataset_accessed": False,
        "training_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

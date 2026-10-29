"""Operator-invoked local fit-only export. Does not upload or rent a VM.

Run through ``python -m vast_training_v1.data.export_guard`` from staging. This command
reads human source data if invoked: use only when the user has authorized the
export and data-use decision. Keep the new evidence directory private.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .bundle import ALLOWED_PURPOSES, _assert_no_reparse_path, export_authorized_fit10_bundle
from .privacy import require_private_parent
from ..source_adapter_v1.local_authority import create_local_source_root


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--purpose", choices=ALLOWED_PURPOSES, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--evidence-dir", type=Path, required=True, help="new private directory, separate from bundle")
    args = parser.parse_args(argv)
    if os.environ.get("ABC_FIT_EXPORT_MONITORED") != "1":
        parser.error("run this export through python -m vast_training_v1.data.export_guard")
    output = _assert_no_reparse_path(args.bundle_dir, include_leaf=False)
    evidence = _assert_no_reparse_path(args.evidence_dir, include_leaf=False)
    if output == evidence or output in evidence.parents or evidence in output.parents:
        parser.error("bundle and evidence directories must be distinct and non-nested")
    if output.exists() or evidence.exists():
        parser.error("bundle and evidence must both be new directories")
    require_private_parent(output.parent)
    require_private_parent(evidence.parent)
    evidence.mkdir(mode=0o700, parents=False)
    root, public_key, key_id = create_local_source_root()
    _write_new(evidence / "public-key.ed25519", public_key)
    _write_new(evidence / "key-id.txt", (key_id + "\n").encode("utf-8"))
    identity = export_authorized_fit10_bundle(output, fold_id=args.fold_id, root=root, purpose=args.purpose)
    records = root.completed_source_evidence()
    _write_new(evidence / "read-evidence.json", (json.dumps({"bundle_identity_sha256": identity,
        "fold_id": args.fold_id, "purpose": args.purpose, "source_evidence": records},
        sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    print(json.dumps({"bundle_identity_sha256": identity, "key_id": key_id,
                      "bundle_dir": str(output), "evidence_dir": str(evidence)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

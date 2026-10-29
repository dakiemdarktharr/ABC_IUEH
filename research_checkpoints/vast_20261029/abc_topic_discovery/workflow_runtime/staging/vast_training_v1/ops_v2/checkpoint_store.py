#!/usr/bin/env python3
"""Integrity-checked checkpoint copy and resume metadata verifier.

The tool treats checkpoint payloads as opaque bytes; it does not deserialize
PyTorch pickle files. It never removes or overwrites a source or destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class CheckpointError(ValueError):
    pass


REQUIRED_META = {
    "schema": "abc-training-checkpoint-v1",
    "run_id": str,
    "plan_sha256": str,
    "data_sha256": str,
    "code_sha256": str,
    "model_sha256": str,
    "fold_id": int,
    "training_seed": int,
    "arm": str,
    "global_step": int,
    "total_steps": int,
    "payload_sha256": str,
    "payload_size_bytes": int,
    "resume_state_complete": bool,
    "state_components": dict,
}
STATE_KEYS = ("model", "optimizer", "sampler", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> tuple[str, int]:
    return _hash_stable(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(_read_stable_bytes(path).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise CheckpointError(f"{path} must contain a JSON object")
    return obj


def _load_json_with_digest(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        raw = _read_stable_bytes(path)
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise CheckpointError(f"{path} must contain a JSON object")
    return obj, hashlib.sha256(raw).hexdigest(), len(raw)


def validate_meta(meta: dict[str, Any]) -> None:
    for key, expected_type in REQUIRED_META.items():
        if key not in meta:
            raise CheckpointError(f"checkpoint metadata is missing {key}")
        if key == "schema":
            if meta[key] != expected_type:
                raise CheckpointError(f"unsupported checkpoint metadata schema: {meta[key]!r}")
        elif isinstance(meta[key], bool) and expected_type is int:
            raise CheckpointError(f"{key} must be an integer")
        elif not isinstance(meta[key], expected_type):
            raise CheckpointError(f"{key} has the wrong type")
    for key in ("plan_sha256", "data_sha256", "code_sha256", "model_sha256"):
        if not SHA_RE.fullmatch(meta[key]):
            raise CheckpointError(f"{key} must be a lowercase SHA-256 hex digest")
    if not SHA_RE.fullmatch(meta["payload_sha256"]):
        raise CheckpointError("payload_sha256 must be a lowercase SHA-256 hex digest")
    if meta["payload_size_bytes"] < 1:
        raise CheckpointError("payload_size_bytes must be positive")
    if not meta["run_id"].strip() or not meta["arm"].strip():
        raise CheckpointError("run_id and arm must be non-empty")
    if meta["fold_id"] < 0 or meta["global_step"] < 1 or meta["total_steps"] < meta["global_step"]:
        raise CheckpointError("fold/step counters are inconsistent")
    if meta["resume_state_complete"] is not True:
        raise CheckpointError("checkpoint is not declared as a complete resume state")
    state = meta["state_components"]
    if any(state.get(name) is not True for name in STATE_KEYS):
        missing = [name for name in STATE_KEYS if state.get(name) is not True]
        raise CheckpointError(f"resume state components are incomplete: {', '.join(missing)}")


def _signature(info: os.stat_result, path: Path) -> tuple[int, int, int, int, int]:
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError(f"not a regular file: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode


@contextmanager
def _open_nofollow(path: Path) -> Iterator[tuple[Any, tuple[int, int, int, int, int]]]:
    if path.is_symlink():
        raise CheckpointError(f"symlink files are refused: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointError(f"cannot safely open {path}: {exc}") from exc
    stream = os.fdopen(fd, "rb")
    try:
        before = _signature(os.fstat(stream.fileno()), path)
        yield stream, before
    finally:
        stream.close()


def _verify_path_still_same(path: Path, expected: tuple[int, int, int, int, int]) -> None:
    try:
        current = _signature(path.lstat(), path)
    except OSError as exc:
        raise CheckpointError(f"source path disappeared during read: {path}") from exc
    if current != expected:
        raise CheckpointError(f"source path changed during read: {path}")


def _read_stable_bytes(path: Path) -> bytes:
    with _open_nofollow(path) as (stream, before):
        content = stream.read()
        after = _signature(os.fstat(stream.fileno()), path)
    _verify_path_still_same(path, before)
    if after != before or len(content) != before[2]:
        raise CheckpointError(f"file changed while reading: {path}")
    return content


def _hash_stable(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with _open_nofollow(path) as (stream, before):
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = _signature(os.fstat(stream.fileno()), path)
    _verify_path_still_same(path, before)
    if before != after or size != before[2]:
        raise CheckpointError(f"source changed while hashing: {path}")
    return digest.hexdigest(), size


def copy_checkpoint(checkpoint: Path, metadata: Path, destination: Path) -> dict[str, Any]:
    if checkpoint.is_symlink() or metadata.is_symlink():
        raise CheckpointError("symlink inputs are refused; pass the intended regular files directly")
    if checkpoint.parent.resolve(strict=True) != metadata.parent.resolve(strict=True):
        raise CheckpointError("checkpoint metadata must be a sibling of the payload")
    if metadata.name != f"{checkpoint.stem}.meta.json":
        raise CheckpointError("trainer sidecar must be named <checkpoint-stem>.meta.json")
    meta, meta_hash, meta_size = _load_json_with_digest(metadata)
    validate_meta(meta)
    payload_hash, payload_size = _hash_stable(checkpoint)
    source_match = re.fullmatch(r"update-(\d+)-([0-9a-f]{64})\.pt", checkpoint.name)
    if source_match is None:
        raise CheckpointError("trainer payload must use update-<step>-<full-sha256>.pt naming")
    if source_match.group(2) != payload_hash or int(source_match.group(1)) != meta["global_step"]:
        raise CheckpointError("content-addressed source filename does not match payload bytes/metadata step")
    if (meta["payload_sha256"], meta["payload_size_bytes"]) != (payload_hash, payload_size):
        raise CheckpointError("trainer sidecar payload hash/size does not match checkpoint bytes")
    if destination.is_symlink():
        raise CheckpointError("destination directory symlink is refused")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve(strict=True)
    payload_name = f"checkpoint-{payload_hash}.pt"
    meta_name = f"checkpoint-{payload_hash}.meta.json"
    manifest_name = f"checkpoint-{payload_hash}.manifest.json"
    payload_target = destination / payload_name
    meta_target = destination / meta_name
    manifest_target = destination / manifest_name
    manifest = {
        "schema": "abc-checkpoint-copy-v1",
        "payload_file": payload_name,
        "payload_sha256": payload_hash,
        "payload_size_bytes": payload_size,
        "metadata_file": meta_name,
        "metadata_sha256": meta_hash,
        "metadata_size_bytes": meta_size,
    }

    # First publish verified temporary files using exclusive hard links. A
    # collision is accepted only when the existing content has the same hash.
    for src, target, expected_hash, expected_size in (
        (checkpoint, payload_target, payload_hash, payload_size),
        (metadata, meta_target, meta_hash, meta_size),
    ):
        if target.parent.resolve(strict=True) != destination:
            raise CheckpointError(f"destination target resolves outside selected mount: {target}")
        if target.exists():
            if target.is_symlink():
                raise CheckpointError(f"destination symlink is refused: {target}")
            actual_hash, actual_size = _hash_stable(target)
            if (actual_hash, actual_size) != (expected_hash, expected_size):
                raise CheckpointError(f"destination collision with different bytes: {target}")
            continue
        fd, temp_name = tempfile.mkstemp(prefix=".checkpoint-copy-", dir=destination)
        temp = Path(temp_name)
        try:
            with _open_nofollow(src) as (source, source_before), os.fdopen(fd, "wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                source_after = _signature(os.fstat(source.fileno()), src)
                output.flush()
                os.fsync(output.fileno())
            _verify_path_still_same(src, source_before)
            if source_before != source_after:
                raise CheckpointError(f"source changed during copy: {src}")
            got_hash, got_size = _hash_stable(temp)
            if (got_hash, got_size) != (expected_hash, expected_size):
                raise CheckpointError(f"copied bytes failed integrity check for {src}")
            src_hash_after, src_size_after = _hash_stable(src)
            if (src_hash_after, src_size_after) != (expected_hash, expected_size):
                raise CheckpointError(f"source changed during copy: {src}")
            try:
                os.link(temp, target)
                _fsync_directory(destination)
            except FileExistsError:
                if target.is_symlink():
                    raise CheckpointError(f"destination symlink is refused: {target}")
                actual_hash, actual_size = _hash_stable(target)
                if (actual_hash, actual_size) != (expected_hash, expected_size):
                    raise CheckpointError(f"destination collision with different bytes: {target}")
        finally:
            # This only removes this call's private temporary file.
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if manifest_target.exists():
        if manifest_target.is_symlink():
            raise CheckpointError(f"manifest symlink is refused: {manifest_target}")
        if _read_stable_bytes(manifest_target) != encoded:
            raise CheckpointError(f"manifest collision with different contents: {manifest_target}")
    else:
        fd, temp_name = tempfile.mkstemp(prefix=".checkpoint-manifest-", dir=destination)
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temp, manifest_target)
                _fsync_directory(destination)
            except FileExistsError:
                if manifest_target.is_symlink():
                    raise CheckpointError(f"manifest symlink is refused: {manifest_target}")
                if _read_stable_bytes(manifest_target) != encoded:
                    raise CheckpointError(f"manifest collision with different contents: {manifest_target}")
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
    verify_copy(manifest_target)
    return {**manifest, "destination": str(destination.resolve()), "manifest_path": str(manifest_target.resolve()),
            "trainer_resume_argument": str(payload_target.resolve())}


def verify_copy(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_symlink():
        raise CheckpointError("manifest symlink is refused")
    if manifest_path.parent.is_symlink():
        raise CheckpointError("destination directory symlink is refused")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "abc-checkpoint-copy-v1":
        raise CheckpointError("unsupported checkpoint copy manifest")
    for filename_key in ("payload_file", "metadata_file"):
        filename = manifest.get(filename_key)
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CheckpointError(f"invalid {filename_key} in manifest")
    root = manifest_path.parent
    root_real = root.resolve(strict=True)
    payload = root / manifest["payload_file"]
    metadata = root / manifest["metadata_file"]
    if payload.is_symlink() or metadata.is_symlink():
        raise CheckpointError("payload and metadata symlinks are refused")
    if payload.resolve(strict=True).parent != root_real or metadata.resolve(strict=True).parent != root_real:
        raise CheckpointError("payload or metadata resolves outside the selected destination directory")
    payload_hash, payload_size = _hash_stable(payload)
    meta, meta_hash, meta_size = _load_json_with_digest(metadata)
    if (payload_hash, payload_size) != (manifest.get("payload_sha256"), manifest.get("payload_size_bytes")):
        raise CheckpointError("checkpoint payload does not match its manifest")
    if (meta_hash, meta_size) != (manifest.get("metadata_sha256"), manifest.get("metadata_size_bytes")):
        raise CheckpointError("checkpoint metadata does not match its manifest")
    validate_meta(meta)
    if (meta["payload_sha256"], meta["payload_size_bytes"]) != (payload_hash, payload_size):
        raise CheckpointError("trainer sidecar payload hash/size does not match copied payload")
    return manifest


def compare_resume(manifest_path: Path, expected_path: Path) -> dict[str, Any]:
    manifest = verify_copy(manifest_path)
    meta, meta_hash, meta_size = _load_json_with_digest(manifest_path.parent / manifest["metadata_file"])
    if (meta_hash, meta_size) != (manifest["metadata_sha256"], manifest["metadata_size_bytes"]):
        raise CheckpointError("checkpoint metadata changed after verification")
    validate_meta(meta)
    expected = _load_json(expected_path)
    fields = ("run_id", "plan_sha256", "data_sha256", "code_sha256", "model_sha256", "fold_id", "training_seed", "arm")
    mismatch = [key for key in fields if expected.get(key) != meta.get(key)]
    if mismatch:
        raise CheckpointError("resume pin mismatch: " + ", ".join(mismatch))
    expected_step = expected.get("expected_global_step")
    if expected_step is not None and expected_step != meta["global_step"]:
        raise CheckpointError("resume step mismatch")
    payload_path = (manifest_path.parent / manifest["payload_file"]).resolve(strict=True)
    return {
        "manifest": str(manifest_path.resolve()),
        "payload_sha256": manifest["payload_sha256"],
        "metadata": meta,
        "trainer_resume_argument": str(payload_path),
        "trainer_resume_flag": "--resume",
        "root_must_bind_expected_payload_sha256": manifest["payload_sha256"],
    }


def _require_different_filesystems(checkpoint: Path, destination: Path) -> None:
    if checkpoint.stat().st_dev == destination.stat().st_dev:
        raise CheckpointError("checkpoint and backup destination are on the same filesystem device")


def validate_backup_configuration(source: Path, destination: Path, attestation_path: Path) -> dict[str, Any]:
    """Check operator attestation and filesystem separation before a fit."""
    if not destination.is_absolute() or not destination.is_dir() or destination.is_symlink():
        raise CheckpointError("backup destination must be an existing absolute non-symlink directory")
    attestation = _load_json(attestation_path)
    if (attestation.get("schema") != "abc-external-storage-attestation-v1"
            or attestation.get("external_storage_verified_by_operator") is not True
            or not isinstance(attestation.get("storage_identity"), str)
            or not attestation["storage_identity"].strip()
            or not isinstance(attestation.get("evidence_reference"), str)
            or not attestation["evidence_reference"].strip()):
        raise CheckpointError("missing external storage operator attestation or evidence reference")
    if attestation.get("destination") != str(destination.resolve(strict=True)):
        raise CheckpointError("attestation destination does not match resolved backup directory")
    _require_different_filesystems(source, destination)
    return attestation


def backup_checkpoint(checkpoint: Path, metadata: Path, destination: Path, attestation_path: Path) -> dict[str, Any]:
    """Back up after validating operator claims and filesystem separation."""
    attestation = validate_backup_configuration(checkpoint, destination, attestation_path)
    copied = copy_checkpoint(checkpoint, metadata, destination)
    verify_copy(Path(copied["manifest_path"]))
    return {**copied, "external_storage_attested_by_operator": True,
            "external_storage_independently_verified": False,
            "storage_identity": attestation["storage_identity"],
            "evidence_reference": attestation["evidence_reference"]}


def verify_restored_receipt(manifest_path: Path, expected_path: Path, receipt_path: Path) -> dict[str, Any]:
    verified = compare_resume(manifest_path, expected_path)
    checkpoint_meta = verified["metadata"]
    receipt = _load_json(receipt_path)
    required = {
        "schema": "abc-resume-restored-v1",
        "resume_verified": True,
        "run_id": checkpoint_meta["run_id"],
        "checkpoint_sha256": verified["payload_sha256"],
        "plan_sha256": checkpoint_meta["plan_sha256"],
        "loaded_global_step": checkpoint_meta["global_step"],
        "optimizer_restored": True,
        "sampler_restored": True,
        "rng_restored": True,
        "model_restored": True,
    }
    mismatches = [key for key, value in required.items()
                  if type(receipt.get(key)) is not type(value) or receipt.get(key) != value]
    if mismatches:
        raise CheckpointError("restored-process receipt mismatch: " + ", ".join(mismatches))
    return {"restored_process_receipt": str(receipt_path.resolve()), "checkpoint_sha256": verified["payload_sha256"], "loaded_global_step": checkpoint_meta["global_step"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cp = sub.add_parser("copy", help="copy opaque checkpoint bytes to a mounted durable destination")
    cp.add_argument("checkpoint", type=Path)
    cp.add_argument("metadata", type=Path, help="trainer-written JSON resume state inventory")
    cp.add_argument("destination", type=Path, help="operator-selected persistent/mounted destination")
    backup = sub.add_parser("backup", help="fail-closed post-checkpoint external-mount backup hook")
    backup.add_argument("checkpoint", type=Path)
    backup.add_argument("metadata", type=Path)
    backup.add_argument("destination", type=Path)
    backup.add_argument("attestation", type=Path)
    ver = sub.add_parser("verify", help="verify copied payload and metadata hashes")
    ver.add_argument("manifest", type=Path)
    resume = sub.add_parser("resume-check", help="compare copied checkpoint pins with the frozen run pins")
    resume.add_argument("manifest", type=Path)
    resume.add_argument("expected", type=Path)
    restored = sub.add_parser("verify-restored", help="verify trainer-written receipt after process restore")
    restored.add_argument("manifest", type=Path)
    restored.add_argument("expected", type=Path)
    restored.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "copy":
            result = copy_checkpoint(args.checkpoint, args.metadata, args.destination)
        elif args.action == "backup":
            result = backup_checkpoint(args.checkpoint, args.metadata, args.destination, args.attestation)
        elif args.action == "verify":
            result = verify_copy(args.manifest)
        elif args.action == "resume-check":
            result = compare_resume(args.manifest, args.expected)
        else:
            result = verify_restored_receipt(args.manifest, args.expected, args.receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, CheckpointError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Train-only, signed-bundle reader/exporter for the ordered fit10 pilot.

No numerical packages are imported here. The payload is packed little-endian
float32 quaternions in wxyz order, concatenated segment by segment. Window
materialization is lazy and each window is bounded to one segment.
"""

from __future__ import annotations

import bisect
import base64
import binascii
import hashlib
import json
import math
import mmap
import numbers
import os
import re
import stat
import struct
import sys
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

BUNDLE_KIND = "abc_vast_prepared_train_bundle"
SCHEMA_VERSION = 1
STATUS = "PREPARED_TRAIN_ONLY"
PURPOSE = "ordered_motion_exploratory_fit_only_micro_pilot"
SCREEN_PURPOSE = "ordered_motion_three_arm_screen"
ALLOWED_PURPOSES = (PURPOSE, SCREEN_PURPOSE)
ROLE = "outer_fit10"
FOLD_ID = 0
ALLOWED_FOLDS = (0, 1, 2)
CONTEXT_SAMPLES = 400
TARGET_SAMPLES = 20
ROW_WIDTH = 4
ROW_BYTES = ROW_WIDTH * 4
WINDOW_ROWS = CONTEXT_SAMPLES + TARGET_SAMPLES
ORIGIN_FORMULA = "sample_count_minus_419"
DATA_FILE = "segments.f32le"
MANIFEST_FILE = "manifest.json"
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SIGNATURE_RE = re.compile(r"^[A-Za-z0-9_+/=-]{32,2048}$")
_IDENTITY_KEYS = (
    "schema_version",
    "kind",
    "status",
    "purpose",
    "role",
    "fold_id",
    "window",
    "payload",
    "segments",
    "total_origins",
    "source_binding",
)
_MANIFEST_KEYS = set(_IDENTITY_KEYS) | {
    "status",
    "root_attestation",
    "bundle_identity_sha256",
}
_SOURCE_BINDING_KEYS = {
    "source_asset_sha256",
    "split_sha256",
    "fit_role_inventory_sha256",
    "fit_origin_inventory_sha256",
    "source_loader_sha256",
    "source_authorization_sha256",
    "source_read_receipt_sha256",
}
_ATTESTATION_KEYS = {
    "algorithm",
    "key_id",
    "identity_signature",
    "authorization_sha256",
    "issued_utc",
    "content_validation",
}
_ATTESTATION_UNSIGNED_KEYS = _ATTESTATION_KEYS - {"identity_signature"}
_CONTENT_VALIDATION = "finite_unit_wxyz_hemisphere_source_v1"


class BundleError(ValueError):
    """The bundle is malformed, unsafe, unauthorized, or not the fixed pilot."""


class BundleAuthorizationError(PermissionError):
    """A trusted root verifier did not authorize this bundle/session."""


def _require_opaque_capability(value: object, label: str) -> Callable[[], Any]:
    if isinstance(value, (ABCMapping, list, tuple, set, frozenset, str, bytes, int, float, bool)):
        raise BundleAuthorizationError(f"{label} must be an opaque process-local capability")
    assert_live = getattr(value, "assert_live", None)
    if not callable(assert_live):
        raise BundleAuthorizationError(f"{label} must expose assert_live()")
    assert_live()
    return assert_live


@dataclass(frozen=True)
class SegmentInput:
    """One already-authorized, contiguous segment supplied by the root reader.

    `float32_wxyz_le` must expose exactly sample_count * 16 contiguous bytes.
    This is a source adapter contract, not authority to open the source.
    """

    segment_index: int
    sample_count: int
    float32_wxyz_le: Any


class TrustedSourceRoot(Protocol):
    """Process-local root capability injected by an authenticated launcher.

    The implementation must authenticate a fresh, single-fold fit10,
    train-only export grant before opening the V3 source reader. It must use the new pilot
    purpose and must not reuse any consumed V1/V2/V3 source claim.
    """

    def issue_fit10_export_grant(self, fold_id: int, *, purpose: str) -> object: ...

    def verify_fit10_export_grant(self, grant: object) -> Mapping[str, str]: ...

    def open_fit10_segments(self, grant: object) -> Iterable[SegmentInput]: ...

    def finalize_fit10_source_binding(self, grant: object) -> Mapping[str, str]: ...

    def attest_bundle_identity(
        self, grant: object, identity_sha256: str
    ) -> Mapping[str, str]: ...


class TrustedBundleVerifier(Protocol):
    """Injected trust root for signature, source binding, and resume checks.

    `verify_bundle` must authenticate the root signature over
    `attestation_signing_bytes(...)` and verify the current train-only permit
    plus the opaque model/recipe/runner/checkpoint run binding.
    It is called on every open, including resume. It must return a process-local
    opaque object (not a mapping or value reconstructed from manifest JSON)
    with an `assert_live()` method and a `bundle_identity_sha256` attribute.
    """

    def verify_bundle(
        self,
        *,
        canonical_manifest: bytes,
        bundle_identity_sha256: str,
        payload_sha256: str,
        expected_identity_sha256: str | None,
        expected_fold_id: int,
        expected_run_binding: object,
    ) -> object: ...


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise BundleError("non-finite JSON number is forbidden")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BundleError("manifest cannot be represented as canonical JSON") from exc


def _sha256(data: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise BundleError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        raise BundleError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _validate_source_binding(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _SOURCE_BINDING_KEYS:
        raise BundleError("source_binding has an unexpected schema")
    result = {}
    for key in sorted(_SOURCE_BINDING_KEYS):
        result[key] = _require_hash(value[key], f"source_binding.{key}")
    return result


def _validate_pre_read_binding(value: Any) -> dict[str, str]:
    keys = _SOURCE_BINDING_KEYS - {"source_read_receipt_sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise BundleError("pre-read binding must contain only static source authorization hashes")
    return {key: _require_hash(value[key], f"source_binding.{key}") for key in sorted(keys)}


def _validate_attestation(value: Any, source_binding: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _ATTESTATION_KEYS:
        raise BundleError("root_attestation has an unexpected schema")
    if value.get("algorithm") != "Ed25519":
        raise BundleError("root_attestation.algorithm must be Ed25519")
    if not isinstance(value.get("key_id"), str) or not value["key_id"] or len(value["key_id"]) > 128:
        raise BundleError("root_attestation.key_id is malformed")
    if not isinstance(value.get("issued_utc"), str) or len(value["issued_utc"]) > 64:
        raise BundleError("root_attestation.issued_utc is malformed")
    if value.get("content_validation") != _CONTENT_VALIDATION:
        raise BundleError("root attestation does not bind the expected content validation")
    authorization_sha256 = _require_hash(
        value.get("authorization_sha256"), "root_attestation.authorization_sha256"
    )
    if authorization_sha256 != source_binding["source_authorization_sha256"]:
        raise BundleError("root attestation does not match the source authorization")
    signature = value.get("identity_signature")
    if not isinstance(signature, str) or not SIGNATURE_RE.fullmatch(signature):
        raise BundleError("root_attestation.identity_signature is malformed")
    try:
        decoded_signature = base64.b64decode(signature, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise BundleError("root_attestation.identity_signature is not base64") from exc
    if len(decoded_signature) != 64:
        raise BundleError("Ed25519 identity signature must contain 64 bytes")
    return {key: value[key] for key in sorted(_ATTESTATION_KEYS)}


def _manifest_identity(manifest: Mapping[str, Any]) -> str:
    identity = {key: manifest[key] for key in _IDENTITY_KEYS}
    return _sha256(_canonical_json(identity))


def attestation_signing_bytes(
    bundle_identity_sha256: str, attestation: Mapping[str, Any]
) -> bytes:
    """Canonical Ed25519 message shared by the trusted signer and verifier.

    The signature covers the exact bundle identity and all authority metadata,
    including key id, authorization digest, issued time, and validation claim.
    """
    _require_hash(bundle_identity_sha256, "bundle_identity_sha256")
    values = dict(attestation)
    if set(values) == _ATTESTATION_KEYS:
        _validate_attestation(
            values,
            {
                "source_authorization_sha256": _require_hash(
                    values.get("authorization_sha256"),
                    "root_attestation.authorization_sha256",
                )
            },
        )
    elif set(values) == _ATTESTATION_UNSIGNED_KEYS:
        if values.get("algorithm") != "Ed25519":
            raise BundleError("root_attestation.algorithm must be Ed25519")
        if not isinstance(values.get("key_id"), str) or not values["key_id"] or len(values["key_id"]) > 128:
            raise BundleError("root_attestation.key_id is malformed")
        _require_hash(values.get("authorization_sha256"), "root_attestation.authorization_sha256")
        if not isinstance(values.get("issued_utc"), str) or len(values["issued_utc"]) > 64:
            raise BundleError("root_attestation.issued_utc is malformed")
        if values.get("content_validation") != _CONTENT_VALIDATION:
            raise BundleError("root attestation does not bind the expected content validation")
    else:
        raise BundleError("root_attestation has an unexpected schema")
    signed_fields = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": values["algorithm"],
        "key_id": values["key_id"],
        "bundle_identity_sha256": bundle_identity_sha256,
        "authorization_sha256": values["authorization_sha256"],
        "issued_utc": values["issued_utc"],
        "content_validation": values["content_validation"],
    }
    return _canonical_json(signed_fields)


def _decode_manifest(raw: bytes) -> dict[str, Any]:
    if len(raw) > 1024 * 1024:
        raise BundleError("manifest exceeds 1 MiB")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except BundleError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BundleError("manifest root must be an object")
    if set(value) != _MANIFEST_KEYS:
        raise BundleError("manifest has unexpected or missing fields")
    return value


def _validate_manifest(raw: bytes) -> tuple[dict[str, Any], str]:
    manifest = _decode_manifest(raw)
    if _strict_int(manifest["schema_version"], "schema_version", minimum=1) != SCHEMA_VERSION:
        raise BundleError("unsupported schema_version")
    if manifest["kind"] != BUNDLE_KIND or manifest["status"] != STATUS:
        raise BundleError("manifest kind/status is not a prepared train bundle")
    if manifest["purpose"] not in ALLOWED_PURPOSES or manifest["role"] != ROLE:
        raise BundleError("bundle purpose/role is not the authorized train-only pilot")
    fold_id = _strict_int(manifest["fold_id"], "fold_id")
    if fold_id not in ALLOWED_FOLDS:
        raise BundleError("fold_id must identify one of the three frozen train folds")
    window = manifest["window"]
    expected_window_keys = {
        "context_samples",
        "target_samples",
        "row_width",
        "dtype",
        "quaternion_order",
        "same_segment_required",
        "origin_formula",
    }
    if not isinstance(window, dict) or set(window) != expected_window_keys:
        raise BundleError("window descriptor has an unexpected schema")
    for key, expected in (
        ("context_samples", CONTEXT_SAMPLES),
        ("target_samples", TARGET_SAMPLES),
        ("row_width", ROW_WIDTH),
    ):
        if _strict_int(window[key], f"window.{key}") != expected:
            raise BundleError(f"window.{key} does not match the fixed tensor contract")
    if window["same_segment_required"] is not True:
        raise BundleError("window must require a single source segment")
    expected_window = {
        "context_samples": CONTEXT_SAMPLES,
        "target_samples": TARGET_SAMPLES,
        "row_width": ROW_WIDTH,
        "dtype": "<f4",
        "quaternion_order": "wxyz",
        "same_segment_required": True,
        "origin_formula": ORIGIN_FORMULA,
    }
    if window != expected_window:
        raise BundleError("window schema does not match the fixed [400,4]/[20,4] contract")
    payload = manifest["payload"]
    if not isinstance(payload, dict) or set(payload) != {"path", "bytes", "sha256"}:
        raise BundleError("payload descriptor has an unexpected schema")
    if payload["path"] != DATA_FILE:
        raise BundleError("payload path is not the fixed data member")
    payload_bytes = _strict_int(payload["bytes"], "payload.bytes", minimum=ROW_BYTES)
    if payload_bytes > MAX_PAYLOAD_BYTES or payload_bytes % ROW_BYTES:
        raise BundleError("payload length exceeds the cap or is not float32 quaternion aligned")
    _require_hash(payload["sha256"], "payload.sha256")
    segments = manifest["segments"]
    if not isinstance(segments, list) or not segments:
        raise BundleError("segments must be a non-empty list")
    cursor_bytes = 0
    total_origins = 0
    for expected_index, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != {
            "segment_index",
            "offset_bytes",
            "sample_count",
            "origin_count",
            "sha256",
        }:
            raise BundleError("segment descriptor has an unexpected schema")
        if _strict_int(segment["segment_index"], "segment_index") != expected_index:
            raise BundleError("segment indexes must be contiguous and ordered")
        offset = _strict_int(segment["offset_bytes"], "segment.offset_bytes")
        if offset != cursor_bytes:
            raise BundleError("segment payload ranges must be contiguous and non-overlapping")
        sample_count = _strict_int(segment["sample_count"], "segment.sample_count", minimum=WINDOW_ROWS)
        origin_count = _strict_int(segment["origin_count"], "segment.origin_count", minimum=1)
        if origin_count != sample_count - 419:
            raise BundleError("origin count must be N-419 within each segment")
        _require_hash(segment["sha256"], "segment.sha256")
        cursor_bytes += sample_count * ROW_BYTES
        total_origins += origin_count
    if cursor_bytes != payload_bytes:
        raise BundleError("segment lengths do not exactly cover the payload")
    if _strict_int(manifest["total_origins"], "total_origins", minimum=1) != total_origins:
        raise BundleError("total_origins does not match the segment-local N-419 sum")
    source_binding = _validate_source_binding(manifest["source_binding"])
    _validate_attestation(manifest["root_attestation"], source_binding)
    identity_sha256 = _manifest_identity(manifest)
    if _require_hash(manifest["bundle_identity_sha256"], "bundle_identity_sha256") != identity_sha256:
        raise BundleError("bundle identity digest mismatch")
    # Canonical bytes prevent multiple signed interpretations of one document.
    if raw != _canonical_json(manifest) + b"\n":
        raise BundleError("manifest bytes are not canonical JSON with one final newline")
    return manifest, identity_sha256


def _path_has_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BundleError(f"cannot inspect path component: {path.name}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _absolute_components(path: Path) -> tuple[Path, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if not parts:
        raise BundleError("empty path")
    current = Path(parts[0])
    components = [current]
    for part in parts[1:]:
        current = current / part
        components.append(current)
    return tuple(components)


def _assert_no_reparse_path(path: Path, *, include_leaf: bool = True) -> Path:
    components = _absolute_components(path)
    checked = components if include_leaf else components[:-1]
    for component in checked:
        if _path_has_reparse(component):
            raise BundleError("symlink or reparse-point path is forbidden")
    return Path(os.path.abspath(os.fspath(path)))


def _open_plain_file(directory: Path, name: str) -> int:
    if name not in {MANIFEST_FILE, DATA_FILE}:
        raise BundleError("internal member name is not allow-listed")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    dir_fd: int | None = None
    if os.name == "posix" and hasattr(os, "O_DIRECTORY") and os.open in getattr(os, "supports_dir_fd", set()):
        directory_fd = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    else:
        descriptor = os.open(directory / name, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise BundleError("bundle member must be a regular file")
    return descriptor


def _read_fd_limited(descriptor: int, limit: int) -> bytes:
    metadata = os.fstat(descriptor)
    if metadata.st_size < 0 or metadata.st_size > limit:
        raise BundleError("file exceeds its allowed size")
    if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise BundleError("manifest must not be writable while it is being verified")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        block = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
        if not block:
            break
        chunks.append(block)
        total += len(block)
        if total > limit:
            raise BundleError("file grew beyond its allowed size while reading")
    if total != metadata.st_size or os.fstat(descriptor).st_size != metadata.st_size:
        raise BundleError("file changed during read")
    return b"".join(chunks)


def _hash_fd(
    descriptor: int, expected_bytes: int, segment_descriptors: Sequence[Mapping[str, Any]]
) -> str:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
        raise BundleError("payload is not a regular file of the declared exact size")
    if before.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise BundleError("payload must not be writable while it is being verified")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    segment_hashes = [hashlib.sha256() for _ in segment_descriptors]
    segment_ends = [
        item["offset_bytes"] + item["sample_count"] * ROW_BYTES
        for item in segment_descriptors
    ]
    segment_index = 0
    total = 0
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
        block_view = memoryview(block)
        block_start = total
        block_end = total + len(block)
        cursor = block_start
        while cursor < block_end:
            if segment_index >= len(segment_descriptors):
                raise BundleError("payload bytes exceed declared segments")
            segment_end = segment_ends[segment_index]
            chunk_end = min(block_end, segment_end)
            local_start = cursor - block_start
            local_end = chunk_end - block_start
            segment_hashes[segment_index].update(block_view[local_start:local_end])
            cursor = chunk_end
            if cursor == segment_end:
                segment_index += 1
        total += len(block)
        if total > expected_bytes:
            raise BundleError("payload grew while hashing")
    after = os.fstat(descriptor)
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if total != expected_bytes or before_id != after_id or segment_index != len(segment_descriptors):
        raise BundleError("payload changed while hashing")
    for index, descriptor_item in enumerate(segment_descriptors):
        if segment_hashes[index].hexdigest() != descriptor_item["sha256"]:
            raise BundleError(f"segment {index} SHA-256 mismatch")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _buffer_bytes(value: Any, expected_bytes: int) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise BundleError("segment source must expose a contiguous bytes buffer") from exc
    if not view.c_contiguous:
        raise BundleError("segment source must be C-contiguous")
    if view.itemsize != 1 and (view.itemsize != 4 or view.format not in {"f", "<f"}):
        raise BundleError("segment source must be float32 wxyz or raw little-endian bytes")
    if view.itemsize == 4 and sys.byteorder != "little":
        raise BundleError("native float32 source is supported only on little-endian hosts")
    try:
        byte_view = view.cast("B")
    except (TypeError, ValueError) as exc:
        raise BundleError("segment source cannot be viewed as bytes") from exc
    if len(byte_view) != expected_bytes:
        raise BundleError("segment byte length does not match sample_count * 16")
    return byte_view


def _validate_quaternion_segment(byte_view: memoryview, sample_count: int) -> None:
    previous: tuple[float, float, float, float] | None = None
    for values in struct.iter_unpack("<4f", byte_view):
        q = tuple(float(component) for component in values)
        if not all(math.isfinite(component) for component in q):
            raise BundleError("segment contains a non-finite quaternion component")
        norm_sq = sum(component * component for component in q)
        if abs(norm_sq - 1.0) > 2e-4:
            raise BundleError("segment contains a non-unit quaternion")
        if previous is not None and sum(a * b for a, b in zip(previous, q)) < -1e-6:
            raise BundleError("segment has a quaternion hemisphere discontinuity")
        previous = q
    if sample_count < WINDOW_ROWS:
        raise BundleError("segment must contain at least 420 rows")


def _write_exclusive_file(path: Path, data: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, mode) if hasattr(os, "fchmod") else None
    finally:
        os.close(descriptor)


def _write_payload_stream(
    destination: Path,
    segments: Iterable[SegmentInput],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    path = destination / DATA_FILE
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    descriptors: list[dict[str, Any]] = []
    global_digest = hashlib.sha256()
    cursor_bytes = 0
    total_origins = 0
    index = 0
    try:
        for segment in segments:
            if not isinstance(segment, SegmentInput):
                raise BundleError("source stream must yield SegmentInput values")
            if _strict_int(segment.segment_index, "segment_index") != index:
                raise BundleError("source segment indexes must be contiguous and ordered")
            sample_count = _strict_int(segment.sample_count, "sample_count", minimum=WINDOW_ROWS)
            segment_bytes = sample_count * ROW_BYTES
            if cursor_bytes + segment_bytes > MAX_PAYLOAD_BYTES:
                raise BundleError("fit10 bundle exceeds the 128 MiB pose-array cap")
            byte_view = _buffer_bytes(segment.float32_wxyz_le, segment_bytes)
            _validate_quaternion_segment(byte_view, sample_count)
            segment_digest = hashlib.sha256(byte_view).hexdigest()
            offset = 0
            while offset < len(byte_view):
                written = os.write(descriptor, byte_view[offset : offset + 1024 * 1024])
                if written <= 0:
                    raise OSError("short write while creating bundle payload")
                block = byte_view[offset : offset + written]
                global_digest.update(block)
                offset += written
            origins = sample_count - 419
            descriptors.append(
                {
                    "segment_index": index,
                    "offset_bytes": cursor_bytes,
                    "sample_count": sample_count,
                    "origin_count": origins,
                    "sha256": segment_digest,
                }
            )
            cursor_bytes += segment_bytes
            total_origins += origins
            index += 1
        if not descriptors:
            raise BundleError("root source returned no fit10 segments")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400) if hasattr(os, "fchmod") else None
    except Exception:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    if not hasattr(os, "fchmod"):
        os.chmod(path, 0o400)
    payload = {"path": DATA_FILE, "bytes": cursor_bytes, "sha256": global_digest.hexdigest()}
    return descriptors, payload, total_origins


def _make_manifest(
    *,
    fold_id: int,
    segments: list[dict[str, Any]],
    payload: dict[str, Any],
    total_origins: int,
    source_binding: Mapping[str, str],
    root_attestation: Mapping[str, str],
    purpose: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "status": STATUS,
        "purpose": purpose,
        "role": ROLE,
        "fold_id": fold_id,
        "window": {
            "context_samples": CONTEXT_SAMPLES,
            "target_samples": TARGET_SAMPLES,
            "row_width": ROW_WIDTH,
            "dtype": "<f4",
            "quaternion_order": "wxyz",
            "same_segment_required": True,
            "origin_formula": ORIGIN_FORMULA,
        },
        "payload": payload,
        "segments": segments,
        "total_origins": total_origins,
        "source_binding": _validate_source_binding(dict(source_binding)),
        "root_attestation": dict(root_attestation),
    }
    manifest["bundle_identity_sha256"] = _manifest_identity(manifest)
    _validate_attestation(manifest["root_attestation"], manifest["source_binding"])
    return manifest


def _create_bundle(
    destination: Path,
    segments: Iterable[SegmentInput],
    *,
    fold_id: int,
    source_binding: Mapping[str, str] | Callable[[], Mapping[str, str]],
    attest_identity: Callable[[str], Mapping[str, str]],
    purpose: str = PURPOSE,
) -> str:
    if purpose not in ALLOWED_PURPOSES:
        raise BundleError("unsupported export purpose")
    fold_id = _strict_int(fold_id, "fold_id")
    if fold_id not in ALLOWED_FOLDS:
        raise BundleError("fold_id must be 0, 1, or 2")
    destination = _assert_no_reparse_path(destination, include_leaf=False)
    if destination.exists() or destination.is_symlink():
        raise BundleError("bundle destination must be a new directory")
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        descriptors, payload, total_origins = _write_payload_stream(destination, segments)
        source_binding = _validate_source_binding(dict(
            source_binding() if callable(source_binding) else source_binding
        ))
        # Root attestation signs this digest of exact payload bytes and source bindings.
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "status": STATUS,
            "purpose": purpose,
            "role": ROLE,
            "fold_id": fold_id,
            "window": {
                "context_samples": CONTEXT_SAMPLES,
                "target_samples": TARGET_SAMPLES,
                "row_width": ROW_WIDTH,
                "dtype": "<f4",
                "quaternion_order": "wxyz",
                "same_segment_required": True,
                "origin_formula": ORIGIN_FORMULA,
            },
            "payload": payload,
            "segments": descriptors,
            "total_origins": total_origins,
            "source_binding": _validate_source_binding(dict(source_binding)),
        }
        identity_sha256 = _sha256(_canonical_json(unsigned))
        attestation = attest_identity(identity_sha256)
        manifest = _make_manifest(
            fold_id=fold_id,
            segments=descriptors,
            payload=payload,
            total_origins=total_origins,
            source_binding=source_binding,
            root_attestation=attestation,
            purpose=purpose,
        )
        if manifest["bundle_identity_sha256"] != identity_sha256:
            raise BundleError("root identity signing input mismatch")
        _write_exclusive_file(
            destination / MANIFEST_FILE,
            _canonical_json(manifest) + b"\n",
            0o400,
        )
        if hasattr(os, "chmod"):
            os.chmod(destination, 0o500)
        return identity_sha256
    except Exception:
        # Leave a private incomplete directory for explicit inspection/quarantine.
        raise


def export_authorized_fit10_bundle(
    destination: str | os.PathLike[str], *, fold_id: int, root: TrustedSourceRoot,
    purpose: str = PURPOSE,
) -> str:
    """Export one fold's fit10 only after a trusted root grants this purpose.

    No source pathname is accepted. The external root object must authenticate a
    fresh, one-use grant before its `open_fit10_segments` method touches V3/HDF5.
    The grant's source bindings and resulting bundle identity are root-derived.
    """

    fold_id = _strict_int(fold_id, "fold_id")
    if fold_id not in ALLOWED_FOLDS:
        raise BundleError("fold_id must be 0, 1, or 2")
    if purpose not in ALLOWED_PURPOSES:
        raise BundleError("unsupported export purpose")
    grant = root.issue_fit10_export_grant(fold_id, purpose=purpose)
    if grant is None:
        raise BundleAuthorizationError("trusted root did not issue a grant")
    if getattr(grant, "purpose", None) != purpose:
        raise BundleAuthorizationError("root grant has the wrong purpose")
    if getattr(grant, "role", None) != ROLE:
        raise BundleAuthorizationError("root grant has the wrong train role")
    grant_fold_id = _strict_int(getattr(grant, "fold_id", None), "grant.fold_id")
    if grant_fold_id != fold_id:
        raise BundleAuthorizationError("root grant is not bound to the requested fold")
    assert_live = _require_opaque_capability(grant, "root grant")
    pre_read_binding = _validate_pre_read_binding(dict(root.verify_fit10_export_grant(grant)))
    segments = root.open_fit10_segments(grant)
    def checked_segments():
        for segment in segments:
            assert_live()
            yield segment

    def finalized_binding():
        assert_live()
        final = _validate_source_binding(dict(root.finalize_fit10_source_binding(grant)))
        if any(final[key] != value for key, value in pre_read_binding.items()):
            raise BundleAuthorizationError("source binding changed after reading")
        return final

    identity = _create_bundle(
        Path(destination),
        checked_segments(),
        fold_id=fold_id,
        source_binding=finalized_binding,
        attest_identity=lambda identity: root.attest_bundle_identity(grant, identity),
        purpose=purpose,
    )
    assert_live()
    return identity


def _enumerate_bundle_files(directory: Path) -> None:
    try:
        entries = list(os.scandir(directory))
    except OSError as exc:
        raise BundleError("cannot enumerate bundle directory") from exc
    names = {entry.name for entry in entries}
    if names != {MANIFEST_FILE, DATA_FILE}:
        raise BundleError("bundle directory must contain exactly manifest.json and segments.f32le")
    for entry in entries:
        if entry.name not in {MANIFEST_FILE, DATA_FILE} or entry.is_symlink():
            raise BundleError("symlink or unrecognized bundle member is forbidden")
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise BundleError("cannot inspect bundle member") from exc
        attributes = getattr(metadata, "st_file_attributes", 0)
        if not stat.S_ISREG(metadata.st_mode) or attributes & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise BundleError("bundle members must be regular files")


def open_prepared_train_bundle(
    directory: str | os.PathLike[str],
    *,
    trusted_verifier: TrustedBundleVerifier,
    expected_fold_id: int,
    expected_run_binding: object,
    expected_identity_sha256: str | None = None,
) -> "PreparedTrainBundle":
    """Open and verify a bundle; the trusted verifier is mandatory on every open.

    Resume callers pass the identity digest bound into their independently
    authenticated checkpoint. The verifier must renew/validate current training
    authority and return an opaque process-local binding object.
    """

    _require_opaque_capability(expected_run_binding, "expected run binding")
    if expected_identity_sha256 is not None:
        _require_hash(expected_identity_sha256, "expected_identity_sha256")
        if getattr(expected_run_binding, "expected_identity_sha256", None) != expected_identity_sha256:
            raise BundleAuthorizationError(
                "resume run binding is not tied to the checkpoint-authenticated bundle identity"
            )
    expected_fold_id = _strict_int(expected_fold_id, "expected_fold_id")
    if expected_fold_id not in ALLOWED_FOLDS:
        raise BundleError("expected_fold_id must be 0, 1, or 2")
    path = _assert_no_reparse_path(Path(directory))
    if not path.is_dir():
        raise BundleError("bundle path must be a directory")
    directory_metadata = path.stat(follow_symlinks=False)
    if directory_metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise BundleError("bundle directory must not be writable while it is being verified")
    _enumerate_bundle_files(path)
    manifest_fd = _open_plain_file(path, MANIFEST_FILE)
    payload_fd = _open_plain_file(path, DATA_FILE)
    try:
        manifest_raw = _read_fd_limited(manifest_fd, 1024 * 1024)
        manifest, identity_sha256 = _validate_manifest(manifest_raw)
        if manifest["fold_id"] != expected_fold_id:
            raise BundleAuthorizationError("bundle fold does not match the authorized run fold")
        if expected_identity_sha256 is not None and identity_sha256 != expected_identity_sha256:
            raise BundleAuthorizationError("bundle identity differs from the authenticated checkpoint")
        payload_desc = manifest["payload"]
        actual_payload_sha256 = _hash_fd(
            payload_fd, payload_desc["bytes"], manifest["segments"]
        )
        if actual_payload_sha256 != payload_desc["sha256"]:
            raise BundleError("payload SHA-256 mismatch")
        try:
            binding = trusted_verifier.verify_bundle(
                canonical_manifest=manifest_raw,
                bundle_identity_sha256=identity_sha256,
                payload_sha256=actual_payload_sha256,
                expected_identity_sha256=expected_identity_sha256,
                expected_fold_id=expected_fold_id,
                expected_run_binding=expected_run_binding,
            )
        except BundleAuthorizationError:
            raise
        except Exception as exc:
            raise BundleAuthorizationError("trusted root rejected the bundle or resume permit") from exc
        if isinstance(binding, (ABCMapping, list, tuple, set, frozenset, str, bytes, int, float, bool, type(None))):
            raise BundleAuthorizationError("trusted verifier must return a process-local opaque binding")
        if getattr(binding, "bundle_identity_sha256", None) != identity_sha256:
            raise BundleAuthorizationError("trusted binding is not tied to this exact bundle identity")
        _require_opaque_capability(binding, "trusted bundle binding")
        before = os.fstat(payload_fd)
        payload_map = mmap.mmap(payload_fd, length=0, access=mmap.ACCESS_READ)
        after = os.fstat(payload_fd)
        before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_id != after_id:
            payload_map.close()
            raise BundleError("payload changed before memory mapping")
        return PreparedTrainBundle(
            directory=path,
            manifest=manifest,
            identity_sha256=identity_sha256,
            binding=binding,
            payload_fd=payload_fd,
            payload_map=payload_map,
        )
    except Exception:
        os.close(payload_fd)
        raise
    finally:
        os.close(manifest_fd)


class PreparedTrainBundle:
    """Verified immutable train-only payload with deterministic origin access."""

    def __init__(
        self,
        *,
        directory: Path,
        manifest: dict[str, Any],
        identity_sha256: str,
        binding: object,
        payload_fd: int,
        payload_map: mmap.mmap,
    ) -> None:
        self.directory = directory
        self.manifest = manifest
        self.identity_sha256 = identity_sha256
        self.binding = binding
        self._payload_fd = payload_fd
        self._payload_map = payload_map
        self._segments = tuple(manifest["segments"])
        self._origin_ends: tuple[int, ...] = self._make_origin_ends()
        self.total_origins = manifest["total_origins"]
        self._closed = False

    def _make_origin_ends(self) -> tuple[int, ...]:
        cumulative = 0
        ends = []
        for segment in self._segments:
            cumulative += segment["origin_count"]
            ends.append(cumulative)
        if cumulative != self.manifest["total_origins"]:
            raise BundleError("internal origin total mismatch")
        return tuple(ends)

    @property
    def identity_summary(self) -> Mapping[str, Any]:
        return {"purpose": self.manifest["purpose"], "role": self.manifest["role"],
                "fold_id": self.manifest["fold_id"]}

    @property
    def source_data_identity_sha256(self) -> str:
        """Digest of root-bound source metadata, without exposing user identifiers."""
        return _sha256(_canonical_json(self.manifest["source_binding"]))

    def _assert_live(self) -> None:
        if self._closed:
            raise BundleError("bundle is closed")
        try:
            self.binding.assert_live()
        except Exception as exc:
            raise BundleAuthorizationError("bundle authorization expired or was revoked") from exc

    def assert_active(self) -> None:
        """Recheck the root-issued run binding before updates/checkpoints."""
        self._assert_live()

    def origin_for_rank(self, rank: int) -> tuple[int, int]:
        """Map a global origin rank to `(segment_index, segment_local_origin)`."""
        self._assert_live()
        rank = _strict_int(rank, "rank")
        if rank >= self.total_origins:
            raise IndexError("origin rank is outside the authorized fit10 pool")
        segment_index = bisect.bisect_right(self._origin_ends, rank)
        segment_start_rank = 0 if segment_index == 0 else self._origin_ends[segment_index - 1]
        local_origin = rank - segment_start_rank
        if local_origin >= self._segments[segment_index]["origin_count"]:
            raise BundleError("origin rank mapping exceeded segment-local range")
        return segment_index, local_origin

    def sample_window(self, segment_index: int, origin: int) -> tuple[bytes, bytes]:
        """Return packed `<f4` context and target bytes from one segment only."""
        self._assert_live()
        segment_index = _strict_int(segment_index, "segment_index")
        origin = _strict_int(origin, "origin")
        if segment_index >= len(self._segments):
            raise IndexError("segment index is outside the bundle")
        segment = self._segments[segment_index]
        if origin >= segment["origin_count"]:
            raise IndexError("origin is outside this segment's N-419 valid starts")
        byte_start = segment["offset_bytes"] + origin * ROW_BYTES
        context_end = byte_start + CONTEXT_SAMPLES * ROW_BYTES
        target_end = context_end + TARGET_SAMPLES * ROW_BYTES
        segment_end = segment["offset_bytes"] + segment["sample_count"] * ROW_BYTES
        if target_end > segment_end:
            raise BundleError("window would cross the segment boundary")
        contexts = self._payload_map[byte_start:context_end]
        targets = self._payload_map[context_end:target_end]
        if len(contexts) != CONTEXT_SAMPLES * ROW_BYTES or len(targets) != TARGET_SAMPLES * ROW_BYTES:
            raise BundleError("payload slice length does not match [400,4]/[20,4]")
        return contexts, targets

    def sample_rank(self, rank: int) -> tuple[bytes, bytes]:
        segment_index, origin = self.origin_for_rank(rank)
        return self.sample_window(segment_index, origin)

    def iter_ranked_windows(self, ranks: Iterable[int]):
        """Yield `(segment_index, origin, context_bytes, target_bytes)` lazily."""
        for rank in ranks:
            segment_index, origin = self.origin_for_rank(rank)
            contexts, targets = self.sample_window(segment_index, origin)
            yield segment_index, origin, contexts, targets

    def close(self) -> None:
        if not self._closed:
            self._payload_map.close()
            os.close(self._payload_fd)
            self._closed = True

    def abort_and_close(self) -> None:
        """Close this view and notify the root binding if it supports abort()."""
        if not self._closed:
            abort = getattr(self.binding, "abort", None)
            try:
                if callable(abort):
                    abort()
            finally:
                self.close()

    def __enter__(self) -> "PreparedTrainBundle":
        self._assert_live()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def make_test_attestation(identity_sha256: str, source_authorization_sha256: str) -> dict[str, str]:
    """Create only a clearly synthetic attestation shape for generated tests.

    This is not an Ed25519 signer and is not accepted as production authority by
    any verifier. Tests use this helper solely to construct structurally valid
    fixtures; production sign/verify callbacks belong to the trusted root.
    """
    _require_hash(identity_sha256, "identity_sha256")
    _require_hash(source_authorization_sha256, "source_authorization_sha256")
    signature = base64.b64encode(bytes(64)).decode("ascii")
    return {
        "algorithm": "Ed25519",
        "key_id": "synthetic-fixture-only",
        "identity_signature": signature,
        "authorization_sha256": source_authorization_sha256,
        "issued_utc": "2026-10-29T00:00:00Z",
        "content_validation": _CONTENT_VALIDATION,
    }


def write_synthetic_bundle(
    destination: str | os.PathLike[str],
    segments: Sequence[SegmentInput],
    *,
    fold_id: int = FOLD_ID,
    source_binding: Mapping[str, str],
    synthetic_attestation: Callable[[str], Mapping[str, str]],
) -> str:
    """Test-only builder. Never call this with real human arrays or bytes."""
    return _create_bundle(
        Path(destination),
        segments,
        fold_id=fold_id,
        source_binding=source_binding,
        attest_identity=synthetic_attestation,
    )


"""Fixed-path, coordinator-gated lazy reader for train-only quaternion windows.

This module never accepts a data path, split, digest, partition, or fold from a
caller.  The only production entrypoint is ``iter_approved_quaternion_windows``;
it requires a frozen protocol and coordinator approval at fixed private paths.
The reader does not fit models or compute fitted statistics.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Iterator

import h5py
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = WORKSPACE_ROOT / "ABC_IUEH"
WORKFLOW_ROOT = WORKSPACE_ROOT / "abc_topic_discovery" / "workflow_runtime"
STAGING_ROOT = Path(__file__).resolve().parent
TRAIN_INVENTORY_ROOT = STAGING_ROOT.parent / "train_origin_inventory"

# The accepted private source path was recorded by the fixed, hash-pinned
# pre-results plan.  It is data/processed/alyx.h5; there is no head_pose.h5 in
# the accepted metadata.  The SHA below remains a reference until a later
# coordinator-approved run verifies the file bytes before h5py opens it.
SOURCE_HDF5_RELATIVE_PATH = "data/processed/alyx.h5"
SOURCE_HDF5_PATH = REPO_ROOT / SOURCE_HDF5_RELATIVE_PATH
SOURCE_HDF5_SHA256 = "0986c483c2dc800ff8be1a2a8d2e82d909193f4c3e7c9f7b1eae3e2ea055ea62"
SPLIT_SHA256 = "9d7f342c15231a3f7f61fd8c9b95b75adc6aff641e7071d95a43a3832db49256"
ARC_CONFIG_SHA256 = "1ee9d2fff488ffa6952ce864177003706761b205901afddc69bca6951b26e753"
SEMANTIC_RUN_ID = "rebind-20261028T163453Z-55853a16fdb7"
SEMANTIC_CONFIG_SHA256 = "fb402662f474184c80ee0834381e9f9ff4140e5c63d2ec6b4d03ba04695401de"
SEMANTIC_RECEIPT_SHA256 = "38093d32ec39f27a48de49ad1f1a78b87595c526328dc31c1a536e17df540238"
ORIGIN_RUN_ID = "origin-inventory-20261028T171723Z-430ac21d"
ORIGIN_RECEIPT_SHA256 = "1f91d94062c320cdf48dfadb95070b660dc000b2bfc4610c39591e42fba94d9d"
ORIGIN_ORDER_SHA256 = "e9af1133a1aa32b27f5fffc6526f23c52d2c3d0e55a5a7447a6330ba994234e3"
ORIGIN_INTENT_SHA256 = "1757baca2b833f48107b81a695ae57c02ee97ec6d9942f304635d3f710b08706"
ORIGIN_MAP_SHA256 = "e4f19c1c5b8a1da4c3ca9a721723f001d21b382669695a72dda46af13c86ff0a"
ORIGIN_BINARY_SHA256 = "da4345d74ec59577ff76d66a148b033c723b919763a2e1123538074236892683"
ORIGIN_TOTAL = 3_330_312
ORIGIN_USERS = 15
ORIGIN_SESSIONS = 28
ORIGIN_SEGMENTS = 52
ORIGIN_FOLDS = 3

# A path-only, hash-pinned metadata record verifies the fixed local HDF5 name.
SOURCE_PATH_EVIDENCE_RELATIVE_PATH = (
    "data/reset_optimizer_probe/20261028_fixed_train_only/plan_before_results.json"
)
SOURCE_PATH_EVIDENCE_SHA256 = "dcb3ebd1e28e8289496235fc1252b40b97a7f233d4f31b867dcccbc41e1e3f77"

ORIGIN_RECEIPT_PATH = (
    REPO_ROOT / "data" / "research_controls" / "origin_bindings" / ORIGIN_RUN_ID /
    "origin_inventory_receipt.json"
)
ORIGIN_MAP_PATH = (
    REPO_ROOT / "data" / "research_controls" / "origin_bindings" / ORIGIN_RUN_ID /
    "origin_map_private.json"
)
ORIGIN_BINARY_PATH = (
    REPO_ROOT / "data" / "research_controls" / "origin_bindings" / ORIGIN_RUN_ID /
    "origin_index_canonical_le_i8.bin"
)
APPROVAL_PATH = (
    REPO_ROOT / "data" / "research_controls" / "q1_train_authorizations" /
    "q1_train_prefix_approval.json"
)
PROTOCOL_PATH = WORKFLOW_ROOT / "protocols" / "q1_train_prefix_protocol.json"

CONTEXT_SAMPLES = 400
HORIZON_SAMPLES = 20
WINDOW_SAMPLES = CONTEXT_SAMPLES + HORIZON_SAMPLES
INNER_SPLIT_SEED = 20261029  # fixed coordinator Q1 selection seed
INNER_FIT_USERS = 8
INNER_SELECTOR_USERS = 2
OUTER_FIT_USERS = 10
OUTER_VALIDATION_USERS = 5
SELECTOR_ORIGINS_PER_USER = 1024
MAX_APPROVED_PREFIX_WINDOWS = 32_768
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ORIGIN_BINARY_BYTES = 64 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUTHORIZATION_ID_RE = re.compile(r"^q1-train-prefix-[a-z0-9][a-z0-9._-]{0,80}$")
PROTOCOL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ROLE_ORDER = {"inner_fit": 0, "inner_selector": 1}

if str(TRAIN_INVENTORY_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_INVENTORY_ROOT))
import train_origin_inventory as origin_inventory  # noqa: E402


@dataclass(frozen=True)
class OriginRow:
    user_id: str = field(repr=False)
    session_id: str = field(repr=False)
    segment_id: str = field(repr=False)
    fold_id: int
    offset_int64: int
    count: int


@dataclass(frozen=True)
class AuthorizedPrefix:
    fold_id: int
    role: str
    origins_per_user: int


@dataclass(frozen=True)
class PlannedWindow:
    evaluation_fold: int
    role: str
    user_id: str = field(repr=False)
    session_id: str = field(repr=False)
    segment_id: str = field(repr=False)
    origin_index: int
    context_start: int
    slice_stop: int


@dataclass(frozen=True)
class MetadataBinding:
    parent: origin_inventory.ParentBinding
    source_path: Path
    source_identity: tuple[int, int, int, int]
    source_path_evidence_sha256: str
    origin_receipt_sha256: str
    origin_map_sha256: str
    origin_binary_sha256: str
    origin_rows: tuple[OriginRow, ...] = field(repr=False)
    origin_values: np.ndarray = field(repr=False)
    ownership: dict = field(repr=False)
    ownership_plan_sha256: str


@dataclass(frozen=True)
class ApprovedAccess:
    authorization_id: str
    protocol_id: str
    protocol_sha256: str
    prefixes: tuple[AuthorizedPrefix, ...]
    approval_sha256: str
    code_sha256: str
    test_sha256: str
    readme_sha256: str


@dataclass(frozen=True)
class QuaternionWindow:
    """One private train window; IDs are only for local split accounting."""

    evaluation_fold: int
    role: str
    user_id: str = field(repr=False)
    session_id: str = field(repr=False)
    segment_id: str = field(repr=False)
    origin_index: int
    context: np.ndarray = field(repr=False)
    target: np.ndarray = field(repr=False)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path, limit: int = MAX_JSON_BYTES) -> str:
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("fixed metadata file exceeds its size bound")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(raw: bytes, label: str) -> dict:
    try:
        doc = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} JSON is malformed") from exc
    if type(doc) is not dict:
        raise ValueError(f"{label} JSON root must be an object")
    return doc


def _path_without_symlink(path: Path, *, required: bool, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        try:
            if component.is_symlink():
                raise ValueError(f"{label} path contains a symlink component")
        except OSError as exc:
            raise ValueError(f"cannot verify {label} path components") from exc
    if not absolute.exists():
        if required:
            raise FileNotFoundError(f"fixed {label} is unavailable")
        return absolute
    resolved = absolute.resolve(strict=True)
    if Path(os.path.abspath(resolved)) != absolute:
        raise ValueError(f"{label} path does not resolve to its fixed spelling")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file")
    return absolute


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    result = Path(path).stat()
    if not stat.S_ISREG(result.st_mode):
        raise ValueError("fixed source HDF5 is not a regular file")
    return (result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns)


def _private_paths() -> dict[str, Path]:
    return {
        "source_hdf5": SOURCE_HDF5_PATH,
        "source_path_evidence": REPO_ROOT / SOURCE_PATH_EVIDENCE_RELATIVE_PATH,
        "origin_receipt": ORIGIN_RECEIPT_PATH,
        "origin_map": ORIGIN_MAP_PATH,
        "origin_binary": ORIGIN_BINARY_PATH,
        "approval": APPROVAL_PATH,
        "protocol": PROTOCOL_PATH,
    }


def _preflight_fixed_data_paths() -> tuple[dict[str, Path], tuple[int, int, int, int]]:
    """Stat/link preflight only. No payload/HDF5/recording bytes are opened."""
    paths = _private_paths()
    expected = {
        "source_hdf5": SOURCE_HDF5_PATH,
        "source_path_evidence": REPO_ROOT / SOURCE_PATH_EVIDENCE_RELATIVE_PATH,
        "origin_receipt": ORIGIN_RECEIPT_PATH,
        "origin_map": ORIGIN_MAP_PATH,
        "origin_binary": ORIGIN_BINARY_PATH,
        "approval": APPROVAL_PATH,
        "protocol": PROTOCOL_PATH,
    }
    if paths != expected:
        raise ValueError("fixed production path roles changed")
    for role, path in paths.items():
        _path_without_symlink(path, required=(role not in {"approval", "protocol"}), label=role)

    # Reuse the reviewed raw-recording identity scan and protected-artifact
    # inode scan. Both inspect path/stat metadata only; they read no payloads.
    source_helpers = origin_inventory._import_sources()
    semantic_root = origin_inventory.STAGING_ROOT.parent / "semantic_rebind"
    if str(semantic_root) not in sys.path:
        sys.path.insert(0, str(semantic_root))
    from semantic_rebind import rebind as semantic_guard

    alias_candidates = [
        (role, path) for role, path in paths.items()
        if path.exists() and role != "source_hdf5"
    ]
    alias_candidates.append(("source HDF5", paths["source_hdf5"]))
    source_helpers._reject_recording_aliases(alias_candidates, REPO_ROOT / "data")

    protected_ids = semantic_guard._protected_artifact_identities()
    for role, path in paths.items():
        # The canonical origin-index stream is itself a metadata input and is
        # intentionally stored as a private ignored .bin file. Its fixed path,
        # symlink-free spelling, exact size and pinned bytes are checked below;
        # `.bin` also appears in the broad protected-payload suffix set.
        if role in {"source_hdf5", "approval", "protocol", "origin_binary"} or not path.exists():
            continue
        result = path.stat()
        if (result.st_dev, result.st_ino) in protected_ids:
            raise ValueError(f"private metadata path aliases a protected payload: {role}")

    source_path = paths["source_hdf5"]
    source_identity = _file_identity(source_path)
    # The fixed source HDF5 must be the only protected artifact with its inode.
    # This catches hard links to the feature cache or any other local HDF5/model
    # file without reading data bytes.
    aliases: list[Path] = []
    protected_suffixes = semantic_guard.PROTECTED_ARTIFACT_SUFFIXES
    data_root = REPO_ROOT / "data"
    for directory, subdirs, filenames in os.walk(data_root):
        current = Path(directory)
        subdirs[:] = [name for name in subdirs if name != "raw"]
        for filename in filenames:
            candidate = current / filename
            if candidate.suffix.lower() not in protected_suffixes:
                continue
            try:
                result = candidate.stat()
            except OSError as exc:
                raise ValueError("cannot verify protected source artifact identities") from exc
            if (result.st_dev, result.st_ino) == source_identity[:2]:
                aliases.append(candidate)
    if len(aliases) != 1 or Path(os.path.abspath(aliases[0])) != Path(os.path.abspath(source_path)):
        raise ValueError("fixed source HDF5 aliases another protected data/model path")
    return paths, source_identity


def _source_path_evidence(path: Path) -> str:
    raw = Path(path).read_bytes()
    digest = _sha256_bytes(raw)
    if digest != SOURCE_PATH_EVIDENCE_SHA256:
        raise ValueError("source HDF5 path evidence bytes differ from the fixed metadata pin")
    doc = _strict_json(raw, "source HDF5 path evidence")
    plan = doc.get("plan")
    if (type(plan) is not dict or
            plan.get("source_hdf5") != SOURCE_HDF5_RELATIVE_PATH or
            plan.get("split") != "data/user_split.json"):
        raise ValueError("fixed metadata no longer identifies the accepted Alyx HDF5 path")
    return digest


def _strict_int(value, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _read_private_metadata(path: Path, *, max_bytes: int, label: str) -> tuple[bytes, dict]:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{label} exceeds its fixed size bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{label} changed while it was read")
    return raw, _strict_json(raw, label)


def _validate_origin_receipt(doc: dict, parent: origin_inventory.ParentBinding,
                             raw: bytes) -> str:
    digest = _sha256_bytes(raw)
    if digest != ORIGIN_RECEIPT_SHA256:
        raise ValueError("actual train-origin receipt bytes differ from the accepted run")
    expected = {
        "schema_version": 1,
        "status": "completed_train_origin_index_only",
        "run_id": ORIGIN_RUN_ID,
        "parent_semantic_run_id": SEMANTIC_RUN_ID,
        "semantic_config_sha256": SEMANTIC_CONFIG_SHA256,
        "semantic_receipt_sha256": SEMANTIC_RECEIPT_SHA256,
        "arc_config_sha256": ARC_CONFIG_SHA256,
        "split_sha256": SPLIT_SHA256,
        "source_hdf5_sha256_reference_only": SOURCE_HDF5_SHA256,
        "cache_relative_path": "ABC_IUEH/data/baseline_runs/20261028_r80/features.h5",
        "cache_bytes": 1_724_653_596,
        "cache_sha256_opaque_file_integrity": origin_inventory.CACHE_SHA256,
        "total_origin_count": ORIGIN_TOTAL,
        "segment_count": ORIGIN_SEGMENTS,
        "session_count": ORIGIN_SESSIONS,
        "train_user_count": ORIGIN_USERS,
        "fold_count": ORIGIN_FOLDS,
        "fold_digest": parent.fold_digest,
        "realized_origin_order_sha256": ORIGIN_ORDER_SHA256,
        "origin_schedule_intent_semantic_id": ORIGIN_INTENT_SHA256,
        "fits_performed": 0,
        "q1_grant": False,
        "new_budget_or_debit": False,
    }
    for key, value in expected.items():
        if type(doc.get(key)) is not type(value) or doc.get(key) != value:
            raise ValueError(f"actual train-origin receipt field differs: {key}")
    source_bindings = doc.get("source_metadata_bindings")
    if (type(source_bindings) is not dict or
            source_bindings.get("integrated_sources_helper_sha256") != origin_inventory.SOURCE_HELPER_CODE_SHA256 or
            source_bindings.get("semantic_code_binding_map_sha256") != origin_inventory.canonical_json_sha256(
                parent.metadata["semantic_config"]["code_byte_bindings"]
            )):
        raise ValueError("train-origin receipt source/helper bindings differ from the fixed metadata parent")
    policy = doc.get("read_policy")
    if (type(policy) is not dict or policy.get("development_or_final_values_read") != 0 or
            policy.get("source_pose_hdf5_opened") is not False or
            policy.get("feature_target_error_scaler_weight_values_read") != 0 or
            policy.get("denied_dataset_value_read_attempts") != 0 or
            policy.get("denied_group_dereference_attempts") != 0 or
            policy.get("denied_foreign_file_access_attempts") != 0):
        raise ValueError("actual origin receipt reports an unexpected signal-read scope")
    private = doc.get("private_mapping")
    if (type(private) is not dict or
            private.get("origin_map_relative_path") != (
                f"data/research_controls/origin_bindings/{ORIGIN_RUN_ID}/origin_map_private.json") or
            private.get("origin_map_sha256") != ORIGIN_MAP_SHA256 or
            private.get("origin_index_binary_relative_path") != (
                f"data/research_controls/origin_bindings/{ORIGIN_RUN_ID}/origin_index_canonical_le_i8.bin") or
            private.get("origin_index_binary_sha256") != ORIGIN_BINARY_SHA256):
        raise ValueError("actual origin receipt private map/index bindings differ from the fixed artifacts")
    return digest


def _parse_origin_map(doc: dict, parent: origin_inventory.ParentBinding,
                      raw: bytes) -> tuple[OriginRow, ...]:
    if _sha256_bytes(raw) != ORIGIN_MAP_SHA256:
        raise ValueError("private train-origin map bytes differ from the accepted origin receipt")
    if (doc.get("schema_version") != 1 or
            doc.get("parent_semantic_run_id") != SEMANTIC_RUN_ID or
            doc.get("origin_schedule_intent_semantic_id") != ORIGIN_INTENT_SHA256 or
            doc.get("canonical_order") != (
                "fold_id ascending; each fold's validation users lexically; session_id lexically; "
                "zero-padded segment_id numerically; stored origin_index values in slice order") or
            doc.get("index_binary_file") != ORIGIN_BINARY_PATH.name or
            doc.get("index_binary_sha256") != ORIGIN_BINARY_SHA256 or
            doc.get("index_binary_int64_count") != ORIGIN_TOTAL):
        raise ValueError("private origin map header differs from the authenticated train receipt")
    rows_doc = doc.get("rows")
    if type(rows_doc) is not list or len(rows_doc) != ORIGIN_SEGMENTS:
        raise ValueError("private origin map has an unexpected row count")
    expected_order = list(origin_inventory._iter_canonical_segments(parent.expected_train))
    if len(expected_order) != ORIGIN_SEGMENTS:
        raise ValueError("fixed metadata parent no longer has the accepted train segment count")
    rows: list[OriginRow] = []
    next_offset = 0
    for raw_row, expected_tuple in zip(rows_doc, expected_order):
        if type(raw_row) is not dict:
            raise ValueError("private origin map row must be an object")
        fold, user, session, segment = expected_tuple
        expected_names = {
            "fold": fold, "user_id": user, "session_id": session,
            "segment_id": segment,
        }
        for key, expected in expected_names.items():
            if raw_row.get(key) != expected or type(raw_row.get(key)) is not type(expected):
                raise ValueError("private origin map order/ownership differs from the fixed train split")
        offset = _strict_int(raw_row.get("offset_int64"), "origin map offset")
        count = _strict_int(raw_row.get("count"), "origin map count", minimum=1)
        if offset != next_offset:
            raise ValueError("private origin map offsets are not contiguous")
        if (raw_row.get("first_origin") != 399 or type(raw_row.get("first_origin")) is not int or
                raw_row.get("last_origin") != 398 + count or type(raw_row.get("last_origin")) is not int or
                not SHA256_RE.fullmatch(str(raw_row.get("origin_values_sha256", "")))):
            raise ValueError("private origin map row window range/digest is malformed")
        rows.append(OriginRow(user, session, segment, fold, offset, count))
        next_offset += count
    if next_offset != ORIGIN_TOTAL:
        raise ValueError("private origin map total differs from the accepted train origin count")
    return tuple(rows)


def selector_origin_indices(origin_count: int, *, requested: int = SELECTOR_ORIGINS_PER_USER) -> np.ndarray:
    """Endpoint-inclusive evenly spaced positions, matching the Q1 1,024 rule."""
    if type(origin_count) is not int or origin_count < 1:
        raise ValueError("origin_count must be a positive integer")
    if type(requested) is not int or requested < 1 or requested > SELECTOR_ORIGINS_PER_USER:
        raise ValueError("requested must be between one and the fixed 1,024-origin bound")
    count = min(origin_count, requested)
    positions = np.linspace(0, origin_count - 1, num=count, dtype=np.int64)
    if len(np.unique(positions)) != count:
        raise AssertionError("endpoint-inclusive origin sampling unexpectedly duplicated a position")
    return positions


def _inner_ownership(expected: origin_inventory.ExpectedTrain) -> dict:
    """Build exact fold roles: outer 10/5, then inner 8/2 within the 10."""
    result: dict[int, dict[str, tuple[str, ...]]] = {}
    users = set(expected.users)
    for fold_id, outer_validation in enumerate(expected.fold_validation_order):
        outer_validation_set = set(outer_validation)
        outer_fit = sorted(users - outer_validation_set)
        if len(outer_fit) != OUTER_FIT_USERS or len(outer_validation_set) != OUTER_VALIDATION_USERS:
            raise ValueError("frozen train fold is not the required outer 10/5 ownership")
        ranked = sorted(outer_fit, key=lambda user: (
            hashlib.sha256(
                f"q1-inner-split-v1\0{INNER_SPLIT_SEED}\0{fold_id}\0{user}".encode("utf-8")
            ).digest(),
            user,
        ))
        inner_selector = tuple(sorted(ranked[:INNER_SELECTOR_USERS]))
        inner_fit = tuple(sorted(ranked[INNER_SELECTOR_USERS:]))
        if (len(inner_fit) != INNER_FIT_USERS or len(inner_selector) != INNER_SELECTOR_USERS or
                set(inner_fit).intersection(inner_selector) or
                set(inner_fit).union(inner_selector) != set(outer_fit)):
            raise ValueError("inner train split is not an exact 8/2 partition of outer fit users")
        result[fold_id] = {
            "outer_fit": tuple(outer_fit),
            "outer_validation": tuple(sorted(outer_validation_set)),
            "inner_fit": inner_fit,
            "inner_selector": inner_selector,
        }
    return result


def _ownership_digest(ownership: dict) -> str:
    canonical = {
        "schema_version": 1,
        "algorithm": "sha256_rank_q1-inner-split-v1_seed_fold_user_first2_selector_rest8_fit",
        "seed": INNER_SPLIT_SEED,
        "folds": [
            {"fold_id": fold, **{key: list(values) for key, values in roles.items()}}
            for fold, roles in sorted(ownership.items())
        ],
    }
    return origin_inventory.canonical_json_sha256(canonical)


def validate_metadata_bindings() -> MetadataBinding:
    """Authenticate fixed metadata/origin inputs; never open source or cache HDF5."""
    # This reviewed validator checks exact fixed JSON paths/hashes, source tuple
    # receipts, folds and current helper pins. It only stats the feature cache.
    parent = origin_inventory.validate_production_parent()
    if (parent.source_hdf5_sha256_reference != SOURCE_HDF5_SHA256 or
            parent.split_sha256 != SPLIT_SHA256 or
            parent.expected_train.expected_windows != ORIGIN_TOTAL):
        raise ValueError("train-only metadata parent differs from this reader's fixed source/split")

    paths, source_identity = _preflight_fixed_data_paths()
    path_evidence_hash = _source_path_evidence(paths["source_path_evidence"])

    receipt_raw, receipt = _read_private_metadata(
        paths["origin_receipt"], max_bytes=MAX_JSON_BYTES, label="actual origin receipt")
    receipt_hash = _validate_origin_receipt(receipt, parent, receipt_raw)
    map_raw, map_doc = _read_private_metadata(
        paths["origin_map"], max_bytes=MAX_JSON_BYTES, label="private origin map")
    rows = _parse_origin_map(map_doc, parent, map_raw)

    if paths["origin_binary"].stat().st_size != ORIGIN_TOTAL * np.dtype("<i8").itemsize:
        raise ValueError("private origin-index binary length differs from its accepted receipt")
    if paths["origin_binary"].stat().st_size > MAX_ORIGIN_BINARY_BYTES:
        raise ValueError("private origin-index binary exceeds its bounded metadata size")
    binary_raw = paths["origin_binary"].read_bytes()
    binary_hash = _sha256_bytes(binary_raw)
    if binary_hash != ORIGIN_BINARY_SHA256:
        raise ValueError("private origin-index binary bytes differ from the accepted receipt")
    origin_values = np.frombuffer(binary_raw, dtype="<i8")
    origin_values.setflags(write=False)

    ownership = _inner_ownership(parent.expected_train)
    return MetadataBinding(
        parent=parent,
        source_path=paths["source_hdf5"],
        source_identity=source_identity,
        source_path_evidence_sha256=path_evidence_hash,
        origin_receipt_sha256=receipt_hash,
        origin_map_sha256=_sha256_bytes(map_raw),
        origin_binary_sha256=binary_hash,
        origin_rows=rows,
        origin_values=origin_values,
        ownership=ownership,
        ownership_plan_sha256=_ownership_digest(ownership),
    )


def _protocol_and_approval_bindings(binding: MetadataBinding) -> dict[str, str]:
    code_hash = _sha256_file(Path(__file__), limit=2 * 1024 * 1024)
    test_path = STAGING_ROOT / "test_q1_train_reader.py"
    readme_path = STAGING_ROOT / "README.md"
    test_hash = _sha256_file(test_path, limit=2 * 1024 * 1024)
    readme_hash = _sha256_file(readme_path, limit=2 * 1024 * 1024)
    return {
        "reader_code_sha256": code_hash,
        "synthetic_test_sha256": test_hash,
        "reader_readme_sha256": readme_hash,
        "source_path_evidence_sha256": binding.source_path_evidence_sha256,
        "source_hdf5_sha256_reference": SOURCE_HDF5_SHA256,
        "source_hdf5_relative_path": SOURCE_HDF5_RELATIVE_PATH,
        "split_sha256": SPLIT_SHA256,
        "arc_config_sha256": ARC_CONFIG_SHA256,
        "semantic_config_sha256": SEMANTIC_CONFIG_SHA256,
        "semantic_receipt_sha256": SEMANTIC_RECEIPT_SHA256,
        "origin_inventory_receipt_sha256": binding.origin_receipt_sha256,
        "origin_map_sha256": binding.origin_map_sha256,
        "origin_index_binary_sha256": binding.origin_binary_sha256,
        "realized_origin_order_sha256": ORIGIN_ORDER_SHA256,
        "ownership_plan_sha256": binding.ownership_plan_sha256,
    }


def _validate_protocol(doc: dict, digest: str, binding: MetadataBinding) -> str:
    if (type(doc.get("schema_version")) is not int or doc.get("schema_version") != 1 or
            doc.get("kind") != "q1_train_prefix_protocol" or
            doc.get("status") != "FROZEN_BEFORE_PREFIX"):
        raise PermissionError("fixed Q1 protocol is not frozen for a prefix run")
    protocol_id = doc.get("protocol_id")
    if not isinstance(protocol_id, str) or not PROTOCOL_ID_RE.fullmatch(protocol_id):
        raise ValueError("fixed Q1 protocol has an invalid protocol_id")
    expected_boundary = {
        "source_partition": "train",
        "source_hdf5_relative_path": SOURCE_HDF5_RELATIVE_PATH,
        "source_hdf5_sha256": SOURCE_HDF5_SHA256,
        "split_sha256": SPLIT_SHA256,
        "fold_digest": binding.parent.fold_digest,
        "outer_train_users": ORIGIN_USERS,
        "fold_count": ORIGIN_FOLDS,
        "outer_fit_users_per_fold": OUTER_FIT_USERS,
        "outer_validation_users_per_fold": OUTER_VALIDATION_USERS,
        "inner_fit_users_per_fold": INNER_FIT_USERS,
        "inner_selector_users_per_fold": INNER_SELECTOR_USERS,
        "inner_split_seed": INNER_SPLIT_SEED,
        "ownership_plan_sha256": binding.ownership_plan_sha256,
        "origin_sampling_strategy": "endpoint_inclusive_evenly_spaced_per_user",
        "selector_origins_per_user": SELECTOR_ORIGINS_PER_USER,
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "shift": 1,
    }
    if not origin_inventory._exact_equal(doc.get("data_boundary"), expected_boundary):
        raise PermissionError("fixed Q1 protocol data boundary differs from this train reader")
    if not SHA256_RE.fullmatch(digest):
        raise ValueError("fixed Q1 protocol hash is malformed")
    return protocol_id


def _validate_authorization(doc: dict, protocol_doc: dict, protocol_sha256: str,
                            binding: MetadataBinding, approval_sha256: str) -> ApprovedAccess:
    if (type(doc.get("schema_version")) is not int or doc.get("schema_version") != 1 or
            doc.get("kind") != "q1_train_prefix_approval"):
        raise PermissionError("fixed coordinator approval has the wrong schema")
    if doc.get("status") != "APPROVED_AFTER_INDEPENDENT_REVIEW":
        raise PermissionError("fixed coordinator approval is absent or not approved")
    authorization_id = doc.get("authorization_id")
    if not isinstance(authorization_id, str) or not AUTHORIZATION_ID_RE.fullmatch(authorization_id):
        raise ValueError("coordinator approval has an invalid authorization_id")
    if doc.get("approved_by_role") != "coordinator":
        raise PermissionError("approval must be issued by the workflow coordinator")
    approved_utc = doc.get("approved_utc")
    if not isinstance(approved_utc, str):
        raise ValueError("approval timestamp is missing")
    try:
        approved_time = datetime.fromisoformat(approved_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("approval timestamp is not ISO-8601") from exc
    if approved_time.tzinfo is None:
        raise ValueError("approval timestamp must carry a timezone")
    protocol_id = _validate_protocol(protocol_doc, protocol_sha256, binding)
    if (doc.get("protocol_id") != protocol_id or
            doc.get("protocol_sha256") != protocol_sha256):
        raise PermissionError("coordinator approval is not bound to the fixed frozen protocol")

    expected_artifacts = _protocol_and_approval_bindings(binding)
    if not origin_inventory._exact_equal(doc.get("reviewed_artifacts"), expected_artifacts):
        raise PermissionError("coordinator approval does not bind the exact reader/tests/docs and source metadata")
    expected_boundary = {
        "source_partition": "train",
        "outer_train_users": ORIGIN_USERS,
        "fold_count": ORIGIN_FOLDS,
        "outer_fit_users_per_fold": OUTER_FIT_USERS,
        "outer_validation_users_per_fold": OUTER_VALIDATION_USERS,
        "inner_fit_users_per_fold": INNER_FIT_USERS,
        "inner_selector_users_per_fold": INNER_SELECTOR_USERS,
        "inner_split_seed": INNER_SPLIT_SEED,
        "ownership_plan_sha256": binding.ownership_plan_sha256,
        "origin_sampling_strategy": "endpoint_inclusive_evenly_spaced_per_user",
        "selector_origins_per_user": SELECTOR_ORIGINS_PER_USER,
    }
    if not origin_inventory._exact_equal(doc.get("ownership"), expected_boundary):
        raise PermissionError("approval fold ownership differs from the fixed 10/5 and 8/2 plan")

    raw_prefixes = doc.get("authorized_prefixes")
    if type(raw_prefixes) is not list or not raw_prefixes:
        raise ValueError("approval must list at least one bounded train-prefix role")
    prefixes: list[AuthorizedPrefix] = []
    seen: set[tuple[int, str]] = set()
    for item in raw_prefixes:
        if type(item) is not dict or set(item) != {"fold_id", "role", "sampling", "origins_per_user"}:
            raise ValueError("authorized prefix row has an unexpected schema")
        fold_id = _strict_int(item.get("fold_id"), "authorized fold_id")
        role = item.get("role")
        if (fold_id not in binding.ownership or not isinstance(role, str) or
                role not in ROLE_ORDER):
            raise ValueError("authorized prefix must use a frozen inner train/selector role")
        if item.get("sampling") != "endpoint_inclusive_evenly_spaced_per_user_v1":
            raise ValueError("authorized prefix must use the fixed endpoint-inclusive per-user sampler")
        per_user = _strict_int(item.get("origins_per_user"), "authorized origins_per_user", minimum=1)
        if per_user > SELECTOR_ORIGINS_PER_USER:
            raise PermissionError("authorized per-user sample exceeds the fixed 1,024-origin bound")
        if role == "inner_selector" and per_user != SELECTOR_ORIGINS_PER_USER:
            raise PermissionError("selector users require exactly 1,024 endpoint-inclusive origins each")
        key = (fold_id, role)
        if key in seen:
            raise ValueError("duplicate authorized fold/role prefix")
        seen.add(key)
        role_users = set(binding.ownership[fold_id][role])
        if not role_users:
            raise ValueError("fixed train role has no users")
        prefixes.append(AuthorizedPrefix(fold_id, role, per_user))
    expected_order = sorted(prefixes, key=lambda p: (p.fold_id, ROLE_ORDER[p.role]))
    if prefixes != expected_order:
        raise ValueError("authorized prefix rows must use canonical fold/role order")
    projected_total = 0
    for item in prefixes:
        role_users = set(binding.ownership[item.fold_id][item.role])
        for user in role_users:
            available = sum(row.count for row in binding.origin_rows if row.user_id == user)
            projected_total += min(item.origins_per_user, available)
    if projected_total > MAX_APPROVED_PREFIX_WINDOWS:
        raise PermissionError("authorized prefix exceeds the reviewed 32,768-window reader bound")

    artifact_hashes = expected_artifacts
    return ApprovedAccess(
        authorization_id=authorization_id,
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        prefixes=tuple(prefixes),
        approval_sha256=approval_sha256,
        code_sha256=artifact_hashes["reader_code_sha256"],
        test_sha256=artifact_hashes["synthetic_test_sha256"],
        readme_sha256=artifact_hashes["reader_readme_sha256"],
    )


def load_fixed_approval(binding: MetadataBinding) -> ApprovedAccess:
    """Read the one exact approval/protocol pair, after all inode preflights."""
    paths, _source_identity = _preflight_fixed_data_paths()
    for role in ("approval", "protocol"):
        path = paths[role]
        if not path.exists():
            raise PermissionError(f"fixed {role} metadata is not present; no source HDF5 may be opened")
    for role in ("approval", "protocol"):
        if paths[role].stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"fixed {role} file exceeds the metadata bound")
    protocol_raw = paths["protocol"].read_bytes()
    if len(protocol_raw) > MAX_JSON_BYTES:
        raise ValueError("fixed Q1 protocol grew beyond the metadata size bound")
    protocol_hash = _sha256_bytes(protocol_raw)
    protocol_doc = _strict_json(protocol_raw, "fixed Q1 protocol")
    approval_raw = paths["approval"].read_bytes()
    if len(approval_raw) > MAX_JSON_BYTES:
        raise ValueError("fixed coordinator approval grew beyond the metadata size bound")
    approval_doc = _strict_json(approval_raw, "fixed coordinator approval")
    return _validate_authorization(
        approval_doc, protocol_doc, protocol_hash, binding, _sha256_bytes(approval_raw))


def build_read_plan(binding: MetadataBinding, approval: ApprovedAccess) -> tuple[PlannedWindow, ...]:
    plan: list[PlannedWindow] = []
    for prefix in approval.prefixes:
        role_users = set(binding.ownership[prefix.fold_id][prefix.role])
        for user in sorted(role_users):
            user_rows = [row for row in binding.origin_rows if row.user_id == user]
            user_origin_count = sum(row.count for row in user_rows)
            if user_origin_count <= 0:
                raise ValueError("approved train user has no authenticated origins")
            selected_positions = selector_origin_indices(
                user_origin_count, requested=prefix.origins_per_user)
            cumulative = 0
            selected_position = 0
            for row in user_rows:
                stop_rank = cumulative + row.count
                while (selected_position < len(selected_positions) and
                       int(selected_positions[selected_position]) < stop_rank):
                    local_index = int(selected_positions[selected_position]) - cumulative
                    if local_index < 0:
                        raise ValueError("per-user sample position is not monotonically ordered")
                    origin_offset = row.offset_int64 + local_index
                    if origin_offset >= len(binding.origin_values):
                        raise ValueError("selected origin position is outside the authenticated binary index")
                    origin = int(binding.origin_values[origin_offset])
                    if origin != 399 + local_index:
                        raise ValueError("selected realized origin differs from its authenticated segment range")
                    start = origin - (CONTEXT_SAMPLES - 1)
                    stop = origin + HORIZON_SAMPLES + 1
                    if start < 0 or stop - start != WINDOW_SAMPLES:
                        raise ValueError("selected train origin does not yield exactly one 400-to-20 window")
                    plan.append(PlannedWindow(
                        evaluation_fold=prefix.fold_id, role=prefix.role,
                        user_id=row.user_id, session_id=row.session_id,
                        segment_id=row.segment_id, origin_index=origin,
                        context_start=start, slice_stop=stop,
                    ))
                    selected_position += 1
                cumulative = stop_rank
            if selected_position != len(selected_positions):
                raise ValueError("per-user origin sample could not be mapped to the fixed private segments")
    if not plan or len(plan) > MAX_APPROVED_PREFIX_WINDOWS:
        raise ValueError("authorized train prefix is empty or exceeds the reader bound")
    return tuple(plan)


def _object_address(obj) -> int:
    return int(h5py.h5o.get_info(obj.id).addr)


def _hard_child(parent, name: str, *, group: bool):
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise ValueError("source HDF5 path contains a soft or external link")
    obj = parent[name]
    if group and not isinstance(obj, h5py.Group):
        raise ValueError("source HDF5 hierarchy has a non-group path component")
    if not group and not isinstance(obj, h5py.Dataset):
        raise ValueError("source HDF5 segment child is not a dataset")
    return obj


def _register_unique(obj, path: str, addresses: dict[int, str]) -> None:
    info = h5py.h5o.get_info(obj.id)
    if int(info.rc) != 1:
        raise ValueError("source HDF5 object has a hard-link alias")
    address = int(info.addr)
    previous = addresses.get(address)
    if previous is not None and previous != path:
        raise ValueError("source HDF5 object aliases two addressed train paths")
    addresses[address] = path


def _dataset_metadata(dataset, *, dtype, ndim: int) -> tuple[int, ...]:
    if (dataset.dtype != np.dtype(dtype) or dataset.ndim != ndim or
            dataset.is_virtual or dataset.external):
        raise ValueError("source HDF5 dataset dtype/dimensions/storage are not local and fixed")
    if dataset.chunks is None:
        raise ValueError("source HDF5 signal dataset is not chunked")
    return tuple(int(dim) for dim in dataset.shape)


def _validate_source_root(source_file: h5py.File, binding: MetadataBinding) -> None:
    attrs = source_file.attrs
    required = {
        "schema_version": 1,
        "source": "whoisalyx",
        "license": "CC-BY-NC-SA-4.0",
        "quaternion_order": "wxyz",
        "coordinate_frame": "source-native OpenVR; no handedness transform",
        "sample_rate_hz": 100.0,
        "timing_dispute": "20 samples / 100 Hz / 100 ms unresolved",
        "interpolation": "offline bracketing SLERP, not a zero-delay sensor pipeline",
        "source_receipt_sha256": binding.parent.metadata_sha256["legacy_receipt"],
        "source_inventory_file_sha256": binding.parent.metadata_sha256["inventory"],
        "source_record_inventory_sha256": binding.parent.metadata["semantic_config"][
            "source_binding"]["authenticated_record_inventory_sha256"],
    }
    for key, expected in required.items():
        actual = attrs.get(key)
        if isinstance(expected, float):
            if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, float, np.integer, np.floating)) or float(actual) != expected:
                raise ValueError(f"source HDF5 root metadata differs: {key}")
        elif type(actual) not in (type(expected), np.int64, np.int32) or actual != expected:
            raise ValueError(f"source HDF5 root metadata differs: {key}")
    try:
        config = json.loads(attrs["config_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("source HDF5 config metadata is malformed") from exc
    if config != binding.parent.metadata["pipeline_config"]:
        raise ValueError("source HDF5 preprocessing config differs from the fixed metadata parent")
    if set(source_file.keys()) != {"users"}:
        raise ValueError("source HDF5 root contains unexpected non-source groups")


def _metadata_paths(plan: tuple[PlannedWindow, ...]) -> set[str]:
    paths = {"/users"}
    for window in plan:
        user = window.user_id
        session = window.session_id
        segment = window.segment_id
        up = f"/users/{user}"
        sp = f"{up}/sessions"
        sessp = f"{sp}/{session}"
        segsp = f"{sessp}/segments"
        gp = f"{segsp}/{segment}"
        paths.update({up, sp, sessp, segsp, gp, f"{gp}/timestamps", f"{gp}/quaternion"})
    return paths


def _preflight_selected_train_metadata(source_file: h5py.File, binding: MetadataBinding,
                                       plan: tuple[PlannedWindow, ...]) -> tuple[dict[str, h5py.Dataset], set[str], Counter]:
    allowed_paths = _metadata_paths(plan)
    # A path appears in the value allowlist once per approved read. Counter
    # multiplicity permits explicitly approved repeat use across roles/folds.
    allowed_reads: Counter = Counter()
    for window in plan:
        path = (f"/users/{window.user_id}/sessions/{window.session_id}/segments/"
                f"{window.segment_id}/quaternion")
        allowed_reads[(path, window.context_start, window.slice_stop)] += 1

    selected_users = sorted({window.user_id for window in plan})
    selected_rows = {(row.user_id, row.session_id, row.segment_id): row for row in binding.origin_rows}
    opened_datasets: dict[str, h5py.Dataset] = {}
    addresses: dict[int, str] = {}

    users = _hard_child(source_file, "users", group=True)
    _register_unique(users, "/users", addresses)
    for user in selected_users:
        if user not in binding.parent.expected_train.user_sessions:
            raise PermissionError("planned source path is outside the exact outer train split")
        user_group = _hard_child(users, user, group=True)
        user_path = f"/users/{user}"
        _register_unique(user_group, user_path, addresses)
        if set(user_group.keys()) != {"sessions"}:
            raise ValueError("selected train user hierarchy differs from source schema")
        sessions = _hard_child(user_group, "sessions", group=True)
        _register_unique(sessions, f"{user_path}/sessions", addresses)
        expected_sessions = binding.parent.expected_train.user_sessions[user]
        if set(sessions.keys()) != set(expected_sessions):
            raise ValueError("selected train sessions differ from authenticated accepted quality metadata")
        for session in sorted(expected_sessions):
            session_group = _hard_child(sessions, session, group=True)
            session_path = f"{user_path}/sessions/{session}"
            _register_unique(session_group, session_path, addresses)
            if set(session_group.keys()) != {"segments"}:
                raise ValueError("selected train session hierarchy differs from source schema")
            segments = _hard_child(session_group, "segments", group=True)
            _register_unique(segments, f"{session_path}/segments", addresses)
            expected_segments = expected_sessions[session]
            if set(segments.keys()) != set(expected_segments):
                raise ValueError("selected train segments differ from authenticated quality metadata")
            session_samples = 0
            for segment in expected_segments:
                segment_group = _hard_child(segments, segment, group=True)
                segment_path = f"{session_path}/segments/{segment}"
                _register_unique(segment_group, segment_path, addresses)
                if set(segment_group.keys()) != {"timestamps", "quaternion"}:
                    raise ValueError("selected source segment has unexpected feature/target payloads")
                timestamps = _hard_child(segment_group, "timestamps", group=False)
                quaternion = _hard_child(segment_group, "quaternion", group=False)
                n_t = _dataset_metadata(timestamps, dtype="float64", ndim=1)
                n_q = _dataset_metadata(quaternion, dtype="float32", ndim=2)
                if n_q[1:] != (4,) or n_t != (n_q[0],) or n_q[0] < 1000:
                    raise ValueError("selected source segment has an invalid sample shape/floor")
                _register_unique(timestamps, f"{segment_path}/timestamps", addresses)
                _register_unique(quaternion, f"{segment_path}/quaternion", addresses)
                origin_row = selected_rows.get((user, session, segment))
                if origin_row is None or origin_row.count != n_q[0] - 419:
                    raise ValueError("source segment sample count differs from its authenticated origin index")
                session_samples += n_q[0]
                if f"{segment_path}/quaternion" in allowed_paths:
                    opened_datasets[f"{segment_path}/quaternion"] = quaternion
            expected_samples = binding.parent.expected_train.session_windows[(user, session)] + 419 * len(expected_segments)
            if session_samples != expected_samples:
                raise ValueError("selected train session sample total differs from quality metadata")
    return opened_datasets, allowed_paths, allowed_reads


class HDF5ReadGuard:
    """Permit exact 420-row slices from exact selected train quaternion datasets."""

    def __init__(self, source_file: h5py.File, *, allowed_metadata_paths: set[str],
                 allowed_reads: Counter):
        self.file_id = int(source_file.id.id)
        self.allowed_metadata_paths = allowed_metadata_paths
        self.remaining_reads = allowed_reads.copy()
        self.audit = {
            "quaternion_slice_calls": 0,
            "quaternion_rows_read": 0,
            "denied_value_reads": 0,
            "denied_group_dereferences": 0,
            "denied_foreign_file_accesses": 0,
        }
        self.metadata_paths_touched: list[str] = []

    def _belongs(self, obj) -> bool:
        try:
            return int(obj.file.id.id) == self.file_id
        except (AttributeError, ValueError, RuntimeError):
            return False

    def __enter__(self):
        self._group_getitem = h5py.Group.__getitem__
        self._dataset_getitem = h5py.Dataset.__getitem__
        self._read_direct = h5py.Dataset.read_direct
        self._array = h5py.Dataset.__array__
        guard = self

        def group_getitem(group, key):
            if not guard._belongs(group):
                guard.audit["denied_foreign_file_accesses"] += 1
                guard.audit["denied_group_dereferences"] += 1
                raise PermissionError("HDF5 group belongs to a foreign FileID")
            if not isinstance(key, str):
                guard.audit["denied_group_dereferences"] += 1
                raise PermissionError("only exact train metadata paths are available")
            path = (group.name.rstrip("/") + "/" + key.lstrip("/")).rstrip("/")
            if path not in guard.allowed_metadata_paths:
                guard.audit["denied_group_dereferences"] += 1
                raise PermissionError("HDF5 group path is outside the selected train metadata allowlist")
            guard.metadata_paths_touched.append(path)
            return guard._group_getitem(group, key)

        def dataset_getitem(dataset, key):
            if not guard._belongs(dataset):
                guard.audit["denied_foreign_file_accesses"] += 1
                guard.audit["denied_value_reads"] += 1
                raise PermissionError("HDF5 dataset belongs to a foreign FileID")
            if not isinstance(key, slice) or key.step not in (None, 1):
                guard.audit["denied_value_reads"] += 1
                raise PermissionError("only one-dimensional unit-step quaternion window slices are allowed")
            try:
                start, stop, step = key.indices(dataset.shape[0])
            except (TypeError, ValueError, OverflowError) as exc:
                guard.audit["denied_value_reads"] += 1
                raise PermissionError("quaternion window slice bounds are invalid") from exc
            path = dataset.name
            allow_key = (path, start, stop)
            if (path.rsplit("/", 1)[-1] != "quaternion" or step != 1 or
                    stop - start != WINDOW_SAMPLES or allow_key not in guard.remaining_reads or
                    guard.remaining_reads[allow_key] <= 0):
                guard.audit["denied_value_reads"] += 1
                raise PermissionError("quaternion value slice is outside the approved train prefix")
            result = guard._dataset_getitem(dataset, key)
            guard.remaining_reads[allow_key] -= 1
            guard.audit["quaternion_slice_calls"] += 1
            guard.audit["quaternion_rows_read"] += int(result.shape[0])
            return result

        def blocked_read_direct(dataset, *args, **kwargs):
            if not guard._belongs(dataset):
                guard.audit["denied_foreign_file_accesses"] += 1
            guard.audit["denied_value_reads"] += 1
            raise PermissionError("read_direct is disabled for this bounded reader")

        def blocked_array(dataset, *args, **kwargs):
            if not guard._belongs(dataset):
                guard.audit["denied_foreign_file_accesses"] += 1
            guard.audit["denied_value_reads"] += 1
            raise PermissionError("whole-dataset array conversion is disabled")

        h5py.Group.__getitem__ = group_getitem
        h5py.Dataset.__getitem__ = dataset_getitem
        h5py.Dataset.read_direct = blocked_read_direct
        h5py.Dataset.__array__ = blocked_array
        return self

    def __exit__(self, exc_type, exc, tb):
        h5py.Group.__getitem__ = self._group_getitem
        h5py.Dataset.__getitem__ = self._dataset_getitem
        h5py.Dataset.read_direct = self._read_direct
        h5py.Dataset.__array__ = self._array


def _validate_quaternion_window(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.dtype != np.dtype("float32") or raw.shape != (WINDOW_SAMPLES, 4):
        raise ValueError("source HDF5 returned an unexpected quaternion window schema")
    if not np.isfinite(raw).all():
        raise ValueError("source quaternion window contains nonfinite values")
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 2e-6:
        raise ValueError("source quaternion window contains nonunit orientations")
    if float(np.min(np.sum(raw[1:] * raw[:-1], axis=1))) < -1e-6:
        raise ValueError("source quaternion window is not hemisphere aligned")
    # Do not repair or renormalize source values. The reader verifies the
    # stored contract and returns two distinct, exact-shape read-only arrays.
    context = np.array(raw[:CONTEXT_SAMPLES], dtype=np.float32, copy=True)
    target = np.array(raw[CONTEXT_SAMPLES:], dtype=np.float32, copy=True)
    context.setflags(write=False)
    target.setflags(write=False)
    return context, target


def _iter_windows_from_open_file(source_file: h5py.File, binding: MetadataBinding,
                                 plan: tuple[PlannedWindow, ...]) -> Iterator[QuaternionWindow]:
    """Synthetic-testable core; production passes only the fixed source FileID."""
    _validate_source_root(source_file, binding)
    datasets, metadata_paths, allowed_reads = _preflight_selected_train_metadata(
        source_file, binding, plan)
    with HDF5ReadGuard(source_file, allowed_metadata_paths=metadata_paths,
                       allowed_reads=allowed_reads) as guard:
        for window in plan:
            path = (f"/users/{window.user_id}/sessions/{window.session_id}/segments/"
                    f"{window.segment_id}/quaternion")
            dataset = datasets.get(path)
            if dataset is None:
                raise PermissionError("authorized train quaternion dataset was not preflighted")
            raw = dataset[window.context_start:window.slice_stop]
            context, target = _validate_quaternion_window(raw)
            yield QuaternionWindow(
                window.evaluation_fold, window.role, window.user_id,
                window.session_id, window.segment_id, window.origin_index,
                context, target,
            )
        if any(count for count in guard.remaining_reads.values()):
            raise PermissionError("the authorized train prefix was not read exactly once")
        if any(guard.audit[key] for key in (
                "denied_value_reads", "denied_group_dereferences", "denied_foreign_file_accesses")):
            raise PermissionError("HDF5 read guard recorded a denied access attempt")


@contextmanager
def _verified_source_file(binding: MetadataBinding):
    """Hash exact source bytes before h5py and retain the same verified handle."""
    current = _file_identity(binding.source_path)
    if current != binding.source_identity:
        raise ValueError("source HDF5 file identity changed after metadata preflight")
    stream = binding.source_path.open("rb")
    try:
        opened = os.fstat(stream.fileno())
        current_path = binding.source_path.stat()
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        path_identity = (current_path.st_dev, current_path.st_ino,
                         current_path.st_size, current_path.st_mtime_ns)
        if identity != binding.source_identity or path_identity != binding.source_identity:
            raise ValueError("source HDF5 inode changed before its byte verification")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != SOURCE_HDF5_SHA256:
            raise ValueError("source HDF5 bytes differ from the locked split reference")
        stream.seek(0)
        with h5py.File(stream, "r") as source_file:
            yield source_file
    finally:
        stream.close()


def iter_approved_quaternion_windows() -> Iterator[QuaternionWindow]:
    """Yield only the fixed, coordinator-approved train prefixes.

    The caller cannot supply paths, hashes, users, split partitions, fold IDs,
    or an ``allow_final`` switch. The function fails before any source HDF5
    byte read/open if fixed metadata, protocol, approval, or file identity is
    wrong. A later standardization pass requires its own separately reviewed
    gate; this reader never fits statistics.
    """
    binding = validate_metadata_bindings()
    approval = load_fixed_approval(binding)
    plan = build_read_plan(binding, approval)
    with _verified_source_file(binding) as source_file:
        yield from _iter_windows_from_open_file(source_file, binding, plan)


def safe_aggregate_template(binding: MetadataBinding, approval: ApprovedAccess) -> dict:
    """Return a public-safe receipt skeleton without participant identifiers."""
    return {
        "schema_version": 1,
        "kind": "q1_train_prefix_reader_access_receipt",
        "authorization_id": approval.authorization_id,
        "protocol_id": approval.protocol_id,
        "source_hdf5_relative_path": SOURCE_HDF5_RELATIVE_PATH,
        "source_hdf5_sha256_reference": SOURCE_HDF5_SHA256,
        "split_sha256": SPLIT_SHA256,
        "semantic_run_id": SEMANTIC_RUN_ID,
        "semantic_config_sha256": SEMANTIC_CONFIG_SHA256,
        "semantic_receipt_sha256": SEMANTIC_RECEIPT_SHA256,
        "origin_inventory_receipt_sha256": binding.origin_receipt_sha256,
        "realized_origin_order_sha256": ORIGIN_ORDER_SHA256,
        "ownership_plan_sha256": binding.ownership_plan_sha256,
        "authorized_prefixes": [
            {"fold_id": item.fold_id, "role": item.role,
             "sampling": "endpoint_inclusive_evenly_spaced_per_user_v1",
             "origins_per_user": item.origins_per_user}
            for item in approval.prefixes
        ],
        "raw_user_session_segment_ids_in_receipt": False,
        "final_development_values_read": 0,
        "feature_target_cache_values_read": 0,
        "model_fits_performed": 0,
        "fitted_statistics_computed": 0,
        "q1_decision_or_global_grant": False,
    }

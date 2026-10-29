"""Root-gated connector from pinned V3 fit10 pools to Vast bundle SegmentInput.

The local authority validates the fixed metadata and signs the resulting bundle.
An operator must explicitly invoke export. A signature confirms bundle integrity;
it does not itself grant data-use permission or authorize a cloud transfer.

The connector prepares whole outer-fit pose segments only. It never prepares
windows, validation/dev/final rows, standardizers, or a training schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Protocol

import numpy as np

from ..data import bundle as _bundle


PURPOSE = _bundle.PURPOSE
ALLOWED_PURPOSES = _bundle.ALLOWED_PURPOSES
ROLE = "outer_fit10"
ALLOWED_FOLDS = (0, 1, 2)
SOURCE_SHA256 = "0986c483c2dc800ff8be1a2a8d2e82d909193f4c3e7c9f7b1eae3e2ea055ea62"
SPLIT_SHA256 = "9d7f342c15231a3f7f61fd8c9b95b75adc6aff641e7071d95a43a3832db49256"
V3_LOADER_SHA256 = "7d1326fdba2da6a47ed20401d2745c7bcc7f5a5f5bd8e4ceb4e120347387eaab"
V3_READER_SHA256 = "2b499f28199f91d0a586fc3611964a16a943c0b206d53eb899ad79a367a32149"
V3_READER_REVIEW_SHA256 = "8c2e569341412a0bc5e74bbaccd51c3c7b95c8466b057418f4e590a2e76be8dd"
MAX_POSE_BYTES = 128 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024
MIN_FREE_RAM_BYTES = 2 * 1024**3
MAX_LOADER_SECONDS = 3600
GRANT_KIND = "vast_fit10_export_grant_v1"
GRANT_NAMESPACE = "vast_training_fit10_export_v1"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
V3_LOADER_PATH = WORKFLOW_ROOT / "staging" / "q1_fit_loader_v3" / "q1_fit_loader.py"
SOURCE_PATH = WORKSPACE_ROOT / "ABC_IUEH" / "data" / "processed" / "alyx.h5"

_V3_MODULE = None

_STATIC_BINDING_KEYS = frozenset({
    "source_asset_sha256",
    "split_sha256",
    "fit_role_inventory_sha256",
    "fit_origin_inventory_sha256",
    "source_loader_sha256",
    "source_authorization_sha256",
})
_SOURCE_BINDING_KEYS = _STATIC_BINDING_KEYS | {"source_read_receipt_sha256"}


class SourceAdapterError(ValueError):
    """A source grant, metadata binding, or pose segment violates this route."""


class SourceAuthorizationError(PermissionError):
    """A root-held capability did not authorize this exact fit10 export."""


@dataclass(frozen=True)
class ExportRequest:
    """Canonical request that the injected root authority must authenticate."""

    grant_namespace: str
    purpose: str
    role: str
    fold_id: int
    source_asset_sha256: str
    split_sha256: str
    source_loader_sha256: str
    source_reader_sha256: str
    source_reader_review_sha256: str
    bundle_module_sha256: str


class RootExportAuthority(Protocol):
    """Methods implemented by the externally authenticated root process.

    Implementations must issue a fresh one-use grant and durable claim for this
    purpose/fold, verify metadata provenance and expiry, and attest the final
    bundle identity. A structural protocol does not authenticate its caller.
    """

    def issue_fit10_export_grant(self, request: ExportRequest) -> object: ...

    def verify_fit10_export_grant(
        self, grant: object, request: ExportRequest
    ) -> Mapping[str, str]: ...

    def finalize_fit10_source_binding(
        self, grant: object, request: ExportRequest, receipt: Mapping[str, Any]
    ) -> Mapping[str, str]: ...

    def attest_bundle_identity(
        self, grant: object, identity_sha256: str
    ) -> Mapping[str, str]: ...


@dataclass
class _GrantState:
    grant: object = field(repr=False)
    request: ExportRequest
    fold_id: int
    binding: Mapping[str, str] | None = None
    verified: bool = False
    opened: bool = False
    complete: bool = False
    receipt: Mapping[str, Any] | None = None
    finalized: bool = False
    expiry_monotonic: float = 0.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SourceAdapterError("pinned V3 loader is not a regular file")
    if path.stat().st_size > max_bytes:
        raise SourceAdapterError("pinned V3 loader exceeds its fixed code-size cap")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_v3_loader():
    """Load the exact reviewed V3 loader bytes without opening its data source."""
    global _V3_MODULE
    if _V3_MODULE is not None:
        if _sha256_file(V3_LOADER_PATH) != V3_LOADER_SHA256:
            raise SourceAuthorizationError("pinned V3 loader changed after import")
        return _V3_MODULE
    if _sha256_file(V3_LOADER_PATH) != V3_LOADER_SHA256:
        raise SourceAuthorizationError("V3 fit10 loader differs from its reviewed pin")
    module_name = "_q1_fit_loader_v3_for_vast_fit10_export"
    spec = importlib.util.spec_from_file_location(module_name, V3_LOADER_PATH)
    if spec is None or spec.loader is None:
        raise SourceAdapterError("cannot load the fixed V3 loader module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if _sha256_file(V3_LOADER_PATH) != V3_LOADER_SHA256:
        sys.modules.pop(module_name, None)
        raise SourceAuthorizationError("V3 loader changed during import")
    if (module.WORKSPACE_ROOT != WORKSPACE_ROOT or
            module.REPO_ROOT != WORKSPACE_ROOT / "ABC_IUEH" or
            module.SOURCE_HDF5_RELATIVE_PATH != "data/processed/alyx.h5" or
            module.SOURCE_HDF5_SHA256 != SOURCE_SHA256 or
            module.SPLIT_SHA256 != SPLIT_SHA256):
        sys.modules.pop(module_name, None)
        raise SourceAuthorizationError("V3 loader constants differ from this fixed route")
    _V3_MODULE = module
    return module


def _require_capability(value: object, label: str) -> None:
    if isinstance(value, (Mapping, list, tuple, set, frozenset,
                          str, bytes, bytearray, int, float, bool)):
        raise SourceAuthorizationError(f"{label} must be an opaque process-local capability")
    if not callable(getattr(value, "assert_live", None)):
        raise SourceAuthorizationError(f"{label} must expose assert_live()")
    value.assert_live()


def _parse_expiry(value: object) -> datetime:
    if type(value) is not str:
        raise SourceAuthorizationError("grant expiry must be a UTC ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceAuthorizationError("grant expiry timestamp is malformed") from exc
    if (parsed.tzinfo is None or parsed.utcoffset() is None or
            parsed.utcoffset().total_seconds() != 0 or parsed <= datetime.now(timezone.utc)):
        raise SourceAuthorizationError("grant expiry is absent, non-UTC, or expired")
    return parsed.astimezone(timezone.utc)


def validate_grant_scope(grant: object, request: ExportRequest) -> datetime:
    """Pure scope validation used before V3 import/source access.

    This validates the envelope only. The root's verifier remains the actual
    authority and must check its fresh, purpose-bound one-use claim.
    """
    _require_capability(grant, "export grant")
    checks = {
        "grant_kind": GRANT_KIND,
        "grant_namespace": GRANT_NAMESPACE,
        "purpose": request.purpose,
        "role": ROLE,
        "fold_id": request.fold_id,
        "source_asset_sha256": SOURCE_SHA256,
        "split_sha256": SPLIT_SHA256,
        "source_loader_sha256": V3_LOADER_SHA256,
        "source_reader_sha256": V3_READER_SHA256,
        "source_reader_review_sha256": V3_READER_REVIEW_SHA256,
    }
    for name, expected in checks.items():
        if getattr(grant, name, None) != expected:
            raise SourceAuthorizationError(f"grant {name} is outside the exact export request")
    if type(getattr(grant, "fold_id", None)) is not int or grant.fold_id not in ALLOWED_FOLDS:
        raise SourceAuthorizationError("grant fold must be exactly 0, 1, or 2")
    for name in ("grant_id", "session_id", "claim_id", "plan_id", "approval_id"):
        value = getattr(grant, name, None)
        if type(value) is not str or not ID_RE.fullmatch(value):
            raise SourceAuthorizationError(f"grant has malformed {name}")
    digest_names = (
        "authorization_sha256", "plan_sha256", "approval_sha256", "recipe_sha256",
        "fit_role_inventory_sha256", "fit_origin_inventory_sha256",
    )
    for name in digest_names:
        value = getattr(grant, name, None)
        if type(value) is not str or not HASH_RE.fullmatch(value):
            raise SourceAuthorizationError(f"grant has malformed {name}")
    if getattr(grant, "one_use", None) is not True:
        raise SourceAuthorizationError("grant is not marked one-use by root")
    if getattr(grant, "binding", None) is None:
        raise SourceAuthorizationError("grant lacks the root-verified V3 metadata binding")
    if (type(getattr(grant, "max_pose_bytes", None)) is not int or
            not 0 < grant.max_pose_bytes <= MAX_POSE_BYTES or
            type(getattr(grant, "max_chunk_bytes", None)) is not int or
            not 16 <= grant.max_chunk_bytes <= MAX_CHUNK_BYTES or
            grant.max_chunk_bytes % 16 != 0 or
            type(getattr(grant, "max_loader_seconds", None)) is not int or
            not 1 <= grant.max_loader_seconds <= MAX_LOADER_SECONDS):
        raise SourceAuthorizationError("grant resource limits exceed the reviewed V3 bounds")
    return _parse_expiry(getattr(grant, "expires_utc", None))


def validate_source_binding(value: object, grant: object, *, receipt_sha256: str | None = None) -> dict[str, str]:
    """Check static grant bindings; a read digest exists only after the read."""
    keys = _STATIC_BINDING_KEYS if receipt_sha256 is None else _SOURCE_BINDING_KEYS
    if not isinstance(value, Mapping) or set(value) != keys:
        raise SourceAuthorizationError("root returned an unexpected source-binding schema")
    binding = {}
    for key in sorted(keys):
        item = value[key]
        if type(item) is not str or not HASH_RE.fullmatch(item):
            raise SourceAuthorizationError(f"source binding {key} is not a SHA-256 digest")
        binding[key] = item
    expected = {
        "source_asset_sha256": SOURCE_SHA256,
        "split_sha256": SPLIT_SHA256,
        "fit_role_inventory_sha256": grant.fit_role_inventory_sha256,
        "fit_origin_inventory_sha256": grant.fit_origin_inventory_sha256,
        "source_loader_sha256": V3_LOADER_SHA256,
        "source_authorization_sha256": grant.authorization_sha256,
    }
    if receipt_sha256 is not None:
        expected["source_read_receipt_sha256"] = receipt_sha256
    if binding != expected:
        raise SourceAuthorizationError("verified source binding differs from the purpose grant or actual receipt")
    return binding


def ordered_pool_segments(rows: Iterable[object], binding: object, segments: Iterable[object]) -> tuple[object, ...]:
    """Prove rank correspondence to the fixed origin map and reorder the pool."""
    by_key = {}
    for segment in segments:
        key = (segment.user_id, segment.session_id, segment.segment_id)
        if key in by_key:
            raise SourceAdapterError("duplicate pool segment")
        by_key[key] = segment
    ordered = []
    previous_stop = -1
    for row in rows:
        key = (row.user_id, row.session_id, row.segment_id)
        segment = by_key.pop(key, None)
        offset, count = int(row.offset_int64), int(row.count)
        if segment is None or offset < previous_stop or count <= 0:
            raise SourceAdapterError("pool segment does not match canonical origin order")
        values = binding.origin_values[offset:offset + count]
        if (len(values) != count or int(values[0]) != 399 or
                int(values[-1]) != 398 + count or
                (count > 1 and not np.array_equal(values[1:], values[:-1] + 1))):
            raise SourceAdapterError("fixed origin map is not the complete consecutive segment range")
        poses = segment.poses
        if (segment.global_origin_offset != offset or segment.origin_count != count or
                not isinstance(poses, np.ndarray) or poses.ndim != 2 or
                poses.shape != (count + 419, 4)):
            raise SourceAdapterError("pool segment identity or sample count differs from origin map")
        ordered.append(segment)
        previous_stop = offset + count
    if by_key or not ordered:
        raise SourceAdapterError("pool contains extra or no segments")
    return tuple(ordered)


def fit_role_inventory_digest(rows: Iterable[object], fold_id: int, role: str) -> str:
    """Hash canonical row membership without returning participant identifiers."""
    if type(fold_id) is not int or fold_id not in ALLOWED_FOLDS or role != ROLE:
        raise SourceAdapterError("role inventory digest is restricted to outer_fit10 folds")
    digest = hashlib.sha256()
    digest.update(f"vast-fit10-role-inventory-v1\0{fold_id}\0{role}\n".encode("utf-8"))
    count = 0
    previous_stop = -1
    seen = set()
    for row in rows:
        identity = (
            str(row.user_id), str(row.session_id), str(row.segment_id),
            int(row.offset_int64), int(row.count), int(row.fold_id),
        )
        if identity[3] < 0 or identity[4] <= 0:
            raise SourceAdapterError("fit row has an invalid origin range")
        if identity[:3] in seen or identity[3] < previous_stop:
            raise SourceAdapterError("fit rows are not in canonical unique origin order")
        seen.add(identity[:3])
        previous_stop = identity[3] + identity[4]
        digest.update(_canonical_json(identity) + b"\n")
        count += 1
    if count == 0:
        raise SourceAdapterError("fit role inventory is empty")
    return digest.hexdigest()


def segment_input_from_pose_segment(segment: object, segment_index: int):
    """Copy one V3 segment into immutable little-endian float32 WXYZ bytes.

    Returning bytes prevents a later owner-backed NumPy `setflags(write=True)`
    from changing the bundle payload after the role checks. No windows are made.
    """
    if type(segment_index) is not int or segment_index < 0:
        raise SourceAdapterError("segment index must be a non-negative integer")
    poses = getattr(segment, "poses", None)
    if (not isinstance(poses, np.ndarray) or poses.dtype != np.dtype("float32") or
            poses.ndim != 2 or poses.shape[1] != 4 or not poses.flags.c_contiguous):
        raise SourceAdapterError("V3 pose segment must be C-contiguous float32[N,4]")
    if poses.shape[0] < _bundle.WINDOW_ROWS:
        raise SourceAdapterError("pose segment is shorter than one fixed 400+20 window")
    sample_count = int(poses.shape[0])
    payload = poses.astype("<f4", copy=False).tobytes(order="C")
    if len(payload) != sample_count * _bundle.ROW_BYTES:
        raise SourceAdapterError("immutable segment byte count differs from its shape")
    return _bundle.SegmentInput(
        segment_index=segment_index,
        sample_count=sample_count,
        float32_wxyz_le=payload,
    )


def _available_physical_memory_bytes() -> int | None:
    if os.name == "nt":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = __import__("ctypes").sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullAvailPhys)
    if sys.platform.startswith("linux"):
        try:
            from ..runner.trainer import available_physical_ram_bytes
            return available_physical_ram_bytes()
        except (OSError, ValueError, RuntimeError):
            return None
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            return None
    return None


def _require_ram(extra_bytes: int) -> None:
    free_bytes = _available_physical_memory_bytes()
    if (type(extra_bytes) is not int or extra_bytes < 0 or free_bytes is None or
            free_bytes < MIN_FREE_RAM_BYTES + extra_bytes):
        raise MemoryError("Vast fit10 export requires 2 GiB free plus bounded pool/copy scratch")


def _grant_expiry_deadline(grant: object) -> float:
    expiry = _parse_expiry(grant.expires_utc)
    remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise SourceAuthorizationError("fit10 export grant expired before source preparation")
    return time.monotonic() + remaining


class V3Fit10SourceRoot:
    """Implements `bundle.TrustedSourceRoot` using an injected root authority.

    Source opening requires an explicit operator invocation and an authority
    implementing the fixed metadata, receipt and signing contract.
    """

    def __init__(self, authority: RootExportAuthority):
        if authority is None:
            raise SourceAuthorizationError("source adapter requires a local export authority")
        self._authority = authority
        self._grants: list[_GrantState] = []
        self._seen_claims: set[tuple[str, str]] = set()

    def _state(self, grant: object) -> _GrantState:
        for state in self._grants:
            if state.grant is grant:
                grant.assert_live()
                if time.monotonic() >= state.expiry_monotonic:
                    raise SourceAuthorizationError("fit10 export lease expired")
                return state
        raise SourceAuthorizationError("grant was not issued by this adapter instance")

    def issue_fit10_export_grant(self, fold_id: int, *, purpose: str) -> object:
        if purpose not in ALLOWED_PURPOSES:
            raise SourceAuthorizationError("unsupported export purpose")
        if type(fold_id) is not int or fold_id not in ALLOWED_FOLDS:
            raise SourceAuthorizationError("only frozen outer-fit folds 0, 1, or 2 are exportable")
        request = ExportRequest(
            grant_namespace=GRANT_NAMESPACE,
            purpose=purpose,
            role=ROLE,
            fold_id=fold_id,
            source_asset_sha256=SOURCE_SHA256,
            split_sha256=SPLIT_SHA256,
            source_loader_sha256=V3_LOADER_SHA256,
            source_reader_sha256=V3_READER_SHA256,
            source_reader_review_sha256=V3_READER_REVIEW_SHA256,
            bundle_module_sha256=_sha256_file(Path(_bundle.__file__)),
        )
        grant = self._authority.issue_fit10_export_grant(request)
        expiry = validate_grant_scope(grant, request)
        claim_key = (grant.session_id, grant.claim_id)
        if claim_key in self._seen_claims:
            raise SourceAuthorizationError("fit10 export claim was already consumed in this process")
        self._seen_claims.add(claim_key)
        self._grants.append(_GrantState(
            grant=grant,
            request=request,
            fold_id=fold_id,
            expiry_monotonic=time.monotonic() + (expiry - datetime.now(timezone.utc)).total_seconds(),
        ))
        return grant

    def verify_fit10_export_grant(self, grant: object) -> Mapping[str, str]:
        state = self._state(grant)
        expiry = validate_grant_scope(grant, state.request)
        if (grant.session_id, grant.claim_id) not in self._seen_claims:
            raise SourceAuthorizationError("grant does not own a fresh adapter claim")
        returned = self._authority.verify_fit10_export_grant(grant, state.request)
        binding = validate_source_binding(returned, grant)
        if state.binding is not None and state.binding != binding:
            raise SourceAuthorizationError("root source binding changed during grant verification")
        state.binding = binding
        state.verified = True
        state.expiry_monotonic = min(
            state.expiry_monotonic,
            time.monotonic() + max(0.0, (expiry - datetime.now(timezone.utc)).total_seconds()),
        )
        return binding

    def open_fit10_segments(self, grant: object) -> Iterable[Any]:
        state = self._state(grant)
        if not state.verified or state.binding is None:
            raise SourceAuthorizationError("grant must be verified before source preparation")
        if state.opened:
            raise SourceAuthorizationError("fit10 grant permits one source open only")
        state.opened = True
        return self._iter_fit10_segments(state)

    def finalize_fit10_source_binding(self, grant: object) -> Mapping[str, str]:
        state = self._state(grant)
        if not state.complete or state.receipt is None or state.binding is None or state.finalized:
            raise SourceAuthorizationError("source receipt is unavailable or already finalized")
        receipt = dict(state.receipt)
        receipt_digest = _sha256(_canonical_json(receipt))
        returned = self._authority.finalize_fit10_source_binding(grant, state.request, receipt)
        binding = validate_source_binding(returned, grant, receipt_sha256=receipt_digest)
        if {key: binding[key] for key in _STATIC_BINDING_KEYS} != state.binding:
            raise SourceAuthorizationError("root changed a static source binding after reading")
        state.finalized = True
        return binding

    def completed_source_evidence(self):
        """Return completed read evidence for the caller to store privately."""
        return self._authority.completed_source_evidence()

    def attest_bundle_identity(self, grant: object, identity_sha256: str) -> Mapping[str, str]:
        state = self._state(grant)
        if not state.finalized:
            raise SourceAuthorizationError("bundle cannot be attested before the actual source receipt is finalized")
        if type(identity_sha256) is not str or not HASH_RE.fullmatch(identity_sha256):
            raise SourceAuthorizationError("bundle identity is not a SHA-256 digest")
        grant.assert_live()
        return self._authority.attest_bundle_identity(grant, identity_sha256)

    def _iter_fit10_segments(self, state: _GrantState):
        grant = state.grant
        loader = _load_pinned_v3_loader()
        reader = loader._load_pinned_reader()
        if (getattr(grant, "source_reader_sha256", None) != V3_READER_SHA256 or
                getattr(grant, "source_reader_review_sha256", None) != V3_READER_REVIEW_SHA256):
            raise SourceAuthorizationError("grant does not bind the reviewed bounded source reader")
        binding = grant.binding
        if type(binding) is not reader.MetadataBinding:
            raise SourceAuthorizationError("grant binding is not the exact pinned reader metadata type")
        if (Path(binding.source_path) != SOURCE_PATH or
                binding.source_identity != reader._file_identity(SOURCE_PATH)):
            raise SourceAuthorizationError("metadata binding does not identify the pinned source asset")

        fold_id = state.fold_id
        fit_users = tuple(loader._role_users(binding, fold_id, ROLE))
        if len(fit_users) != 10 or len(set(fit_users)) != 10:
            raise SourceAuthorizationError("frozen role does not resolve to exactly ten fit users")
        rows = loader._canonical_fit_rows(binding, fold_id, ROLE)
        row_users = {row.user_id for row in rows}
        if row_users != set(fit_users):
            raise SourceAuthorizationError("canonical source rows escape the exact fit10 membership")
        role_digest = fit_role_inventory_digest(rows, fold_id, ROLE)
        origin_digest = loader._pool_origin_digest(rows, binding, fold_id, ROLE)
        if (role_digest != grant.fit_role_inventory_sha256 or
                origin_digest != grant.fit_origin_inventory_sha256):
            raise SourceAuthorizationError("V3 role/origin map differs from the root grant binding")
        total_origins = sum(int(row.count) for row in rows)
        pose_bytes = sum((int(row.count) + 419) * 4 * np.dtype(np.float32).itemsize for row in rows)
        if (total_origins <= 0 or pose_bytes > grant.max_pose_bytes or
                pose_bytes > loader.MAX_POSE_ARRAY_BYTES or pose_bytes > _bundle.MAX_PAYLOAD_BYTES):
            raise MemoryError("outer-fit pose segments exceed the new export grant's cap")

        # Reserve the existing two-GiB free-memory floor and enough room for
        # source pose arrays, up to two adjacent immutable segment copies,
        # bounded V3 read scratch, and metadata overhead before any source open.
        preflight_extra = 2 * pose_bytes + 8 * grant.max_chunk_bytes + 64 * 1024**2
        _require_ram(preflight_extra)
        if time.monotonic() >= state.expiry_monotonic:
            raise SourceAuthorizationError("fit10 grant expired before source open")

        access = loader.FitPoolPlan(
            plan_id=grant.plan_id,
            purpose=state.request.purpose,
            role_mode=ROLE,
            fold_ids=(fold_id,),
            seed_values=(),
            arm_ids=(),
            standardizer_seed=0,
            windows_per_schedule=0,
            recipe_sha256=grant.recipe_sha256,
            max_pose_bytes=grant.max_pose_bytes,
            max_chunk_bytes=grant.max_chunk_bytes,
            max_loader_seconds=grant.max_loader_seconds,
            plan_sha256=grant.plan_sha256,
            approval_id=grant.approval_id,
            approval_sha256=grant.approval_sha256,
            independent_review_sha256=V3_LOADER_SHA256,
            numpy_version=np.__version__,
            reviewed_artifacts={
                "v3_loader_sha256": V3_LOADER_SHA256,
                "v3_reader_sha256": V3_READER_SHA256,
                "v3_reader_review_sha256": V3_READER_REVIEW_SHA256,
                "bundle_module_sha256": state.request.bundle_module_sha256,
            },
            fitting_operations_allowed=("fit10_pose_segment_bundle_export_only",),
            device="CPU",
            cuda_contexts=0,
            gpu_updates=0,
            optimizer_updates=0,
        )
        pool_digest = origin_digest
        prepared = (rows, total_origins, pool_digest, (), None)
        loader_deadline = min(
            time.monotonic() + grant.max_loader_seconds,
            state.expiry_monotonic,
        )
        pool = None
        try:
            with loader._verified_fit_source_file(reader, binding, loader_deadline) as source_file:
                pool = loader._load_fit_pool_from_open_file(
                    source_file, binding, access, fold_id, prepared,
                    source_hash_verified=True,
                    deadline=loader_deadline,
                )
            if (pool.fold_id != fold_id or pool.role_mode != ROLE or
                    pool.source_hash_verified is not True or
                    pool._audit.denied_reads != 0 or
                    pool._audit.fit_role_users != 10 or
                    pool.origin_pool_sha256 != origin_digest or
                    len(pool._segments) != len(rows)):
                raise SourceAuthorizationError("V3 returned a pool outside the verified fit10 role")
            receipt = pool._audit.safe_aggregate(access, fold_id, total_origins)
            receipt["fit_origin_pool_identity_sha256"] = pool.origin_pool_sha256
            receipt["source_loader_sha256"] = V3_LOADER_SHA256
            receipt["source_reader_sha256"] = V3_READER_SHA256
            receipt["source_reader_review_sha256"] = V3_READER_REVIEW_SHA256
            ordered_segments = ordered_pool_segments(rows, binding, pool._segments)

            expected_pose_bytes = pose_bytes
            actual_pose_bytes = sum(int(item.poses.nbytes) for item in pool._segments)
            if actual_pose_bytes != expected_pose_bytes:
                raise SourceAdapterError("V3 pose arrays do not match canonical fit origin row lengths")
            for index, segment in enumerate(ordered_segments):
                grant.assert_live()
                if time.monotonic() >= state.expiry_monotonic:
                    raise SourceAuthorizationError("fit10 export grant expired during segment streaming")
                _require_ram(2 * max(int(segment.poses.nbytes), 1) + 8 * grant.max_chunk_bytes)
                yield segment_input_from_pose_segment(segment, index)
            grant.assert_live()
            state.receipt = receipt
            state.complete = True
        finally:
            if pool is not None:
                pool._segments = ()


def get_trusted_source_root():
    """Return (root, public key, key ID) for a fresh local export session."""
    from .local_authority import create_local_source_root
    return create_local_source_root()

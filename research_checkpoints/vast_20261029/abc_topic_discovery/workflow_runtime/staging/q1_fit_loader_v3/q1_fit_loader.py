"""Coordinator-gated lazy batches over a bounded outer-fit quaternion pool.

This module is an implementation draft only. It has no CLI, trainer, GPU path,
or arbitrary data-path input. Production reads require fixed, independently
reviewed plan and approval files that are intentionally absent at this stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import h5py
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = WORKSPACE_ROOT / "ABC_IUEH"
WORKFLOW_ROOT = WORKSPACE_ROOT / "abc_topic_discovery" / "workflow_runtime"
STAGING_ROOT = Path(__file__).resolve().parent
READER_PATH = (WORKFLOW_ROOT / "staging" / "q1_train_reader" /
               "q1_train_reader.py")
READER_REVIEW_PATH = (WORKFLOW_ROOT / "reviews" /
                      "q1_train_reader_independent_20261029" / "review.md")
PLAN_PATH = (REPO_ROOT / "data" / "research_controls" / "q1_fit_loader_v3" /
             "fit_pool_plan.json")
APPROVAL_PATH = (REPO_ROOT / "data" / "research_controls" / "q1_fit_loader_v3" /
                 "fit_pool_approval.json")
PLAN_RELATIVE_PATH = "data/research_controls/q1_fit_loader_v3/fit_pool_plan.json"
APPROVAL_RELATIVE_PATH = "data/research_controls/q1_fit_loader_v3/fit_pool_approval.json"
CPU_CONSUMER_PATH = (WORKFLOW_ROOT / "staging" / "q1_cpu_preparation_v3" /
                     "cpu_preparation.py")

# Independent bounded-reader review pins. Verify code bytes before importing it.
PINNED_READER_SHA256 = "2b499f28199f91d0a586fc3611964a16a943c0b206d53eb899ad79a367a32149"
PINNED_READER_REVIEW_SHA256 = "8c2e569341412a0bc5e74bbaccd51c3c7b95c8466b057418f4e590a2e76be8dd"
SOURCE_HDF5_RELATIVE_PATH = "data/processed/alyx.h5"
SOURCE_HDF5_SHA256 = "0986c483c2dc800ff8be1a2a8d2e82d909193f4c3e7c9f7b1eae3e2ea055ea62"
SPLIT_SHA256 = "9d7f342c15231a3f7f61fd8c9b95b75adc6aff641e7071d95a43a3832db49256"
ORIGIN_ORDER_SHA256 = "e9af1133a1aa32b27f5fffc6526f23c52d2c3d0e55a5a7447a6330ba994234e3"

# Narrow compatibility bridge for the known legacy source root. These are
# authenticated fixed metadata identities, not caller-supplied pins.
LEGACY_INVENTORY_SHA256 = "5e0445d6fed5586bab3a0edfddb201259ab5cd70679df0cb4866492044111bb0"
LEGACY_RECEIPT_SHA256 = "d7bff10d5e4a1d08b60f27aa2570a8dbe1719b123464099642ee1f689fb55e16"
VERSIONED_RECEIPT_SHA256 = "2d5355f7447f3b395f492d1e3b9e2e8139681ef3609596fd82a5b1e95960a0fa"
METADATA_BRIDGE_SHA256 = "43927785270c5551a614831265aed7e4cd1c4a11b0d31489d92252e803c4e0fd"
SEMANTIC_CONFIG_SHA256 = "fb402662f474184c80ee0834381e9f9ff4140e5c63d2ec6b4d03ba04695401de"
SEMANTIC_RECEIPT_SHA256 = "38093d32ec39f27a48de49ad1f1a78b87595c526328dc31c1a536e17df540238"
PIPELINE_CONFIG_SHA256 = "1b3bb05567fa5c8fe94a4b89422edf969f2ba4998dbdabc48da8617cb2553f3b"
ARC_CONFIG_SHA256 = "1ee9d2fff488ffa6952ce864177003706761b205901afddc69bca6951b26e753"
SOURCE_HELPER_SHA256 = "157dd0d9c094b39f04e92322f0f4f862ee2aced6cd38aa97f63ce5e437726cc4"
SEMANTIC_REBIND_SHA256 = "ba85608f8ea1490dd5474c600374da4b5874f3ee74dbb3df9c8cfe48cfe9ce26"
SOURCE_PATH_EVIDENCE_SHA256 = "dcb3ebd1e28e8289496235fc1252b40b97a7f233d4f31b867dcccbc41e1e3f77"
LEGACY_RECORD_DIGEST_SHA256 = "284bb0b566399ece3f62dd7db7572eff23cada3cf8e1ca97cf707e16e2fe2bcd"
SOURCE_INVENTORY_SEMANTIC_ID = "b71a5e661d03abebc9471619cc8a049d7067b1891b539d74fb840c57adcd3daf"

CONTEXT_SAMPLES = 400
HORIZON_SAMPLES = 20
WINDOW_SAMPLES = CONTEXT_SAMPLES + HORIZON_SAMPLES
FIT_POOL_ROLE_COUNTS = {"outer_fit10": 10, "inner_fit8_diagnostic": 8}
TRAINING_SAMPLER = "numpy_pcg64_global_uniform_with_replacement_v1"
STANDARDIZER_SAMPLER = "numpy_pcg64_global_uniform_without_replacement_v1"
SAMPLER_VERSION = "numpy_default_rng_seedsequence_training_seed_fold_v2"
TRAINING_SEEDS = (20261026, 20261027)
CANONICAL_ARMS = (
    "raw_vector", "derivative_vector", "same_feature_vector",
    "fixed_weight", "learned_weight",
)
STANDARDIZER_SEED = 20261028
MAX_POSE_ARRAY_BYTES = 128 * 1024 * 1024
MAX_LOGICAL_READ_BYTES = 1024 * 1024
MAX_BATCH_SIZE = 128
MAX_WINDOWS_PER_SCHEDULE = 32_768
MAX_TOTAL_TRAINING_WINDOWS = 3 * 2 * 5 * MAX_WINDOWS_PER_SCHEDULE
MAX_LOADER_SECONDS = 3600
STANDARDIZER_ORIGINS_PER_FOLD = 65_536
MAX_JSON_BYTES = 1024 * 1024
MAX_LAUNCH_ENVELOPE_BYTES = 16 * 1024
MAX_LAUNCH_TTL_SECONDS = 900
MAX_ATTEMPT_LOCK_BYTES = 16 * 1024
TRUSTED_LAUNCH_ENV = "ABC_Q1_TRUSTED_CPU_PREP_LAUNCH"
LAUNCH_LOCK_ROOT = (REPO_ROOT / "data" / "research_controls" /
                    "q1_fit_loader" / "launch_attempt_locks")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PLAN_ID_RE = re.compile(r"^q1-fit-pool-plan-[a-z0-9][a-z0-9._-]{0,80}$")
AUTH_ID_RE = re.compile(r"^q1-fit-pool-approval-[a-z0-9][a-z0-9._-]{0,80}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_reader_module = None
_launch_use_state: dict[str, dict] = {}
_seen_launch_ids: dict[str, str] = {}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> str:
    path = Path(path)
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError("reviewed artifact exceeds its fixed size bound")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_pinned_reader():
    """Verify the immutable helper bytes before importing or reusing its module."""
    global _reader_module
    absolute = Path(os.path.abspath(READER_PATH))
    if absolute != Path(os.path.abspath(
            WORKFLOW_ROOT / "staging" / "q1_train_reader" / "q1_train_reader.py")):
        raise ValueError("bounded reader path is not the fixed reviewed path")
    if not absolute.is_file() or absolute.is_symlink():
        raise FileNotFoundError("fixed bounded reader source is unavailable")
    for component in absolute.parents:
        if component.is_symlink():
            raise ValueError("bounded reader source path contains a symlink component")
    if Path(os.path.abspath(absolute.resolve(strict=True))) != absolute:
        raise ValueError("bounded reader source path does not retain its fixed spelling")
    digest = _sha256_file(absolute)
    if digest != PINNED_READER_SHA256:
        raise PermissionError("bounded reader bytes differ from the independently accepted review pin")
    if (not READER_REVIEW_PATH.is_file() or
            _sha256_file(READER_REVIEW_PATH) != PINNED_READER_REVIEW_SHA256):
        raise PermissionError("bounded reader independent-review bytes differ from the accepted pin")
    if _reader_module is not None:
        return _reader_module
    module_name = "_q1_train_reader_pinned_20261029"
    spec = importlib.util.spec_from_file_location(module_name, absolute)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the pinned bounded reader helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _reader_module = module
    return module


def _strict_json(raw: bytes, label: str) -> dict:
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds its metadata size limit")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate object key")
            result[key] = value
        return result

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite number")
        return value

    def reject_constant(token):
        raise ValueError(f"{label} contains a nonstandard number")

    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                         parse_constant=reject_constant, parse_float=finite_float)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is malformed JSON") from exc
    if type(doc) is not dict:
        raise ValueError(f"{label} root must be an object")
    return doc


def _strict_int(value, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{label} must be an integer at least {minimum}{suffix}")
    return value


def _require_sha(value, label: str) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _fixed_paths() -> dict[str, Path]:
    return {"plan": PLAN_PATH, "approval": APPROVAL_PATH}


def _preflight_plan_paths(reader) -> dict[str, Path]:
    paths = _fixed_paths()
    if paths != {"plan": PLAN_PATH, "approval": APPROVAL_PATH}:
        raise ValueError("fit-pool plan paths differ from their fixed private locations")
    for role, path in paths.items():
        absolute = Path(os.path.abspath(path))
        for component in (absolute, *absolute.parents):
            if component.is_symlink():
                raise ValueError(f"fixed {role} path contains a symlink component")
        if not absolute.exists() or not absolute.is_file():
            raise PermissionError(f"fixed fit-pool {role} metadata is absent; source HDF5 remains closed")
        if absolute.stat().st_size > MAX_JSON_BYTES:
            raise ValueError(f"fixed fit-pool {role} metadata exceeds its size limit")
    sources = reader.origin_inventory._import_sources()
    sources._reject_recording_aliases(list(paths.items()), REPO_ROOT / "data")
    semantic_parent = WORKFLOW_ROOT / "staging"
    if str(semantic_parent) not in sys.path:
        sys.path.insert(0, str(semantic_parent))
    from semantic_rebind import rebind as semantic_guard
    protected = semantic_guard._protected_artifact_identities()
    for role, path in paths.items():
        st = path.stat()
        if (st.st_dev, st.st_ino) in protected:
            raise ValueError(f"fixed fit-pool {role} metadata aliases a protected payload")
    return paths


def _read_fixed_docs(reader) -> tuple[bytes, dict, bytes, dict]:
    paths = _preflight_plan_paths(reader)
    plan_bytes = paths["plan"].read_bytes()
    approval_bytes = paths["approval"].read_bytes()
    return (plan_bytes, _strict_json(plan_bytes, "fixed fit-pool plan"),
            approval_bytes, _strict_json(approval_bytes, "fixed fit-pool approval"))


def _current_artifact_hashes(reader) -> dict[str, str]:
    return {
        "fit_loader_code_sha256": _sha256_file(Path(__file__)),
        "fit_loader_test_sha256": _sha256_file(STAGING_ROOT / "test_q1_fit_loader.py"),
        "fit_loader_readme_sha256": _sha256_file(STAGING_ROOT / "README.md"),
        "bounded_reader_code_sha256": PINNED_READER_SHA256,
        "bounded_reader_review_sha256": PINNED_READER_REVIEW_SHA256,
        "source_path_evidence_sha256": reader.SOURCE_PATH_EVIDENCE_SHA256,
    }


@dataclass(frozen=True)
class FitPoolPlan:
    plan_id: str
    purpose: str
    role_mode: str
    fold_ids: tuple[int, ...]
    seed_values: tuple[int, ...]
    arm_ids: tuple[str, ...]
    standardizer_seed: int
    windows_per_schedule: int
    recipe_sha256: str
    max_pose_bytes: int
    max_chunk_bytes: int
    max_loader_seconds: int
    plan_sha256: str
    approval_id: str
    approval_sha256: str
    independent_review_sha256: str
    numpy_version: str
    reviewed_artifacts: dict[str, str] = field(repr=False)
    fitting_operations_allowed: tuple[str, ...] = ()
    device: str = "CPU"
    cuda_contexts: int = 0
    gpu_updates: int = 0
    optimizer_updates: int = 0
    cpu_consumer_source_sha256: str | None = None
    cpu_max_wall_seconds: int = 0
    cpu_max_rss_bytes: int = 0
    cpu_min_free_memory_bytes: int = 0
    cpu_min_free_disk_bytes: int = 0


@dataclass(frozen=True)
class FitPoolPreflight:
    """ID-free validated plan summary available without opening source values."""
    fold_id: int
    purpose: str
    role_mode: str
    device: str
    cuda_contexts: int
    optimizer_updates: int
    gpu_updates: int
    fitting_operations_allowed: tuple[str, ...]
    source_hdf5_relative_path: str
    source_hdf5_sha256_reference: str
    plan_sha256: str
    approval_sha256: str
    recipe_sha256: str
    cpu_consumer_source_sha256: str
    standardizer_seed: int
    standardizer_origin_count: int
    standardizer_schedule_sha256: str
    standardizer_origin_identity_sha256: str
    max_pose_bytes: int
    max_logical_read_bytes: int
    max_loader_seconds: int
    cpu_max_wall_seconds: int
    cpu_max_rss_bytes: int
    cpu_min_free_memory_bytes: int
    cpu_min_free_disk_bytes: int


@dataclass(frozen=True)
class PreparedSchedule:
    fold_id: int
    kind: str
    seed: int
    arm_id: str | None
    ranks: np.ndarray = field(repr=False)
    schedule_sha256: str
    origin_identity_sha256: str


@dataclass(frozen=True)
class PoseSegment:
    user_id: str = field(repr=False)
    session_id: str = field(repr=False)
    segment_id: str = field(repr=False)
    global_origin_offset: int
    origin_count: int
    poses: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class FitBatch:
    fold_id: int
    role_mode: str
    schedule_kind: str
    seed: int
    arm_id: str | None
    schedule_sha256: str
    origin_identity_sha256: str
    first_schedule_position: int
    global_origin_offsets: np.ndarray = field(repr=False)
    actual_origin_indices: np.ndarray = field(repr=False)
    contexts: np.ndarray = field(repr=False)
    targets20: np.ndarray | None = field(repr=False)

    @property
    def model_inputs(self) -> np.ndarray:
        """The only feature tensor exposed to a model: past context only."""
        return self.contexts

    @property
    def endpoint20(self) -> np.ndarray:
        if self.targets20 is None:
            raise PermissionError("CPU-preparation batches do not expose forecast targets")
        result = self.targets20[:, -1, :]
        result.setflags(write=False)
        return result

    def clone_contexts_for_augmentation(self) -> np.ndarray:
        """Return a writable copy; the immutable source pool stays unchanged."""
        return np.array(self.contexts, dtype=np.float32, copy=True)


@dataclass
class ReadAudit:
    file_id: int
    fit_role_users: int
    metadata_paths: list[str] = field(default_factory=list, repr=False)
    value_rows_by_dataset: dict[str, int] = field(default_factory=dict, repr=False)
    dataset_slice_calls: int = 0
    max_logical_slice_bytes: int = 0
    denied_reads: int = 0
    pose_array_bytes: int = 0
    source_hash_verified: bool = False
    source_root_metadata_binding_mode: str = "unknown"
    elapsed_loader_seconds: float = 0.0

    def safe_aggregate(self, plan: FitPoolPlan, fold_id: int,
                       pool_origin_count: int) -> dict:
        quaternion_rows = sum(rows for path, rows in self.value_rows_by_dataset.items()
                              if path.endswith("/quaternion"))
        timestamp_rows = sum(rows for path, rows in self.value_rows_by_dataset.items()
                             if path.endswith("/timestamps"))
        return {
            "schema_version": 1,
            "kind": "q1_fit_pool_loader_aggregate_receipt",
            "plan_id": plan.plan_id,
            "authorization_id": plan.approval_id,
            "fold_id": fold_id,
            "purpose": plan.purpose,
            "role_mode": plan.role_mode,
            "fit_user_count": self.fit_role_users,
            "fit_origin_count": pool_origin_count,
            "source_hdf5_relative_path": SOURCE_HDF5_RELATIVE_PATH,
            "source_hdf5_sha256_reference": SOURCE_HDF5_SHA256,
            "source_hash_verified": self.source_hash_verified,
            "source_root_metadata_binding_mode": self.source_root_metadata_binding_mode,
            "pose_array_bytes": self.pose_array_bytes,
            "max_logical_slice_bytes": self.max_logical_slice_bytes,
            "hdf5_dataset_slice_calls": self.dataset_slice_calls,
            "quaternion_rows_loaded": quaternion_rows,
            "timestamp_rows_validated": timestamp_rows,
            "hdf5_denied_reads": self.denied_reads,
            "plan_sha256": plan.plan_sha256,
            "approval_sha256": plan.approval_sha256,
            "recipe_sha256": plan.recipe_sha256,
            "q1_grant": False,
            "model_fits_performed_by_loader": 0,
            "fitting_operations_allowed": list(plan.fitting_operations_allowed),
            "device": plan.device,
            "cuda_contexts": plan.cuda_contexts,
            "optimizer_updates": plan.optimizer_updates,
            "cpu_consumer_source_sha256": plan.cpu_consumer_source_sha256,
            "raw_user_session_segment_ids_in_receipt": False,
        }


def _expected_data_bindings(reader, binding) -> dict:
    pipeline_sha = reader.origin_inventory.canonical_json_sha256(
        binding.parent.metadata["pipeline_config"])
    return {
        "source_partition": "train",
        "source_hdf5_relative_path": SOURCE_HDF5_RELATIVE_PATH,
        "source_hdf5_sha256_reference": SOURCE_HDF5_SHA256,
        "split_sha256": SPLIT_SHA256,
        "fold_digest": binding.parent.fold_digest,
        "fold_ids": [0, 1, 2],
        "train_user_count": 15,
        "outer_fit_users_per_fold": 10,
        "outer_validation_users_per_fold": 5,
        "inner_fit_users_per_fold": 8,
        "inner_selector_users_per_fold": 2,
        "origin_inventory_receipt_sha256": binding.origin_receipt_sha256,
        "origin_map_sha256": binding.origin_map_sha256,
        "origin_index_binary_sha256": binding.origin_binary_sha256,
        "origin_order_sha256": ORIGIN_ORDER_SHA256,
        "source_path_evidence_sha256": binding.source_path_evidence_sha256,
        "fit_segment_value_dataset_names": ["timestamps", "quaternion"],
        "semantic_run_id": reader.SEMANTIC_RUN_ID,
        "semantic_config_sha256": reader.SEMANTIC_CONFIG_SHA256,
        "semantic_receipt_sha256": reader.SEMANTIC_RECEIPT_SHA256,
        "arc_config_sha256": reader.ARC_CONFIG_SHA256,
        "pipeline_config_sha256": pipeline_sha,
        "ownership_plan_sha256": binding.ownership_plan_sha256,
        "context_samples": CONTEXT_SAMPLES,
        "horizon_samples": HORIZON_SAMPLES,
        "shift": 1,
    }


def _validate_plan_and_approval(reader, binding, plan_doc: dict, plan_raw: bytes,
                                approval_doc: dict, approval_raw: bytes) -> FitPoolPlan:
    if (type(plan_doc.get("schema_version")) is not int or plan_doc.get("schema_version") != 1 or
            plan_doc.get("kind") != "q1_fit_pool_plan"):
        raise PermissionError("fit-pool plan schema or kind is invalid")
    plan_id = plan_doc.get("plan_id")
    if not isinstance(plan_id, str) or not PLAN_ID_RE.fullmatch(plan_id):
        raise ValueError("fit-pool plan_id is malformed")
    purpose = plan_doc.get("purpose")
    role_mode = plan_doc.get("role_mode")
    purpose_role = {
        "train_only_timing_prefix": "outer_fit10",
        "inner_fit_pool_diagnostic": "inner_fit8_diagnostic",
        "fit_only_cpu_preparation": "outer_fit10",
    }
    if purpose not in purpose_role or purpose_role[purpose] != role_mode:
        raise PermissionError("plan purpose does not authorize its fit-only role mode")
    cpu_prep = purpose == "fit_only_cpu_preparation"
    if plan_doc.get("status") != ("FROZEN_BEFORE_CPU_PREPARATION" if cpu_prep
                                   else "FROZEN_BEFORE_PREFIX"):
        raise PermissionError("fit-pool plan is not frozen for its separately locked purpose")
    base_keys = {
        "schema_version", "kind", "status", "plan_id", "purpose", "role_mode",
        "data_bindings", "sampling", "resource_policy", "recipe", "recipe_sha256",
        "reviewed_artifacts",
    }
    cpu_keys = {
        "cpu_consumer_source_sha256", "fitting_operations_allowed", "device_policy",
        "cpu_resource_policy",
    }
    if set(plan_doc) != (base_keys | cpu_keys if cpu_prep else base_keys):
        raise ValueError("fit-pool plan has an unexpected schema")
    if not reader.origin_inventory._exact_equal(
            plan_doc.get("data_bindings"), _expected_data_bindings(reader, binding)):
        raise PermissionError("fit-pool plan source, split, fold, origin, config, or role lock differs")

    artifacts = _current_artifact_hashes(reader)
    if not reader.origin_inventory._exact_equal(plan_doc.get("reviewed_artifacts"), artifacts):
        raise PermissionError("fit-pool plan does not bind the exact loader, tests, README, and reader review")

    sampling = plan_doc.get("sampling")
    training_sampling_keys = {
        "training_sampler", "sampler_version", "numpy_version", "fold_ids",
        "seed_values", "arm_ids", "windows_per_schedule", "standardizer_sampler",
        "standardizer_seed", "standardizer_origins_per_fold", "batch_size",
    }
    cpu_sampling_keys = {
        "numpy_version", "fold_ids", "standardizer_sampler", "standardizer_seed",
        "standardizer_origins_per_fold", "batch_size",
    }
    expected_sampling_keys = cpu_sampling_keys if cpu_prep else training_sampling_keys
    if type(sampling) is not dict or set(sampling) != expected_sampling_keys:
        raise ValueError("fit-pool sampling plan has an unexpected schema")
    if sampling.get("standardizer_sampler") != STANDARDIZER_SAMPLER:
        raise PermissionError("fit-pool sampling algorithm differs from the reviewed sampler")
    if not cpu_prep and (sampling.get("training_sampler") != TRAINING_SAMPLER or
                         sampling.get("sampler_version") != SAMPLER_VERSION):
        raise PermissionError("fit-pool training sampler differs from the reviewed sampler")
    if sampling.get("numpy_version") != np.__version__:
        raise PermissionError("fit-pool NumPy version differs from the frozen sampler implementation")
    fold_ids = sampling.get("fold_ids")
    if (type(fold_ids) is not list or fold_ids != [0, 1, 2] or
            any(type(fold) is not int for fold in fold_ids)):
        raise PermissionError("fit-pool plan must lock all three canonical outer folds")
    seeds = sampling.get("seed_values")
    arms = sampling.get("arm_ids")
    if cpu_prep:
        if seeds is not None or arms is not None or "windows_per_schedule" in sampling:
            raise PermissionError("CPU preparation cannot bind training seeds, arms, or schedules")
        seeds_tuple: tuple[int, ...] = ()
        arms_tuple: tuple[str, ...] = ()
        windows = 0
    else:
        if (type(seeds) is not list or tuple(seeds) != TRAINING_SEEDS or
                any(type(seed) is not int for seed in seeds)):
            raise ValueError("timing prefix requires the two exact protocol training seeds")
        if (type(arms) is not list or tuple(arms) != CANONICAL_ARMS or
                any(type(arm) is not str for arm in arms)):
            raise ValueError("timing prefix requires the five canonical arms in fixed order")
        seeds_tuple = tuple(seeds)
        arms_tuple = tuple(arms)
        windows = _strict_int(sampling.get("windows_per_schedule"), "windows_per_schedule",
                              minimum=1, maximum=MAX_WINDOWS_PER_SCHEDULE)
    standardizer_seed = _strict_int(sampling.get("standardizer_seed"), "standardizer_seed",
                                    maximum=2**32 - 1)
    if standardizer_seed != STANDARDIZER_SEED:
        raise PermissionError("standardizer seed differs from the frozen protocol seed")
    if sampling.get("standardizer_origins_per_fold") != STANDARDIZER_ORIGINS_PER_FOLD or type(
            sampling.get("standardizer_origins_per_fold")) is not int:
        raise PermissionError("standardizer schedule must contain exactly 65,536 distinct fit origins per fold")
    if sampling.get("batch_size") != MAX_BATCH_SIZE or type(sampling.get("batch_size")) is not int:
        raise PermissionError("fit-pool batch size must be the fixed 128-window bound")

    resources = plan_doc.get("resource_policy")
    if type(resources) is not dict or set(resources) != {
            "max_pose_array_bytes", "max_logical_read_bytes", "max_batch_size",
            "max_windows_per_schedule", "max_total_training_windows", "max_loader_seconds"}:
        raise ValueError("fit-pool resource policy has an unexpected schema")
    max_pose_bytes = _strict_int(resources.get("max_pose_array_bytes"), "max_pose_array_bytes",
                                 minimum=1, maximum=MAX_POSE_ARRAY_BYTES)
    max_chunk_bytes = _strict_int(resources.get("max_logical_read_bytes"), "max_logical_read_bytes",
                                  minimum=16, maximum=MAX_LOGICAL_READ_BYTES)
    if max_chunk_bytes % 16:
        raise ValueError("logical read bound must align to complete float32 quaternion rows")
    if resources.get("max_batch_size") != MAX_BATCH_SIZE or type(
            resources.get("max_batch_size")) is not int:
        raise PermissionError("resource policy exceeds or changes the fixed batch-size bound")
    expected_schedule_cap = 0 if cpu_prep else windows
    if resources.get("max_windows_per_schedule") != expected_schedule_cap or type(
            resources.get("max_windows_per_schedule")) is not int:
        raise PermissionError("resource policy and sampler window cap differ")
    expected_total = 0 if cpu_prep else 3 * len(seeds_tuple) * len(arms_tuple) * windows
    if (resources.get("max_total_training_windows") != expected_total or
            type(resources.get("max_total_training_windows")) is not int or
            expected_total > MAX_TOTAL_TRAINING_WINDOWS):
        raise PermissionError("fit-pool total training-window cap is invalid")
    max_seconds = _strict_int(resources.get("max_loader_seconds"), "max_loader_seconds",
                              minimum=1, maximum=MAX_LOADER_SECONDS)

    fitting_operations: tuple[str, ...] = ()
    device = "CPU"
    optimizer_updates = 0
    cpu_consumer_sha: str | None = None
    cpu_max_wall_seconds = 0
    cpu_max_rss_bytes = 0
    cpu_min_free_memory_bytes = 0
    cpu_min_free_disk_bytes = 0
    if cpu_prep:
        fitting = plan_doc.get("fitting_operations_allowed")
        if type(fitting) is not list or fitting != ["population_feature_standardizer"]:
            raise PermissionError("CPU preparation only permits the population feature standardizer")
        fitting_operations = tuple(fitting)
        device_policy = plan_doc.get("device_policy")
        if (type(device_policy) is not dict or set(device_policy) != {
                "device", "cuda_contexts", "gpu_updates", "optimizer_updates"} or
                device_policy.get("device") != "CPU" or
                type(device_policy.get("cuda_contexts")) is not int or
                device_policy.get("cuda_contexts") != 0 or
                type(device_policy.get("gpu_updates")) is not int or
                device_policy.get("gpu_updates") != 0 or
                type(device_policy.get("optimizer_updates")) is not int or
                device_policy.get("optimizer_updates") != 0):
            raise PermissionError("CPU preparation must lock CPU-only execution and zero updates")
        cpu_consumer_sha = _require_sha(
            plan_doc.get("cpu_consumer_source_sha256"), "cpu_consumer_source_sha256")
        consumer_path = Path(os.path.abspath(CPU_CONSUMER_PATH))
        for component in (consumer_path, *consumer_path.parents):
            if component.is_symlink():
                raise PermissionError("CPU preparation consumer source path contains a symlink")
        if not consumer_path.is_file():
            raise PermissionError("fixed CPU preparation consumer source is absent")
        source_helpers = reader.origin_inventory._import_sources()
        source_helpers._reject_recording_aliases(
            [("CPU preparation consumer", consumer_path)], REPO_ROOT / "data")
        consumer_stat = consumer_path.stat()
        semantic_parent = reader.origin_inventory.STAGING_ROOT.parent
        if str(semantic_parent) not in sys.path:
            sys.path.insert(0, str(semantic_parent))
        from semantic_rebind import rebind as semantic_guard
        if (consumer_stat.st_dev, consumer_stat.st_ino) in semantic_guard._protected_artifact_identities():
            raise PermissionError("CPU preparation consumer aliases a protected data/model payload")
        if _sha256_file(consumer_path) != cpu_consumer_sha:
            raise PermissionError("CPU preparation consumer bytes differ from the locked source hash")
        cpu_resources = plan_doc.get("cpu_resource_policy")
        cpu_resource_limits = {
            "max_wall_seconds": 900,
            "max_rss_bytes": 2 * 1024 * 1024 * 1024,
            "min_free_memory_bytes": 2 * 1024 * 1024 * 1024,
            "min_free_disk_bytes": 2 * 1024 * 1024 * 1024,
        }
        if type(cpu_resources) is not dict or set(cpu_resources) != set(cpu_resource_limits):
            raise ValueError("CPU preparation resource policy has an unexpected schema")
        for key, expected_value in cpu_resource_limits.items():
            if type(cpu_resources.get(key)) is not int or cpu_resources[key] != expected_value:
                raise PermissionError("CPU preparation resource policy differs from reviewed fixed limits")
        cpu_max_wall_seconds = cpu_resource_limits["max_wall_seconds"]
        cpu_max_rss_bytes = cpu_resource_limits["max_rss_bytes"]
        cpu_min_free_memory_bytes = cpu_resource_limits["min_free_memory_bytes"]
        cpu_min_free_disk_bytes = cpu_resource_limits["min_free_disk_bytes"]
        if max_seconds > cpu_max_wall_seconds:
            raise PermissionError("loader time cap exceeds the CPU preparation wall-time limit")

    recipe = plan_doc.get("recipe")
    if type(recipe) is not dict or not recipe:
        raise ValueError("fit-pool plan must embed the complete nonempty recipe document")
    try:
        recipe_bytes = json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("fit-pool recipe is not canonical JSON") from exc
    recipe_sha = _require_sha(plan_doc.get("recipe_sha256"), "recipe_sha256")
    if _sha256_bytes(recipe_bytes) != recipe_sha:
        raise PermissionError("fit-pool recipe bytes differ from the locked recipe hash")
    plan_sha = _sha256_bytes(plan_raw)
    if (type(approval_doc.get("schema_version")) is not int or
            approval_doc.get("schema_version") != 1 or
            approval_doc.get("kind") != "q1_fit_pool_approval" or
            approval_doc.get("status") != "APPROVED_AFTER_INDEPENDENT_REVIEW"):
        raise PermissionError("fixed fit-pool approval is absent or not independently approved")
    if set(approval_doc) != {
            "schema_version", "kind", "status", "approval_id", "approved_by_role",
            "approved_utc", "plan_id", "plan_sha256", "recipe_sha256",
            "reviewed_artifacts", "independent_review_sha256"}:
        raise ValueError("fit-pool approval has an unexpected schema")
    approval_id = approval_doc.get("approval_id")
    if not isinstance(approval_id, str) or not AUTH_ID_RE.fullmatch(approval_id):
        raise ValueError("fit-pool approval_id is malformed")
    if approval_doc.get("approved_by_role") != "coordinator":
        raise PermissionError("fit-pool approval must be issued by the coordinator")
    if not isinstance(approval_doc.get("approved_utc"), str):
        raise ValueError("fit-pool approval timestamp is missing")
    try:
        from datetime import datetime
        approved_time = datetime.fromisoformat(approval_doc["approved_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("fit-pool approval timestamp is malformed") from exc
    if approved_time.tzinfo is None:
        raise ValueError("fit-pool approval timestamp must carry a timezone")
    if (approval_doc.get("plan_id") != plan_id or approval_doc.get("plan_sha256") != plan_sha or
            approval_doc.get("recipe_sha256") != recipe_sha):
        raise PermissionError("fit-pool approval does not lock the exact plan and recipe")
    if not reader.origin_inventory._exact_equal(approval_doc.get("reviewed_artifacts"), artifacts):
        raise PermissionError("fit-pool approval reviewed artifact hashes differ from the fixed loader bundle")
    _require_sha(approval_doc.get("independent_review_sha256"), "independent_review_sha256")

    return FitPoolPlan(
        plan_id=plan_id, purpose=purpose, role_mode=role_mode,
        fold_ids=tuple(fold_ids), seed_values=seeds_tuple, arm_ids=arms_tuple,
        standardizer_seed=standardizer_seed, windows_per_schedule=windows,
        recipe_sha256=recipe_sha, max_pose_bytes=max_pose_bytes,
        max_chunk_bytes=max_chunk_bytes, max_loader_seconds=max_seconds,
        plan_sha256=plan_sha, approval_id=approval_id,
        approval_sha256=_sha256_bytes(approval_raw),
        independent_review_sha256=approval_doc["independent_review_sha256"],
        numpy_version=np.__version__,
        reviewed_artifacts=artifacts, fitting_operations_allowed=fitting_operations,
        device=device, cuda_contexts=0, gpu_updates=0,
        optimizer_updates=optimizer_updates,
        cpu_consumer_source_sha256=cpu_consumer_sha,
        cpu_max_wall_seconds=cpu_max_wall_seconds, cpu_max_rss_bytes=cpu_max_rss_bytes,
        cpu_min_free_memory_bytes=cpu_min_free_memory_bytes,
        cpu_min_free_disk_bytes=cpu_min_free_disk_bytes,
    )


def _load_fixed_plan_and_approval(binding=None) -> tuple[object, FitPoolPlan]:
    """Authenticate metadata and fixed plan/approval before any signal open."""
    reader = _load_pinned_reader()
    if binding is None:
        binding = reader.validate_metadata_bindings()
    plan_raw, plan_doc, approval_raw, approval_doc = _read_fixed_docs(reader)
    access = _validate_plan_and_approval(
        reader, binding, plan_doc, plan_raw, approval_doc, approval_raw)
    return binding, access


@dataclass(frozen=True)
class TrustedCPUPrepLaunch:
    launch_id: str
    expires_at: datetime


def _validate_trusted_cpu_prep_launch(
    access: FitPoolPlan, *, now: datetime | None = None,
) -> TrustedCPUPrepLaunch:
    """Require root-provided launch pins before any production source access.

    The environment envelope is a trusted-caller capability supplied by the
    coordinating root tool after review. The loader does not authenticate an
    OS identity, create an envelope, or fall back to plan/approval JSON alone.
    """
    if (access.purpose != "fit_only_cpu_preparation" or access.role_mode != "outer_fit10" or
            access.device != "CPU" or access.cuda_contexts != 0 or
            access.gpu_updates != 0 or access.optimizer_updates != 0 or
            access.fitting_operations_allowed != ("population_feature_standardizer",) or
            access.seed_values or access.arm_ids or access.windows_per_schedule != 0):
        raise PermissionError(
            "production source loading is locked to CPU fit-only preparation; other scopes need separate review"
        )
    raw = os.environ.get(TRUSTED_LAUNCH_ENV)
    if not isinstance(raw, str) or not raw:
        raise PermissionError("trusted root launch capability is missing")
    if len(raw.encode("utf-8")) > MAX_LAUNCH_ENVELOPE_BYTES:
        raise PermissionError("trusted root launch capability exceeds its fixed size bound")
    envelope = _strict_json(raw.encode("utf-8"), "trusted CPU-preparation launch")
    expected_keys = {
        "purpose", "role_mode", "plan_sha256", "approval_sha256", "recipe_sha256",
        "independent_review_sha256", "loader_source_sha256", "consumer_source_sha256",
        "launch_id", "expires_utc",
    }
    if set(envelope) != expected_keys:
        raise PermissionError("trusted root launch capability has an unexpected schema")
    if (envelope.get("purpose") != access.purpose or
            envelope.get("role_mode") != access.role_mode):
        raise PermissionError("trusted root launch scope differs from the validated CPU plan")

    actual_loader_sha = _sha256_file(Path(__file__))
    if not CPU_CONSUMER_PATH.is_file() or CPU_CONSUMER_PATH.is_symlink():
        raise PermissionError("fixed CPU consumer source is unavailable for trusted launch binding")
    actual_consumer_sha = _sha256_file(CPU_CONSUMER_PATH)
    expected_pins = {
        "plan_sha256": access.plan_sha256,
        "approval_sha256": access.approval_sha256,
        "recipe_sha256": access.recipe_sha256,
        "independent_review_sha256": access.independent_review_sha256,
        "loader_source_sha256": actual_loader_sha,
        "consumer_source_sha256": actual_consumer_sha,
    }
    for key, expected in expected_pins.items():
        supplied = _require_sha(envelope.get(key), f"trusted launch {key}")
        if supplied != expected:
            raise PermissionError(f"trusted root launch {key} differs from validated bytes")
    if (access.reviewed_artifacts.get("fit_loader_code_sha256") != actual_loader_sha or
            access.cpu_consumer_source_sha256 != actual_consumer_sha):
        raise PermissionError("validated plan code pins differ from current loader/consumer bytes")

    launch_id = envelope.get("launch_id")
    if not isinstance(launch_id, str) or not ID_RE.fullmatch(launch_id):
        raise PermissionError("trusted launch_id is missing or malformed")
    expires_text = envelope.get("expires_utc")
    if not isinstance(expires_text, str):
        raise PermissionError("trusted launch expiry is missing")
    try:
        expires_at = datetime.fromisoformat(expires_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PermissionError("trusted launch expiry is malformed") from exc
    if expires_at.tzinfo is None:
        raise PermissionError("trusted launch expiry must carry a timezone")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("launch clock must be timezone-aware")
    remaining = expires_at.astimezone(timezone.utc) - now.astimezone(timezone.utc)
    if remaining <= timedelta(0) or remaining > timedelta(seconds=MAX_LAUNCH_TTL_SECONDS):
        raise PermissionError("trusted launch expiry must be future-dated within 900 seconds")
    return TrustedCPUPrepLaunch(launch_id, expires_at.astimezone(timezone.utc))


def _claim_production_fold(
    access: FitPoolPlan,
    launch: TrustedCPUPrepLaunch,
    fold_id: int,
    *,
    lock_root: Path | None = None,
    use_state: dict | None = None,
    seen_launch_ids: dict | None = None,
) -> None:
    """Exclusively lock this plan once and mark one fold before source I/O.

    `lock_root` and state arguments are test seams; production call sites use
    the fixed ignored location and module-level process state.
    """
    if type(fold_id) is not int or fold_id not in access.fold_ids:
        raise PermissionError("production fold is outside the trusted plan")
    use_state = _launch_use_state if use_state is None else use_state
    seen_launch_ids = _seen_launch_ids if seen_launch_ids is None else seen_launch_ids
    plan_sha = access.plan_sha256
    if launch.expires_at.tzinfo is None or launch.expires_at.utcoffset() is None:
        raise PermissionError("trusted launch expiry must be timezone-aware")
    expiry_utc = launch.expires_at.astimezone(timezone.utc).isoformat()
    root = Path(os.path.abspath(LAUNCH_LOCK_ROOT if lock_root is None else lock_root))
    if lock_root is None and root != Path(os.path.abspath(LAUNCH_LOCK_ROOT)):
        raise PermissionError("production attempt-lock root is not the fixed ignored directory")
    _assert_no_reparse_components(root)
    root_identity = None

    state = use_state.get(plan_sha)
    if state is not None:
        if (state["launch_id"] != launch.launch_id or
                state["approval_sha256"] != access.approval_sha256 or
                state["recipe_sha256"] != access.recipe_sha256):
            raise PermissionError("in-process launch state differs from the trusted plan capability")
        if state["lock_root"] != str(root):
            raise PermissionError("attempt-lock root changed during this in-process plan attempt")
        root_identity = _attempt_directory_identity(root)
        if root_identity != state["lock_root_identity"]:
            raise PermissionError("attempt-lock root filesystem identity changed during this plan attempt")
        lock_path = root / f"plan_{plan_sha}.attempt.json"
        _verify_plan_attempt_lock(
            lock_path, state["parent_plan_lock_identity"], access,
            launch_id=state["launch_id"], expires_utc=state["expires_utc"],
        )
        if state["expires_utc"] != expiry_utc:
            raise PermissionError("trusted launch expiry changed during this plan attempt")
        for claimed_fold, claim_identity in state["fold_claim_identities"].items():
            claim_path = root / f"plan_{plan_sha}.fold_{claimed_fold}.claim.json"
            _verify_attempt_lock(claim_path, claim_identity)
        if fold_id in state["folds"]:
            raise PermissionError("this trusted launch already attempted the requested fold")
    else:
        if launch.launch_id in seen_launch_ids:
            raise PermissionError("trusted launch_id was already used for a different plan")
        root.mkdir(parents=True, exist_ok=True)
        _assert_no_reparse_components(root)
        root_identity = _attempt_directory_identity(root)
        lock_path = root / f"plan_{plan_sha}.attempt.json"
        lock_doc = {
            "schema_version": 1,
            "kind": "q1_fit_pool_production_attempt_lock",
            "plan_sha256": plan_sha,
            "approval_sha256": access.approval_sha256,
            "recipe_sha256": access.recipe_sha256,
            "launch_id": launch.launch_id,
            "expires_utc": expiry_utc,
        }
        payload = (json.dumps(lock_doc, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        parent_identity = _write_exclusive_attempt_file(
            lock_path, payload,
            exists_message="this plan already has an exclusive production attempt lock",
        )
        state = {
            "launch_id": launch.launch_id,
            "approval_sha256": access.approval_sha256,
            "recipe_sha256": access.recipe_sha256,
            "lock_root": str(root),
            "lock_root_identity": root_identity,
            "expires_utc": expiry_utc,
            "parent_plan_lock_identity": parent_identity,
            "folds": set(),
            "fold_claim_identities": {},
        }
        # Retain state immediately after the durable plan lock. A later failure
        # must not let this process replace or redirect the attempt.
        use_state[plan_sha] = state
        seen_launch_ids[launch.launch_id] = plan_sha

    claim_path = root / f"plan_{plan_sha}.fold_{fold_id}.claim.json"
    claim_doc = {
        "schema_version": 1,
        "kind": "q1_fit_pool_production_fold_claim",
        "plan_sha256": plan_sha,
        "approval_sha256": access.approval_sha256,
        "recipe_sha256": access.recipe_sha256,
        "launch_id": launch.launch_id,
        "expires_utc": expiry_utc,
        "fold_id": fold_id,
        "parent_plan_lock_identity": state["parent_plan_lock_identity"],
    }
    claim_payload = (json.dumps(claim_doc, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _verify_plan_attempt_lock(
        lock_path, state["parent_plan_lock_identity"], access,
        launch_id=launch.launch_id, expires_utc=expiry_utc,
    )
    claim_identity = _write_exclusive_attempt_file(
        claim_path, claim_payload,
        exists_message="durable fold attempt claim already exists",
    )
    _verify_plan_attempt_lock(
        lock_path, state["parent_plan_lock_identity"], access,
        launch_id=launch.launch_id, expires_utc=expiry_utc,
    )
    state["fold_claim_identities"][fold_id] = claim_identity
    state["folds"].add(fold_id)


def _is_reparse_metadata(metadata) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & 0x400
    )  # FILE_ATTRIBUTE_REPARSE_POINT on Windows (covers junctions too).


def _assert_no_reparse_components(path: Path) -> None:
    """Reject symlink/reparse components from a fixed attempt-lock path."""
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PermissionError("attempt-lock path metadata is unavailable") from exc
        if _is_reparse_metadata(metadata):
            raise PermissionError("attempt-lock path contains a symlink or reparse point")


def _attempt_file_identity(metadata, digest: str) -> dict[str, int | str]:
    if not stat.S_ISREG(metadata.st_mode) or int(getattr(metadata, "st_nlink", 1)) != 1:
        raise PermissionError("attempt-lock artifact must be a single-link regular file")
    device = int(metadata.st_dev)
    inode = int(metadata.st_ino)
    if inode <= 0:
        raise PermissionError("attempt-lock filesystem did not report a stable file identity")
    return {
        "device": device,
        "inode": inode,
        "size": int(metadata.st_size),
        "mtime_ns": int(getattr(metadata, "st_mtime_ns", 0)),
        "sha256": digest,
    }


def _attempt_directory_identity(path: Path) -> dict[str, int]:
    _assert_no_reparse_components(path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PermissionError("attempt-lock root directory is unavailable") from exc
    if _is_reparse_metadata(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("attempt-lock root must be a real directory, not a reparse point")
    device = int(metadata.st_dev)
    inode = int(metadata.st_ino)
    if inode <= 0:
        raise PermissionError("attempt-lock root filesystem did not report a stable directory identity")
    return {"device": device, "inode": inode}


def _write_exclusive_attempt_file(
    path: Path, payload: bytes, *,
    exists_message: str = "immutable attempt-lock artifact already exists",
) -> dict[str, int | str]:
    if len(payload) > MAX_ATTEMPT_LOCK_BYTES:
        raise ValueError("attempt-lock artifact exceeds its fixed size limit")
    _assert_no_reparse_components(path.parent)
    _assert_no_reparse_components(path)
    try:
        with path.open("xb") as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode):
                raise PermissionError("new attempt-lock artifact is not a regular file")
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            opened_after = os.fstat(stream.fileno())
        _assert_no_reparse_components(path)
        path_after = path.lstat()
        if (_is_reparse_metadata(path_after) or
                (opened_after.st_dev, opened_after.st_ino) !=
                (path_after.st_dev, path_after.st_ino) or
                (opened_after.st_size, opened_after.st_mtime_ns,
                 opened_after.st_nlink) !=
                (path_after.st_size, path_after.st_mtime_ns,
                 path_after.st_nlink) or
                opened_before.st_ino != opened_after.st_ino or
                opened_after.st_size != len(payload)):
            raise PermissionError("attempt-lock path identity changed during exclusive write")
        return _attempt_file_identity(opened_after, _sha256_bytes(payload))
    except FileExistsError as exc:
        raise PermissionError(exists_message) from exc


def _verify_attempt_lock(path: Path, expected_identity: dict[str, int | str]) -> bytes:
    """Check a plan lock's path, file identity, and exact durable bytes."""
    _assert_no_reparse_components(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise PermissionError("durable plan attempt lock is unavailable") from exc
    if _is_reparse_metadata(before):
        raise PermissionError("durable plan attempt lock is a symlink or reparse point")
    observed = _attempt_file_identity(before, str(expected_identity["sha256"]))
    for key in ("device", "inode", "size", "mtime_ns"):
        if observed[key] != expected_identity[key]:
            raise PermissionError("durable plan attempt lock identity changed")
    if before.st_size > MAX_ATTEMPT_LOCK_BYTES:
        raise PermissionError("durable plan attempt lock exceeds its fixed size limit")
    try:
        with path.open("rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if (opened_before.st_dev, opened_before.st_ino) != (before.st_dev, before.st_ino):
                raise PermissionError("durable plan attempt lock changed before open")
            payload = stream.read(MAX_ATTEMPT_LOCK_BYTES + 1)
            opened_after = os.fstat(stream.fileno())
        _assert_no_reparse_components(path)
        after = path.lstat()
    except OSError as exc:
        raise PermissionError("durable plan attempt lock could not be verified") from exc
    if (len(payload) > MAX_ATTEMPT_LOCK_BYTES or
            _sha256_bytes(payload) != expected_identity["sha256"] or
            (opened_after.st_dev, opened_after.st_ino, opened_after.st_size,
             opened_after.st_mtime_ns) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) or
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) or
            _is_reparse_metadata(after)):
        raise PermissionError("durable plan attempt lock bytes or identity changed")
    return payload


def _verify_plan_attempt_lock(
    path: Path,
    expected_identity: dict[str, int | str],
    access: FitPoolPlan,
    *,
    launch_id: str,
    expires_utc: str,
) -> None:
    payload = _verify_attempt_lock(path, expected_identity)
    document = _strict_json(payload, "durable plan attempt lock")
    expected = {
        "schema_version": 1,
        "kind": "q1_fit_pool_production_attempt_lock",
        "plan_sha256": access.plan_sha256,
        "approval_sha256": access.approval_sha256,
        "recipe_sha256": access.recipe_sha256,
        "launch_id": launch_id,
        "expires_utc": expires_utc,
    }
    if document != expected:
        raise PermissionError("durable parent plan attempt lock document differs from the frozen launch")


def _role_users(binding, fold_id: int, role_mode: str) -> tuple[str, ...]:
    if type(fold_id) is not int or fold_id not in binding.ownership:
        raise ValueError("requested fold is outside the fixed train-fold plan")
    if role_mode == "outer_fit10":
        users = binding.ownership[fold_id]["outer_fit"]
    elif role_mode == "inner_fit8_diagnostic":
        users = binding.ownership[fold_id]["inner_fit"]
    else:
        raise PermissionError("only outer-fit10 and explicitly diagnostic inner-fit8 are permitted")
    expected = FIT_POOL_ROLE_COUNTS[role_mode]
    if len(users) != expected:
        raise ValueError("fixed train fit-role ownership count differs")
    if any(user not in binding.parent.expected_train.user_sessions for user in users):
        raise PermissionError("fit role contains a non-train user")
    return tuple(sorted(users))


def _canonical_fit_rows(binding, fold_id: int, role_mode: str):
    users = set(_role_users(binding, fold_id, role_mode))
    # Canonical origin rows are stored in each user's held-out outer fold;
    # for a given fold's fit pool, selected users therefore come from the two
    # *other* row folds as well. Role membership, not row.fold_id equality,
    # defines the pool.
    rows = [row for row in binding.origin_rows if row.user_id in users]
    rows.sort(key=lambda row: row.offset_int64)
    expected_segments = sum(len(binding.parent.expected_train.user_sessions[u][s])
                            for u in users
                            for s in binding.parent.expected_train.user_sessions[u])
    if len(rows) != expected_segments or not rows:
        raise ValueError("canonical fit origin rows do not cover the exact authorized role")
    previous_stop = -1
    for row in rows:
        if row.offset_int64 < previous_stop:
            raise ValueError("fit origin rows overlap in the canonical global order")
        if row.offset_int64 + row.count > len(binding.origin_values):
            raise ValueError("fit origin row exceeds the pinned canonical origin index")
        values = binding.origin_values[row.offset_int64:row.offset_int64 + row.count]
        if values.size != row.count or (values.size and int(values[0]) != 399) or (
                values.size and int(values[-1]) != 398 + row.count):
            raise ValueError("fit segment origin range differs from the fixed gap-free origin contract")
        if values.size > 1 and not np.array_equal(values[1:], values[:-1] + 1):
            raise ValueError("fit segment origins are not consecutive within their source segment")
        previous_stop = row.offset_int64 + row.count
    return tuple(rows)


def _pool_origin_digest(rows, binding, fold_id: int, role_mode: str) -> str:
    h = hashlib.sha256()
    h.update(f"q1-fit-pool-origin-map-v1\0{fold_id}\0{role_mode}\0".encode("utf-8"))
    for row in rows:
        header = (f"{row.fold_id}\0{row.user_id}\0{row.session_id}\0{row.segment_id}\0"
                  f"{row.offset_int64}\0{row.count}\n").encode("utf-8")
        h.update(header)
        values = binding.origin_values[row.offset_int64:row.offset_int64 + row.count]
        h.update(memoryview(values).cast("B"))
    return h.hexdigest()


def _schedule_ranks(total: int, *, fold_id: int, seed: int,
                    windows: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("approved fit pool has no origins")
    # Each seed produces one global batch schedule shared by all five arms in
    # that fold, so arm comparisons use identical sampled origins.
    sequence = np.random.SeedSequence([seed, fold_id])
    generator = np.random.default_rng(sequence)
    ranks = generator.integers(0, total, size=windows, endpoint=False, dtype=np.int64)
    ranks.setflags(write=False)
    return ranks


def _schedule_digest(ranks: np.ndarray, fold_id: int, kind: str,
                     seed: int, arm_id: str | None) -> str:
    h = hashlib.sha256()
    header = f"q1-fit-schedule-v1\0{fold_id}\0{kind}\0{seed}\0{arm_id or ''}\0".encode()
    h.update(header)
    h.update(memoryview(np.asarray(ranks, dtype="<i8")).cast("B"))
    return h.hexdigest()


def _origin_identity_digest(ranks: np.ndarray, rows, binding, fold_id: int,
                            role_mode: str,
                            pool_digest: str) -> str:
    counts = np.asarray([row.count for row in rows], dtype=np.int64)
    cumulative = np.cumsum(counts)
    segment_indices = np.searchsorted(cumulative, ranks, side="right")
    offsets = np.asarray([row.offset_int64 for row in rows], dtype=np.int64)
    before = np.concatenate((np.asarray([0], dtype=np.int64), cumulative[:-1]))
    global_offsets = offsets[segment_indices] + (ranks - before[segment_indices])
    actual_indices = binding.origin_values[global_offsets]
    h = hashlib.sha256()
    h.update(f"q1-fit-selected-origin-identities-v1\0{fold_id}\0{role_mode}\n".encode())
    # Pool digest authenticates the private ID-to-global-origin mapping without
    # copying those identifiers into a public aggregate receipt.
    h.update(pool_digest.encode("ascii"))
    h.update(memoryview(np.asarray(global_offsets, dtype="<i8")).cast("B"))
    h.update(memoryview(np.asarray(actual_indices, dtype="<i8")).cast("B"))
    return h.hexdigest()


def _prepare_fold_schedules(binding, access: FitPoolPlan, fold_id: int):
    if type(fold_id) is not int or fold_id not in access.fold_ids:
        raise PermissionError("requested fold is not in the frozen fit-pool plan")
    rows = _canonical_fit_rows(binding, fold_id, access.role_mode)
    total = sum(row.count for row in rows)
    if total < STANDARDIZER_ORIGINS_PER_FOLD:
        raise PermissionError("approved fit pool is smaller than the distinct-origin standardizer sample")
    pool_digest = _pool_origin_digest(rows, binding, fold_id, access.role_mode)
    schedules: list[PreparedSchedule] = []
    if access.purpose != "fit_only_cpu_preparation":
        for seed in access.seed_values:
            seed_ranks = _schedule_ranks(
                total, fold_id=fold_id, seed=seed,
                windows=access.windows_per_schedule)
            for arm_id in access.arm_ids:
                ranks = seed_ranks
                digest = _schedule_digest(ranks, fold_id, "training_with_replacement", seed, arm_id)
                identity = _origin_identity_digest(ranks, rows, binding, fold_id,
                                                   access.role_mode, pool_digest)
                schedules.append(PreparedSchedule(
                    fold_id, "training_with_replacement", seed, arm_id, ranks,
                    digest, identity))
    std_seed = np.random.SeedSequence([STANDARDIZER_SEED, fold_id])
    std_rng = np.random.default_rng(std_seed)
    std_ranks = std_rng.choice(
        total, size=STANDARDIZER_ORIGINS_PER_FOLD, replace=False).astype(np.int64, copy=False)
    if len(np.unique(std_ranks)) != STANDARDIZER_ORIGINS_PER_FOLD:
        raise AssertionError("standardizer schedule contains duplicate origins")
    std_ranks.setflags(write=False)
    std_digest = _schedule_digest(std_ranks, fold_id, "standardizer_without_replacement",
                                  access.standardizer_seed, None)
    std_identity = _origin_identity_digest(std_ranks, rows, binding, fold_id,
                                           access.role_mode, pool_digest)
    standardizer = PreparedSchedule(
        fold_id, "standardizer_without_replacement", access.standardizer_seed,
        None, std_ranks, std_digest, std_identity)
    return rows, total, pool_digest, tuple(schedules), standardizer


@dataclass
class _DatasetAudit:
    allowed_file_id: int
    allowed_paths: frozenset[str]
    max_bytes: int
    deadline: float
    audit: ReadAudit


def _read_dataset_rows(dataset, *, start: int, stop: int, audit: _DatasetAudit) -> np.ndarray:
    """One logged, bounded row slice from the exact opened file and fit path."""
    try:
        file_id = int(dataset.file.id.id)
    except (AttributeError, ValueError, RuntimeError) as exc:
        audit.audit.denied_reads += 1
        raise PermissionError("fit dataset has no live HDF5 FileID") from exc
    if file_id != audit.allowed_file_id or dataset.name not in audit.allowed_paths:
        audit.audit.denied_reads += 1
        raise PermissionError("fit read path or FileID is outside the authorized pool")
    if time.monotonic() > audit.deadline:
        audit.audit.denied_reads += 1
        raise TimeoutError("approved fit-pool load exceeded its fixed time allowance")
    if (type(start) is not int or type(stop) is not int or start < 0 or stop <= start or
            stop > dataset.shape[0]):
        audit.audit.denied_reads += 1
        raise PermissionError("fit chunk indices are invalid or outside the selected segment")
    logical_bytes = (stop - start) * int(np.prod(dataset.shape[1:] or (1,))) * dataset.dtype.itemsize
    if logical_bytes <= 0 or logical_bytes > audit.max_bytes:
        audit.audit.denied_reads += 1
        raise PermissionError("fit chunk exceeds the frozen logical byte bound")
    result = dataset[start:stop]
    audit.audit.dataset_slice_calls += 1
    audit.audit.max_logical_slice_bytes = max(audit.audit.max_logical_slice_bytes, logical_bytes)
    audit.audit.value_rows_by_dataset[dataset.name] = (
        audit.audit.value_rows_by_dataset.get(dataset.name, 0) + (stop - start))
    return result


def _get_selected_child(parent, name: str, *, expected_path: str, source_file,
                        fit_users: set[str], audit: ReadAudit, group: bool):
    if (not isinstance(name, str) or not name or name in {".", ".."} or "/" in name):
        audit.denied_reads += 1
        raise PermissionError("HDF5 child name must be one nonempty path component")
    parent_name = getattr(parent, "name", None)
    if not isinstance(parent_name, str) or not parent_name.startswith("/"):
        audit.denied_reads += 1
        raise PermissionError("HDF5 parent does not have a canonical absolute name")
    canonical_path = ("/" + name) if parent_name == "/" else (
        parent_name.rstrip("/") + "/" + name)
    if not isinstance(expected_path, str) or expected_path != canonical_path:
        audit.denied_reads += 1
        raise PermissionError("expected HDF5 child path is not the exact canonical parent/name")
    file_id = int(source_file.id.id)
    if int(parent.file.id.id) != file_id:
        audit.denied_reads += 1
        raise PermissionError("HDF5 metadata path belongs to a foreign FileID")
    if not expected_path.startswith("/users/") and expected_path != "/users":
        audit.denied_reads += 1
        raise PermissionError("HDF5 metadata path is outside the users root")
    if expected_path.startswith("/users/"):
        user = expected_path.split("/", 3)[2]
        if user not in fit_users:
            audit.denied_reads += 1
            raise PermissionError("HDF5 metadata path is outside the approved fit role")
    if not expected_path.startswith(parent.name.rstrip("/") + "/") and not (
            parent.name == "/" and expected_path == "/users"):
        audit.denied_reads += 1
        raise PermissionError("HDF5 metadata path does not match its parent group")
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        audit.denied_reads += 1
        raise ValueError("fit HDF5 hierarchy has a soft or external link")
    obj = parent[name]
    if int(obj.file.id.id) != file_id:
        audit.denied_reads += 1
        raise PermissionError("HDF5 child belongs to a foreign FileID")
    if getattr(obj, "name", None) != expected_path:
        audit.denied_reads += 1
        raise PermissionError("dereferenced HDF5 object name differs from the exact selected path")
    audit.metadata_paths.append(expected_path)
    if group and not isinstance(obj, h5py.Group):
        raise ValueError("fit HDF5 hierarchy contains a non-group path component")
    if not group and not isinstance(obj, h5py.Dataset):
        raise ValueError("fit HDF5 segment child is not a dataset")
    return obj


def _validate_timestamp_chunk(timestamp_chunk: np.ndarray,
                              previous_timestamp: float | None) -> float:
    if timestamp_chunk.dtype != np.dtype("float64") or timestamp_chunk.ndim != 1:
        raise ValueError("fit timestamp chunk has an unexpected dtype or shape")
    if not np.isfinite(timestamp_chunk).all():
        raise ValueError("fit timestamps contain nonfinite values")
    deltas = np.diff(timestamp_chunk)
    if previous_timestamp is not None:
        deltas = np.concatenate((np.asarray([timestamp_chunk[0] - previous_timestamp]), deltas))
    if deltas.size and (not np.isfinite(deltas).all() or np.any(deltas <= 0.0) or
                        np.any(np.abs(deltas - 0.01) > 1e-6)):
        raise ValueError("fit segment timestamps are not a gap-free 100 Hz grid")
    return float(timestamp_chunk[-1])


def _validate_pose_chunk(pose_chunk: np.ndarray,
                         previous_pose: np.ndarray | None) -> np.ndarray:
    if pose_chunk.dtype != np.dtype("float32") or pose_chunk.ndim != 2 or pose_chunk.shape[1] != 4:
        raise ValueError("fit quaternion chunk has an unexpected dtype or shape")
    if not np.isfinite(pose_chunk).all():
        raise ValueError("fit segment contains nonfinite values")
    norms = np.linalg.norm(pose_chunk.astype(np.float64), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 2e-6:
        raise ValueError("fit segment contains a nonunit quaternion")
    adjacent = np.sum(pose_chunk[1:] * pose_chunk[:-1], axis=1)
    if previous_pose is not None:
        adjacent = np.concatenate((np.asarray([np.dot(previous_pose, pose_chunk[0])]), adjacent))
    if adjacent.size and float(np.min(adjacent)) < -1e-6:
        raise ValueError("fit segment quaternion signs are not chronologically aligned")
    return pose_chunk[-1].copy()


def _selected_record_digest(records: list[dict]) -> str:
    """Digest authenticated source metadata fields, never recording bytes."""
    if type(records) is not list:
        raise PermissionError("legacy source inventory records are malformed")
    rows = []
    for record in records:
        if type(record) is not dict:
            raise PermissionError("legacy source inventory record is malformed")
        if not {"source_path", "expected_bytes", "md5"}.issubset(record):
            raise PermissionError("legacy source inventory record lacks consumed fields")
        rows.append({key: record[key] for key in ("source_path", "expected_bytes", "md5")})
    rows.sort(key=lambda row: row["source_path"])
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_receipt_tuples(inventory_records: list[dict], receipt: dict,
                             *, label: str) -> None:
    fields = ("source_path", "source_revision", "user_id", "session_id",
              "relative_path", "url", "expected_bytes", "md5")
    receipt_records = receipt.get("records") if type(receipt) is dict else None
    if (type(inventory_records) is not list or type(receipt_records) is not list or
            len(inventory_records) != 70 or len(receipt_records) != 70):
        raise PermissionError(f"{label} does not bind the exact 70 source records")
    inventory_by_path: dict[str, dict] = {}
    session_parts: dict[tuple[str, str], list[str]] = {}
    for row in inventory_records:
        if type(row) is not dict or any(key not in row for key in fields):
            raise PermissionError(f"{label} inventory tuple is malformed")
        path = row["source_path"]
        if not isinstance(path, str) or path in inventory_by_path:
            raise PermissionError(f"{label} inventory source paths are not unique")
        if type(row["expected_bytes"]) is not int or row["expected_bytes"] <= 0:
            raise PermissionError(f"{label} inventory byte metadata is malformed")
        if not isinstance(row["md5"], str) or len(row["md5"]) != 32:
            raise PermissionError(f"{label} inventory checksum metadata is malformed")
        inventory_by_path[path] = row
        session_parts.setdefault((row["user_id"], row["session_id"]), []).append(path)
    expected_parts = {key: sorted(value) for key, value in session_parts.items()}
    seen: set[str] = set()
    for row in receipt_records:
        if type(row) is not dict or any(key not in row for key in (*fields, "expected_session_paths")):
            raise PermissionError(f"{label} receipt tuple is malformed")
        path = row["source_path"]
        source = inventory_by_path.get(path)
        if source is None or path in seen:
            raise PermissionError(f"{label} receipt does not uniquely cover the inventory")
        if any(row.get(key) != source.get(key) for key in fields):
            raise PermissionError(f"{label} receipt tuple differs from the authenticated inventory")
        session = (source["user_id"], source["session_id"])
        if row.get("expected_session_paths") != expected_parts[session]:
            raise PermissionError(f"{label} session-part list differs from the authenticated inventory")
        seen.add(path)
    if len(seen) != 70 or seen != set(inventory_by_path):
        raise PermissionError(f"{label} receipt does not cover all 70 inventory tuples")


def _validate_legacy_root_lineage(reader, binding, *, source_hash_verified: bool) -> None:
    """Authenticate the exact legacy receipt chain before deriving root attrs."""
    if source_hash_verified is not True:
        raise PermissionError("legacy root attribute derivation requires verified same-FD source bytes")
    parent = binding.parent
    metadata = parent.metadata
    hashes = parent.metadata_sha256
    expected_hashes = {
        "inventory": LEGACY_INVENTORY_SHA256,
        "legacy_receipt": LEGACY_RECEIPT_SHA256,
        "versioned_receipt": VERSIONED_RECEIPT_SHA256,
        "metadata_bridge": METADATA_BRIDGE_SHA256,
        "semantic_config": SEMANTIC_CONFIG_SHA256,
        "semantic_receipt": SEMANTIC_RECEIPT_SHA256,
        "pipeline_config": PIPELINE_CONFIG_SHA256,
        "arc_config": ARC_CONFIG_SHA256,
        "split": SPLIT_SHA256,
        "source_helper": SOURCE_HELPER_SHA256,
    }
    if any(hashes.get(key) != value for key, value in expected_hashes.items()):
        raise PermissionError("legacy source metadata bytes differ from the pinned lineage")
    if (parent.source_hdf5_sha256_reference != SOURCE_HDF5_SHA256 or
            parent.split_sha256 != SPLIT_SHA256 or
            Path(binding.source_path) != Path(reader.SOURCE_HDF5_PATH) or
            binding.source_path_evidence_sha256 != SOURCE_PATH_EVIDENCE_SHA256 or
            binding.origin_receipt_sha256 != reader.ORIGIN_RECEIPT_SHA256 or
            binding.origin_map_sha256 != reader.ORIGIN_MAP_SHA256 or
            binding.origin_binary_sha256 != reader.ORIGIN_BINARY_SHA256):
        raise PermissionError("legacy source path, split, or origin receipt chain differs from its fixed pins")
    if len(parent.expected_train.users) != 15 or parent.expected_train.expected_windows != 3_330_312:
        raise PermissionError("legacy source metadata does not retain the exact train partition")

    inventory = metadata.get("inventory")
    legacy = metadata.get("legacy_receipt")
    versioned = metadata.get("versioned_receipt")
    resealed_from = versioned.get("resealed_from") if type(versioned) is dict else None
    versioned_bindings = versioned.get("inventory_bindings") if type(versioned) is dict else None
    if (type(inventory) is not dict or type(legacy) is not dict or
            type(versioned) is not dict or
            type(legacy.get("schema_version", 1)) is not int or
            legacy.get("schema_version", 1) != 1 or
            legacy.get("source_manifest_sha256") != LEGACY_INVENTORY_SHA256 or
            type(versioned.get("schema_version")) is not int or
            versioned.get("schema_version") != 2 or
            type(resealed_from) is not dict or
            set(resealed_from) != {"schema_version", "source_manifest_sha256",
                                   "source_receipt_sha256"} or
            type(resealed_from.get("schema_version")) is not int or
            resealed_from.get("schema_version") != 1 or
            resealed_from.get("source_manifest_sha256") != LEGACY_INVENTORY_SHA256 or
            resealed_from.get("source_receipt_sha256") != LEGACY_RECEIPT_SHA256 or
            type(versioned.get("record_inventory_sha256")) is not str or
            versioned.get("record_inventory_sha256") != LEGACY_RECORD_DIGEST_SHA256 or
            type(versioned_bindings) is not list or len(versioned_bindings) != 1 or
            versioned_bindings != [{
                "manifest_sha256": LEGACY_INVENTORY_SHA256,
                "record_inventory_sha256": LEGACY_RECORD_DIGEST_SHA256,
            }]):
        raise PermissionError("legacy inventory byte and record digests are not bound by the versioned receipt chain")
    inventory_records = inventory.get("records")
    _validate_receipt_tuples(inventory_records, legacy, label="legacy receipt")
    _validate_receipt_tuples(inventory_records, versioned, label="integrated versioned receipt")
    if _selected_record_digest(inventory_records) != LEGACY_RECORD_DIGEST_SHA256:
        raise PermissionError("legacy record digest differs from the authenticated 70-row lineage")

    bridge = metadata.get("metadata_bridge")
    bridge_result = bridge.get("result", {}) if type(bridge) is dict else {}
    before = bridge.get("input_byte_hashes_before", {}) if type(bridge) is dict else {}
    after = bridge.get("input_byte_hashes_after", {}) if type(bridge) is dict else {}
    if (type(bridge) is not dict or bridge.get("legacy_metadata_bytes_unchanged") is not True or
            before != after or before.get("data\\alyx_inventory.json") != LEGACY_INVENTORY_SHA256 or
            before.get("data\\download_receipt.json") != LEGACY_RECEIPT_SHA256 or
            bridge.get("HDF5_and_split_rewritten") is not False or
            bridge_result.get("inventory_file_sha256") != LEGACY_INVENTORY_SHA256 or
            bridge_result.get("legacy_receipt_file_sha256") != LEGACY_RECEIPT_SHA256 or
            bridge_result.get("new_receipt_file_sha256") != VERSIONED_RECEIPT_SHA256 or
            bridge_result.get("new_receipt_schema_version") != 2 or
            bridge_result.get("record_inventory_sha256") != LEGACY_RECORD_DIGEST_SHA256 or
            bridge_result.get("inventory_records") != 70 or bridge_result.get("receipt_records") != 70 or
            bridge_result.get("legacy_receipt_validated") is not True or
            bridge_result.get("recording_payloads_read") != 0 or
            bridge.get("signal_open_spy_attempts") != 0 or
            bridge.get("human_signal_rows_read") != 0):
        raise PermissionError("G1 metadata bridge does not prove unchanged authenticated lineage")

    semantic_config = metadata.get("semantic_config")
    semantic_receipt = metadata.get("semantic_receipt")
    source_binding = (semantic_config.get("source_binding", {})
                      if type(semantic_config) is dict else {})
    source_validation = (semantic_receipt.get("source_validation", {})
                         if type(semantic_receipt) is dict else {})
    if (source_binding.get("source_hdf5_sha256_reference") != SOURCE_HDF5_SHA256 or
            source_binding.get("authenticated_record_inventory_sha256") != LEGACY_RECORD_DIGEST_SHA256 or
            source_binding.get("source_inventory_semantic_id") != SOURCE_INVENTORY_SEMANTIC_ID or
            source_binding.get("inventory_records") != 70 or
            source_validation.get("source_hdf5_sha256_reference") != SOURCE_HDF5_SHA256 or
            source_validation.get("authenticated_record_inventory_sha256") != LEGACY_RECORD_DIGEST_SHA256 or
            source_validation.get("source_inventory_semantic_id") != SOURCE_INVENTORY_SEMANTIC_ID):
        raise PermissionError("semantic rebind does not bind the authenticated record digest and source")
    code_bindings = semantic_config.get("code_byte_bindings", {})
    if (code_bindings.get("ABC_IUEH/src/abc_iueh/sources.py") != SOURCE_HELPER_SHA256 or
            code_bindings.get("workflow_runtime/staging/semantic_rebind/rebind.py") !=
            SEMANTIC_REBIND_SHA256):
        raise PermissionError("semantic source metadata validators differ from their fixed code pins")


class _RootAttrsWithLegacyDerivations:
    """Read-only overlay of exactly the two absent, receipt-derived attrs."""

    def __init__(self, actual_attrs, derived: dict[str, str]):
        self._actual_attrs = actual_attrs
        self._derived = derived

    def get(self, key, default=None):
        if key in self._derived:
            return self._derived[key]
        return self._actual_attrs.get(key, default)

    def __getitem__(self, key):
        if key in self._derived:
            return self._derived[key]
        return self._actual_attrs[key]


class _RootMetadataWithLegacyDerivations:
    """Forward every real root attribute/key except the two authenticated gaps."""

    def __init__(self, source_file, derived: dict[str, str]):
        self.attrs = _RootAttrsWithLegacyDerivations(source_file.attrs, derived)
        self._source_file = source_file

    def keys(self):
        return self._source_file.keys()


def _validate_source_root_with_legacy_bridge(reader, source_file, binding, *,
                                             source_hash_verified: bool) -> str:
    """Use the pinned reader unchanged, with one exact absent-attrs bridge."""
    if isinstance(source_file, h5py.File):
        # Construct the exact root group on the already-open, same FileID. This
        # avoids a path lookup through File['/'] while keeping root metadata
        # reads and the underlying signal-access capability tied to one file.
        root_metadata = h5py.Group(h5py.h5g.open(source_file.id, b"/"))
        if (root_metadata.name != "/" or
                int(root_metadata.file.id.id) != int(source_file.id.id)):
            raise PermissionError("source root metadata view does not retain the verified FileID")
    else:
        # Lightweight synthetic fixtures exercise the identical validator
        # contract without opening the production file.
        root_metadata = source_file
    attrs = root_metadata.attrs
    names = set(attrs.keys())
    inventory_key = "source_inventory_file_sha256"
    record_key = "source_record_inventory_sha256"
    inventory_present, record_present = inventory_key in names, record_key in names
    if inventory_present and record_present:
        # Includes present-null and wrong-value cases: the reviewed validator
        # must reject them without any legacy fallback.
        reader._validate_source_root(root_metadata, binding)
        return "direct_root_attributes"
    if inventory_present != record_present:
        raise PermissionError("legacy root metadata bridge rejects a partially absent attribute pair")
    if attrs.get("source_receipt_sha256") != LEGACY_RECEIPT_SHA256:
        raise PermissionError("legacy root metadata bridge requires the exact pinned receipt pointer")
    _validate_legacy_root_lineage(reader, binding,
                                  source_hash_verified=source_hash_verified)
    derived = {
        inventory_key: LEGACY_INVENTORY_SHA256,
        record_key: LEGACY_RECORD_DIGEST_SHA256,
    }
    reader._validate_source_root(_RootMetadataWithLegacyDerivations(root_metadata, derived), binding)
    return "legacy_receipt_chain_derived"


def _load_fit_pool_from_open_file(source_file, binding, access: FitPoolPlan, fold_id: int,
                                  prepared, *, source_hash_verified: bool,
                                  deadline: float) -> "FitPool":
    reader = _load_pinned_reader()
    rows, total_origins, pool_digest, schedules, standardizer = prepared
    fit_users = set(_role_users(binding, fold_id, access.role_mode))
    audit = ReadAudit(int(source_file.id.id), len(fit_users), source_hash_verified=source_hash_verified)
    audit.source_root_metadata_binding_mode = _validate_source_root_with_legacy_bridge(
        reader, source_file, binding, source_hash_verified=source_hash_verified)
    users = _get_selected_child(source_file, "users", expected_path="/users",
                                source_file=source_file, fit_users=fit_users,
                                audit=audit, group=True)
    addresses: dict[int, str] = {}
    reader._register_unique(users, "/users", addresses)
    row_by_key = {(row.user_id, row.session_id, row.segment_id): row for row in rows}
    dataset_specs = []
    total_pose_bytes = 0

    # Metadata-only pass over precisely the approved fit role. No user outside
    # this set is enumerated or dereferenced.
    for user in sorted(fit_users):
        user_path = f"/users/{user}"
        user_group = _get_selected_child(users, user, expected_path=user_path,
                                         source_file=source_file, fit_users=fit_users,
                                         audit=audit, group=True)
        reader._register_unique(user_group, user_path, addresses)
        expected_sessions = binding.parent.expected_train.user_sessions[user]
        if set(user_group.keys()) != {"sessions"}:
            raise ValueError("fit user hierarchy differs from the accepted train schema")
        session_path = f"{user_path}/sessions"
        sessions = _get_selected_child(user_group, "sessions", expected_path=session_path,
                                       source_file=source_file, fit_users=fit_users,
                                       audit=audit, group=True)
        reader._register_unique(sessions, session_path, addresses)
        if set(sessions.keys()) != set(expected_sessions):
            raise ValueError("fit sessions differ from accepted train quality metadata")
        for session in sorted(expected_sessions):
            sess_path = f"{session_path}/{session}"
            sess = _get_selected_child(sessions, session, expected_path=sess_path,
                                       source_file=source_file, fit_users=fit_users,
                                       audit=audit, group=True)
            reader._register_unique(sess, sess_path, addresses)
            if set(sess.keys()) != {"segments"}:
                raise ValueError("fit session hierarchy differs from the fixed schema")
            segs_path = f"{sess_path}/segments"
            segs = _get_selected_child(sess, "segments", expected_path=segs_path,
                                       source_file=source_file, fit_users=fit_users,
                                       audit=audit, group=True)
            reader._register_unique(segs, segs_path, addresses)
            expected_segments = expected_sessions[session]
            if set(segs.keys()) != set(expected_segments):
                raise ValueError("fit segments differ from accepted train quality metadata")
            session_samples = 0
            for segment in sorted(expected_segments):
                seg_path = f"{segs_path}/{segment}"
                seg = _get_selected_child(segs, segment, expected_path=seg_path,
                                          source_file=source_file, fit_users=fit_users,
                                          audit=audit, group=True)
                reader._register_unique(seg, seg_path, addresses)
                if set(seg.keys()) != {"timestamps", "quaternion"}:
                    raise ValueError("fit segment contains an unexpected payload")
                ts_path, q_path = f"{seg_path}/timestamps", f"{seg_path}/quaternion"
                timestamps = _get_selected_child(seg, "timestamps", expected_path=ts_path,
                                                 source_file=source_file, fit_users=fit_users,
                                                 audit=audit, group=False)
                quaternion = _get_selected_child(seg, "quaternion", expected_path=q_path,
                                                 source_file=source_file, fit_users=fit_users,
                                                 audit=audit, group=False)
                n_t = reader._dataset_metadata(timestamps, dtype="float64", ndim=1)
                n_q = reader._dataset_metadata(quaternion, dtype="float32", ndim=2)
                if n_q[1:] != (4,) or n_t != (n_q[0],) or n_q[0] < 1000:
                    raise ValueError("fit segment has invalid shape, sample floor, or storage")
                row = row_by_key.get((user, session, segment))
                if row is None or row.count != n_q[0] - 419:
                    raise ValueError("fit segment sample count differs from the fixed origin index")
                reader._register_unique(timestamps, ts_path, addresses)
                reader._register_unique(quaternion, q_path, addresses)
                total_pose_bytes += int(n_q[0]) * 4 * np.dtype("float32").itemsize
                session_samples += int(n_q[0])
                dataset_specs.append((row, int(n_q[0]), timestamps, quaternion, ts_path, q_path))
            expected_samples = (binding.parent.expected_train.session_windows[(user, session)] +
                               419 * len(expected_segments))
            if session_samples != expected_samples:
                raise ValueError("fit session sample count differs from accepted quality metadata")

    if total_pose_bytes > access.max_pose_bytes or total_pose_bytes > MAX_POSE_ARRAY_BYTES:
        raise MemoryError("fit pose arrays exceed the approved 128 MiB preallocation cap")
    if total_origins != sum(spec[0].count for spec in dataset_specs):
        raise ValueError("fit metadata rows and approved pool origin count differ")

    arrays: list[PoseSegment] = []
    logical_rows = max(1, access.max_chunk_bytes // (4 * 4))
    dataset_audit = _DatasetAudit(
        int(source_file.id.id),
        frozenset(path for spec in dataset_specs for path in (spec[4], spec[5])),
        access.max_chunk_bytes, deadline, audit)
    started = time.monotonic()
    for row, sample_count, timestamps, quaternion, ts_path, q_path in dataset_specs:
        poses = np.empty((sample_count, 4), dtype=np.float32)
        previous_pose = None
        previous_timestamp = None
        for start in range(0, sample_count, logical_rows):
            if time.monotonic() > deadline:
                raise TimeoutError("fit pool load exceeded the approved time allowance")
            stop = min(sample_count, start + logical_rows)
            t_chunk = _read_dataset_rows(timestamps, start=start, stop=stop, audit=dataset_audit)
            previous_timestamp = _validate_timestamp_chunk(t_chunk, previous_timestamp)
            q_chunk = _read_dataset_rows(quaternion, start=start, stop=stop, audit=dataset_audit)
            previous_pose = _validate_pose_chunk(q_chunk, previous_pose)
            poses[start:stop] = q_chunk
            del q_chunk, t_chunk
        poses.setflags(write=False)
        arrays.append(PoseSegment(
            row.user_id, row.session_id, row.segment_id,
            row.offset_int64, row.count, poses))
    audit.pose_array_bytes = total_pose_bytes
    audit.elapsed_loader_seconds = time.monotonic() - started
    if audit.denied_reads != 0:
        raise PermissionError("fit pool read audit recorded a denied access")
    return FitPool(
        binding=binding, access=access, fold_id=fold_id, segments=tuple(arrays),
        total_origin_count=total_origins, origin_pool_sha256=pool_digest,
        schedules=schedules, standardizer_schedule=standardizer, audit=audit,
        source_hash_verified=source_hash_verified, deadline=deadline,
    )


class FitPool:
    """Read-only pose buffers with deterministic, lazy Bx400/Bx20 gathers."""

    def __init__(self, *, binding, access: FitPoolPlan, fold_id: int,
                 segments: tuple[PoseSegment, ...], total_origin_count: int,
                 origin_pool_sha256: str, schedules: tuple[PreparedSchedule, ...],
                 standardizer_schedule: PreparedSchedule, audit: ReadAudit,
                 source_hash_verified: bool, deadline: float):
        self._binding = binding
        self._access = access
        self.fold_id = fold_id
        self.role_mode = access.role_mode
        self._segments = segments
        self.total_origin_count = total_origin_count
        self.origin_pool_sha256 = origin_pool_sha256
        self.training_schedules = schedules
        self.standardizer_schedule = standardizer_schedule
        self._audit = audit
        self.source_hash_verified = source_hash_verified
        self._deadline = deadline
        self._standardizer_iteration_started = False
        self._counts = np.asarray([segment.origin_count for segment in segments], dtype=np.int64)
        self._cumulative = np.cumsum(self._counts)
        if int(self._cumulative[-1]) != total_origin_count:
            raise ValueError("loaded pose segments do not cover the fixed fit origin pool")

    def _validate_prepared_schedule(self, schedule: PreparedSchedule) -> np.ndarray:
        candidates = (*self.training_schedules, self.standardizer_schedule)
        if not any(schedule is candidate for candidate in candidates):
            raise PermissionError("schedule was not generated from this fixed approved fold plan")
        if schedule.fold_id != self.fold_id or schedule.ranks.dtype != np.dtype("int64"):
            raise PermissionError("prepared schedule fold or integer dtype differs")
        if schedule.ranks.flags.writeable:
            raise PermissionError("prepared schedule indices must be immutable")
        if len(schedule.ranks) == 0 or np.any(schedule.ranks < 0) or np.any(
                schedule.ranks >= self.total_origin_count):
            raise PermissionError("prepared schedule contains an out-of-range origin rank")
        if schedule.kind == "standardizer_without_replacement":
            if len(schedule.ranks) != STANDARDIZER_ORIGINS_PER_FOLD or len(
                    np.unique(schedule.ranks)) != len(schedule.ranks):
                raise PermissionError("standardizer schedule is not 65,536 distinct origins")
            expected = np.random.default_rng(np.random.SeedSequence(
                [STANDARDIZER_SEED, self.fold_id])).choice(
                    self.total_origin_count, size=STANDARDIZER_ORIGINS_PER_FOLD,
                    replace=False).astype(np.int64, copy=False)
            if not np.array_equal(schedule.ranks, expected):
                raise PermissionError("standardizer ranks differ from the fixed sampler seed")
        elif schedule.kind == "training_with_replacement":
            if schedule.arm_id not in self._access.arm_ids:
                raise PermissionError("training schedule arm is outside the frozen recipe")
            if schedule.seed not in self._access.seed_values or len(schedule.ranks) != self._access.windows_per_schedule:
                raise PermissionError("training schedule seed or window count differs from the approval")
            expected = _schedule_ranks(
                self.total_origin_count, fold_id=self.fold_id, seed=schedule.seed,
                windows=self._access.windows_per_schedule)
            if not np.array_equal(schedule.ranks, expected):
                raise PermissionError("training ranks differ from the fixed global sampler schedule")
        else:
            raise PermissionError("unknown fit-pool sampler kind")
        digest = _schedule_digest(schedule.ranks, schedule.fold_id, schedule.kind,
                                  schedule.seed, schedule.arm_id)
        if digest != schedule.schedule_sha256:
            raise PermissionError("prepared schedule digest differs from its immutable ranks")
        return schedule.ranks

    def iter_batches(self, schedule: PreparedSchedule) -> Iterator[FitBatch]:
        ranks = self._validate_prepared_schedule(schedule)
        if self._access.purpose == "fit_only_cpu_preparation":
            if schedule is not self.standardizer_schedule:
                raise PermissionError("CPU preparation only permits its standardizer schedule")
            if self._standardizer_iteration_started:
                raise PermissionError("CPU preparation standardizer schedule is single-pass")
            self._standardizer_iteration_started = True
        for first in range(0, len(ranks), MAX_BATCH_SIZE):
            if time.monotonic() > self._deadline:
                raise TimeoutError("approved fit-pool access exceeded its fixed time allowance")
            selection = ranks[first:first + MAX_BATCH_SIZE]
            size = len(selection)
            contexts = np.empty((size, CONTEXT_SAMPLES, 4), dtype=np.float32)
            targets = (None if self._access.purpose == "fit_only_cpu_preparation"
                       else np.empty((size, HORIZON_SAMPLES, 4), dtype=np.float32))
            global_offsets = np.empty(size, dtype=np.int64)
            origin_indices = np.empty(size, dtype=np.int64)
            seg_ids = np.searchsorted(self._cumulative, selection, side="right")
            before = np.concatenate((np.asarray([0], dtype=np.int64), self._cumulative[:-1]))
            for j, (rank, seg_index) in enumerate(zip(selection, seg_ids)):
                segment = self._segments[int(seg_index)]
                local_origin_rank = int(rank) - int(before[int(seg_index)])
                global_offset = segment.global_origin_offset + local_origin_rank
                origin = int(self._binding.origin_values[global_offset])
                if origin != 399 + local_origin_rank:
                    raise ValueError("loaded fit origin differs from the immutable canonical index")
                start = origin - (CONTEXT_SAMPLES - 1)
                stop = origin + HORIZON_SAMPLES + 1
                if start < 0 or stop > len(segment.poses) or stop - start != WINDOW_SAMPLES:
                    raise ValueError("fit window crosses a segment boundary or has the wrong shift")
                contexts[j] = segment.poses[start:origin + 1]
                if targets is not None:
                    targets[j] = segment.poses[origin + 1:stop]
                global_offsets[j] = global_offset
                origin_indices[j] = origin
            contexts.setflags(write=False)
            if targets is not None:
                targets.setflags(write=False)
            global_offsets.setflags(write=False)
            origin_indices.setflags(write=False)
            yield FitBatch(
                self.fold_id, self.role_mode, schedule.kind, schedule.seed,
                schedule.arm_id, schedule.schedule_sha256,
                schedule.origin_identity_sha256, first,
                global_offsets, origin_indices, contexts, targets)

    def safe_aggregate_receipt(self) -> dict:
        receipt = self._audit.safe_aggregate(self._access, self.fold_id, self.total_origin_count)
        receipt["fit_origin_pool_identity_sha256"] = self.origin_pool_sha256
        receipt["sampler_schedules"] = [
            {"kind": schedule.kind, "seed": schedule.seed, "arm_id": schedule.arm_id,
             "window_count": len(schedule.ranks),
             "schedule_sha256": schedule.schedule_sha256,
             "origin_identity_sha256": schedule.origin_identity_sha256}
            for schedule in (*self.training_schedules, self.standardizer_schedule)
        ]
        return receipt


@contextmanager
def _verified_fit_source_file(reader, binding, deadline: float):
    """Verify opaque bytes and open h5py on the same bounded-time file handle."""
    current = reader._file_identity(binding.source_path)
    if current != binding.source_identity:
        raise ValueError("source HDF5 identity changed after metadata and approval checks")
    stream = binding.source_path.open("rb")
    try:
        opened = os.fstat(stream.fileno())
        path_stat = binding.source_path.stat()
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        path_identity = (path_stat.st_dev, path_stat.st_ino, path_stat.st_size, path_stat.st_mtime_ns)
        if opened_identity != binding.source_identity or path_identity != binding.source_identity:
            raise ValueError("source HDF5 inode changed before its fixed byte verification")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if time.monotonic() > deadline:
                raise TimeoutError("source checksum exceeded the approved loader time allowance")
            digest.update(block)
        if time.monotonic() > deadline:
            raise TimeoutError("source checksum exceeded the approved loader time allowance")
        opened_after = os.fstat(stream.fileno())
        path_after = binding.source_path.stat()
        opened_after_identity = (opened_after.st_dev, opened_after.st_ino,
                                 opened_after.st_size, opened_after.st_mtime_ns)
        path_after_identity = (path_after.st_dev, path_after.st_ino,
                               path_after.st_size, path_after.st_mtime_ns)
        if (opened_after_identity != binding.source_identity or
                path_after_identity != binding.source_identity):
            raise ValueError("source HDF5 identity changed during its fixed byte verification")
        if digest.hexdigest() != SOURCE_HDF5_SHA256:
            raise ValueError("source HDF5 bytes differ from the fixed split reference")
        stream.seek(0)
        with h5py.File(stream, "r") as source_file:
            yield source_file
    finally:
        stream.close()


def load_approved_fit_pool(fold_id: int) -> FitPool:
    """Load one approved fit-only pool after every fixed gate passes.

    The new plan and approval are separate fixed private files. No role, seed,
    path, sampler, recipe, window count, or resource cap can be supplied here.
    """
    if type(fold_id) is not int:
        raise ValueError("fold_id must be an integer (booleans are not accepted)")
    started = time.monotonic()
    reader = _load_pinned_reader()
    binding, access = _load_fixed_plan_and_approval()
    launch = _validate_trusted_cpu_prep_launch(access)
    if fold_id not in access.fold_ids:
        raise PermissionError("requested fold is not authorized by the frozen fit-pool plan")
    _claim_production_fold(access, launch, fold_id)
    launch_remaining = max(
        0.0, (launch.expires_at - datetime.now(timezone.utc)).total_seconds()
    )
    deadline = min(started + access.max_loader_seconds,
                   time.monotonic() + launch_remaining)
    if time.monotonic() > deadline:
        raise TimeoutError("metadata and approval validation exceeded the fixed loader time allowance")
    # Generate and validate all approved schedules against metadata before the
    # source-file checksum or HDF5 open.
    prepared = _prepare_fold_schedules(binding, access, fold_id)
    if time.monotonic() > deadline:
        raise TimeoutError("fit-pool schedule preparation exceeded the fixed loader time allowance")
    with _verified_fit_source_file(reader, binding, deadline) as source_file:
        pool = _load_fit_pool_from_open_file(
            source_file, binding, access, fold_id, prepared,
            source_hash_verified=True, deadline=deadline)
    return pool


def preflight_approved_fit_pool(fold_id: int) -> FitPoolPreflight:
    """Validate CPU-preparation plan and exact schedule without source values.

    This performs only the bounded reader's fixed metadata preflight, plan and
    approval validation, and deterministic in-memory schedule generation. It
    does not checksum or open the source HDF5. The source-consuming loader
    repeats all fixed checks immediately before its opaque checksum.
    """
    if type(fold_id) is not int:
        raise ValueError("fold_id must be an integer (booleans are not accepted)")
    binding, access = _load_fixed_plan_and_approval()
    if access.purpose != "fit_only_cpu_preparation" or access.role_mode != "outer_fit10":
        raise PermissionError("CPU consumer requires the separately locked outer-fit preparation purpose")
    if fold_id not in access.fold_ids:
        raise PermissionError("requested fold is not authorized by the frozen CPU-preparation plan")
    _, _, _, schedules, standardizer = _prepare_fold_schedules(binding, access, fold_id)
    if schedules:
        raise AssertionError("CPU preparation unexpectedly produced training schedules")
    return FitPoolPreflight(
        fold_id=fold_id, purpose=access.purpose, role_mode=access.role_mode,
        device=access.device, cuda_contexts=access.cuda_contexts,
        optimizer_updates=access.optimizer_updates,
        gpu_updates=0, fitting_operations_allowed=access.fitting_operations_allowed,
        source_hdf5_relative_path=SOURCE_HDF5_RELATIVE_PATH,
        source_hdf5_sha256_reference=SOURCE_HDF5_SHA256,
        plan_sha256=access.plan_sha256, approval_sha256=access.approval_sha256,
        recipe_sha256=access.recipe_sha256,
        cpu_consumer_source_sha256=access.cpu_consumer_source_sha256 or "",
        standardizer_seed=access.standardizer_seed,
        standardizer_origin_count=len(standardizer.ranks),
        standardizer_schedule_sha256=standardizer.schedule_sha256,
        standardizer_origin_identity_sha256=standardizer.origin_identity_sha256,
        max_pose_bytes=access.max_pose_bytes, max_logical_read_bytes=access.max_chunk_bytes,
        max_loader_seconds=access.max_loader_seconds,
        cpu_max_wall_seconds=access.cpu_max_wall_seconds,
        cpu_max_rss_bytes=access.cpu_max_rss_bytes,
        cpu_min_free_memory_bytes=access.cpu_min_free_memory_bytes,
        cpu_min_free_disk_bytes=access.cpu_min_free_disk_bytes,
    )


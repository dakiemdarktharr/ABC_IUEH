"""Synthetic fit-pool and fixed-metadata checks; no production payload reads."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from datetime import datetime, timedelta, timezone

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = ROOT / "ABC_IUEH"
STAGING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGING_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import q1_fit_loader as loader


ROWS_PER_SEGMENT = 10_000
SAMPLES_PER_SEGMENT = ROWS_PER_SEGMENT + 419


def _pipeline_dict():
    from abc_iueh.config import PipelineConfig
    return PipelineConfig().to_dict()


def _synthetic_binding(tmp: Path):
    reader = loader._load_pinned_reader()
    users = tuple(f"fit_user_{i:02d}" for i in range(15))
    folds = (users[:5], users[5:10], users[10:])
    user_fold = {user: fold for fold, group in enumerate(folds) for user in group}
    user_sessions = {user: {"session_0": ("0000",)} for user in users}
    session_windows = {(user, "session_0"): ROWS_PER_SEGMENT for user in users}
    expected = reader.origin_inventory.ExpectedTrain(
        users=users, user_sessions=user_sessions, user_fold=user_fold,
        fold_validation_order=folds, expected_windows=15 * ROWS_PER_SEGMENT,
        session_windows=session_windows,
    )
    metadata = {
        "semantic_config": {"source_binding": {
            "authenticated_record_inventory_sha256": "1" * 64,
        }},
        "pipeline_config": _pipeline_dict(),
    }
    parent = SimpleNamespace(
        expected_train=expected,
        metadata=metadata,
        metadata_sha256={"legacy_receipt": "2" * 64, "inventory": "3" * 64},
        fold_digest="4" * 64,
        split_sha256=loader.SPLIT_SHA256,
    )
    rows = []
    origin_values = []
    offset = 0
    for fold, group in enumerate(folds):
        for user in sorted(group):
            rows.append(reader.OriginRow(
                user, "session_0", "0000", fold, offset, ROWS_PER_SEGMENT))
            origin_values.extend(range(399, 399 + ROWS_PER_SEGMENT))
            offset += ROWS_PER_SEGMENT
    ownership = reader._inner_ownership(expected)
    return reader.MetadataBinding(
        parent=parent,
        source_path=tmp / "synthetic.h5",
        source_identity=(0, 0, 0, 0),
        source_path_evidence_sha256=reader.SOURCE_PATH_EVIDENCE_SHA256,
        origin_receipt_sha256=reader.ORIGIN_RECEIPT_SHA256,
        origin_map_sha256=reader.ORIGIN_MAP_SHA256,
        origin_binary_sha256=reader.ORIGIN_BINARY_SHA256,
        origin_rows=tuple(rows),
        origin_values=np.asarray(origin_values, dtype="<i8"),
        ownership=ownership,
        ownership_plan_sha256=reader._ownership_digest(ownership),
    )


def _synthetic_quaternions(n=SAMPLES_PER_SEGMENT):
    angle = np.arange(n, dtype=np.float64) * 0.0001
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = np.cos(angle / 2).astype(np.float32)
    q[:, 3] = np.sin(angle / 2).astype(np.float32)
    return q


def _write_synthetic_h5(path: Path, binding, *, alias_fit_to_final=False,
                         soft_fit_link=False, virtual_fit=False,
                         external_fit=False, timestamp_gap=False,
                         bad_quaternion_norm=False, bad_quaternion_sign=False):
    cfg_json = json.dumps(binding.parent.metadata["pipeline_config"], sort_keys=True)
    q_values = _synthetic_quaternions()
    if bad_quaternion_norm:
        q_values[1024] *= 2.0
    if bad_quaternion_sign:
        q_values[1024] *= -1.0
    with h5py.File(path, "w") as store:
        attrs = store.attrs
        attrs["schema_version"] = 1
        attrs["source"] = "whoisalyx"
        attrs["license"] = "CC-BY-NC-SA-4.0"
        attrs["quaternion_order"] = "wxyz"
        attrs["coordinate_frame"] = "source-native OpenVR; no handedness transform"
        attrs["sample_rate_hz"] = 100.0
        attrs["timing_dispute"] = "20 samples / 100 Hz / 100 ms unresolved"
        attrs["interpolation"] = "offline bracketing SLERP, not a zero-delay sensor pipeline"
        attrs["source_receipt_sha256"] = binding.parent.metadata_sha256["legacy_receipt"]
        attrs["source_inventory_file_sha256"] = binding.parent.metadata_sha256["inventory"]
        attrs["source_record_inventory_sha256"] = binding.parent.metadata["semantic_config"][
            "source_binding"]["authenticated_record_inventory_sha256"]
        attrs["config_json"] = cfg_json
        users_group = store.create_group("users")
        for user in binding.parent.expected_train.users:
            user_group = users_group.create_group(user)
            session = user_group.create_group("sessions").create_group("session_0")
            segment = session.create_group("segments").create_group("0000")
            timestamps = np.arange(SAMPLES_PER_SEGMENT, dtype=np.float64) / 100.0
            if timestamp_gap:
                timestamps[2048:] += 0.08
            segment.create_dataset("timestamps", data=timestamps, chunks=(2048,),
                                   compression="gzip", compression_opts=1)
            if soft_fit_link and user == binding.ownership[0]["outer_fit"][0]:
                segment["quaternion"] = h5py.SoftLink("/users/synthetic_final/q")
            elif virtual_fit and user == binding.ownership[0]["outer_fit"][0]:
                layout = h5py.VirtualLayout(shape=(SAMPLES_PER_SEGMENT, 4), dtype=np.float32)
                source = h5py.VirtualSource(str(path.with_name("missing_virtual_source.h5")),
                                            "/q", shape=(SAMPLES_PER_SEGMENT, 4))
                layout[:, :] = source[:, :]
                segment.create_virtual_dataset("quaternion", layout)
            elif external_fit and user == binding.ownership[0]["outer_fit"][0]:
                segment.create_dataset(
                    "quaternion", shape=(SAMPLES_PER_SEGMENT, 4), dtype=np.float32,
                    external=[(path.with_suffix(".external.raw").name, 0, h5py.h5f.UNLIMITED)])
            else:
                segment.create_dataset("quaternion", data=q_values, chunks=(2048, 4),
                                       compression="gzip", compression_opts=1)
        for user in ("synthetic_development", "synthetic_final"):
            group = users_group.create_group(user)
            group.create_dataset("q", data=q_values[:32], chunks=(32, 4))
        if alias_fit_to_final:
            selected = binding.ownership[0]["outer_fit"][0]
            ds = store[f"users/{selected}/sessions/session_0/segments/0000/quaternion"]
            users_group["synthetic_final"].create_group("aliases")["fit_q"] = ds


def _synthetic_access(binding, *, role_mode="outer_fit10", purpose=None,
                     cpu_consumer_path=None):
    reader = loader._load_pinned_reader()
    purpose = purpose or ("train_only_timing_prefix" if role_mode == "outer_fit10"
                          else "inner_fit_pool_diagnostic")
    cpu_prep = purpose == "fit_only_cpu_preparation"
    cpu_consumer_path = cpu_consumer_path or loader.CPU_CONSUMER_PATH
    artifacts = loader._current_artifact_hashes(reader)
    recipe = {"model": "synthetic-only", "steps_per_schedule": 1}
    recipe_bytes = json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode()
    plan_doc = {
        "schema_version": 1,
        "kind": "q1_fit_pool_plan",
        "status": ("FROZEN_BEFORE_CPU_PREPARATION" if cpu_prep
                   else "FROZEN_BEFORE_PREFIX"),
        "plan_id": "q1-fit-pool-plan-synthetic-v1",
        "purpose": purpose,
        "role_mode": role_mode,
        "data_bindings": loader._expected_data_bindings(reader, binding),
        "sampling": ({
            "numpy_version": np.__version__,
            "fold_ids": [0, 1, 2],
            "standardizer_sampler": loader.STANDARDIZER_SAMPLER,
            "standardizer_seed": loader.STANDARDIZER_SEED,
            "standardizer_origins_per_fold": 65_536,
            "batch_size": 128,
        } if cpu_prep else {
            "training_sampler": loader.TRAINING_SAMPLER,
            "sampler_version": loader.SAMPLER_VERSION,
            "numpy_version": np.__version__,
            "fold_ids": [0, 1, 2],
            "seed_values": list(loader.TRAINING_SEEDS),
            "arm_ids": list(loader.CANONICAL_ARMS),
            "windows_per_schedule": 257,
            "standardizer_sampler": loader.STANDARDIZER_SAMPLER,
            "standardizer_seed": loader.STANDARDIZER_SEED,
            "standardizer_origins_per_fold": 65_536,
            "batch_size": 128,
        }),
        "resource_policy": {
            "max_pose_array_bytes": loader.MAX_POSE_ARRAY_BYTES,
            "max_logical_read_bytes": 16_384,
            "max_batch_size": 128,
            "max_windows_per_schedule": 0 if cpu_prep else 257,
            "max_total_training_windows": 0 if cpu_prep else 3 * 2 * 5 * 257,
            "max_loader_seconds": 600,
        },
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "reviewed_artifacts": artifacts,
    }
    if cpu_prep:
        plan_doc.update({
            "cpu_consumer_source_sha256": loader._sha256_file(cpu_consumer_path),
            "fitting_operations_allowed": ["population_feature_standardizer"],
            "device_policy": {
                "device": "CPU", "cuda_contexts": 0,
                "gpu_updates": 0, "optimizer_updates": 0,
            },
            "cpu_resource_policy": {
                "max_wall_seconds": 900,
                "max_rss_bytes": 2 * 1024 * 1024 * 1024,
                "min_free_memory_bytes": 2 * 1024 * 1024 * 1024,
                "min_free_disk_bytes": 2 * 1024 * 1024 * 1024,
            },
        })
    plan_raw = json.dumps(plan_doc, sort_keys=True, separators=(",", ":")).encode()
    approval_doc = {
        "schema_version": 1,
        "kind": "q1_fit_pool_approval",
        "status": "APPROVED_AFTER_INDEPENDENT_REVIEW",
        "approval_id": "q1-fit-pool-approval-synthetic-v1",
        "approved_by_role": "coordinator",
        "approved_utc": "2026-10-29T12:00:00+00:00",
        "plan_id": plan_doc["plan_id"],
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "recipe_sha256": plan_doc["recipe_sha256"],
        "reviewed_artifacts": artifacts,
        "independent_review_sha256": "b" * 64,
    }
    approval_raw = json.dumps(approval_doc, sort_keys=True, separators=(",", ":")).encode()
    with mock.patch.object(loader, "CPU_CONSUMER_PATH", cpu_consumer_path):
        access = loader._validate_plan_and_approval(
            reader, binding, plan_doc, plan_raw, approval_doc, approval_raw)
    return reader, plan_doc, plan_raw, approval_doc, approval_raw, access


def _cpu_plan_doc_from_timing(plan_doc):
    doc = json.loads(json.dumps(plan_doc))
    doc["purpose"] = "fit_only_cpu_preparation"
    doc["status"] = "FROZEN_BEFORE_CPU_PREPARATION"
    doc["sampling"] = {
        "numpy_version": np.__version__,
        "fold_ids": [0, 1, 2],
        "standardizer_sampler": loader.STANDARDIZER_SAMPLER,
        "standardizer_seed": loader.STANDARDIZER_SEED,
        "standardizer_origins_per_fold": loader.STANDARDIZER_ORIGINS_PER_FOLD,
        "batch_size": loader.MAX_BATCH_SIZE,
    }
    doc["resource_policy"]["max_windows_per_schedule"] = 0
    doc["resource_policy"]["max_total_training_windows"] = 0
    doc.update({
        "cpu_consumer_source_sha256": "0" * 64,
        "fitting_operations_allowed": ["population_feature_standardizer"],
        "device_policy": {
            "device": "CPU", "cuda_contexts": 0,
            "gpu_updates": 0, "optimizer_updates": 0,
        },
        "cpu_resource_policy": {
            "max_wall_seconds": 900,
            "max_rss_bytes": 2 * 1024 * 1024 * 1024,
            "min_free_memory_bytes": 2 * 1024 * 1024 * 1024,
            "min_free_disk_bytes": 2 * 1024 * 1024 * 1024,
        },
    })
    return doc


def _synthetic_trusted_launch(access, *, launch_id="q1-cpu-launch-synthetic-001",
                              now=None, expiry_delta=timedelta(seconds=300), **overrides):
    """Test-only environment capability; production code never creates one."""
    now = now or datetime.now(timezone.utc)
    document = {
        "purpose": "fit_only_cpu_preparation",
        "role_mode": "outer_fit10",
        "plan_sha256": access.plan_sha256,
        "approval_sha256": access.approval_sha256,
        "recipe_sha256": access.recipe_sha256,
        "independent_review_sha256": access.independent_review_sha256,
        "loader_source_sha256": access.reviewed_artifacts["fit_loader_code_sha256"],
        "consumer_source_sha256": access.cpu_consumer_source_sha256,
        "launch_id": launch_id,
        "expires_utc": (now + expiry_delta).isoformat(),
    }
    document.update(overrides)
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _legacy_root_bridge_fixture(tmp: Path):
    reader = loader._load_pinned_reader()
    records = []
    for i in range(70):
        records.append({
            "source_path": f"fixture/source_{i:02d}",
            "source_revision": "fixture-revision",
            "user_id": "fixture-user",
            "session_id": "fixture-session",
            "relative_path": f"part_{i:02d}.csv",
            "url": f"https://fixture.invalid/part_{i:02d}.csv",
            "expected_bytes": 1000 + i,
            "md5": hashlib.md5(f"fixture-{i}".encode()).hexdigest(),
        })
    session_parts = sorted(row["source_path"] for row in records)
    receipt_rows = [dict(row, expected_session_paths=session_parts) for row in records]
    inventory = {"records": records}
    legacy = {"source_manifest_sha256": loader.LEGACY_INVENTORY_SHA256,
              "records": [dict(row) for row in receipt_rows]}
    semantic_digest = loader._selected_record_digest(records)
    versioned = {
        "schema_version": 2,
        "resealed_from": {
            "schema_version": 1,
            "source_manifest_sha256": loader.LEGACY_INVENTORY_SHA256,
            "source_receipt_sha256": loader.LEGACY_RECEIPT_SHA256,
        },
        "record_inventory_sha256": semantic_digest,
        "inventory_bindings": [{
            "manifest_sha256": loader.LEGACY_INVENTORY_SHA256,
            "record_inventory_sha256": semantic_digest,
        }],
        "records": [dict(row) for row in receipt_rows],
    }
    metadata_bridge = {
        "legacy_metadata_bytes_unchanged": True,
        "input_byte_hashes_before": {
            "data\\alyx_inventory.json": loader.LEGACY_INVENTORY_SHA256,
            "data\\download_receipt.json": loader.LEGACY_RECEIPT_SHA256,
        },
        "input_byte_hashes_after": {
            "data\\alyx_inventory.json": loader.LEGACY_INVENTORY_SHA256,
            "data\\download_receipt.json": loader.LEGACY_RECEIPT_SHA256,
        },
        "HDF5_and_split_rewritten": False,
        "signal_open_spy_attempts": 0,
        "human_signal_rows_read": 0,
        "result": {
            "inventory_file_sha256": loader.LEGACY_INVENTORY_SHA256,
            "legacy_receipt_file_sha256": loader.LEGACY_RECEIPT_SHA256,
            "new_receipt_file_sha256": loader.VERSIONED_RECEIPT_SHA256,
            "new_receipt_schema_version": 2,
            "record_inventory_sha256": semantic_digest,
            "inventory_records": 70,
            "receipt_records": 70,
            "legacy_receipt_validated": True,
            "recording_payloads_read": 0,
        },
    }
    source_binding = {
        "source_hdf5_sha256_reference": loader.SOURCE_HDF5_SHA256,
        "authenticated_record_inventory_sha256": semantic_digest,
        "source_inventory_semantic_id": loader.SOURCE_INVENTORY_SEMANTIC_ID,
        "inventory_records": 70,
    }
    code_bindings = {
        "ABC_IUEH/src/abc_iueh/sources.py": loader.SOURCE_HELPER_SHA256,
        "workflow_runtime/staging/semantic_rebind/rebind.py":
            loader.SEMANTIC_REBIND_SHA256,
    }
    metadata = {
        "inventory": inventory,
        "legacy_receipt": legacy,
        "versioned_receipt": versioned,
        "metadata_bridge": metadata_bridge,
        "semantic_config": {"source_binding": source_binding,
                            "code_byte_bindings": code_bindings},
        "semantic_receipt": {"source_validation": dict(source_binding)},
        "pipeline_config": _pipeline_dict(),
    }
    hashes = {
        "inventory": loader.LEGACY_INVENTORY_SHA256,
        "legacy_receipt": loader.LEGACY_RECEIPT_SHA256,
        "versioned_receipt": loader.VERSIONED_RECEIPT_SHA256,
        "metadata_bridge": loader.METADATA_BRIDGE_SHA256,
        "semantic_config": loader.SEMANTIC_CONFIG_SHA256,
        "semantic_receipt": loader.SEMANTIC_RECEIPT_SHA256,
        "pipeline_config": loader.PIPELINE_CONFIG_SHA256,
        "arc_config": loader.ARC_CONFIG_SHA256,
        "split": loader.SPLIT_SHA256,
        "source_helper": loader.SOURCE_HELPER_SHA256,
    }
    parent = SimpleNamespace(
        metadata=metadata, metadata_sha256=hashes,
        source_hdf5_sha256_reference=loader.SOURCE_HDF5_SHA256,
        split_sha256=loader.SPLIT_SHA256,
        expected_train=SimpleNamespace(users=tuple(f"u{i:02d}" for i in range(15)),
                                       expected_windows=3_330_312),
    )
    binding = SimpleNamespace(
        parent=parent, source_path=reader.SOURCE_HDF5_PATH,
        source_path_evidence_sha256=loader.SOURCE_PATH_EVIDENCE_SHA256,
        origin_receipt_sha256=reader.ORIGIN_RECEIPT_SHA256,
        origin_map_sha256=reader.ORIGIN_MAP_SHA256,
        origin_binary_sha256=reader.ORIGIN_BINARY_SHA256,
    )
    attrs = {
        "schema_version": 1,
        "source": "whoisalyx",
        "license": "CC-BY-NC-SA-4.0",
        "quaternion_order": "wxyz",
        "coordinate_frame": "source-native OpenVR; no handedness transform",
        "sample_rate_hz": 100.0,
        "timing_dispute": "20 samples / 100 Hz / 100 ms unresolved",
        "interpolation": "offline bracketing SLERP, not a zero-delay sensor pipeline",
        "source_receipt_sha256": loader.LEGACY_RECEIPT_SHA256,
        "config_json": json.dumps(metadata["pipeline_config"], sort_keys=True),
    }
    return reader, binding, attrs, semantic_digest


class _FakeRoot:
    def __init__(self, attrs):
        self.attrs = dict(attrs)
    def keys(self):
        return ("users",)


class LegacyRootBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="q1-legacy-root-bridge-", dir=STAGING_ROOT))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.reader, self.binding, self.attrs, self.semantic_digest = _legacy_root_bridge_fixture(self.tmp)
        self.expected_digest_patch = mock.patch.object(
            loader, "LEGACY_RECORD_DIGEST_SHA256", self.semantic_digest)
        self.expected_digest_patch.start()
        self.addCleanup(self.expected_digest_patch.stop)

    def test_exact_legacy_receipt_chain_derives_only_the_two_missing_root_attrs(self):
        source = _FakeRoot(self.attrs)
        mode = loader._validate_source_root_with_legacy_bridge(
            self.reader, source, self.binding, source_hash_verified=True)
        self.assertEqual(mode, "legacy_receipt_chain_derived")
        view = loader._RootMetadataWithLegacyDerivations(source, {
            "source_inventory_file_sha256": loader.LEGACY_INVENTORY_SHA256,
            "source_record_inventory_sha256": self.semantic_digest,
        })
        self.assertEqual(view.attrs.get("config_json"), self.attrs["config_json"])
        self.assertEqual(view.attrs.get("source_receipt_sha256"), self.attrs["source_receipt_sha256"])
        self.assertEqual(view.attrs.get("source_inventory_file_sha256"), loader.LEGACY_INVENTORY_SHA256)
        self.assertEqual(tuple(view.keys()), ("users",))
        self.assertNotIn("source_inventory_file_sha256", source.attrs)
        self.assertNotIn("source_record_inventory_sha256", source.attrs)

    def test_complete_modern_attributes_stay_on_the_original_reader_validator(self):
        attrs = dict(self.attrs)
        attrs["source_inventory_file_sha256"] = loader.LEGACY_INVENTORY_SHA256
        attrs["source_record_inventory_sha256"] = self.semantic_digest
        mode = loader._validate_source_root_with_legacy_bridge(
            self.reader, _FakeRoot(attrs), self.binding, source_hash_verified=False)
        self.assertEqual(mode, "direct_root_attributes")

    def test_partial_pair_and_present_null_are_rejected_without_legacy_fallback(self):
        partial = dict(self.attrs, source_inventory_file_sha256=loader.LEGACY_INVENTORY_SHA256)
        with self.assertRaisesRegex(PermissionError, "partially absent"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(partial), self.binding, source_hash_verified=True)
        present_null = dict(self.attrs, source_inventory_file_sha256=None,
                             source_record_inventory_sha256=self.semantic_digest)
        with self.assertRaisesRegex(ValueError, "source_inventory_file_sha256"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(present_null), self.binding, source_hash_verified=True)

    def test_wrong_nonnull_value_and_wrong_receipt_pointer_are_denied(self):
        mismatch = dict(self.attrs, source_inventory_file_sha256="0" * 64,
                        source_record_inventory_sha256=self.semantic_digest)
        with self.assertRaisesRegex(ValueError, "source_inventory_file_sha256"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(mismatch), self.binding, source_hash_verified=True)
        wrong_receipt = dict(self.attrs, source_receipt_sha256="0" * 64)
        with self.assertRaisesRegex(PermissionError, "exact pinned receipt"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(wrong_receipt), self.binding, source_hash_verified=True)

    def test_wrong_same_fd_hash_or_parent_source_reference_denies_legacy_derivation(self):
        with self.assertRaisesRegex(PermissionError, "same-FD"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding, source_hash_verified=False)
        original = self.binding.parent.source_hdf5_sha256_reference
        self.binding.parent.source_hdf5_sha256_reference = "0" * 64
        with self.assertRaisesRegex(PermissionError, "source path, split"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding, source_hash_verified=True)
        self.binding.parent.source_hdf5_sha256_reference = original

    def test_bad_metadata_digest_row_count_or_session_parts_denies_bridge(self):
        original_hash = self.binding.parent.metadata_sha256["inventory"]
        self.binding.parent.metadata_sha256["inventory"] = "0" * 64
        with self.assertRaisesRegex(PermissionError, "metadata bytes"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding, source_hash_verified=True)
        self.binding.parent.metadata_sha256["inventory"] = original_hash

        original = self.binding.parent.metadata["legacy_receipt"]["records"]
        self.binding.parent.metadata["legacy_receipt"]["records"] = original[:-1]
        with self.assertRaisesRegex(PermissionError, "exact 70"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding, source_hash_verified=True)
        self.binding.parent.metadata["legacy_receipt"]["records"] = original

        original_session_paths = original[0]["expected_session_paths"]
        original[0]["expected_session_paths"] = ["wrong-part.csv"]
        with self.assertRaisesRegex(PermissionError, "session-part list"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding, source_hash_verified=True)
        original[0]["expected_session_paths"] = original_session_paths

    def test_wrong_record_digest_and_semantic_binding_deny_bridge(self):
        with mock.patch.object(loader, "LEGACY_RECORD_DIGEST_SHA256", "0" * 64):
            with self.assertRaisesRegex(PermissionError, "record digest"):
                loader._validate_source_root_with_legacy_bridge(
                    self.reader, _FakeRoot(self.attrs), self.binding,
                    source_hash_verified=True)
        self.binding.parent.metadata["semantic_config"]["source_binding"][
            "source_inventory_semantic_id"] = "0" * 64
        with self.assertRaisesRegex(PermissionError, "semantic rebind"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding, source_hash_verified=True)

    def test_versioned_v2_receipt_must_bind_exact_legacy_bytes_and_records(self):
        receipt = self.binding.parent.metadata["versioned_receipt"]
        mutations = (
            ("schema_version", 1),
            ("record_inventory_sha256", "0" * 64),
            ("inventory_bindings", [{
                "manifest_sha256": loader.LEGACY_INVENTORY_SHA256,
                "record_inventory_sha256": "0" * 64,
            }]),
            ("resealed_from", {
                "schema_version": 1,
                "source_manifest_sha256": "0" * 64,
                "source_receipt_sha256": loader.LEGACY_RECEIPT_SHA256,
            }),
            ("resealed_from", {
                "schema_version": 1,
                "source_manifest_sha256": loader.LEGACY_INVENTORY_SHA256,
                "source_receipt_sha256": "0" * 64,
            }),
            ("resealed_from", {
                "schema_version": 2,
                "source_manifest_sha256": loader.LEGACY_INVENTORY_SHA256,
                "source_receipt_sha256": loader.LEGACY_RECEIPT_SHA256,
            }),
        )
        for field, wrong_value in mutations:
            with self.subTest(field=field):
                original = receipt[field]
                receipt[field] = wrong_value
                with self.assertRaisesRegex(PermissionError, "versioned receipt chain"):
                    loader._validate_source_root_with_legacy_bridge(
                        self.reader, _FakeRoot(self.attrs), self.binding,
                        source_hash_verified=True)
                receipt[field] = original

    def test_g1_bridge_guard_counters_are_read_from_the_authenticated_top_level(self):
        bridge = self.binding.parent.metadata["metadata_bridge"]
        for field in ("signal_open_spy_attempts", "human_signal_rows_read"):
            original = bridge[field]
            bridge[field] = 1
            with self.assertRaisesRegex(PermissionError, "G1 metadata bridge"):
                loader._validate_source_root_with_legacy_bridge(
                    self.reader, _FakeRoot(self.attrs), self.binding,
                    source_hash_verified=True)
            bridge[field] = original

        result = bridge["result"]
        original = result["new_receipt_schema_version"]
        result["new_receipt_schema_version"] = 1
        with self.assertRaisesRegex(PermissionError, "G1 metadata bridge"):
            loader._validate_source_root_with_legacy_bridge(
                self.reader, _FakeRoot(self.attrs), self.binding,
                source_hash_verified=True)
        result["new_receipt_schema_version"] = original


class FitLoaderSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="q1-fit-loader-synthetic-", dir=STAGING_ROOT))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.binding = _synthetic_binding(self.tmp)
        self.source_path = self.tmp / "synthetic.h5"
        _write_synthetic_h5(self.source_path, self.binding)
        self.reader, self.plan_doc, self.plan_raw, self.approval_doc, self.approval_raw, self.access = (
            _synthetic_access(self.binding))

    def _load_synthetic_pool(self, *, fold_id=0, access=None, path=None):
        access = access or self.access
        path = path or self.source_path
        prepared = loader._prepare_fold_schedules(self.binding, access, fold_id)
        deadline = time_monotonic_plus(access.max_loader_seconds)
        with h5py.File(path, "r") as source_file:
            return loader._load_fit_pool_from_open_file(
                source_file, self.binding, access, fold_id, prepared,
                source_hash_verified=False, deadline=deadline)

    def test_plan_and_approval_lock_roles_source_recipe_sampler_and_resources(self):
        self.assertEqual(self.access.role_mode, "outer_fit10")
        self.assertEqual(self.access.purpose, "train_only_timing_prefix")
        self.assertEqual(self.access.seed_values, loader.TRAINING_SEEDS)
        self.assertEqual(self.access.arm_ids, loader.CANONICAL_ARMS)
        self.assertEqual(self.access.windows_per_schedule, 257)
        self.assertEqual(self.access.standardizer_seed, 20261028)
        self.assertEqual(self.access.max_pose_bytes, 128 * 1024 * 1024)
        self.assertEqual(self.access.max_chunk_bytes, 16_384)
        self.assertEqual(self.access.plan_sha256, hashlib.sha256(self.plan_raw).hexdigest())

        wrong_source = json.loads(json.dumps(self.plan_doc))
        wrong_source["data_bindings"]["source_partition"] = "development"
        bad_raw = json.dumps(wrong_source, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(PermissionError, "source, split, fold, origin"):
            loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_source, bad_raw,
                self.approval_doc, self.approval_raw)

        wrong_sampler = json.loads(json.dumps(self.plan_doc))
        wrong_sampler["sampling"]["training_sampler"] = "caller_indices"
        bad_raw = json.dumps(wrong_sampler, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(PermissionError, "training sampler"):
            loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_sampler, bad_raw,
                self.approval_doc, self.approval_raw)

        wrong_recipe = json.loads(json.dumps(self.plan_doc))
        wrong_recipe["recipe"]["steps_per_schedule"] = 2
        bad_raw = json.dumps(wrong_recipe, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(PermissionError, "recipe bytes differ"):
            loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_recipe, bad_raw,
                self.approval_doc, self.approval_raw)

        wrong_seed = json.loads(json.dumps(self.plan_doc))
        wrong_seed["sampling"]["seed_values"] = [True, 20261027]
        bad_raw = json.dumps(wrong_seed, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "exact protocol training seeds"):
            loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_seed, bad_raw,
                self.approval_doc, self.approval_raw)

        wrong_memory = json.loads(json.dumps(self.plan_doc))
        wrong_memory["resource_policy"]["max_pose_array_bytes"] = loader.MAX_POSE_ARRAY_BYTES + 1
        bad_raw = json.dumps(wrong_memory, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "max_pose_array_bytes"):
            loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_memory, bad_raw,
                self.approval_doc, self.approval_raw)

        wrong_role = json.loads(json.dumps(self.plan_doc))
        wrong_role["role_mode"] = "outer_validation5"
        bad_raw = json.dumps(wrong_role, sort_keys=True, separators=(",", ":")).encode()
        with self.assertRaisesRegex(PermissionError, "purpose does not authorize"):
            loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_role, bad_raw,
                self.approval_doc, self.approval_raw)

    def test_strict_json_rejects_duplicate_and_nonfinite_values_recursively(self):
        malformed = (
            b'{"a":1,"a":2}',
            b'{"outer":{"a":1,"a":2}}',
            b'{"outer":[NaN]}',
            b'{"outer":{"value":Infinity}}',
            b'{"outer":[-Infinity]}',
            b'{"outer":{"value":1e999}}',
        )
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "malformed JSON"):
                loader._strict_json(raw, "synthetic plan")
        self.assertEqual(loader._strict_json(b'{"outer":{"value":1.25}}', "fixture"),
                         {"outer": {"value": 1.25}})

    def test_outer_fit10_and_explicit_inner_fit8_diagnostic_only(self):
        self.assertEqual(len(loader._role_users(self.binding, 0, "outer_fit10")), 10)
        self.assertEqual(len(loader._role_users(
            self.binding, 0, "inner_fit8_diagnostic")), 8)
        with self.assertRaises(PermissionError):
            loader._role_users(self.binding, 0, "outer_validation")
        with self.assertRaises(ValueError):
            loader._role_users(self.binding, True, "outer_fit10")
        with self.assertRaises(ValueError):
            loader._role_users(self.binding, 9, "outer_fit10")

        allowed = set(loader._role_users(self.binding, 0, "outer_fit10"))
        audit = loader.ReadAudit(file_id=0, fit_role_users=len(allowed))
        with h5py.File(self.source_path, "r") as source_file:
            users_group = source_file["users"]
            audit.file_id = int(source_file.id.id)
            with self.assertRaisesRegex(PermissionError, "outside the approved fit role"):
                loader._get_selected_child(
                    users_group, "synthetic_final", expected_path="/users/synthetic_final",
                    source_file=source_file, fit_users=allowed, audit=audit, group=True)
        self.assertNotIn("/users/synthetic_final", audit.metadata_paths)

    def test_selected_child_name_path_and_fileid_gates_precede_dereference(self):
        selected_user = loader._role_users(self.binding, 0, "outer_fit10")[0]

        class FakeSource:
            def __init__(self, file_id=5):
                self.id = SimpleNamespace(id=file_id)

        class FakeObject:
            def __init__(self, file_id=5, name="/unused"):
                self.file = FakeSource(file_id)
                self.name = name

        class SpyParent:
            def __init__(self, name, file_id=5, result=None):
                self.name = name
                self.file = FakeSource(file_id)
                self.get_calls = 0
                self.item_calls = 0
                self.result = result or FakeObject()
            def get(self, name, *, getlink=False):
                self.get_calls += 1
                return h5py.HardLink()
            def __getitem__(self, name):
                self.item_calls += 1
                return self.result

        cases = (
            ("quaternion", f"/users/{selected_user}/timestamps", 5),
            ("../other", f"/users/{selected_user}/../other", 5),
            ("child", "/users/other_user/child", 5),
            ("child", f"/users/{selected_user}/child", 6),
            ("child/extra", f"/users/{selected_user}/child/extra", 5),
        )
        for name, expected_path, parent_file_id in cases:
            parent = SpyParent(f"/users/{selected_user}", file_id=parent_file_id)
            audit = loader.ReadAudit(file_id=5, fit_role_users=10)
            with self.subTest(name=name, expected_path=expected_path), self.assertRaises(
                    PermissionError):
                loader._get_selected_child(
                    parent, name, expected_path=expected_path, source_file=FakeSource(5),
                    fit_users={selected_user}, audit=audit, group=True)
            self.assertEqual(parent.get_calls, 0)
            self.assertEqual(parent.item_calls, 0)

        canonical = f"/users/{selected_user}/child"
        for result, message in (
                (FakeObject(5, "/users/another/child"), "object name"),
                (FakeObject(6, canonical), "foreign FileID")):
            parent = SpyParent(f"/users/{selected_user}", result=result)
            audit = loader.ReadAudit(file_id=5, fit_role_users=10)
            with self.assertRaisesRegex(PermissionError, message):
                loader._get_selected_child(
                    parent, "child", expected_path=canonical, source_file=FakeSource(5),
                    fit_users={selected_user}, audit=audit, group=True)
            self.assertEqual(parent.get_calls, 1)
            self.assertEqual(parent.item_calls, 1)

    def test_lazy_batch_shapes_exact_context_targets_and_readonly_arrays(self):
        pool = self._load_synthetic_pool()
        schedule = pool.training_schedules[0]
        batches = list(pool.iter_batches(schedule))
        self.assertEqual([len(batch.contexts) for batch in batches], [128, 128, 1])
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(pool.training_schedules), 10)
        self.assertEqual(len(pool.standardizer_schedule.ranks), 65_536)
        self.assertEqual(len(np.unique(pool.standardizer_schedule.ranks)), 65_536)
        standardizer_first_batch = next(pool.iter_batches(pool.standardizer_schedule))
        self.assertEqual(standardizer_first_batch.contexts.shape, (128, 400, 4))
        self.assertEqual(standardizer_first_batch.schedule_kind,
                         "standardizer_without_replacement")
        self.assertEqual(pool._audit.pose_array_bytes, 10 * SAMPLES_PER_SEGMENT * 16)
        self.assertLessEqual(pool._audit.max_logical_slice_bytes, 16_384)
        self.assertEqual(pool._audit.denied_reads, 0)
        self.assertTrue(all("synthetic_final" not in path and "synthetic_development" not in path
                            for path in pool._audit.metadata_paths))
        self.assertTrue(all(path.startswith("/users/fit_user_")
                            for path in pool._audit.value_rows_by_dataset))
        self.assertTrue(all(not segment.poses.flags.writeable for segment in pool._segments))

        q = _synthetic_quaternions()
        self.assertEqual(batches[0].contexts.shape, (128, 400, 4))
        self.assertEqual(batches[0].targets20.shape, (128, 20, 4))
        self.assertIs(batches[0].model_inputs, batches[0].contexts)
        self.assertFalse(batches[0].contexts.flags.writeable)
        self.assertFalse(batches[0].targets20.flags.writeable)
        self.assertEqual(batches[0].endpoint20.shape, (128, 4))
        np.testing.assert_array_equal(batches[0].endpoint20, batches[0].targets20[:, -1, :])
        for batch in batches:
            for j, global_offset in enumerate(batch.global_origin_offsets):
                row = next(row for row in self.binding.origin_rows
                           if row.offset_int64 <= global_offset < row.offset_int64 + row.count)
                local_rank = int(global_offset) - row.offset_int64
                origin = int(self.binding.origin_values[int(global_offset)])
                start = origin - 399
                np.testing.assert_array_equal(batch.contexts[j], q[start:origin + 1])
                np.testing.assert_array_equal(batch.targets20[j], q[origin + 1:origin + 21])
                self.assertEqual(int(batch.actual_origin_indices[j]), origin)
        cloned = batches[0].clone_contexts_for_augmentation()
        self.assertTrue(cloned.flags.writeable)
        cloned[0, 0, 0] = 123.0
        self.assertNotEqual(float(batches[0].contexts[0, 0, 0]), 123.0)
        receipt = pool.safe_aggregate_receipt()
        receipt_text = json.dumps(receipt, sort_keys=True)
        self.assertNotIn("fit_user_00", receipt_text)
        self.assertNotIn("fit_user_01", receipt_text)
        self.assertNotIn("session_0", receipt_text)
        self.assertEqual(receipt["q1_grant"], False)
        self.assertEqual(receipt["source_root_metadata_binding_mode"], "direct_root_attributes")
        self.assertEqual(receipt["quaternion_rows_loaded"], 10 * SAMPLES_PER_SEGMENT)
        self.assertEqual(receipt["timestamp_rows_validated"], 10 * SAMPLES_PER_SEGMENT)
        self.assertEqual(len(receipt["sampler_schedules"]), 11)

    def test_sampler_schedule_and_origin_identity_hashes_are_stable_and_seeded(self):
        rows1, total1, digest1, schedules1, standardizer1 = loader._prepare_fold_schedules(
            self.binding, self.access, 0)
        rows2, total2, digest2, schedules2, standardizer2 = loader._prepare_fold_schedules(
            self.binding, self.access, 0)
        self.assertEqual(total1, 10 * ROWS_PER_SEGMENT)
        self.assertEqual(digest1, digest2)
        self.assertEqual([s.schedule_sha256 for s in schedules1],
                         [s.schedule_sha256 for s in schedules2])
        self.assertEqual([s.origin_identity_sha256 for s in schedules1],
                         [s.origin_identity_sha256 for s in schedules2])
        np.testing.assert_array_equal(standardizer1.ranks, standardizer2.ranks)
        for arm_schedule in schedules1[:5]:
            np.testing.assert_array_equal(arm_schedule.ranks, schedules1[0].ranks)
            self.assertEqual(arm_schedule.origin_identity_sha256,
                             schedules1[0].origin_identity_sha256)
        self.assertIs(schedules1[0].ranks, schedules1[4].ranks)
        self.assertNotEqual(schedules1[0].schedule_sha256, schedules1[5].schedule_sha256)
        expected_training = np.random.default_rng(np.random.SeedSequence([20261026, 0])).integers(
            0, total1, size=257, endpoint=False, dtype=np.int64)
        np.testing.assert_array_equal(schedules1[0].ranks, expected_training)
        expected_standardizer = np.random.default_rng(
            np.random.SeedSequence([20261028, 0])).choice(
                total1, size=65_536, replace=False).astype(np.int64, copy=False)
        np.testing.assert_array_equal(standardizer1.ranks, expected_standardizer)
        self.assertEqual(len(rows1), len(rows2))
        self.assertTrue(all(not schedule.ranks.flags.writeable for schedule in schedules1))
        pool = self._load_synthetic_pool()
        original = pool.training_schedules[0]
        changed_ranks = np.array(original.ranks, copy=True)
        changed_ranks[0] = -1
        changed_ranks.setflags(write=False)
        altered = replace(original, ranks=changed_ranks)
        with self.assertRaisesRegex(PermissionError, "not generated from this fixed approved"):
            list(pool.iter_batches(altered))

        _, _, _, _, _, diagnostic_access = _synthetic_access(
            self.binding, role_mode="inner_fit8_diagnostic")
        diagnostic_pool = self._load_synthetic_pool(access=diagnostic_access)
        self.assertEqual(diagnostic_pool._audit.fit_role_users, 8)
        allowed_users = set(loader._role_users(self.binding, 0, "inner_fit8_diagnostic"))
        seen_users = {path.split("/")[2] for path in diagnostic_pool._audit.metadata_paths
                      if path.startswith("/users/")}
        self.assertTrue(seen_users.issubset(allowed_users))

    def test_cpu_preparation_has_only_standardizer_schedule_and_context_batches(self):
        cpu_access = replace(
            self.access, purpose="fit_only_cpu_preparation", role_mode="outer_fit10",
            seed_values=(), arm_ids=(), standardizer_seed=20261028,
            windows_per_schedule=0,
            fitting_operations_allowed=("population_feature_standardizer",),
            device="CPU", cuda_contexts=0, optimizer_updates=0,
            cpu_consumer_source_sha256="0" * 64)
        _, total, _, schedules, standardizer = loader._prepare_fold_schedules(
            self.binding, cpu_access, 0)
        self.assertEqual(total, 10 * ROWS_PER_SEGMENT)
        self.assertEqual(schedules, ())
        expected = np.random.default_rng(np.random.SeedSequence([20261028, 0])).choice(
            total, size=65_536, replace=False).astype(np.int64, copy=False)
        np.testing.assert_array_equal(standardizer.ranks, expected)

        pool = self._load_synthetic_pool(access=cpu_access)
        self.assertEqual(pool.training_schedules, ())
        batch = next(pool.iter_batches(pool.standardizer_schedule))
        self.assertEqual(batch.contexts.shape, (128, 400, 4))
        self.assertIsNone(batch.targets20)
        self.assertIs(batch.model_inputs, batch.contexts)
        with self.assertRaisesRegex(PermissionError, "single-pass"):
            next(pool.iter_batches(pool.standardizer_schedule))
        with self.assertRaisesRegex(PermissionError, "do not expose forecast targets"):
            _ = batch.endpoint20
        forged_ranks = np.asarray([0], dtype=np.int64)
        forged_ranks.setflags(write=False)
        forged = loader.PreparedSchedule(
            0, "training_with_replacement", loader.TRAINING_SEEDS[0],
            loader.CANONICAL_ARMS[0], forged_ranks,
            loader._schedule_digest(forged_ranks, 0, "training_with_replacement",
                                    loader.TRAINING_SEEDS[0], loader.CANONICAL_ARMS[0]),
            "0" * 64)
        with self.assertRaisesRegex(PermissionError, "not generated from this fixed approved"):
            list(pool.iter_batches(forged))
        receipt = pool.safe_aggregate_receipt()
        self.assertEqual(receipt["purpose"], "fit_only_cpu_preparation")
        self.assertEqual(receipt["fitting_operations_allowed"], ["population_feature_standardizer"])
        self.assertEqual(receipt["optimizer_updates"], 0)
        self.assertEqual(receipt["model_fits_performed_by_loader"], 0)
        self.assertEqual(len(receipt["sampler_schedules"]), 1)

    def test_wrong_cpu_purpose_and_device_fail_before_source_access(self):
        wrong_role = _cpu_plan_doc_from_timing(self.plan_doc)
        wrong_role["role_mode"] = "inner_fit8_diagnostic"
        bad_raw = json.dumps(wrong_role, sort_keys=True, separators=(",", ":")).encode()

        def validate_wrong_role():
            return loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_role, bad_raw,
                self.approval_doc, self.approval_raw)

        with mock.patch.object(loader, "_load_pinned_reader", return_value=self.reader), \
                mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                  side_effect=validate_wrong_role), \
                mock.patch.object(loader, "_verified_fit_source_file",
                                  side_effect=AssertionError("source checksum/open must not run")), \
                mock.patch.object(loader.h5py, "File",
                                  side_effect=AssertionError("HDF5 must not open")):
            with self.assertRaisesRegex(PermissionError, "purpose does not authorize"):
                loader.load_approved_fit_pool(0)

        wrong_device = _cpu_plan_doc_from_timing(self.plan_doc)
        wrong_device["device_policy"]["device"] = "GPU"
        bad_raw = json.dumps(wrong_device, sort_keys=True, separators=(",", ":")).encode()

        def validate_wrong_device():
            return loader._validate_plan_and_approval(
                self.reader, self.binding, wrong_device, bad_raw,
                self.approval_doc, self.approval_raw)

        with mock.patch.object(loader, "_load_pinned_reader", return_value=self.reader), \
                mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                  side_effect=validate_wrong_device), \
                mock.patch.object(loader, "_verified_fit_source_file",
                                  side_effect=AssertionError("source checksum/open must not run")), \
                mock.patch.object(loader.h5py, "File",
                                  side_effect=AssertionError("HDF5 must not open")):
            with self.assertRaisesRegex(PermissionError, "CPU-only execution"):
                loader.load_approved_fit_pool(0)

    def test_cpu_plan_consumer_hash_and_metadata_preflight_precede_source_access(self):
        consumer_path = self.tmp / "synthetic_cpu_consumer.py"
        consumer_path.write_text("# synthetic consumer fixture\n", encoding="utf-8")
        (reader, plan_doc, plan_raw, approval_doc, approval_raw, access) = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation",
            cpu_consumer_path=consumer_path)
        self.assertEqual(access.purpose, "fit_only_cpu_preparation")
        self.assertEqual(access.role_mode, "outer_fit10")
        self.assertEqual(access.fitting_operations_allowed,
                         ("population_feature_standardizer",))
        self.assertEqual(access.device, "CPU")
        self.assertEqual(access.cuda_contexts, 0)
        self.assertEqual(access.optimizer_updates, 0)
        self.assertEqual(access.cpu_consumer_source_sha256,
                         loader._sha256_file(consumer_path))

        with mock.patch.object(loader, "CPU_CONSUMER_PATH", consumer_path), \
                mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                  return_value=(self.binding, access)), \
                mock.patch.object(loader, "_verified_fit_source_file",
                                  side_effect=AssertionError("preflight must not open source")), \
                mock.patch.object(loader.h5py, "File",
                                  side_effect=AssertionError("preflight must not open HDF5")):
            preflight = loader.preflight_approved_fit_pool(0)
        self.assertEqual(preflight.purpose, "fit_only_cpu_preparation")
        self.assertEqual(preflight.device, "CPU")
        self.assertEqual(preflight.cuda_contexts, 0)
        self.assertEqual(preflight.optimizer_updates, 0)
        self.assertEqual(preflight.standardizer_origin_count, 65_536)
        self.assertEqual(preflight.cpu_consumer_source_sha256,
                         loader._sha256_file(consumer_path))

        consumer_path.write_text("# changed bytes\n", encoding="utf-8")
        wrong_hash_raw = json.dumps(
            plan_doc, sort_keys=True, separators=(",", ":")).encode()

        def validate_changed_consumer():
            return loader._validate_plan_and_approval(
                reader, self.binding, plan_doc, wrong_hash_raw,
                approval_doc, approval_raw)

        with mock.patch.object(loader, "CPU_CONSUMER_PATH", consumer_path), \
                mock.patch.object(loader, "_load_pinned_reader", return_value=reader), \
                mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                  side_effect=validate_changed_consumer), \
                mock.patch.object(loader, "_verified_fit_source_file",
                                  side_effect=AssertionError("source checksum/open must not run")), \
                mock.patch.object(loader.h5py, "File",
                                  side_effect=AssertionError("HDF5 must not open")):
            with self.assertRaisesRegex(PermissionError, "consumer bytes differ"):
                loader.load_approved_fit_pool(0)

    def test_trusted_launch_pins_scope_expiry_and_actual_code_hashes(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        now = datetime.now(timezone.utc)
        valid_raw = _synthetic_trusted_launch(cpu_access, now=now)
        with mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: valid_raw}):
            trusted = loader._validate_trusted_cpu_prep_launch(cpu_access, now=now)
        self.assertTrue(trusted.launch_id.startswith("q1-cpu-launch-"))
        self.assertEqual(trusted.expires_at, now + timedelta(seconds=300))

        invalid_envelopes = (
            _synthetic_trusted_launch(cpu_access, now=now, plan_sha256="0" * 64),
            _synthetic_trusted_launch(cpu_access, now=now, approval_sha256="0" * 64),
            _synthetic_trusted_launch(cpu_access, now=now, recipe_sha256="0" * 64),
            _synthetic_trusted_launch(cpu_access, now=now,
                                      independent_review_sha256="0" * 64),
            _synthetic_trusted_launch(cpu_access, now=now,
                                      loader_source_sha256="0" * 64),
            _synthetic_trusted_launch(cpu_access, now=now,
                                      consumer_source_sha256="0" * 64),
            _synthetic_trusted_launch(cpu_access, now=now,
                                      expiry_delta=timedelta(seconds=-1)),
            _synthetic_trusted_launch(cpu_access, now=now,
                                      expiry_delta=timedelta(seconds=901)),
        )
        for raw in invalid_envelopes:
            with self.subTest(raw=raw), mock.patch.dict(
                    os.environ, {loader.TRUSTED_LAUNCH_ENV: raw}), self.assertRaises(
                        PermissionError):
                loader._validate_trusted_cpu_prep_launch(cpu_access, now=now)

    def test_missing_or_forged_launch_capability_fails_before_source_and_lock(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        malformed = "not-json"
        forged = _synthetic_trusted_launch(
            cpu_access, plan_sha256="0" * 64)
        cases = (None, malformed, forged)
        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp:
            for index, raw in enumerate(cases):
                lock_root = Path(tmp) / f"lock_{index}"
                source_calls = mock.Mock(side_effect=AssertionError("source boundary must stay closed"))
                with self.subTest(index=index), \
                        mock.patch.object(loader, "_load_pinned_reader", return_value=self.reader), \
                        mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                          return_value=(self.binding, cpu_access)), \
                        mock.patch.object(loader, "_verified_fit_source_file", source_calls), \
                        mock.patch.object(loader, "LAUNCH_LOCK_ROOT", lock_root), \
                        mock.patch.dict(loader._launch_use_state, {}, clear=True), \
                        mock.patch.dict(loader._seen_launch_ids, {}, clear=True):
                    if raw is None:
                        with mock.patch.dict(os.environ, {}, clear=True):
                            with self.assertRaises((PermissionError, ValueError)):
                                loader.load_approved_fit_pool(0)
                    else:
                        with mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: raw}):
                            with self.assertRaises((PermissionError, ValueError)):
                                loader.load_approved_fit_pool(0)
                source_calls.assert_not_called()
                self.assertFalse(lock_root.exists())

    def test_plan_lock_is_exclusive_and_one_fold_attempt_is_allowed(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        raw = _synthetic_trusted_launch(cpu_access)
        source_calls = []

        def stop_at_source(*args, **kwargs):
            source_calls.append(1)
            raise RuntimeError("synthetic source boundary stop")

        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp:
            lock_root = Path(tmp) / "attempt_locks"
            with mock.patch.object(loader, "_load_pinned_reader", return_value=self.reader), \
                    mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                      return_value=(self.binding, cpu_access)), \
                    mock.patch.object(loader, "_verified_fit_source_file", stop_at_source), \
                    mock.patch.object(loader, "LAUNCH_LOCK_ROOT", lock_root), \
                    mock.patch.dict(loader._launch_use_state, {}, clear=True), \
                    mock.patch.dict(loader._seen_launch_ids, {}, clear=True), \
                    mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: raw}):
                for fold_id in (0, 1, 2):
                    with self.assertRaisesRegex(RuntimeError, "synthetic source boundary"):
                        loader.load_approved_fit_pool(fold_id)
                lock_path = lock_root / f"plan_{cpu_access.plan_sha256}.attempt.json"
                self.assertTrue(lock_path.is_file())
                state = loader._launch_use_state[cpu_access.plan_sha256]
                self.assertEqual(state["folds"], {0, 1, 2})
                parent_lock = json.loads(lock_path.read_text(encoding="utf-8"))
                self.assertEqual(state["expires_utc"], parent_lock["expires_utc"])
                for fold_id in (0, 1, 2):
                    claim_path = lock_root / (
                        f"plan_{cpu_access.plan_sha256}.fold_{fold_id}.claim.json")
                    self.assertTrue(claim_path.is_file())
                    claim = json.loads(claim_path.read_text(encoding="utf-8"))
                    self.assertEqual(claim["kind"], "q1_fit_pool_production_fold_claim")
                    self.assertEqual(claim["fold_id"], fold_id)
                    self.assertEqual(claim["plan_sha256"], cpu_access.plan_sha256)
                    self.assertEqual(claim["approval_sha256"], cpu_access.approval_sha256)
                    self.assertEqual(claim["recipe_sha256"], cpu_access.recipe_sha256)
                    self.assertEqual(claim["launch_id"], state["launch_id"])
                    self.assertEqual(claim["expires_utc"], parent_lock["expires_utc"])
                    self.assertEqual(claim["parent_plan_lock_identity"],
                                     state["parent_plan_lock_identity"])
                self.assertEqual(state["lock_root"], str(Path(os.path.abspath(lock_root))))
                with self.assertRaisesRegex(PermissionError, "already attempted"):
                    loader.load_approved_fit_pool(0)
                self.assertEqual(len(source_calls), 3)

                # Simulate a fresh process: the exclusive plan lock still denies relaunch.
                loader._launch_use_state.clear()
                loader._seen_launch_ids.clear()
                with self.assertRaisesRegex(PermissionError, "exclusive production attempt lock"):
                    loader.load_approved_fit_pool(0)
                self.assertEqual(len(source_calls), 3)

    def test_changed_expiry_for_same_launch_cannot_claim_later_fold(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        now = datetime.now(timezone.utc)
        launch_id = "q1-cpu-launch-expiry-continuity"
        first_envelope = _synthetic_trusted_launch(
            cpu_access, now=now, launch_id=launch_id,
            expiry_delta=timedelta(seconds=300))
        extended_envelope = _synthetic_trusted_launch(
            cpu_access, now=now, launch_id=launch_id,
            expiry_delta=timedelta(seconds=600))
        source_calls = []

        def stop_at_source(*args, **kwargs):
            source_calls.append(1)
            raise RuntimeError("synthetic source boundary stop")

        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp, \
                mock.patch.object(loader, "_load_pinned_reader", return_value=self.reader), \
                mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                  return_value=(self.binding, cpu_access)), \
                mock.patch.object(loader, "_verified_fit_source_file", stop_at_source), \
                mock.patch.object(loader, "LAUNCH_LOCK_ROOT", Path(tmp) / "attempt_locks"), \
                mock.patch.dict(loader._launch_use_state, {}, clear=True), \
                mock.patch.dict(loader._seen_launch_ids, {}, clear=True):
            with mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: first_envelope}):
                with self.assertRaisesRegex(RuntimeError, "synthetic source boundary"):
                    loader.load_approved_fit_pool(0)
            fold0_claim = Path(tmp) / "attempt_locks" / (
                f"plan_{cpu_access.plan_sha256}.fold_0.claim.json")
            fold1_claim = Path(tmp) / "attempt_locks" / (
                f"plan_{cpu_access.plan_sha256}.fold_1.claim.json")
            self.assertTrue(fold0_claim.is_file())
            self.assertFalse(fold1_claim.exists())

            with mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: extended_envelope}):
                with self.assertRaisesRegex(PermissionError, "expiry changed"):
                    loader.load_approved_fit_pool(1)
            self.assertFalse(fold1_claim.exists())
            self.assertEqual(len(source_calls), 1)

    def test_durable_claim_replay_and_claim_tamper_fail_closed(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        raw = _synthetic_trusted_launch(cpu_access)
        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp, \
                mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: raw}):
            launch = loader._validate_trusted_cpu_prep_launch(cpu_access)
            lock_root = Path(tmp) / "attempt_locks"
            state = {}
            used_ids = {}
            loader._claim_production_fold(
                cpu_access, launch, 0, lock_root=lock_root,
                use_state=state, seen_launch_ids=used_ids)
            claim_path = lock_root / f"plan_{cpu_access.plan_sha256}.fold_0.claim.json"
            claim_before = claim_path.read_bytes()

            # Clearing volatile fold state cannot reuse an existing durable claim.
            state[cpu_access.plan_sha256]["folds"].clear()
            with self.assertRaisesRegex(PermissionError, "fold attempt claim already exists"):
                loader._claim_production_fold(
                    cpu_access, launch, 0, lock_root=lock_root,
                    use_state=state, seen_launch_ids=used_ids)
            self.assertEqual(claim_path.read_bytes(), claim_before)

            with claim_path.open("ab") as stream:
                stream.write(b"tamper")
                stream.flush()
                os.fsync(stream.fileno())
            with self.assertRaisesRegex(PermissionError, "attempt lock (identity|bytes)"):
                loader._claim_production_fold(
                    cpu_access, launch, 1, lock_root=lock_root,
                    use_state=state, seen_launch_ids=used_ids)
            self.assertFalse((lock_root /
                              f"plan_{cpu_access.plan_sha256}.fold_1.claim.json").exists())

    def test_durable_claim_parent_lock_tamper_fails_before_next_claim(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        raw = _synthetic_trusted_launch(cpu_access)
        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp, \
                mock.patch.dict(os.environ, {loader.TRUSTED_LAUNCH_ENV: raw}):
            launch = loader._validate_trusted_cpu_prep_launch(cpu_access)
            lock_root = Path(tmp) / "attempt_locks"
            state = {}
            used_ids = {}
            loader._claim_production_fold(
                cpu_access, launch, 0, lock_root=lock_root,
                use_state=state, seen_launch_ids=used_ids)
            plan_lock = lock_root / f"plan_{cpu_access.plan_sha256}.attempt.json"
            with plan_lock.open("ab") as stream:
                stream.write(b"tamper")
                stream.flush()
                os.fsync(stream.fileno())
            with self.assertRaisesRegex(PermissionError, "lock (identity|bytes)"):
                loader._claim_production_fold(
                    cpu_access, launch, 1, lock_root=lock_root,
                    use_state=state, seen_launch_ids=used_ids)
            self.assertFalse((lock_root /
                              f"plan_{cpu_access.plan_sha256}.fold_1.claim.json").exists())

    def test_lock_root_cannot_change_mid_plan(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        launch = loader.TrustedCPUPrepLaunch(
            "q1-cpu-launch-synthetic-root-change",
            datetime.now(timezone.utc) + timedelta(seconds=300))
        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp:
            root_a = Path(tmp) / "locks_a"
            root_b = Path(tmp) / "locks_b"
            state = {}
            used_ids = {}
            loader._claim_production_fold(
                cpu_access, launch, 0, lock_root=root_a,
                use_state=state, seen_launch_ids=used_ids)
            with self.assertRaisesRegex(PermissionError, "root changed"):
                loader._claim_production_fold(
                    cpu_access, launch, 1, lock_root=root_b,
                    use_state=state, seen_launch_ids=used_ids)
            self.assertFalse(root_b.exists())

            # Reusing the same spelling for a replaced directory is also denied.
            retired_root = Path(tmp) / "locks_a_retired"
            os.rename(root_a, retired_root)
            root_a.mkdir()
            with self.assertRaisesRegex(PermissionError, "root filesystem identity changed"):
                loader._claim_production_fold(
                    cpu_access, launch, 1, lock_root=root_a,
                    use_state=state, seen_launch_ids=used_ids)
            self.assertFalse((root_a /
                              f"plan_{cpu_access.plan_sha256}.fold_1.claim.json").exists())

    def test_lock_root_junction_reparse_metadata_fails_before_any_claim(self):
        cpu_access = _synthetic_access(
            self.binding, purpose="fit_only_cpu_preparation")[5]
        launch = loader.TrustedCPUPrepLaunch(
            "q1-cpu-launch-synthetic-reparse",
            datetime.now(timezone.utc) + timedelta(seconds=300))
        with tempfile.TemporaryDirectory(dir=STAGING_ROOT) as tmp:
            lock_root = Path(tmp) / "locks"
            lock_root.mkdir()
            absolute_root = Path(os.path.abspath(lock_root))
            real_lstat = Path.lstat

            def report_reparse(path):
                metadata = real_lstat(path)
                if Path(os.path.abspath(path)) == absolute_root:
                    return SimpleNamespace(st_mode=metadata.st_mode,
                                           st_file_attributes=0x400)
                return metadata

            with mock.patch.object(Path, "lstat", new=report_reparse):
                with self.assertRaisesRegex(PermissionError, "reparse point"):
                    loader._claim_production_fold(
                        cpu_access, launch, 0, lock_root=lock_root,
                        use_state={}, seen_launch_ids={})
            self.assertEqual(list(lock_root.iterdir()), [])

    def test_timing_prefix_source_load_stays_blocked_without_separate_scope(self):
        source_calls = mock.Mock(side_effect=AssertionError("source boundary must stay closed"))
        with mock.patch.object(loader, "_load_pinned_reader", return_value=self.reader), \
                mock.patch.object(loader, "_load_fixed_plan_and_approval",
                                  return_value=(self.binding, self.access)), \
                mock.patch.object(loader, "_verified_fit_source_file", source_calls), \
                mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(PermissionError, "locked to CPU fit-only preparation"):
                loader.load_approved_fit_pool(0)
        source_calls.assert_not_called()

    def test_preallocation_memory_cap_fails_before_any_dataset_value_slice(self):
        read_calls = []
        original = loader._read_dataset_rows

        def spy_read(*args, **kwargs):
            read_calls.append(1)
            return original(*args, **kwargs)

        too_small = replace(self.access, max_pose_bytes=1024)
        prepared = loader._prepare_fold_schedules(self.binding, too_small, 0)
        with h5py.File(self.source_path, "r") as source_file, mock.patch.object(
                loader, "_read_dataset_rows", spy_read):
            with self.assertRaisesRegex(MemoryError, "preallocation cap"):
                loader._load_fit_pool_from_open_file(
                    source_file, self.binding, too_small, 0, prepared,
                    source_hash_verified=False,
                    deadline=time_monotonic_plus(60))
        self.assertEqual(read_calls, [])

    def test_hardlink_softlink_and_timestamp_gap_reject_before_quaternion_reads(self):
        original = loader._read_dataset_rows
        read_calls = []

        def spy_read(*args, **kwargs):
            read_calls.append((args[0].name, kwargs["start"], kwargs["stop"]))
            return original(*args, **kwargs)

        for options, message in (
                ({"alias_fit_to_final": True}, "hard-link alias"),
                ({"soft_fit_link": True}, "soft or external link"),
                ({"virtual_fit": True}, "dtype/dimensions/storage"),
                ({"external_fit": True}, "dtype/dimensions/storage"),
                ({"timestamp_gap": True}, "gap-free 100 Hz"),
                ({"bad_quaternion_norm": True}, "nonunit quaternion"),
                ({"bad_quaternion_sign": True}, "chronologically aligned")):
            path = self.tmp / f"bad_{len(read_calls)}.h5"
            _write_synthetic_h5(path, self.binding, **options)
            prepared = loader._prepare_fold_schedules(self.binding, self.access, 0)
            read_calls.clear()
            with h5py.File(path, "r") as source_file, mock.patch.object(
                    loader, "_read_dataset_rows", spy_read):
                with self.assertRaisesRegex((ValueError, PermissionError), message):
                    loader._load_fit_pool_from_open_file(
                        source_file, self.binding, self.access, 0, prepared,
                        source_hash_verified=False,
                        deadline=time_monotonic_plus(60))
            if "hard-link" in message or "soft or external" in message or "storage" in message:
                self.assertEqual(read_calls, [])
            else:
                timestamp_rows = sum(stop - start for path, start, stop in read_calls
                                     if "/timestamps" in path)
                quaternion_rows = sum(stop - start for path, start, stop in read_calls
                                      if "/quaternion" in path)
                if "gap-free" in message:
                    self.assertEqual(timestamp_rows, 3 * 1024)
                    self.assertEqual(quaternion_rows, 2 * 1024)
                else:
                    self.assertEqual(timestamp_rows, 2 * 1024)
                    self.assertEqual(quaternion_rows, 2 * 1024)

    def test_source_hash_mismatch_precedes_hdf5_open(self):
        reader = self.reader
        wrong = self.tmp / "synthetic_wrong_source.bin"
        wrong.write_bytes(b"synthetic bytes are not the production source")
        binding = replace(
            self.binding, source_path=wrong,
            source_identity=reader._file_identity(wrong))
        opened = []
        actual_file = h5py.File

        def spy_file(*args, **kwargs):
            opened.append(args[0] if args else kwargs.get("name"))
            return actual_file(*args, **kwargs)

        with mock.patch.object(loader.h5py, "File", spy_file):
            with self.assertRaisesRegex(ValueError, "source HDF5 bytes differ"):
                with loader._verified_fit_source_file(
                        reader, binding, time_monotonic_plus(60)):
                    self.fail("HDF5 must not open after a checksum mismatch")
        self.assertEqual(opened, [])

    def test_exact_dataset_fileid_path_and_chunk_bytes_are_checked_before_read(self):
        class FakeFile:
            def __init__(self, id_value):
                self.id = SimpleNamespace(id=id_value)

        class FakeDataset:
            def __init__(self, file_id, name, rows=1024):
                self.file = FakeFile(file_id)
                self.name = name
                self.shape = (rows, 4)
                self.dtype = np.dtype("float32")
                self.calls = []

            def __getitem__(self, key):
                self.calls.append(key)
                start, stop = key.start, key.stop
                return np.ones((stop - start, 4), dtype=np.float32)

        audit = loader.ReadAudit(file_id=5, fit_role_users=10)
        guard = loader._DatasetAudit(5, frozenset({"/users/fit_u/q"}), 16_384,
                                     time_monotonic_plus(20), audit)
        good = FakeDataset(5, "/users/fit_u/q")
        result = loader._read_dataset_rows(good, start=0, stop=1024, audit=guard)
        self.assertEqual(result.shape, (1024, 4))
        self.assertEqual(audit.max_logical_slice_bytes, 16_384)
        for dataset, start, stop in (
                (FakeDataset(6, "/users/fit_u/q"), 0, 2),
                (FakeDataset(5, "/users/other/q"), 0, 2),
                (FakeDataset(5, "/users/fit_u/q"), 0, 1025)):
            with self.assertRaises(PermissionError):
                loader._read_dataset_rows(dataset, start=start, stop=stop, audit=guard)
            self.assertEqual(dataset.calls, [])
        self.assertEqual(audit.denied_reads, 3)
        expired = FakeDataset(5, "/users/fit_u/q")
        expired_guard = loader._DatasetAudit(
            5, frozenset({"/users/fit_u/q"}), 16_384, 0.0, audit)
        with self.assertRaises(TimeoutError):
            loader._read_dataset_rows(expired, start=0, stop=2, audit=expired_guard)
        self.assertEqual(expired.calls, [])

    def test_missing_fixed_plan_fails_before_source_checksum_or_hdf5(self):
        path_spy = self._install_no_signal_open_spies()
        missing = self.tmp / "missing"
        reader = loader._load_pinned_reader()
        with mock.patch.object(loader, "PLAN_PATH", missing / "fit_pool_plan.json"), \
                mock.patch.object(loader, "APPROVAL_PATH", missing / "fit_pool_approval.json"), \
                mock.patch.object(loader, "_verified_fit_source_file",
                                  side_effect=AssertionError("source must stay closed")), \
                path_spy, mock.patch.object(
                    h5py, "File", side_effect=AssertionError("HDF5 must stay closed")):
            with self.assertRaisesRegex(PermissionError, "fit-pool plan metadata is absent"):
                loader.load_approved_fit_pool(0)

    def _install_no_signal_open_spies(self):
        import builtins
        import io
        original_path_open = Path.open
        original_builtin_open = builtins.open
        original_io_open = io.open
        original_os_open = os.open
        reader = loader._load_pinned_reader()
        cache = reader.origin_inventory.production_inputs().cache.resolve()
        raw_root = (REPO_ROOT / "data" / "raw").resolve()
        blocked = {str(reader.SOURCE_HDF5_PATH.resolve()).casefold(), str(cache).casefold()}

        def check(value):
            try:
                path = Path(os.fspath(value)).resolve(strict=False)
            except (TypeError, ValueError, OSError):
                return
            key = str(path).casefold()
            if key in blocked or path.is_relative_to(raw_root):
                raise AssertionError("actual HDF5/cache/recording open was attempted")

        def path_open(path, *args, **kwargs):
            check(path)
            return original_path_open(path, *args, **kwargs)

        def builtin_open(file, *args, **kwargs):
            check(file)
            return original_builtin_open(file, *args, **kwargs)

        def io_open(file, *args, **kwargs):
            check(file)
            return original_io_open(file, *args, **kwargs)

        def os_open(path, *args, **kwargs):
            check(path)
            return original_os_open(path, *args, **kwargs)

        stack = (
            mock.patch.object(Path, "open", path_open),
            mock.patch.object(builtins, "open", builtin_open),
            mock.patch.object(io, "open", io_open),
            mock.patch.object(os, "open", os_open),
        )

        class Combined:
            def __enter__(self):
                for context in stack:
                    context.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                for context in reversed(stack):
                    context.__exit__(exc_type, exc, tb)
                return False

        return Combined()


def time_monotonic_plus(seconds):
    return __import__("time").monotonic() + seconds


class PinnedImportTests(unittest.TestCase):
    def test_helper_code_hash_is_checked_before_first_import(self):
        old_module = loader._reader_module
        old_hash = loader.PINNED_READER_SHA256
        loader._reader_module = None
        loader.PINNED_READER_SHA256 = "0" * 64
        import_spy = mock.Mock(side_effect=AssertionError("helper import occurred before hash check"))
        try:
            with mock.patch.object(importlib.util, "spec_from_file_location", import_spy):
                with self.assertRaisesRegex(PermissionError, "independently accepted review pin"):
                    loader._load_pinned_reader()
            import_spy.assert_not_called()
        finally:
            loader.PINNED_READER_SHA256 = old_hash
            loader._reader_module = old_module


if __name__ == "__main__":
    unittest.main()


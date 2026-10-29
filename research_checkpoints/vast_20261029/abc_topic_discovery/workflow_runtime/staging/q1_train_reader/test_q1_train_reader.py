"""Synthetic and metadata-only checks for the staged Q1 train reader."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = ROOT / "ABC_IUEH"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

import q1_train_reader as reader
from abc_iueh.config import PipelineConfig


def _pipeline_dict():
    return PipelineConfig().to_dict()


def _synthetic_binding(tmpdir: Path) -> reader.MetadataBinding:
    users = tuple(f"train_{i:02d}" for i in range(15))
    validation = (users[:5], users[5:10], users[10:])
    user_fold = {user: fold for fold, group in enumerate(validation) for user in group}
    user_sessions = {user: {"session_0": ("0000",)} for user in users}
    windows_per_segment = 581  # n_samples - (400 + 20 - 1)
    session_windows = {(user, "session_0"): windows_per_segment for user in users}
    expected = reader.origin_inventory.ExpectedTrain(
        users=users, user_sessions=user_sessions, user_fold=user_fold,
        fold_validation_order=validation, expected_windows=15 * windows_per_segment,
        session_windows=session_windows,
    )
    rows = []
    origins = []
    offset = 0
    for fold, group in enumerate(validation):
        for user in sorted(group):
            rows.append(reader.OriginRow(
                user, "session_0", "0000", fold, offset, windows_per_segment))
            origins.extend(range(399, 399 + windows_per_segment))
            offset += windows_per_segment
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
    )
    ownership = reader._inner_ownership(expected)
    source_path = tmpdir / "synthetic_source.h5"
    return reader.MetadataBinding(
        parent=parent,
        source_path=source_path,
        source_identity=(0, 0, 0, 0),
        source_path_evidence_sha256=reader.SOURCE_PATH_EVIDENCE_SHA256,
        origin_receipt_sha256=reader.ORIGIN_RECEIPT_SHA256,
        origin_map_sha256=reader.ORIGIN_MAP_SHA256,
        origin_binary_sha256=reader.ORIGIN_BINARY_SHA256,
        origin_rows=tuple(rows),
        origin_values=np.asarray(origins, dtype="<i8"),
        ownership=ownership,
        ownership_plan_sha256=reader._ownership_digest(ownership),
    )


def _synthetic_quaternions(n=1000):
    angle = np.arange(n, dtype=np.float64) * 0.001
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = np.cos(angle / 2).astype(np.float32)
    q[:, 3] = np.sin(angle / 2).astype(np.float32)
    return q


def _write_synthetic_h5(path: Path, binding: reader.MetadataBinding, *,
                         alias_train_to_final=False, soft_train_quaternion=False,
                         virtual_train_quaternion=False):
    q_values = _synthetic_quaternions()
    cfg_json = json.dumps(_pipeline_dict(), sort_keys=True)
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

        users = store.create_group("users")
        for user in binding.parent.expected_train.users:
            user_group = users.create_group(user)
            sessions = user_group.create_group("sessions")
            session = sessions.create_group("session_0")
            segments = session.create_group("segments")
            segment = segments.create_group("0000")
            segment.create_dataset(
                "timestamps", data=np.arange(1000, dtype=np.float64) / 100.0,
                chunks=(1000,), compression="gzip", compression_opts=1,
            )
            if virtual_train_quaternion:
                layout = h5py.VirtualLayout(shape=(1000, 4), dtype=np.float32)
                source = h5py.VirtualSource(str(path) + ".backing", "/q", shape=(1000, 4))
                layout[:, :] = source[:, :]
                segment.create_virtual_dataset("quaternion", layout)
            elif soft_train_quaternion:
                segment["quaternion"] = h5py.SoftLink("/users/synthetic_development/q")
            else:
                segment.create_dataset(
                    "quaternion", data=q_values, chunks=(1000, 4),
                    compression="gzip", compression_opts=1,
                )

        # These non-train groups are present to prove the reader does not
        # dereference either partition. They use a deliberately different schema.
        for user in ("synthetic_development", "synthetic_final"):
            group = users.create_group(user)
            group.create_dataset("q", data=q_values, chunks=(1000, 4), compression="gzip")
        if alias_train_to_final:
            selected_alias_user = binding.ownership[0]["outer_fit"][0]
            train_ds = store[
                f"users/{selected_alias_user}/sessions/session_0/segments/0000/quaternion"
            ]
            users["synthetic_final"].create_group("aliases")["train_quaternion"] = train_ds


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="q1-reader-synthetic-", dir=reader.STAGING_ROOT)
        self.tmp = Path(self._tmp.name)
        self.binding = _synthetic_binding(self.tmp)
        self.source_path = self.tmp / "synthetic_source.h5"
        _write_synthetic_h5(self.source_path, self.binding)
        self.source_file = h5py.File(self.source_path, "r")

    def tearDown(self):
        if getattr(self, "source_file", None) is not None:
            self.source_file.close()
        self._tmp.cleanup()

    def _plan(self):
        approval = SimpleNamespace(prefixes=(
            reader.AuthorizedPrefix(0, "inner_fit", 2),
            reader.AuthorizedPrefix(0, "inner_selector", reader.SELECTOR_ORIGINS_PER_USER),
        ))
        return reader.build_read_plan(self.binding, approval)

    def test_exact_context_target_indices_norm_sign_and_readonly(self):
        plan = self._plan()
        captured = []
        original_init = reader.HDF5ReadGuard.__init__

        def capture_init(guard, *args, **kwargs):
            original_init(guard, *args, **kwargs)
            captured.append(guard)

        with mock.patch.object(reader.HDF5ReadGuard, "__init__", capture_init):
            windows = list(reader._iter_windows_from_open_file(self.source_file, self.binding, plan))
        self.assertEqual(len(windows), 8 * 2 + 2 * 581)
        self.assertEqual(len(captured), 1)
        guard = captured[0]
        self.assertEqual(guard.audit["quaternion_slice_calls"], len(plan))
        self.assertEqual(guard.audit["quaternion_rows_read"], len(plan) * 420)
        self.assertEqual(guard.audit["denied_value_reads"], 0)
        self.assertEqual(guard.audit["denied_group_dereferences"], 0)
        self.assertEqual(guard.audit["denied_foreign_file_accesses"], 0)
        self.assertFalse(any("synthetic_final" in path or "synthetic_development" in path
                             for path in guard.metadata_paths_touched))

        q = _synthetic_quaternions()
        for window in windows:
            self.assertGreaterEqual(window.origin_index, 399)
            self.assertEqual(window.context.shape, (400, 4))
            self.assertEqual(window.target.shape, (20, 4))
            self.assertEqual(window.context.dtype, np.float32)
            self.assertEqual(window.target.dtype, np.float32)
            start = window.origin_index - 399
            stop = start + reader.WINDOW_SAMPLES
            np.testing.assert_array_equal(window.context, q[start:start + 400])
            np.testing.assert_array_equal(window.target, q[start + 400:stop])
            self.assertFalse(window.context.flags.writeable)
            self.assertFalse(window.target.flags.writeable)
            np.testing.assert_allclose(np.linalg.norm(window.context, axis=1), 1.0, atol=2e-6)
            self.assertGreaterEqual(float(np.min(np.sum(window.context[1:] * window.context[:-1], axis=1))), -1e-6)

    def test_outer_and_inner_ownership_is_exactly_10_5_then_8_2(self):
        ownership = reader._inner_ownership(self.binding.parent.expected_train)
        all_train = set(self.binding.parent.expected_train.users)
        self.assertEqual(set(ownership), {0, 1, 2})
        for fold in ownership.values():
            self.assertEqual(len(fold["outer_fit"]), 10)
            self.assertEqual(len(fold["outer_validation"]), 5)
            self.assertEqual(set(fold["outer_fit"]).intersection(fold["outer_validation"]), set())
            self.assertEqual(set(fold["outer_fit"]).union(fold["outer_validation"]), all_train)
            self.assertEqual(len(fold["inner_fit"]), 8)
            self.assertEqual(len(fold["inner_selector"]), 2)
            self.assertEqual(set(fold["inner_fit"]).intersection(fold["inner_selector"]), set())
            self.assertEqual(set(fold["inner_fit"]).union(fold["inner_selector"]), set(fold["outer_fit"]))

    def test_hash_rank_split_and_1024_even_sampler_match_shared_contract(self):
        expected = self.binding.parent.expected_train
        for fold_id, outer_validation in enumerate(expected.fold_validation_order):
            outer_fit = sorted(set(expected.users) - set(outer_validation))
            ranked = sorted(outer_fit, key=lambda user: (
                hashlib.sha256(
                    f"q1-inner-split-v1\0{20261029}\0{fold_id}\0{user}".encode("utf-8")
                ).digest(), user,
            ))
            self.assertEqual(self.binding.ownership[fold_id]["inner_selector"], tuple(sorted(ranked[:2])))
            self.assertEqual(self.binding.ownership[fold_id]["inner_fit"], tuple(sorted(ranked[2:])))
        positions = reader.selector_origin_indices(20_001)
        expected_positions = np.linspace(0, 20_000, num=1024, dtype=np.int64)
        np.testing.assert_array_equal(positions, expected_positions)
        self.assertEqual(int(positions[0]), 0)
        self.assertEqual(int(positions[-1]), 20_000)

    def test_plan_is_only_selected_train_prefix_and_uses_shift_one(self):
        plan = self._plan()
        self.assertEqual(len(plan), 8 * 2 + 2 * 581)
        self.assertEqual([row.role for row in plan],
                         ["inner_fit"] * (8 * 2) + ["inner_selector"] * (2 * 581))
        for row in plan:
            self.assertEqual(row.context_start, row.origin_index - 399)
            self.assertEqual(row.slice_stop, row.context_start + 420)
            self.assertEqual(row.slice_stop - row.context_start, 400 + 20)
            self.assertIn(row.user_id, self.binding.parent.expected_train.users)

    def test_guard_denies_dev_final_foreign_file_and_oversized_or_wrong_slice(self):
        plan = self._plan()
        _datasets, allowed_paths, allowed_reads = reader._preflight_selected_train_metadata(
            self.source_file, self.binding, plan)
        users_group = self.source_file["users"]
        selected_user = self.binding.ownership[0]["outer_fit"][0]
        train_ds = self.source_file[
            f"users/{selected_user}/sessions/session_0/segments/0000/quaternion"
        ]
        foreign_path = self.tmp / "foreign.h5"
        with h5py.File(foreign_path, "w") as foreign:
            g = foreign.create_group("users").create_group("foreign")
            q = g.create_dataset("quaternion", data=_synthetic_quaternions(), chunks=(1000, 4))
            foreign_ds = q
            guard = reader.HDF5ReadGuard(
                self.source_file, allowed_metadata_paths=allowed_paths,
                allowed_reads=allowed_reads,
            )
            with guard:
                with self.assertRaises(PermissionError):
                    users_group["synthetic_final"]
                with self.assertRaises(PermissionError):
                    foreign_ds[0:420]
                with self.assertRaises(PermissionError):
                    train_ds[0:421]
                with self.assertRaises(PermissionError):
                    train_ds[0:420:2]
                with self.assertRaises(PermissionError):
                    train_ds.read_direct(np.empty((420, 4), dtype=np.float32))
            self.assertGreaterEqual(guard.audit["denied_group_dereferences"], 1)
            self.assertGreaterEqual(guard.audit["denied_foreign_file_accesses"], 1)
            self.assertGreaterEqual(guard.audit["denied_value_reads"], 4)

    def test_hardlink_alias_and_soft_link_are_rejected_before_value_reads(self):
        alias_path = self.tmp / "alias.h5"
        _write_synthetic_h5(alias_path, self.binding, alias_train_to_final=True)
        with h5py.File(alias_path, "r") as alias_file:
            with self.assertRaisesRegex(ValueError, "hard-link alias"):
                list(reader._iter_windows_from_open_file(alias_file, self.binding, self._plan()))

        soft_path = self.tmp / "soft.h5"
        _write_synthetic_h5(soft_path, self.binding, soft_train_quaternion=True)
        with h5py.File(soft_path, "r") as soft_file:
            with self.assertRaisesRegex(ValueError, "soft or external link"):
                list(reader._iter_windows_from_open_file(soft_file, self.binding, self._plan()))

    def test_virtual_or_external_storage_is_rejected(self):
        if not hasattr(h5py, "VirtualLayout"):
            self.skipTest("installed h5py has no virtual-dataset support")
        virtual_path = self.tmp / "virtual.h5"
        _write_synthetic_h5(virtual_path, self.binding, virtual_train_quaternion=True)
        with h5py.File(virtual_path, "r") as virtual_file:
            with self.assertRaisesRegex(ValueError, "dtype/dimensions/storage"):
                list(reader._iter_windows_from_open_file(virtual_file, self.binding, self._plan()))

    def test_source_byte_mismatch_fails_before_h5py_open(self):
        wrong_path = self.tmp / "not_hdf5.bin"
        wrong_path.write_bytes(b"synthetic bytes, not source data")
        identity = reader._file_identity(wrong_path)
        binding = replace(self.binding, source_path=wrong_path, source_identity=identity)
        opens = []
        original_file = h5py.File

        def spy_file(*args, **kwargs):
            opens.append(args[0] if args else kwargs.get("name"))
            return original_file(*args, **kwargs)

        with mock.patch.object(reader.h5py, "File", spy_file):
            with self.assertRaisesRegex(ValueError, "source HDF5 bytes differ"):
                with reader._verified_source_file(binding):
                    self.fail("the h5py context must not be reached")
        self.assertEqual(opens, [])

    def test_bad_quaternion_norm_or_sign_is_rejected(self):
        q = _synthetic_quaternions()
        bad_norm = q.copy()
        bad_norm[30] *= 2.0
        with self.assertRaisesRegex(ValueError, "nonunit"):
            reader._validate_quaternion_window(bad_norm[:420])
        bad_sign = q.copy()
        bad_sign[30] *= -1.0
        with self.assertRaisesRegex(ValueError, "hemisphere"):
            reader._validate_quaternion_window(bad_sign[:420])


class ProductionMetadataTests(unittest.TestCase):
    def _install_no_payload_open_spies(self):
        original = Path.open
        cache_path = reader.origin_inventory.production_inputs().cache
        raw_root = (REPO_ROOT / "data" / "raw" / "whoisalyx").resolve()
        blocked = {
            str(reader.SOURCE_HDF5_PATH.resolve()).casefold(),
            str(cache_path.resolve()).casefold(),
        }
        opened = []

        def path_open(path, *args, **kwargs):
            absolute = Path(os.path.abspath(path))
            key = str(absolute.resolve(strict=False)).casefold()
            try:
                in_raw = absolute.resolve(strict=False).is_relative_to(raw_root)
            except AttributeError:
                in_raw = str(raw_root).casefold() in key
            if key in blocked or in_raw:
                raise AssertionError("actual source HDF5/cache/source recording open was attempted")
            opened.append(key)
            return original(path, *args, **kwargs)

        return mock.patch.object(Path, "open", path_open), opened

    def _validated_metadata_only(self):
        path_spy, opened = self._install_no_payload_open_spies()
        with path_spy, mock.patch.object(
                reader.h5py, "File", side_effect=AssertionError("actual HDF5 open forbidden in this test")):
            binding = reader.validate_metadata_bindings()
        self.assertFalse(any("alyx.h5" in path.casefold() or "features.h5" in path.casefold()
                             for path in opened))
        return binding

    def test_exact_actual_origin_receipt_map_and_parent_validate_json_only(self):
        binding = self._validated_metadata_only()
        self.assertEqual(binding.origin_receipt_sha256, reader.ORIGIN_RECEIPT_SHA256)
        self.assertEqual(binding.origin_map_sha256, reader.ORIGIN_MAP_SHA256)
        self.assertEqual(binding.origin_binary_sha256, reader.ORIGIN_BINARY_SHA256)
        self.assertEqual(len(binding.origin_rows), 52)
        self.assertEqual(len(binding.parent.expected_train.users), 15)
        self.assertEqual(binding.parent.expected_train.expected_windows, 3_330_312)
        self.assertEqual(binding.parent.fold_digest, "db9618ba19224f1c48735a80c45b95d07e75e406633b51350137feb0d0569df2")
        self.assertGreater(len(binding.origin_values), 3_000_000)

    def test_wrong_actual_receipt_field_and_split_fail_before_hdf5(self):
        binding = self._validated_metadata_only()
        raw = reader.ORIGIN_RECEIPT_PATH.read_bytes()
        doc = reader._strict_json(raw, "test receipt")
        doc["realized_origin_order_sha256"] = "0" * 64
        doc["split_sha256"] = "0" * 64
        with mock.patch.object(reader.h5py, "File", side_effect=AssertionError("HDF5 must remain unopened")):
            with self.assertRaisesRegex(ValueError, "actual train-origin receipt field differs"):
                reader._validate_origin_receipt(doc, binding.parent, raw)
            wrong_receipt = reader._strict_json(raw, "receipt")
            wrong_receipt["split_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "actual train-origin receipt field differs: split_sha256"):
                reader._validate_origin_receipt(wrong_receipt, binding.parent, raw)

    def test_no_fixed_protocol_or_approval_fails_before_source_open(self):
        binding = self._validated_metadata_only()
        path_spy, opened = self._install_no_payload_open_spies()
        missing_protocol = self.tmp_path() / "missing_protocol.json"
        missing_approval = self.tmp_path() / "missing_approval.json"
        with mock.patch.object(reader, "PROTOCOL_PATH", missing_protocol), \
                mock.patch.object(reader, "APPROVAL_PATH", missing_approval), \
                path_spy, mock.patch.object(
                    reader.h5py, "File", side_effect=AssertionError("HDF5 must remain unopened")):
            with self.assertRaisesRegex(PermissionError, "fixed (approval|protocol) metadata is not present"):
                reader.load_fixed_approval(binding)
        self.assertFalse(any("alyx.h5" in path.casefold() for path in opened))

    def tmp_path(self):
        path = Path(tempfile.mkdtemp(prefix="q1-reader-missing-approval-", dir=reader.STAGING_ROOT))
        self.addCleanup(lambda: __import__("shutil").rmtree(path, ignore_errors=True))
        return path

    def test_authorization_requires_exact_frozen_scope_and_hashes(self):
        tmp = Path(tempfile.mkdtemp(prefix="q1-reader-auth-", dir=reader.STAGING_ROOT))
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        binding = _synthetic_binding(tmp)
        boundary = {
            "source_partition": "train",
            "source_hdf5_relative_path": reader.SOURCE_HDF5_RELATIVE_PATH,
            "source_hdf5_sha256": reader.SOURCE_HDF5_SHA256,
            "split_sha256": reader.SPLIT_SHA256,
            "fold_digest": binding.parent.fold_digest,
            "outer_train_users": 15,
            "fold_count": 3,
            "outer_fit_users_per_fold": 10,
            "outer_validation_users_per_fold": 5,
            "inner_fit_users_per_fold": 8,
            "inner_selector_users_per_fold": 2,
            "inner_split_seed": reader.INNER_SPLIT_SEED,
            "ownership_plan_sha256": binding.ownership_plan_sha256,
            "origin_sampling_strategy": "endpoint_inclusive_evenly_spaced_per_user",
            "selector_origins_per_user": reader.SELECTOR_ORIGINS_PER_USER,
            "context_samples": 400,
            "horizon_samples": 20,
            "shift": 1,
        }
        protocol = {
            "schema_version": 1,
            "kind": "q1_train_prefix_protocol",
            "status": "FROZEN_BEFORE_PREFIX",
            "protocol_id": "q1-train-prefix-protocol-test-v1",
            "data_boundary": boundary,
        }
        protocol_raw = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
        ownership = {
            "source_partition": "train",
            "outer_train_users": 15,
            "fold_count": 3,
            "outer_fit_users_per_fold": 10,
            "outer_validation_users_per_fold": 5,
            "inner_fit_users_per_fold": 8,
            "inner_selector_users_per_fold": 2,
            "inner_split_seed": reader.INNER_SPLIT_SEED,
            "ownership_plan_sha256": binding.ownership_plan_sha256,
            "origin_sampling_strategy": "endpoint_inclusive_evenly_spaced_per_user",
            "selector_origins_per_user": reader.SELECTOR_ORIGINS_PER_USER,
        }
        doc = {
            "schema_version": 1,
            "kind": "q1_train_prefix_approval",
            "status": "APPROVED_AFTER_INDEPENDENT_REVIEW",
            "authorization_id": "q1-train-prefix-test-v1",
            "approved_by_role": "coordinator",
            "approved_utc": "2026-10-29T12:00:00+00:00",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha,
            "reviewed_artifacts": reader._protocol_and_approval_bindings(binding),
            "ownership": ownership,
            "authorized_prefixes": [
                {"fold_id": 0, "role": "inner_fit",
                 "sampling": "endpoint_inclusive_evenly_spaced_per_user_v1", "origins_per_user": 2},
                {"fold_id": 0, "role": "inner_selector",
                 "sampling": "endpoint_inclusive_evenly_spaced_per_user_v1",
                 "origins_per_user": reader.SELECTOR_ORIGINS_PER_USER},
            ],
        }
        approval_raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        access = reader._validate_authorization(doc, protocol, protocol_sha, binding,
                                                hashlib.sha256(approval_raw).hexdigest())
        self.assertEqual(access.authorization_id, "q1-train-prefix-test-v1")
        self.assertEqual(len(reader.build_read_plan(binding, access)), 8 * 2 + 2 * 581)
        self.assertNotIn("user_id", reader.safe_aggregate_template(binding, access))

        bad_status = dict(doc, status="PENDING_REVIEW")
        with self.assertRaisesRegex(PermissionError, "not approved"):
            reader._validate_authorization(bad_status, protocol, protocol_sha, binding,
                                           hashlib.sha256(approval_raw).hexdigest())
        bad_protocol = dict(protocol, data_boundary=dict(boundary, source_partition="development"))
        bad_protocol_raw = json.dumps(bad_protocol, sort_keys=True, separators=(",", ":")).encode()
        bad_protocol_sha = hashlib.sha256(bad_protocol_raw).hexdigest()
        with self.assertRaisesRegex(PermissionError, "data boundary"):
            reader._validate_authorization(doc, bad_protocol, bad_protocol_sha, binding,
                                           hashlib.sha256(approval_raw).hexdigest())
        bad_count = json.loads(json.dumps(doc))
        bad_count["authorized_prefixes"][0]["origins_per_user"] = True
        with self.assertRaisesRegex(ValueError, "integer at least"):
            reader._validate_authorization(bad_count, protocol, protocol_sha, binding,
                                           hashlib.sha256(approval_raw).hexdigest())


if __name__ == "__main__":
    unittest.main()

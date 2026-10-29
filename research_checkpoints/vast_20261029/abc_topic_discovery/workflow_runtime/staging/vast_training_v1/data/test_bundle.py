"""Generated-only tests for the VAST train-bundle contract.

These tests synthesize quaternion bytes in memory. They never import NumPy,
h5py, Torch, or any project data reader, and they never inspect a human-data
path.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
import tempfile
import unittest
from pathlib import Path

from bundle import (
    BundleAuthorizationError,
    BundleError,
    CONTEXT_SAMPLES,
    DATA_FILE,
    FOLD_ID,
    MANIFEST_FILE,
    PURPOSE,
    SCREEN_PURPOSE,
    ROLE,
    SegmentInput,
    attestation_signing_bytes,
    export_authorized_fit10_bundle,
    make_test_attestation,
    open_prepared_train_bundle,
    write_synthetic_bundle,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _source_binding() -> dict[str, str]:
    return {
        "source_asset_sha256": _digest("fixture-source"),
        "split_sha256": _digest("fixture-split"),
        "fit_role_inventory_sha256": _digest("fixture-fit-role"),
        "fit_origin_inventory_sha256": _digest("fixture-fit-origins"),
        "source_loader_sha256": _digest("fixture-loader"),
        "source_authorization_sha256": _digest("fixture-source-authorization"),
        "source_read_receipt_sha256": _digest("fixture-source-receipt"),
    }


def _rows(quaternion: tuple[float, float, float, float], n: int = 420) -> bytes:
    row = struct.pack("<4f", *quaternion)
    return row * n


class _OpaqueBinding:
    def __init__(
        self,
        identity: str,
        run_binding: object | None = None,
        expected_identity_sha256: str | None = None,
    ) -> None:
        self.bundle_identity_sha256 = identity
        self.run_binding = run_binding
        self.expected_identity_sha256 = expected_identity_sha256
        self.active = True

    def assert_live(self) -> None:
        if not self.active:
            raise RuntimeError("expired synthetic grant")

    def abort(self) -> None:
        self.active = False

    def __reduce__(self):
        raise TypeError("synthetic binding must not be serialized")


class _SyntheticVerifier:
    def __init__(self, *, deny: bool = False) -> None:
        self.deny = deny
        self.calls = []

    def verify_bundle(
        self,
        *,
        canonical_manifest: bytes,
        bundle_identity_sha256: str,
        payload_sha256: str,
        expected_identity_sha256: str | None,
        expected_fold_id: int,
        expected_run_binding: object | None,
    ) -> object:
        self.calls.append((bundle_identity_sha256, expected_run_binding, expected_fold_id))
        if self.deny:
            raise BundleAuthorizationError("synthetic denial")
        manifest = json.loads(canonical_manifest)
        if manifest["role"] != ROLE or manifest["fold_id"] != expected_fold_id:
            raise BundleAuthorizationError("synthetic scope mismatch")
        if manifest["payload"]["sha256"] != payload_sha256:
            raise BundleAuthorizationError("synthetic payload mismatch")
        if expected_identity_sha256 not in (None, bundle_identity_sha256):
            raise BundleAuthorizationError("synthetic resume identity mismatch")
        return _OpaqueBinding(bundle_identity_sha256, expected_run_binding)


class _SyntheticExportGrant:
    def __init__(self, fold_id: int) -> None:
        self.purpose = PURPOSE
        self.role = ROLE
        self.fold_id = fold_id

    def assert_live(self) -> None:
        return None


class _SyntheticExportRoot:
    def __init__(self, *, grant_fold: int) -> None:
        self.grant_fold = grant_fold
        self.events = []
        self.source_binding = _source_binding()

    def issue_fit10_export_grant(self, fold_id: int, *, purpose: str) -> object:
        self.events.append(("issue", fold_id))
        grant = _SyntheticExportGrant(self.grant_fold)
        grant.purpose = purpose
        return grant

    def verify_fit10_export_grant(self, grant: object) -> dict[str, str]:
        self.events.append(("verify", grant.fold_id))
        return {k: v for k, v in self.source_binding.items() if k != "source_read_receipt_sha256"}

    def open_fit10_segments(self, grant: object):
        self.events.append(("open", grant.fold_id))
        return [
            SegmentInput(0, 420, _rows((1.0, 0.0, 0.0, 0.0))),
            SegmentInput(1, 420, _rows((0.0, 1.0, 0.0, 0.0))),
        ]

    def attest_bundle_identity(self, grant: object, identity_sha256: str) -> dict[str, str]:
        self.events.append(("attest", grant.fold_id))
        return make_test_attestation(
            identity_sha256, self.source_binding["source_authorization_sha256"]
        )

    def finalize_fit10_source_binding(self, grant: object) -> dict[str, str]:
        self.events.append(("finalize", grant.fold_id))
        return self.source_binding


class TrainBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vast-bundle-fixture-")
        self.root = Path(self.temp.name)
        self.bundle_dir = self.root / "bundle"
        # Distinct constant unit quaternions make segment crossing observable.
        self.expected0 = _rows((1.0, 0.0, 0.0, 0.0))
        self.expected1 = _rows((0.0, 1.0, 0.0, 0.0))

    def tearDown(self) -> None:
        for path in self.root.rglob("*"):
            try:
                os.chmod(path, 0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        self.temp.cleanup()

    def _build(self, destination: Path | None = None) -> str:
        target = destination or self.bundle_dir
        source_binding = _source_binding()
        return write_synthetic_bundle(
            target,
            [
                SegmentInput(0, 420, self.expected0),
                SegmentInput(1, 420, self.expected1),
            ],
            fold_id=FOLD_ID,
            source_binding=source_binding,
            synthetic_attestation=lambda identity: make_test_attestation(
                identity, source_binding["source_authorization_sha256"]
            ),
        )

    def _open(self, *, expected_identity: str | None = None, verifier=None):
        run_binding = _OpaqueBinding(
            _digest("run-binding"), expected_identity_sha256=expected_identity
        )
        return open_prepared_train_bundle(
            self.bundle_dir,
            trusted_verifier=verifier or _SyntheticVerifier(),
            expected_fold_id=FOLD_ID,
            expected_run_binding=run_binding,
            expected_identity_sha256=expected_identity,
        )

    def test_windows_are_segment_local_and_have_400_20_shapes(self) -> None:
        identity = self._build()
        with self._open(expected_identity=identity) as bundle:
            self.assertEqual(bundle.total_origins, 2)
            self.assertEqual(bundle.origin_for_rank(0), (0, 0))
            self.assertEqual(bundle.origin_for_rank(1), (1, 0))
            context0, target0 = bundle.sample_rank(0)
            context1, target1 = bundle.sample_rank(1)
            self.assertEqual(len(context0), CONTEXT_SAMPLES * 16)
            self.assertEqual(len(target0), 20 * 16)
            self.assertEqual(context0, self.expected0[: CONTEXT_SAMPLES * 16])
            self.assertEqual(target0, self.expected0[CONTEXT_SAMPLES * 16 :])
            self.assertEqual(context1, self.expected1[: CONTEXT_SAMPLES * 16])
            self.assertEqual(target1, self.expected1[CONTEXT_SAMPLES * 16 :])
            # A 420-row segment has exactly one valid start. The next index may
            # not spill into the following segment.
            with self.assertRaises(IndexError):
                bundle.sample_window(0, 1)

    def test_resume_reopens_same_bytes_and_rank_mapping_exactly(self) -> None:
        identity = self._build()
        ranks = (1, 0, 1, 1, 0)
        first_verifier = _SyntheticVerifier()
        with self._open(expected_identity=identity, verifier=first_verifier) as first:
            first_rows = [(first.origin_for_rank(rank), first.sample_rank(rank)) for rank in ranks]
        second_verifier = _SyntheticVerifier()
        with self._open(expected_identity=identity, verifier=second_verifier) as resumed:
            resumed_rows = [(resumed.origin_for_rank(rank), resumed.sample_rank(rank)) for rank in ranks]
            self.assertEqual(resumed.identity_sha256, identity)
            self.assertEqual(first_rows, resumed_rows)
            self.assertEqual(resumed.binding.run_binding, second_verifier.calls[0][1])
        self.assertEqual(len(first_verifier.calls), 1)
        self.assertEqual(len(second_verifier.calls), 1)

    def test_illegal_role_is_denied(self) -> None:
        self._build()
        manifest_path = self.bundle_dir / MANIFEST_FILE
        os.chmod(manifest_path, 0o600)
        manifest = json.loads(manifest_path.read_bytes())
        manifest["role"] = "outer_validation5"
        manifest_path.write_bytes(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
        os.chmod(manifest_path, 0o400)
        with self.assertRaises(BundleError):
            self._open()

    def test_corrupted_payload_is_denied_before_verifier(self) -> None:
        self._build()
        payload = self.bundle_dir / DATA_FILE
        os.chmod(payload, 0o600)
        raw = bytearray(payload.read_bytes())
        raw[0] ^= 0x01
        payload.write_bytes(raw)
        os.chmod(payload, 0o400)
        verifier = _SyntheticVerifier()
        with self.assertRaises(BundleError):
            self._open(verifier=verifier)
        self.assertEqual(verifier.calls, [])

    def test_unexpected_source_member_denies_bundle_layout(self) -> None:
        self._build()
        os.chmod(self.bundle_dir, 0o700)
        (self.bundle_dir / "source.h5").write_bytes(b"synthetic fixture only")
        os.chmod(self.bundle_dir, 0o500)
        with self.assertRaises(BundleError):
            self._open()

    def test_wrong_fold_is_denied_before_trusted_verifier(self) -> None:
        self._build()
        verifier = _SyntheticVerifier()
        with self.assertRaises(BundleAuthorizationError):
            open_prepared_train_bundle(
                self.bundle_dir,
                trusted_verifier=verifier,
                expected_fold_id=1,
                expected_run_binding=_OpaqueBinding(_digest("wrong-fold-run")),
            )
        self.assertEqual(verifier.calls, [])

    def test_resume_identity_must_be_carried_by_root_binding(self) -> None:
        identity = self._build()
        verifier = _SyntheticVerifier()
        run_binding = _OpaqueBinding(
            _digest("run-binding"), expected_identity_sha256=_digest("different-bundle")
        )
        with self.assertRaises(BundleAuthorizationError):
            open_prepared_train_bundle(
                self.bundle_dir,
                trusted_verifier=verifier,
                expected_fold_id=FOLD_ID,
                expected_run_binding=run_binding,
                expected_identity_sha256=identity,
            )
        self.assertEqual(verifier.calls, [])

    def test_verifier_denial_and_nonserializable_binding_contract(self) -> None:
        self._build()
        with self.assertRaises(BundleAuthorizationError):
            self._open(verifier=_SyntheticVerifier(deny=True))
        identity = self._build(self.root / "bundle2")
        with open_prepared_train_bundle(
            self.root / "bundle2",
            trusted_verifier=_SyntheticVerifier(),
            expected_fold_id=FOLD_ID,
            expected_run_binding=_OpaqueBinding(identity),
        ) as bundle:
            with self.assertRaises(TypeError):
                pickle.dumps(bundle.binding)
            self.assertEqual(bundle.identity_sha256, identity)

    def test_attestation_message_binds_issuer_metadata(self) -> None:
        source = _source_binding()
        signature = make_test_attestation(_digest("identity"), source["source_authorization_sha256"])
        before = attestation_signing_bytes(_digest("identity"), signature)
        changed = dict(signature)
        changed["issued_utc"] = "2026-10-30T00:00:00Z"
        after = attestation_signing_bytes(_digest("identity"), changed)
        self.assertNotEqual(before, after)

    def test_exporter_checks_fold_grant_before_open_and_writes_only_allowlist(self) -> None:
        root = _SyntheticExportRoot(grant_fold=1)
        with self.assertRaises(BundleAuthorizationError):
            export_authorized_fit10_bundle(self.root / "denied", fold_id=0, root=root)
        self.assertEqual(root.events, [("issue", 0)])

        root = _SyntheticExportRoot(grant_fold=0)
        identity = export_authorized_fit10_bundle(self.bundle_dir, fold_id=0, root=root)
        self.assertEqual([event[0] for event in root.events], ["issue", "verify", "open", "finalize", "attest"])
        self.assertEqual(
            {entry.name for entry in self.bundle_dir.iterdir()},
            {MANIFEST_FILE, DATA_FILE},
        )
        self.assertEqual(identity, json.loads((self.bundle_dir / MANIFEST_FILE).read_bytes())["bundle_identity_sha256"])

    def test_screen_has_distinct_signed_purpose_and_post_read_receipt(self) -> None:
        root = _SyntheticExportRoot(grant_fold=0)
        export_authorized_fit10_bundle(self.bundle_dir, fold_id=0, root=root, purpose=SCREEN_PURPOSE)
        manifest = json.loads((self.bundle_dir / MANIFEST_FILE).read_bytes())
        self.assertEqual(manifest["purpose"], SCREEN_PURPOSE)
        self.assertEqual([event[0] for event in root.events],
                         ["issue", "verify", "open", "finalize", "attest"])
        self.assertEqual(manifest["source_binding"]["source_read_receipt_sha256"],
                         root.source_binding["source_read_receipt_sha256"])

    def test_changed_static_binding_after_read_is_rejected(self) -> None:
        root = _SyntheticExportRoot(grant_fold=0)
        finalize = root.finalize_fit10_source_binding
        def changed(grant):
            return {**finalize(grant), "split_sha256": _digest("different-split")}
        root.finalize_fit10_source_binding = changed
        with self.assertRaises(BundleAuthorizationError):
            export_authorized_fit10_bundle(self.bundle_dir, fold_id=0, root=root)
        self.assertFalse((self.bundle_dir / MANIFEST_FILE).exists())

    def test_symlink_member_denies_bundle_when_platform_supports_it(self) -> None:
        self._build()
        payload = self.bundle_dir / DATA_FILE
        moved = self.root / "payload-external.bin"
        os.chmod(self.bundle_dir, 0o700)
        if os.name == "nt":
            os.chmod(payload, 0o600)
        try:
            payload.rename(moved)
        except OSError as exc:
            self.skipTest(f"platform cannot move the protected fixture: {exc}")
        try:
            payload.symlink_to(moved)
        except (OSError, NotImplementedError) as exc:
            moved.rename(payload)
            self.skipTest(f"symlink creation unavailable: {exc}")
        os.chmod(self.bundle_dir, 0o500)
        with self.assertRaises(BundleError):
            self._open()


if __name__ == "__main__":
    unittest.main()

"""Generated-only Ed25519 integration checks; no source recordings are read."""
import base64
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vast_training_v1.data.bundle import (  # noqa: E402
    BundleAuthorizationError, SegmentInput, attestation_signing_bytes,
    open_prepared_train_bundle, write_synthetic_bundle,
)
from vast_training_v1.data.ed25519_verifier import Ed25519BundleVerifier  # noqa: E402

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    Ed25519PrivateKey = None


class Lease:
    def __init__(self, identity=None):
        self.bundle_identity_sha256 = identity
        self.active = True
    def assert_live(self):
        if not self.active:
            raise RuntimeError("revoked")


class Authority:
    def __init__(self):
        self.calls = 0
    def authorize_verified_bundle(self, *, manifest, bundle_identity_sha256, **kwargs):
        self.calls += 1
        return Lease(bundle_identity_sha256)


@unittest.skipIf(Ed25519PrivateKey is None, "cryptography is not installed")
class VerifierTests(unittest.TestCase):
    def test_real_signature_and_untrusted_key(self):
        key = Ed25519PrivateKey.generate()
        pub = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        digest = hashlib.sha256(b"synthetic").hexdigest()
        binding = {k: digest for k in (
            "source_asset_sha256", "split_sha256", "fit_role_inventory_sha256",
            "fit_origin_inventory_sha256", "source_loader_sha256",
            "source_authorization_sha256", "source_read_receipt_sha256")}
        def sign(identity):
            metadata = {"algorithm": "Ed25519", "key_id": "fixture", "authorization_sha256": digest,
                        "issued_utc": "2026-10-29T00:00:00Z",
                        "content_validation": "finite_unit_wxyz_hemisphere_source_v1"}
            metadata["identity_signature"] = base64.b64encode(
                key.sign(attestation_signing_bytes(identity, metadata))).decode("ascii")
            return metadata
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bundle"
            identity = write_synthetic_bundle(path, [SegmentInput(0, 420, struct.pack("<4f", 1, 0, 0, 0) * 420)],
                                              source_binding=binding, synthetic_attestation=sign)
            authority = Authority()
            verifier = Ed25519BundleVerifier(pinned_public_keys={"fixture": pub}, authority=authority)
            try:
                with open_prepared_train_bundle(path, trusted_verifier=verifier, expected_fold_id=0,
                                                expected_run_binding=Lease(), expected_identity_sha256=None) as bundle:
                    self.assertEqual(bundle.identity_sha256, identity)
                    bundle.binding.active = False
                    with self.assertRaises(BundleAuthorizationError):
                        bundle.assert_active()
                self.assertEqual(authority.calls, 1)
                wrong_key = Ed25519PrivateKey.generate().public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                verifier = Ed25519BundleVerifier(pinned_public_keys={"fixture": wrong_key}, authority=authority)
                with self.assertRaises(BundleAuthorizationError):
                    open_prepared_train_bundle(path, trusted_verifier=verifier, expected_fold_id=0,
                                               expected_run_binding=Lease())
                self.assertEqual(authority.calls, 1)
            finally:
                os.chmod(path, 0o700)
                for item in path.iterdir():
                    os.chmod(item, 0o600)

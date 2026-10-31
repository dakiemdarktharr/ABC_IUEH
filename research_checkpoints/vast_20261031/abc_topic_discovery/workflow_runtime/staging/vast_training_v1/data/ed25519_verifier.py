"""Signature verification for a separately authenticated training authority.

Keys must be pinned by a trusted launcher, not supplied by a bundle. Signature
verification establishes who signed the manifest; it does not itself grant a
data-use, train, resume, or budget authorization.
"""
from __future__ import annotations

import base64
from typing import Any, Mapping

from .bundle import BundleAuthorizationError, _validate_manifest, attestation_signing_bytes


class Ed25519BundleVerifier:
    def __init__(self, *, pinned_public_keys: Mapping[str, bytes], authority: Any):
        if not pinned_public_keys or any(
            not isinstance(k, str) or not k or not isinstance(v, bytes) or len(v) != 32
            for k, v in pinned_public_keys.items()
        ):
            raise ValueError("out-of-band Ed25519 public keys must be 32 bytes")
        if not callable(getattr(authority, "authorize_verified_bundle", None)):
            raise ValueError("a separate live training authority is required")
        self._keys = dict(pinned_public_keys)
        self._authority = authority

    def verify_bundle(self, *, canonical_manifest: bytes, bundle_identity_sha256: str,
                      payload_sha256: str, expected_identity_sha256: str | None,
                      expected_fold_id: int, expected_run_binding: object) -> object:
        manifest, identity = _validate_manifest(canonical_manifest)
        if (identity != bundle_identity_sha256 or manifest["payload"]["sha256"] != payload_sha256
                or manifest["fold_id"] != expected_fold_id
                or expected_identity_sha256 not in (None, identity)):
            raise BundleAuthorizationError("manifest differs from the requested run binding")
        attestation = manifest["root_attestation"]
        key = self._keys.get(attestation["key_id"])
        if key is None:
            raise BundleAuthorizationError("signing key is not pinned by the launcher")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(key).verify(
                base64.b64decode(attestation["identity_signature"], validate=True),
                attestation_signing_bytes(identity, attestation),
            )
        except ImportError as exc:
            raise BundleAuthorizationError("Ed25519 verification dependency is unavailable") from exc
        except Exception as exc:
            raise BundleAuthorizationError("Ed25519 signature verification failed") from exc
        return self._authority.authorize_verified_bundle(
            manifest=manifest, bundle_identity_sha256=identity,
            expected_run_binding=expected_run_binding,
            expected_identity_sha256=expected_identity_sha256,
        )

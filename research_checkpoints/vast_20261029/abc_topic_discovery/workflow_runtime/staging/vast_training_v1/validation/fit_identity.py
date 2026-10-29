"""Verify signed fit-bundle identities against an out-of-band pinned key."""
import base64
from pathlib import Path


def verify_fit_identities(manifests, *, public_keys, key_ids):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from ..data import bundle
    from ..source_adapter_v1.source_adapter import SOURCE_SHA256, SPLIT_SHA256
    folds = {0, 1, 2}
    if (set(manifests) != folds or set(public_keys) != folds or set(key_ids) != folds or
            any(type(public_keys[fold]) is not bytes or len(public_keys[fold]) != 32 or
                type(key_ids[fold]) is not str or not key_ids[fold] for fold in folds)):
        raise ValueError('each fold requires its manifest, independently pinned 32-byte key, and key id')
    result = {}
    for fold, path in sorted(manifests.items()):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError('fit manifest must be a bounded regular file')
        manifest, identity = bundle._validate_manifest(path.read_bytes())
        attestation = manifest['root_attestation']
        source = manifest['source_binding']
        if (manifest['fold_id'] != fold or manifest['purpose'] != bundle.SCREEN_PURPOSE or
            manifest['role'] != bundle.ROLE or attestation['key_id'] != key_ids[fold] or
            source['source_asset_sha256'] != SOURCE_SHA256 or source['split_sha256'] != SPLIT_SHA256):
            raise ValueError('fit manifest scope differs from pinned screen source')
        verifier = Ed25519PublicKey.from_public_bytes(public_keys[fold])
        verifier.verify(base64.b64decode(attestation['identity_signature'], validate=True),
                        bundle.attestation_signing_bytes(identity, attestation))
        result[fold] = identity
    return result

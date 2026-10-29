"""Local export authority for an operator-invoked, single-fold train bundle.

The operator's call is the decision to prepare the requested train-only bundle.
The signature authenticates bundle integrity and its local origin, not data-use
permission, a cloud transfer, training, or institutional approval. Keep the
private key local; pin the returned public key independently at the consumer.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import uuid


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


class _Grant:
    def assert_live(self):
        if datetime.now(timezone.utc) >= self._expires:
            raise PermissionError('local export grant expired')


class LocalExportAuthority:
    """Issue one-use local grants and Ed25519 attestations for fixed metadata.

    Call ``create_local_source_root`` to obtain this authority with metadata
    checked by the existing reader. No remote service or data source is opened
    at construction. This Python process is not an isolation/security boundary.
    """
    def __init__(self, binding, loader, adapter, private_key):
        self._binding = binding
        self._loader = loader
        self._adapter = adapter
        self._key = private_key
        self._states = {}
        from cryptography.hazmat.primitives import serialization
        self.public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.key_id = 'local-ed25519-' + hashlib.sha256(self.public_key).hexdigest()[:24]

    def issue_fit10_export_grant(self, request):
        a = self._adapter
        if (request.purpose not in a.ALLOWED_PURPOSES or request.role != a.ROLE or
                request.fold_id not in a.ALLOWED_FOLDS or
                request.source_asset_sha256 != a.SOURCE_SHA256 or
                request.split_sha256 != a.SPLIT_SHA256 or
                request.source_loader_sha256 != a.V3_LOADER_SHA256 or
                request.source_reader_sha256 != a.V3_READER_SHA256 or
                request.source_reader_review_sha256 != a.V3_READER_REVIEW_SHA256):
            raise PermissionError('local export request differs from the fixed train source')
        rows = self._loader._canonical_fit_rows(self._binding, request.fold_id, request.role)
        role_hash = a.fit_role_inventory_digest(rows, request.fold_id, request.role)
        origin_hash = self._loader._pool_origin_digest(rows, self._binding, request.fold_id, request.role)
        token = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=a.MAX_LOADER_SECONDS)
        decision = {'kind': 'operator_invoked_local_fit10_export', 'id': token,
                    'purpose': request.purpose, 'role': request.role, 'fold': request.fold_id,
                    'source_sha256': request.source_asset_sha256, 'split_sha256': request.split_sha256,
                    'role_sha256': role_hash, 'origin_sha256': origin_hash,
                    'bundle_module_sha256': request.bundle_module_sha256}
        decision_hash = _digest(decision)
        grant = _Grant()
        for key, value in {
            'grant_kind': a.GRANT_KIND, 'grant_namespace': a.GRANT_NAMESPACE,
            'purpose': request.purpose, 'role': request.role, 'fold_id': request.fold_id,
            'source_asset_sha256': a.SOURCE_SHA256, 'split_sha256': a.SPLIT_SHA256,
            'source_loader_sha256': a.V3_LOADER_SHA256, 'source_reader_sha256': a.V3_READER_SHA256,
            'source_reader_review_sha256': a.V3_READER_REVIEW_SHA256,
            'grant_id': token, 'session_id': token, 'claim_id': token, 'plan_id': token,
            'approval_id': token, 'authorization_sha256': decision_hash,
            'plan_sha256': decision_hash, 'approval_sha256': decision_hash,
            'recipe_sha256': _digest({'purpose': request.purpose, 'operation': 'fit10_pose_segment_bundle_export_only'}),
            'fit_role_inventory_sha256': role_hash, 'fit_origin_inventory_sha256': origin_hash,
            'one_use': True, 'binding': self._binding, 'max_pose_bytes': a.MAX_POSE_BYTES,
            'max_chunk_bytes': a.MAX_CHUNK_BYTES, 'max_loader_seconds': a.MAX_LOADER_SECONDS,
            '_expires': expires, 'expires_utc': expires.isoformat(),
        }.items():
            setattr(grant, key, value)
        self._states[id(grant)] = {'grant': grant, 'request': request, 'decision': decision,
                                   'verified': False, 'finalized': False, 'signed': False}
        return grant

    def _state(self, grant, request=None):
        state = self._states.get(id(grant))
        if state is None or state['grant'] is not grant or (request is not None and state['request'] != request):
            raise PermissionError('unknown or mismatched local grant')
        grant.assert_live()
        return state

    def verify_fit10_export_grant(self, grant, request):
        state = self._state(grant, request)
        if state['verified']:
            raise PermissionError('local grant already verified')
        state['verified'] = True
        return {'source_asset_sha256': grant.source_asset_sha256, 'split_sha256': grant.split_sha256,
                'fit_role_inventory_sha256': grant.fit_role_inventory_sha256,
                'fit_origin_inventory_sha256': grant.fit_origin_inventory_sha256,
                'source_loader_sha256': grant.source_loader_sha256,
                'source_authorization_sha256': grant.authorization_sha256}

    def finalize_fit10_source_binding(self, grant, request, receipt):
        state = self._state(grant, request)
        if not state['verified'] or state['finalized']:
            raise PermissionError('local source receipt lifecycle is invalid')
        expected = {'fold_id': grant.fold_id, 'purpose': grant.purpose, 'role_mode': grant.role,
                    'plan_id': grant.plan_id, 'authorization_id': grant.approval_id,
                    'source_hash_verified': True, 'hdf5_denied_reads': 0,
                    'fit_user_count': 10, 'source_loader_sha256': grant.source_loader_sha256,
                    'fit_origin_pool_identity_sha256': grant.fit_origin_inventory_sha256}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('actual read receipt is inconsistent with the export grant')
        state['finalized'] = True
        state['receipt'] = dict(receipt)
        return {'source_asset_sha256': grant.source_asset_sha256, 'split_sha256': grant.split_sha256,
                'fit_role_inventory_sha256': grant.fit_role_inventory_sha256,
                'fit_origin_inventory_sha256': grant.fit_origin_inventory_sha256,
                'source_loader_sha256': grant.source_loader_sha256,
                'source_authorization_sha256': grant.authorization_sha256,
                'source_read_receipt_sha256': _digest(receipt)}

    def completed_source_evidence(self):
        """Return actual completed read aggregates and their local decision records."""
        return tuple({'decision': dict(state['decision']), 'receipt': dict(state['receipt'])}
                     for state in self._states.values() if state.get('signed'))

    def attest_bundle_identity(self, grant, identity_sha256):
        state = self._state(grant)
        if not state['finalized'] or state['signed']:
            raise PermissionError('bundle attestation requires a completed actual source read')
        from ..data.bundle import attestation_signing_bytes
        attestation = {'algorithm': 'Ed25519', 'key_id': self.key_id,
                       'authorization_sha256': grant.authorization_sha256,
                       'issued_utc': datetime.now(timezone.utc).isoformat(),
                       'content_validation': 'finite_unit_wxyz_hemisphere_source_v1'}
        attestation['identity_signature'] = base64.b64encode(
            self._key.sign(attestation_signing_bytes(identity_sha256, attestation))).decode('ascii')
        state['signed'] = True
        return attestation


def create_local_source_root():
    """Validate pinned metadata and create a fresh local source root and key.

    Returns (root, public_key, key_id). The source HDF5 remains unopened until
    export. The caller must store/pin the public key separately from the bundle.
    A new call creates a new in-memory key and cannot sign prior bundles.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from . import source_adapter as adapter
    loader = adapter._load_pinned_v3_loader()
    reader = loader._load_pinned_reader()
    binding = reader.validate_metadata_bindings()
    authority = LocalExportAuthority(binding, loader, adapter, Ed25519PrivateKey.generate())
    return adapter.V3Fit10SourceRoot(authority), authority.public_key, authority.key_id

"""Stdlib-only authority lifecycle test with synthetic metadata and fake signer."""
import sys
import types
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from . import local_authority


class LocalAuthorityTests(unittest.TestCase):
    def test_receipt_and_signing_lifecycle(self):
        serialization = types.ModuleType('cryptography.hazmat.primitives.serialization')
        serialization.Encoding = SimpleNamespace(Raw=1)
        serialization.PublicFormat = SimpleNamespace(Raw=1)
        crypto = types.ModuleType('cryptography')
        hazmat = types.ModuleType('cryptography.hazmat')
        primitives = types.ModuleType('cryptography.hazmat.primitives')
        primitives.serialization = serialization
        modules = {'cryptography': crypto, 'cryptography.hazmat': hazmat,
                   'cryptography.hazmat.primitives': primitives,
                   'cryptography.hazmat.primitives.serialization': serialization}
        class Key:
            def public_key(self): return self
            def public_bytes(self, *args): return b'k' * 32
            def sign(self, value): return b's' * 64
        a = SimpleNamespace(ALLOWED_PURPOSES=('screen',), ROLE='outer_fit10', ALLOWED_FOLDS=(0, 1, 2),
            SOURCE_SHA256='a'*64, SPLIT_SHA256='b'*64, V3_LOADER_SHA256='c'*64,
            V3_READER_SHA256='d'*64, V3_READER_REVIEW_SHA256='e'*64,
            GRANT_KIND='grant', GRANT_NAMESPACE='namespace', MAX_LOADER_SECONDS=60,
            MAX_POSE_BYTES=1024, MAX_CHUNK_BYTES=64,
            fit_role_inventory_digest=lambda rows, fold, role: '1'*64)
        loader = SimpleNamespace(_canonical_fit_rows=lambda *args: (object(),),
                                 _pool_origin_digest=lambda *args: '2'*64)
        request = SimpleNamespace(purpose='screen', role=a.ROLE, fold_id=0,
            source_asset_sha256=a.SOURCE_SHA256, split_sha256=a.SPLIT_SHA256,
            source_loader_sha256=a.V3_LOADER_SHA256, source_reader_sha256=a.V3_READER_SHA256,
            source_reader_review_sha256=a.V3_READER_REVIEW_SHA256, bundle_module_sha256='f'*64)
        with patch.dict(sys.modules, modules):
            authority = local_authority.LocalExportAuthority(object(), loader, a, Key())
        grant = authority.issue_fit10_export_grant(request)
        static = authority.verify_fit10_export_grant(grant, request)
        self.assertNotIn('source_read_receipt_sha256', static)
        with self.assertRaises(PermissionError): authority.attest_bundle_identity(grant, '0'*64)
        receipt = dict(fold_id=0, purpose='screen', role_mode=a.ROLE, plan_id=grant.plan_id,
                       authorization_id=grant.approval_id, source_hash_verified=True,
                       hdf5_denied_reads=0, fit_user_count=10, source_loader_sha256=a.V3_LOADER_SHA256,
                       fit_origin_pool_identity_sha256='2'*64)
        binding = authority.finalize_fit10_source_binding(grant, request, receipt)
        self.assertEqual(binding['source_read_receipt_sha256'], local_authority._digest(receipt))
        attestation = authority.attest_bundle_identity(grant, '0'*64)
        self.assertEqual(attestation['algorithm'], 'Ed25519')
        self.assertEqual(authority.completed_source_evidence()[0]['receipt'], receipt)
        with self.assertRaises(PermissionError): authority.attest_bundle_identity(grant, '0'*64)

if __name__ == '__main__': unittest.main()

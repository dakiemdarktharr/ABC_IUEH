"""Generated manifests and Ed25519 keys; no source recordings are read."""
import base64
import os
from pathlib import Path
import struct
import tempfile
import unittest

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    Ed25519PrivateKey = None

from ..data import bundle
from .fit_identity import verify_fit_identities


@unittest.skipIf(Ed25519PrivateKey is None, 'cryptography is not installed')
class FitIdentityTests(unittest.TestCase):
    def test_three_fold_specific_keys_and_wrong_pin_rejection(self):
        from ..source_adapter_v1.source_adapter import SOURCE_SHA256, SPLIT_SHA256
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); paths={}; keys={}; ids={}; identities={}; dirs=[]
            binding={name:'a'*64 for name in bundle._SOURCE_BINDING_KEYS}
            binding['source_asset_sha256']=SOURCE_SHA256
            binding['split_sha256']=SPLIT_SHA256
            try:
                for fold in range(3):
                    private=Ed25519PrivateKey.generate()
                    keys[fold]=private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                    ids[fold]=f'synthetic-fold-{fold}'
                    def sign(identity, private=private, fold=fold):
                        attestation={'algorithm':'Ed25519','key_id':ids[fold],
                                     'authorization_sha256':binding['source_authorization_sha256'],
                                     'issued_utc':'2026-10-29T00:00:00Z',
                                     'content_validation':'finite_unit_wxyz_hemisphere_source_v1'}
                        attestation['identity_signature']=base64.b64encode(private.sign(
                            bundle.attestation_signing_bytes(identity, attestation))).decode('ascii')
                        return attestation
                    directory=root/f'fold-{fold}'; dirs.append(directory)
                    identities[fold]=bundle._create_bundle(directory,
                        [bundle.SegmentInput(0,420,struct.pack('<4f',1,0,0,0)*420)],
                        fold_id=fold, purpose=bundle.SCREEN_PURPOSE, source_binding=binding,
                        attest_identity=sign)
                    paths[fold]=directory/bundle.MANIFEST_FILE
                self.assertEqual(verify_fit_identities(paths,public_keys=keys,key_ids=ids),identities)
                with self.assertRaises(InvalidSignature):
                    verify_fit_identities(paths,public_keys={**keys,1:keys[0]},key_ids=ids)
                with self.assertRaises(ValueError):
                    verify_fit_identities(paths,public_keys=keys,key_ids={**ids,1:ids[0]})
                with self.assertRaises(ValueError):
                    verify_fit_identities(paths,public_keys={0:keys[0]},key_ids=ids)
            finally:
                for directory in dirs:
                    if directory.exists():
                        os.chmod(directory,0o700)
                        for item in directory.iterdir(): os.chmod(item,0o600)


if __name__ == '__main__': unittest.main()

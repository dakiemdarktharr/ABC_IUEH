"""Synthetic signature, mismatch and canonical-byte checks; no recordings."""
import base64
import hashlib
import unittest
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    Ed25519PrivateKey=None
from . import signed_plan as signed


@unittest.skipIf(Ed25519PrivateKey is None, 'cryptography is not installed')
class SignedPlanTests(unittest.TestCase):
    def test_signed_plan_requires_all_fold_and_checkpoint_pins(self):
        private=Ed25519PrivateKey.generate()
        pub=private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
        plan={'kind':'abc_three_arm_validation_plan_v1','purpose':'ordered_motion_three_arm_screen',
              'role':'outer_validation5','protocol_sha256':'a'*64,'source_sha256':'b'*64,'split_sha256':'c'*64,
              'fold_origin_sha256':{str(f):'d'*64 for f in range(3)},
              'fold_origin_stream_sha256':{str(f):'e'*64 for f in range(3)},
              'fit_identity_sha256':{str(f):'f'*64 for f in range(3)},
              'checkpoint_sha256':{cell:'1'*64 for cell in signed.CELLS},
              'code_sha256':{key:'2'*64 for key in signed.CODE_KEYS}}
        sig=base64.b64encode(private.sign(signed.canonical(plan))).decode('ascii')
        raw=signed.canonical({'plan':plan,'signature':sig})+b'\n'; digest=hashlib.sha256(raw).hexdigest()
        self.assertEqual(signed.verify_signed_plan(raw,public_key=pub,expected_plan_sha256=digest,observed=plan),plan)
        altered={**plan,'source_sha256':'3'*64}
        with self.assertRaises(ValueError): signed.verify_signed_plan(raw,public_key=pub,expected_plan_sha256=digest,observed=altered)
        wrong=Ed25519PrivateKey.generate().public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
        with self.assertRaises(ValueError): signed.verify_signed_plan(raw,public_key=wrong,expected_plan_sha256=digest,observed=plan)
        with self.assertRaises(ValueError): signed.verify_signed_plan(raw+b' ',public_key=pub,expected_plan_sha256=digest,observed=plan)
        broken={**plan,'checkpoint_sha256':dict(plan['checkpoint_sha256'])}; broken['checkpoint_sha256'].pop(signed.CELLS[0])
        signature=base64.b64encode(private.sign(signed.canonical(broken))).decode('ascii')
        bad=signed.canonical({'plan':broken,'signature':signature})+b'\n'
        with self.assertRaises(ValueError): signed.verify_signed_plan(bad,public_key=pub,expected_plan_sha256=hashlib.sha256(bad).hexdigest(),observed=broken)


if __name__ == '__main__': unittest.main()

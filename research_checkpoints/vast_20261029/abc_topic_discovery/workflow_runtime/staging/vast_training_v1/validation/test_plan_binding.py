"""Synthetic measured-file binding test. No human source data is accessed."""
import base64
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
except ImportError:
    Ed25519PrivateKey=None
from . import plan_binding, signed_plan
from .source import _origin_stream_digest


@unittest.skipIf(Ed25519PrivateKey is None,'cryptography unavailable')
class PlanBindingTests(unittest.TestCase):
    def test_actual_file_hashes_and_signed_fold_bindings(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp)
            def file(name, data):
                path=root/name; path.write_bytes(data); return path
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            source_file=file('synthetic-source',b'generated only')
            split=file('synthetic-split',b'split'); protocol=file('protocol',b'protocol')
            code={key:file('code-'+key,key.encode()) for key in signed_plan.CODE_KEYS}
            checkpoints={key:file(f'checkpoint-{i}',key.encode()) for i,key in enumerate(signed_plan.CELLS)}
            class FakeSource:
                role='outer_validation5'
                def __init__(self, fold):
                    self.fold_id=fold; self._users=['u']
                    self._rows=[SimpleNamespace(user_id='u',offset_int64=fold*3,count=2)]
                    self._binding=object(); self._loader=SimpleNamespace(_pool_origin_digest=lambda rows,binding,f,r: str(f+1)*64)
                    self.origin_identity_sha256=str(fold+1)*64
                    self.origin_stream_sha256=_origin_stream_digest(self._rows,self._users)
                    self.source_asset_sha256=digest(source_file); self.split_sha256=digest(split)
            sources={fold:FakeSource(fold) for fold in range(3)}
            identities={0:'a'*64,1:'b'*64,2:'c'*64}
            plan={'kind':'abc_three_arm_validation_plan_v1','purpose':'ordered_motion_three_arm_screen',
                  'role':'outer_validation5','protocol_sha256':digest(protocol),'source_sha256':digest(source_file),
                  'split_sha256':digest(split),'fold_origin_sha256':{str(i):sources[i].origin_identity_sha256 for i in range(3)},
                  'fold_origin_stream_sha256':{str(i):sources[i].origin_stream_sha256 for i in range(3)},
                  'fit_identity_sha256':{str(i):v for i,v in identities.items()},
                  'checkpoint_sha256':{key:digest(path) for key,path in checkpoints.items()},
                  'code_sha256':{key:digest(path) for key,path in code.items()}}
            private=Ed25519PrivateKey.generate(); pub=private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
            sig=base64.b64encode(private.sign(signed_plan.canonical(plan))).decode('ascii')
            raw=signed_plan.canonical({'plan':plan,'signature':sig})+b'\n'
            kwargs=dict(public_key=pub,plan_sha256=hashlib.sha256(raw).hexdigest(),source_file=source_file,
                split_file=split,protocol_file=protocol,code_files=code,checkpoint_files=checkpoints,
                fit_manifests={},fit_public_keys={},fit_key_ids={},fold_sources=sources)
            with patch.object(plan_binding,'FixedValidationSegments',FakeSource), \
                 patch.object(plan_binding,'verify_fit_identities',return_value=identities):
                self.assertEqual(plan_binding.verify_actual_inputs(raw,**kwargs),plan)
                bad=checkpoints[signed_plan.CELLS[0]]; bad.write_bytes(b'changed')
                with self.assertRaises(ValueError): plan_binding.verify_actual_inputs(raw,**kwargs)
                bad.write_bytes(signed_plan.CELLS[0].encode())
                code['metric'].write_bytes(b'changed code')
                with self.assertRaises(ValueError): plan_binding.verify_actual_inputs(raw,**kwargs)


if __name__ == '__main__': unittest.main()

"""Synthetic backup transport only; no rclone/network/source recordings."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .. import remote_backup as rb
from .. import checkpoint_store as store


class FakeTransport:
    def __init__(self): self.files={}; self.events=[]; self.corrupt=None; self.fail=None
    def put_if_absent(self, path, remote):
        self.events.append(('put',remote))
        if self.fail == remote: raise OSError('synthetic network failure')
        self.files.setdefault(remote,Path(path).read_bytes())
    def download_to(self, remote, destination):
        self.events.append(('download',remote))
        if self.fail == remote: raise OSError('synthetic network failure')
        Path(destination).write_bytes(self.files[remote])
    def read_digest(self, remote):
        self.events.append(('read',remote))
        data=self.files[remote]
        if self.corrupt == remote: data+=b'corrupt'
        return hashlib.sha256(data).hexdigest(),len(data)


class RemoteBackupTests(unittest.TestCase):
    def inputs(self, root):
        payload=b'synthetic checkpoint bytes'; digest=hashlib.sha256(payload).hexdigest()
        path=root/f'update-100-{digest}.pt'; path.write_bytes(payload)
        metadata=path.with_name(path.stem+'.meta.json')
        values={'schema':'abc-training-checkpoint-v1','run_id':'fake','plan_sha256':'a'*64,
                'data_sha256':'b'*64,'code_sha256':'c'*64,'model_sha256':'d'*64,
                'fold_id':0,'training_seed':1,'arm':'fixture','global_step':100,'total_steps':5000,
                'payload_sha256':digest,'payload_size_bytes':len(payload),'resume_state_complete':True,
                'state_components':{key:True for key in store.STATE_KEYS}}
        metadata.write_text(json.dumps(values),encoding='utf-8')
        return path,metadata,digest

    def test_verified_manifest_last_and_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,meta,digest=self.inputs(Path(tmp)); transport=FakeTransport()
            result=rb.backup_checkpoint(path,meta,'drive:private/run',transport=transport)
            self.assertEqual(result['payload_sha256'],digest)
            self.assertEqual([x[0] for x in transport.events],['put','read']*3)
            self.assertEqual(transport.events[-2][1],result['manifest_remote'])
            manifest=json.loads(transport.files[result['manifest_remote']])
            self.assertEqual(manifest['payload_sha256'],digest)
            self.assertEqual(rb.backup_checkpoint(path,meta,'drive:private/run',transport=transport),result)

    def test_collision_or_bad_read_prevents_manifest(self):
        for collision in ('payload','metadata','readback','network'):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as tmp:
                path,meta,digest=self.inputs(Path(tmp)); transport=FakeTransport(); prefix='drive:private/run'
                payload=f'{prefix}/checkpoint-{digest}.pt'; metadata=f'{prefix}/checkpoint-{digest}.meta.json'
                if collision == 'payload': transport.files[payload]=b'different'
                if collision == 'metadata': transport.files[metadata]=b'different'
                if collision == 'readback': transport.corrupt=metadata
                if collision == 'network': transport.fail=metadata
                with self.assertRaises((rb.RemoteBackupError,OSError)):
                    rb.backup_checkpoint(path,meta,prefix,transport=transport)
                self.assertNotIn(f'{prefix}/checkpoint-{digest}.manifest.json',transport.files)
                if collision == 'payload': self.assertEqual(transport.files[payload],b'different')
                if collision == 'metadata': self.assertEqual(transport.files[metadata],b'different')

    def test_conflicting_manifest_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,meta,digest=self.inputs(Path(tmp)); transport=FakeTransport(); prefix='drive:private/run'
            target=f'{prefix}/checkpoint-{digest}.manifest.json'; transport.files[target]=b'bad'
            with self.assertRaises(rb.RemoteBackupError): rb.backup_checkpoint(path,meta,prefix,transport=transport)
            self.assertEqual(transport.files[target],b'bad')

    def test_invalid_scope_or_metadata_never_contacts_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,meta,_=self.inputs(Path(tmp)); transport=FakeTransport()
            for remote in ('local/path','drive:../escape','drive:/absolute','drive:private//other'):
                with self.assertRaises(rb.RemoteBackupError): rb.backup_checkpoint(path,meta,remote,transport=transport)
            meta.write_text('{}',encoding='utf-8')
            with self.assertRaises(store.CheckpointError): rb.backup_checkpoint(path,meta,'drive:private/run',transport=transport)
            self.assertEqual(transport.events,[])


    def test_restore_and_remote_reverification_with_independent_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,meta,digest=self.inputs(root); transport=FakeTransport(); prefix='drive:private/run'
            info=rb.backup_checkpoint(path,meta,prefix,transport=transport)
            source_meta=json.loads(meta.read_text())
            expected={key:source_meta[key] for key in ('run_id','plan_sha256','data_sha256','code_sha256',
                      'model_sha256','fold_id','training_seed','arm')}
            expected['expected_global_step']=100
            pins=root/'expected.json'; pins.write_text(json.dumps(expected))
            result=rb.restore_checkpoint(prefix,root/'restore',manifest_sha256=info['manifest_sha256'],
                                         payload_sha256=digest,expected_path=pins,transport=transport)
            self.assertEqual(Path(result['resume_path']).read_bytes(),path.read_bytes())
            rb.verify_remote_copy(prefix,Path(result['manifest_path']),info['manifest_sha256'],transport=transport)
            bad_expected={**expected,'expected_global_step':101}
            bad_pins=root/'bad-expected.json'; bad_pins.write_text(json.dumps(bad_expected))
            with self.assertRaises(store.CheckpointError):
                rb.restore_checkpoint(prefix,root/'wrong-step',manifest_sha256=info['manifest_sha256'],
                                      payload_sha256=digest,expected_path=bad_pins,transport=transport)
            self.assertFalse((root/'wrong-step'/Path(info['manifest_remote']).name).exists())
            transport.files[info['manifest_remote']]=b'changed'
            with self.assertRaises(rb.RemoteBackupError):
                rb.verify_remote_copy(prefix,Path(result['manifest_path']),info['manifest_sha256'],transport=transport)
            with self.assertRaises(rb.RemoteBackupError):
                rb.restore_checkpoint(prefix,root/'bad',manifest_sha256=info['manifest_sha256'],
                                      payload_sha256=digest,expected_path=pins,transport=transport)
            self.assertFalse((root/'bad'/Path(info['manifest_remote']).name).exists())


    def test_rclone_errors_are_redacted(self):
        import subprocess
        error=subprocess.CalledProcessError(1,['rclone','token-secret'],stderr=b'token-secret')
        with patch.object(rb.subprocess,'run',side_effect=error):
            with self.assertRaises(rb.RemoteBackupError) as caught:
                rb.RcloneTransport().check('drive:private/run')
        self.assertNotIn('token-secret',str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_rclone_copy_flags_and_download_hash(self):
        calls=[]
        def run(args,**kwargs):
            calls.append(args)
            if args[1]=='cat': kwargs['stdout'].write(b'downloaded')
        with patch.object(rb.subprocess,'run',side_effect=run):
            transport=rb.RcloneTransport('fake-rclone',42)
            transport.put_if_absent(Path('synthetic'), 'drive:private/item')
            digest=transport.read_digest('drive:private/item')
        self.assertEqual(calls[0],['fake-rclone','copyto','synthetic','drive:private/item','--ignore-existing','--immutable'])
        self.assertEqual(digest,(hashlib.sha256(b'downloaded').hexdigest(),10))


if __name__ == '__main__': unittest.main()

"""Synthetic journal tests with in-memory transport; no network."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import bootstrap, ledger, remote_journal as journal
from .test_guard import HISTORY, rows

class MemoryTransport:
    def __init__(self): self.files = {}
    def put_if_absent(self, local, remote): self.files.setdefault(remote, Path(local).read_bytes())
    def read_digest(self, remote):
        raw = self.files[remote]; return hashlib.sha256(raw).hexdigest(), len(raw)
    def download_to(self, remote, target): Path(target).write_bytes(self.files[remote])
    def list_files(self, prefix): return [x[len(prefix)+1:] for x in self.files if x.startswith(prefix+'/')]

class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); raw, head=rows(HISTORY)
        self.orig=self.root/'original'; self.orig.write_bytes(raw)
        for key,value in [('HISTORY_SHA256',hashlib.sha256(raw).hexdigest()),('HISTORY_HEAD',head)]:
            p=patch.object(ledger,key,value); p.start(); self.addCleanup(p.stop)
        self.led=self.root/'ledger'; snap=ledger.initialize_ledger(self.led,self.orig)
        ext=self.root/'anchor-dir'; ext.mkdir(); self.anchor=ext/'anchor'
        bootstrap.initialize_anchor(self.anchor,snap)
        self.pin=self.root/'head'; journal.initialize_head(self.pin)
        self.empty_head=self.pin.read_bytes()
        self.transport=MemoryTransport()
    def reserve(self, ident): return bootstrap.AnchoredReservation(self.led,self.orig,self.anchor,ident*16,30)
    def test_two_publishes_and_recovery_latest(self):
        a=self.reserve('a'); first=journal.publish(a,self.anchor,self.pin,'fake:journal',transport=self.transport); a.close()
        b=self.reserve('b'); second=journal.publish(b,self.anchor,self.pin,'fake:journal',transport=self.transport); b.close()
        self.assertNotEqual(first,second)
        recovered=self.root/'new'; recovered.mkdir(); anchor_dir=recovered/'ext'; anchor_dir.mkdir()
        with self.assertRaises(ValueError):
            journal.recover(remote_prefix='fake:journal',expected_head=first,original_path=self.orig,
                ledger_path=recovered/'ledger',anchor_path=anchor_dir/'anchor',head_path=recovered/'head',transport=self.transport)
        journal.recover(remote_prefix='fake:journal',expected_head=second,original_path=self.orig,
            ledger_path=recovered/'ledger',anchor_path=anchor_dir/'anchor',head_path=recovered/'head',transport=self.transport)
        self.assertEqual((recovered/'ledger').read_bytes(),self.led.read_bytes())
        self.assertEqual((anchor_dir/'anchor').read_bytes(),self.anchor.read_bytes())
        with self.assertRaises(FileExistsError):
            journal.recover(remote_prefix='fake:journal',expected_head=second,original_path=self.orig,
                ledger_path=recovered/'ledger',anchor_path=anchor_dir/'anchor',head_path=recovered/'head',transport=self.transport)
    def test_upload_failure_and_wrong_local_head_stop(self):
        a=self.reserve('a')
        with patch.object(self.transport,'read_digest',return_value=('0'*64,0)):
            with self.assertRaises(ValueError): journal.publish(a,self.anchor,self.pin,'fake:journal',transport=self.transport)
        self.assertIsNone(journal._head(self.pin)); a.close()
    def test_remote_tamper_fails(self):
        a=self.reserve('a'); journal.publish(a,self.anchor,self.pin,'fake:journal',transport=self.transport); a.close()
        key=next(k for k in self.transport.files if '/ledger-' in k); self.transport.files[key]=b'tampered'
        b=self.reserve('b')
        with self.assertRaises(ValueError): journal.publish(b,self.anchor,self.pin,'fake:journal',transport=self.transport)
        b.close()
    def test_stale_head_cannot_publish(self):
        a=self.reserve('a'); journal.publish(a,self.anchor,self.pin,'fake:journal',transport=self.transport); a.close()
        self.pin.write_bytes(self.empty_head)
        b=self.reserve('b')
        with self.assertRaises(ValueError): journal.publish(b,self.anchor,self.pin,'fake:journal',transport=self.transport)
        b.close()

    def test_crash_after_upload_blocks_future_stale_head(self):
        a=self.reserve('a')
        with patch.object(journal,'_advance_head',side_effect=OSError('synthetic crash')):
            with self.assertRaises(OSError): journal.publish(a,self.anchor,self.pin,'fake:journal',transport=self.transport)
        a.close(); self.assertIsNone(journal._head(self.pin))
        b=self.reserve('b')
        with self.assertRaises(ValueError): journal.publish(b,self.anchor,self.pin,'fake:journal',transport=self.transport)
        b.close()

    def test_invalid_manifest_anchor_is_rejected(self):
        import json
        a=self.reserve('a'); first=journal.publish(a,self.anchor,self.pin,'fake:journal',transport=self.transport); a.close()
        b=self.reserve('b'); second=journal.publish(b,self.anchor,self.pin,'fake:journal',transport=self.transport); b.close()
        row=json.loads(self.transport.files[f'fake:journal/journal-{second}.json'])
        row['anchor_head']='f'*64
        raw=journal._canonical(row)+b'\n'; digest=hashlib.sha256(raw).hexdigest()
        self.transport.files[f'fake:journal/journal-{digest}.json']=raw
        with self.assertRaises(ValueError): journal._all_generations(self.transport,'fake:journal',self.orig.read_bytes())

if __name__ == '__main__': unittest.main()

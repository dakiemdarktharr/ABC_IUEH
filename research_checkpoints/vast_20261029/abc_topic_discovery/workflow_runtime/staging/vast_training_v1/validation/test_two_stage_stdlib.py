"""Pure-stdlib, generated two-stage fixture; no human source, NumPy or Torch."""
import math
import hashlib
import json
import struct
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import two_stage as stage
from .source import _origin_stream_digest
from types import SimpleNamespace

class FakeSource:
    fold_id=0; role='outer_validation5'; origin_identity_sha256='a'*64
    expected_origin_counts=(3,3,3,3,3)
    source_asset_sha256='c'*64; split_sha256='d'*64
    def __init__(self, path):
        self.path=path; self.target_requested=False
        digest=hashlib.sha256()
        for u in range(5):
            for i in range(3): digest.update(stage._ORIGIN.pack(u,u*10+i))
        self.origin_stream_sha256=digest.hexdigest()
    def iter_contexts(self):
        for u in range(5):
            for i in range(3): yield u,u*10+i,bytes(400*4*4)
    def iter_targets(self):
        if not self.path.exists() or self.path.stat().st_size != 15*stage._RECORD.size:
            raise AssertionError('targets requested before predictions were persisted')
        self.target_requested=True
        for u in range(5):
            for i in range(3):
                q=(1.,0.,0.,0.) if i!=1 else (math.sqrt(.5),math.sqrt(.5),0.,0.)
                yield u,u*10+i,q

class TwoStageTests(unittest.TestCase):
    def test_metadata_origin_digest_matches_fixture_order(self):
        rows=[SimpleNamespace(user_id=u, offset_int64=u*10, count=3) for u in range(5)]
        source=FakeSource(Path('unused'))
        self.assertEqual(_origin_stream_digest(rows, tuple(range(5))), source.origin_stream_sha256)
        rows[1].offset_int64 += 1
        self.assertNotEqual(_origin_stream_digest(rows, tuple(range(5))), source.origin_stream_sha256)

    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_freeze_then_exact_user_and_pooled_metrics(self, _memory):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; output=root/'result.json'; source=FakeSource(pred); lock_path=root/'prediction-lock.json'
            pins={cell:'a'*64 for cell in stage.CELLS}
            predict=lambda context:{cell:(1.,0.,0.,0.) for cell in stage.CELLS}
            lock=stage.freeze_predictions(source=source,predict=predict,prediction_path=pred,lock_path=lock_path,checkpoint_hashes=pins,predictor_code_sha256='e'*64,ram_guard=lambda:None)
            self.assertFalse(source.target_requested)
            try:
                result=stage.score_frozen_predictions(source=source,prediction_path=pred,lock_path=lock_path,expected_lock_sha256=lock,output_path=output,scratch_parent=root,ram_guard=lambda:None)
                self.assertTrue(source.target_requested)
                summary=result['users'][0]['predictors'][stage.CELLS[0]]
                self.assertAlmostEqual(summary['mean_degrees'],30,places=5)
                self.assertAlmostEqual(summary['p95_degrees'],81,places=5)
                self.assertAlmostEqual(summary['p99_degrees'],88.2,places=5)
                self.assertEqual(result['pooled_origins'][stage.CELLS[0]]['origin_count'],15)
                self.assertEqual(result['users'][0]['anonymous_user_index'],0)
            finally: os.chmod(pred,0o600); os.chmod(lock_path,0o600)

    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_tampered_predictions_rejected_before_target_read(self, _memory):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; source=FakeSource(pred); lock_path=root/'prediction-lock.json'
            lock=stage.freeze_predictions(source=source,predict=lambda _: {cell:(1.,0.,0.,0.) for cell in stage.CELLS},prediction_path=pred,lock_path=lock_path,checkpoint_hashes={cell:'b'*64 for cell in stage.CELLS},predictor_code_sha256='e'*64,ram_guard=lambda:None)
            os.chmod(pred,0o600)
            with pred.open('r+b') as f: f.seek(2); f.write(b'x')
            with self.assertRaises(ValueError):
                stage.score_frozen_predictions(source=source,prediction_path=pred,lock_path=lock_path,expected_lock_sha256=lock,output_path=root/'out.json',scratch_parent=root,ram_guard=lambda:None)
            self.assertFalse(source.target_requested)
            os.chmod(lock_path,0o600)


    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_invalid_locked_origin_order_rejected_before_targets(self, _memory):
        # Model an independently retained digest for structurally invalid bytes.
        # Cryptographic integrity alone must not make malformed records acceptable.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; lock_path=root/'lock.json'; source=FakeSource(pred)
            stage.freeze_predictions(source=source,predict=lambda _: {cell:(1.,0.,0.,0.) for cell in stage.CELLS},prediction_path=pred,lock_path=lock_path,checkpoint_hashes={cell:'b'*64 for cell in stage.CELLS},predictor_code_sha256='e'*64,ram_guard=lambda:None)
            try:
                os.chmod(pred,0o600); os.chmod(lock_path,0o600)
                data=bytearray(pred.read_bytes())
                struct.pack_into('<Q', data, stage._RECORD.size+1, 0)
                pred.write_bytes(data)
                lock=json.loads(lock_path.read_bytes())
                lock['predictions_sha256']=hashlib.sha256(data).hexdigest()
                raw=stage._canonical(lock)+b'\n'; lock_path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'origin order'):
                    stage.score_frozen_predictions(source=source,prediction_path=pred,lock_path=lock_path,expected_lock_sha256=hashlib.sha256(raw).hexdigest(),output_path=root/'out.json',scratch_parent=root,ram_guard=lambda:None)
                self.assertFalse(source.target_requested)
            finally:
                os.chmod(pred,0o600); os.chmod(lock_path,0o600)


    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_wrong_context_origins_cannot_create_lock(self, _memory):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; lock_path=root/'lock.json'; source=FakeSource(pred)
            source.origin_stream_sha256='f'*64
            with self.assertRaisesRegex(ValueError, 'origin stream'):
                stage.freeze_predictions(source=source,predict=lambda _: {cell:(1.,0.,0.,0.) for cell in stage.CELLS},prediction_path=pred,lock_path=lock_path,checkpoint_hashes={cell:'b'*64 for cell in stage.CELLS},predictor_code_sha256='e'*64,ram_guard=lambda:None)
            self.assertFalse(lock_path.exists())
            self.assertFalse(source.target_requested)


    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_prediction_file_mutation_before_targets_cannot_change_locked_snapshot(self, _memory):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; lock_path=root/'lock.json'
            class MutatingSource(FakeSource):
                def iter_targets(self):
                    os.chmod(self.path,0o600)
                    with self.path.open('r+b') as output:
                        output.seek(9); output.write(struct.pack('<4f',0.,1.,0.,0.))
                    yield from super().iter_targets()
            source=MutatingSource(pred)
            lock=stage.freeze_predictions(source=source,predict=lambda _: {cell:(1.,0.,0.,0.) for cell in stage.CELLS},prediction_path=pred,lock_path=lock_path,checkpoint_hashes={cell:'b'*64 for cell in stage.CELLS},predictor_code_sha256='e'*64,ram_guard=lambda:None)
            try:
                result=stage.score_frozen_predictions(source=source,prediction_path=pred,lock_path=lock_path,expected_lock_sha256=lock,output_path=root/'out.json',scratch_parent=root,ram_guard=lambda:None)
                self.assertTrue(source.target_requested)
                self.assertAlmostEqual(result['users'][0]['predictors'][stage.CELLS[0]]['mean_degrees'],30,places=5)
            finally:
                os.chmod(pred,0o600); os.chmod(lock_path,0o600)

    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_snapshot_memory_reserve_checked_before_target(self, memory):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; lock_path=root/'lock.json'; source=FakeSource(pred)
            lock=stage.freeze_predictions(source=source,predict=lambda _: {cell:(1.,0.,0.,0.) for cell in stage.CELLS},prediction_path=pred,lock_path=lock_path,checkpoint_hashes={cell:'b'*64 for cell in stage.CELLS},predictor_code_sha256='e'*64,ram_guard=lambda:None)
            try:
                memory.return_value=2*1024**3
                with self.assertRaises(MemoryError):
                    stage.score_frozen_predictions(source=source,prediction_path=pred,lock_path=lock_path,expected_lock_sha256=lock,output_path=root/'out.json',scratch_parent=root,ram_guard=lambda:None)
                self.assertFalse(source.target_requested)
            finally:
                os.chmod(pred,0o600); os.chmod(lock_path,0o600)

    @patch.object(stage, '_available_ram_bytes', return_value=1024**3)
    def test_low_memory_fails_before_context_and_output(self, _memory):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'predictions.bin'; source=FakeSource(pred)
            with self.assertRaises(MemoryError):
                stage.freeze_predictions(source=source,predict=lambda _: {},prediction_path=pred,lock_path=root/'lock.json',checkpoint_hashes={cell:'b'*64 for cell in stage.CELLS},predictor_code_sha256='e'*64,ram_guard=lambda:None)
            self.assertFalse(pred.exists())
            self.assertFalse(source.target_requested)

if __name__ == '__main__': unittest.main()

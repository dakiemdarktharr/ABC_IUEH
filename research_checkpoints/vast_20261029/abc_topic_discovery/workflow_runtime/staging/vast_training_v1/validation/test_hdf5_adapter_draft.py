"""Generated fake HDF5 hierarchy for selected open-file adapter; no files read."""
import itertools
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest
from . import hdf5_adapter_draft as adapter
from . import two_stage as stage
from .source import _origin_stream_digest

try:
    import numpy as np
except ImportError:
    np=None


class Group(dict):
    def __init__(self, name, file, **children): super().__init__(children); self.name=name; self.file=file


@unittest.skipIf(np is None,'NumPy unavailable')
class AdapterTests(unittest.TestCase):
    def make(self):
        file=SimpleNamespace(id=SimpleNamespace(id=7)); users_group=Group('/users',file); file.name='/'; file.file=file
        file.users=users_group
        users=[f'u{i}' for i in range(5)]; rows=[]; expected={}; session_windows={}; arrays={}
        for i,user in enumerate(users):
            path=f'/users/{user}/sessions/s/segments/g'; q=np.zeros((1000,4),np.float32); q[:,0]=1
            quat=SimpleNamespace(name=path+'/quaternion',file=file,shape=q.shape,data=q)
            ts=SimpleNamespace(name=path+'/timestamps',file=file,shape=(1000,))
            seg=Group(path,file,timestamps=ts,quaternion=quat)
            segs=Group(f'/users/{user}/sessions/s/segments',file,g=seg)
            sess=Group(f'/users/{user}/sessions/s',file,segments=segs)
            sessions=Group(f'/users/{user}/sessions',file,s=sess)
            users_group[user]=Group(f'/users/{user}',file,sessions=sessions)
            rows.append(SimpleNamespace(user_id=user,session_id='s',segment_id='g',fold_id=0,offset_int64=i*581,count=581))
            expected[user]={'s':{'g'}}; session_windows[(user,'s')]=581; arrays[user]=q
        reads=[]
        class Loader:
            ReadAudit=staticmethod(lambda *args,**kwargs:SimpleNamespace(denied_reads=0))
            _DatasetAudit=staticmethod(lambda *args:SimpleNamespace())
            @staticmethod
            def _validate_source_root_with_legacy_bridge(*a,**kw): pass
            @staticmethod
            def _get_selected_child(parent,name,**kw): return getattr(parent,name) if parent is file else parent[name]
            @staticmethod
            def _read_dataset_rows(dataset,*,start,stop,audit): reads.append((dataset.name,start,stop)); return dataset.data[start:stop]
        class Reader:
            @staticmethod
            def _register_unique(obj,path,addresses):
                if id(obj) in addresses: raise ValueError('alias')
                addresses[id(obj)]=path
            @staticmethod
            def _dataset_metadata(obj,*,dtype,ndim): return obj.shape
        binding=SimpleNamespace(ownership={0:{'outer_validation':users,'outer_fit':['fit'+str(i) for i in range(10)]}},
            parent=SimpleNamespace(expected_train=SimpleNamespace(user_sessions=expected,session_windows=session_windows)))
        source=SimpleNamespace(_users=users,_rows=rows,_binding=binding,_loader=Loader(),_reader=Reader(),
                               fold_id=0,expected_origin_counts=(581,)*5)
        return source,file,reads

    def test_context_and_endpoint_slice_origins(self):
        source,file,reads=self.make(); guards=[]
        first=list(itertools.islice(adapter.iter_selected_open_file(source,file,phase='contexts',
                    check_guard=lambda:guards.append(1),deadline=999999999),2))
        self.assertEqual([(a,b) for _,a,b in reads],[(0,400),(1,401)])
        self.assertEqual([(x[0],x[1]) for x in first],[(0,0),(0,1)])
        reads.clear()
        target=list(itertools.islice(adapter.iter_selected_open_file(source,file,phase='endpoints',
                    check_guard=lambda:guards.append(1),deadline=999999999),2))
        self.assertEqual([(a,b) for _,a,b in reads],[(419,420),(420,421)])
        self.assertEqual([(x[0],x[1]) for x in target],[(0,0),(0,1)])
        self.assertTrue(guards)


    @patch.object(stage, '_available_ram_bytes', return_value=4*1024**3)
    def test_generated_adapter_freezes_before_targets_and_scores(self, _memory):
        base,file,reads=self.make()
        class Source:
            role='outer_validation5'; source_asset_sha256='a'*64; split_sha256='b'*64
            origin_identity_sha256='c'*64
            def __init__(self,lock):
                self.lock=lock; self.targets_requested=False
                self.origin_stream_sha256=_origin_stream_digest(base._rows,base._users)
            def __getattr__(self,name): return getattr(base,name)
            def iter_contexts(self):
                return adapter.iter_selected_open_file(self,file,phase='contexts',check_guard=lambda:None,deadline=999999999)
            def iter_targets(self):
                if not self.lock.exists(): raise AssertionError('no prediction lock')
                self.targets_requested=True
                return adapter.iter_selected_open_file(self,file,phase='endpoints',check_guard=lambda:None,deadline=999999999)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); pred=root/'pred'; lock_path=root/'lock'; source=Source(lock_path)
            lock=stage.freeze_predictions(source=source,predict=lambda _:{cell:(1.,0.,0.,0.) for cell in stage.CELLS},
                prediction_path=pred,lock_path=lock_path,checkpoint_hashes={cell:'d'*64 for cell in stage.CELLS},
                predictor_code_sha256='e'*64,ram_guard=lambda:None)
            self.assertFalse(source.targets_requested)
            result=stage.score_frozen_predictions(source=source,prediction_path=pred,lock_path=lock_path,
                expected_lock_sha256=lock,output_path=root/'out',scratch_parent=root,ram_guard=lambda:None)
            self.assertTrue(source.targets_requested)
            self.assertEqual(result['pooled_origins'][stage.CELLS[0]]['origin_count'],5*581)
            self.assertEqual(result['pooled_origins'][stage.CELLS[0]]['mean_degrees'],0)
            os.chmod(pred,0o600); os.chmod(lock_path,0o600)

    def test_scope_failure_precedes_data_slices(self):
        source,file,reads=self.make(); source._binding.ownership[0]['outer_fit'][0]=source._users[0]
        with self.assertRaises(PermissionError):
            next(adapter.iter_selected_open_file(source,file,phase='contexts',check_guard=lambda:None,deadline=0))
        self.assertEqual(reads,[])


if __name__ == '__main__': unittest.main()

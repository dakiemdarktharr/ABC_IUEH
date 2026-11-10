"""Synthetic-only checks for split access, sampling, metric and recovery guards."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from abc_iueh import boxrr_v2_experiment as e
from abc_iueh.boxrr_experiment import MemoryUser, RamCorpus


def corpus(lengths=(500,700)):
    users=[]
    for i,n in enumerate(lengths):
        q=np.zeros((n,4),dtype=np.float32);q[:,0]=1
        users.append(MemoryUser(f'synthetic-{i}',[q],np.array([n-419],dtype=np.int64)))
    return RamCorpus(users)


def manifest_fixture(tmp_path):
    p=tmp_path/'fit.bin';p.write_bytes(b'synthetic-placeholder')
    row={'user':'synthetic-fit','partition':'fit','path':p.name,'bytes':p.stat().st_size,
         'sha256':e.file_sha256(p),'eligible_windows':81}
    return {'source':'BOXRR-23','training_allowed':True,'source_revision':e.REVISION,
            'config':{'context_samples':400,'horizon_samples':20,'target_rate_hz':100},
            'shards':[row,{**row,'user':'synthetic-final','partition':'final','path':'does-not-exist.h5'}]}


def test_training_verification_never_reads_final_files(tmp_path):
    m=manifest_fixture(tmp_path)
    assert e.require_corpus(tmp_path,m,{'fit'})==['synthetic-fit']
    with pytest.raises(ValueError,match='locked'):
        e.require_corpus(tmp_path,m,{'final'})


def test_reject_nonofficial_changed_timebase_duplicate_and_traversal(tmp_path):
    m=manifest_fixture(tmp_path);m['source']='synthetic'
    with pytest.raises(ValueError,match='official'):e.require_corpus(tmp_path,m,{'fit'})
    m=manifest_fixture(tmp_path);m['config']['horizon_samples']=10
    with pytest.raises(ValueError,match='interface'):e.require_corpus(tmp_path,m,{'fit'})
    m=manifest_fixture(tmp_path);m['shards'][1]['user']=m['shards'][0]['user']
    with pytest.raises(ValueError,match='Duplicate'):e.require_corpus(tmp_path,m,{'fit'})
    m=manifest_fixture(tmp_path);m['shards'][0]['path']='../outside.bin'
    with pytest.raises(ValueError,match='unsafe'):e.require_corpus(tmp_path,m,{'fit'})


def test_sampler_uses_declared_mixture_and_replays():
    c=corpus();sampler=e.MixedSampler(c)
    counts=np.array([81,281]);np.testing.assert_allclose(sampler.probability,.25+.5*counts/counts.sum())
    a=sampler.batch(np.random.default_rng(31),17);b=sampler.batch(np.random.default_rng(31),17)
    assert a[0].shape==(17,400,4) and a[1].shape==(17,20,4)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])


def test_score_full_rotation_degrees_and_both_weightings():
    c=corpus()
    class Offset(torch.nn.Module):
        def forward(self,x):
            v=torch.tensor([0.,0.,np.pi/6],dtype=x.dtype).expand(len(x),20,3)
            return e.geometry.exp_so3(v)
    result,errors,draws=e.score(Offset(),c,'cpu',windows=7,batch_size=3,keep_errors=True)
    assert result['window_weighted_deg']==pytest.approx(30,abs=2e-5)
    assert result['user_balanced_deg']==pytest.approx(30,abs=2e-5)
    assert errors.shape==(2,7) and draws.shape==(2,7,3)
    assert np.all(draws[0,:,0]==0) and np.all(draws[1,:,0]==1)
    assert np.all(draws[:,:,2]>=0)


def test_distinct_population_and_user_metrics():
    c=corpus()
    angle=np.pi/6
    c.users[1].segments[0][:]=np.array([np.cos(angle/2),0,np.sin(angle/2),0],dtype=np.float32)
    class Identity(torch.nn.Module):
        def forward(self,x):
            y=torch.zeros((len(x),20,4),dtype=x.dtype);y[...,0]=1
            return y
    result=e.score(Identity(),c,'cpu',windows=5,batch_size=3)
    assert result['user_balanced_deg']==pytest.approx(15,abs=2e-5)
    assert result['window_weighted_deg']==pytest.approx(30*281/(81+281),abs=2e-5)


def test_bootstrap_resamples_users_and_weight_scale_is_irrelevant():
    error=np.array([[1,1,1],[9,9,9]],dtype=float)
    a=e.bootstrap(error,[.1,.9]);b=e.bootstrap(error,[1,9])
    assert a==b
    assert a['window_weighted_ci95']==[1.,9.]
    assert a['user_balanced_ci95']==[1.,9.]


def test_kinematic_constant_angular_velocity():
    t=torch.arange(400,dtype=torch.float64)[None,:,None]
    velocity=torch.tensor([[[0.,.004,0.]]],dtype=torch.float64)
    x=e.geometry.exp_so3(t*velocity)
    expected=e.geometry.exp_so3(torch.arange(400,420,dtype=torch.float64)[None,:,None]*velocity)
    for w in (1,5):
        assert e.geometry.geodesic_radians(e.Kinematic(w)(x),expected).max()<1e-10


def test_lock_source_change_rejected_before_checkpoint_load(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'sources',lambda:{'module':'changed'})
    e.atomic_json(tmp_path/'protocol.json',{'sources':{'module':'old'}})
    e.atomic_json(tmp_path/'all_models_locked.json',{'protocol_sha256':e.file_sha256(tmp_path/'protocol.json'),
        'sources':{'module':'old'},'models':{n:{} for n in e.NAMES}})
    with pytest.raises(ValueError,match='lock mismatch'):e.verify_locks(tmp_path)


def test_checkpoint_internal_binding_is_checked(tmp_path):
    e.atomic_json(tmp_path/'protocol.json',{'synthetic_fixture':True})
    path=tmp_path/'model.pt'
    torch.save({'binding':{'protocol_sha256':'wrong','model':'tcn_v2'},'config':{}},path)
    with pytest.raises(ValueError,match='internal binding'):
        e.checked_checkpoint(path,{'configs':{'tcn_v2':{}}},tmp_path,'tcn_v2','cpu')


def test_protocol_preserves_interleaved_manifest_evaluation_order(tmp_path):
    manifest={'shards':[{'user':u,'partition':p} for u,p in
              [('outer-a','observed_dev'),('fit-a','fit'),('inner-b','inner'),('final-a','final'),
               ('outer-c','observed_dev'),('inner-d','inner')]]}
    e.atomic_json(tmp_path/'manifest.json',manifest)
    plan=e.protocol(tmp_path,manifest,steps=2,batch_size=4,base_seconds=60,
                    refine_steps=2,refine_seconds=60,width=32)
    assert plan['development_user_order']==['outer-a','inner-b','outer-c','inner-d']
    assert plan['development_user_order']!=plan['split']['inner']+plan['split']['observed_dev']
    assert plan['final_user_order']==['final-a']


def test_completed_evaluation_replay_checks_receipt_before_loading_data(tmp_path,monkeypatch):
    plan={'development_user_order':['synthetic-a']}
    lock={'protocol_sha256':'pinned'}
    monkeypatch.setattr(e,'verify_locks',lambda _: (plan,lock))
    errors=tmp_path/'development_errors.npz';errors.write_bytes(b'synthetic_saved_errors')
    value={'protocol_sha256':'pinned','evaluation_partition':'development',
           'errors_sha256':e.file_sha256(errors),'evaluation_user_order':['synthetic-a']}
    e.atomic_json(tmp_path/'development_results.json',value)
    assert e.evaluate(tmp_path/'missing_corpus',tmp_path,device='cpu',partition='development')==value
    errors.write_bytes(b'altered_saved_errors')
    with pytest.raises(ValueError,match='receipt'):
        e.evaluate(tmp_path/'missing_corpus',tmp_path,device='cpu',partition='development')


@pytest.mark.parametrize('name',['ordered_v2','energy_v2','jepa_v2','gru_v2'])
def test_tiny_synthetic_phase_checkpoint_binding_and_optimizer_step(tmp_path,name):
    torch.set_num_threads(1)
    c=corpus((430,440))
    configuration={'width':8,'latent_dim':16,'use_gravity':True}
    if name=='energy_v2':configuration['energy_warmup']=0
    plan={'configs':{name:configuration},'seed':123,'base_lr':.001,'refine_lr':.0002,
          'weight_decay':.0001,'gradient_clip':1.,'development_windows_per_user':2,'batch_size':4,
          'eval_every':2}
    e.atomic_json(tmp_path/'protocol.json',plan)
    result=e.train_phase(name,c.subset(['synthetic-0']),c.subset(['synthetic-1']),plan,tmp_path,'cpu',
                         phase='base',steps=2,max_seconds=60)
    assert result['completed_steps']==2
    state=e.checked_checkpoint(tmp_path/name/'base_last.pt',plan,tmp_path,name,'cpu')
    assert state['phase_step']==2
    assert any(int(s['step'])==2 for s in state['optimizer']['state'].values())
    assert result['selected_checkpoint_phase']=='base'

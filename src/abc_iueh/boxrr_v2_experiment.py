"""Expanded official BOXRR experiment: development-only search, physically sealed test.

The final archive is not present during training. All configurations, checkpoint
hashes and the development-selected primary model are locked before final loading.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from . import boxrr_models as geometry
from .boxrr_experiment import RamCorpus, atomic_json, atomic_torch, read_json
from .boxrr_source import file_sha256, REVISION

NAMES=('ordered_v2','mlp_v2','tcn_v2','energy_v2','jepa_v2','desp_v2','gru_v2')
SEED=20261110


def build_model(name,config,seed=SEED):
    if name in ('energy_v2','jepa_v2'):
        from .boxrr_v2_latent import build_latent_model
        return build_latent_model(name,seed=seed,**config)
    if name=='gru_v2':
        from .boxrr_v2_extra import build_extra_model
        return build_extra_model(name,seed=seed,**config)
    from .boxrr_v2_deterministic import build_model as deterministic
    return deterministic(name,seed=seed,**config)


def objective(model,x,y,step,total_steps,phase):
    if getattr(model,'arm',None) in ('energy_v2','jepa_v2'):
        from .boxrr_v2_latent import training_loss
        return training_loss(model,x,y,step,total_steps,phase=phase)
    predicted=model(x)
    errors=geometry.geodesic_radians(predicted,y)
    loss=errors[:,-1].mean()+.1*errors.mean()
    return loss,{'endpoint_radians':errors[:,-1].mean()}


def after_step(model):
    if getattr(model,'arm',None) in ('energy_v2','jepa_v2'):
        from .boxrr_v2_latent import after_optimizer_step
        after_optimizer_step(model)


def sources():
    folder=Path(__file__).parent
    names=('boxrr_v2_experiment.py','boxrr_v2_deterministic.py','boxrr_v2_latent.py',
           'boxrr_v2_extra.py','boxrr_models.py','boxrr_experiment.py','boxrr_source.py')
    return {n:file_sha256(folder/n) for n in names}


def require_corpus(root,manifest,partitions,*,test_locked=False):
    """Verify only allowed shards; never open final arrays during a training check."""
    if manifest.get('source')!='BOXRR-23' or manifest.get('training_allowed') is not True:
        raise ValueError('Only an authorized official BOXRR corpus is accepted')
    if manifest.get('source_revision')!=REVISION:raise ValueError('Official source revision differs')
    config=manifest['config']
    if (config['context_samples'],config['horizon_samples'],config['target_rate_hz'])!=(400,20,100):
        raise ValueError('The fixed challenge interface/timebase changed')
    if 'final' in partitions and not test_locked:raise ValueError('Final requires locked model selection')
    users=[r['user'] for r in manifest['shards']]
    if len(set(users))!=len(users):raise ValueError('Duplicate user/cross-partition leakage')
    if any(r['partition'] not in ('fit','inner','observed_dev','final') for r in manifest['shards']):
        raise ValueError('Unknown partition')
    allowed=[]
    root=Path(root).resolve()
    for row in manifest['shards']:
        if row['partition'] not in partitions:continue
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing/unsafe allowed shard')
        if path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']:
            raise ValueError('Allowed shard checksum differs')
        if row.get('eligible_windows',0)<1:raise ValueError('Invalid window population')
        allowed.append(row['user'])
    if not allowed:raise ValueError('Empty requested corpus')
    return allowed


class MixedSampler:
    """Half participant-balanced, half window-balanced, fixed before model errors."""
    def __init__(self,corpus):
        self.corpus=corpus
        counts=np.array([u.cumulative[-1] for u in corpus.users],dtype=np.float64)
        self.probability=.5/len(counts)+.5*counts/counts.sum()

    def batch(self,rng,count):
        indices=rng.choice(len(self.corpus.users),size=count,p=self.probability)
        data=np.empty((count,420,4),dtype=np.float32)
        for j,index in enumerate(indices):
            user=self.corpus.users[int(index)]
            draw=int(rng.integers(int(user.cumulative[-1])))
            segment=int(np.searchsorted(user.cumulative,draw,side='right'))
            start=draw-(int(user.cumulative[segment-1]) if segment else 0)
            data[j]=user.segments[segment][start:start+420]
        return torch.from_numpy(data[:,:400]),torch.from_numpy(data[:,400:])


def population_weights(corpus):
    weights=np.asarray([u.cumulative[-1] for u in corpus.users],dtype=np.float64)
    return weights/weights.sum()


def score(model,corpus,device,*,windows=256,batch_size=128,seed=SEED+1,keep_errors=False):
    rng=np.random.default_rng(seed)
    model.eval()
    means=[];curves=[];all_errors=[];origins=[];diagnostic_users={}
    with torch.inference_mode():
        for index in range(len(corpus.users)):
            errors=[];draws=[];diagnostic_totals={}
            for start in range(0,windows,batch_size):
                count=min(batch_size,windows-start)
                x,y,where=corpus.batch(rng,count,index,return_draws=True)
                if hasattr(model,'forward_with_diagnostics'):
                    predicted,details=model.forward_with_diagnostics(x.to(device))
                    for key,value in details.items():
                        diagnostic_totals[key]=diagnostic_totals.get(key,0.)+float(value.detach().double().sum().cpu())
                else:predicted=model(x.to(device))
                if predicted.shape!=(count,20,4) or not torch.isfinite(predicted).all():
                    raise ValueError('Invalid model prediction')
                if not torch.allclose(predicted.norm(dim=-1),torch.ones_like(predicted[...,0]),atol=2e-4,rtol=2e-4):
                    raise ValueError('Output quaternion norm differs from one')
                error=geometry.geodesic_radians(predicted,y.to(device))*180/math.pi
                errors.append(error.cpu().numpy());draws.append(where)
            values=np.concatenate(errors)
            means.append(float(values[:,-1].astype(np.float64).mean()))
            curves.append(values.astype(np.float64).mean(0))
            for key,value in diagnostic_totals.items():diagnostic_users.setdefault(key,[]).append(value/windows)
            if keep_errors:all_errors.append(values[:,-1]);origins.append(np.concatenate(draws))
    weights=population_weights(corpus)
    result={'window_weighted_deg':float(np.dot(weights,means)),
            'user_balanced_deg':float(np.mean(means)),
            'per_user_deg':means,'population_weights':weights.tolist(),
            'window_weighted_horizon_deg':np.dot(weights,np.asarray(curves)).tolist(),
            'user_balanced_horizon_deg':np.mean(curves,axis=0).tolist()}
    if diagnostic_users:
        result['inference_diagnostics']={k:{'user_balanced':float(np.mean(v)),
                                           'window_weighted':float(np.dot(weights,v))}
                                         for k,v in diagnostic_users.items()}
    if keep_errors:return result,np.asarray(all_errors),np.asarray(origins)
    return result


def status(output,**fields):
    value={'state':'running','updated_unix':time.time(),**fields}
    atomic_json(Path(output)/'status.json',value)


def protocol(root,manifest,*,steps,batch_size,base_seconds,refine_steps,refine_seconds,width):
    if min(steps,batch_size,base_seconds,refine_steps,refine_seconds,width)<=0:
        raise ValueError('Training bounds must be positive')
    split={p:[r['user'] for r in manifest['shards'] if r['partition']==p]
           for p in ('fit','inner','observed_dev','final')}
    reference=Path(root).parent/'reference_tcn.pt'
    return {'schema':'boxrr-round2-experiment-v1','manifest_sha256':file_sha256(Path(root)/'manifest.json'),
        'previous_tcn_sha256':file_sha256(reference) if reference.is_file() else None,
        'sources':sources(),'model_names':list(NAMES),'seed':SEED,'split':split,
        'configs':{n:{'width':width,'latent_dim':128,'use_gravity':True} for n in NAMES},
        'development_user_order':[r['user'] for r in manifest['shards'] if r['partition'] in ('inner','observed_dev')],
        'final_user_order':[r['user'] for r in manifest['shards'] if r['partition']=='final'],
        'base_steps':steps,'batch_size':batch_size,'base_max_seconds':base_seconds,
        'jepa_pretrain_steps':5000,'jepa_pretrain_max_seconds':600,
        'refine_steps':refine_steps,'refine_top_k':3,'refine_max_seconds':refine_seconds,
        'eval_every':2000,'development_windows_per_user':256,'final_windows_per_user':1024,
        'train_sampling':'0.5 uniform_user + 0.5 uniform_eligible_window; live with replacement',
        'optimizer':'AdamW','base_lr':.001,'refine_lr':.0002,'weight_decay':.0001,
        'gradient_clip':1.,'loss':'endpoint_geodesic + .1 trajectory; latent arms separately documented',
        'development':'old29inner + old72previously_observed_outer, all excluded from fitting/SSL',
        'selection_metric':'window-weighted endpoint20 mean geodesic degrees on all101 development users',
        'secondary_metric':'equal-user mean endpoint20 degrees',
        'final_policy':'new-user archive absent during all training; unlock once after checkpoint and winner hashes',
        'timing':'100Hz; endpoint20 is +200ms by indexing; official100ms wording remains unresolved',
        'targets':{'all_models_below_deg':3.,'any_model_below_deg':1.,'guaranteed':False}}


def numeric(value):
    if torch.is_tensor(value):value=float(value.detach().cpu())
    else:value=float(value)
    return value if math.isfinite(value) else None


def checked_checkpoint(path,plan,output,name,device):
    state=torch.load(path,map_location=device,weights_only=True)
    expected=file_sha256(Path(output)/'protocol.json')
    if (state.get('binding',{}).get('protocol_sha256')!=expected or
        state.get('binding',{}).get('model')!=name or state.get('config')!=plan['configs'][name]):
        raise ValueError('Checkpoint internal binding/config differs')
    return state


def train_phase(name,fit,development,plan,output,device,*,phase,steps,max_seconds,initial=None):
    output=Path(output);directory=output/name;directory.mkdir(exist_ok=True)
    model=build_model(name,plan['configs'][name],plan['seed']).to(device)
    initial_binding=None;warmup_offset=0
    if initial is not None:
        initial_state=checked_checkpoint(initial,plan,output,name,device)
        model.load_state_dict(initial_state['model']);initial_binding=initial_state['binding']
        if phase=='refine' and name=='energy_v2':warmup_offset=int(initial_state['phase_step'])
        del initial_state
    lr=plan['refine_lr'] if phase=='refine' else plan['base_lr']
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=plan['weight_decay'])
    sampler=MixedSampler(fit)
    rng=np.random.default_rng(plan['seed']+(900000 if phase=='refine' else 800000 if phase=='pretrain' else 0))
    torch.manual_seed(plan['seed']+(900000 if phase=='refine' else 800000 if phase=='pretrain' else 0))
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(plan['seed'])
    history=[];best=float('inf');best_step=-1;step=0;losses=[]
    phase_last=directory/f'{phase}_last.pt'
    binding={'protocol_sha256':file_sha256(output/'protocol.json'),'model':name,'phase':phase}
    if phase=='refine':
        previous=read_json(directory/'base_lock.json')
        best=previous['best_development_deg'];best_step=previous['selected_step']
    started=time.monotonic()

    def checkpoint():
        return {'binding':binding,'config':plan['configs'][name],'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),'phase_step':step,'best_development_deg':best,
                'sampler_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state() if str(device).startswith('cuda') else None,
                'elapsed_seconds':time.monotonic()-started}

    def validate():
        nonlocal best,best_step
        measured=score(model,development,device,windows=plan['development_windows_per_user'],batch_size=plan['batch_size'])
        selected=measured['window_weighted_deg']
        if selected<best:
            best=selected;best_step=step
            atomic_torch(directory/'best.pt',checkpoint())
        row={'phase':phase,'step':step,'elapsed_seconds':time.monotonic()-started,
             'window_weighted_deg':selected,'user_balanced_deg':measured['user_balanced_deg'],'best_deg':best}
        history.append(row)
        atomic_json(directory/f'{phase}_history.json',history)
        atomic_torch(phase_last,checkpoint())
        status(output,stage=phase,current_arm=name,step=step,metrics=row)
        print(json.dumps({'event':'development','model':name,**row}),flush=True)

    if phase!='pretrain':validate()
    while step<steps and time.monotonic()-started<max_seconds:
        model.train()
        x,y=sampler.batch(rng,plan['batch_size']);x=x.to(device);y=y.to(device)
        progress=step/max(1,steps-1)
        current_lr=lr*(.05+.95*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=current_lr
        optimizer.zero_grad(set_to_none=True)
        loss,diagnostics=objective(model,x,y,step+warmup_offset,steps+warmup_offset,'pretrain' if phase=='pretrain' else 'finetune')
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite objective: {name}/{phase}/{step}')
        loss.backward()
        if name=='energy_v2':
            density=list(model.density_head.parameters());energy=list(model.energy_head.parameters())
            auxiliaries={id(p) for p in density+energy}
            forecast=[p for p in model.parameters() if id(p) not in auxiliaries]
            norms=[torch.nn.utils.clip_grad_norm_(group,plan['gradient_clip'],error_if_nonfinite=True)
                   for group in (forecast,density,energy)]
            # clip_grad_norm_ returns a CPU zero for an inactive warm-up group.
            grad=torch.stack([n.to(x.device) for n in norms]).max()
            diagnostics.update({f'gradient_group_{i}':n.detach() for i,n in enumerate(norms)})
        else:grad=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();after_step(model);step+=1
        losses.append(float(loss.detach()))
        if step%200==0:
            row={'event':'train','model':name,'phase':phase,'step':step,'mean_loss':float(np.mean(losses)),
                 'lr':current_lr,'gradient_norm':float(grad),'elapsed_seconds':time.monotonic()-started,
                 'diagnostics':{k:numeric(v) for k,v in diagnostics.items()}}
            losses=[]
            print(json.dumps(row,allow_nan=False),flush=True)
            status(output,stage=phase,current_arm=name,step=step)
            with (directory/f'{phase}_training.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        if step%plan['eval_every']==0:
            if phase=='pretrain':atomic_torch(phase_last,checkpoint())
            else:validate()
    if phase=='pretrain':atomic_torch(phase_last,checkpoint())
    elif not history or history[-1]['step']!=step:validate()
    result={'binding':binding,'completed_steps':step,'elapsed_seconds':time.monotonic()-started,
            'termination':'steps' if step>=steps else 'time_cap','last_sha256':file_sha256(phase_last),
            'selected_step':best_step,'best_development_deg':best if math.isfinite(best) else None,
            'optimizer_reset':True,'initial_checkpoint_binding':initial_binding,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'all_parameters':sum(p.numel() for p in model.parameters())}
    if name in ('energy_v2','jepa_v2'):
        from .boxrr_v2_latent import parameter_counts
        result['parameter_counts']=parameter_counts(model)
    else:result['parameter_counts']={'total':result['all_parameters'],'trainable':result['trainable_parameters'],
                                     'inference':result['all_parameters']}
    if phase!='pretrain':result['best_sha256']=file_sha256(directory/'best.pt')
    if phase!='pretrain':
        selected_state=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        result['selected_checkpoint_phase']=selected_state['binding']['phase']
        result['selected_checkpoint_phase_step']=selected_state['phase_step']
    atomic_json(directory/f'{phase}_lock.json',result)
    del model,optimizer
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result


def train(root,output,*,device='cuda',steps=20000,batch_size=128,base_seconds=1500,
          refine_steps=40000,refine_seconds=1800,width=32):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=read_json(root/'manifest.json')
    allowed=require_corpus(root,manifest,{'fit','inner','observed_dev'})
    if any((root/r['path']).exists() for r in manifest['shards'] if r['partition']=='final'):
        raise ValueError('Final files must be physically absent during model search')
    plan=protocol(root,manifest,steps=steps,batch_size=batch_size,base_seconds=base_seconds,
                  refine_steps=refine_steps,refine_seconds=refine_seconds,width=width)
    if (output/'protocol.json').exists():raise FileExistsError('Use a fresh output for each frozen round')
    atomic_json(output/'protocol.json',plan)
    status(output,stage='loading_training')
    corpus=RamCorpus.load(root,manifest,allowed)
    fit=corpus.subset(plan['split']['fit'])
    development=corpus.subset(plan['split']['inner']+plan['split']['observed_dev'])
    print(json.dumps({'event':'loaded','fit_users':len(fit.users),'development_users':len(development.users),
                      'ram_bytes':corpus.bytes,'final_arrays_loaded':False}),flush=True)
    locks={}
    for name in NAMES:
        initial=None
        if name=='jepa_v2':
            train_phase(name,fit,development,plan,output,device,phase='pretrain',
                        steps=plan['jepa_pretrain_steps'],max_seconds=plan['jepa_pretrain_max_seconds'])
            initial=output/name/'pretrain_last.pt'
        locks[name]=train_phase(name,fit,development,plan,output,device,phase='base',steps=steps,
                               max_seconds=base_seconds,initial=initial)
    promoted=sorted(NAMES,key=lambda n:(locks[n]['best_development_deg'],n))[:plan['refine_top_k']]
    atomic_json(output/'promotion.json',{'rule':'top3 development window-weighted endpoint20',
                                       'promoted':promoted,'base_locks':locks})
    for name in promoted:
        locks[name]=train_phase(name,fit,development,plan,output,device,phase='refine',steps=refine_steps,
                               max_seconds=refine_seconds,initial=output/name/'best.pt')
    selected=min(NAMES,key=lambda n:(locks[n]['best_development_deg'],n))
    receipt={'protocol_sha256':file_sha256(output/'protocol.json'),'sources':sources(),
             'models':{n:{'config':plan['configs'][n],'best_sha256':file_sha256(output/n/'best.pt'),
                          'development_deg':locks[n]['best_development_deg']} for n in NAMES},
             'primary_selected_model':selected,'selected_using':'development_only',
             'final_arrays_loaded':False,'locked_unix':time.time()}
    atomic_json(output/'all_models_locked.json',receipt)
    status(output,state='awaiting_final_archive',stage='all_models_locked',selected=selected,completed_arms=list(NAMES))
    del corpus,fit,development;gc.collect()
    return receipt


def verify_locks(output):
    output=Path(output);plan=read_json(output/'protocol.json');locks=read_json(output/'all_models_locked.json')
    if (locks['protocol_sha256']!=file_sha256(output/'protocol.json') or locks['sources']!=sources()
        or plan['sources']!=sources() or set(locks['models'])!=set(NAMES)):
        raise ValueError('Model/protocol/source lock mismatch')
    for n in NAMES:
        if file_sha256(output/n/'best.pt')!=locks['models'][n]['best_sha256']:raise ValueError('Checkpoint changed')
        checked_checkpoint(output/n/'best.pt',plan,output,n,'cpu')
    if locks['primary_selected_model'] not in NAMES or locks['selected_using']!='development_only':
        raise ValueError('Invalid prespecified primary model')
    return plan,locks


def bootstrap(errors,weights,seed=SEED+97):
    means=np.asarray(errors,dtype=np.float64).mean(1);weights=np.asarray(weights,dtype=np.float64)
    rng=np.random.default_rng(seed);draws=rng.integers(len(means),size=(5000,len(means)))
    macro=means[draws].mean(1)
    weighted=(means[draws]*weights[draws]).sum(1)/weights[draws].sum(1)
    return {'window_weighted_ci95':np.quantile(weighted,[.025,.975]).tolist(),
            'user_balanced_ci95':np.quantile(macro,[.025,.975]).tolist()}


class Kinematic(torch.nn.Module):
    def __init__(self,width):super().__init__();self.width=width
    def forward(self,x):
        last=geometry.normalize_quaternion(x[:,-1])
        if self.width==0:return last[:,None].expand(-1,20,-1)
        delta=geometry.multiply_wxyz(geometry.conjugate_wxyz(x[:,-self.width-1]),last)
        v=geometry.log_so3(delta)/self.width
        h=torch.arange(1,21,device=x.device,dtype=x.dtype)[None,:,None]
        return geometry.multiply_wxyz(last[:,None],geometry.exp_so3(h*v[:,None]))


class EnergyControl(torch.nn.Module):
    def __init__(self,model,mode):super().__init__();self.model=model;self.mode=mode
    def forward(self,x):
        if self.mode=='proposal':return self.model.proposal_prediction(x)
        return self.model.decode_density(*self.model.encode(x),use_energy=False)[0]


def evaluate(root,output,*,device='cuda',partition='final'):
    root=Path(root);output=Path(output);plan,locks=verify_locks(output)
    if partition not in ('final','development'):raise ValueError('Unknown evaluation partition')
    is_final=partition=='final';result_name='results.json' if is_final else 'development_results.json'
    if (output/result_name).exists():
        existing=read_json(output/result_name)
        if (existing.get('protocol_sha256')!=locks['protocol_sha256']
            or existing.get('evaluation_partition')!=partition
            or existing.get('errors_sha256')!=file_sha256(output/(partition+'_errors.npz'))
            or existing.get('evaluation_user_order')!=plan['final_user_order' if is_final else 'development_user_order']):
            raise ValueError('Existing evaluation receipt differs from the locked experiment')
        return existing
    manifest=read_json(root/'manifest.json')
    if file_sha256(root/'manifest.json')!=plan['manifest_sha256']:raise ValueError('Final corpus lock differs')
    allowed=require_corpus(root,manifest,{'final'} if is_final else {'inner','observed_dev'},test_locked=is_final)
    opened={'all_models_lock_sha256':file_sha256(output/'all_models_locked.json'),
            'models':list(NAMES),'evaluation_seed':SEED+2,'windows_per_user':plan['final_windows_per_user']}
    marker=output/('final_opened.json' if is_final else 'development_report_opened.json')
    if marker.exists():
        prior=read_json(marker)
        if any(prior.get(k)!=v for k,v in opened.items()):raise ValueError('Final replay configuration changed')
    else:atomic_json(marker,{**opened,'opened_unix':time.time()})
    status(output,stage='loading_'+partition,completed_arms=list(NAMES))
    corpus=RamCorpus.load(root,manifest,allowed)
    user_order=[u.user for u in corpus.users]
    if user_order!=plan['final_user_order' if is_final else 'development_user_order']:
        raise ValueError('Evaluation user order differs from the frozen protocol')
    report={'schema':'boxrr-round2-results-v1','primary_selected_model':locks['primary_selected_model'],
            'protocol_sha256':file_sha256(output/'protocol.json'),'evaluation_users':len(allowed),
            'evaluation_partition':partition,'used_for_model_selection':not is_final,
            'evaluation_user_order':user_order,
            'windows_per_user':plan['final_windows_per_user'],'metric':'endpoint20 fullSO3 geodesic degrees',
            'models':{},'baselines':{},'diagnostics':{},'bootstrap_unit':'user','final_is_official_challenge_test':False}
    arrays={};reference_draws=None
    candidates=[(n,None) for n in NAMES]+[('zero_velocity',0),('cav1',1),('cav5',5)]
    candidates += [('energy_proposal','proposal'),('energy_q_only','q_only')]
    if plan.get('previous_tcn_sha256'):
        if file_sha256(root.parent/'reference_tcn.pt')!=plan['previous_tcn_sha256']:
            raise ValueError('Pinned prior BOXRR TCN checkpoint differs')
        candidates.append(('previous_tcn','previous_tcn'))
    for name,kinematic in candidates:
        if kinematic is None:
            model=build_model(name,plan['configs'][name],plan['seed']).to(device)
            model.load_state_dict(checked_checkpoint(output/name/'best.pt',plan,output,name,device)['model'])
        elif kinematic in ('proposal','q_only'):
            energy=build_model('energy_v2',plan['configs']['energy_v2'],plan['seed']).to(device)
            energy.load_state_dict(checked_checkpoint(output/'energy_v2'/'best.pt',plan,output,'energy_v2',device)['model'])
            model=EnergyControl(energy,kinematic).to(device)
        elif kinematic=='previous_tcn':
            model=geometry.build_model('tcn').to(device)
            state=torch.load(root.parent/'reference_tcn.pt',map_location=device,weights_only=True)
            model.load_state_dict(state['model'])
        else:model=Kinematic(kinematic).to(device)
        measured,errors,draws=score(model,corpus,device,windows=plan['final_windows_per_user'],
                                   batch_size=plan['batch_size'],seed=SEED+2,keep_errors=True)
        if reference_draws is None:reference_draws=draws
        elif not np.array_equal(reference_draws,draws):raise ValueError('Models used different final origins')
        measured.update(bootstrap(errors,measured['population_weights']))
        measured['sampled_windows_under1_fraction']=float((errors<1).mean())
        measured['sampled_window_median_deg']=float(np.median(errors))
        measured['sampled_window_p95_deg']=float(np.quantile(errors,.95))
        measured['user_means_under1']=int((errors.mean(1)<1).sum())
        group='models' if kinematic is None else 'diagnostics' if kinematic in ('proposal','q_only') else 'baselines'
        report[group][name]=measured
        arrays[name]=errors
        atomic_json(output/(partition+'_results.partial.json'),report)
        print(json.dumps({'event':partition,'model':name,'window_weighted_deg':measured['window_weighted_deg'],
                          'user_balanced_deg':measured['user_balanced_deg']}),flush=True)
        del model
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    arrays['origins']=reference_draws
    arrays['user_order']=np.asarray(user_order,dtype='U')
    arrays['population_weights']=population_weights(corpus)
    errors_path=output/(partition+'_errors.npz')
    np.savez_compressed(errors_path,**arrays)
    report['errors_sha256']=file_sha256(errors_path)
    report['duplicate_draws']=int(sum(len(d)-len(np.unique(d,axis=0)) for d in reference_draws))
    report['all_below3']=all(r['window_weighted_deg']<3 for r in report['models'].values())
    report['any_below1']=any(r['window_weighted_deg']<1 for r in report['models'].values())
    atomic_json(output/result_name,report)
    status(output,state='complete',stage=partition+'_evaluated',completed_arms=list(NAMES),final_remains_sealed=not is_final)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('train','evaluate','development'),required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--base-seconds',type=int,default=1500)
    p.add_argument('--refine-steps',type=int,default=40000);p.add_argument('--refine-seconds',type=int,default=1800)
    p.add_argument('--width',type=int,default=32);args=p.parse_args()
    torch.set_num_threads(1)
    if args.stage=='train':train(args.corpus,args.output,device=args.device,steps=args.steps,batch_size=args.batch_size,
        base_seconds=args.base_seconds,refine_steps=args.refine_steps,refine_seconds=args.refine_seconds,width=args.width)
    else:evaluate(args.corpus,args.output,device=args.device,partition='development' if args.stage=='development' else 'final')


if __name__=='__main__':main()

"""Budget-bounded three-model BOXRR study; no final-user evaluation path."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import boxrr_models as geo
from .boxrr_experiment import RamCorpus, atomic_json, atomic_torch, read_json
from .boxrr_source import file_sha256
from .boxrr_v2_experiment import require_corpus, population_weights, score

MANIFEST_SHA = '3e0d7fba1a28bf09a03c5d4f00d773b9583fe309df4417a28380882f5b680293'
MODELS = ('direct', 'cde', 'imm', 'tcn_control')
SEED = 20261114


def build_model(name, config=None, seed=SEED):
    torch.manual_seed(seed)
    config = dict(config or {})
    if name == 'direct':
        from .boxrr_v5_direct import DirectSO3Conv
        return DirectSO3Conv(**config)
    if name == 'cde':
        from .boxrr_v5_cde import SGNeuralCDE
        return SGNeuralCDE(**config)
    if name == 'imm':
        from .boxrr_v5_imm import IMMResidual
        return IMMResidual(**config)
    if name == 'tcn_control':
        from .boxrr_v2_deterministic import build_model as old
        return old('tcn_v2', seed=seed, width=config.get('width', 64), latent_dim=128, use_gravity=True)
    raise ValueError('Unknown v5 model')


def sources():
    directory = Path(__file__).parent
    names = ('boxrr_v5_experiment.py', 'boxrr_v5_direct.py', 'boxrr_v5_cde.py', 'boxrr_v5_imm.py',
             'boxrr_models.py', 'boxrr_v2_deterministic.py', 'boxrr_v2_experiment.py',
             'boxrr_experiment.py', 'boxrr_source.py')
    return {n: file_sha256(directory/n) for n in names}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def set_runtime():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def objective(model, x, y):
    error = geo.geodesic_radians(model(x), y)
    return error[:, -1].mean()+.1*error.mean(), error[:, -1].mean()


class PopulationSampler:
    """Exactly the eligible-window risk used by the primary metric."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.weights = population_weights(corpus)

    def batch(self, rng, count):
        who = rng.choice(len(self.corpus.users), size=count, p=self.weights)
        data = np.empty((count, 420, 4), dtype=np.float32)
        for j, index in enumerate(who):
            u = self.corpus.users[int(index)]
            draw = int(rng.integers(int(u.cumulative[-1])))
            seg = int(np.searchsorted(u.cumulative, draw, side='right'))
            start = draw-(int(u.cumulative[seg-1]) if seg else 0)
            data[j] = u.segments[seg][start:start+420]
        return torch.from_numpy(data[:, :400]), torch.from_numpy(data[:, 400:])


def protocol():
    return {'schema': 'boxrr-three-model-v5', 'manifest_sha256': MANIFEST_SHA,
        'source_sha256': sources(), 'seed': SEED, 'model_order': list(MODELS),
        'configs': {'direct': {'width': 64}, 'cde': {'width': 64}, 'imm': {'width': 64}, 'tcn_control': {'width': 64}},
        'base_steps': 20000, 'refine_steps': 40000, 'base_seconds': 1800, 'refine_seconds': 3600,
        'control_base_seconds': 900, 'control_refine_seconds': 1800,
        'batch_size': 128, 'eval_every': 2000, 'save_every': 1000,
        'selection_windows': 64, 'selection_seed': 20261111,
        'report_windows': 1024, 'report_seed': 20261112,
        'fit_diagnostic_users': 64, 'fit_diagnostic_windows': 32, 'fit_diagnostic_seed': 20261116,
        'sampling': 'uniform eligible window over all600fit users; same for all4arms',
        'sampling_change_from_v2': 'v2 used .5uniformuser+.5window; new scratchTCN control shares v5sampling',
        'optimizer': 'AdamW', 'base_lr': .001, 'refine_lr': .0002, 'weight_decay': .0001,
        'gradient_clip': 1., 'loss': 'endpoint geodesic + .1 mean trajectory geodesic radians',
        'refine_optimizer_reset': True, 'model_selection': 'minimum weighted development endpoint20',
        'metric': 'window-weighted mean physical fullSO3 endpoint20 degrees',
        'timebase': '100Hz; endpoint20 is +200ms; organizer100ms wording unresolved',
        'final_policy': 'never_upload_or_evaluate', 'legacy_assets': 'excluded',
        'target_deg': 2., 'guaranteed': False, 'source_round': 'new6USDround, campaign20USDcap'}


def finite_number(x):
    y = float(x)
    if not math.isfinite(y):
        raise ValueError('Nonfinite training diagnostic')
    return y


def rng_state(rng, device=None):
    return {'numpy': rng.bit_generator.state, 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device is not None and device.type == 'cuda' else []}


def restore_rng(rng, state, device):
    rng.bit_generator.state = state['numpy']
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def clone_cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [clone_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(clone_cpu(v) for v in value)
    return value


def training_state(model, optimizer, rng, *, binding, phase, step, elapsed, best,
                   total_steps, compute_steps):
    device = next(model.parameters()).device
    return clone_cpu({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'rng': rng_state(rng, device), 'binding': binding, 'phase': phase, 'step': step,
        'elapsed': elapsed, 'best': best, 'total_steps': total_steps,
        'compute_steps': compute_steps})


def load_trusted(path, device):
    # Only source-bound checkpoints from this run are accepted, never arbitrary uploaded weights.
    return torch.load(path, map_location=device, weights_only=False)


def checked_checkpoint(path, binding, device, *, sha256=None, phase=None):
    if sha256 is not None and file_sha256(path) != sha256:
        raise ValueError('Checkpoint checksum differs')
    state = load_trusted(path, device)
    if state.get('binding') != binding or (phase is not None and state.get('phase') != phase):
        raise ValueError('Checkpoint name/config/seed/protocol binding differs')
    if state.get('phase') not in ('base', 'refine'):
        raise ValueError('Checkpoint phase differs')
    if not isinstance(state.get('step'), int) or state['step'] < 0:
        raise ValueError('Invalid checkpoint step')
    if state['total_steps'] < state['step'] or state['compute_steps'] < state['total_steps']:
        raise ValueError('Invalid checkpoint update accounting')
    finite_number(state['elapsed']); finite_number(state['best'])
    return state


def checked_phase(directory, phase, binding, device):
    lock = read_json(directory/(phase+'_lock.json'))
    if (lock.get('binding') != binding or lock.get('phase') != phase or
            lock.get('termination') not in ('steps', 'time_cap')):
        raise ValueError('Completed phase binding differs')
    state = checked_checkpoint(directory/(phase+'_last.pt'), binding, device,
                               sha256=lock['last_sha256'], phase=phase)
    selected = checked_checkpoint(directory/(phase+'_best.pt'), binding, device,
                                  sha256=lock['best_sha256'])
    if (state['step'] != lock['completed_steps'] or
            state['total_steps'] != lock['total_completed_steps'] or
            state['compute_steps'] != lock['total_compute_steps'] or
            selected['best'] != lock['best_development_deg']):
        raise ValueError('Completed phase accounting differs')
    return lock, selected


def train_phase(name, config, fit, dev, output, p, device, *, phase, deadline):
    directory = output/name
    directory.mkdir(exist_ok=True)
    binding = {'protocol_sha256': digest(p), 'name': name, 'config': config, 'seed': p['seed']}
    done = directory/(phase+'_lock.json')
    if done.exists():
        return checked_phase(directory, phase, binding, device)[0]
    # This marker survives failed/restarted processes. Downtime consumes the cap
    # conservatively; restarting never buys a fresh phase wall-time allowance.
    clock_path = directory/(phase+'_clock.json')
    if not clock_path.exists():
        atomic_json(clock_path, {'binding': binding, 'phase': phase, 'started_unix': time.time()})
    clock = read_json(clock_path)
    if clock['binding'] != binding or clock['phase'] != phase:
        raise ValueError('Phase clock binding differs')
    elapsed = max(0., time.time()-finite_number(clock['started_unix']))
    began = time.monotonic()
    model = build_model(name, config, p['seed']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['base_lr'] if phase == 'base' else p['refine_lr'],
                                  weight_decay=p['weight_decay'])
    rng = np.random.default_rng(p['seed']+(0 if phase == 'base' else 100))
    steps = p['base_steps'] if phase == 'base' else p['refine_steps']
    seconds = p['base_seconds'] if phase == 'base' else p['refine_seconds']
    if name == 'tcn_control':
        seconds = p['control_base_seconds'] if phase == 'base' else p['control_refine_seconds']
    last = directory/(phase+'_last.pt')
    history_path = directory/(phase+'_history.json')
    history, selected = [], None
    step, best, total_before, compute_before = 0, float('inf'), 0, 0
    if last.exists():
        state = checked_checkpoint(last, binding, device, phase=phase)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(rng, state['rng'], device)
        step, best = state['step'], state['best']
        elapsed = max(elapsed, state['elapsed'])
        total_before = state['total_steps']-step
        compute_before = state['compute_steps']-step
        # History and selection share the same atomic recovery boundary as weights,
        # optimizer and RNG. External JSON/best mirrors are never resume authority.
        history = state['history']
        selected = state['selected_best']
        if selected['binding'] != binding or selected['best'] != best:
            raise ValueError('Resume selected checkpoint binding differs')
    elif phase == 'refine':
        base_lock, state = checked_phase(directory, 'base', binding, device)
        model.load_state_dict(state['model'])
        total_before = state['total_steps']
        compute_before = base_lock['total_compute_steps']
        best = state['best']
        selected = clone_cpu(state)
    sampler = PopulationSampler(fit)

    def spent(): return elapsed+time.monotonic()-began
    def snapshot():
        return training_state(model, optimizer, rng, binding=binding, phase=phase, step=step,
                              elapsed=spent(), best=best, total_steps=total_before+step,
                              compute_steps=compute_before+step)
    def save():
        state = snapshot()
        state.update(history=history, selected_best=selected)
        atomic_torch(last, state)
        atomic_torch(directory/'best.pt', selected)
        atomic_json(history_path, history)

    def evaluate():
        nonlocal best, selected
        result = score(model, dev, device, windows=p['selection_windows'], batch_size=p['batch_size'],
                       seed=p['selection_seed'])
        value = finite_number(result['window_weighted_deg'])
        history.append({'step': step, 'total_steps': total_before+step,
                        'compute_steps': compute_before+step, 'elapsed_seconds': spent(),
                        'weighted_deg': value, 'user_macro_deg': result['user_balanced_deg']})
        if value < best:
            best = value
            selected = snapshot()
        print(json.dumps({'model': name, 'phase': phase, 'step': step, 'dev_deg': value,
                          'best_deg': best, 'elapsed': spent()}), flush=True)

    if step == 0:
        evaluate(); save()
    termination = 'steps'
    while step < steps:
        if (output/'cancel').exists(): termination = 'cancel'; break
        if time.time() >= deadline-120: termination = 'deadline'; break
        if spent() >= seconds: termination = 'time_cap'; break
        model.train(); x, y = sampler.batch(rng, p['batch_size'])
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, endpoint = objective(model, x, y)
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss: '+name)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip'], error_if_nonfinite=True)
        rate = p['base_lr' if phase == 'base' else 'refine_lr']*(.05+.95*.5*(1+math.cos(math.pi*step/steps)))
        for group in optimizer.param_groups: group['lr'] = rate
        optimizer.step(); step += 1
        if step % 200 == 0:
            record = {'model': name, 'phase': phase, 'step': step, 'total_steps': total_before+step,
                      'compute_steps': compute_before+step,
                      'loss': finite_number(loss.detach()), 'endpoint_rad': finite_number(endpoint.detach()),
                      'gradient_norm': finite_number(norm), 'lr': rate, 'elapsed_seconds': spent()}
            with (directory/'training.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
            atomic_json(output/'status.json', {'state': 'training', **record, 'updated_unix': time.time()})
        if step % p['eval_every'] == 0: evaluate()
        if step % p['save_every'] == 0: save()
    if termination in ('steps', 'time_cap') and (not history or history[-1]['step'] != step): evaluate()
    save()
    if termination in ('cancel', 'deadline'):
        return {'binding': binding, 'phase': phase, 'state': 'interrupted',
                'termination': termination, 'completed_steps': step,
                'total_completed_steps': total_before+step,
                'total_compute_steps': compute_before+step, 'elapsed_seconds': spent()}
    # Each phase pins its own immutable selection, even if refine later changes
    # the convenience best.pt mirror. This makes base replay checks meaningful.
    atomic_torch(directory/(phase+'_best.pt'), selected)
    lock = {'binding': binding, 'phase': phase, 'completed_steps': step,
            'state': 'complete', 'total_completed_steps': total_before+step,
            'total_compute_steps': compute_before+step, 'elapsed_seconds': spent(), 'termination': termination,
            'best_development_deg': best, 'last_sha256': file_sha256(last),
            'best_sha256': file_sha256(directory/(phase+'_best.pt')), 'final_evaluated': False,
            'update_accounting': 'lineage total_steps; completed-phase compute_steps; crash-replayed unsaved updates excluded'}
    atomic_json(done, lock)
    del model, optimizer; gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return lock


def bootstrap_delta(delta, weights, seed=20261117):
    means = delta.mean(1)
    draws = np.random.default_rng(seed).integers(len(weights), size=(5000, len(weights)))
    return np.quantile((means[draws]*weights[draws]).sum(1)/weights[draws].sum(1), [.025, .975]).tolist()


def report(fit, dev, output, p, device, deadline):
    results, errors = {}, {}
    if tuple(p['model_order']) != MODELS:
        raise ValueError('Complete report requires all four prespecified arms')
    fit_small = fit.subset([u.user for u in fit.users[:p['fit_diagnostic_users']]])
    report_origins = None
    for name in p['model_order']:
        if time.time() >= deadline-60: raise TimeoutError('Report deadline reached')
        directory = output/name
        binding = {'protocol_sha256': digest(p), 'name': name, 'config': p['configs'][name], 'seed': p['seed']}
        checked_phase(directory, 'base', binding, device)
        _, state = checked_phase(directory, 'refine', binding, device)
        path = directory/'refine_best.pt'
        model = build_model(name, p['configs'][name], p['seed']).to(device)
        model.load_state_dict(state['model'])
        result, e, origins = score(model, dev, device, windows=p['report_windows'],
                                 batch_size=p['batch_size'], seed=p['report_seed'], keep_errors=True)
        if report_origins is None: report_origins = origins
        elif not np.array_equal(report_origins, origins):
            raise ValueError('Paired report origins differ between arms')
        fit_result = score(model, fit_small, device, windows=p['fit_diagnostic_windows'],
                           batch_size=p['batch_size'], seed=p['fit_diagnostic_seed'])
        result.update(checkpoint_sha256=file_sha256(path), selected_phase=state['phase'],
                      selected_step=state['step'], selected_total_steps=state['total_steps'],
                      selected_compute_steps=state['compute_steps'],
                      inference_parameters=sum(v.numel() for v in model.parameters()),
                      fit_subset_diagnostic=fit_result, fit_subset_users=len(fit_small.users),
                      fit_subset_not_population_estimate=True)
        results[name] = result; errors[name] = e
        # Static CPU inference artifact, exact source/config/state binding retained.
        atomic_torch(output/name/'inference.pt', {'name': name, 'config': p['configs'][name],
            'state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
            'protocol_sha256': digest(p), 'checkpoint_sha256': file_sha256(path)})
        atomic_json(output/'development_results.partial.json', results)
        del model; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    weights = population_weights(dev)
    np.savez_compressed(output/'development_errors.npz', **errors, origins=report_origins,
                        population_weights=weights, user_order=np.array([u.user for u in dev.users]))
    names = [n for n in p['model_order'] if n in results and n != 'tcn_control']
    winner = min(names, key=lambda n: results[n]['window_weighted_deg'])
    contrasts = {}
    if 'tcn_control' in errors:
        for name in names:
            delta = errors[name].astype('float64')-errors['tcn_control']
            contrasts[name] = {'mean_delta_deg': float(np.dot(weights, delta.mean(1))),
                               'paired_user_ci95': bootstrap_delta(delta, weights)}
    receipt = {'state': 'complete', 'schema': 'boxrr-v5-development-results', 'models': results,
        'winner_on_observed_development': winner, 'contrasts_to_scratch_control': contrasts,
        'protocol_sha256': digest(p), 'final_evaluated': False,
        'selection_adjusted': False, 'covers_training_seed_uncertainty': False,
        'all_three_below2': all(results[n]['window_weighted_deg'] < 2 for n in names),
        'any_below2': any(results[n]['window_weighted_deg'] < 2 for n in names)}
    atomic_json(output/'development_results.json', receipt)
    return receipt


def preflight(output, device):
    output.mkdir(parents=True, exist_ok=True)
    t = torch.arange(400, device=device, dtype=torch.float32)*.01
    x = geo.exp_so3(torch.stack((.5*t, .1*torch.sin(t*3), .05*t*t), -1))[None].repeat(4,1,1)
    target = x[:, -1:].repeat(1,20,1)
    records = {}
    for name in MODELS:
        model = build_model(name, {'width':64}).to(device)
        started = time.monotonic(); predicted = model(x)
        if predicted.shape != (4,20,4) or not torch.isfinite(predicted).all(): raise ValueError('Preflight output')
        if not torch.allclose(predicted.norm(dim=-1), torch.ones((4,20),device=device),atol=2e-4):
            raise ValueError('Preflight norm')
        error = geo.geodesic_radians(predicted,target).mean(); error.backward()
        grads = [v.grad for v in model.parameters() if v.requires_grad]
        if not all(g is not None and torch.isfinite(g).all() for g in grads):
            raise ValueError('Inactive or nonfinite parameter gradient: '+name)
        opt = torch.optim.AdamW(model.parameters(),lr=.001); opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            if geo.geodesic_radians(model(x),model(-x)).max() > 2e-5:
                raise ValueError('Antipode invariance failed')
        if device.type == 'cuda': torch.cuda.synchronize()
        records[name] = {'passed': True, 'parameters':sum(v.numel() for v in model.parameters()),
                         'forward_backward_seconds':time.monotonic()-started}
    value={'passed':True,'device':str(device),'software_fixture_only':True,'models':records,
           'torch':torch.__version__,'cuda':torch.version.cuda,'new_human_training_steps':0}
    atomic_json(output/'preflight.json',value); print(json.dumps(value),flush=True)
    return value


def study(root, output, device, deadline, *, expected_protocol=None):
    output.mkdir(parents=True, exist_ok=True)
    p = protocol()
    if expected_protocol is not None and read_json(expected_protocol) != p:
        raise ValueError('Expected launch protocol/source binding differs')
    if file_sha256(root/'manifest.json') != MANIFEST_SHA: raise ValueError('Frozen manifest changed')
    manifest = read_json(root/'manifest.json')
    for row in manifest['shards']:
        if row['partition']=='final' and (root/row['path']).exists(): raise ValueError('Final file present')
    fit_users = require_corpus(root,manifest,('fit',))
    dev_users = require_corpus(root,manifest,('inner','observed_dev'))
    if (len(fit_users),len(dev_users))!=(600,101):raise ValueError('Cohort count changed')
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=p:raise ValueError('Immutable protocol changed')
    if not path.exists():atomic_json(path,p)
    all_data = RamCorpus.load(root,manifest,[*fit_users,*dev_users])
    fit,dev=all_data.subset(fit_users),all_data.subset(dev_users)
    for phase in ('base','refine'):
        for name in p['model_order']:
            if time.time() >= deadline-900 or (output/'cancel').exists():
                atomic_json(output/'status.json',{'state':'stopped','reason':'deadline_or_cancel','phase':phase,'model':name})
                return
            phase_result = train_phase(name,p['configs'][name],fit,dev,output,p,device,phase=phase,deadline=deadline-600)
            if phase_result.get('state') != 'complete':
                atomic_json(output/'status.json', {'state': 'stopped', 'phase': phase, 'model': name,
                            'reason': phase_result['termination'], 'resumable': True})
                return
    result=report(fit,dev,output,p,device,deadline)
    atomic_json(output/'status.json',{'state':'complete','winner':result['winner_on_observed_development'],
                                    'updated_unix':time.time()})


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('preflight','train'))
    a.add_argument('--corpus',type=Path);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--expected-protocol',type=Path)
    a.add_argument('--device',default='cuda');a.add_argument('--deadline',type=float,default=0)
    args=a.parse_args();set_runtime();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA required')
    if args.action=='preflight':preflight(args.output,device)
    else:
        if not args.corpus or not args.deadline>time.time():raise ValueError('Corpus/deadline required')
        study(args.corpus,args.output,device,args.deadline,expected_protocol=args.expected_protocol)


if __name__=='__main__':main()

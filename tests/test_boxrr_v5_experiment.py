"""CPU synthetic fixtures for v5 recovery and scientific-protocol boundaries."""
import copy
import json
import math
import time

import numpy as np
import pytest
import torch

from abc_iueh import boxrr_v5_experiment as e
from abc_iueh.boxrr_experiment import MemoryUser, RamCorpus

CPU = torch.device('cpu')
torch.set_num_threads(1)


def corpus(lengths=(430, 450)):
    users = []
    for index, length in enumerate(lengths):
        t = torch.arange(length, dtype=torch.float32)
        q = e.geo.exp_so3(torch.stack((.002*(index+1)*t, .03*torch.sin(t/50), .001*t), -1))
        users.append(MemoryUser(f'fixture-{index}', [q.numpy()], np.array([length-419])))
    return RamCorpus(users)


class Toy(torch.nn.Module):
    """Tiny test model, deliberately stochastic in training to exercise RNG replay."""
    def __init__(self, stochastic=True, callback=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.02, -.01, .015]))
        self.register_buffer('updates_seen', torch.tensor(0))
        self.stochastic, self.callback = stochastic, callback

    def forward(self, x):
        noise = 0.
        if self.training:
            self.updates_seen.add_(1)
            if self.stochastic: noise = torch.rand_like(self.weight)*.001
            if self.callback is not None: self.callback()
        steps = torch.arange(1, 21, device=x.device, dtype=x.dtype)/20
        displacement = (self.weight+noise)[None, None]*steps[None, :, None]
        return e.geo.multiply_wxyz(x[:, -1:], e.geo.exp_so3(displacement))


def plan(**overrides):
    value = {'seed': 123, 'configs': {name: {} for name in e.MODELS},
        'model_order': list(e.MODELS), 'source_sha256': {'fixture': 'fixed'},
        'base_steps': 6, 'refine_steps': 3, 'base_seconds': 1000., 'refine_seconds': 1000.,
        'control_base_seconds': 1000., 'control_refine_seconds': 1000.,
        'base_lr': .007, 'refine_lr': .0023, 'weight_decay': .0001,
        'gradient_clip': 1., 'batch_size': 3, 'save_every': 2, 'eval_every': 2,
        'selection_windows': 2, 'selection_seed': 999,
        'report_windows': 5, 'report_seed': 1000,
        'fit_diagnostic_users': 1, 'fit_diagnostic_windows': 2, 'fit_diagnostic_seed': 1001}
    value.update(overrides)
    return value


@pytest.fixture
def toy_runtime(monkeypatch):
    def factory(name, config=None, seed=123):
        torch.manual_seed(seed)
        return Toy()
    def selection(model, *_args, **_kwargs):
        model.eval()
        value = 100.-int(model.updates_seen)
        return {'window_weighted_deg': value, 'user_balanced_deg': value}
    monkeypatch.setattr(e, 'build_model', factory)
    monkeypatch.setattr(e, 'score', selection)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: pytest.fail('CPU test initialized CUDA RNG'))


def run_phase(output, p, phase='base', name='direct'):
    output.mkdir(exist_ok=True)
    c = corpus()
    return e.train_phase(name, {}, c, c, output, p, CPU, phase=phase, deadline=time.time()+2000)


def state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b): assert_tree_equal(left, right)
    else:
        assert a == b


def test_population_sampler_replays_and_uses_exact_eligible_window_weights():
    c = corpus()
    sampler = e.PopulationSampler(c)
    np.testing.assert_allclose(sampler.weights, np.array([11, 31])/42)
    first = sampler.batch(np.random.default_rng(65), 19)
    second = sampler.batch(np.random.default_rng(65), 19)
    assert_tree_equal(first, second)
    reference = np.random.default_rng(65)
    who = reference.choice(2, size=19, p=np.array([11, 31])/42)
    for row, user_index in enumerate(who):
        user = c.users[user_index]
        start = int(reference.integers(int(user.cumulative[-1])))
        expected = torch.from_numpy(user.segments[0][start:start+420])
        assert torch.equal(torch.cat((first[0][row], first[1][row])), expected)


def test_score_uses_full_angle_endpoint_degrees_and_preserves_origins():
    c = corpus()
    angle = math.pi/6
    for user in c.users: user.segments[0][:] = [1., 0., 0., 0.]
    c.users[1].segments[0][:] = [math.cos(angle/2), 0., math.sin(angle/2), 0.]
    class Identity(torch.nn.Module):
        def forward(self, x):
            q = torch.zeros(len(x), 20, 4); q[..., 0] = 1.
            return q
    result, errors, origins = e.score(Identity(), c, CPU, windows=7, batch_size=3, seed=56, keep_errors=True)
    repeated = e.score(Identity(), c, CPU, windows=7, batch_size=7, seed=56, keep_errors=True)
    assert result['window_weighted_deg'] == pytest.approx(30*31/42, abs=2e-5)
    assert result['user_balanced_deg'] == pytest.approx(15., abs=2e-5)
    assert errors.shape == (2, 7)
    np.testing.assert_array_equal(origins, repeated[2])


def test_checkpoint_resume_replays_optimizer_rng_origins_history_and_selection(tmp_path, monkeypatch, toy_runtime):
    p = plan()
    sampled = []
    original_batch = e.PopulationSampler.batch
    def record_batch(self, rng, count):
        batch = original_batch(self, rng, count)
        sampled.append(tuple(value.clone() for value in batch))
        return batch
    monkeypatch.setattr(e.PopulationSampler, 'batch', record_batch)
    full = tmp_path/'full'
    run_phase(full, p)
    expected_batches = sampled.copy(); sampled.clear()
    original_save = e.atomic_torch
    crashed = False
    def fail_after_durable_last(path, value):
        nonlocal crashed
        original_save(path, value)
        if path.name == 'base_last.pt' and value['step'] == 2 and not crashed:
            crashed = True
            raise RuntimeError('injected after atomic last, before JSON/best mirrors')
    monkeypatch.setattr(e, 'atomic_torch', fail_after_durable_last)
    interrupted = tmp_path/'interrupted'
    with pytest.raises(RuntimeError, match='injected'):
        run_phase(interrupted, p)
    assert state(interrupted/'direct/base_last.pt')['step'] == 2
    # These non-authoritative mirrors may be stale, damaged, or ahead of last.pt.
    e.atomic_json(interrupted/'direct/base_history.json', [{'step': 999, 'weighted_deg': -10}])
    (interrupted/'direct/best.pt').write_bytes(b'damaged mirror')
    run_phase(interrupted, p)
    actual, expected = state(interrupted/'direct/base_last.pt'), state(full/'direct/base_last.pt')
    for key in ('model', 'optimizer', 'rng', 'step', 'best', 'total_steps', 'compute_steps'):
        assert_tree_equal(actual[key], expected[key])
    for actual_row, expected_row in zip(actual['history'], expected['history'], strict=True):
        assert {k: v for k, v in actual_row.items() if k != 'elapsed_seconds'} == {
            k: v for k, v in expected_row.items() if k != 'elapsed_seconds'}
    assert_tree_equal(actual['selected_best']['model'], expected['selected_best']['model'])
    assert_tree_equal(sampled, expected_batches)
    assert e.read_json(interrupted/'direct/base_history.json') == actual['history']


@pytest.mark.parametrize('reason', ['cancel', 'deadline'])
def test_cancel_and_deadline_are_resumable_and_do_not_lock_partial_phase(tmp_path, monkeypatch, toy_runtime, reason):
    p = plan()
    trigger = [True]
    original_time = time.time
    deadline = original_time()+2000
    def stop_once():
        if trigger[0]:
            trigger[0] = False
            if reason == 'cancel': (tmp_path/'cancel').write_text('fixture')
            else: monkeypatch.setattr(e.time, 'time', lambda: deadline-100)
    def factory(name, config=None, seed=123):
        torch.manual_seed(seed)
        return Toy(callback=stop_once)
    monkeypatch.setattr(e, 'build_model', factory)
    c = corpus()
    result = e.train_phase('direct', {}, c, c, tmp_path, p, CPU, phase='base', deadline=deadline)
    assert result['state'] == 'interrupted' and result['termination'] == reason
    assert result['completed_steps'] == 1
    assert not (tmp_path/'direct/base_lock.json').exists()
    if reason == 'cancel': (tmp_path/'cancel').unlink()
    monkeypatch.setattr(e.time, 'time', original_time)
    resumed = run_phase(tmp_path, p)
    assert resumed['state'] == 'complete' and resumed['completed_steps'] == 6


def test_refine_distinguishes_selected_lineage_from_total_phase_compute(tmp_path, monkeypatch, toy_runtime):
    p = plan()
    def selected_early(model, *_args, **_kwargs):
        model.eval()
        value = 1.+abs(int(model.updates_seen)-2)
        return {'window_weighted_deg': value, 'user_balanced_deg': value}
    monkeypatch.setattr(e, 'score', selected_early)
    base = run_phase(tmp_path, p)
    assert base['completed_steps'] == 6
    assert state(tmp_path/'direct/base_best.pt')['total_steps'] == 2
    refined = run_phase(tmp_path, p, 'refine')
    last = state(tmp_path/'direct/refine_last.pt')
    assert refined['total_completed_steps'] == last['total_steps'] == 5
    assert refined['total_compute_steps'] == last['compute_steps'] == 9
    assert all(int(v['step']) == 3 for v in last['optimizer']['state'].values())
    assert state(tmp_path/'direct/refine_best.pt')['phase'] == 'base'


def test_protocol_learning_rates_control_actual_optimizer_schedule(tmp_path, toy_runtime):
    p = plan(base_steps=1, refine_steps=1)
    run_phase(tmp_path, p)
    assert state(tmp_path/'direct/base_last.pt')['optimizer']['param_groups'][0]['lr'] == p['base_lr']
    run_phase(tmp_path, p, 'refine')
    assert state(tmp_path/'direct/refine_last.pt')['optimizer']['param_groups'][0]['lr'] == p['refine_lr']


def test_conservative_phase_clock_counts_resume_downtime(tmp_path, toy_runtime):
    p = plan(base_seconds=30.)
    directory = tmp_path/'direct'; directory.mkdir()
    binding = {'protocol_sha256': e.digest(p), 'name': 'direct', 'config': {}, 'seed': p['seed']}
    e.atomic_json(directory/'base_clock.json', {'binding': binding, 'phase': 'base', 'started_unix': time.time()-60})
    result = run_phase(tmp_path, p)
    assert result['termination'] == 'time_cap'
    assert result['completed_steps'] == 0
    assert result['elapsed_seconds'] >= 60


def test_immutable_phase_selection_survives_refinement_and_detects_tampering(tmp_path, toy_runtime):
    p = plan()
    original = run_phase(tmp_path, p)
    original_sha = e.file_sha256(tmp_path/'direct/base_best.pt')
    run_phase(tmp_path, p, 'refine')
    assert e.file_sha256(tmp_path/'direct/base_best.pt') == original_sha
    assert state(tmp_path/'direct/best.pt')['phase'] == 'refine'
    assert run_phase(tmp_path, p) == original
    (tmp_path/'direct/base_best.pt').write_bytes(b'changed immutable selection')
    with pytest.raises(ValueError, match='checksum'):
        run_phase(tmp_path, p)


@pytest.mark.parametrize('field, value', [('name', 'foreign'), ('config', {'width': 32}),
                                        ('seed', 99), ('protocol_sha256', 'changed-source')])
def test_checked_checkpoint_rejects_every_binding_component(tmp_path, toy_runtime, field, value):
    p = plan(base_steps=1)
    run_phase(tmp_path, p)
    path = tmp_path/'direct/base_last.pt'
    saved = state(path); expected = copy.deepcopy(saved['binding'])
    saved['binding'][field] = value
    e.atomic_torch(path, saved)
    with pytest.raises(ValueError, match='binding'):
        e.checked_checkpoint(path, expected, CPU, sha256=e.file_sha256(path), phase='base')


def test_preflight_accepts_equivalent_antipodal_output_signs(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'build_model', lambda *_args, **_kwargs: Toy(stochastic=False))
    value = e.preflight(tmp_path, CPU)
    assert value['passed'] and set(value['models']) == set(e.MODELS)
    assert value['new_human_training_steps'] == 0


def test_report_requires_all_four_completed_arms(tmp_path):
    c = corpus()
    with pytest.raises(FileNotFoundError):
        e.report(c, c, tmp_path, plan(), CPU, time.time()+2000)
    assert not (tmp_path/'development_results.json').exists()


def test_complete_report_uses_immutable_bound_selection_and_paired_origins(tmp_path, monkeypatch):
    p = plan(base_steps=2, refine_steps=1, save_every=1, eval_every=1)
    def factory(name, config=None, seed=123):
        torch.manual_seed(seed)
        return Toy()
    monkeypatch.setattr(e, 'build_model', factory)
    c = corpus()
    for name in e.MODELS:
        run_phase(tmp_path, p, name=name)
        run_phase(tmp_path, p, phase='refine', name=name)
        (tmp_path/name/'best.pt').write_bytes(b'untrusted convenience mirror')
    receipt = e.report(c, c, tmp_path, p, CPU, time.time()+2000)
    assert receipt['state'] == 'complete' and set(receipt['models']) == set(e.MODELS)
    assert receipt['final_evaluated'] is False
    assert all(v['mean_delta_deg'] == 0 for v in receipt['contrasts_to_scratch_control'].values())
    with np.load(tmp_path/'development_errors.npz') as stored:
        assert stored['origins'].shape == (2, p['report_windows'], 3)
        for name in e.MODELS:
            assert stored[name].shape == (2, p['report_windows'])
            assert receipt['models'][name]['checkpoint_sha256'] == e.file_sha256(tmp_path/name/'refine_best.pt')


def test_frozen_final_presence_rejected_before_loading_any_array(tmp_path, monkeypatch):
    final = tmp_path/'final.fixture'; final.write_bytes(b'synthetic, never a recording')
    (tmp_path/'manifest.json').write_text(json.dumps({'shards': [{'partition': 'final', 'path': final.name}]}))
    monkeypatch.setattr(e, 'file_sha256', lambda _: e.MANIFEST_SHA)
    monkeypatch.setattr(e, 'require_corpus', lambda *_: pytest.fail('Must stop before cohort loading'))
    with pytest.raises(ValueError, match='Final file present'):
        e.study(tmp_path, tmp_path/'output', CPU, time.time()+2000)


def test_study_stops_after_interrupted_base_without_refine_or_report(tmp_path, monkeypatch):
    root = tmp_path/'corpus'; root.mkdir()
    (root/'manifest.json').write_text(json.dumps({'shards': []}))
    monkeypatch.setattr(e, 'file_sha256', lambda _: e.MANIFEST_SHA)
    monkeypatch.setattr(e, 'require_corpus', lambda _root, _manifest, parts: list(range(600 if parts == ('fit',) else 101)))
    monkeypatch.setattr(e, 'protocol', plan)
    class Loaded:
        def subset(self, _users): return corpus()
    monkeypatch.setattr(e.RamCorpus, 'load', lambda *_: Loaded())
    calls = []
    def interrupted(name, *_args, **kwargs):
        calls.append((name, kwargs['phase']))
        return {'state': 'interrupted', 'termination': 'deadline'}
    monkeypatch.setattr(e, 'train_phase', interrupted)
    monkeypatch.setattr(e, 'report', lambda *_: pytest.fail('Incomplete study cannot report complete'))
    output = tmp_path/'output'
    expected = tmp_path/'expected_protocol.json'; e.atomic_json(expected, plan())
    e.study(root, output, CPU, time.time()+2000, expected_protocol=expected)
    assert calls == [('direct', 'base')]
    assert e.read_json(output/'status.json')['resumable'] is True


def test_expected_launch_source_binding_rejected_before_corpus_io(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'protocol', plan)
    expected = plan(); expected['source_sha256']['fixture'] = 'changed-after-launch-lock'
    path = tmp_path/'expected.json'; e.atomic_json(path, expected)
    monkeypatch.setattr(e, 'file_sha256', lambda _: pytest.fail('Corpus I/O before launch binding check'))
    monkeypatch.setattr(e.RamCorpus, 'load', lambda *_: pytest.fail('Array loading before launch binding check'))
    with pytest.raises(ValueError, match='launch protocol/source binding'):
        e.study(tmp_path/'absent-corpus', tmp_path/'output', CPU, time.time()+2000, expected_protocol=path)


def test_source_and_protocol_bindings_cover_all_training_models():
    p = e.protocol()
    expected = {'boxrr_v5_experiment.py', 'boxrr_v5_direct.py', 'boxrr_v5_cde.py', 'boxrr_v5_imm.py',
                'boxrr_models.py', 'boxrr_v2_deterministic.py', 'boxrr_v2_experiment.py',
                'boxrr_experiment.py', 'boxrr_source.py'}
    assert set(p['source_sha256']) == expected
    assert all(len(value) == 64 for value in p['source_sha256'].values())
    changed = copy.deepcopy(p); changed['source_sha256']['boxrr_v5_direct.py'] = '0'*64
    assert e.digest(changed) != e.digest(p)
    assert p['final_policy'] == 'never_upload_or_evaluate'
    assert tuple(p['model_order']) == e.MODELS

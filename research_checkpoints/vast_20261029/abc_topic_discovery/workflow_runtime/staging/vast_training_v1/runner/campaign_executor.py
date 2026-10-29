"""Run the locked screen sequentially; each fit is independently budget-gated.

This executes real work when called. It does not grant a lease or data access.
An interrupted or failed attempt is never retried automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from .campaign_plan import build_plan
from .checkpoint import ensure_private_directory, sha256_file
from .screen_plan import UPDATES_PER_FIT
from .trainer import THREE_ARM_DRAFT_SHA256
from . import lease_clock
from ..ops_v2.checkpoint_store import verify_copy
from ..ops_v2 import remote_backup
from ..time_budget.ledger import _acquire, _unlock

MAX_LOG_BYTES = 8 * 1024 * 1024


class CampaignStop(RuntimeError):
    """Campaign cannot safely proceed without operator reconciliation."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sync_dir(path: Path):
    if os.name == 'posix':
        fd = os.open(str(path), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def _new_json(path: Path, obj: dict):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(_canonical(obj) + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    _sync_dir(path.parent)


def _read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOG_BYTES:
        raise CampaignStop('missing, unsafe or oversized campaign record')
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise CampaignStop('invalid campaign record') from exc


def _json_lines(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOG_BYTES:
        raise CampaignStop('missing, unsafe or oversized launcher output')
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try: value = json.loads(line)
        except ValueError: continue
        if type(value) is dict: records.append(value)
    return records


def _verify_outcome(stdout: Path, fit: dict, config: dict):
    records = _json_lines(stdout)
    fits = [x for x in records if 'fit' in x]
    supervisors = [x for x in records if 'time_supervisor' in x]
    if len(fits) != 1 or len(supervisors) != 1:
        raise CampaignStop('launcher did not emit exactly one fit and supervisor result')
    entry, supervisor = fits[0], supervisors[0].get('time_supervisor')
    report = entry.get('fit')
    if type(report) is not dict or type(supervisor) is not dict:
        raise CampaignStop('launcher result is malformed')
    if (supervisor.get('exit_code') != 0 or supervisor.get('timed_out') is not False
        or supervisor.get('orphaned') is not False or supervisor.get('external_stop_signal') is not None
        or supervisor.get('same_process_group_empty_verified') is not True):
        raise CampaignStop('supervisor did not report a clean finished process group')
    expected = {'purpose': 'ordered_motion_three_arm_screen', 'role': 'outer_fit10',
                'fold_id': fit['fold_id'], 'seed': fit['training_seed'], 'arm': fit['arm_id'],
                'completed_updates': UPDATES_PER_FIT,
                'data_identity_sha256': config['bundles'][str(fit['fold_id'])]['identity_sha256'],
                'model_sha256': config['model_sha256'][fit['arm_id']],
                'run_id': config['run_ids'][fit['fit']]}
    if any(report.get(k) != v for k, v in expected.items()):
        raise CampaignStop('fit identity or completed update count does not match the locked plan')
    checkpoint_hash = report.get('checkpoint_sha256')
    backups = entry.get('verified_backups')
    if type(checkpoint_hash) is not str or len(checkpoint_hash) != 64 or type(backups) is not list or not backups:
        raise CampaignStop('final checkpoint backup is absent')
    final = backups[-1]
    if type(final) is not dict or final.get('step') != UPDATES_PER_FIT or final.get('payload_sha256') != checkpoint_hash:
        raise CampaignStop('final backup does not match final checkpoint')
    checkpoint_receipts = [x['checkpoint_backup_verified'] for x in records if 'checkpoint_backup_verified' in x]
    if not checkpoint_receipts or checkpoint_receipts[-1] != final:
        raise CampaignStop('final backup lacks matching local controller receipt')
    expected_meta = {'fold_id': fit['fold_id'], 'training_seed': fit['training_seed'],
                     'arm': fit['arm_id'], 'global_step': UPDATES_PER_FIT,
                     'total_steps': UPDATES_PER_FIT, 'data_sha256': expected['data_identity_sha256'],
                     'model_sha256': expected['model_sha256'], 'code_sha256': config['runner_sha256'],
                     'payload_sha256': checkpoint_hash}
    def verify_metadata(manifest_path, manifest):
        meta_path = manifest_path.parent / manifest['metadata_file']
        if meta_path.is_symlink() or not meta_path.is_file() or meta_path.stat().st_size > MAX_LOG_BYTES:
            raise CampaignStop('backup metadata is missing or unsafe')
        meta_bytes = meta_path.read_bytes()
        if hashlib.sha256(meta_bytes).hexdigest() != manifest.get('metadata_sha256'):
            raise CampaignStop('backup metadata changed after verification')
        meta = json.loads(meta_bytes)
        if manifest.get('payload_sha256') != checkpoint_hash or any(meta.get(k) != v for k, v in expected_meta.items()):
            raise CampaignStop('verified backup metadata differs from fit or code pins')

    if 'backup_rclone_remote' in config:
        # Pins originate from the local controller's pre-upload checkpoint
        # record, not from remote metadata. This is an in-process consistency
        # binding, not an independent signed off-host authority.
        run_id = config['run_ids'][fit['fit']]
        pins = {'run_id': run_id, 'plan_sha256': THREE_ARM_DRAFT_SHA256,
                'data_sha256': expected['data_identity_sha256'], 'code_sha256': config['runner_sha256'],
                'model_sha256': expected['model_sha256'], 'fold_id': fit['fold_id'],
                'training_seed': fit['training_seed'], 'arm': fit['arm_id']}
        if type(run_id) is not str or not re.fullmatch(r'[0-9a-f]{32}', run_id):
            raise CampaignStop('local controller run ID is missing')
        if any(final.get(k) != v for k, v in pins.items()):
            raise CampaignStop('local controller checkpoint pins do not match the plan')
        prefix = remote_backup._remote_prefix(config['backup_rclone_remote'])
        expected_remote = f'{prefix}/checkpoint-{checkpoint_hash}.manifest.json'
        if final.get('manifest_remote') != expected_remote:
            raise CampaignStop('remote manifest is outside the pinned prefix')
        with tempfile.TemporaryDirectory(prefix='abc-campaign-verify-') as temp:
            root = Path(temp); expected_path = root / 'expected.json'
            _new_json(expected_path, {**pins, 'expected_global_step': UPDATES_PER_FIT})
            restored = remote_backup.restore_checkpoint(prefix, root / 'restored',
                manifest_sha256=final['manifest_sha256'], payload_sha256=checkpoint_hash,
                expected_path=expected_path)
            manifest_path = Path(restored['manifest_path'])
            manifest = remote_backup.verify_remote_copy(prefix, manifest_path, final['manifest_sha256'])
            verify_metadata(manifest_path, manifest)
    else:
        manifest_path = Path(final.get('manifest_path', ''))
        if not manifest_path.is_absolute() or manifest_path.is_symlink():
            raise CampaignStop('invalid backup manifest path')
        backup_root = Path(config['backup_dir']).resolve(strict=True)
        if manifest_path.resolve(strict=True).parent != backup_root:
            raise CampaignStop('backup manifest is outside configured destination')
        if sha256_file(manifest_path) != final.get('manifest_sha256'):
            raise CampaignStop('backup manifest hash differs from launcher output')
        manifest = verify_copy(manifest_path)
        verify_metadata(manifest_path, manifest)
    return {'checkpoint_sha256': checkpoint_hash, 'manifest_sha256': final['manifest_sha256']}


def _check_python(python):
    python = Path(python)
    if (not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK)
            or not (python.parent.parent / 'pyvenv.cfg').is_file()):
        raise CampaignStop('configured virtual environment Python is missing or not executable')


def execute_campaign(config: dict, state_dir: Path, *, continue_verified: bool = False) -> dict:
    """Run in frozen order, or skip only individually reverified prior successes."""
    plan = build_plan(config)
    _check_python(config['python_executable'])
    config_hash = _digest(config)
    root = ensure_private_directory(Path(state_dir))
    lock = _acquire(str(root / '.campaign.lock'))
    try:
        record_path = root / 'campaign.json'
        clock_path = root / 'lease-clock.jsonl'
        identity_base = {'schema': 'abc-screen-campaign-v2', 'config_sha256': config_hash,
                         'fit_ids': [row['fit'] for row in plan['fits']]}
        if record_path.exists():
            identity = _read_json(record_path)
            if (not continue_verified or type(identity) is not dict
                    or {k: identity.get(k) for k in identity_base} != identity_base
                    or set(identity) != set(identity_base) | {'lease_clock_genesis'}):
                raise CampaignStop('existing campaign requires matching config and --continue-verified')
            genesis = identity['lease_clock_genesis']
        else:
            if continue_verified:
                raise CampaignStop('cannot continue a campaign without its original record')
            # Refuse an existing directory containing unknown files before initialising.
            if any(p.name != '.campaign.lock' for p in root.iterdir()):
                raise CampaignStop('new campaign state directory must be empty')
            genesis = lease_clock.create(clock_path, _read_json(Path(config['ledger'])))
            identity = {**identity_base, 'lease_clock_genesis': genesis}
            _new_json(record_path, identity)
        # A later receipt after a missing earlier fit is evidence of a gap, not
        # permission to rerun the missing fit. Check the whole sequence before
        # launching any further work.
        missing = False
        expected_clock_head = genesis
        for row in plan['fits']:
            name = row['fit']
            paths = [root / (name + suffix) for suffix in ('.started.json', '.finished.json', '.stdout', '.stderr')]
            exists = any(p.exists() or p.is_symlink() for p in paths)
            if missing and exists:
                raise CampaignStop('campaign receipts are not a contiguous prefix; reconcile before continuing')
            if exists:
                if not paths[0].is_file() or not paths[1].is_file():
                    raise CampaignStop('incomplete fit attempt requires reconciliation')
                prior = _read_json(paths[1])
                if type(prior) is not dict or prior.get('status') != 'complete' or not isinstance(prior.get('lease_clock_head'), str):
                    raise CampaignStop('failed or unanchored fit requires reconciliation')
                expected_clock_head = prior['lease_clock_head']
            if not exists:
                missing = True
        lease_clock.check(clock_path, genesis, expected_head=expected_clock_head)
        completed = 0
        for fit in plan['fits']:
            name = fit['fit']
            argv = build_plan(config, fit_id=name)['selected_command']['argv']
            argv += ['--shared-lease-clock', str(clock_path), '--lease-clock-genesis', genesis]
            started, finished = root / (name + '.started.json'), root / (name + '.finished.json')
            stdout_path, stderr_path = root / (name + '.stdout'), root / (name + '.stderr')
            command_sha = _digest(argv)
            start_record = {'schema': 'abc-screen-fit-start-v1', 'fit': name,
                            'config_sha256': config_hash, 'argv_sha256': command_sha}
            if any(p.exists() or p.is_symlink() for p in (started, finished, stdout_path, stderr_path)):
                if not continue_verified or not started.is_file() or not finished.is_file():
                    raise CampaignStop(f'{name}: incomplete or existing attempt requires reconciliation')
                done = _read_json(finished)
                if (_read_json(started) != start_record or type(done) is not dict
                    or done.get('schema') != 'abc-screen-fit-finish-v1' or done.get('fit') != name
                    or done.get('config_sha256') != config_hash or done.get('argv_sha256') != command_sha
                    or done.get('exit_code') != 0 or done.get('status') != 'complete'
                    or stdout_path.is_symlink() or stderr_path.is_symlink()
                    or not stdout_path.is_file() or not stderr_path.is_file()
                    or sha256_file(stdout_path) != done.get('stdout_sha256')
                    or sha256_file(stderr_path) != done.get('stderr_sha256')):
                    raise CampaignStop(f'{name}: prior receipt or logs do not verify')
                if _verify_outcome(stdout_path, fit, config) != done.get('verified_backup'):
                    raise CampaignStop(f'{name}: prior backup changed')
                completed += 1
                continue
            _new_json(started, start_record)
            # Exclusive logs and started marker exist before launching; a crash is
            # never mistaken for a never-started fit, even if no result is written.
            with stdout_path.open('xb') as out, stderr_path.open('xb') as err:
                try:
                    proc = subprocess.run(argv, cwd=str(Path(__file__).resolve().parents[2]),
                                          stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False)
                    exit_code = proc.returncode
                finally:
                    out.flush(); os.fsync(out.fileno()); err.flush(); os.fsync(err.fileno())
            lease_clock.evaluate(clock_path, genesis, _read_json(Path(config['ledger'])), mark=True)
            clock_head = lease_clock.check(clock_path, genesis)
            verified = None
            status = 'failed'
            if exit_code == 0:
                try:
                    verified = _verify_outcome(stdout_path, fit, config)
                    status = 'complete'
                except (CampaignStop, OSError, ValueError, KeyError) as exc:
                    reason = str(exc)
            else:
                reason = f'launcher exited {exit_code}'
            finish = {'schema': 'abc-screen-fit-finish-v1', 'fit': name,
                      'config_sha256': config_hash, 'argv_sha256': command_sha,
                      'exit_code': exit_code, 'status': status, 'stdout_sha256': sha256_file(stdout_path),
                      'stderr_sha256': sha256_file(stderr_path), 'verified_backup': verified,
                      'lease_clock_head': clock_head}
            if status != 'complete': finish['reason'] = reason
            _new_json(finished, finish)
            if status != 'complete':
                raise CampaignStop(f'{name}: {reason}; do not retry without reconciliation')
            completed += 1
        return {'status': 'complete', 'fits_verified': completed, 'config_sha256': config_hash}
    finally:
        _unlock(lock); lock.close()


def request_instance_stop(instance_id: str) -> dict:
    """Request a provider stop; CLI success is not proof of stopped billing."""
    if type(instance_id) is not str or not re.fullmatch(r'[1-9][0-9]{0,19}', instance_id):
        raise ValueError('instance ID must be a positive decimal identifier')
    try:
        result = subprocess.run(['vastai', 'stop', 'instance', instance_id], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=30)
        return {'stop_requested': True, 'cli_accepted': result.returncode == 0,
                'provider_stop_verified': False, 'stopped_storage_may_still_be_billed': True}
    except (OSError, subprocess.TimeoutExpired):
        return {'stop_requested': False, 'cli_accepted': False,
                'provider_stop_verified': False, 'stopped_storage_may_still_be_billed': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('state_dir', type=Path, help='private campaign receipts directory')
    parser.add_argument('--continue-verified', action='store_true',
                        help='reverify and skip completed fit receipts; never retry a failed attempt')
    parser.add_argument('--stop-instance-id', help='request Vast instance stop on campaign exit; decimal ID only')
    args = parser.parse_args(argv)
    if args.stop_instance_id is not None and not re.fullmatch(r'[1-9][0-9]{0,19}', args.stop_instance_id):
        parser.error('stop-instance-id must be a positive decimal identifier')
    exit_code = 0
    try:
        result = execute_campaign(json.loads(args.config.read_text(encoding='utf-8')),
                                  args.state_dir, continue_verified=args.continue_verified)
    except (CampaignStop, OSError, ValueError) as exc:
        print('CAMPAIGN STOP: ' + str(exc), flush=True)
        exit_code = 2
    except KeyboardInterrupt:
        print('CAMPAIGN STOP: interrupted; reconcile the active fit', flush=True)
        exit_code = 130
    except Exception as exc:
        print('CAMPAIGN STOP: unexpected ' + type(exc).__name__, flush=True)
        exit_code = 2
    else:
        print(json.dumps(result, sort_keys=True))
    if args.stop_instance_id is not None:
        stopped = request_instance_stop(args.stop_instance_id)
        print(json.dumps({'instance_stop': stopped}, sort_keys=True), flush=True)
        if not stopped['cli_accepted']:
            print('STOP REQUIRED: verify and stop the instance manually; storage can still incur charges.', flush=True)
            exit_code = exit_code or 2
    return exit_code


if __name__ == '__main__': raise SystemExit(main())

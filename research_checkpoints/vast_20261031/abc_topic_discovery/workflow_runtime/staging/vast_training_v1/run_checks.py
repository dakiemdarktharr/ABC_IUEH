"""Verify a copied runtime and exercise only generated fixtures on CPU.

Run from the runtime's staging directory:
python -m vast_training_v1.run_checks
No lease, network call or human recording is used by this entry point.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


def verify_manifest(root: Path):
    manifest = json.loads((root / 'code-only-manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'abc-vast-code-only-v1'
            or manifest.get('human_data_included') is not False
            or manifest.get('bundle_included') is not False
            or type(manifest.get('files')) is not dict or not manifest['files']):
        raise ValueError('invalid code-only manifest')
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('unsafe manifest path')
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('manifest path escapes the runtime root')
        if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
            raise ValueError('missing or linked runtime component: ' + name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('runtime bytes differ from manifest: ' + name)
        if path.suffix == '.py': ast.parse(data, filename=name)
    return len(manifest['files'])


def main():
    runtime = Path(__file__).resolve().parent
    staging, root = runtime.parent, runtime.parents[3]
    count = verify_manifest(root)
    env = dict(os.environ)
    paths = [staging, runtime / 'runner', runtime / 'ops_v2', runtime / 'data',
             staging / 'ordered_motion_model_v3']
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths))
    env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='abc-vast-generated-') as temp:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(runtime),
            str(staging / 'ordered_motion_model_v3' / 'test_ordered_motion_model.py'),
            '--basetemp', str(Path(temp) / 'pytest')], cwd=staging, env=env, check=False)
    print(json.dumps({'manifest_files_verified': count, 'synthetic_test_exit_code': result.returncode,
                      'human_data_accessed': False, 'paid_run_authorized': False}), flush=True)
    return result.returncode


if __name__ == '__main__': raise SystemExit(main())

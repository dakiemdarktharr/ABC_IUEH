"""Bounded two-stage prediction freeze and target scoring for synthetic sources.

The actual HDF5 validation reader is deliberately disabled. Provider callbacks
must be independently bound to the fixed validation origin map before real use.
This module does not create authorization or open any human data source.
"""
from __future__ import annotations
import hashlib
import heapq
import json
import math
import os
import shutil
from pathlib import Path
import struct
import tempfile

CELLS = tuple(f'{a}:{s}' for a in ('ordered_product','active_order_control','contemporaneous_raw_vector')
              for s in (20261026,20261027))
_RECORD = struct.Struct('<BQ24f')
_FLOAT = struct.Struct('<d')
_ORIGIN = struct.Struct('<BQ')
MAX_SORT_VALUES = 8192
MAX_FOLD_ORIGINS = 2_000_000


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _available_ram_bytes():
    if os.name == 'nt':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),
                      ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
                      ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
                      ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),
                      ('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
        state=Memory(); state.dwLength=ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)): return None
        return int(state.ullAvailPhys)
    # The shared implementation considers nested cgroup v1/v2 limits, not only
    # host RAM or the root cgroup. A failed probe must fail closed.
    try:
        from ..runner.trainer import available_physical_ram_bytes
        return available_physical_ram_bytes()
    except (OSError, ValueError, RuntimeError):
        return None


def _check_ram(ram_guard):
    free = _available_ram_bytes()
    if free is None or free < 2 * 1024**3:
        raise MemoryError('validation requires at least 2 GiB available RAM')
    ram_guard()


def _scope(source):
    counts = tuple(source.expected_origin_counts)
    digest = source.origin_identity_sha256
    transcript = source.origin_stream_sha256
    for value in (source.source_asset_sha256, source.split_sha256):
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('source asset and split must have SHA-256 identities')
    if (source.role != 'outer_validation5' or type(source.fold_id) is not int or source.fold_id not in (0,1,2)
        or len(counts) != 5 or any(type(n) is not int or n <= 0 for n in counts)
        or type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
        or type(transcript) is not str or len(transcript) != 64 or any(c not in '0123456789abcdef' for c in transcript)):
        raise ValueError('invalid validation scope')
    if sum(counts) > MAX_FOLD_ORIGINS:
        raise ValueError('full fold exceeds bounded exact-quantile capacity; do not subsample')
    return counts, digest, transcript


def _quaternion(q):
    if len(q) != 4 or not all(math.isfinite(float(x)) for x in q): raise ValueError('invalid quaternion')
    norm = math.sqrt(sum(float(x)*float(x) for x in q))
    if abs(norm-1) > .002 or norm < 1e-8: raise ValueError('quaternion is not unit norm')
    return tuple(float(x)/norm for x in q)


def _angle(p, q):
    p, q = _quaternion(p), _quaternion(q)
    scalar = abs(sum(a*b for a,b in zip(p,q)))
    x = q[0]*p[1]-p[0]*q[1] - (q[2]*p[3]-q[3]*p[2])
    y = q[0]*p[2]-p[0]*q[2] - (q[3]*p[1]-q[1]*p[3])
    z = q[0]*p[3]-p[0]*q[3] - (q[1]*p[2]-q[2]*p[1])
    return 2 * math.degrees(math.atan2(math.sqrt(x*x+y*y+z*z), scalar))


def freeze_predictions(*, source, predict, prediction_path, lock_path, checkpoint_hashes, predictor_code_sha256, ram_guard):
    """Write all six predictions per origin; never request any target here.

    ``predict(context_bytes)`` maps each fixed cell to its endpoint20 quaternion.
    Output file is exclusive/private and remains untrusted until this function
    returns a lock containing its SHA-256, source identity and exact cell pins.
    """
    counts, origin_hash, expected_origins = _scope(source)
    if set(checkpoint_hashes) != set(CELLS) or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in checkpoint_hashes.values()):
        raise ValueError('all six independently verified checkpoint digests required')
    if type(predictor_code_sha256) is not str or len(predictor_code_sha256) != 64 or any(c not in '0123456789abcdef' for c in predictor_code_sha256):
        raise ValueError('a pinned predictor code digest is required')
    path = Path(prediction_path)
    digest, origins, actual, previous = hashlib.sha256(), hashlib.sha256(), [0]*5, [-1]*5
    seen_user = -1
    _check_ram(ram_guard)
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as out:
        for index, (user, origin, context) in enumerate(source.iter_contexts()):
            if index % 128 == 0: _check_ram(ram_guard)
            if (type(user) is not int or not 0 <= user < 5 or user < seen_user or
                type(origin) is not int or not 0 <= origin < 2**64 or origin <= previous[user] or
                type(context) is not bytes or len(context) != 400*4*4):
                raise ValueError('invalid canonical validation context stream')
            predictions = predict(context)
            if set(predictions) != set(CELLS): raise ValueError('prediction cells differ from frozen screen')
            flat = [v for cell in CELLS for v in _quaternion(predictions[cell])]
            record = _RECORD.pack(user, origin, *flat)
            out.write(record); digest.update(record); origins.update(_ORIGIN.pack(user, origin))
            seen_user, previous[user] = user, origin
            actual[user] += 1
        if tuple(actual) != counts or origins.hexdigest() != expected_origins:
            raise ValueError('context origin stream differs from fixed scope')
        out.flush(); os.fsync(out.fileno())
    os.chmod(path, 0o400)
    lock = {'kind':'validation_prediction_lock_v1', 'fold_id':source.fold_id, 'role':source.role,
            'origin_identity_sha256':origin_hash, 'origin_counts':list(counts),
            'source_asset_sha256':source.source_asset_sha256, 'split_sha256':source.split_sha256,
            'predictor_code_sha256':predictor_code_sha256,
            'metric_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cells':list(CELLS), 'checkpoint_sha256':dict(checkpoint_hashes),
            'record_count':sum(counts), 'record_bytes':_RECORD.size,
            'predictions_sha256':digest.hexdigest(), 'origin_stream_sha256':origins.hexdigest()}
    raw = _canonical(lock) + b'\n'
    fd = os.open(Path(lock_path), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.chmod(lock_path, 0o400)
    return hashlib.sha256(raw).hexdigest()


def _percentiles(path, count, scratch, *, ram_guard):
    """Exact NumPy-linear P95/P99 by bounded sorted runs and external merge."""
    if count <= 0: raise ValueError('empty quantile input')
    runs=[]
    try:
        with path.open('rb') as src:
            while True:
                _check_ram(ram_guard)
                block=src.read(MAX_SORT_VALUES * _FLOAT.size)
                if not block: break
                if len(block)%_FLOAT.size: raise ValueError('invalid error spool length')
                values=sorted(item[0] for item in struct.iter_unpack('<d',block))
                fd, name=tempfile.mkstemp(prefix='run-',dir=scratch)
                with os.fdopen(fd,'wb') as out:
                    for value in values: out.write(_FLOAT.pack(value))
                runs.append(Path(name))
        locations=[]
        for quantile in (.95,.99):
            position=(count-1)*quantile
            locations.extend((math.floor(position),math.ceil(position)))
        needed=set(locations); found={}
        handles=[p.open('rb') for p in runs]
        try:
            def values(handle):
                while True:
                    b=handle.read(_FLOAT.size)
                    if not b: return
                    if len(b)!=_FLOAT.size: raise ValueError('truncated sorted run')
                    yield _FLOAT.unpack(b)[0]
            for index,value in enumerate(heapq.merge(*(values(h) for h in handles))):
                if index % MAX_SORT_VALUES == 0: _check_ram(ram_guard)
                if index in needed: found[index]=value
                if index >= max(needed): break
        finally:
            for handle in handles: handle.close()
        if len(found)!=len(needed): raise ValueError('quantile spool is incomplete')
        result=[]
        for q in (.95,.99):
            pos=(count-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
            result.append(found[lo] + (found[hi]-found[lo])*(pos-lo))
        return result
    finally:
        for path in runs: path.unlink(missing_ok=True)


def score_frozen_predictions(*, source, prediction_path, lock_path, expected_lock_sha256, output_path, scratch_parent, ram_guard):
    """Verify complete prediction bytes before requesting target20 values."""
    counts, identity, expected_origins = _scope(source)
    lock_file = Path(lock_path)
    if lock_file.is_symlink() or not lock_file.is_file() or lock_file.stat().st_size > 16384:
        raise ValueError('prediction lock must be a bounded regular file')
    raw_lock = lock_file.read_bytes()
    if hashlib.sha256(raw_lock).hexdigest() != expected_lock_sha256:
        raise ValueError('prediction lock bytes differ from the independently retained digest')
    lock = json.loads(raw_lock)
    if raw_lock != _canonical(lock) + b'\n': raise ValueError('prediction lock is not canonical')
    expected_keys={'kind','fold_id','role','origin_identity_sha256','origin_counts','cells','checkpoint_sha256',
                   'record_count','record_bytes','predictions_sha256','source_asset_sha256','split_sha256',
                   'predictor_code_sha256','metric_code_sha256','origin_stream_sha256'}
    if (set(lock)!=expected_keys or lock['kind']!='validation_prediction_lock_v1' or
        lock['fold_id']!=source.fold_id or lock['role']!=source.role or
        lock['origin_identity_sha256']!=identity or lock['origin_stream_sha256']!=expected_origins or lock['origin_counts']!=list(counts) or
        lock['source_asset_sha256']!=source.source_asset_sha256 or lock['split_sha256']!=source.split_sha256 or
        lock['metric_code_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
        lock['cells']!=list(CELLS) or lock['record_count']!=sum(counts) or lock['record_bytes']!=_RECORD.size):
        raise ValueError('prediction lock differs from fixed validation scope')
    def valid_hash(value):
        return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if (type(lock['checkpoint_sha256']) is not dict or set(lock['checkpoint_sha256']) != set(CELLS) or
        not all(valid_hash(value) for value in lock['checkpoint_sha256'].values()) or
        not all(valid_hash(lock[key]) for key in ('predictor_code_sha256', 'predictions_sha256', 'origin_stream_sha256'))):
        raise ValueError('prediction lock lacks valid digest pins')
    pred = Path(prediction_path)
    if pred.is_symlink() or not pred.is_file() or pred.stat().st_size != sum(counts)*_RECORD.size:
        raise ValueError('prediction artifact is missing or has wrong length')
    output=Path(output_path)
    required_scratch = sum(counts) * 6 * 2 * _FLOAT.size + 64 * 1024 * 1024
    if shutil.disk_usage(scratch_parent).free < required_scratch:
        raise OSError('insufficient free scratch for exact full-origin quantiles')
    _check_ram(ram_guard)
    out_fd=os.open(output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    try:
        with pred.open('rb') as handle:
            # Hold one immutable in-process bytes snapshot before target access.
            # Re-reading the path after hash verification would permit concurrent
            # writes to alter predictions after the target stage has started.
            expected_bytes = sum(counts) * _RECORD.size
            free = _available_ram_bytes()
            if free is None or free < 2 * 1024**3 + expected_bytes + 16 * 1024**2:
                raise MemoryError('validation needs 2 GiB free plus prediction snapshot overhead')
            frozen = handle.read(expected_bytes + 1)
            if len(frozen) != expected_bytes or hashlib.sha256(frozen).hexdigest() != lock['predictions_sha256']:
                raise ValueError('prediction artifact differs from lock')
            _check_ram(ram_guard)
            origins, actual, previous = hashlib.sha256(), [0]*5, [-1]*5
            seen_user = -1
            for index in range(sum(counts)):
                if index % 128 == 0: _check_ram(ram_guard)
                values = _RECORD.unpack_from(frozen,index * _RECORD.size); user, origin = values[:2]
                if user >= 5 or user < seen_user or origin <= previous[user]:
                    raise ValueError('invalid frozen origin order')
                for cell in range(6): _quaternion(values[2+4*cell:6+4*cell])
                origins.update(_ORIGIN.pack(user, origin)); actual[user] += 1
                seen_user, previous[user] = user, origin
            if tuple(actual) != counts or origins.hexdigest() != lock['origin_stream_sha256']:
                raise ValueError('prediction artifact differs from lock or fixed scope')
            with tempfile.TemporaryDirectory(prefix='abc-validation-',dir=scratch_parent) as scratch:
                root=Path(scratch)
                files={(u,c): (root/f'{u}-{c}.bin').open('wb') for u in range(6) for c in range(6)}
                totals={(u,c):0.0 for u in range(6) for c in range(6)}
                seen=[0]*5
                try:
                    targets=iter(source.iter_targets())
                    for idx in range(sum(counts)):
                        if idx % 128 == 0: _check_ram(ram_guard)
                        unpacked=_RECORD.unpack_from(frozen, idx * _RECORD.size); user, origin=unpacked[:2]
                        try: target_user,target_origin,target=next(targets)
                        except StopIteration as exc: raise ValueError('target stream is incomplete') from exc
                        if (user,origin)!=(target_user,target_origin) or not 0 <= user < 5:
                            raise ValueError('target origin differs from frozen prediction identity')
                        for cell in range(6):
                            error=_angle(unpacked[2+4*cell:6+4*cell], target)
                            for bucket in (user,5): files[(bucket,cell)].write(_FLOAT.pack(error)); totals[(bucket,cell)]+=error
                        seen[user]+=1
                    if next(targets,None) is not None or tuple(seen)!=counts:
                        raise ValueError('target stream differs from full frozen origin scope')
                finally:
                    for file in files.values(): file.close()
                summaries=[]
                for bucket in range(6):
                    n=counts[bucket] if bucket<5 else sum(counts)
                    values={}
                    for cell,name in enumerate(CELLS):
                        p95,p99=_percentiles(root/f'{bucket}-{cell}.bin',n,root,ram_guard=ram_guard)
                        values[name]={'origin_count':n,'mean_degrees':totals[(bucket,cell)]/n,'p95_degrees':p95,'p99_degrees':p99}
                    summaries.append(values)
                result={'kind':'three_arm_outer_validation_fold','fold_id':source.fold_id,'role':source.role,
                        'origin_identity_sha256':identity,'prediction_lock_sha256':expected_lock_sha256,
                        'users':[{'anonymous_user_index':i,'predictors':summaries[i]} for i in range(5)],
                        'pooled_origins':summaries[5],'metric':'endpoint20 geodesic degrees',
                        'quantile_method':'linear, exact external merge','interpretation':'exploratory; not final holdout'}
                with os.fdopen(out_fd,'w',encoding='utf-8') as out:
                    out_fd=-1; json.dump(result,out,sort_keys=True,separators=(',', ':'),allow_nan=False); out.write('\n'); out.flush(); os.fsync(out.fileno())
                return result
    finally:
        if out_fd >= 0: os.close(out_fd)

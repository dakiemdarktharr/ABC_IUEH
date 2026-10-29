"""Three-arm outer-validation evaluator; no source access on import."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
from .source import FixedValidationSegments

ARMS = ('ordered_product', 'active_order_control', 'contemporaneous_raw_vector')
SEEDS = (20261026, 20261027)
ROLE = 'outer_validation5'
PURPOSE = 'ordered_motion_three_arm_screen'


def validate_checkpoints(fold, checkpoints, *, fit_identities, torch, trainer, checkpoint):
    """Validate the full 18-fit screen before loading this fold's six models."""
    expected = {(f, a, s) for f in range(3) for a in ARMS for s in SEEDS}
    if set(checkpoints) != expected:
        raise ValueError('all 18 completed screen checkpoints are required')
    if set(fit_identities) != {0, 1, 2}:
        raise ValueError('three independently verified fit bundle identities are required')
    models, hashes, fold_data = {}, {}, {}
    for (f, arm, seed), (path, digest) in sorted(checkpoints.items()):
        state = checkpoint.load_checkpoint(Path(path), expected_sha256=digest, torch=torch)
        recipe = trainer.ScreenRecipe(fold_id=f, training_seed=seed, arm_id=arm).as_dict()
        recipe_hash = hashlib.sha256(trainer._canonical_json(recipe)).hexdigest()
        _, model_hash = trainer._model_binding(arm)
        if (state.get('recipe') != recipe or state.get('recipe_sha256') != recipe_hash or
            state.get('draft_sha256') != trainer.THREE_ARM_DRAFT_SHA256 or
            state.get('model_sha256') != model_hash or state.get('runner_sha256') != trainer._runner_sha256() or
            state.get('completed_updates') != 5000 or state.get('sampler_cursor_draws') != 640000):
            raise ValueError('checkpoint is incomplete or differs from fixed screen, model, or runner')
        if state.get('data_identity_sha256') != fit_identities[f]:
            raise ValueError('checkpoint fit bundle differs from independently signed identity')
        fold_data.setdefault(f, set()).add(state.get('data_identity_sha256'))
        if f == fold:
            model, _, _ = trainer._load_model(arm, seed)
            model.load_state_dict(state['model_state_dict'], strict=True)
            model.eval()
            models[(arm, seed)] = model
            hashes[f'{arm}:{seed}'] = digest
    if any(len(items) != 1 or None in items for items in fold_data.values()):
        raise ValueError('all six fits in each fold must share one training data identity')
    return models, hashes


def endpoint20_errors(prediction, target, *, torch):
    """Actual sign-invariant SO(3) geodesic degrees at horizon index 19."""
    if tuple(prediction.shape) != tuple(target.shape) or tuple(target.shape[1:]) != (20, 4):
        raise ValueError('model and target must have shape [B,20,4]')
    p, q = prediction[:, 19, :].to(torch.float64), target[:, 19, :].to(torch.float64)
    if not bool(torch.isfinite(p).all() and torch.isfinite(q).all()):
        raise ValueError('orientation is non-finite')
    pn, qn = torch.linalg.vector_norm(p, dim=-1), torch.linalg.vector_norm(q, dim=-1)
    if bool((pn < 1e-8).any() or (qn < 1e-8).any() or
            (torch.abs(pn - 1) > 2e-3).any() or (torch.abs(qn - 1) > 2e-3).any()):
        raise ValueError('orientation is not unit norm')
    p, q = p / pn[:, None], q / qn[:, None]
    scalar = torch.abs((p * q).sum(dim=-1))
    vec = q[:, :1] * p[:, 1:] - p[:, :1] * q[:, 1:] - torch.linalg.cross(q[:, 1:], p[:, 1:], dim=-1)
    return 2 * torch.atan2(torch.linalg.vector_norm(vec, dim=-1), scalar) * (180 / math.pi)


def summarize_user(errors, *, np):
    result, sizes = {}, set()
    for cell, values in errors.items():
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or np.any(vector < 0) or np.any(vector > 180):
            raise ValueError('invalid endpoint error vector')
        sizes.add(int(vector.size))
        result[cell] = {'origin_count': int(vector.size), 'mean_degrees': float(vector.mean()),
                        'p95_degrees': float(np.quantile(vector, .95)), 'p99_degrees': float(np.quantile(vector, .99))}
    if len(sizes) != 1:
        raise ValueError('predictors must score the same origins')
    return result


def evaluate_fold(*, fold_id, checkpoints, fit_identities, segments, output_path, device='cpu', batch_size=128):
    """Evaluate frozen origin stream; segment source must be independently verified.

    Stream yields (anonymous user index, canonical offset, float32 quaternion
    segment). It supplies fold_id, role, expected_origin_counts, and a verified
    origin_identity_sha256; labels alone are not proof of source membership.
    """
    import numpy as np
    import torch
    from ..runner import checkpoint, trainer
    if type(fold_id) is not int or fold_id not in (0, 1, 2): raise ValueError('invalid fold')
    if type(segments) is not FixedValidationSegments:
        raise ValueError('production evaluator requires the fixed validation source type')
    if segments.role != ROLE or segments.fold_id != fold_id: raise ValueError('wrong validation fold or role')
    expected = tuple(segments.expected_origin_counts)
    if len(expected) != 5 or any(type(n) is not int or n <= 0 for n in expected): raise ValueError('expected five users')
    if type(batch_size) is not int or not 1 <= batch_size <= 128: raise ValueError('bad batch size')
    models, hashes = validate_checkpoints(fold_id, checkpoints, fit_identities=fit_identities, torch=torch, trainer=trainer, checkpoint=checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    for model in models.values(): model.to(device=device)
    errors = {i: {f'{a}:{s}': [] for a, s in sorted(models)} for i in range(5)}
    counts, last_user, last_offset = [0] * 5, -1, -1
    with torch.no_grad():
        for user, offset, poses in segments:
            if (type(user) is not int or not 0 <= user < 5 or type(offset) is not int or
                user < last_user or (user == last_user and offset <= last_offset)):
                raise ValueError('segments differ from canonical user/origin order')
            last_user, last_offset = user, offset
            if poses.dtype != np.float32 or poses.ndim != 2 or poses.shape[1] != 4 or len(poses) < 420:
                raise ValueError('invalid validation segment')
            for first in range(0, len(poses) - 419, batch_size):
                starts = range(first, min(first + batch_size, len(poses) - 419))
                x = torch.from_numpy(np.stack([poses[i:i+400] for i in starts])).to(device=device)
                y = torch.from_numpy(np.stack([poses[i+400:i+420] for i in starts])).to(device=device)
                for (arm, seed), model in models.items():
                    errors[user][f'{arm}:{seed}'].append(endpoint20_errors(model(x), y, torch=torch).cpu().numpy())
                counts[user] += len(starts)
    if tuple(counts) != expected: raise ValueError('full fixed origin counts were not consumed')
    if not isinstance(segments.read_receipt, dict): raise ValueError('source read did not complete')
    summaries, pooled = [], {f'{a}:{s}': [] for a, s in sorted(models)}
    for user in range(5):
        vectors = {key: np.concatenate(chunks) for key, chunks in errors[user].items()}
        summaries.append({'anonymous_user_index': user, 'predictors': summarize_user(vectors, np=np)})
        for key, vector in vectors.items(): pooled[key].append(vector)
    report = {'kind': 'three_arm_outer_validation_fold', 'purpose': PURPOSE, 'fold_id': fold_id, 'role': ROLE,
              'scope': 'outer-validation users within train15; no old dev or final',
              'validation_origin_identity_sha256': segments.origin_identity_sha256, 'source_read_receipt': segments.read_receipt, 'checkpoint_sha256': hashes,
              'users': summaries, 'pooled_origins': summarize_user({k: np.concatenate(v) for k,v in pooled.items()}, np=np),
              'metric': 'endpoint20 sign-invariant geodesic degrees', 'quantile_method': 'numpy linear',
              'interpretation': 'exploratory; not final-holdout results'}
    fd = os.open(Path(output_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    return report

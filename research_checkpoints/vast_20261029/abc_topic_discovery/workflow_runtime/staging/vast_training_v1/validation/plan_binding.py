"""Draft file-backed provenance binding; it does not grant data access.

Call only after a prospective plan has been independently reviewed and signed.
This function reads files named by its caller to compute hashes; never invoke it
on the real source during generated-only testing. No source values are decoded.
"""
from __future__ import annotations

import json
from pathlib import Path
from . import signed_plan
from .fit_identity import verify_fit_identities
from .source import FixedValidationSegments, _origin_stream_digest
from ..ops_v2.checkpoint_store import sha256_file


def verify_actual_inputs(raw_plan, *, public_key, plan_sha256, source_file, split_file,
                         protocol_file, code_files, checkpoint_files, fit_manifests,
                         fit_public_keys, fit_key_ids, fold_sources):
    """Authenticate the plan, then independently hash the exact selected files.

    All three fold metadata objects must be the fixed metadata-only source type.
    A signed plan and matching bytes still do not constitute coordinator approval,
    a target gate, a sandbox or a completed independent methodology review.
    """
    try:
        declared=json.loads(raw_plan)['plan']
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('invalid plan envelope') from exc
    signed_plan.verify_signed_plan(raw_plan,public_key=public_key,
                                   expected_plan_sha256=plan_sha256,observed=declared)
    if (type(code_files) is not dict or set(code_files)!=set(signed_plan.CODE_KEYS) or
        type(checkpoint_files) is not dict or set(checkpoint_files)!=set(signed_plan.CELLS) or
        type(fold_sources) is not dict or set(fold_sources)!={0,1,2}):
        raise ValueError('file inputs differ from complete fixed scope')
    def file_hash(path):
        return sha256_file(Path(path))[0]
    source_hash=file_hash(source_file); split_hash=file_hash(split_file)
    origins={}; streams={}
    for fold in range(3):
        source=fold_sources[fold]
        if (type(source) is not FixedValidationSegments or source.fold_id != fold or
            source.role != 'outer_validation5' or source.source_asset_sha256 != source_hash or
            source.split_sha256 != split_hash):
            raise ValueError('fold source does not match fixed scope or source file pins')
        origin=source._loader._pool_origin_digest(source._rows,source._binding,fold,source.role)
        stream=_origin_stream_digest(source._rows,source._users)
        if origin != source.origin_identity_sha256 or stream != source.origin_stream_sha256:
            raise ValueError('fold metadata identity changed')
        origins[str(fold)]=origin; streams[str(fold)]=stream
    identities=verify_fit_identities(fit_manifests,public_keys=fit_public_keys,key_ids=fit_key_ids)
    observed={'kind':'abc_three_arm_validation_plan_v1','purpose':'ordered_motion_three_arm_screen',
              'role':'outer_validation5','protocol_sha256':file_hash(protocol_file),
              'source_sha256':source_hash,'split_sha256':split_hash,'fold_origin_sha256':origins,
              'fold_origin_stream_sha256':streams,'fit_identity_sha256':{str(k):v for k,v in identities.items()},
              'checkpoint_sha256':{key:file_hash(path) for key,path in checkpoint_files.items()},
              'code_sha256':{key:file_hash(path) for key,path in code_files.items()}}
    return signed_plan.verify_signed_plan(raw_plan,public_key=public_key,
                                          expected_plan_sha256=plan_sha256,observed=observed)

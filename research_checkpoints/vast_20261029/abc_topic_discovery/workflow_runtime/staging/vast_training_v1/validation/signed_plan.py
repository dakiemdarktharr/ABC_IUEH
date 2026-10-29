"""Verify a prospective, independently pinned signed validation plan.

This is a draft integrity primitive, not a review, authorization, source opener,
or a replacement for the existing protocol. A plan must be produced before
any validation outcome and pin the exact actual source/checkpoint/code bytes.
"""
from __future__ import annotations
import base64
import hashlib
import json
import re

HASH = re.compile(r'^[0-9a-f]{64}$')
CELLS = tuple(f'{fold}:{arm}:{seed}' for fold in range(3)
              for arm in ('ordered_product','active_order_control','contemporaneous_raw_vector')
              for seed in (20261026,20261027))
CODE_KEYS = frozenset((
    'predictor', 'metric', 'source', 'hdf5_adapter', 'window_reader',
    'plan_verifier', 'plan_binding', 'v3_loader', 'bounded_reader', 'checkpoint_loader',
    'fit_identity', 'runner', 'ordered_model', 'raw_vector_model', 'review',
))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hashes(value, keys):
    return (type(value) is dict and set(value) == set(keys) and
            all(type(digest) is str and HASH.fullmatch(digest) for digest in value.values()))


def verify_signed_plan(raw, *, public_key, expected_plan_sha256, observed):
    """Check signature, immutable plan bytes and exact independently measured pins.

    ``observed`` is a locally measured dictionary with the same plan schema;
    hash strings copied from the plan without measuring actual artifacts do not
    establish provenance. The caller retains the public key and plan-byte hash
    independently, obtains review/authorization and enforces a target gate.
    """
    if (type(raw) is not bytes or len(raw) > 65536 or type(public_key) is not bytes or len(public_key) != 32 or
        type(expected_plan_sha256) is not str or not HASH.fullmatch(expected_plan_sha256) or
        hashlib.sha256(raw).hexdigest() != expected_plan_sha256):
        raise ValueError('signed plan bytes or independent public-key/hash pin invalid')
    try:
        envelope=json.loads(raw)
        if type(envelope) is not dict or set(envelope) != {'plan','signature'} or canonical(envelope)+b'\n' != raw:
            raise ValueError('signed plan envelope is not canonical')
        plan=envelope['plan']
        signature=base64.b64decode(envelope['signature'],validate=True)
        if len(signature) != 64: raise ValueError('invalid signature size')
    except (TypeError,ValueError,UnicodeDecodeError) as exc:
        raise ValueError('invalid signed plan envelope') from exc
    keys={'kind','purpose','role','protocol_sha256','source_sha256','split_sha256',
          'fold_origin_sha256','fold_origin_stream_sha256','fit_identity_sha256','checkpoint_sha256','code_sha256'}
    if (type(plan) is not dict or set(plan) != keys or
        plan['kind'] != 'abc_three_arm_validation_plan_v1' or
        plan['purpose'] != 'ordered_motion_three_arm_screen' or plan['role'] != 'outer_validation5' or
        not all(type(plan[k]) is str and HASH.fullmatch(plan[k]) for k in ('protocol_sha256','source_sha256','split_sha256')) or
        not _hashes(plan['fold_origin_sha256'], ('0','1','2')) or
        not _hashes(plan['fold_origin_stream_sha256'], ('0','1','2')) or
        not _hashes(plan['fit_identity_sha256'], ('0','1','2')) or
        not _hashes(plan['checkpoint_sha256'], CELLS) or not _hashes(plan['code_sha256'], CODE_KEYS)):
        raise ValueError('signed plan differs from the exact fixed validation scope')
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, canonical(plan))
    except Exception as exc:
        raise ValueError('signed plan signature failed') from exc
    if type(observed) is not dict or canonical(observed) != canonical(plan):
        raise ValueError('independently measured provenance differs from signed plan')
    return plan

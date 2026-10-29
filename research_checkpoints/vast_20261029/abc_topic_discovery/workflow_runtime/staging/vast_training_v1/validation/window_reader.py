"""Bounded source-window mapping for a future reviewed HDF5 reader.

These functions do not open files, select users, or confer data access. The
caller must supply previously authenticated segment rows and a checked reader
bound to the same file and allowlisted dataset. This module is not connected to
FixedValidationSegments, whose production source iterator remains disabled.
"""
from __future__ import annotations

import math
import struct

CONTEXT_ROWS = 400
ENDPOINT_OFFSET = 419
ROW_BYTES = 16
_RECORD = struct.Struct('<4f')


def _validated_quaternions(payload, count):
    if type(payload) is not bytes or len(payload) != count * ROW_BYTES:
        raise ValueError('row reader returned a different shape or dtype')
    previous = None
    for q in struct.iter_unpack('<4f', payload):
        if not all(math.isfinite(value) for value in q):
            raise ValueError('nonfinite quaternion in source slice')
        if abs(sum(value*value for value in q) - 1.0) > 4e-6:
            raise ValueError('nonunit quaternion in source slice')
        if previous is not None and sum(a*b for a,b in zip(previous,q)) < -1e-6:
            raise ValueError('source quaternion slice is not hemisphere-aligned')
        previous=q


def iter_segment_contexts(*, user_index, global_offset, origin_count, sample_count,
                          read_rows, check_guard):
    """Yield each [i, i+400) only; never request that origin's endpoint."""
    _check_scope(user_index, global_offset, origin_count, sample_count)
    for index in range(origin_count):
        check_guard()
        context=read_rows(index, index+CONTEXT_ROWS)
        _validated_quaternions(context, CONTEXT_ROWS)
        yield user_index, global_offset+index, context


def iter_segment_endpoints(*, user_index, global_offset, origin_count, sample_count,
                           read_rows, check_guard):
    """Yield i+419 only. Caller must enforce a verified target-read gate."""
    _check_scope(user_index, global_offset, origin_count, sample_count)
    for index in range(origin_count):
        check_guard()
        raw=read_rows(index+ENDPOINT_OFFSET, index+ENDPOINT_OFFSET+1)
        _validated_quaternions(raw, 1)
        yield user_index, global_offset+index, _RECORD.unpack(raw)


def _check_scope(user, offset, count, samples):
    if (type(user) is not int or user not in range(5) or
        type(offset) is not int or offset < 0 or
        type(count) is not int or count <= 0 or
        type(samples) is not int or samples != count + ENDPOINT_OFFSET or
        offset + count > 2**64):
        raise ValueError('segment differs from fixed outer-validation origin range')

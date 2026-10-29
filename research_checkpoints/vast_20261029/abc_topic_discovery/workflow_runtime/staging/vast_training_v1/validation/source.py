"""Fixed outer-validation reader, separate from the fit-only V3 source route.

New code: independent review and prospective evaluation lock are outstanding.
Iteration is disabled and raises before source bytes are opened. Metadata-only
construction selects no old-dev/final quaternion values.
"""
import time
import hashlib
import struct

ROLE = 'outer_validation5'


def _origin_stream_digest(rows, users):
    # Fixed origin offsets in canonical user/row order; metadata only.
    transcript = hashlib.sha256()
    for row in rows:
        user = users.index(row.user_id)
        for origin in range(int(row.offset_int64), int(row.offset_int64) + int(row.count)):
            transcript.update(struct.pack('<BQ', user, origin))
    return transcript.hexdigest()


class FixedValidationSegments:
    def __init__(self, fold_id):
        if type(fold_id) is not int or fold_id not in (0, 1, 2): raise ValueError('invalid fold')
        import numpy as np
        from ..source_adapter_v1 import source_adapter as adapter
        self.fold_id, self.role = fold_id, ROLE
        self._np = np
        self._loader = adapter._load_pinned_v3_loader()
        self._reader = self._loader._load_pinned_reader()
        self._binding = self._reader.validate_metadata_bindings()
        users = tuple(sorted(self._binding.ownership[fold_id]['outer_validation']))
        fit_users = set(self._binding.ownership[fold_id]['outer_fit'])
        if len(users) != 5 or len(set(users)) != 5 or len(fit_users) != 10 or fit_users.intersection(users):
            raise ValueError('fixed fold does not have disjoint five validation and ten fit users')
        if any(user not in self._binding.parent.expected_train.user_sessions for user in users):
            raise ValueError('validation user is outside the fixed train partition')
        self._users = users
        rows = tuple(sorted((r for r in self._binding.origin_rows if r.user_id in users),
                            key=lambda r: (r.user_id, r.offset_int64)))
        self._rows = rows
        if len(rows) != sum(len(segments) for user in users
                            for segments in self._binding.parent.expected_train.user_sessions[user].values()):
            raise ValueError('validation rows do not cover exact expected segments')
        keys, previous_stop = set(), -1
        counts = [0] * 5
        for row in sorted(rows, key=lambda r: r.offset_int64):
            key = (row.user_id, row.session_id, row.segment_id)
            if key in keys or row.fold_id != fold_id or row.offset_int64 < previous_stop or row.count <= 0:
                raise ValueError('duplicate or overlapping validation origin rows')
            keys.add(key)
            values = self._binding.origin_values[row.offset_int64:row.offset_int64 + row.count]
            if (len(values) != row.count or int(values[0]) != 399 or int(values[-1]) != 398 + row.count or
                    (row.count > 1 and not np.array_equal(values[1:], values[:-1] + 1))):
                raise ValueError('validation origin slice is not the complete fixed segment range')
            expected_segments = self._binding.parent.expected_train.user_sessions[row.user_id]
            if row.session_id not in expected_segments or row.segment_id not in expected_segments[row.session_id]:
                raise ValueError('validation origin segment is absent from fixed metadata')
            counts[users.index(row.user_id)] += int(row.count)
            previous_stop = row.offset_int64 + row.count
        self.expected_origin_counts = tuple(counts)
        # Bind the two-stage record sequence to the fixed metadata ranges.
        # This is metadata only; this does not authorize a human-data read.
        self.origin_stream_sha256 = _origin_stream_digest(rows, users)
        self.source_asset_sha256 = adapter.SOURCE_SHA256
        self.split_sha256 = adapter.SPLIT_SHA256
        self.origin_identity_sha256 = self._loader._pool_origin_digest(rows, self._binding, fold_id, ROLE)
        self._opened = False
        self.read_receipt = None

    def __iter__(self):
        # Protocol v5 requires predictions locked before targets are passed to
        # evaluation. This draft streams both together, so it cannot open data.
        raise PermissionError('validation source disabled: prediction lock before target access is not implemented')


"""Synthetic source bridge checks; no source files, GPU, or cloud are accessed."""
import hashlib
import json
import time
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from . import source_adapter as adapter


@dataclass
class Row:
    user_id: str
    session_id: str
    segment_id: str
    offset_int64: int
    count: int
    fold_id: int = 0


def segment(row):
    return SimpleNamespace(user_id=row.user_id, session_id=row.session_id,
                           segment_id=row.segment_id, global_origin_offset=row.offset_int64,
                           origin_count=row.count,
                           poses=np.zeros((row.count + 419, 4), dtype=np.float32))


class SyntheticAdapterTests(unittest.TestCase):
    def test_reorders_segments_to_frozen_origin_rank(self):
        rows = [Row('z', 's', '0', 0, 3), Row('a', 's', '0', 3, 2)]
        values = np.array([399, 400, 401, 399, 400], dtype=np.int64)
        segments = [segment(rows[1]), segment(rows[0])]
        actual = adapter.ordered_pool_segments(rows, SimpleNamespace(origin_values=values), segments)
        self.assertEqual(actual, (segments[1], segments[0]))
        self.assertEqual(len(adapter.fit_role_inventory_digest(rows, 0, adapter.ROLE)), 64)

    def test_rejects_gaps_wrong_counts_and_extra_segments(self):
        row = Row('u', 's', '0', 0, 3)
        binding = SimpleNamespace(origin_values=np.array([399, 401, 401], dtype=np.int64))
        with self.assertRaises(adapter.SourceAdapterError):
            adapter.ordered_pool_segments([row], binding, [segment(row)])
        binding.origin_values = np.array([399, 400, 401], dtype=np.int64)
        wrong = segment(row)
        wrong.origin_count = 2
        with self.assertRaises(adapter.SourceAdapterError):
            adapter.ordered_pool_segments([row], binding, [wrong])
        extra = segment(Row('v', 's', '0', 3, 1))
        with self.assertRaises(adapter.SourceAdapterError):
            adapter.ordered_pool_segments([row], binding, [segment(row), extra])

    def test_actual_receipt_bound_only_after_complete(self):
        class Grant:
            fit_role_inventory_sha256 = '1' * 64
            fit_origin_inventory_sha256 = '2' * 64
            authorization_sha256 = '3' * 64
            def assert_live(self):
                pass
        grant = Grant()
        static = {
            'source_asset_sha256': adapter.SOURCE_SHA256,
            'split_sha256': adapter.SPLIT_SHA256,
            'fit_role_inventory_sha256': grant.fit_role_inventory_sha256,
            'fit_origin_inventory_sha256': grant.fit_origin_inventory_sha256,
            'source_loader_sha256': adapter.V3_LOADER_SHA256,
            'source_authorization_sha256': grant.authorization_sha256,
        }
        class Authority:
            def finalize_fit10_source_binding(self, actual_grant, request, receipt):
                self.asserted = (actual_grant, request, receipt)
                return {**static, 'source_read_receipt_sha256': hashlib.sha256(
                    json.dumps(receipt, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
        root = object.__new__(adapter.V3Fit10SourceRoot)
        root._authority = Authority()
        state = adapter._GrantState(grant, object(), 0, binding=static, verified=True,
                                    expiry_monotonic=time.monotonic() + 100)
        root._grants = [state]
        with self.assertRaises(adapter.SourceAuthorizationError):
            root.finalize_fit10_source_binding(grant)
        state.receipt = {'synthetic_read_count': 4}
        state.complete = True
        result = root.finalize_fit10_source_binding(grant)
        self.assertEqual(len(result['source_read_receipt_sha256']), 64)
        with self.assertRaises(adapter.SourceAuthorizationError):
            root.finalize_fit10_source_binding(grant)
        with self.assertRaises(adapter.SourceAuthorizationError):
            adapter.validate_source_binding({**static, 'source_read_receipt_sha256': '0' * 64}, grant)


if __name__ == '__main__':
    unittest.main()

"""No-data scope tests; importing trainer does not load numerical libraries."""
import unittest
from pathlib import Path
import tempfile

import trainer
from fit_api import PilotRecipe, ScreenRecipe


class FakeBundle:
    total_origins = 1
    identity_sha256 = "a" * 64
    def __init__(self, summary):
        self.identity_summary = summary
    def assert_active(self):
        pass


class ScopeTests(unittest.TestCase):
    def test_pilot_bundle_rejected_for_screen(self):
        bundle = FakeBundle({"purpose": PilotRecipe().purpose, "role": "outer_fit10", "fold_id": 0})
        trainer._bundle_scope_check(bundle, PilotRecipe())
        with self.assertRaises(PermissionError):
            trainer._bundle_scope_check(bundle, ScreenRecipe())
    def test_missing_summary_rejected(self):
        with self.assertRaises(PermissionError):
            trainer._bundle_scope_check(FakeBundle(None), PilotRecipe())

    def test_cgroup_v2_uses_tightest_parent_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root / "child"
            child.mkdir()
            for folder, limit, usage in ((root, "1000", "800"), (child, "900", "100")):
                (folder / "memory.max").write_text(limit)
                (folder / "memory.current").write_text(usage)
            self.assertEqual(trainer._cgroup_available_bytes(root, "0::/child\n"), 200)


if __name__ == "__main__":
    unittest.main()

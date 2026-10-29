"""CLI wiring tests with no reader or human source."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import export
from .bundle import SCREEN_PURPOSE


class _Root:
    def completed_source_evidence(self):
        return ({"receipt": {"generated_only": True}},)


class ExportTests(unittest.TestCase):
    def test_separate_pin_and_evidence_written_after_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            bundle, evidence = base / "bundle", base / "evidence"
            def fake_export(destination, **kwargs):
                self.assertEqual(kwargs["purpose"], SCREEN_PURPOSE)
                self.assertTrue((evidence / "public-key.ed25519").is_file())
                self.assertFalse((evidence / "read-evidence.json").exists())
                destination.mkdir()
                return "a" * 64
            with patch.dict("os.environ", {"ABC_FIT_EXPORT_MONITORED": "1"}), \
                 patch.object(export, "require_private_parent"), \
                 patch.object(export, "create_local_source_root", return_value=(_Root(), b"k" * 32, "fixture")), \
                 patch.object(export, "export_authorized_fit10_bundle", side_effect=fake_export):
                self.assertEqual(export.main(["--fold-id", "0", "--purpose", SCREEN_PURPOSE,
                    "--bundle-dir", str(bundle), "--evidence-dir", str(evidence)]), 0)
            self.assertEqual((evidence / "public-key.ed25519").read_bytes(), b"k" * 32)
            self.assertIn("generated_only", (evidence / "read-evidence.json").read_text())


if __name__ == "__main__":
    unittest.main()

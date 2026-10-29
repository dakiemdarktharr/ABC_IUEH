import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from checkpoint_store import (
    CheckpointError,
    backup_checkpoint,
    compare_resume,
    copy_checkpoint,
    verify_copy,
    verify_restored_receipt,
)


def digest(char: str) -> str:
    return hashlib.sha256(char.encode("utf-8")).hexdigest()


def metadata(payload: bytes) -> dict:
    return {
        "schema": "abc-training-checkpoint-v1",
        "run_id": "synthetic-run",
        "plan_sha256": digest("plan"),
        "data_sha256": digest("synthetic-data"),
        "code_sha256": digest("code"),
        "model_sha256": digest("model"),
        "fold_id": 0,
        "training_seed": 20261026,
        "arm": "synthetic-arm",
        "global_step": 100,
        "total_steps": 5000,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "resume_state_complete": True,
        "state_components": {
            "model": True,
            "optimizer": True,
            "sampler": True,
            "python_rng": True,
            "numpy_rng": True,
            "torch_cpu_rng": True,
            "torch_cuda_rng": True,
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class CheckpointStoreTests(unittest.TestCase):
    def test_backup_requires_attestation_and_distinct_device_and_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "checkpoint.pt"
            source.write_bytes(b"synthetic")
            destination = root / "remote"
            destination.mkdir()
            attestation = root / "attestation.json"
            write_json(attestation, {})
            with self.assertRaises(CheckpointError):
                backup_checkpoint(source, root / "metadata.json", destination, attestation)
            write_json(attestation, {"schema": "abc-external-storage-attestation-v1",
                "external_storage_verified_by_operator": True, "destination": str(destination.resolve()),
                "storage_identity": "synthetic remote", "evidence_reference": "synthetic evidence"})
            with self.assertRaises(CheckpointError):
                backup_checkpoint(source, root / "metadata.json", destination, attestation)
            with patch("checkpoint_store._require_different_filesystems"), patch("checkpoint_store.copy_checkpoint", side_effect=CheckpointError("copy failed")):
                with self.assertRaises(CheckpointError):
                    backup_checkpoint(source, root / "metadata.json", destination, attestation)

    def test_copy_is_verified_no_overwrite_and_resume_pins_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root / "persistent"
            payload = b"synthetic checkpoint bytes; not a torch pickle"
            source = root / f"update-100-{hashlib.sha256(payload).hexdigest()}.pt"
            meta_path = source.with_suffix(".meta.json")
            source.write_bytes(payload)
            write_json(meta_path, metadata(payload))
            with patch("checkpoint_store._fsync_directory") as fsync_dir:
                copied = copy_checkpoint(source, meta_path, destination)
            self.assertEqual(fsync_dir.call_count, 3)
            manifest = Path(copied["manifest_path"])
            verify_copy(manifest)
            # The result points to the exact hash-named file that the trainer
            # must consume via its --resume argument.
            self.assertEqual(Path(copied["trainer_resume_argument"]).name, f"checkpoint-{copied['payload_sha256']}.pt")
            expected = {key: metadata(payload)[key] for key in (
                "run_id", "plan_sha256", "data_sha256", "code_sha256", "model_sha256",
                "fold_id", "training_seed", "arm",
            )}
            expected["expected_global_step"] = 100
            expected_path = root / "expected.json"
            write_json(expected_path, expected)
            verified = compare_resume(manifest, expected_path)
            self.assertEqual(verified["metadata"]["global_step"], 100)
            self.assertEqual(verified["trainer_resume_argument"], copied["trainer_resume_argument"])
            self.assertEqual(verified["root_must_bind_expected_payload_sha256"], copied["payload_sha256"])
            self.assertTrue(source.exists(), "copy must preserve the instance-local source")

            # Repeating an exact content copy is safe; different content can't
            # take the same hash-derived destination name.
            copy_checkpoint(source, meta_path, destination)
            self.assertEqual(len(list(destination.glob("checkpoint-*.manifest.json"))), 1)

    def test_resume_pin_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "persistent"
            payload = b"synthetic bytes"
            source = root / f"update-100-{hashlib.sha256(payload).hexdigest()}.pt"
            meta_path = source.with_suffix(".meta.json")
            source.write_bytes(payload)
            write_json(meta_path, metadata(payload))
            copied = copy_checkpoint(source, meta_path, dest)
            manifest = Path(copied["manifest_path"])
            expected = {key: metadata(payload)[key] for key in (
                "run_id", "plan_sha256", "data_sha256", "code_sha256", "model_sha256",
                "fold_id", "training_seed", "arm",
            )}
            expected["code_sha256"] = digest("different-code")
            expected_path = root / "expected.json"
            write_json(expected_path, expected)
            with self.assertRaises(CheckpointError):
                compare_resume(manifest, expected_path)

    def test_incomplete_resume_state_and_bad_restore_receipt_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "persistent"
            payload = b"synthetic bytes"
            source = root / f"update-100-{hashlib.sha256(payload).hexdigest()}.pt"
            meta_path = source.with_suffix(".meta.json")
            source.write_bytes(payload)
            bad_meta = metadata(payload)
            bad_meta["state_components"]["torch_cuda_rng"] = False
            write_json(meta_path, bad_meta)
            with self.assertRaises(CheckpointError):
                copy_checkpoint(source, meta_path, dest)

            write_json(meta_path, metadata(payload))
            copied = copy_checkpoint(source, meta_path, dest)
            expected = {key: metadata(payload)[key] for key in (
                "run_id", "plan_sha256", "data_sha256", "code_sha256", "model_sha256",
                "fold_id", "training_seed", "arm",
            )}
            expected_path = root / "expected.json"
            write_json(expected_path, expected)
            receipt = {
                "schema": "abc-resume-restored-v1",
                "resume_verified": True,
                "run_id": metadata(payload)["run_id"],
                "checkpoint_sha256": copied["payload_sha256"],
                "plan_sha256": metadata(payload)["plan_sha256"],
                "loaded_global_step": 99,
                "optimizer_restored": True,
                "sampler_restored": True,
                "rng_restored": True,
                "model_restored": True,
            }
            receipt_path = root / "receipt.json"
            write_json(receipt_path, receipt)
            with self.assertRaises(CheckpointError):
                verify_restored_receipt(Path(copied["manifest_path"]), expected_path, receipt_path)
            receipt["loaded_global_step"] = 100
            write_json(receipt_path, receipt)
            verify_restored_receipt(Path(copied["manifest_path"]), expected_path, receipt_path)
            receipt["optimizer_restored"] = 1  # Python equality must not accept an integer as a boolean.
            write_json(receipt_path, receipt)
            with self.assertRaises(CheckpointError):
                verify_restored_receipt(Path(copied["manifest_path"]), expected_path, receipt_path)

    def test_sidecar_payload_digest_must_match_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b"synthetic bytes"
            source = root / f"update-100-{hashlib.sha256(payload).hexdigest()}.pt"
            meta_path = source.with_suffix(".meta.json")
            source.write_bytes(payload)
            meta = metadata(payload)
            meta["payload_sha256"] = digest("different-payload")
            write_json(meta_path, meta)
            with self.assertRaises(CheckpointError):
                copy_checkpoint(source, meta_path, root / "persistent")

    @unittest.skipUnless(os.name == "posix", "POSIX symlink test")
    def test_existing_payload_symlink_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b"synthetic bytes"
            source = root / f"update-100-{hashlib.sha256(payload).hexdigest()}.pt"
            meta_path = source.with_suffix(".meta.json")
            destination = root / "persistent"
            external = root / "external.pt"
            source.write_bytes(payload)
            external.write_bytes(payload)
            write_json(meta_path, metadata(payload))
            destination.mkdir()
            target = destination / f"checkpoint-{hashlib.sha256(payload).hexdigest()}.pt"
            target.symlink_to(external)
            with self.assertRaises(CheckpointError):
                copy_checkpoint(source, meta_path, destination)

    @unittest.skipUnless(os.name == "posix", "POSIX symlink test")
    def test_verify_copy_rejects_payload_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b"synthetic bytes"
            source = root / f"update-100-{hashlib.sha256(payload).hexdigest()}.pt"
            meta_path = source.with_suffix(".meta.json")
            destination = root / "persistent"
            source.write_bytes(payload)
            write_json(meta_path, metadata(payload))
            copied = copy_checkpoint(source, meta_path, destination)
            target = Path(copied["trainer_resume_argument"])
            external = root / "external.pt"
            external.write_bytes(payload)
            target.unlink()
            target.symlink_to(external)
            with self.assertRaises(CheckpointError):
                verify_copy(Path(copied["manifest_path"]))


if __name__ == "__main__":
    unittest.main()

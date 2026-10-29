"""Build an explicit code-only runtime tree; never include recordings or bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
STAGING = Path("abc_topic_discovery/workflow_runtime/staging")
RUNTIME = STAGING / "vast_training_v1"
FILES = (
    "ABC_IUEH/src/abc_iueh/__init__.py", "ABC_IUEH/src/abc_iueh/arc_model.py",
    "ABC_IUEH/requirements-lock.txt", "ABC_IUEH/requirements-gpu-lock.txt",
    str(STAGING / "ordered_motion_model_v3/ordered_motion_model.py"),
    *(str(RUNTIME / "data" / name) for name in ("__init__.py", "bundle.py", "ed25519_verifier.py")),
    *(str(RUNTIME / "runner" / name) for name in ("__init__.py", "checkpoint.py", "fit_api.py", "launcher.py", "lease_clock.py", "raw_vector_adapter.py", "campaign_plan.py", "campaign_executor.py", "campaign_executor.md", "campaign-template.json", "test_campaign_plan.py", "test_campaign_executor.py", "test_checkpoint_resume.py", "test_checkpoint_policy.py", "test_full_fit_synthetic.py", "test_models_synthetic.py", "test_scope.py", "test_launcher.py", "test_lease_clock.py", "test_runtime_binding.py",
        "trainer.py", "requirements-vast.lock", "screen_plan.py", "README.md")),
    *(str(RUNTIME / "time_budget" / name) for name in ("__init__.py", "guard.py", "ledger.py", "combined.py", "bootstrap.py", "supervisor_linux.py", "remote_journal.py", "test_remote_journal.py", "test_guard.py", "test_ledger.py", "test_supervisor_linux.py", "README.md")),
    *(str(RUNTIME / "ops_v2" / name) for name in ("__init__.py", "budget_gate.py", "budget_guard.py",
        "budget-template.json", "offer-template.json", "usd-ledger-template.json", "checkpoint_store.py", "remote_backup.py", "run_guard.py", "vast_preflight.py", "cuda_smoke.py",
        "bootstrap_env.sh", "requirements-lock.txt", "requirements-gpu-lock.txt", "README.md", "SHA256SUMS.txt")),
    *(str(RUNTIME / "ops_v2" / name) for name in ("legacy-v1-pins/requirements-cu126.lock",
        "tests/test_budget_gate.py", "tests/test_budget_guard.py", "tests/test_checkpoint_store.py", "tests/test_remote_backup.py", "tests/test_run_guard.py")),
)


def assemble(destination: Path) -> dict[str, str]:
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("destination must be a new directory")
    destination.mkdir(parents=False, mode=0o700)
    hashes = {}
    for name in FILES:
        source = WORKSPACE / name
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"missing or unsafe code input: {name}")
        data = source.read_bytes()
        dest = destination / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as out:
            out.write(data)
        hashes[name.replace("\\", "/")] = hashlib.sha256(data).hexdigest()
    (destination / "code-only-manifest.json").write_text(json.dumps({
        "schema": "abc-vast-code-only-v1", "files": hashes,
        "human_data_included": False, "bundle_included": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new local output directory")
    args = parser.parse_args(argv)
    hashes = assemble(args.destination)
    print(json.dumps({"destination": str(args.destination.absolute()), "file_count": len(hashes),
                      "human_data_included": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

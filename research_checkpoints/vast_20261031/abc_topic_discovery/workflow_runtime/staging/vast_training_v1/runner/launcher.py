"""Single-fit operator launcher. Run as ``python -m vast_training_v1.runner.launcher``.

The pinned key, bundle identity, resume digest, quote, ledger and backup mount
are operator-supplied outside the bundle. This launcher neither rents an
instance nor obtains provider billing. It does not establish data-use rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
from datetime import datetime, timezone

from ..data.bundle import open_prepared_train_bundle
from ..data.ed25519_verifier import Ed25519BundleVerifier
from ..ops_v2.budget_gate import BudgetGate, BudgetStop, BUDGET_POLICY
from ..ops_v2.checkpoint_store import backup_checkpoint, compare_resume, verify_restored_receipt, validate_backup_configuration
from ..ops_v2 import remote_backup
from . import trainer
from .checkpoint import sha256_file, ensure_private_directory


def _object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected an operator-supplied JSON object")
    return data


def _hash(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 pin")
    return value


class _Lease:
    def __init__(self, identity: str):
        self.bundle_identity_sha256 = identity
        self.expected_identity_sha256 = identity
        self.active = True
    def assert_live(self):
        if not self.active:
            raise PermissionError("run lease has ended")
    def abort(self):
        self.active = False


class LiveBudgetGate:
    """Conservative local lease clock; operator files remain billing inputs.

    The meter does not query Vast. The operator ledger must be refreshed at
    least hourly and include all externally posted campaign charges.
    """
    def __init__(self, quote_path: Path, ledger_path: Path, seconds_per_update: float,
                 *, shared_lease_clock=None, lease_clock_genesis=None):
        self.quote_path, self.ledger_path = quote_path, ledger_path
        self.seconds_per_update = seconds_per_update
        self.start = time.monotonic()
        self.last_core = self.last_final = 0.0
        initial = _object(ledger_path)
        self._check_operator_ledger(initial, max_age=3600 if shared_lease_clock else 60)
        stamp = datetime.fromisoformat(initial['as_of_utc'].replace('Z', '+00:00'))
        self.initial_elapsed = (float(initial["current_lease_elapsed_seconds"])
                                + (datetime.now(timezone.utc) - stamp).total_seconds())
        self.price_contract = self._price_contract(_object(quote_path))
        self.last_core = float(initial["committed_core_usd"])
        self.last_final = float(initial["committed_final_usd"])
        if bool(shared_lease_clock) != bool(lease_clock_genesis):
            raise BudgetStop("shared lease clock and its genesis pin are both required")
        self.shared_lease_clock = shared_lease_clock
        self.lease_clock_genesis = lease_clock_genesis
        self.last_shared_monotonic = None
        if shared_lease_clock:
            from . import lease_clock
            _, self.last_shared_monotonic = lease_clock.evaluate(shared_lease_clock, lease_clock_genesis, initial, mark=True)
        self.last_decision_json = None

    @property
    def budget_policy(self):
        return dict(BUDGET_POLICY)

    @staticmethod
    def _price_contract(quote):
        keys = ('provider', 'offer_id', 'currency', 'compute_usd_per_gpu_hour',
                'billing_increment_hours', 'minimum_billable_hours', 'limits')
        return json.dumps({key: quote.get(key) for key in keys}, sort_keys=True, allow_nan=False)

    @staticmethod
    def _check_operator_ledger(ledger: dict, *, max_age: int) -> None:
        if ledger.get("schema") != "vast-campaign-ledger-v1" or ledger.get("root_authenticated") is not True:
            raise BudgetStop("operator ledger is not authenticated")
        try:
            stamp = datetime.fromisoformat(ledger["as_of_utc"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("missing timezone")
            age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
            if not 0 <= age <= max_age:
                raise ValueError("operator ledger is stale")
            for field in ("current_lease_elapsed_seconds", "committed_core_usd", "committed_final_usd"):
                val = ledger[field]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0 <= val < float("inf"):
                    raise ValueError("ledger value is invalid")
        except (KeyError, ValueError, TypeError) as exc:
            raise BudgetStop("operator ledger is stale or malformed") from exc

    def before_update(self, *, remaining_updates: int) -> bool:
        ledger = _object(self.ledger_path)
        self._check_operator_ledger(ledger, max_age=3600)
        core, final = float(ledger["committed_core_usd"]), float(ledger["committed_final_usd"])
        if core < self.last_core or final < self.last_final:
            raise BudgetStop("operator ledger reduced previously recorded campaign charges")
        self.last_core, self.last_final = core, final
        if self.shared_lease_clock:
            from . import lease_clock
            elapsed, self.last_shared_monotonic = lease_clock.evaluate(self.shared_lease_clock,
                self.lease_clock_genesis, ledger, minimum_monotonic=self.last_shared_monotonic)
            ledger["current_lease_elapsed_seconds"] = elapsed
        else:
            ledger["current_lease_elapsed_seconds"] = max(float(ledger["current_lease_elapsed_seconds"]),
                                                           self.initial_elapsed + time.monotonic() - self.start)
        # Timestamp represents this local calculation, not a provider billing refresh.
        ledger["as_of_utc"] = datetime.now(timezone.utc).isoformat()
        quote = _object(self.quote_path)
        if self._price_contract(quote) != self.price_contract:
            raise BudgetStop('lease billing identity or rate changed; reconcile cumulative costs before continuing')
        gate = BudgetGate(quote, lambda: ledger, self.seconds_per_update)
        try:
            return gate.before_update(remaining_updates=remaining_updates)
        finally:
            self.last_decision_json = gate.last_decision_json


class TrainingController:
    """One process, one pinned recipe, and synchronous verified backup.

    The operator controls the independently supplied pins and files. This is a
    process-local consistency check, not an isolation or authorization boundary.
    """
    def __init__(self, *, recipe, identity, expected_runner, expected_model, run_id,
                 resume_digest, backup_dir=None, storage_attestation=None,
                 resume_manifest=None, resume_expected=None, remote_prefix=None,
                 resume_remote_manifest_sha256=None, remote_transport=None):
        self.capability_id = secrets.token_hex(16)
        self.recipe = recipe.as_dict()
        self.plan_sha = trainer.PILOT_DRAFT_SHA256 if isinstance(recipe, trainer.PilotRecipe) else trainer.THREE_ARM_DRAFT_SHA256
        if type(run_id) is not str or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("an independently preassigned 32-hex run ID is required")
        self.run_id = run_id
        self.recipe_sha = hashlib.sha256(trainer._canonical_json(self.recipe)).hexdigest()
        self.identity = _hash(identity)
        self.expected_runner = _hash(expected_runner)
        self.expected_model = _hash(expected_model)
        self.resume_digest = _hash(resume_digest) if resume_digest else None
        if (backup_dir is None) == (remote_prefix is None):
            raise ValueError("select exactly one local or rclone backup")
        if backup_dir is not None and storage_attestation is None:
            raise ValueError("local backup requires a storage attestation")
        if remote_prefix is not None and storage_attestation is not None:
            raise ValueError("rclone backup cannot use a local mount attestation")
        self.backup_dir = backup_dir
        self.storage_attestation = storage_attestation
        self.remote_prefix = remote_backup._remote_prefix(remote_prefix) if remote_prefix else None
        self.remote_transport = remote_transport if remote_transport is not None else (remote_backup.RcloneTransport() if remote_prefix else None)
        self.resume_remote_manifest_sha256 = _hash(resume_remote_manifest_sha256) if resume_remote_manifest_sha256 else None
        self.resume_manifest = resume_manifest
        self.resume_expected = resume_expected
        self.lease = _Lease(identity)
        self.backups = []
        if bool(resume_manifest) != bool(resume_expected) or bool(resume_manifest) != bool(resume_digest):
            raise ValueError("resume requires independent manifest, expected pins and payload digest")
        if bool(self.resume_remote_manifest_sha256) != bool(resume_manifest and self.remote_prefix):
            raise ValueError("remote resume requires an independently pinned remote manifest SHA-256")
        if resume_manifest:
            compared = compare_resume(resume_manifest, resume_expected)
            if compared["payload_sha256"] != self.resume_digest:
                raise PermissionError("resume backup does not match the external digest")
            if compared["metadata"]["run_id"] != self.run_id:
                raise PermissionError("resume run ID differs from the independently preassigned pin")
            if self.remote_prefix:
                remote_backup.verify_remote_copy(self.remote_prefix, resume_manifest,
                    self.resume_remote_manifest_sha256, transport=self.remote_transport)

    def authorize_verified_bundle(self, *, manifest, bundle_identity_sha256,
                                  expected_run_binding, expected_identity_sha256):
        self.lease.assert_live()
        if (expected_run_binding is not self.lease or bundle_identity_sha256 != self.identity
                or expected_identity_sha256 != self.identity
                or manifest["purpose"] != self.recipe["purpose"]
                or manifest["role"] != self.recipe["role"]
                or manifest["fold_id"] != self.recipe["fold_id"]):
            raise PermissionError("signed bundle scope differs from this fit")
        return self.lease

    def authorize_training(self, *, recipe, recipe_sha256, data_identity_sha256,
                           model_sha256, runner_sha256, resume_checkpoint_sha256,
                           completed_updates):
        self.lease.assert_live()
        if (recipe != self.recipe or recipe_sha256 != self.recipe_sha
                or data_identity_sha256 != self.identity or model_sha256 != self.expected_model
                or runner_sha256 != self.expected_runner
                or resume_checkpoint_sha256 != self.resume_digest):
            raise PermissionError("training or resume pins differ from the operator's frozen run")

    def checkpoint_committed(self, *, checkpoint_sha256, completed_updates,
                             data_identity_sha256, recipe_sha256, checkpoint_path, metadata_path):
        self.lease.assert_live()
        meta = _object(Path(metadata_path))
        if (data_identity_sha256 != self.identity or recipe_sha256 != self.recipe_sha
                or meta.get("payload_sha256") != checkpoint_sha256
                or meta.get("global_step") != completed_updates
                or meta.get("data_sha256") != self.identity
                or meta.get("code_sha256") != self.expected_runner
                or meta.get("model_sha256") != self.expected_model
                or meta.get("recipe_sha256") != self.recipe_sha
                or meta.get("plan_sha256") != self.plan_sha
                or meta.get("fold_id") != self.recipe["fold_id"]
                or meta.get("training_seed") != self.recipe["training_seed"]
                or meta.get("arm") != self.recipe["arm_id"]
                or not isinstance(meta.get("run_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", meta.get("run_id", ""))
                or meta.get("run_id") != self.run_id
                or sha256_file(Path(checkpoint_path)) != checkpoint_sha256):
            raise PermissionError("checkpoint metadata differs from current run")
        try:
            if self.remote_prefix:
                result = remote_backup.backup_checkpoint(Path(checkpoint_path), Path(metadata_path),
                    self.remote_prefix, transport=self.remote_transport)
                record = {"manifest_remote": result["manifest_remote"], "manifest_sha256": result["manifest_sha256"]}
            else:
                result = backup_checkpoint(Path(checkpoint_path), Path(metadata_path),
                                           self.backup_dir, self.storage_attestation)
                manifest = Path(result["manifest_path"])
                record = {"manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest)}
            if result.get("payload_sha256") != checkpoint_sha256:
                raise PermissionError("backup digest differs from checkpoint")
        except BaseException:
            self.lease.abort()
            raise
        self.backups.append({**record, "payload_sha256": checkpoint_sha256, "step": completed_updates,
                             "run_id": self.run_id, "plan_sha256": self.plan_sha,
                             "data_sha256": self.identity, "code_sha256": self.expected_runner,
                             "model_sha256": self.expected_model, "fold_id": self.recipe["fold_id"],
                             "training_seed": self.recipe["training_seed"], "arm": self.recipe["arm_id"]})
        print(json.dumps({"checkpoint_backup_verified": self.backups[-1]}, sort_keys=True), flush=True)

    def resume_restored(self, *, receipt_path, checkpoint_sha256, completed_updates):
        self.lease.assert_live()
        if not self.resume_manifest or checkpoint_sha256 != self.resume_digest:
            raise PermissionError("no independently pinned resume manifest")
        if self.remote_prefix:
            remote_backup.verify_remote_copy(self.remote_prefix, self.resume_manifest,
                self.resume_remote_manifest_sha256, transport=self.remote_transport)
        result = verify_restored_receipt(self.resume_manifest, self.resume_expected, Path(receipt_path))
        if result["loaded_global_step"] != completed_updates:
            raise PermissionError("restored step differs from backup")


def main(argv=None):
    launcher_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "screen"), required=True)
    parser.add_argument("--run-id", required=True, help="independently preassigned 32-hex run ID")
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=trainer.SCREEN_ARMS, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--public-key", type=Path, required=True, help="separately pinned raw 32-byte Ed25519 key")
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    backup = parser.add_mutually_exclusive_group(required=True)
    backup.add_argument("--backup-dir", type=Path)
    backup.add_argument("--backup-rclone-remote", help="configured remote prefix; no credentials")
    parser.add_argument("--storage-attestation", type=Path, help="required for local mount")
    parser.add_argument("--quote", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True, help="fresh cumulative ledger; reread before every update")
    parser.add_argument("--shared-lease-clock", type=Path)
    parser.add_argument("--lease-clock-genesis")
    parser.add_argument("--worst-case-seconds-per-update", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--resume-expected", type=Path)
    parser.add_argument("--resume-remote-manifest-sha256", help="independently retained remote manifest digest")
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--time-ledger", type=Path, required=True)
    parser.add_argument("--time-original", type=Path, required=True)
    parser.add_argument("--time-anchor", type=Path, required=True)
    parser.add_argument("--time-journal-remote", help="single-writer rclone prefix for time ledger journal")
    parser.add_argument("--time-journal-head", type=Path, help="locally retained, independently reconciled journal head")
    parser.add_argument("--reservation-seconds", type=int, required=True)
    parser.add_argument("--shutdown-seconds", type=float, required=True)
    parser.add_argument("--interrupt-grace-seconds", type=float, required=True)
    parser.add_argument("--term-grace-seconds", type=float, required=True)
    parser.add_argument("--kill-verify-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    if os.name != "posix" or not Path('/proc').is_dir():
        raise PermissionError("supervised launcher requires Linux /proc")
    if not args.time_anchor.is_file():
        raise PermissionError("independent time anchor is required")
    if bool(args.time_journal_remote) != bool(args.time_journal_head):
        parser.error("time journal remote and head must be supplied together")
    if args.backup_rclone_remote and not args.time_journal_remote:
        parser.error("rclone backup requires the off-host time ledger journal")
    if args.time_journal_head is not None and not args.time_journal_head.is_file():
        raise PermissionError("independent time journal head is required")
    checkpoint_root = ensure_private_directory(args.checkpoint_dir)
    if args.backup_dir is not None:
        if args.storage_attestation is None or not args.storage_attestation.is_file():
            raise PermissionError("local backup needs an independent storage attestation")
        validate_backup_configuration(checkpoint_root, args.backup_dir, args.storage_attestation)
    else:
        if args.storage_attestation is not None:
            parser.error("storage attestation applies only to local mount backups")
        remote_backup.RcloneTransport().check(args.backup_rclone_remote)
    trainer.require_ram_floor()
    trainer.configure_pre_torch_environment(require_cuda=True)
    if bool(args.resume) != bool(args.resume_sha256) or bool(args.resume) != bool(args.resume_receipt):
        parser.error("resume needs checkpoint, independent digest and new receipt path")
    recipe = trainer._make_recipe(args.mode, args.fold_id, args.seed, args.arm)
    controller = TrainingController(recipe=recipe, identity=args.bundle_sha256, run_id=args.run_id,
        expected_runner=args.runner_sha256, expected_model=args.model_sha256,
        resume_digest=args.resume_sha256, backup_dir=args.backup_dir,
        storage_attestation=args.storage_attestation, resume_manifest=args.resume_manifest,
        resume_expected=args.resume_expected, remote_prefix=args.backup_rclone_remote,
        resume_remote_manifest_sha256=args.resume_remote_manifest_sha256)
    usd_gate = LiveBudgetGate(args.quote, args.ledger, args.worst_case_seconds_per_update,
        shared_lease_clock=args.shared_lease_clock, lease_clock_genesis=args.lease_clock_genesis)
    # Refuse stale prices, incorrect limits and insufficient USD before any
    # time debit, bundle read, CUDA import or remote journal publication.
    usd_gate.before_update(remaining_updates=recipe.updates)
    from ..time_budget.bootstrap import AnchoredReservation
    from ..time_budget.supervisor_linux import ContainmentFailure, supervise
    session = AnchoredReservation(args.time_ledger, args.time_original, args.time_anchor,
                                  secrets.token_hex(16), args.reservation_seconds,
                                  started_monotonic=launcher_started)

    def worker(gate):
        if trainer._runner_sha256() != controller.expected_runner:
            raise PermissionError("runner differs from the independently reviewed pin")
        if trainer._model_binding(recipe.arm_id)[1] != controller.expected_model:
            raise PermissionError("model differs from the independently reviewed pin")
        verifier = Ed25519BundleVerifier(pinned_public_keys={args.key_id: args.public_key.read_bytes()}, authority=controller)
        bundle = open_prepared_train_bundle(args.bundle_dir, trusted_verifier=verifier,
            expected_fold_id=recipe.fold_id, expected_run_binding=controller.lease,
            expected_identity_sha256=controller.identity)
        report = trainer.run_fit(bundle=bundle, root_controller=controller, budget_gate=gate,
            recipe=recipe, checkpoint_directory=args.checkpoint_dir, resume_checkpoint=args.resume,
            resume_checkpoint_sha256=args.resume_sha256, resume_receipt_path=args.resume_receipt,
            expected_run_id=controller.run_id)
        if report.get("run_id") != controller.run_id:
            raise PermissionError("trainer run ID differs from the locally pinned checkpoint")
        print(json.dumps({"fit": report, "verified_backups": controller.backups,
                          "provider_billing_enforced": False}, sort_keys=True), flush=True)
        return 0

    try:
        if args.time_journal_remote:
            from ..time_budget.remote_journal import publish
            journal_head = publish(session, args.time_anchor, args.time_journal_head, args.time_journal_remote)
            print(json.dumps({"time_journal_published_sha256": journal_head}, sort_keys=True), flush=True)
        result = supervise(session, usd_gate, worst_update_seconds=args.worst_case_seconds_per_update,
            shutdown_seconds=args.shutdown_seconds, interrupt_grace=args.interrupt_grace_seconds,
            term_grace=args.term_grace_seconds, kill_verify=args.kill_verify_seconds, worker=worker)
    except ContainmentFailure:
        print("STOP HOST: group survived; retaining reservation lock until host is stopped", file=__import__('sys').stderr, flush=True)
        while True:
            time.sleep(60)
    except BaseException:
        session.close()
        raise
    session.close()
    print(json.dumps({"time_supervisor": result, "provider_billing_enforced": False}, sort_keys=True))
    if result['timed_out']:
        return 124
    if result['external_stop_signal']:
        return 128 + result['external_stop_signal']
    return 125 if result['orphaned'] else result['exit_code']


if __name__ == "__main__":
    raise SystemExit(main())

"""List fixed screen fits or generate one unexecuted launcher command."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import secrets

from .screen_plan import ARMS, FOLDS, UPDATES_PER_FIT, TOTAL_FITS, screen_fits


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("every runner/model/bundle pin must be a lowercase SHA-256")
    return value


def _path(value):
    if not isinstance(value, str) or not PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError("all target-machine paths must be absolute POSIX paths")
    return value


def generate_run_ids():
    """Generate new IDs locally, before any campaign execution or transfer."""
    result = {}
    for fit in screen_fits():
        name = f"fold-{fit['fold_id']}-seed-{fit['training_seed']}-{fit['arm_id']}"
        run_id = secrets.token_hex(16)
        if run_id in result.values():
            raise RuntimeError("unexpected run ID collision; generate a new set")
        result[name] = run_id
    return result


def build_plan(config: dict, *, fit_id: str | None = None) -> dict:
    required = {"bundles", "model_sha256", "runner_sha256", "checkpoint_root", "run_ids", "python_executable",
                "quote", "ledger", "worst_case_seconds_per_update", "per_fit_overhead_seconds",
                "time_ledger", "time_original", "time_anchor", "reservation_seconds",
                "shutdown_seconds", "interrupt_grace_seconds", "term_grace_seconds", "kill_verify_seconds"}
    if not isinstance(config, dict): raise ValueError("campaign config must be an object")
    local = set(config) == required | {"backup_dir", "storage_attestation"}
    remote = set(config) == required | {"backup_rclone_remote", "time_journal_remote", "time_journal_head"}
    if not (local or remote):
        raise ValueError("campaign config must select exactly one documented backup route")
    if remote:
        from ..ops_v2.remote_backup import _remote_prefix
        _remote_prefix(config["backup_rclone_remote"])
        _remote_prefix(config["time_journal_remote"])
        _path(config["time_journal_head"])
    bundles, models = config["bundles"], config["model_sha256"]
    if not isinstance(bundles, dict) or set(bundles) != {str(f) for f in FOLDS}:
        raise ValueError("exactly three fold bundles are required")
    if not isinstance(models, dict) or set(models) != set(ARMS):
        raise ValueError("exactly three arm model pins are required")
    for sha in models.values():
        _sha(sha)
    runner = _sha(config["runner_sha256"])
    for bundle in bundles.values():
        if not isinstance(bundle, dict) or set(bundle) != {"directory", "identity_sha256", "public_key", "key_id"}:
            raise ValueError("bundle entry requires directory, identity, public key and key id")
        _path(bundle["directory"]); _path(bundle["public_key"]); _sha(bundle["identity_sha256"])
        if not isinstance(bundle["key_id"], str) or not bundle["key_id"]:
            raise ValueError("bundle key id is required")
    for field in ("checkpoint_root", "quote", "ledger", "time_ledger", "time_original", "time_anchor", "python_executable"):
        _path(config[field])
    python = PurePosixPath(config['python_executable'])
    if (python.parent.name != 'bin' or python.parent.parent.name != '.venv'
            or python.name not in ('python', 'python3', 'python3.11')):
        raise ValueError('python_executable must name the selected .venv/bin/python')
    if local:
        _path(config["backup_dir"]); _path(config["storage_attestation"])
    seconds = config["worst_case_seconds_per_update"]
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not 0 < seconds < float("inf"):
        raise ValueError("worst-case seconds per update must be a measured positive finite bound")
    overhead = config['per_fit_overhead_seconds']
    if type(overhead) not in (int, float) or not math.isfinite(overhead) or overhead <= 0:
        raise ValueError('per_fit_overhead_seconds must bound initialization, all backup and verification I/O')
    reservation = config["reservation_seconds"]
    if type(reservation) is not int or not 0 < reservation <= 12450:
        raise ValueError("reservation must be whole positive seconds within the historical non-export remainder")
    durations = {}
    for field in ("shutdown_seconds", "interrupt_grace_seconds", "term_grace_seconds", "kill_verify_seconds"):
        value = config[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
        durations[field] = value
    if durations["shutdown_seconds"] <= 0 or seconds + durations["shutdown_seconds"] >= reservation:
        raise ValueError("reservation must fit an update and shutdown allowance")
    if UPDATES_PER_FIT * seconds + overhead + durations['shutdown_seconds'] >= reservation:
        raise ValueError('full fit, all backup I/O and shutdown do not fit the per-process reservation')
    if TOTAL_FITS * reservation > 12450:
        raise ValueError('all 18 permanent reservations exceed the maximum historical non-export remainder')
    if sum(durations[k] for k in ("interrupt_grace_seconds", "term_grace_seconds", "kill_verify_seconds")) > durations["shutdown_seconds"]:
        raise ValueError("signal grace periods must fit shutdown allowance")
    paths = [PurePosixPath(config[k]) for k in ("time_ledger", "time_original", "time_anchor", "ledger")]
    if len(set(paths)) != len(paths) or paths[2].parent == paths[0].parent:
        raise ValueError("time ledger, history, anchor and USD ledger must be distinct; anchor needs a separate directory")
    fits = []
    for fit in screen_fits():
        fold, seed, arm = fit["fold_id"], fit["training_seed"], fit["arm_id"]
        name = f"fold-{fold}-seed-{seed}-{arm}"
        fits.append({"fit": name, "fold_id": fold, "training_seed": seed, "arm_id": arm})
    run_ids = config["run_ids"]
    if (type(run_ids) is not dict or set(run_ids) != {row["fit"] for row in fits}
            or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{32}", value)
                   for value in run_ids.values()) or len(set(run_ids.values())) != len(fits)):
        raise ValueError("run_ids must preassign one unique 32-hex ID per fixed fit")
    plan = {"schema": "abc-vast-screen-command-plan-v2", "status": "NOT_EXECUTED",
        "fits": fits, "shared_campaign_budget_usd": 16, "training_soft_stop_usd": 4,
        "reserve_usd": 12, "automatic_resume_or_skip": False,
        "full_campaign_reserved_seconds": TOTAL_FITS * reservation,
        "warning": "Generate one fit at a time; launcher reads the current locked anchor and permanently reserves each process. Reconcile cumulative time and USD, refresh billing and verify backups. No command grants a run."}
    if fit_id is not None:
        selected = next((row for row in fits if row["fit"] == fit_id), None)
        if selected is None:
            raise ValueError("fit ID is outside the locked screen")
        fold, seed, arm = selected["fold_id"], selected["training_seed"], selected["arm_id"]
        bundle = bundles[str(fold)]
        name = selected["fit"]
        checkpoint = str(PurePosixPath(config["checkpoint_root"]) / name)
        cmd = [config["python_executable"], "-m", "vast_training_v1.runner.launcher", "--mode", "screen", "--run-id", run_ids[name],
            "--fold-id", str(fold), "--seed", str(seed), "--arm", arm,
            "--bundle-dir", bundle["directory"], "--bundle-sha256", bundle["identity_sha256"],
            "--public-key", bundle["public_key"], "--key-id", bundle["key_id"],
            "--runner-sha256", runner, "--model-sha256", models[arm], "--checkpoint-dir", checkpoint,
            *( ["--backup-dir", config["backup_dir"], "--storage-attestation", config["storage_attestation"]]
               if local else ["--backup-rclone-remote", config["backup_rclone_remote"],
                    "--time-journal-remote", config["time_journal_remote"],
                    "--time-journal-head", config["time_journal_head"]] ),
            "--quote", config["quote"], "--ledger", config["ledger"],
            "--worst-case-seconds-per-update", str(seconds),
            "--time-ledger", config["time_ledger"], "--time-original", config["time_original"],
            "--time-anchor", config["time_anchor"], "--reservation-seconds", str(reservation),
            "--shutdown-seconds", str(durations["shutdown_seconds"]),
            "--interrupt-grace-seconds", str(durations["interrupt_grace_seconds"]),
            "--term-grace-seconds", str(durations["term_grace_seconds"]),
            "--kill-verify-seconds", str(durations["kill_verify_seconds"])]
        plan["selected_command"] = {"fit": name, "argv": cmd,
            "anchor_binding": "read_current_at_launch_under_lock", "reservation": "new_per_process_no_refund"}
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path, help="new manifest file; no command is executed")
    parser.add_argument("--fit", help="select exactly one fit ID; omit to list fits only")
    args = parser.parse_args(argv)
    plan = build_plan(json.loads(args.config.read_text(encoding="utf-8")), fit_id=args.fit)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Generated {'one unexecuted command' if args.fit else 'fit inventory'} in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

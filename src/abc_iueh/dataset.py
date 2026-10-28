"""Motion-only CSV ingestion, HDF5 storage and user-level split manifests."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from .config import PipelineConfig
from .preprocess import prepare_session
from .sources import (_authenticate_inventory, _session_inventory, _validate_download_receipt,
                      _reject_recording_aliases, _validated_record, _verify_record,
                      sha256_file, utc_now, write_json)

QUATERNION_COLUMNS = ("hmd_rot_w", "hmd_rot_x", "hmd_rot_y", "hmd_rot_z")


def read_alyx_motion(paths):
    """Select named columns only; never infer scalar order from CSV position."""
    times, orientations = [], []
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if len(columns) != len(set(columns)):
                raise ValueError("duplicate CSV column names")
            required = {"delta_time_ms", *QUATERNION_COLUMNS}
            if not required.issubset(columns):
                raise ValueError("missing named HMD quaternion or elapsed-time columns")
            for row in reader:
                if None in row:
                    raise ValueError("CSV row contains more fields than its header")
                try:
                    times.append(float(row["delta_time_ms"]) / 1000.0)
                except (ValueError, TypeError):
                    times.append(float("nan"))
                try:
                    orientations.append(tuple(float(row[key]) for key in QUATERNION_COLUMNS))
                except (ValueError, TypeError):
                    orientations.append((float("nan"),) * 4)
    return np.asarray(times, dtype=np.float64), np.asarray(orientations, dtype=np.float64).reshape(-1, 4)


def _same_file_if_present(left, right):
    """Compare filesystem identity without opening either file's contents."""
    try:
        return Path(left).samefile(Path(right))
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError("cannot verify output identity against protected provenance files") from exc


def _guard_ingest_write_paths(write_paths, protected_paths):
    """Keep every ingest write target clear of inputs and immutable snapshots."""
    named_writes = [(label, Path(path)) for label, path in write_paths]
    named_protected = [(label, Path(path)) for label, path in protected_paths]
    for index, (label, path) in enumerate(named_writes):
        resolved = path.resolve()
        for other_label, other in named_writes[:index]:
            if resolved == other.resolve() or _same_file_if_present(path, other):
                raise ValueError(f"{label} and {other_label} must be distinct write targets")
        for protected_label, protected in named_protected:
            if resolved == protected.resolve() or _same_file_if_present(path, protected):
                raise ValueError(f"{label} must not overwrite {protected_label}")


def ingest_alyx(data_root, output, report, config: PipelineConfig, *, inventory_path, receipt_path=None):
    """Authenticate inventory and receipt metadata before opening any recording."""
    data_root, output, report = Path(data_root), Path(output), Path(report)
    inventory_path = Path(inventory_path)
    receipt_path = Path(receipt_path) if receipt_path is not None else data_root / "download_receipt.json"
    if inventory_path.resolve() == receipt_path.resolve():
        raise ValueError("inventory and receipt inputs must be distinct metadata files")
    _reject_recording_aliases((("inventory", inventory_path), ("receipt", receipt_path)), data_root)
    inventory_bytes = inventory_path.read_bytes()
    inventory_document = json.loads(inventory_bytes)
    inventory, record_digest = _authenticate_inventory(inventory_document, data_root)
    inventory_file_digest = hashlib.sha256(inventory_bytes).hexdigest()
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    records = _validate_download_receipt(receipt, inventory=inventory, record_digest=record_digest,
                                         manifest_digest=inventory_file_digest)
    groups = defaultdict(list)
    for record in records:
        groups[(record["user_id"], record["session_id"])].append(record)
    if not groups:
        raise ValueError("no downloaded sessions")
    inventory_sessions = _session_inventory(inventory)
    temporary = output.with_name(output.name + ".part")
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    receipt_version = receipt.get("schema_version", 1)
    quality = {"schema_version": 1, "created_utc": utc_now(), "source": "whoisalyx",
               "license": "CC-BY-NC-SA-4.0", "config": config.to_dict(),
               "source_receipt_sha256": receipt_digest,
               "source_receipt_schema_version": receipt_version,
               "source_inventory_file_sha256": inventory_file_digest,
               "source_record_inventory_sha256": record_digest, "sessions": []}
    provenance_root = output.parent / "provenance"
    provenance = provenance_root / ("receipt_" + receipt_digest + ".json")
    inventory_snapshot = provenance_root / ("inventory_" + inventory_file_digest + ".json")
    recording_paths = [_validated_record(r, data_root) for r in inventory]
    protected_paths = [("source inventory", inventory_path), ("source receipt", receipt_path),
                       *(("source recording", path) for path in recording_paths),
                       ("receipt snapshot", provenance), ("inventory snapshot", inventory_snapshot)]
    _guard_ingest_write_paths((("HDF5 output", output), ("quality report", report),
                               ("temporary HDF5", temporary)), protected_paths)
    _reject_recording_aliases((
        ("HDF5 output", output), ("quality report", report), ("temporary HDF5", temporary),
        ("receipt snapshot", provenance), ("inventory snapshot", inventory_snapshot),
    ), data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    if provenance.exists() and provenance.read_bytes() != receipt_bytes:
        raise ValueError("receipt snapshot checksum collision or corrupted provenance")
    if not provenance.exists():
        provenance.write_bytes(receipt_bytes)
    if inventory_snapshot.exists() and inventory_snapshot.read_bytes() != inventory_bytes:
        raise ValueError("inventory snapshot checksum collision or corrupted provenance")
    if not inventory_snapshot.exists():
        inventory_snapshot.write_bytes(inventory_bytes)
    started = time.perf_counter()
    try:
        with h5py.File(temporary, "w") as store:
            store.attrs["schema_version"] = 1
            store.attrs["source"] = "whoisalyx"
            store.attrs["license"] = "CC-BY-NC-SA-4.0"
            store.attrs["quaternion_order"] = "wxyz"
            store.attrs["coordinate_frame"] = "source-native OpenVR; no handedness transform"
            store.attrs["sample_rate_hz"] = config.target_rate_hz
            store.attrs["timing_dispute"] = "20 samples / 100 Hz / 100 ms unresolved"
            store.attrs["interpolation"] = "offline bracketing SLERP, not a zero-delay sensor pipeline"
            store.attrs["config_json"] = json.dumps(config.to_dict(), sort_keys=True)
            store.attrs["source_receipt_sha256"] = quality["source_receipt_sha256"]
            store.attrs["source_receipt_schema_version"] = receipt_version
            store.attrs["source_inventory_file_sha256"] = inventory_file_digest
            store.attrs["source_record_inventory_sha256"] = record_digest
            for (user, session), records in sorted(groups.items()):
                # A split recording is one source session, so trimming, filtering
                # and the duration cap are applied once, not once per file.
                records.sort(key=lambda r: (Path(r["relative_path"]).name != "vr-controllers.csv", r["source_path"]))
                expected = inventory_sessions[(user, session)]
                downloaded_paths = {r["source_path"] for r in records}
                complete = (isinstance(expected, list) and expected and
                            all(r.get("expected_session_paths") == expected for r in records) and
                            downloaded_paths == set(expected))
                if not complete:
                    quality["sessions"].append({"user_id": user, "session_id": session,
                        "source_files": len(records), "accepted": False,
                        "reason": "source_session_incomplete_or_unverified", "segments": 0, "output_samples": 0,
                        "stats": {"expected_source_files": len(expected) if expected else None,
                                  "downloaded_source_files": len(records)}})
                    continue
                paths = []
                for record in records:
                    path = _validated_record(record, data_root)
                    digest = _verify_record(path, record)
                    if digest != record["sha256"]:
                        raise ValueError("downloaded recording changed since its receipt")
                    paths.append(path)
                timestamps, q = read_alyx_motion(paths)
                prepared = prepare_session(timestamps, q, config)
                row = {"user_id": user, "session_id": session, "source_files": len(records),
                       "accepted": prepared.accepted, "reason": prepared.reason, "stats": prepared.stats}
                row["segments"] = len(prepared.segments)
                row["output_samples"] = sum(len(s.timestamps) for s in prepared.segments)
                quality["sessions"].append(row)
                if not prepared.accepted:
                    continue
                group = store.require_group(f"users/{user}/sessions/{session}")
                group.attrs["source_revision"] = records[0]["source_revision"]
                group.attrs["source_paths_json"] = json.dumps([r["source_path"] for r in records])
                group.attrs["stats_json"] = json.dumps(prepared.stats, allow_nan=False)
                for index, segment in enumerate(prepared.segments):
                    chunk = min(4096, len(segment.timestamps))
                    target = group.create_group(f"segments/{index:04d}")
                    target.create_dataset("timestamps", data=segment.timestamps, dtype="f8",
                                          chunks=(chunk,), compression="gzip", compression_opts=1)
                    target.create_dataset("quaternion", data=segment.quaternion, dtype="f4",
                                          chunks=(chunk, 4), compression="gzip", compression_opts=1)
            quality["eligible_users"] = sorted(store.get("users", {}).keys())
            quality["eligible_user_count"] = len(quality["eligible_users"])
            quality["confirmatory_data_gate_passed"] = len(quality["eligible_users"]) >= config.minimum_confirmatory_users
            quality["accepted_sessions"] = sum(s["accepted"] for s in quality["sessions"])
            quality["output_samples"] = sum(s["output_samples"] for s in quality["sessions"])
            store.attrs["quality_summary_json"] = json.dumps({k: quality[k] for k in
                ("eligible_user_count", "confirmatory_data_gate_passed", "accepted_sessions", "output_samples")})
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    quality["elapsed_seconds"] = time.perf_counter() - started
    quality["hdf5_bytes"] = output.stat().st_size
    quality["hdf5_sha256"] = sha256_file(output)
    write_json(report, quality)
    return quality


def user_inventory(hdf5_path):
    with h5py.File(hdf5_path, "r") as store:
        return sorted(store.get("users", {}).keys())


def create_user_split(hdf5_path, output, config: PipelineConfig, *, smoke=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("split manifest already exists; do not silently reselect users")
    users = user_inventory(hdf5_path)
    if not users:
        raise ValueError("no eligible users")
    ordered = sorted(users, key=lambda user: hashlib.sha256(f"{config.split_seed}:{user}".encode()).hexdigest())
    if smoke:
        partitions = {"smoke": ordered, "train": [], "development": [], "final": []}
    else:
        if len(users) < config.minimum_confirmatory_users:
            raise ValueError(f"confirmatory data gate needs {config.minimum_confirmatory_users} eligible users; got {len(users)}")
        final_count = max(config.minimum_final_users, round(0.43 * len(users)))
        development_count = config.development_users
        if final_count + development_count >= len(users):
            raise ValueError("split leaves no training users")
        partitions = {"final": ordered[:final_count],
                      "development": ordered[final_count:final_count + development_count],
                      "train": ordered[final_count + development_count:]}
    manifest = {"schema_version": 1, "created_utc": utc_now(), "unit": "user_before_windowing",
                "seed": config.split_seed, "mode": "smoke_only_no_training" if smoke else "locked_confirmatory",
                "training_allowed": not smoke, "holdout_pretraining_allowed": False,
                "source_hdf5_sha256": sha256_file(hdf5_path), "partitions": partitions}
    write_json(output, manifest)
    return manifest


def iter_windows(hdf5_path, split_path, partition="train", *, context=400, horizon=20,
                 stride=1, allow_final=False):
    if context != 400 or horizon != 20 or stride < 1:
        raise ValueError("preserve 400 -> 20 samples and a positive stride")
    if partition == "final" and not allow_final:
        raise PermissionError("final holdout access is disabled by default")
    manifest = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if manifest.get("source_hdf5_sha256") != sha256_file(hdf5_path):
        raise ValueError("HDF5 changed after user split was locked")
    partitions = manifest["partitions"]
    if partition not in partitions:
        raise ValueError("unknown split partition")
    seen = set()
    for users in partitions.values():
        if len(users) != len(set(users)) or seen.intersection(users):
            raise ValueError("user leakage across split partitions")
        seen.update(users)
    if partition == "train" and not manifest.get("training_allowed"):
        raise PermissionError("smoke manifests cannot be used for training")
    with h5py.File(hdf5_path, "r") as store:
        if seen != set(store["users"].keys()):
            raise ValueError("split inventory does not match HDF5 users")
        for user in partitions[partition]:
            for session_id, session in store[f"users/{user}/sessions"].items():
                for segment_id, group in session["segments"].items():
                    # One segment resident in RAM, no materialized overlapping
                    # window table and no context crossing session/gap borders.
                    q = group["quaternion"][:]
                    q.setflags(write=False)
                    for start in range(0, len(q) - context - horizon + 1, stride):
                        yield {"user_id": user, "session_id": session_id,
                               "segment_id": segment_id, "start": start,
                               "context": q[start:start + context],
                               "target": q[start + context:start + context + horizon]}


def check_hdf5(hdf5_path, config: PipelineConfig):
    """Dataset correctness only; never report forecast errors on smoke users."""
    summary = {"users": 0, "sessions": 0, "segments": 0, "samples": 0, "windows": 0,
               "maximum_norm_error": 0.0, "minimum_adjacent_dot": 1.0}
    with h5py.File(hdf5_path, "r") as store:
        if store.attrs["quaternion_order"] != "wxyz":
            raise ValueError("wrong quaternion convention")
        for user in store.get("users", {}).values():
            summary["users"] += 1
            for session in user["sessions"].values():
                summary["sessions"] += 1
                for segment in session["segments"].values():
                    timestamps, q = segment["timestamps"][:], segment["quaternion"][:]
                    if q.dtype != np.float32 or q.shape != (len(timestamps), 4):
                        raise ValueError("wrong quaternion dtype or dimensions")
                    if not np.isfinite(q).all() or not np.isfinite(timestamps).all():
                        raise ValueError("nonfinite stored samples")
                    if not np.allclose(np.diff(timestamps), 1 / config.target_rate_hz, atol=1e-8, rtol=0):
                        raise ValueError("nonuniform stored clock")
                    norm_error = float(np.max(np.abs(np.linalg.norm(q, axis=-1) - 1)))
                    dots = np.sum(q[1:] * q[:-1], axis=-1)
                    minimum_dot = float(dots.min()) if len(dots) else 1.0
                    if norm_error > 2e-6 or minimum_dot < -1e-6:
                        raise ValueError("nonunit or hemisphere-discontinuous quaternion")
                    summary["maximum_norm_error"] = max(summary["maximum_norm_error"], norm_error)
                    summary["minimum_adjacent_dot"] = min(summary["minimum_adjacent_dot"], minimum_dot)
                    summary["segments"] += 1
                    summary["samples"] += len(q)
                    summary["windows"] += max(0, len(q) - config.context_samples - config.horizon_samples + 1)
    return summary

"""Build a bounded BOXRR pilot with immutable per-user shards and an 80/20 user split."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

import bson
from filelock import FileLock
import fpzip
import h5py
import numpy as np

from .boxrr import head_rotation, remove_startup_clock_duplicates
from .boxrr_source import (REPO, REVISION, UserArchive, download_archive, file_sha256,
                           http_session, official_archives, authorization_reference)
from .config import PipelineConfig
from .preprocess import prepare_session

GIB = 1024**3
CONFIG = PipelineConfig()
MAX_MEMBER = 64 * 1024**2
MAX_USER_SAMPLES = 16_000_000
SEED = 20261026
ADAPTER_POLICY = "xror_v2_remove_only_repeated_initial_timestamp"


def json_write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _sessions(archive, user):
    seen = set()
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            parts = PurePosixPath(member.name)
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile() or parts.suffix != ".xror":
                raise ValueError("Unexpected TAR member type")
            if len(parts.parts) != 2 or parts.parts[0] != user or parts.stem in seen:
                raise ValueError("Recording user or unique session identity differs")
            seen.add(parts.stem)
            if not 0 < member.size <= MAX_MEMBER:
                yield parts.stem, None, "recording_size_limit"
                continue
            with tar.extractfile(member) as stream:
                packed = stream.read(member.size + 1)
            if len(packed) != member.size:
                raise ValueError("Truncated XROR member")
            try:
                doc = bson.decode(packed)
                frames = fpzip.decompress(doc["frames"])
            except (bson.errors.BSONError, fpzip.FpzipError, KeyError, TypeError, ValueError):
                yield parts.stem, None, "invalid_bson_or_fpzip"
                continue
            if frames.ndim != 4 or frames.shape[:2] != (1, 1):
                yield parts.stem, None, "invalid_frame_dimensions"
                continue
            doc["frames"] = frames[0, 0]
            yield parts.stem, doc, None


def _write_user(archive, user, shard):
    counts = Counter()
    sessions = []
    samples = 0
    with h5py.File(shard, "x") as store:
        store.attrs.update(source="BOXRR-23", quaternion_order="wxyz", sample_rate_hz=100.,
                           source_revision=REVISION, config_json=json.dumps(CONFIG.to_dict()),
                           adapter_policy=ADAPTER_POLICY,
                           coordinate_frame="source-native XROR Unity; no handedness conversion",
                           timing_dispute="20 samples / 100 Hz / 100 ms unresolved")
        group = store.create_group("sessions")
        for session_id, doc, reason in _sessions(archive, user):
            if reason:
                counts[reason] += 1
                continue
            try:
                t, q = head_rotation(doc)
            except (KeyError, ValueError, TypeError):
                counts["missing_or_invalid_head_schema"] += 1
                continue
            t, q, removed = remove_startup_clock_duplicates(t, q)
            counts["startup_duplicate_rows_removed"] += removed
            result = prepare_session(t, q, CONFIG)
            result.stats["startup_duplicate_rows_removed"] = removed
            if not result.accepted:
                counts[result.reason] += 1
                continue
            if samples + result.stats["output_samples"] > MAX_USER_SAMPLES:
                counts["user_sample_cap"] += 1
                break
            counts["accepted"] += 1
            session = group.create_group(session_id)
            session.attrs["quality"] = json.dumps(result.stats)
            segments = session.create_group("segments")
            for index, item in enumerate(result.segments):
                segment = segments.create_group(f"{index:06d}")
                segment.create_dataset("quaternion", data=item.quaternion, chunks=True, compression="gzip", shuffle=True)
                segment.create_dataset("timestamps", data=item.timestamps, chunks=True, compression="gzip", shuffle=True)
                samples += len(item.quaternion)
            sessions.append(session_id)
    return {"samples": samples, "sessions": sessions, "reasons": dict(counts)}


def _cached_user(root, entry, fetch):
    user_dir = root / "users" / entry.user
    if user_dir.exists():
        row = json.loads((user_dir / "record.json").read_bytes())
        if row["archive"] != asdict(entry):
            raise ValueError("Archive identity changed on resume")
        if row["samples"] and file_sha256(user_dir / "data.h5") != row["sha256"]:
            raise ValueError("Cached shard was modified")
        return row
    with tempfile.TemporaryDirectory(prefix=".user-", dir=root) as tmp:
        work = Path(tmp)
        archive = work / "source.tar"
        fetch(entry, archive)
        if archive.stat().st_size != entry.size or file_sha256(archive) != entry.sha256:
            raise ValueError("Archive bytes differ from official metadata")
        user_out = work / "complete"
        user_out.mkdir()
        shard = user_out / "data.h5"
        result = _write_user(archive, entry.user, shard)
        if result["samples"]:
            result.update(bytes=shard.stat().st_size, sha256=file_sha256(shard))
        else:
            shard.unlink()
            result.update(bytes=0, sha256=None)
        result["archive"] = asdict(entry)
        json_write(user_out / "record.json", result)
        user_out.rename(user_dir)
    return result


def prepare(entries, fetch, root, *, source, authorization_sha256=None, minimum_bytes=6 * GIB,
            maximum_bytes=8 * GIB, minimum_users=30):
    """Use source=synthetic only with generated fixtures. Bind real intake to authorization."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".prepare.lock"), timeout=0):
        binding = {"source": source, "revision": REVISION, "config": CONFIG.to_dict(),
                   "adapter_policy": ADAPTER_POLICY,
                   "authorization_sha256": authorization_sha256, "minimum_bytes": minimum_bytes,
                   "maximum_bytes": maximum_bytes, "minimum_users": minimum_users,
                   "split_seed": SEED, "max_user_samples": MAX_USER_SAMPLES}
        if source not in ("BOXRR-23", "synthetic") or not 0 <= minimum_bytes <= maximum_bytes:
            raise ValueError("Invalid pilot binding")
        if source == "BOXRR-23" and not authorization_sha256:
            raise PermissionError("Operator authorization reference missing")
        config_path = root / "intake.json"
        if config_path.exists():
            if json.loads(config_path.read_bytes()) != binding:
                raise ValueError("Pilot intake settings changed on resume")
        else:
            json_write(config_path, binding)
        if (root / "manifest.json").exists():
            return verify(root)
        (root / "users").mkdir(exist_ok=True)
        rows, total, users, sessions = [], 0, set(), set()
        for entry in entries:
            if entry.user in users:
                raise ValueError("Duplicate user in source listing")
            users.add(entry.user)
            if shutil.disk_usage(root).free < entry.size + 2 * GIB:
                raise OSError("Insufficient disk for bounded intake")
            result = _cached_user(root, entry, fetch)
            if not result["samples"]:
                continue
            if sessions.intersection(result["sessions"]):
                raise ValueError("A session occurs under different users")
            if total + result["bytes"] > maximum_bytes:
                break
            sessions.update(result["sessions"])
            rows.append({"user": entry.user, "path": f"users/{entry.user}/data.h5",
                         "source_archive": asdict(entry),
                         **{name: result[name] for name in ("bytes", "sha256", "samples")}})
            total += result["bytes"]
            print(f"Accepted users={len(rows)} bytes={total}", flush=True)
            if total >= minimum_bytes and len(rows) >= minimum_users:
                break
        if total < minimum_bytes or len(rows) < minimum_users:
            raise ValueError(f"Pilot incomplete: {total} bytes, {len(rows)} users. Intake is resumable.")
        ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{SEED}:{row['user']}".encode()).digest())
        validation_count = max(1, round(len(rows) * .2))
        if validation_count >= len(rows):
            raise ValueError("Pilot must leave training users")
        for index, row in enumerate(ordered):
            row["partition"] = "validation" if index < validation_count else "train"
        manifest = {"schema": "abc-boxrr-pilot-v1", "source": source, "source_repo": REPO,
                    "source_revision": REVISION, "split_unit": "user", "split_seed": SEED,
                    "config": CONFIG.to_dict(), "adapter_policy": ADAPTER_POLICY, "bytes": total, "shards": ordered,
                    "training_allowed": source == "BOXRR-23", "authorization_sha256": authorization_sha256}
        temp_manifest = root / "manifest.part.json"
        temp_manifest.unlink(missing_ok=True)
        json_write(temp_manifest, manifest)
        temp_manifest.rename(root / "manifest.json")
        return verify(root)


def verify(root):
    root = Path(root)
    doc = json.loads((root / "manifest.json").read_bytes())
    if (doc["schema"] != "abc-boxrr-pilot-v1" or doc["source"] not in ("BOXRR-23", "synthetic") or
            doc["source_repo"] != REPO or doc["source_revision"] != REVISION or
            doc["split_unit"] != "user" or doc["split_seed"] != SEED or doc["config"] != CONFIG.to_dict() or
            doc["adapter_policy"] != ADAPTER_POLICY or
            doc["training_allowed"] != (doc["source"] == "BOXRR-23")):
        raise ValueError("Pilot contract differs")
    seen, partitions, total = set(), set(), 0
    for row in doc["shards"]:
        source_archive = UserArchive(**row["source_archive"])
        if source_archive.user != row["user"]:
            raise ValueError("Shard user differs from its source archive")
        if row["user"] in seen or row["partition"] not in ("train", "validation"):
            raise ValueError("User leakage or invalid partition")
        seen.add(row["user"])
        partitions.add(row["partition"])
        expected = f"users/{row['user']}/data.h5"
        path = root / row["path"]
        if (row["path"] != expected or ".." in PurePosixPath(row["path"]).parts or
                not path.resolve().is_relative_to(root.resolve()) or path.is_symlink()):
            raise ValueError("Unsafe shard path")
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError("Shard size or SHA-256 differs")
        total += row["bytes"]
    if total != doc["bytes"] or partitions != {"train", "validation"}:
        raise ValueError("Incomplete pilot")
    order = sorted(seen, key=lambda user: hashlib.sha256(f"{SEED}:{user}".encode()).digest())
    validation = set(order[:max(1, round(len(order) * .2))])
    if any((row["partition"] == "validation") != (row["user"] in validation) for row in doc["shards"]):
        raise ValueError("User split differs from the fixed seed")
    return doc


def windows(root, partition, stride=1):
    """Window within a single gap-free segment. Shard this iterator across workers externally."""
    if partition not in ("train", "validation") or type(stride) is not int or stride <= 0:
        raise ValueError("Invalid partition or stride")
    doc = verify(root)
    for row in doc["shards"]:
        if row["partition"] != partition:
            continue
        with h5py.File(Path(root) / row["path"], "r") as store:
            for session in store["sessions"].values():
                for segment in session["segments"].values():
                    q = segment["quaternion"][:]
                    q.setflags(write=False)
                    for start in range(0, len(q) - 419, stride):
                        yield q[start:start + 400], q[start + 400:start + 420]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--authorization-record", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.destination), indent=2))
        return
    authorization = authorization_reference(args.authorization_record)
    with http_session() as session:
        prepare(official_archives(session),
                lambda entry, path: download_archive(session, entry, path),
                args.destination, source="BOXRR-23", authorization_sha256=authorization)


if __name__ == "__main__":
    main()

"""Pinned, bounded WhoIsAlyx downloads; BOXRR is deliberately not enabled."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE_REPO = "cschell/who-is-alyx"
SOURCE_REVISION = "5da510121f99fc8a6cc51e570770ca93e5745646"
# SHA-256 of the sorted (source path, SHA-256 of raw DVC pointer bytes) inventory.
# This anchors the reviewed metadata independently of upstream Git object IDs.
POINTER_INVENTORY_SHA256 = "a3a8cdcdf2a70ce6a28b0a09c7c688568bb457d7e2f8faae1dd3dc233932ecd9"
# SHA-256 of the sorted (source path, byte count, DVC MD5) tuples consumed by
# the downloader/ingester. This also authenticates the fields parsed from old
# manifests that predate raw-pointer SHA-256 fields.
RECORD_INVENTORY_SHA256 = "284bb0b566399ece3f62dd7db7572eff23cada3cf8e1ca97cf707e16e2fe2bcd"
DOWNLOAD_RECEIPT_SCHEMA_VERSION = 2
REMOTE = "https://downloads.hci.informatik.uni-wuerzburg.de/cschell/who-is-alyx/"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
MOTION_PATTERN = re.compile(r"^players/(\d+)/([^/]+)/(vr-controllers[^/]*\.csv)\.dvc$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json_new(path, value):
    """Create a new metadata receipt without replacing any prior attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_get(url, limit=8 * 1024 * 1024):
    request = Request(url, headers={"User-Agent": "ABC-IUEH-research/0.1"})
    with urlopen(request, timeout=45) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("metadata response exceeds configured size limit")
    return content


def _verify_pointer_inventory(records):
    entries = sorted(({"path": r["source_path"], "sha256": r["pointer_sha256"]}
                      for r in records), key=lambda item: item["path"])
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != POINTER_INVENTORY_SHA256:
        raise ValueError("DVC pointer inventory differs from the reviewed SHA-256 snapshot")


def _record_inventory_digest(records):
    rows = [{key: record[key] for key in ("source_path", "expected_bytes", "md5")}
            for record in records]
    rows.sort(key=lambda row: row["source_path"])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verify_record_inventory(records):
    """Verify the full reviewed parsed-pointer inventory, not a caller hash."""
    if not isinstance(records, list) or not records:
        raise ValueError("reviewed recording inventory must be a nonempty list")
    paths = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("recording inventory entries must be objects")
        _validate_record_identity(record)
        paths.append(record["source_path"])
    if len(paths) != len(set(paths)):
        raise ValueError("recording inventory contains duplicate source paths")
    digest = _record_inventory_digest(records)
    if digest != RECORD_INVENTORY_SHA256:
        raise ValueError("recording inventory differs from the reviewed consumed-field snapshot")
    return digest


def _validate_record_identity(record):
    """Validate provenance-derived fields without touching recording bytes."""
    if not isinstance(record, dict) or record.get("source") != "whoisalyx":
        raise PermissionError("Only the reviewed WhoIsAlyx source is enabled; BOXRR remains blocked")
    match = MOTION_PATTERN.fullmatch(record.get("source_path", ""))
    if not match or not 42 <= int(match[1]) <= 76 or record.get("source_revision") != SOURCE_REVISION:
        raise ValueError("record is outside the reviewed motion cohort/revision")
    user, session, filename = match.groups()
    if record.get("user_id") != f"alyx_{int(user):02d}" or record.get("session_id") != session:
        raise ValueError("recording/user/session provenance mismatch")
    if record.get("relative_path") != str(PurePosixPath("raw", "whoisalyx", user, session, filename)):
        raise ValueError("recording destination differs from source provenance")
    md5 = record.get("md5", "")
    if not isinstance(md5, str) or not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError("invalid source checksum")
    if record.get("url") != REMOTE + md5[:2] + "/" + md5[2:]:
        raise ValueError("recording URL is outside the reviewed university remote")
    if type(record.get("expected_bytes")) is not int or record["expected_bytes"] <= 0:
        raise ValueError("invalid expected size")
    return match


def _authenticate_inventory(manifest, data_root=None):
    """Return the full pinned inventory and its tuple digest before payload I/O."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported source inventory schema")
    if (manifest.get("source") != "whoisalyx" or
            manifest.get("license") != "CC-BY-NC-SA-4.0"):
        raise PermissionError("unreviewed data source or license")
    if manifest.get("revision") != SOURCE_REVISION:
        raise PermissionError("source revision has not been reviewed")
    records = manifest.get("records")
    digest = _verify_record_inventory(records)
    declared_record_digest = manifest.get("record_inventory_sha256")
    if declared_record_digest is not None and declared_record_digest != digest:
        raise ValueError("inventory declares a different consumed-field digest")
    if "recording_files" in manifest and manifest["recording_files"] != len(records):
        raise ValueError("inventory file count does not match its records")
    if "expected_total_bytes" in manifest and manifest["expected_total_bytes"] != sum(r["expected_bytes"] for r in records):
        raise ValueError("inventory byte total does not match its records")
    if "potential_users" in manifest and manifest["potential_users"] != len({r["user_id"] for r in records}):
        raise ValueError("inventory user count does not match its records")
    pointer_hashes = [r.get("pointer_sha256") for r in records]
    declared_pointer_digest = manifest.get("pointer_inventory_sha256")
    if any(value is not None for value in pointer_hashes):
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in pointer_hashes):
            raise ValueError("inventory has incomplete raw pointer SHA-256 metadata")
        _verify_pointer_inventory(records)
    elif declared_pointer_digest is not None:
        raise ValueError("inventory declares a pointer snapshot without per-record pointer hashes")
    if declared_pointer_digest is not None and declared_pointer_digest != POINTER_INVENTORY_SHA256:
        raise ValueError("inventory declares an unreviewed pointer snapshot")
    if data_root is not None:
        for record in records:
            _validated_record(record, data_root)
    return records, digest


def _session_inventory(records):
    sessions = {}
    for record in records:
        sessions.setdefault((record["user_id"], record["session_id"]), []).append(record["source_path"])
    return {key: sorted(paths) for key, paths in sessions.items()}


def _bind_receipt_records(records, inventory):
    """Bind a receipt subset to pinned records and complete source sessions."""
    if not isinstance(records, list):
        raise ValueError("download receipt records must be a list")
    trusted = {record["source_path"]: record for record in inventory}
    sessions = _session_inventory(inventory)
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("download receipt entries must be objects")
        _validate_record_identity(record)
        source_path = record["source_path"]
        if source_path in seen:
            raise ValueError("download receipt contains duplicate source paths")
        seen.add(source_path)
        expected = trusted.get(source_path)
        if expected is None or any(record.get(key) != expected.get(key)
                                   for key in ("source_path", "source_revision", "user_id", "session_id",
                                               "relative_path", "url", "expected_bytes", "md5")):
            raise ValueError("receipted record differs from the authenticated inventory")
        expected_session_paths = sessions[(expected["user_id"], expected["session_id"])]
        if record.get("expected_session_paths") != expected_session_paths:
            raise ValueError("receipted session parts differ from the complete authenticated inventory")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("receipt record is missing its verified payload SHA-256")
        if type(record.get("downloaded_bytes_this_run")) is not int or record["downloaded_bytes_this_run"] < 0:
            raise ValueError("receipt record has an invalid transfer byte count")
        if not isinstance(record.get("cached"), bool):
            raise ValueError("receipt record is missing its cache status")
        if not isinstance(record.get("verified_utc"), str):
            raise ValueError("receipt record is missing its verification time")
    return records


def _validate_inventory_binding(binding, record_digest):
    if not isinstance(binding, dict):
        raise ValueError("invalid inventory binding")
    manifest_digest = binding.get("manifest_sha256")
    bound_records = binding.get("record_inventory_sha256")
    if (not isinstance(manifest_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest)
            or bound_records != record_digest):
        raise ValueError("receipt inventory binding does not match the authenticated inventory")


def _validate_download_receipt(receipt, *, inventory, record_digest, manifest_digest,
                               require_manifest_binding=True):
    if not isinstance(receipt, dict) or receipt.get("source") != "whoisalyx":
        raise PermissionError("recordings require a reviewed-source download receipt")
    if (receipt.get("license") != "CC-BY-NC-SA-4.0" or
            receipt.get("noncommercial_research_confirmed") is not True):
        raise PermissionError("recording receipt lacks the reviewed license/use confirmation")
    version = receipt.get("schema_version", 1)
    if type(version) is not int or version not in (1, DOWNLOAD_RECEIPT_SCHEMA_VERSION):
        raise ValueError("unsupported download receipt schema")
    if version == 1:
        if receipt.get("source_manifest_sha256") != manifest_digest:
            raise ValueError("legacy receipt is not bound to the supplied inventory bytes")
    else:
        if receipt.get("record_inventory_sha256") != record_digest:
            raise ValueError("receipt is not bound to the reviewed record inventory")
        bindings = receipt.get("inventory_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError("versioned receipt has no inventory bindings")
        seen = set()
        for binding in bindings:
            _validate_inventory_binding(binding, record_digest)
            if binding["manifest_sha256"] in seen:
                raise ValueError("versioned receipt contains duplicate inventory bindings")
            seen.add(binding["manifest_sha256"])
        if require_manifest_binding and manifest_digest not in seen:
            raise ValueError("versioned receipt is not bound to the supplied inventory bytes")
    return _bind_receipt_records(receipt.get("records"), inventory)


def _reject_recording_aliases(paths, data_root):
    """Reject metadata/output paths that identify a local recording inode.

    `Path.resolve()` catches ordinary symlinks and paths under the raw tree but
    cannot identify a hard link stored elsewhere. This scan uses directory and
    file identity metadata only; it never opens recording contents.
    """
    raw_root = (Path(data_root).resolve() / "raw" / "whoisalyx").resolve()
    candidates = []
    for label, value in paths:
        path = Path(value)
        if path.resolve().is_relative_to(raw_root):
            raise ValueError(f"{label} path must not target a recording object")
        candidates.append((label, path))

    try:
        raw_root.stat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError("cannot verify metadata paths against the raw recording tree") from exc
    if not raw_root.is_dir():
        raise ValueError("raw recording root is not a directory")

    def walk_error(error):
        raise error

    try:
        for directory, _subdirs, filenames in os.walk(raw_root, onerror=walk_error):
            for filename in filenames:
                recording = Path(directory) / filename
                for label, candidate in candidates:
                    try:
                        if candidate.samefile(recording):
                            raise ValueError(f"{label} path is a hard-link alias of a raw recording")
                    except FileNotFoundError:
                        # A non-existing output path or a concurrently removed
                        # source file cannot be the same current file object.
                        continue
                    except OSError as exc:
                        raise ValueError("cannot verify metadata file identity against recordings") from exc
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("cannot scan recording identities without reading payloads") from exc


def reseal_legacy_receipt(inventory_path, receipt_path, output_path, data_root):
    """Metadata-only validate/reseal a legacy receipt without opening recordings."""
    inventory_path, receipt_path, output_path = map(Path, (inventory_path, receipt_path, output_path))
    _reject_recording_aliases((
        ("inventory", inventory_path), ("receipt", receipt_path), ("new receipt", output_path),
    ), data_root)
    inventory_bytes = inventory_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    receipt = json.loads(receipt_bytes)
    records, record_digest = _authenticate_inventory(inventory, data_root)
    inventory_digest = hashlib.sha256(inventory_bytes).hexdigest()
    if receipt.get("schema_version", 1) != 1:
        raise ValueError("metadata-only reseal accepts only the original unversioned receipt")
    _validate_download_receipt(receipt, inventory=records, record_digest=record_digest,
                               manifest_digest=inventory_digest)
    if output_path.resolve() in {inventory_path.resolve(), receipt_path.resolve()}:
        raise ValueError("new reseal output must not overwrite either historical input")
    if output_path.exists():
        raise FileExistsError("new reseal output already exists; preserve prior validation receipts")
    resealed = {"schema_version": DOWNLOAD_RECEIPT_SCHEMA_VERSION,
                "source": receipt["source"], "license": receipt["license"],
                "noncommercial_research_confirmed": True,
                "record_inventory_sha256": record_digest,
                "inventory_bindings": [{"manifest_sha256": inventory_digest,
                                         "record_inventory_sha256": record_digest}],
                "resealed_from": {"schema_version": 1,
                                  "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                                  "source_manifest_sha256": inventory_digest},
                "created_utc": utc_now(), "records": receipt["records"]}
    _write_json_new(output_path, resealed)
    return {"inventory_file_sha256": inventory_digest,
            "legacy_receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "record_inventory_sha256": record_digest,
            "inventory_records": len(records), "receipt_records": len(receipt["records"]),
            "legacy_receipt_validated": True, "new_receipt_schema_version": DOWNLOAD_RECEIPT_SCHEMA_VERSION,
            "recording_payloads_read": 0, "new_receipt_file_sha256": sha256_file(output_path)}


def discover_alyx(output, revision=SOURCE_REVISION):
    """Read only public repository metadata, not recording bodies."""
    if revision != SOURCE_REVISION:
        raise ValueError("unreviewed revision; review source terms and update the pinned revision in code first")
    output = Path(output)
    evidence = output.parent / "source_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    tree = json.loads(_small_get(f"https://api.github.com/repos/{SOURCE_REPO}/git/trees/{revision}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("source tree is truncated; cannot declare complete inventory")
    raw_base = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{revision}/"
    for name in ("Readme.md", "License.md", ".dvc/config"):
        (evidence / name.replace("/", "_")).write_bytes(_small_get(raw_base + name))
    remote_config = (evidence / ".dvc_config").read_text(encoding="utf-8")
    if REMOTE not in remote_config:
        raise ValueError("upstream DVC remote changed; review before downloading")
    candidates = []
    for entry in tree["tree"]:
        match = MOTION_PATTERN.fullmatch(entry["path"])
        if match and int(match[1]) >= 42:
            candidates.append((entry, match.groups()))

    def read_pointer(candidate):
        entry, (user, session, filename) = candidate
        pointer_bytes = _small_get(raw_base + entry["path"], limit=16384)
        pointer_sha256 = hashlib.sha256(pointer_bytes).hexdigest()
        pointer = pointer_bytes.decode("utf-8")
        checksum = re.search(r"(?:^|\s)md5:\s*([0-9a-f]{32})(?:\s|$)", pointer)
        size = re.search(r"(?:^|\s)size:\s*(\d+)(?:\s|$)", pointer)
        leaf = re.search(r"(?:^|\s)path:\s*(\S+)", pointer)
        if not checksum or not size or not leaf or leaf[1] != filename:
            raise ValueError(f"unexpected DVC pointer schema: {entry['path']}")
        md5 = checksum[1]
        relative = str(PurePosixPath("raw", "whoisalyx", user, session, filename))
        return {"source": "whoisalyx", "user_id": f"alyx_{int(user):02d}",
                "session_id": session, "source_path": entry["path"],
                "pointer_blob_sha": entry["sha"], "source_revision": revision,
                "pointer_sha256": pointer_sha256,
                "expected_bytes": int(size[1]), "md5": md5,
                "url": REMOTE + md5[:2] + "/" + md5[2:], "relative_path": relative}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(read_pointer, candidates))
    records.sort(key=lambda r: r["source_path"])
    _verify_pointer_inventory(records)
    _verify_record_inventory(records)
    manifest = {"schema_version": 1, "created_utc": utc_now(), "source": "whoisalyx",
                "repository": SOURCE_REPO, "revision": revision,
                "pointer_inventory_sha256": POINTER_INVENTORY_SHA256,
                "record_inventory_sha256": RECORD_INVENTORY_SHA256,
                "license": "CC-BY-NC-SA-4.0", "license_url": LICENSE_URL,
                "recording_files": len(records), "potential_users": len({r["user_id"] for r in records}),
                "expected_total_bytes": sum(r["expected_bytes"] for r in records),
                "payload_downloaded": False, "records": records}
    write_json(output, manifest)
    return manifest


def _validated_record(record, root):
    _validate_record_identity(record)
    relative = PurePosixPath(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("raw", "whoisalyx"):
        raise ValueError("unsafe recording path")
    root = Path(root).resolve()
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("recording path escapes data root")
    return destination


def _verify_record(path, record):
    if path.stat().st_size != record["expected_bytes"]:
        raise ValueError("recording length differs from upstream pointer")
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != record["md5"]:
        raise ValueError("recording checksum differs from upstream pointer")
    return sha.hexdigest()


def fetch_alyx(manifest_path, data_root, *, noncommercial=False, max_files=2,
               max_total_bytes=256 * 1024 * 1024, users=None, receipt_path=None,
               previous_receipt_path=None):
    if not noncommercial:
        raise PermissionError("Confirm noncommercial research with --noncommercial after reviewing source terms")
    if max_files < 1 or max_total_bytes < 1:
        raise ValueError("download limits must be positive")
    manifest_path, data_root = Path(manifest_path), Path(data_root)
    receipt_path = Path(receipt_path) if receipt_path is not None else data_root / "download_receipt_v2.json"
    if receipt_path.exists():
        raise FileExistsError("new fetch receipt already exists; choose a new path to preserve receipt history")
    if receipt_path.resolve() == manifest_path.resolve():
        raise ValueError("new fetch receipt path must be distinct metadata, not a source inventory or recording")
    if previous_receipt_path is None:
        legacy_path = data_root / "download_receipt.json"
        previous_receipt_path = legacy_path if legacy_path.exists() else None
    else:
        previous_receipt_path = Path(previous_receipt_path)
    if previous_receipt_path is not None and previous_receipt_path.resolve() in {manifest_path.resolve(), receipt_path.resolve()}:
        raise ValueError("previous receipt path must be distinct metadata, not an inventory or recording")
    metadata_paths = [("source inventory", manifest_path), ("new receipt", receipt_path)]
    if previous_receipt_path is not None:
        metadata_paths.append(("previous receipt", previous_receipt_path))
    _reject_recording_aliases(metadata_paths, data_root)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    records, record_digest = _authenticate_inventory(manifest, data_root)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    session_parts = _session_inventory(records)
    selected = [{**r, "expected_session_paths": sorted(session_parts[(r["user_id"], r["session_id"])])}
                for r in records if not users or r["user_id"] in users][:max_files]
    if not selected:
        raise ValueError("no matching recordings")
    for record in selected:
        _validated_record(record, data_root)
    if sum(r["expected_bytes"] for r in selected) > max_total_bytes:
        raise ValueError("selected recordings exceed byte budget; narrow users/files or explicitly change limit")
    if previous_receipt_path is not None:
        previous_bytes = previous_receipt_path.read_bytes()
        previous = json.loads(previous_bytes)
        previous_records = _validate_download_receipt(previous, inventory=records,
            record_digest=record_digest, manifest_digest=manifest_digest,
            require_manifest_binding=False)
        previous_version = previous.get("schema_version", 1)
        if previous_version == 1:
            bindings = []
            legacy_receipt_sha256 = hashlib.sha256(previous_bytes).hexdigest()
        else:
            bindings = list(previous["inventory_bindings"])
            legacy_receipt_sha256 = (previous.get("resealed_from", {}).get("source_receipt_sha256") or
                                     previous.get("legacy_receipt_sha256"))
        previous_receipt_sha256 = hashlib.sha256(previous_bytes).hexdigest()
    else:
        previous = None
        previous_records = []
        bindings = []
        legacy_receipt_sha256 = None
        previous_receipt_sha256 = None
    receipts = {r["source_path"]: r for r in previous_records}
    budget_records = {**receipts, **{r["source_path"]: r for r in selected}}
    if sum(r["expected_bytes"] for r in budget_records.values()) > max_total_bytes:
        raise ValueError("existing plus selected recordings exceed the total byte budget")
    for record in selected:
        destination = _validated_record(record, data_root)
        started = time.perf_counter()
        destination.parent.mkdir(parents=True, exist_ok=True)
        transferred = 0
        cached = destination.exists()
        if not cached:
            partial = destination.with_name(destination.name + ".part")
            try:
                request = Request(record["url"], headers={"User-Agent": "ABC-IUEH-research/0.1"})
                with urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                    if urlparse(response.url).netloc != urlparse(REMOTE).netloc:
                        raise ValueError("recording redirected outside reviewed host")
                    length = response.headers.get("Content-Length")
                    if length and int(length) != record["expected_bytes"]:
                        raise ValueError("upstream content length changed")
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        transferred += len(block)
                        if transferred > record["expected_bytes"]:
                            raise ValueError("download exceeds expected recording size")
                        stream.write(block)
                _verify_record(partial, record)
                partial.replace(destination)
            except BaseException:
                if partial.exists():
                    partial.unlink()
                raise
        digest = _verify_record(destination, record)
        receipts[record["source_path"]] = {**record, "sha256": digest, "verified_utc": utc_now(),
                                             "downloaded_bytes_this_run": transferred, "cached": cached,
                                             "elapsed_seconds": time.perf_counter() - started}
        inventory_bindings = {binding["manifest_sha256"]: binding for binding in bindings}
        inventory_bindings[manifest_digest] = {"manifest_sha256": manifest_digest,
                                                "record_inventory_sha256": record_digest}
        receipt_document = {"schema_version": DOWNLOAD_RECEIPT_SCHEMA_VERSION,
                            "source": "whoisalyx", "license": manifest["license"],
                            "noncommercial_research_confirmed": True,
                            "record_inventory_sha256": record_digest,
                            "inventory_bindings": [inventory_bindings[key] for key in sorted(inventory_bindings)],
                            "created_utc": utc_now(),
                            "records": [receipts[key] for key in sorted(receipts)]}
        if previous_receipt_sha256:
            receipt_document["previous_receipt_sha256"] = previous_receipt_sha256
        if legacy_receipt_sha256:
            receipt_document["legacy_receipt_sha256"] = legacy_receipt_sha256
        _write_json_new(receipt_path, receipt_document)
        print(json.dumps({"event": "recording_verified", "recording": record["source_path"],
                          "bytes": record["expected_bytes"], "cached": cached}), flush=True)
    return list(receipts.values())

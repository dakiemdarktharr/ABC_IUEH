"""Pinned, bounded WhoIsAlyx downloads; BOXRR is deliberately not enabled."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE_REPO = "cschell/who-is-alyx"
SOURCE_REVISION = "5da510121f99fc8a6cc51e570770ca93e5745646"
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
        blob_digest = hashlib.sha1(b"blob " + str(len(pointer_bytes)).encode() + b"\0" + pointer_bytes).hexdigest()
        if blob_digest != entry["sha"]:
            raise ValueError("DVC pointer does not match pinned repository blob")
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
                "expected_bytes": int(size[1]), "md5": md5,
                "url": REMOTE + md5[:2] + "/" + md5[2:], "relative_path": relative}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(read_pointer, candidates))
    records.sort(key=lambda r: r["source_path"])
    manifest = {"schema_version": 1, "created_utc": utc_now(), "source": "whoisalyx",
                "repository": SOURCE_REPO, "revision": revision,
                "license": "CC-BY-NC-SA-4.0", "license_url": LICENSE_URL,
                "recording_files": len(records), "potential_users": len({r["user_id"] for r in records}),
                "expected_total_bytes": sum(r["expected_bytes"] for r in records),
                "payload_downloaded": False, "records": records}
    write_json(output, manifest)
    return manifest


def _validated_record(record, root):
    if record.get("source") != "whoisalyx":
        raise PermissionError("Only the reviewed WhoIsAlyx source is enabled; BOXRR remains blocked")
    match = MOTION_PATTERN.fullmatch(record.get("source_path", ""))
    if not match or not 42 <= int(match[1]) <= 76 or record.get("source_revision") != SOURCE_REVISION:
        raise ValueError("record is outside the reviewed motion cohort/revision")
    user, session, filename = match.groups()
    if record.get("user_id") != f"alyx_{int(user):02d}" or record.get("session_id") != session:
        raise ValueError("recording/user/session provenance mismatch")
    expected_relative = str(PurePosixPath("raw", "whoisalyx", user, session, filename))
    if record.get("relative_path") != expected_relative:
        raise ValueError("recording destination differs from source provenance")
    md5 = record.get("md5", "")
    if not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError("invalid source checksum")
    expected_url = REMOTE + md5[:2] + "/" + md5[2:]
    if record.get("url") != expected_url:
        raise ValueError("recording URL is outside the reviewed university remote")
    relative = PurePosixPath(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("raw", "whoisalyx"):
        raise ValueError("unsafe recording path")
    root = Path(root).resolve()
    destination = root.joinpath(*relative.parts).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("recording path escapes data root")
    if type(record["expected_bytes"]) is not int or record["expected_bytes"] <= 0:
        raise ValueError("invalid expected size")
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
               max_total_bytes=256 * 1024 * 1024, users=None):
    if not noncommercial:
        raise PermissionError("Confirm noncommercial research with --noncommercial after reviewing source terms")
    if max_files < 1 or max_total_bytes < 1:
        raise ValueError("download limits must be positive")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("source") != "whoisalyx" or manifest.get("license") != "CC-BY-NC-SA-4.0":
        raise PermissionError("unreviewed data source or license")
    if manifest.get("revision") != SOURCE_REVISION:
        raise PermissionError("source revision has not been reviewed")
    session_parts = {}
    for record in manifest["records"]:
        key = (record["user_id"], record["session_id"])
        session_parts.setdefault(key, []).append(record["source_path"])
    selected = [{**r, "expected_session_paths": sorted(session_parts[(r["user_id"], r["session_id"])])}
                for r in manifest["records"] if not users or r["user_id"] in users][:max_files]
    if not selected:
        raise ValueError("no matching recordings")
    for record in selected:
        _validated_record(record, data_root)
    if sum(r["expected_bytes"] for r in selected) > max_total_bytes:
        raise ValueError("selected recordings exceed byte budget; narrow users/files or explicitly change limit")
    receipt_path = Path(data_root) / "download_receipt.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {"records": []}
    for record in previous["records"]:
        _validated_record(record, data_root)
    receipts = {r["source_path"]: r for r in previous["records"]}
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
        write_json(receipt_path, {"source": "whoisalyx", "license": manifest["license"],
                                  "noncommercial_research_confirmed": True,
                                  "source_manifest_sha256": sha256_file(manifest_path),
                                  "records": list(receipts.values())})
        print(json.dumps({"event": "recording_verified", "recording": record["source_path"],
                          "bytes": record["expected_bytes"], "cached": cached}), flush=True)
    return list(receipts.values())

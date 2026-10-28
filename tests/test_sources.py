import json
import hashlib
import pytest

from abc_iueh.sources import (REMOTE, SOURCE_REVISION, _validated_record,
                              fetch_alyx, write_json)


def record(user=42, size=100):
    md5 = "a" * 32
    return {"source": "whoisalyx", "user_id": f"alyx_{user:02d}",
            "session_id": "2022-05-19", "source_revision": SOURCE_REVISION,
            "source_path": f"players/{user}/2022-05-19/vr-controllers.csv.dvc",
            "relative_path": f"raw/whoisalyx/{user}/2022-05-19/vr-controllers.csv",
            "expected_bytes": size, "md5": md5, "url": REMOTE + "aa/" + "a" * 30}


def manifest(tmp_path, records, source="whoisalyx"):
    path = tmp_path / "inventory.json"
    write_json(path, {"schema_version": 1, "source": source, "license": "CC-BY-NC-SA-4.0",
                      "revision": SOURCE_REVISION, "records": records})
    return path


def no_network(*args, **kwargs):
    pytest.fail("invalid request must be rejected before any network call")


def test_no_download_without_noncommercial_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr("abc_iueh.sources.urlopen", no_network)
    with pytest.raises(PermissionError):
        fetch_alyx(manifest(tmp_path, [record()]), tmp_path)


def test_boxrr_stays_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("abc_iueh.sources.urlopen", no_network)
    with pytest.raises(PermissionError):
        fetch_alyx(manifest(tmp_path, [record()], source="boxrr"), tmp_path, noncommercial=True)


def test_negative_size_cannot_cancel_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("abc_iueh.sources.urlopen", no_network)
    with pytest.raises(ValueError, match="size"):
        fetch_alyx(manifest(tmp_path, [record(size=1000), record(user=43, size=-999)]),
                   tmp_path, noncommercial=True, max_total_bytes=500)


@pytest.mark.parametrize("patch", [{"user_id": "alyx_43"}, {"session_id": "another"},
                         {"relative_path": "raw/whoisalyx/../../escape"},
                         {"source_revision": "b" * 40}, {"url": "https://example.com/data"},
                         {"source_path": "players/42/2022-05-19/eye-tracking.csv.dvc"}])
def test_source_identity_and_paths_are_bound(tmp_path, patch):
    with pytest.raises(ValueError):
        _validated_record({**record(), **patch}, tmp_path)


def _mock_discovery(monkeypatch, mutation=None):
    from abc_iueh import sources

    entries, pointers = [], {}
    for user, letter, size in [(42, "a", 100), (43, "b", 200)]:
        path = f"players/{user}/2022-05-19/vr-controllers.csv.dvc"
        body = (f"outs:\n- md5: {letter * 32}\n  size: {size}\n  path: vr-controllers.csv\n").encode()
        entries.append({"path": path, "sha": str(user) * 20})
        pointers[path] = body
    if mutation == "content":
        pointers[entries[0]["path"]] = pointers[entries[0]["path"]].replace(b"a" * 32, b"c" * 32)
    elif mutation == "comment":
        pointers[entries[0]["path"]] += b"# changed without changing parsed fields\n"
    elif mutation == "path":
        old = entries[0]["path"]
        entries[0]["path"] = old.replace("2022-05-19", "2022-05-20")
        pointers[entries[0]["path"]] = pointers.pop(old)
    elif mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries.append(entries[0].copy())
    raw_base = f"https://raw.githubusercontent.com/{sources.SOURCE_REPO}/{sources.SOURCE_REVISION}/"

    def small_get(url, limit=8 * 1024 * 1024):
        if "/git/trees/" in url:
            return json.dumps({"truncated": False, "tree": entries}).encode()
        path = url.removeprefix(raw_base)
        if path == ".dvc/config":
            return f"[remote]\nurl = {sources.REMOTE}\n".encode()
        if path in {"Readme.md", "License.md"}:
            return b"synthetic source terms\n"
        if path in pointers:
            return pointers[path]
        pytest.fail("unexpected metadata request")

    monkeypatch.setattr(sources, "_small_get", small_get)
    # Frozen digest of these two original fixtures, not their mutated versions.
    monkeypatch.setattr(sources, "POINTER_INVENTORY_SHA256",
                        "7106c6596b4c4e81b19f5ef11f93e23d1b081e91c84c183802b9c882e07eb533")
    parsed_records = []
    for user, letter, size in [(42, "a", 100), (43, "b", 200)]:
        md5 = letter * 32
        session = "2022-05-19"
        filename = "vr-controllers.csv"
        parsed_records.append({"source": "whoisalyx", "user_id": f"alyx_{user:02d}",
            "session_id": session, "source_path": f"players/{user}/{session}/{filename}.dvc",
            "source_revision": SOURCE_REVISION, "relative_path": f"raw/whoisalyx/{user}/{session}/{filename}",
            "expected_bytes": size, "md5": md5,
            "url": sources.REMOTE + md5[:2] + "/" + md5[2:]})
    monkeypatch.setattr(sources, "RECORD_INVENTORY_SHA256",
                        sources._record_inventory_digest(parsed_records))


def test_discovery_accepts_reviewed_sha256_inventory(tmp_path, monkeypatch):
    from abc_iueh.sources import discover_alyx
    _mock_discovery(monkeypatch)
    result = discover_alyx(tmp_path / "inventory.json")
    assert result["recording_files"] == 2
    body = b"outs:\n- md5: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n  size: 100\n  path: vr-controllers.csv\n"
    assert result["records"][0]["pointer_sha256"] == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize("mutation", ["content", "comment", "path", "missing", "duplicate"])
def test_discovery_rejects_tampered_pointer_snapshot(tmp_path, monkeypatch, mutation):
    from abc_iueh.sources import discover_alyx
    _mock_discovery(monkeypatch, mutation)
    with pytest.raises(ValueError, match="SHA-256 snapshot"):
        discover_alyx(tmp_path / "inventory.json")
    assert not (tmp_path / "inventory.json").exists()

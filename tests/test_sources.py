import json
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
    write_json(path, {"source": source, "license": "CC-BY-NC-SA-4.0",
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

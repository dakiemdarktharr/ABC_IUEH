import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from abc_iueh.config import PipelineConfig
from abc_iueh.dataset import (check_hdf5, create_user_split, ingest_alyx,
                              iter_windows, read_alyx_motion)
from abc_iueh.geometry import exp_so3
from abc_iueh import sources
from abc_iueh.sources import SOURCE_REVISION, REMOTE, sha256_file, write_json


def synthetic_record(root, *, filename="vr-controllers.csv", offset=0.0):
    path = root / "raw/whoisalyx/42/2022-05-19" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(2251) / 90 + offset
    angle = np.deg2rad(45) * np.sin(2 * np.pi * 0.4 * t)
    q = exp_so3(np.column_stack((angle, np.zeros_like(t), np.zeros_like(t))))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        # Deliberately not wxyz physical column order.
        writer.writerow(["hmd_rot_z", "unused_controller_button", "hmd_rot_x",
                         "delta_time_ms", "hmd_rot_w", "hmd_rot_y"])
        for timestamp, orientation in zip(t, q):
            writer.writerow([orientation[3], 999, orientation[1], timestamp * 1000,
                             orientation[0], orientation[2]])
    md5 = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    return {"source": "whoisalyx", "user_id": "alyx_42", "session_id": "2022-05-19",
            "source_path": f"players/42/2022-05-19/{filename}.dvc",
            "source_revision": SOURCE_REVISION, "relative_path": path.relative_to(root).as_posix(),
            "expected_bytes": path.stat().st_size, "md5": md5,
            "url": REMOTE + md5[:2] + "/" + md5[2:], "sha256": sha256_file(path)}


def receipt(root, records, monkeypatch, extra_inventory_records=()):
    inventory_records = [dict(record) for record in records] + [dict(record) for record in extra_inventory_records]
    record_digest = sources._record_inventory_digest(inventory_records)
    monkeypatch.setattr(sources, "RECORD_INVENTORY_SHA256", record_digest)
    inventory_path = root / "alyx_inventory.json"
    write_json(inventory_path, {"schema_version": 1, "source": "whoisalyx",
        "license": "CC-BY-NC-SA-4.0", "revision": SOURCE_REVISION,
        "records": inventory_records})
    inventory_digest = sha256_file(inventory_path)
    grouped = {}
    for record in inventory_records:
        grouped.setdefault((record["user_id"], record["session_id"]), []).append(record["source_path"])
    records = [{**r, "expected_session_paths": sorted(grouped[(r["user_id"], r["session_id"])]),
                "downloaded_bytes_this_run": 0, "cached": True,
                "verified_utc": "2026-10-28T00:00:00+00:00"}
               for r in records]
    write_json(root / "download_receipt.json", {"schema_version": 2, "source": "whoisalyx",
               "license": "CC-BY-NC-SA-4.0", "noncommercial_research_confirmed": True,
               "record_inventory_sha256": record_digest,
               "inventory_bindings": [{"manifest_sha256": inventory_digest,
                                       "record_inventory_sha256": record_digest}],
               "records": records})


def test_named_columns_milliseconds_and_no_extra_fields(tmp_path):
    record = synthetic_record(tmp_path)
    t, q = read_alyx_motion([tmp_path / record["relative_path"]])
    assert t[-1] == pytest.approx(25.0)
    assert q.shape == (2251, 4)
    np.testing.assert_allclose(q[0], [1, 0, 0, 0], atol=1e-12)


def test_real_schema_end_to_end_and_no_smoke_training(tmp_path, monkeypatch):
    record = synthetic_record(tmp_path)
    receipt(tmp_path, [record], monkeypatch)
    hdf5, report = tmp_path / "processed/data.h5", tmp_path / "quality.json"
    quality = ingest_alyx(tmp_path, hdf5, report, PipelineConfig(),
                          inventory_path=tmp_path / "alyx_inventory.json")
    assert quality["eligible_user_count"] == 1
    assert not quality["confirmatory_data_gate_passed"]
    summary = check_hdf5(hdf5, PipelineConfig())
    assert summary["samples"] >= 1900
    assert summary["windows"] == summary["samples"] - 419
    with pytest.raises(ValueError, match="30"):
        create_user_split(hdf5, tmp_path / "confirmatory.json", PipelineConfig())
    split_path = tmp_path / "smoke.json"
    create_user_split(hdf5, split_path, PipelineConfig(), smoke=True)
    window = next(iter_windows(hdf5, split_path, "smoke"))
    assert window["context"].shape == (400, 4)
    assert window["target"].shape == (20, 4)
    with pytest.raises(ValueError):
        window["context"][0, 0] = 0
    with pytest.raises(PermissionError):
        next(iter_windows(hdf5, split_path, "train"))
    with pytest.raises(PermissionError):
        next(iter_windows(hdf5, split_path, "final"))
    with pytest.raises(FileExistsError):
        create_user_split(hdf5, split_path, PipelineConfig(), smoke=True)


def test_multiple_files_share_session_and_clock_reset_is_not_repaired(tmp_path, monkeypatch):
    records = [synthetic_record(tmp_path), synthetic_record(tmp_path, filename="vr-controllers2.csv")]
    receipt(tmp_path, records, monkeypatch)
    result = ingest_alyx(tmp_path, tmp_path / "out.h5", tmp_path / "quality.json", PipelineConfig(),
                         inventory_path=tmp_path / "alyx_inventory.json")
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["source_files"] == 2
    assert result["sessions"][0]["reason"] == "backward_timestamps"
    assert result["eligible_user_count"] == 0


def test_incomplete_source_session_cannot_qualify_on_its_active_prefix(tmp_path, monkeypatch):
    record = synthetic_record(tmp_path)
    missing = dict(record)
    missing["source_path"] = "players/42/2022-05-19/vr-controllers2.csv.dvc"
    missing["relative_path"] = "raw/whoisalyx/42/2022-05-19/vr-controllers2.csv"
    missing["md5"] = "b" * 32
    missing["expected_bytes"] = 1
    missing["url"] = REMOTE + "bb/" + "b" * 30
    missing.pop("sha256")
    receipt(tmp_path, [record], monkeypatch, extra_inventory_records=[missing])
    def no_recording_read(*args, **kwargs):
        pytest.fail("incomplete source session must be rejected before payload I/O")
    monkeypatch.setattr("abc_iueh.dataset._verify_record", no_recording_read)
    result = ingest_alyx(tmp_path, tmp_path / "out.h5", tmp_path / "quality.json", PipelineConfig(),
                         inventory_path=tmp_path / "alyx_inventory.json")
    assert result["sessions"][0]["reason"] == "source_session_incomplete_or_unverified"
    assert result["eligible_user_count"] == 0


def make_many_users(path, count=35):
    with h5py.File(path, "w") as store:
        store.attrs["quaternion_order"] = "wxyz"
        for i in range(count):
            segment = store.create_group(f"users/synthetic_{i:02d}/sessions/session/segments/0000")
            q = np.tile([1, 0, 0, 0], (450, 1)).astype(np.float32)
            segment.create_dataset("quaternion", data=q)
            segment.create_dataset("timestamps", data=np.arange(450) / 100)


def test_split_is_user_disjoint_reproducible_and_data_bound(tmp_path):
    path = tmp_path / "data.h5"
    make_many_users(path)
    a = create_user_split(path, tmp_path / "a.json", PipelineConfig())
    b = create_user_split(path, tmp_path / "b.json", PipelineConfig())
    assert a["partitions"] == b["partitions"]
    assert {k: len(v) for k, v in a["partitions"].items()} == {"train": 15, "development": 5, "final": 15}
    assert set(a["partitions"]["train"]).isdisjoint(a["partitions"]["final"])
    assert len(list(iter_windows(path, tmp_path / "a.json", "train", stride=20))) == 30
    with h5py.File(path, "a") as store:
        store.attrs["changed"] = "yes"
    with pytest.raises(ValueError, match="changed"):
        next(iter_windows(path, tmp_path / "a.json", "train"))


def test_ingest_does_not_overwrite_raw_or_receipt(tmp_path, monkeypatch):
    record = synthetic_record(tmp_path)
    receipt(tmp_path, [record], monkeypatch)
    raw = tmp_path / record["relative_path"]
    with pytest.raises(ValueError, match="overwrite"):
        ingest_alyx(tmp_path, raw, tmp_path / "quality.json", PipelineConfig(),
                    inventory_path=tmp_path / "alyx_inventory.json")
    assert sha256_file(raw) == record["sha256"]

import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import pytest

from abc_iueh import sources
from abc_iueh.config import PipelineConfig
from abc_iueh.dataset import ingest_alyx
from abc_iueh.sources import REMOTE, SOURCE_REVISION, fetch_alyx, reseal_legacy_receipt, sha256_file, write_json


def _record(payload, *, day="2022-05-19", filename="vr-controllers.csv"):
    user = 42
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return {"source": "whoisalyx", "user_id": "alyx_42", "session_id": day,
            "source_path": f"players/{user}/{day}/{filename}.dvc",
            "source_revision": SOURCE_REVISION,
            "relative_path": f"raw/whoisalyx/{user}/{day}/{filename}",
            "expected_bytes": len(payload), "md5": md5,
            "url": REMOTE + md5[:2] + "/" + md5[2:], "sha256": hashlib.sha256(payload).hexdigest()}


def _inventory(root, monkeypatch, records):
    records = [copy.deepcopy(record) for record in records]
    record_digest = sources._record_inventory_digest(records)
    monkeypatch.setattr(sources, "RECORD_INVENTORY_SHA256", record_digest)
    path = root / "alyx_inventory.json"
    write_json(path, {"schema_version": 1, "source": "whoisalyx", "license": "CC-BY-NC-SA-4.0",
                      "revision": SOURCE_REVISION, "records": records})
    return path, records, record_digest


def _receipt(root, inventory_path, records, record_digest, *, schema=2):
    inventory_file_digest = sha256_file(inventory_path)
    session_paths = sources._session_inventory(records)
    receipted = []
    for record in records:
        receipted.append({**record,
            "expected_session_paths": session_paths[(record["user_id"], record["session_id"])],
            "downloaded_bytes_this_run": record["expected_bytes"], "cached": False,
            "verified_utc": "2026-10-28T00:00:00+00:00", "elapsed_seconds": 0.01})
    receipt = {"schema_version": schema, "source": "whoisalyx", "license": "CC-BY-NC-SA-4.0",
               "noncommercial_research_confirmed": True, "records": receipted}
    if schema == 1:
        receipt["source_manifest_sha256"] = inventory_file_digest
        receipt.pop("schema_version")
    else:
        receipt["record_inventory_sha256"] = record_digest
        receipt["inventory_bindings"] = [{"manifest_sha256": inventory_file_digest,
                                           "record_inventory_sha256": record_digest}]
    path = root / "download_receipt.json"
    write_json(path, receipt)
    return path, receipt


def _no_recording_io(monkeypatch):
    calls = {"verify": 0, "parse": 0, "network": 0}

    def fail_verify(*args, **kwargs):
        calls["verify"] += 1
        raise AssertionError("recording checksum reader was called before metadata rejection")

    def fail_parse(*args, **kwargs):
        calls["parse"] += 1
        raise AssertionError("CSV reader was called before metadata rejection")

    def fail_network(*args, **kwargs):
        calls["network"] += 1
        raise AssertionError("network was called before metadata rejection")

    monkeypatch.setattr(sources, "_verify_record", fail_verify)
    monkeypatch.setattr("abc_iueh.dataset._verify_record", fail_verify)
    monkeypatch.setattr("abc_iueh.dataset.read_alyx_motion", fail_parse)
    monkeypatch.setattr(sources, "urlopen", fail_network)
    return calls


@pytest.mark.parametrize("mutation", ["self_consistent_tuple", "shortened_inventory"])
def test_fetch_rejects_inventory_tampering_before_cache_or_network(tmp_path, monkeypatch, mutation):
    payloads = [b"cached synthetic part A", b"cached synthetic part B"]
    records = [_record(payloads[0]), _record(payloads[1], day="2022-05-20")]
    inventory_path, records, _ = _inventory(tmp_path, monkeypatch, records)
    document = json.loads(inventory_path.read_text(encoding="utf-8"))
    if mutation == "self_consistent_tuple":
        forged_payload = b"forged cached synthetic object"
        forged = document["records"][0]
        forged_md5 = hashlib.md5(forged_payload, usedforsecurity=False).hexdigest()
        forged.update(expected_bytes=len(forged_payload), md5=forged_md5,
                      url=REMOTE + forged_md5[:2] + "/" + forged_md5[2:])
        cached = tmp_path / forged["relative_path"]
        cached.parent.mkdir(parents=True)
        cached.write_bytes(forged_payload)
    else:
        document["records"].pop()
    write_json(inventory_path, document)
    calls = _no_recording_io(monkeypatch)
    with pytest.raises(ValueError, match="inventory|snapshot|count|byte"):
        fetch_alyx(inventory_path, tmp_path, noncommercial=True, max_files=2)
    assert calls == {"verify": 0, "parse": 0, "network": 0}


def test_fetch_rejects_forged_cached_receipt_record_before_payload_io(tmp_path, monkeypatch):
    payloads = [b"cached synthetic part A", b"cached synthetic part B"]
    records = [_record(payloads[0]), _record(payloads[1], day="2022-05-20")]
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, records)
    receipt_path, receipt = _receipt(tmp_path, inventory_path, records, record_digest)
    forged_payload = b"counterfeit cached bytes"
    forged_md5 = hashlib.md5(forged_payload, usedforsecurity=False).hexdigest()
    receipt["records"][0].update(expected_bytes=len(forged_payload), md5=forged_md5,
        url=REMOTE + forged_md5[:2] + "/" + forged_md5[2:],
        sha256=hashlib.sha256(forged_payload).hexdigest())
    write_json(receipt_path, receipt)
    cache_path = tmp_path / records[0]["relative_path"]
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(forged_payload)
    calls = _no_recording_io(monkeypatch)
    with pytest.raises(ValueError, match="authenticated inventory"):
        fetch_alyx(inventory_path, tmp_path, noncommercial=True, max_files=2)
    assert calls == {"verify": 0, "parse": 0, "network": 0}


def test_legacy_metadata_only_reseal_preserves_original_bytes(tmp_path, monkeypatch):
    record = _record(b"opaque synthetic payload")
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, [record])
    receipt_path, _ = _receipt(tmp_path, inventory_path, records, record_digest, schema=1)
    inventory_before, receipt_before = inventory_path.read_bytes(), receipt_path.read_bytes()
    calls = _no_recording_io(monkeypatch)
    output = tmp_path / "resealed_receipt.json"
    summary = reseal_legacy_receipt(inventory_path, receipt_path, output, tmp_path)
    resealed = json.loads(output.read_text(encoding="utf-8"))
    assert summary["legacy_receipt_validated"] and summary["recording_payloads_read"] == 0
    assert resealed["schema_version"] == 2
    assert resealed["resealed_from"]["source_receipt_sha256"] == hashlib.sha256(receipt_before).hexdigest()
    assert resealed["record_inventory_sha256"] == record_digest
    assert inventory_path.read_bytes() == inventory_before
    assert receipt_path.read_bytes() == receipt_before
    assert calls == {"verify": 0, "parse": 0, "network": 0}


@pytest.mark.parametrize("mutation", ["unknown_path", "forged_tuple", "wrong_revision",
                                       "wrong_session_parts", "duplicate_record", "wrong_binding",
                                       "consent_false", "consent_string_false", "consent_integer_one"])
def test_direct_ingest_rejects_forged_receipt_before_recording_io(tmp_path, monkeypatch, mutation):
    payload = b"synthetic opaque recording bytes"
    records = [_record(payload)]
    if mutation == "duplicate_record":
        records.append(_record(b"second opaque part", filename="vr-controllers2.csv"))
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, records)
    receipt_path, receipt = _receipt(tmp_path, inventory_path, records, record_digest)
    row = receipt["records"][0]
    if mutation == "unknown_path":
        row["source_path"] = "players/42/2022-05-19/not-motion.csv.dvc"
    elif mutation == "forged_tuple":
        forged_payload = b"forged size/md5-consistent tuple"
        forged_md5 = hashlib.md5(forged_payload, usedforsecurity=False).hexdigest()
        row.update(expected_bytes=len(forged_payload), md5=forged_md5,
                   url=REMOTE + forged_md5[:2] + "/" + forged_md5[2:],
                   sha256=hashlib.sha256(forged_payload).hexdigest())
    elif mutation == "wrong_revision":
        row["source_revision"] = "0" * 40
    elif mutation == "wrong_session_parts":
        row["expected_session_paths"] = ["players/42/2022-05-19/vr-controllers-fake.csv.dvc"]
    elif mutation == "duplicate_record":
        receipt["records"].append(copy.deepcopy(receipt["records"][0]))
    elif mutation == "wrong_binding":
        receipt["inventory_bindings"][0]["record_inventory_sha256"] = "0" * 64
    elif mutation == "consent_false":
        receipt["noncommercial_research_confirmed"] = False
    elif mutation == "consent_string_false":
        receipt["noncommercial_research_confirmed"] = "false"
    elif mutation == "consent_integer_one":
        receipt["noncommercial_research_confirmed"] = 1
    write_json(receipt_path, receipt)
    calls = _no_recording_io(monkeypatch)
    with pytest.raises((ValueError, PermissionError)):
        ingest_alyx(tmp_path, tmp_path / "out.h5", tmp_path / "quality.json", PipelineConfig(),
                    inventory_path=inventory_path, receipt_path=receipt_path)
    assert calls == {"verify": 0, "parse": 0, "network": 0}


@pytest.mark.parametrize("metadata_kind", ["inventory", "receipt"])
def test_ingest_rejects_hardlink_metadata_alias_before_reading_alias(tmp_path, monkeypatch, metadata_kind):
    payload = b"synthetic opaque recording bytes"
    record = _record(payload)
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, [record])
    receipt_path, _ = _receipt(tmp_path, inventory_path, records, record_digest)
    recording = tmp_path / record["relative_path"]
    recording.parent.mkdir(parents=True, exist_ok=True)
    recording.write_bytes(payload)
    alias = tmp_path / f"hardlink_{metadata_kind}.json"
    os.link(recording, alias)
    calls = _no_recording_io(monkeypatch)
    original_read_bytes = Path.read_bytes
    alias_read_calls = []

    def track_alias_read(path):
        if path == alias:
            alias_read_calls.append(str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_alias_read)
    if metadata_kind == "inventory":
        inventory_input, receipt_input = alias, receipt_path
    else:
        inventory_input, receipt_input = inventory_path, alias
    with pytest.raises(ValueError, match="hard-link alias"):
        ingest_alyx(tmp_path, tmp_path / "out.h5", tmp_path / "quality.json", PipelineConfig(),
                    inventory_path=inventory_input, receipt_path=receipt_input)
    assert alias_read_calls == []
    assert calls == {"verify": 0, "parse": 0, "network": 0}


def test_direct_ingest_uses_valid_authenticated_metadata_and_preserves_snapshots(tmp_path, monkeypatch):
    payload = (b"delta_time_ms,hmd_rot_w,hmd_rot_x,hmd_rot_y,hmd_rot_z\n"
               b"0,1,0,0,0\n10,1,0,0,0\n20,1,0,0,0\n")
    record = _record(payload)
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, [record])
    legacy_receipt_path, _ = _receipt(tmp_path, inventory_path, records, record_digest, schema=1)
    legacy_receipt_bytes = legacy_receipt_path.read_bytes()
    recording = tmp_path / record["relative_path"]
    recording.parent.mkdir(parents=True, exist_ok=True)
    recording.write_bytes(payload)
    def no_network(*args, **kwargs):
        raise AssertionError("verified cache should not require network")
    monkeypatch.setattr(sources, "urlopen", no_network)
    fetch_alyx(inventory_path, tmp_path, noncommercial=True, max_files=1,
               max_total_bytes=len(payload), previous_receipt_path=legacy_receipt_path)
    receipt_path = tmp_path / "download_receipt_v2.json"
    generated_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert generated_receipt["schema_version"] == 2
    assert generated_receipt["record_inventory_sha256"] == record_digest
    assert generated_receipt["previous_receipt_sha256"] == hashlib.sha256(legacy_receipt_bytes).hexdigest()
    assert legacy_receipt_path.read_bytes() == legacy_receipt_bytes
    output, report = tmp_path / "processed/out.h5", tmp_path / "quality.json"
    result = ingest_alyx(tmp_path, output, report, PipelineConfig(),
                         inventory_path=inventory_path, receipt_path=receipt_path)
    assert result["source_inventory_file_sha256"] == sha256_file(inventory_path)
    assert result["source_record_inventory_sha256"] == record_digest
    assert result["source_receipt_schema_version"] == 2
    with h5py.File(output, "r") as store:
        assert store.attrs["source_record_inventory_sha256"] == record_digest
        assert store.attrs["source_inventory_file_sha256"] == sha256_file(inventory_path)
        assert store.attrs["source_receipt_schema_version"] == 2
    assert (output.parent / "provenance" / f"inventory_{sha256_file(inventory_path)}.json").read_bytes() == inventory_path.read_bytes()
    assert (output.parent / "provenance" / f"receipt_{sha256_file(receipt_path)}.json").read_bytes() == receipt_path.read_bytes()


def test_ingest_rejects_outputs_that_collide_with_authenticated_inventory(tmp_path, monkeypatch):
    record = _record(b"synthetic opaque recording bytes")
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, [record])
    receipt_path, _ = _receipt(tmp_path, inventory_path, records, record_digest)
    with pytest.raises(ValueError, match="overwrite"):
        ingest_alyx(tmp_path, inventory_path, tmp_path / "quality.json", PipelineConfig(),
                    inventory_path=inventory_path, receipt_path=receipt_path)


@pytest.mark.parametrize("snapshot_kind", ["receipt", "inventory"])
@pytest.mark.parametrize("write_kind", ["report", "temporary"])
def test_ingest_rejects_snapshot_write_collisions_without_changing_snapshot(
        tmp_path, monkeypatch, snapshot_kind, write_kind):
    record = _record(b"synthetic opaque recording bytes")
    inventory_path, records, record_digest = _inventory(tmp_path, monkeypatch, [record])
    receipt_path, _ = _receipt(tmp_path, inventory_path, records, record_digest)
    if snapshot_kind == "receipt":
        digest = sha256_file(receipt_path)
        snapshot_name = f"receipt_{digest}.json"
    else:
        digest = sha256_file(inventory_path)
        snapshot_name = f"inventory_{digest}.json"
    snapshot_dir = tmp_path / "processed" / "provenance"
    snapshot_dir.mkdir(parents=True)
    snapshot = snapshot_dir / snapshot_name
    sentinel = b"preexisting immutable snapshot\n"
    snapshot.write_bytes(sentinel)

    output = tmp_path / "processed" / "out.h5"
    report = tmp_path / "quality.json"
    if write_kind == "report":
        report = snapshot
    else:
        os.link(snapshot, output.with_name(output.name + ".part"))
    calls = _no_recording_io(monkeypatch)
    with pytest.raises(ValueError, match="overwrite"):
        ingest_alyx(tmp_path, output, report, PipelineConfig(),
                    inventory_path=inventory_path, receipt_path=receipt_path)
    assert snapshot.read_bytes() == sentinel
    assert calls == {"verify": 0, "parse": 0, "network": 0}

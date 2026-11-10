"""End-to-end generated BSON/fpzip/TAR fixtures, user splits and resumable output."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
import uuid

import bson
import fpzip
import numpy as np
import pytest

from abc_iueh import boxrr_pilot as pilot
from abc_iueh.boxrr_source import UserArchive, file_sha256
from abc_iueh import boxrr_drive as drive


def archive(root, number, *, rate=100, member_name=None):
    user = str(uuid.UUID(int=number))
    session = str(uuid.UUID(int=number + 1000))
    t = np.arange(rate * 25, dtype=np.float32) / rate
    angle = 0.7 * np.sin(t * 1.7 + number / 10)
    q = np.column_stack((np.sin(angle / 2), t * 0, t * 0, np.cos(angle / 2)))
    doc = {"info": {"hardware": {"devices": [{"type": "HMD", "joint": "HEAD",
            "axes": ["i", "j", "k", "1"]}]}}, "frames": fpzip.compress(np.column_stack((t, q)))}
    packed = bson.encode(doc)
    path = root / f"{user}.tar"
    with tarfile.open(path, "w") as tf:
        item = tarfile.TarInfo(member_name or f"{user}/{session}.xror")
        item.size = len(packed)
        tf.addfile(item, io.BytesIO(packed))
    return UserArchive(user, f"users/0/{user}.tar", path.stat().st_size, file_sha256(path))


def entries(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [archive(raw, n) for n in range(1, 6)]
    rows.append(archive(raw, 6, rate=60))
    return raw, rows


def make_pilot(tmp_path):
    raw, rows = entries(tmp_path)
    dest = tmp_path / "pilot"
    def fetch(row, path):
        shutil.copyfile(raw / f"{row.user}.tar", path)
    result = pilot.prepare(rows, fetch, dest, source="synthetic", minimum_bytes=0,
                           maximum_bytes=1024**2, minimum_users=5)
    return dest, result


def test_full_tar_decode_filter_split_windows(tmp_path):
    dest, doc = make_pilot(tmp_path)
    train = {r["user"] for r in doc["shards"] if r["partition"] == "train"}
    valid = {r["user"] for r in doc["shards"] if r["partition"] == "validation"}
    assert len(train) == 4 and len(valid) == 1 and train.isdisjoint(valid)
    assert doc["training_allowed"] is False
    for partition in ("train", "validation"):
        x, y = next(pilot.windows(dest, partition))
        assert x.shape == (400, 4) and y.shape == (20, 4)
        assert x.dtype == y.dtype == np.float32 and not x.flags.writeable
        assert np.isfinite(x).all() and np.isfinite(y).all()
    # All raw archives were temporary and remain outside the deployable corpus.
    assert not list(dest.rglob("*.tar"))


def test_interrupted_intake_reuses_verified_users(tmp_path):
    raw, rows = entries(tmp_path)
    calls = []
    def fetch(row, path):
        calls.append(row.user)
        if row == rows[2]:
            raise OSError("interrupted download")
        shutil.copyfile(raw / f"{row.user}.tar", path)
    dest = tmp_path / "pilot"
    with pytest.raises(OSError):
        pilot.prepare(rows, fetch, dest, source="synthetic", minimum_bytes=0,
                      maximum_bytes=1024**2, minimum_users=5)
    assert not (dest / "manifest.json").exists()
    resumed = []
    def fetch_again(row, path):
        resumed.append(row.user)
        shutil.copyfile(raw / f"{row.user}.tar", path)
    pilot.prepare(rows, fetch_again, dest, source="synthetic", minimum_bytes=0,
                  maximum_bytes=1024**2, minimum_users=5)
    assert rows[0].user not in resumed and rows[1].user not in resumed


def test_target_bytes_must_be_reached(tmp_path):
    raw, rows = entries(tmp_path)
    dest = tmp_path / "pilot"
    with pytest.raises(ValueError, match="incomplete"):
        pilot.prepare(rows, lambda row, path: shutil.copyfile(raw / f"{row.user}.tar", path),
                      dest, source="synthetic", minimum_bytes=100 * 1024**2, maximum_bytes=101 * 1024**2,
                      minimum_users=5)
    assert not (dest / "manifest.json").exists()
    rejected = json.loads((dest / "users" / rows[-1].user / "record.json").read_bytes())
    assert rejected["reasons"]["native_rate_out_of_range"] == 1


def test_corrupt_cached_shard_fails(tmp_path):
    dest, doc = make_pilot(tmp_path)
    with (dest / doc["shards"][0]["path"]).open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="differs"):
        pilot.verify(dest)


def test_remote_source_identity_is_bound_to_shard_user(tmp_path):
    dest, doc = make_pilot(tmp_path)
    doc["shards"][0]["source_archive"] = doc["shards"][1]["source_archive"]
    (dest / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="source archive"):
        pilot.verify(dest)


def test_source_pin_checked_before_decode(tmp_path):
    raw, rows = entries(tmp_path)
    wrong = UserArchive(rows[0].user, rows[0].path, rows[0].size, "0" * 64)
    with pytest.raises(ValueError, match="official metadata"):
        pilot.prepare([wrong], lambda row, path: shutil.copyfile(raw / f"{row.user}.tar", path),
                      tmp_path / "pilot", source="synthetic", minimum_bytes=0, minimum_users=2)


def test_official_archive_resource_bound():
    user = str(uuid.UUID(int=1))
    assert UserArchive(user, f"users/0/{user}.tar", 4 * 1024**3, "a" * 64).size == 4 * 1024**3
    with pytest.raises(ValueError, match="archive descriptor"):
        UserArchive(user, f"users/0/{user}.tar", 4 * 1024**3 + 1, "a" * 64)


def test_traversal_rejected_without_extraction(tmp_path):
    row = archive(tmp_path, 1, member_name="../escape.xror")
    with pytest.raises(ValueError, match="Unsafe"):
        list(pilot._sessions(tmp_path / f"{row.user}.tar", row.user))
    assert not (tmp_path.parent / "escape.xror").exists()


def test_recordings_without_head_are_rejected_and_counted(tmp_path, monkeypatch):
    brush = {"info": {"hardware": {"devices": [{"type": "OTHER", "axes": ["x"]}]}}}
    monkeypatch.setattr(pilot, "_sessions", lambda *args: iter([
        ("brush", brush, None), ("damaged", None, "invalid_bson_or_fpzip")]))
    result = pilot._write_user(tmp_path / "unused", "user", tmp_path / "out.h5")
    assert result["samples"] == 0
    assert result["reasons"] == {"missing_or_invalid_head_schema": 1, "invalid_bson_or_fpzip": 1}


def test_gate_blocks_before_network_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["boxrr_pilot", str(tmp_path / "out")])
    monkeypatch.setattr(pilot, "http_session", lambda: pytest.fail("network must not open"))
    with pytest.raises(PermissionError):
        pilot.main()
    assert not (tmp_path / "out").exists()


def test_synthetic_cannot_be_uploaded_as_training(tmp_path):
    dest, _ = make_pilot(tmp_path)
    with pytest.raises(ValueError, match="complete"):
        drive.upload(None, "unused", dest, "unused", tmp_path / "upload")


def test_restore_with_pinned_index_and_remote_bytes(tmp_path, monkeypatch):
    dest, doc = make_pilot(tmp_path)
    objects, index = {}, {"schema": "abc-boxrr-drive-v1", "files": []}
    for name in [r["path"] for r in doc["shards"]] + ["manifest.json"]:
        raw = (dest / name).read_bytes()
        fid = str(len(objects))
        objects[fid] = raw
        index["files"].append({"path": name, "id": fid, "bytes": len(raw),
                               "sha256": hashlib.sha256(raw).hexdigest()})
    objects["index"] = json.dumps(index).encode()
    monkeypatch.setattr(drive, "get_file", lambda api, fid, path: path.write_bytes(objects[fid]))
    target = tmp_path / "restore"
    drive.fetch(None, "index", hashlib.sha256(objects["index"]).hexdigest(), target)
    assert pilot.verify(target) == doc
    objects["0"] += b"corrupt"
    with pytest.raises(ValueError, match="differs"):
        drive.fetch(None, "index", hashlib.sha256(objects["index"]).hexdigest(), tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_existing_pilot_cannot_change_intake_settings(tmp_path):
    dest, _ = make_pilot(tmp_path)
    with pytest.raises(ValueError, match="settings changed"):
        pilot.prepare([], None, dest, source="synthetic", minimum_bytes=0,
                      maximum_bytes=2 * 1024**2, minimum_users=5)


def test_shard_batch_model_optimizer_and_checkpoint_on_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    model_path = (Path(__file__).resolve().parents[1] /
                  "research_checkpoints/vast_20261031/abc_topic_discovery/workflow_runtime/staging/"
                  "ordered_motion_model_v3/ordered_motion_model.py")
    assert file_sha256(model_path) == "731ba2dde12293b8111d4937f1fb7de593566d017a74ac31513330f8b57dd5ee"
    spec = importlib.util.spec_from_file_location("boxrr_test_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest, _ = make_pilot(tmp_path)
    x, y = next(pilot.windows(dest, "train"))
    x, y = torch.from_numpy(x.copy())[None], torch.from_numpy(y.copy())[None]
    model = module.OrderedMotionProduct(seed=pilot.SEED).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = module.endpoint20_geodesic_loss(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    path = tmp_path / "generated-checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
    restored = module.OrderedMotionProduct(seed=pilot.SEED).cpu()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_training_sampler_and_resume_on_generated_fixture(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from abc_iueh import boxrr_train as trainer
    dest, doc = make_pilot(tmp_path)
    sampler = trainer.Batches(dest, doc, "train", pilot.SEED)
    assert len(sampler.users) == 4
    expected = {row["path"] for row in doc["shards"] if row["partition"] == "train"}
    assert {path for path, _, _ in sampler.users} == expected
    x, y = sampler.batch(3)
    assert tuple(x.shape) == (3, 400, 4) and tuple(y.shape) == (3, 20, 4)
    sampler.close()
    model_path = (Path(__file__).resolve().parents[1] /
                  "research_checkpoints/vast_20261031/abc_topic_discovery/workflow_runtime/staging/"
                  "ordered_motion_model_v3/ordered_motion_model.py")
    with pytest.raises(ValueError, match="Synthetic"):
        trainer.train(dest, model_path, tmp_path / "blocked", steps=1, batch_size=2)
    # Exercise the optimizer boundary without changing fixture provenance on disk.
    monkeypatch.setattr(trainer, "verify", lambda root: {**doc, "training_allowed": True})
    first = trainer.train(dest, model_path, tmp_path / "run", steps=1, batch_size=2)
    second = trainer.train(dest, model_path, tmp_path / "run", steps=2, batch_size=2, resume=True)
    assert first["step"] == 1 and second["step"] == 2
    assert first["checkpoint_sha256"] != second["checkpoint_sha256"]
    assert second["validation_evaluated"] is False
    validation = trainer.evaluate(dest, model_path, tmp_path / "run", windows_per_user=3, batch_size=2)
    assert validation["partition"] == "validation" and validation["users"] == 1
    assert np.isfinite([validation["model_deg"], validation["zero_velocity_deg"],
                        validation["constant_angular_velocity_1_deg"]]).all()


def test_hf_pagination_accepts_encoded_path_but_rejects_foreign_host():
    from abc_iueh.boxrr_source import _pages, TREE
    class Response:
        def __init__(self, next_url):
            self.links = {"next": {"url": next_url}} if next_url else {}
        def raise_for_status(self): pass
        def json(self): return [{"type": "file"}]
    class Session:
        def __init__(self, next_url): self.next_url, self.calls = next_url, 0
        def get(self, url, timeout):
            self.calls += 1
            return Response(self.next_url if self.calls == 1 else None)
    encoded = TREE.replace("/users", "/users%2F0") + "?cursor=opaque"
    assert len(list(_pages(Session(encoded), TREE + "/0"))) == 2
    with pytest.raises(ValueError, match="pagination"):
        list(_pages(Session("https://example.com/foreign"), TREE + "/0"))

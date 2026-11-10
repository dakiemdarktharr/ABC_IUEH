"""Synthetic-only experiment boundary and scientific-protocol tests."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from abc_iueh import boxrr_experiment as exp
from abc_iueh.boxrr_train import Batches, model_module


BASELINE = (Path(__file__).resolve().parents[1] /
            "research_checkpoints/vast_20261031/abc_topic_discovery/workflow_runtime/staging/"
            "ordered_motion_model_v3/ordered_motion_model.py")


def generated_q(length, velocity=.003):
    angle = np.arange(length, dtype=np.float32) * velocity
    return np.stack([np.cos(angle / 2), np.sin(angle / 2), np.zeros(length), np.zeros(length)],
                    axis=-1).astype(np.float32)


def generated_corpus(tmp_path):
    rows = []
    for i in range(4):
        user = f"generated-{i}"
        relative = f"users/{user}/data.h5"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as store:
            session = store.create_group("sessions/session")
            for j, length in enumerate((440, 530)):
                group = session.create_group(f"segments/segment{j}")
                group.create_dataset("quaternion", data=generated_q(length, .001 * (i + j + 1)))
        rows.append({"user": user, "path": relative, "partition": "validation" if i == 3 else "train"})
    manifest = {"source": "synthetic", "training_allowed": False, "shards": rows}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_inner_split_disjoint_stable_and_outer_unchanged():
    rows = [{"user": f"u{i}", "partition": "train" if i < 287 else "validation"} for i in range(359)]
    split = exp.split_users({"shards": rows})
    assert {k: len(v) for k, v in split.items()} == {"fit": 258, "inner": 29, "outer": 72}
    assert not (set(split["fit"]) & set(split["inner"]) or
                set(split["fit"]) & set(split["outer"]) or set(split["inner"]) & set(split["outer"]))
    reversed_split = exp.split_users({"shards": list(reversed(rows))})
    assert all(set(split[k]) == set(reversed_split[k]) for k in split)
    assert split["outer"] == [f"u{i}" for i in range(287, 359)]


def test_duplicate_user_partition_is_rejected():
    manifest = {"shards": [{"user": "x", "partition": "train"},
                           {"user": "x", "partition": "validation"}]}
    with pytest.raises(ValueError, match="leakage"):
        exp.split_users(manifest, inner_count=1)


def test_official_entrypoint_refuses_synthetic_before_output(tmp_path, monkeypatch):
    manifest = generated_corpus(tmp_path)
    monkeypatch.setattr(exp, "verify", lambda _: manifest)
    monkeypatch.setattr(exp, "model_module", lambda _: pytest.fail("Model must not be loaded"))
    with pytest.raises(ValueError, match="Synthetic"):
        exp.train_all(tmp_path, BASELINE, tmp_path / "blocked", device="cpu")
    assert not (tmp_path / "blocked").exists()


def test_ram_loader_only_opens_allowed_users_and_segments_stay_separate(tmp_path):
    manifest = generated_corpus(tmp_path)
    # Delete the disallowed user file; selecting the fit set must never open it.
    (tmp_path / manifest["shards"][-1]["path"]).unlink()
    allowed = [row["user"] for row in manifest["shards"][:-1]]
    corpus = exp.RamCorpus.load(tmp_path, manifest, allowed)
    x, y, draws = corpus.batch(np.random.default_rng(3), 1000, return_draws=True)
    assert x.shape == (1000, 400, 4) and y.shape == (1000, 20, 4)
    for i, (u, segment, start) in enumerate(draws):
        expected = corpus.users[u].segments[segment][start:start + 420]
        np.testing.assert_array_equal(torch.cat((x[i], y[i])).numpy(), expected)
        assert start + 420 <= len(corpus.users[u].segments[segment])
    assert all(not s.flags.writeable for u in corpus.users for s in u.segments)


def test_legacy_sampler_is_byte_identical_to_existing_sampler(tmp_path):
    manifest = generated_corpus(tmp_path)
    users = [r["user"] for r in manifest["shards"] if r["partition"] == "train"]
    ram = exp.RamCorpus.load(tmp_path, manifest, users)
    old = Batches(tmp_path, manifest, "train", exp.LEGACY_SEED)
    rng = np.random.default_rng(exp.LEGACY_SEED)
    try:
        for user_index in range(len(users)):
            for count in (128, 128):
                x, y = ram.batch(rng, count, user_index)
                ox, oy = old.batch(count, user_index)
                torch.testing.assert_close(x, ox, rtol=0, atol=0)
                torch.testing.assert_close(y, oy, rtol=0, atol=0)
    finally:
        old.close()


def test_live_training_sampler_is_byte_identical_to_existing_sampler(tmp_path):
    manifest = generated_corpus(tmp_path)
    users = [r["user"] for r in manifest["shards"] if r["partition"] == "train"]
    ram = exp.RamCorpus.load(tmp_path, manifest, users)
    old = Batches(tmp_path, manifest, "train", exp.SEED)
    rng = np.random.default_rng(exp.SEED)
    try:
        x, y = ram.batch(rng, 256)
        ox, oy = old.batch(256)
        torch.testing.assert_close(x, ox, rtol=0, atol=0)
        torch.testing.assert_close(y, oy, rtol=0, atol=0)
    finally:
        old.close()


def test_bootstrap_resamples_users_and_paired_constant_difference():
    assert exp.bootstrap_mean(np.array([2., 2., 2.])) == [2.0, 2.0]
    trajectory = np.array([1., 5., 9.], dtype=np.float32)[:, None, None] * np.ones((3, 10, 20))
    values = exp.summarise_errors({"ordered": trajectory, "improved": trajectory - .5},
                                 np.full((3, 10), 20.0))
    assert values["ordered"]["endpoint20_mean_deg"] == 5.0
    assert values["improved"]["paired_user_delta_ci95_deg"] == [-.5, -.5]
    assert values["improved"]["fraction_users_better_than_ordered"] == 1.0
    assert values["ordered"]["speed_bins"][1]["windows"] == 30
    assert values["ordered"]["speed_bins"][1]["users"] == 3


def test_baselines_use_context_only_and_exact_constant_rotation():
    q = torch.from_numpy(generated_q(420, .004))[None]
    baselines = exp.baseline_predictions(q[:, :400])
    for name in ("cav1", "cav5", "cav20"):
        error = exp.models.geodesic_radians(baselines[name], q[:, 400:])
        assert float(error.max()) < 3e-6
    expected_zv_endpoint_deg = .004 * 20 * 180 / np.pi
    actual = torch.rad2deg(exp.models.geodesic_radians(baselines["zero_velocity"][:, -1], q[:, -1]))
    assert float(actual) == pytest.approx(expected_zv_endpoint_deg, abs=3e-5)
    assert float(exp.context_speed(q[:, :400])) == pytest.approx(.004 * 100 * 180 / np.pi, abs=3e-4)


def test_outer_requires_every_lock_before_loading_any_signals(tmp_path, monkeypatch):
    (tmp_path / "protocol.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp, "verify", lambda _: pytest.fail("Corpus accessed before lock"))
    monkeypatch.setattr(exp.RamCorpus, "load", lambda *args: pytest.fail("Outer accessed before lock"))
    with pytest.raises(FileNotFoundError):
        exp.evaluate_all(tmp_path / "unused", BASELINE, tmp_path, device="cpu")


def test_lock_hash_tampering_is_rejected(tmp_path):
    exp.atomic_json(tmp_path / "protocol.json", {"models": list(exp.NAMES)})
    protocol_sha = exp.file_sha256(tmp_path / "protocol.json")
    locks = {}
    for name in exp.NAMES:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "best.pt").write_bytes(b"generated checkpoint bytes")
        lock = {"binding": exp._binding(protocol_sha, name),
                "checkpoint_sha256": exp.file_sha256(directory / "best.pt")}
        exp.atomic_json(directory / "lock.json", lock)
        locks[name] = lock
    exp.atomic_json(tmp_path / "all_models_locked.json", {"protocol_sha256": protocol_sha, "models": locks})
    assert len(exp.verify_all_locks(tmp_path)["models"]) == 5
    (tmp_path / "tcn" / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        exp.verify_all_locks(tmp_path)


def test_all_architectures_train_checkpoint_and_restore_on_generated_arrays(tmp_path):
    torch.set_num_threads(1)
    manifest = generated_corpus(tmp_path)
    corpus = exp.RamCorpus.load(tmp_path, manifest, [r["user"] for r in manifest["shards"][:3]])
    fit, inner = corpus.subset(["generated-0", "generated-1"]), corpus.subset(["generated-2"])
    output = tmp_path / "component_smoke"
    output.mkdir()
    protocol = {"learning_rate": .001, "minimum_learning_rate": .00005,
                "weight_decay": .01, "batch_size": 4, "steps": 2,
                "max_model_seconds": 60, "gradient_clip": 1., "eval_every": 1,
                "inner_windows_per_user": 4}
    exp.atomic_json(output / "protocol.json", protocol)
    baseline = model_module(BASELINE)
    for name in exp.NAMES:
        lock = exp.train_one(name, baseline, fit, inner, protocol, output, "cpu")
        assert lock["completed_steps"] == 2 and lock["outer_evaluated"] is False
        assert lock["selected_step"] in (0, 1, 2)
        assert lock["inference_parameters"] <= lock["trainable_parameters"] <= lock["all_parameters_including_targets"]
        if name == "jepa":
            assert lock["inference_parameters"] == 104066
            assert lock["trainable_parameters"] == 128834
            assert lock["all_parameters_including_targets"] == 154370
        if name == "energy":
            history = exp.read_json(output / name / "history.json")
            assert all("energy_proposal_endpoint20_deg" in item for item in history["inner"])
        state = exp._load_checkpoint(output / name / "best.pt", lock["binding"], "cpu")
        restored = exp.models.build_model(name, baseline_module=baseline, seed=exp.SEED)
        restored.load_state_dict(state["model"])
        restored.eval()
        x, _ = fit.batch(np.random.default_rng(4), 2)
        with torch.inference_mode():
            prediction = restored(x)
        assert prediction.shape == (2, 20, 4) and torch.isfinite(prediction).all()
        with pytest.raises(FileExistsError):
            exp.train_one(name, baseline, fit, inner, protocol, output, "cpu")
        assert exp.train_one(name, baseline, fit, inner, protocol, output, "cpu", resume=True) == lock
    status = exp.read_json(output / "status.json")
    assert set(status["completed_arms"]) == set(exp.NAMES)


def test_evaluate_sample_pairs_all_models_on_same_windows(tmp_path):
    manifest = generated_corpus(tmp_path)
    corpus = exp.RamCorpus.load(tmp_path, manifest, ["generated-3"])
    class ZeroVelocity(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, None].expand(-1, 20, -1)
    fitted = {name: ZeroVelocity() for name in exp.NAMES}
    errors, speed, future, metadata, _ = exp.evaluate_sample(
        corpus, fitted, seed=22, windows=20, device="cpu", batch_size=7)
    for name in exp.NAMES:
        np.testing.assert_array_equal(errors[name], errors["zero_velocity"])
    np.testing.assert_array_equal(future, errors["zero_velocity"][:, :, -1])
    repeated = exp.evaluate_sample(corpus, fitted, seed=22, windows=20, device="cpu", batch_size=4)
    assert repeated[3]["sample_sha256"] == metadata["sample_sha256"]
    np.testing.assert_array_equal(repeated[0]["ordered"], errors["ordered"])


def test_json_writes_reject_nonfinite_report_values(tmp_path):
    with pytest.raises(ValueError):
        exp.atomic_json(tmp_path / "invalid.json", {"metric": float("nan")})
    assert not (tmp_path / "invalid.json").exists()

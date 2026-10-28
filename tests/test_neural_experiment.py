import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from abc_iueh.geometry import exp_so3, geodesic_deg, multiply_wxyz, normalize_wxyz
from abc_iueh.neural import adjacent_features
from abc_iueh.sources import sha256_file, write_json
from abc_iueh.stage import LockedDevelopmentStage, build_feature_cache
import abc_iueh.neural_experiment as experiment


def motion(length, speed=.002, offset=0):
    time = np.arange(length, dtype=np.float64)
    q = exp_so3(np.column_stack((speed * time + offset, .000001 * time**2,
                                .003 * np.sin(time / 19)))).astype(np.float32)
    q.flags.writeable = False
    return q


def test_axis_scaler_counts_each_train_increment_once_including_chunk_boundaries():
    segments = {("train_b", "s", "0"): motion(426, speed=.01),
                ("train_a", "s", "0"): motion(13, speed=-.004)}
    result = experiment._fit_input_statistics(segments, chunk_increments=7)
    all_increments = np.concatenate([adjacent_features(q[None])[0] for q in segments.values()])
    assert result["fit_partition"] == "train" and result["count"] == 425 + 12
    assert result["ddof"] == 0 and result["epsilon"] == 1e-8
    np.testing.assert_allclose(result["mean"], all_increments.astype(np.float64).mean(axis=0), atol=1e-16)
    np.testing.assert_allclose(result["std"], all_increments.astype(np.float64).std(axis=0), atol=1e-16)


def test_global_origin_sampling_preserves_boundaries_and_samples_with_replacement():
    q_a, q_b = motion(422, offset=.2), motion(424, offset=.7)
    segments = {("z_user", "s", "1"): q_b, ("a_user", "s", "0"): q_a,
                ("short", "s", "0"): motion(15)}
    sampler = experiment._TrainingOrigins(segments)
    assert sampler.total == 3 + 5
    assert sampler.counts.tolist() == [3, 5]
    assert all(np.shares_memory(view, segments[key]) and not view.flags.writeable
               for key, view in zip(sampler.keys, sampler.views))
    context, target = sampler.batch_at(np.arange(8))
    for index in range(8):
        source, start = (q_a, index) if index < 3 else (q_b, index - 3)
        np.testing.assert_array_equal(context[index], source[start:start + 400])
        np.testing.assert_array_equal(target[index], source[start + 400:start + 420])
    indexes = np.random.default_rng(92).integers(0, 8, size=128)
    sampled = sampler.sample(np.random.default_rng(92), 128)
    expected = sampler.batch_at(indexes)
    np.testing.assert_array_equal(sampled[0], expected[0])
    np.testing.assert_array_equal(sampled[1], expected[1])
    np.testing.assert_array_equal(sampler.batch_at(indexes.astype(np.uint64))[0], expected[0])
    assert len(np.unique(indexes)) < len(indexes)
    for invalid in ([-1], [8], [1.2], [], [[0]]):
        with pytest.raises(ValueError):
            sampler.batch_at(invalid)


class DummyMLP:
    """A deterministic test double; no human data or real model fitting."""
    instances = []

    def __init__(self, seed=20261026, input_mean=None, input_std=None):
        self.seed = seed
        self.mean = input_mean
        self.std = input_std
        self.train_calls = []
        self.predict_calls = []
        self.instances.append(self)

    def train_step(self, context, target, learning_rate):
        assert context.shape[1:] == (400, 4) and target.shape[1:] == (20, 4)
        assert context.dtype == target.dtype == np.float32
        assert learning_rate == .001
        self.train_calls.append((context.copy(), target.copy()))
        return 1 / len(self.train_calls)

    def predict(self, context):
        self.predict_calls.append(context.copy())
        offset = .1 if self.seed == 20261026 else -.1
        result = np.repeat(context[:, -1:, :], 20, axis=1).copy()
        # Large first-19 errors make trajectory-mean scoring fail this test.
        result[:, :19] = multiply_wxyz(exp_so3([1, 0, 0]), result[:, :19])
        result[:, -1] = multiply_wxyz(exp_so3([offset, 0, 0]), result[:, -1])
        return result.astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, synthetic_seed=np.int64(self.seed))


def test_fixed_step_helper_uses_all_20_targets_and_only_training_origins():
    q = motion(423)
    sampler = experiment._TrainingOrigins({("u", "s", "0"): q})
    model = DummyMLP()
    receipt = experiment._train_model(model, sampler, seed=20261026, steps=3,
                                      batch_size=4, learning_rate=.001)
    assert len(model.train_calls) == 3 and not model.predict_calls
    assert receipt["first_step"] == 1 and receipt["last_step"] == 3
    assert receipt["sampled_training_windows"] == 12
    assert receipt["population_training_windows"] == 4
    assert receipt["sampling_seed"] == 20261026
    assert receipt["first_step_loss"] == 1 and receipt["last_step_loss"] == 1 / 3
    np.testing.assert_allclose(receipt["mean_step_loss"], (1 + .5 + 1 / 3) / 3)
    rng = np.random.default_rng(20261026)
    for context, target in model.train_calls:
        indexes = rng.integers(0, 4, size=4)
        expected_context, expected_target = sampler.batch_at(indexes)
        np.testing.assert_array_equal(context, expected_context)
        np.testing.assert_array_equal(target, expected_target)


def add_cache_partition(cache, partition, segments):
    # Preserve reverse insertion in the fixture. HDF5 traversal is lexical.
    for (user, session, segment), q in sorted(segments.items(), reverse=True):
        if len(q) < 420:
            continue
        group = cache.create_group(f"{partition}/{user}/{session}/{segment}")
        origins = np.arange(399, len(q) - 20, dtype=np.int64)
        group.create_dataset("origin_index", data=origins)
        group.create_dataset("target_q", data=q[origins + 20])


def test_evaluation_full_stride1_matches_cache_order_and_means_errors(tmp_path):
    q_a, q_z = motion(422, offset=.2), motion(424, offset=.7)
    segments = {("z_user", "s", "1"): q_z, ("a_user", "s", "0"): q_a}
    models = {f"MLP_seed_{seed}": DummyMLP(seed) for seed in (20261026, 20261027)}
    with h5py.File(tmp_path / "cache.h5", "w") as cache, h5py.File(tmp_path / "errors.h5", "w") as errors:
        add_cache_partition(cache, "train", segments)
        error_group = errors.create_group("train")
        sums, counts = experiment._evaluate_partition(cache, "train", segments, models,
                                                      error_group, batch_size=2)
        assert counts == {"a_user": 3, "z_user": 5}
        assert all(len(call) <= 2 for model in models.values() for call in model.predict_calls)
        expected_windows = np.concatenate([experiment._window_view(q_a), experiment._window_view(q_z)])
        expected_errors, predictions = [], []
        for method, model in models.items():
            np.testing.assert_array_equal(np.concatenate(model.predict_calls), expected_windows[:, :400])
            prediction = model.predict(expected_windows[:, :400])[:, -1]
            angles = geodesic_deg(prediction, expected_windows[:, -1])
            expected_errors.append(angles)
            predictions.append(prediction)
            np.testing.assert_array_equal(error_group[method][:], angles)
            np.testing.assert_allclose(sums[method]["a_user"], angles[:3].sum())
            np.testing.assert_allclose(sums[method]["z_user"], angles[3:].sum())
        mean_errors = np.mean(expected_errors, axis=0)
        np.testing.assert_array_equal(error_group[experiment._MEAN_ERROR_METHOD][:], mean_errors)
        prediction_ensemble = normalize_wxyz(np.mean(predictions, axis=0))
        ensemble_errors = geodesic_deg(prediction_ensemble, expected_windows[:, -1])
        assert np.all(np.abs(mean_errors - ensemble_errors) > 1)


@pytest.mark.parametrize("corrupt", ["origin", "target", "missing_segment"])
def test_evaluation_refuses_misaligned_feature_cache(tmp_path, corrupt):
    segments = {("u", "s", "0"): motion(422)}
    models = {str(seed): DummyMLP(seed) for seed in (20261026, 20261027)}
    with h5py.File(tmp_path / "cache.h5", "w") as cache, h5py.File(tmp_path / "errors.h5", "w") as errors:
        add_cache_partition(cache, "train", segments)
        group = cache["train/u/s/0"]
        if corrupt == "origin":
            group["origin_index"][0] = 400
        elif corrupt == "target":
            group["target_q"][0] = segments[("u", "s", "0")][400]
        else:
            segments[("u", "s", "1")] = motion(420)
        with pytest.raises(ValueError, match="origins|targets|segments"):
            experiment._evaluate_partition(cache, "train", segments, models, errors.create_group("train"))


def frozen_config():
    return json.loads((Path(__file__).parents[1] / "configs/baseline_features.json").read_text())


def sealed_synthetic_bundle(tmp_path):
    source, split, run_dir = tmp_path / "source.h5", tmp_path / "split.json", tmp_path / "run"
    users = [f"synthetic_user_{i:02d}" for i in range(35)]
    partitions = {"train": users[:15], "development": users[15:20], "final": users[20:]}
    training = {}
    with h5py.File(source, "w") as store:
        store.attrs["quaternion_order"] = "wxyz"
        store.attrs["sample_rate_hz"] = 100
        for index, user in enumerate(users):
            q = motion(422 + index % 2, speed=.001 * (index + 1))
            if user in partitions["train"]:
                training[(user, "s", "0")] = q
            elif user in partitions["final"]:
                # An interpreted final sample would make feature/scaler/model
                # arithmetic fail, while whole-file checksum remains valid.
                q = np.full(q.shape, np.nan, dtype=np.float32)
            store.create_dataset(f"users/{user}/sessions/s/segments/0/quaternion", data=q)
    write_json(split, {"mode": "locked_confirmatory", "unit": "user_before_windowing",
                       "training_allowed": True, "holdout_pretraining_allowed": False,
                       "source_hdf5_sha256": sha256_file(source), "partitions": partitions})
    stage = LockedDevelopmentStage(source, split)
    build_feature_cache(stage, run_dir / "features.h5", run_dir / "scalers.json")
    return stage, run_dir, training


def test_run_neural_private_artifacts_train_only_scaler_and_aggregate_summary(tmp_path, monkeypatch):
    stage, run_dir, training = sealed_synthetic_bundle(tmp_path)
    cache_path = run_dir / "features.h5"
    real_reader = stage.iter_segments
    reads = []

    def traced_reader(partition):
        assert partition != "final", "sealed final partition was accessed"
        reads.append(partition)
        if partition == "development":
            assert all(len(model.train_calls) == 3 for model in DummyMLP.instances[-2:])
        yield from real_reader(partition)

    monkeypatch.setattr(stage, "iter_segments", traced_reader)
    monkeypatch.setattr(experiment, "RelativeMLP", DummyMLP)
    real_train = experiment._train_model

    def short_synthetic_fit(model, origins, **settings):
        assert settings["steps"] == 2000 and settings["batch_size"] == 128
        return real_train(model, origins, **dict(settings, steps=3, batch_size=4))

    monkeypatch.setattr(experiment, "_train_model", short_synthetic_fit)
    summary, error_path = experiment.run_neural(stage, cache_path, run_dir, frozen_config())
    directory = run_dir / "neural"
    assert error_path == directory / "errors.h5"
    assert set(p.name for p in directory.iterdir()) == {
        "errors.h5", "scaler.npz", "MLP_seed_20261026.npz", "MLP_seed_20261027.npz",
        "private_user_stats.json", "summary.json",
    }
    assert summary["audit"]["final_quaternion_datasets_read"] == 0
    assert reads == ["train", "development"]
    assert summary["input_scaling"]["increments"] == sum(len(q) - 1 for q in training.values())
    all_train = np.concatenate([adjacent_features(q[None])[0] for q in training.values()])
    with np.load(directory / "scaler.npz", allow_pickle=False) as scaler:
        np.testing.assert_allclose(scaler["mean"], all_train.astype(np.float64).mean(axis=0), atol=1e-15)
        np.testing.assert_allclose(scaler["std"], all_train.astype(np.float64).std(axis=0), atol=1e-15)
    encoded = json.dumps(summary)
    assert not any(user in encoded for users in stage.partitions.values() for user in users)
    private = json.loads((directory / "private_user_stats.json").read_text())
    assert set(private["train"]["user_counts"]) == set(stage.partitions["train"])
    assert summary["results"]["train"]["methods"][experiment._MEAN_ERROR_METHOD]["windows"] == 52
    assert summary["results"]["development"]["methods"][experiment._MEAN_ERROR_METHOD]["windows"] == 18
    with pytest.raises(FileExistsError):
        experiment.run_neural(stage, cache_path, run_dir, frozen_config())


@pytest.mark.parametrize("change", [
    {"partitions": ["train", "development", "final"]},
    {"final_access_allowed": True}, {"context_samples": 399},
    {"neural_baseline": dict(frozen_config()["neural_baseline"], steps=1999)},
    {"neural_baseline": dict(frozen_config()["neural_baseline"], early_stopping=True)},
    {"uncertainty": {"cluster": "window"}},
])
def test_runner_rejects_nonfrozen_settings_before_any_data_access(tmp_path, change):
    config = copy.deepcopy(frozen_config())
    config.update(change)
    with pytest.raises(ValueError, match="frozen"):
        experiment.run_neural(None, tmp_path / "absent.h5", tmp_path, config)
    assert not (tmp_path / "neural").exists()


def test_runner_requires_own_sealed_cache_before_any_quaternion_access(tmp_path):
    with pytest.raises(ValueError, match="sealed features"):
        experiment.run_neural(None, tmp_path / "other.h5", tmp_path, frozen_config())
    with pytest.raises(FileNotFoundError):
        experiment.run_neural(None, tmp_path / "features.h5", tmp_path, frozen_config())
    assert not (tmp_path / "neural").exists()


def test_runner_rejects_tampered_seal_before_any_quaternion_access(tmp_path, monkeypatch):
    stage, run_dir, _ = sealed_synthetic_bundle(tmp_path)
    with h5py.File(run_dir / "features.h5", "r+") as cache:
        cache["train/synthetic_user_00/s/0/target_q"][0] = [1, 0, 0, 0]

    def forbidden_reader(partition):
        raise AssertionError("tampered bundle reached quaternion reader")

    monkeypatch.setattr(stage, "iter_segments", forbidden_reader)
    with pytest.raises(ValueError, match="SHA-256"):
        experiment.run_neural(stage, run_dir / "features.h5", run_dir, frozen_config())
    assert not (run_dir / "neural").exists()

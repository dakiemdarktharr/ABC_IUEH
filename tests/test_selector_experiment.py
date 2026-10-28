"""Synthetic sealed-bundle and matched-training runner checks."""
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from abc_iueh.evidence import build_segment_features
from abc_iueh.experiments import run_analytic
from abc_iueh.geometry import exp_so3
from abc_iueh.selector import PARAMETERS, SEEDS, VARIANTS, SelectorMLP, normalise_features
from abc_iueh.selector_metrics import method_name
from abc_iueh.sources import sha256_file, write_json
from abc_iueh.stage import LockedDevelopmentStage, build_feature_cache
import abc_iueh.selector_experiment as experiment


ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return json.loads((ROOT / "configs/selector_study.json").read_text(encoding="utf-8"))


def motion(length, shift=0):
    t = np.arange(length)
    q = exp_so3(np.column_stack((.002 * t + shift, .000002 * t**2, .006 * np.sin(t / 13)))).astype("f4")
    q.flags.writeable = False
    return q


def add_segment(cache, partition, user, q, session="s", segment="0"):
    data = build_segment_features(q)
    group = cache.create_group(f"{partition}/{user}/{session}/{segment}")
    for key, value in data.items():
        if key == "features":
            for variant, x in value.items():
                group.create_dataset(f"features/{variant}", data=x)
        else:
            group.create_dataset(key, data=value)


class GuardedFixture:
    def __init__(self):
        self.partitions = {"train": ("a_train", "z_train"), "development": ("dev",), "final": ("sealed",)}
        self.data = {"train": {("z_train", "s", "0"): motion(424, .2),
                               ("a_train", "s", "0"): motion(422, .7)},
                     "development": {("dev", "s", "0"): motion(423, .5)}}
        self.reads = []

    def iter_segments(self, partition):
        if partition not in ("train", "development"):
            raise AssertionError("final source reader reached")
        self.reads.append(partition)
        for (u, s, g), q in self.data[partition].items():
            yield u, s, g, q


def make_small_cache(path):
    stage = GuardedFixture()
    with h5py.File(path, "w") as cache:
        for partition, segments in stage.data.items():
            for (user, session, segment), q in segments.items():
                add_segment(cache, partition, user, q, session, segment)
    return stage


def identity_scalers():
    return {variant: {"fit_partition": "train", "ddof": 0, "epsilon": 1e-8,
                      "mean": [0] * 10, "scale": [1] * 10} for variant in VARIANTS}


def test_contiguous_train_preload_exact_lexical_order_and_labels_separate(tmp_path):
    make_small_cache(tmp_path / "cache.h5")
    with h5py.File(tmp_path / "cache.h5", "r") as cache:
        features, labels = experiment._load_training_arrays(cache, identity_scalers(), "cpu", chunk_size=2)
        assert tuple(features) == VARIANTS and labels.shape == (8, 4)
        ordered = [cache["train/a_train/s/0"], cache["train/z_train/s/0"]]
        for variant in VARIANTS:
            expected = np.concatenate([group[f"features/{variant}"][:] for group in ordered])
            np.testing.assert_array_equal(features[variant], expected)
            assert features[variant].shape == (8, 10) and features[variant].dtype == torch.float32
        np.testing.assert_array_equal(labels, np.concatenate([group["expert_loss"][:] for group in ordered]).astype("f4"))


def test_shared_global_batches_and_initial_states_for_every_variant(monkeypatch):
    class TracedSelector(SelectorMLP):
        def __init__(self, seed):
            super().__init__(seed)
            self.seen = []

        def forward(self, features):
            self.seen.append(features.detach().cpu().clone())
            return super().forward(features)

    monkeypatch.setattr(experiment, "SelectorMLP", TracedSelector)
    features = {variant: torch.arange(70, dtype=torch.float32).reshape(7, 10) / 50 + i
                for i, variant in enumerate(VARIANTS)}
    labels = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    models, fits, indexes = experiment._train_matched_seed(
        features, labels, seed=SEEDS[0], steps=4, batch_size=5, learning_rate=.001, device="cpu")
    expected = np.random.default_rng(SEEDS[0]).integers(0, 7, size=(4, 5), dtype=np.int64)
    np.testing.assert_array_equal(indexes, expected)
    assert len(models) == len(fits) == 6
    assert len({fit["initial_state_sha256"] for fit in fits}) == 1
    assert {fit["global_origin_batches_sha256"] for fit in fits} == {hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest()}
    for variant, fit in zip(VARIANTS, fits):
        model = models[method_name(variant, SEEDS[0])]
        assert len(model.seen) == 4
        for step in range(4):
            torch.testing.assert_close(model.seen[step], features[variant][indexes[step]], rtol=0, atol=0)
        assert fit["optimizer"] == "SGD" and fit["momentum"] == fit["weight_decay"] == 0
        assert fit["optimizer_steps"] == 4 and fit["sampled_training_windows"] == 20
        assert len(fit["step_losses"]) == 4 and np.isfinite(fit["step_losses"]).all()
        assert fit["checkpoint_selection"] == "fixed_final_step_no_development_selection"


def test_predeclared_two_variant_sensitivity_reuses_matched_helpers(tmp_path):
    make_small_cache(tmp_path / "cache.h5")
    variants = ("h1_a0", "h20_a0")
    models = {}
    with (h5py.File(tmp_path / "cache.h5", "r") as cache,
          h5py.File(tmp_path / "errors.h5", "w") as errors,
          h5py.File(tmp_path / "diagnostics.h5", "w") as diagnostics):
        features, labels = experiment._load_training_arrays(cache, identity_scalers(), "cpu", variants=variants)
        assert tuple(features) == variants
        for seed in SEEDS:
            fitted, receipts, indexes = experiment._train_matched_seed(
                features, labels, seed=seed, steps=2, batch_size=4, learning_rate=.001,
                device="cpu", variants=variants)
            assert len(fitted) == len(receipts) == 2 and indexes.shape == (2, 4)
            models.update(fitted)
        sums, counts = experiment._evaluate_partition(
            cache, "train", models, identity_scalers(), errors.create_group("train"),
            diagnostics.create_group("train"), device="cpu", variants=variants, batch_size=2)
        assert len(sums) == len(errors["train"]) == 6
        assert len(diagnostics["train"]) == 4 and counts == {"a_train": 3, "z_train": 5}
    for unsupported in (("motion_only",), ("h20_a0", "h1_a0"), ("h1_a0", "h1_a0")):
        with pytest.raises(ValueError, match="declared"):
            experiment._variant_selection(unsupported)


@pytest.mark.parametrize("field", ["current_q", "target_q", "velocities", "expert_loss", "origin_index"])
def test_independent_source_check_rejects_changed_endpoints_velocities_labels_or_origins(tmp_path, field):
    stage = make_small_cache(tmp_path / "cache.h5")
    with h5py.File(tmp_path / "cache.h5", "r+") as cache:
        receipt = experiment._validate_source_partition(cache, stage, "train", chunk_size=2)
        assert receipt["windows_verified"] == 8
        dataset = cache[f"train/a_train/s/0/{field}"]
        dataset[0] = np.asarray(dataset[0]) + 1
        with pytest.raises(ValueError, match="cached|Cached|origins"):
            experiment._validate_source_partition(cache, stage, "train", chunk_size=2)
    assert stage.reads == ["train", "train"]
    with pytest.raises(PermissionError):
        experiment._validate_source_partition(None, stage, "final")


def test_evaluation_full_coverage_exact_seed_error_mean_and_private_diagnostics(tmp_path):
    stage = make_small_cache(tmp_path / "cache.h5")
    models = {method_name(variant, seed): SelectorMLP(seed) for variant in VARIANTS for seed in SEEDS}
    scalers = identity_scalers()
    with (h5py.File(tmp_path / "cache.h5", "r") as cache,
          h5py.File(tmp_path / "errors.h5", "w") as errors,
          h5py.File(tmp_path / "diagnostics.h5", "w") as diagnostics):
        sums, counts = experiment._evaluate_partition(cache, "train", models, scalers,
                                                       errors.create_group("train"), diagnostics.create_group("train"),
                                                       device="cpu", batch_size=2)
        assert counts == {"a_train": 3, "z_train": 5}
        assert len(errors["train"]) == 18 and len(diagnostics["train"]) == 12
        assert errors["train"].attrs["row_order_sha256"] == diagnostics["train"].attrs["row_order_sha256"]
        for variant in VARIANTS:
            seed_values = [errors[f"train/{method_name(variant, seed)}"][:] for seed in SEEDS]
            mean = errors[f"train/{method_name(variant)}"][:]
            np.testing.assert_array_equal(mean, np.mean(seed_values, axis=0))
            assert sums[method_name(variant)]["a_train"] == pytest.approx(mean[:3].sum())
            assert sums[method_name(variant)]["z_train"] == pytest.approx(mean[3:].sum())
            for seed in SEEDS:
                group = diagnostics[f"train/{method_name(variant, seed)}"]
                assert set(group) == {"expected_loss_deg", "hard_error_deg", "chosen_expert"}
                assert group["chosen_expert"].dtype == np.uint8
        with pytest.raises(PermissionError):
            experiment._evaluate_partition(cache, "final", models, scalers, errors, diagnostics, device="cpu")


@pytest.mark.parametrize("change", [
    {"final_access_allowed": True}, {"partitions": ["train", "development", "final"]},
    {"variants": ["h1_a0", "h20_a0"]}, {"evidence_length": 128},
    {"device_required": "cpu"}, {"unknown_ignored_field": True},
])
def test_frozen_config_changes_refused_before_any_access(tmp_path, change):
    settings = config()
    settings.update(change)
    with pytest.raises(ValueError, match="frozen"):
        experiment.run_selector_study(None, tmp_path / "absent", tmp_path / "out", settings)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,value", [
    ("steps", 1999), ("batch_size", 64), ("learning_rate", .002), ("optimizer", "Adam"),
    ("momentum", .1), ("weight_decay", .1), ("early_stopping", True),
    ("seeds", [20261026]), ("same_origin_batches_per_seed_across_variants", False),
])
def test_nested_training_changes_are_not_silently_ignored(field, value):
    settings = config()
    settings["selector"][field] = value
    with pytest.raises(ValueError, match="frozen"):
        experiment.validate_selector_config(settings)


def test_real_study_refuses_cpu_and_optional_cuda_probe_executes():
    with pytest.raises(ValueError, match="requires CUDA"):
        experiment._cuda_runtime("cpu")
    if torch.cuda.is_available():
        device, receipt = experiment._cuda_runtime("cuda")
        assert device.type == "cuda" and receipt["actual_cuda_probe"]["numerical_check_passed"]
        assert receipt["actual_cuda_probe"]["input_is_cuda"] and receipt["actual_cuda_probe"]["output_is_cuda"]
    else:
        with pytest.raises(RuntimeError, match="cannot fall back"):
            experiment._cuda_runtime("cuda")


def sealed_synthetic_bundle(tmp_path):
    source, split, run_dir = tmp_path / "source.h5", tmp_path / "split.json", tmp_path / "baseline"
    users = [f"synthetic_user_{i:02d}" for i in range(35)]
    partitions = {"train": users[:15], "development": users[15:20], "final": users[20:]}
    with h5py.File(source, "w") as store:
        store.attrs["quaternion_order"], store.attrs["sample_rate_hz"] = "wxyz", 100
        for i, user in enumerate(users):
            q = motion(421 + i % 2, shift=.01 * i)
            if user in partitions["final"]:
                q = np.full(q.shape, np.nan, dtype="f4")
            store.create_dataset(f"users/{user}/sessions/s/segments/0/quaternion", data=q)
    write_json(split, {"mode": "locked_confirmatory", "unit": "user_before_windowing", "training_allowed": True,
                       "holdout_pretraining_allowed": False, "source_hdf5_sha256": sha256_file(source), "partitions": partitions})
    stage = LockedDevelopmentStage(source, split)
    baseline_config = json.loads((ROOT / "configs/baseline_features.json").read_text())
    build_feature_cache(stage, run_dir / "features.h5", run_dir / "scalers.json", config_path=ROOT / "configs/baseline_features.json")
    run_analytic(stage, run_dir, baseline_config, config_path=ROOT / "configs/baseline_features.json")
    settings = config()
    settings.update({"baseline_feature_cache_sha256": sha256_file(run_dir / "features.h5"),
                     "baseline_scalers_sha256": sha256_file(run_dir / "scalers.json"),
                     "source_hdf5_sha256": stage.hdf5_sha256, "split_sha256": stage.split_sha256})
    config_path = tmp_path / "selector_config.json"
    write_json(config_path, settings)
    return stage, run_dir, settings, config_path


def test_full_runner_synthetic_private_artifacts_guarded_development_and_no_checkpoint_selection(tmp_path, monkeypatch):
    # Only this synthetic integration test substitutes CPU for the hard CUDA
    # boundary. The production entry point itself always refuses CPU.
    stage, baseline, settings, config_path = sealed_synthetic_bundle(tmp_path)
    monkeypatch.setattr(experiment, "_cuda_runtime", lambda device: (torch.device("cpu"), {"synthetic_cpu_test_double": True}))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    original_train = experiment._train_matched_seed
    fit_calls, reads = [], []

    def traced_synthetic_training(features, labels, **kwargs):
        assert reads == ["train"]
        fit_calls.append(kwargs["seed"])
        # Exercise the real fixed 2000-step recipe on this tiny synthetic
        # cohort, so checkpoint/receipt step counts remain truthful.
        return original_train(features, labels, **kwargs)

    original_reader = stage.iter_segments

    def source_reader(partition):
        assert partition != "final"
        if partition == "development":
            assert fit_calls == list(SEEDS)
            assert len(list((tmp_path / "selector/checkpoints").glob("*.pt"))) == 12
        reads.append(partition)
        yield from original_reader(partition)

    monkeypatch.setattr(experiment, "_train_matched_seed", traced_synthetic_training)
    monkeypatch.setattr(stage, "iter_segments", source_reader)
    out = tmp_path / "selector"
    summary = experiment.run_selector_study(stage, baseline, out, settings, config_path=config_path)
    assert reads == ["train", "development"] and fit_calls == list(SEEDS)
    assert summary["audit"]["final_quaternion_datasets_read"] == 0
    assert summary["final_holdout_evaluated"] is False
    assert summary["architecture"] == [10, 16, 4] and summary["parameters"] == 244
    assert summary["windows"] == {"train": 37, "development": 13}
    encoded = json.dumps(summary)
    assert not any(user in encoded for users in stage.partitions.values() for user in users)
    for checkpoint in (out / "checkpoints").glob("*.pt"):
        saved = torch.load(checkpoint, weights_only=True)
        assert sum(value.numel() for value in saved["state_dict"].values()) == PARAMETERS
        assert all(torch.isfinite(value).all() for value in saved["state_dict"].values())
    receipt = json.loads((out / "receipt.json").read_text())
    assert len(receipt["checkpoint_sha256"]) == 12
    assert receipt["artifact_sha256"]["errors.h5"] == sha256_file(out / "errors.h5")
    with pytest.raises(FileExistsError):
        experiment.run_selector_study(stage, baseline, out, settings, config_path=config_path)


def test_tampered_bundle_or_changed_saved_selector_config_rejected_before_source_read(tmp_path, monkeypatch):
    stage, baseline, settings, config_path = sealed_synthetic_bundle(tmp_path)

    def forbidden(partition):
        raise AssertionError("invalid sealed bundle reached source quaternion reader")

    monkeypatch.setattr(stage, "iter_segments", forbidden)
    changed = copy.deepcopy(settings)
    changed["selector"]["steps"] = 1
    write_json(config_path, changed)
    with pytest.raises(ValueError, match="frozen"):
        experiment.run_selector_study(stage, baseline, tmp_path / "out", settings, config_path=config_path)
    write_json(config_path, settings)
    with h5py.File(baseline / "features.h5", "r+") as cache:
        cache["train/synthetic_user_00/s/0/target_q"][0] = [1, 0, 0, 0]
    with pytest.raises(ValueError, match="SHA-256"):
        experiment.run_selector_study(stage, baseline, tmp_path / "out", settings, config_path=config_path)
    assert not (tmp_path / "out").exists()

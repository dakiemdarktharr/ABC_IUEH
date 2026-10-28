"""Synthetic-only prospective R sensitivity, seal and training-budget checks."""
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
from abc_iueh.selector import SEEDS, SelectorMLP
from abc_iueh.selector_metrics import cache_row_order_sha256, method_name, summarize_selector_partition
from abc_iueh.sources import sha256_file, write_json
from abc_iueh.stage import LockedDevelopmentStage, build_feature_cache, PopulationMoments
from abc_iueh import selector_experiment as experiment
from abc_iueh import selector_sensitivity as sensitivity


ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def motion(length, shift):
    t = np.arange(length)
    return exp_so3(np.column_stack((.002 * t + shift, .000002 * t**2,
                                    .006 * np.sin(t / 13)))).astype("f4")


@pytest.fixture
def bundle(tmp_path):
    source, split, baseline = tmp_path / "source.h5", tmp_path / "split.json", tmp_path / "baseline"
    users = [f"synthetic_{i:02d}" for i in range(35)]
    partitions = {"train": users[:15], "development": users[15:20], "final": users[20:]}
    with h5py.File(source, "w") as store:
        store.attrs["quaternion_order"], store.attrs["sample_rate_hz"] = "wxyz", 100
        for i, user in enumerate(users):
            q = motion(421 + i % 2, .01 * i)
            if user in partitions["final"]:
                q[:] = np.nan  # Any accidental final feature construction fails.
            store.create_dataset(f"users/{user}/sessions/s/segments/0/quaternion", data=q)
    write_json(split, {"mode": "locked_confirmatory", "unit": "user_before_windowing", "training_allowed": True,
        "holdout_pretraining_allowed": False, "source_hdf5_sha256": sha256_file(source), "partitions": partitions})
    stage = LockedDevelopmentStage(source, split)
    baseline_config = ROOT / "configs/baseline_features.json"
    build_feature_cache(stage, baseline / "features.h5", baseline / "scalers.json", config_path=baseline_config)
    run_analytic(stage, baseline, read(baseline_config), config_path=baseline_config)
    main_config = read(ROOT / "configs/selector_study.json")
    main_config.update({"baseline_feature_cache_sha256": sha256_file(baseline / "features.h5"),
        "baseline_scalers_sha256": sha256_file(baseline / "scalers.json"),
        "source_hdf5_sha256": stage.hdf5_sha256, "split_sha256": stage.split_sha256})
    main_path, config_path = tmp_path / "main_config.json", tmp_path / "sensitivity_config.json"
    write_json(main_path, main_config)
    config = read(ROOT / "configs/selector_sensitivity.json")
    config["parent_main_config_sha256"] = sha256_file(main_path)
    write_json(config_path, config)
    return stage, baseline, config, {"config_path": config_path, "main_config_path": main_path}


@pytest.mark.parametrize("R", [128, 256])
def test_guarded_exact_raw_features_independent_train_scalers_and_immutable_parent(bundle, tmp_path, R):
    stage, baseline, config, paths = bundle
    original = {p.name: sha256_file(p) for p in baseline.iterdir() if p.is_file()}
    source_reader, reads = stage.iter_segments, []

    def guarded(partition):
        assert partition in ("train", "development")
        reads.append(partition)
        yield from source_reader(partition)

    stage.iter_segments = guarded
    child = tmp_path / f"R{R}"
    receipt = sensitivity.build_sensitivity_feature_cache(stage, baseline, child, R, config, **paths)
    assert reads == ["train", "development"]
    assert receipt["windows"] == {"train": 37, "development": 13}
    assert receipt["audit"]["final_quaternion_datasets_read"] == 0
    scalers = read(child / "scalers.json")["variants"]
    moments = {v: PopulationMoments() for v in sensitivity.VARIANTS}
    with h5py.File(child / "features.h5", "r") as cache, h5py.File(baseline / "features.h5", "r") as parent:
        assert set(cache) == {"train", "development"} and cache.attrs["evidence_length"] == R
        for partition in ("train", "development"):
            assert cache_row_order_sha256(cache, partition) == cache_row_order_sha256(parent, partition)
            for user, session, segment, q in source_reader(partition):
                data = build_segment_features(q, R, include_complete_horizon=False)
                group = cache[f"{partition}/{user}/{session}/{segment}"]
                assert set(group["features"]) == set(sensitivity.VARIANTS)
                for v in sensitivity.VARIANTS:
                    np.testing.assert_array_equal(group[f"features/{v}"][:], data["features"][v])
                    assert group[f"features/{v}"].dtype == np.float32
                    if partition == "train":
                        moments[v].update(data["features"][v])
                for field in sensitivity._SHARED:
                    np.testing.assert_array_equal(group[field][:], data[field])
        for v in sensitivity.VARIANTS:
            assert scalers[v] == moments[v].as_dict()
            assert scalers[v]["count"] == 37
    assert original == {p.name: sha256_file(p) for p in baseline.iterdir() if p.is_file()}
    sensitivity.validate_sensitivity_feature_bundle(stage, baseline, child, R, config, **paths)
    with pytest.raises(FileExistsError, match="never overwrite"):
        sensitivity.build_sensitivity_feature_cache(stage, baseline, child, R, config, **paths)


@pytest.mark.parametrize("change", [
    {"evidence_lengths": [256, 128]}, {"evidence_lengths": [128]}, {"evidence_lengths": [80, 128, 256]},
    {"final_access_allowed": True}, {"R_selection_allowed": True}, {"device_required": "cpu"},
    {"variants": ["h1_a0", "h20_a0", "motion_only"]}, {"unknown": True}, {"context_samples": True},
])
def test_frozen_config_changes_fail_without_any_source_read(tmp_path, change):
    config = read(ROOT / "configs/selector_sensitivity.json")
    config.update(change)
    with pytest.raises(ValueError, match="frozen"):
        sensitivity.run_sensitivity(None, tmp_path / "absent", tmp_path / "absent_main", tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,value", [("steps", 1), ("batch_size", 64), ("seeds", [20261026]),
    ("learning_rate", .002), ("early_stopping", True), ("momentum", .1)])
def test_frozen_training_budget_cannot_silently_change(field, value):
    config = read(ROOT / "configs/selector_sensitivity.json")
    config["selector"][field] = value
    with pytest.raises(ValueError, match="frozen"):
        sensitivity.validate_sensitivity_config(config)


@pytest.mark.parametrize("R", [True, 80, 340, 257])
def test_undeclared_R_refused_before_access(R):
    with pytest.raises(ValueError, match="prospectively"):
        sensitivity.build_sensitivity_feature_cache(None, "missing", "missing", R, {})


@pytest.mark.parametrize("field", ["origin_index", "current_q", "target_q", "velocities", "expert_loss",
                                  "features/h1_a0"])
def test_seal_detects_any_cache_tamper_and_child_schema_rejects_shared_nan(bundle, tmp_path, field):
    stage, baseline, config, paths = bundle
    child = tmp_path / "child"
    sensitivity.build_sensitivity_feature_cache(stage, baseline, child, 128, config, **paths)
    with h5py.File(child / "features.h5", "r+") as cache:
        data = cache[f"train/synthetic_00/s/0/{field}"]
        data[0] = np.asarray(data[0]) + 1
    with pytest.raises(ValueError, match="sealed sensitivity"):
        sensitivity.validate_sensitivity_feature_bundle(stage, baseline, child, 128, config, **paths)
    if field != "features/h1_a0":
        with h5py.File(child / "features.h5", "r") as cache, h5py.File(baseline / "features.h5", "r") as parent:
            with pytest.raises(ValueError, match="origins|shared"):
                sensitivity.validate_sensitivity_cache_binding(cache, parent, stage, 128)


@pytest.mark.parametrize("kind", ["final", "extra_variant", "dtype", "external", "nan_feature", "nan_shared", "count"])
def test_strict_child_schema_and_final_guards(bundle, tmp_path, kind):
    stage, baseline, config, paths = bundle
    child = tmp_path / "child"
    sensitivity.build_sensitivity_feature_cache(stage, baseline, child, 128, config, **paths)
    with h5py.File(child / "features.h5", "r+") as cache:
        group = cache["train/synthetic_00/s/0"]
        if kind == "final":
            cache.create_group("final")
        elif kind == "extra_variant":
            group.create_dataset("features/motion_only", data=np.zeros((2, 10), dtype="f4"))
        elif kind == "dtype":
            del group["features/h1_a0"]
            group.create_dataset("features/h1_a0", data=np.zeros((2, 10), dtype="f8"))
        elif kind == "external":
            del group["features/h1_a0"]
            group["features/h1_a0"] = h5py.ExternalLink(str(baseline / "features.h5"),
                                                       "/train/synthetic_00/s/0/features/h1_a0")
        elif kind == "nan_feature":
            group["features/h1_a0"][0] = np.nan
        elif kind == "nan_shared":
            group["expert_loss"][0] = np.nan
        else:
            cache.attrs["window_counts_json"] = json.dumps({"train": 36, "development": 13})
        with h5py.File(baseline / "features.h5", "r") as parent:
            with pytest.raises(ValueError):
                sensitivity.validate_sensitivity_cache_binding(cache, parent, stage, 128)


def test_scaler_tamper_rejected_even_if_hash_is_rewritten(bundle, tmp_path):
    stage, baseline, config, paths = bundle
    child = tmp_path / "child"
    sensitivity.build_sensitivity_feature_cache(stage, baseline, child, 128, config, **paths)
    scalers = read(child / "scalers.json")
    scalers["variants"]["h1_a0"]["mean"][0] += .1
    write_json(child / "scalers.json", scalers)
    receipt = read(child / "feature_receipt.json")
    receipt["artifact_sha256"]["scalers.json"] = sha256_file(child / "scalers.json")
    write_json(child / "feature_receipt.json", receipt)
    with pytest.raises(ValueError, match="train-only population"):
        sensitivity.validate_sensitivity_feature_bundle(stage, baseline, child, 128, config, **paths)


def test_parent_hash_change_fails_before_source_reader(bundle, tmp_path, monkeypatch):
    stage, baseline, config, paths = bundle
    with h5py.File(baseline / "features.h5", "r+") as cache:
        cache["train/synthetic_00/s/0/target_q"][0] = [1, 0, 0, 0]
    monkeypatch.setattr(stage, "iter_segments", lambda partition: pytest.fail("tampered parent reached source reader"))
    with pytest.raises(ValueError, match="SHA-256"):
        sensitivity.build_sensitivity_feature_cache(stage, baseline, tmp_path / "child", 128, config, **paths)
    assert not (tmp_path / "child").exists()


def test_changed_saved_config_and_missing_main_completion_fail_before_cuda(bundle, tmp_path, monkeypatch):
    stage, baseline, config, paths = bundle
    monkeypatch.setattr(experiment, "_cuda_runtime", lambda device: pytest.fail("invalid binding reached CUDA"))
    changed = copy.deepcopy(config)
    changed["selector"]["steps"] = 1
    write_json(paths["config_path"], changed)
    with pytest.raises(ValueError, match="saved sensitivity"):
        sensitivity.run_sensitivity(stage, baseline, tmp_path / "missing", tmp_path / "out", config, **paths)
    write_json(paths["config_path"], config)
    with pytest.raises(FileNotFoundError):
        sensitivity.run_sensitivity(stage, baseline, tmp_path / "missing", tmp_path / "out", config, **paths)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("binding", ["main_config", "plan", "source"])
def test_plan_source_and_parent_config_bindings_are_immutable(bundle, tmp_path, monkeypatch, binding):
    stage, baseline, config, paths = bundle
    plan_path = tmp_path / "sensitivity_plan.md"
    plan_path.write_text(sensitivity._PLAN.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sensitivity, "_PLAN", plan_path)
    child = tmp_path / "child"
    sensitivity.build_sensitivity_feature_cache(stage, baseline, child, 128, config, **paths)
    if binding == "main_config":
        with paths["main_config_path"].open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif binding == "plan":
        with plan_path.open("a", encoding="utf-8") as stream:
            stream.write("\nchanged plan\n")
    else:
        original = sensitivity._sources()
        monkeypatch.setattr(sensitivity, "_sources", lambda: {**original, "selector_sensitivity.py": "0" * 64})
    monkeypatch.setattr(stage, "iter_segments", lambda partition: pytest.fail("binding validation reached source q"))
    with pytest.raises(ValueError, match="SHA-256|bindings"):
        sensitivity.validate_sensitivity_feature_bundle(stage, baseline, child, 128, config, **paths)


def test_short_real_helper_end_to_end_aggregates_and_truthful_budget_refusal(bundle, tmp_path):
    stage, baseline, config, paths = bundle
    child = tmp_path / "child"
    sensitivity.build_sensitivity_feature_cache(stage, baseline, child, 256, config, **paths)
    scalers = read(child / "scalers.json")["variants"]
    models, all_fits = {}, []
    with (h5py.File(child / "features.h5", "r") as cache,
          h5py.File(tmp_path / "errors.h5", "x") as errors,
          h5py.File(tmp_path / "diagnostics.h5", "x") as diagnostics,
          h5py.File(baseline / "analytic_errors.h5", "r") as baseline_errors):
        features, labels = experiment._load_training_arrays(cache, scalers, "cpu", variants=sensitivity.VARIANTS)
        for seed in SEEDS:
            fitted, fits, indexes = experiment._train_matched_seed(features, labels, seed=seed, steps=2,
                batch_size=4, learning_rate=.001, device="cpu", variants=sensitivity.VARIANTS)
            models.update(fitted)
            all_fits.extend(fits)
            assert indexes.shape == (2, 4)
            for fit in fits:
                assert fit["optimizer_steps"] == 2 and len(fit["step_losses"]) == 2
                assert fit["sampled_training_windows"] == 8
                with pytest.raises(ValueError, match="truthfully"):
                    sensitivity._validate_fit(fit, seed, len(labels), fit["global_origin_batches_sha256"],
                                               fit["initial_state_sha256"])
        baseline_stats = read(baseline / "analytic_user_statistics.json")
        for partition in ("train", "development"):
            eg, dg = errors.create_group(partition), diagnostics.create_group(partition)
            sums, counts = experiment._evaluate_partition(cache, partition, models, scalers, eg, dg,
                device="cpu", batch_size=7, variants=sensitivity.VARIANTS)
            reference = baseline_stats[partition]
            summary = summarize_selector_partition(eg, sums, counts, dg, partition=partition, resamples=20,
                cache=cache, baseline_error_group=baseline_errors[partition],
                baseline_user_sums=reference["user_sums"], baseline_user_counts=reference["user_counts"],
                variants=sensitivity.VARIANTS)
            assert summary["feature_variants"] == list(sensitivity.VARIANTS)
            assert len(eg) == 6 and len(dg) == 4
            assert not any(user in json.dumps(summary) for users in stage.partitions.values() for user in users)
            assert summary["final_holdout"] == "untouched_not_evaluated_by_this_API"
    assert len(all_fits) == 4
    assert stage.verify_unchanged()["final_quaternion_datasets_read"] == 0


def test_complete_synthetic_main_and_both_children_keep_true_2000_step_receipts(bundle, tmp_path, monkeypatch):
    """CPU is substituted only at the CUDA boundary; every SGD step is real."""
    stage, baseline, config, paths = bundle
    monkeypatch.setattr(experiment, "_cuda_runtime", lambda device: (torch.device("cpu"), {"synthetic_cpu_test_double": True}))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    main, output = tmp_path / "main", tmp_path / "sensitivity"
    experiment.run_selector_study(stage, baseline, main, read(paths["main_config_path"]),
                                  config_path=paths["main_config_path"])
    parent_hashes = {p.relative_to(main).as_posix(): sha256_file(p) for p in main.rglob("*") if p.is_file()}
    summary = sensitivity.run_sensitivity(stage, baseline, main, output, config, **paths)
    assert summary["evidence_lengths"] == [128, 256] and summary["R_selection_allowed"] is False
    assert summary["final_holdout_evaluated"] is False
    assert summary["audit"]["final_quaternion_datasets_read"] == 0
    assert parent_hashes == {p.relative_to(main).as_posix(): sha256_file(p) for p in main.rglob("*") if p.is_file()}
    for R in (128, 256):
        child = output / f"R{R}" / "study"
        receipt, fits = read(child / "receipt.json"), read(child / "training_receipts.json")["fits"]
        assert len(receipt["checkpoint_sha256"]) == len(fits) == 4
        assert all(fit["optimizer_steps"] == 2000 and len(fit["step_losses"]) == 2000 for fit in fits)
        assert all(fit["sampled_training_windows"] == 256000 for fit in fits)
        assert receipt["artifact_sha256"]["summary.json"] == sha256_file(child / "summary.json")
        for seed in SEEDS:
            main_indexes = np.load(main / f"sampled_train_origins_seed_{seed}.npy", allow_pickle=False)
            child_indexes = np.load(child / f"sampled_train_origins_seed_{seed}.npy", allow_pickle=False)
            np.testing.assert_array_equal(main_indexes, child_indexes)
            for fit in [f for f in fits if f["seed"] == seed]:
                parent_fit = next(f for f in read(main / "training_receipts.json")["fits"] if f["seed"] == seed)
                assert fit["initial_state_sha256"] == parent_fit["initial_state_sha256"]
        assert not any(user in json.dumps(summary["results_by_R"][str(R)])
                       for users in stage.partitions.values() for user in users)
    with pytest.raises(FileExistsError):
        sensitivity.run_sensitivity(stage, baseline, main, output, config, **paths)
    # A changed completed main cannot be trusted for another prospective child.
    with h5py.File(main / "errors.h5", "r+") as errors:
        errors[f"train/{method_name('h1_a0', SEEDS[0])}"][0] += 1
    with pytest.raises(ValueError, match="main artifact SHA-256"):
        sensitivity.run_sensitivity(stage, baseline, main, tmp_path / "refused", config, **paths)

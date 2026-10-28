import json
from pathlib import Path
import sys
import types
import h5py
import numpy as np
import pytest

from abc_iueh.geometry import exp_so3
from abc_iueh.sources import sha256_file, write_json
from abc_iueh.stage import (
    LockedDevelopmentStage, PopulationMoments, build_feature_cache, seal_feature_receipt,
    validate_cache_binding, validate_experiment_config, validate_feature_bundle,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/baseline_features.json"


def locked_fixture(tmp_path):
    source, manifest = tmp_path / "source.h5", tmp_path / "split.json"
    users = [f"synthetic_{i:02d}" for i in range(35)]
    parts = {"train": users[:15], "development": users[15:20], "final": users[20:]}
    with h5py.File(source, "w") as store:
        store.attrs["quaternion_order"] = "wxyz"
        store.attrs["sample_rate_hz"] = 100
        for index, user in enumerate(users):
            t = np.arange(450)[:, None]
            q = exp_so3(t * np.array([0.0001 * (index + 1), 0.0002, 0])).astype(np.float32)
            if user in parts["final"]:
                q[:] = np.nan  # Sentinel: any final feature extraction fails.
            store.create_dataset(f"users/{user}/sessions/session/segments/0000/quaternion", data=q)
    write_json(manifest, {"mode": "locked_confirmatory", "unit": "user_before_windowing",
                          "training_allowed": True, "holdout_pretraining_allowed": False,
                          "source_hdf5_sha256": sha256_file(source), "partitions": parts})
    return source, manifest


def test_stage_forbids_final_and_detects_changed_split(tmp_path):
    source, manifest = locked_fixture(tmp_path)
    stage = LockedDevelopmentStage(source, manifest)
    with pytest.raises(PermissionError):
        next(stage.iter_segments("final"))
    assert len(list(stage.iter_segments("train"))) == 15
    assert stage.verify_unchanged()["final_quaternion_datasets_read"] == 0
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        stage.verify_unchanged()


def test_feature_cache_never_reads_final_and_scalers_use_train_only(tmp_path):
    source, manifest = locked_fixture(tmp_path)
    stage = LockedDevelopmentStage(source, manifest)
    cache, scales = tmp_path / "features.h5", tmp_path / "scalers.json"
    result = build_feature_cache(stage, cache, scales, config_path=CONFIG_PATH)
    assert result["windows"] == {"train": 465, "development": 155}
    assert result["audit"]["accessed_user_count"] == 20
    assert result["schema_version"] == 1
    assert validate_feature_bundle(stage, tmp_path, CONFIG_PATH) == result
    with h5py.File(cache, "r") as store:
        assert set(store.keys()) == {"train", "development"}
        x = np.concatenate([store[f"train/synthetic_{i:02d}/session/0000/features/h1_a0"][:]
                            for i in range(15)])
    scaler = json.loads(scales.read_text())["variants"]["h1_a0"]
    assert scaler["count"] == 465
    np.testing.assert_allclose(scaler["mean"], x.astype(np.float64).mean(axis=0), atol=1e-12)


def test_population_moments_are_chunk_invariant():
    x = np.random.default_rng(7).normal(size=(100, 10))
    state = PopulationMoments()
    state.update(x[:27])
    state.update(x[27:])
    result = state.as_dict()
    np.testing.assert_allclose(result["mean"], x.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(result["std"], x.std(axis=0), atol=1e-15)


@pytest.fixture
def feature_bundle(tmp_path):
    source, manifest = locked_fixture(tmp_path)
    stage = LockedDevelopmentStage(source, manifest)
    build_feature_cache(stage, tmp_path / "features.h5", tmp_path / "scalers.json", config_path=CONFIG_PATH)
    return stage, tmp_path


def unseal(directory):
    path = directory / "feature_receipt.json"
    receipt = json.loads(path.read_text())
    for key in ("schema_version", "sealed_utc", "scalers_sha256", "experiment_plan_sha256", "config_sha256"):
        receipt.pop(key)
    write_json(path, receipt)
    return receipt


@pytest.mark.parametrize("key,value", [
    ("stage", "other"), ("seed", 7), ("partitions", ["train", "final"]),
    ("context_samples", 399), ("horizon_samples", 10), ("stride", 2),
    ("evidence_length", 79), ("complete_horizon", False), ("rank_temperature", 2.0),
    ("feature_standardization", "development"), ("final_access_allowed", True),
    ("selection_after_final_results_allowed", True), ("model_or_feature_artifacts_public", True),
    ("seed", True), ("context_samples", 400.0), ("complete_horizon", 1),
])
def test_frozen_experiment_config_rejects_unsupported_fields(key, value):
    config = json.loads(CONFIG_PATH.read_text())
    config[key] = value
    with pytest.raises(ValueError, match="frozen experiment config"):
        validate_experiment_config(config)


@pytest.mark.parametrize("section,key,value", [
    ("variants", "h1_a0", [20, 0]), ("static_mixture", "fit_partition", "development"),
    ("static_mixture", "simplex_step", 0.5), ("uncertainty", "bootstrap_resamples", 100),
    ("neural_baseline", "steps", 1), ("neural_baseline", "seeds", [7, 8]),
    ("neural_baseline", "early_stopping", True),
])
def test_frozen_experiment_config_rejects_nested_plan_changes(section, key, value):
    config = json.loads(CONFIG_PATH.read_text())
    config[section][key] = value
    with pytest.raises(ValueError, match="frozen experiment config"):
        validate_experiment_config(config)


def test_frozen_experiment_config_rejects_unknown_or_missing_options():
    config = json.loads(CONFIG_PATH.read_text())
    assert validate_experiment_config(config) is config
    config["unknown"] = "ignored"
    with pytest.raises(ValueError):
        validate_experiment_config(config)
    del config["unknown"], config["stride"]
    with pytest.raises(ValueError):
        validate_experiment_config(config)


def test_seal_existing_receipt_is_idempotent_and_never_reads_source_quaternions(feature_bundle, monkeypatch):
    stage, directory = feature_bundle
    original = unseal(directory)
    with pytest.raises(ValueError, match="not sealed"):
        validate_feature_bundle(stage, directory, CONFIG_PATH)
    getitem = h5py.Dataset.__getitem__
    group_getitem = h5py.Group.__getitem__

    def forbid_source_quaternion_values(dataset, item):
        if Path(dataset.file.filename).resolve() == stage.hdf5_path.resolve() and dataset.name.endswith("/quaternion"):
            raise AssertionError("bundle validation read source quaternion values")
        return getitem(dataset, item)

    def forbid_final_source_groups(group, item):
        if (Path(group.file.filename).resolve() == stage.hdf5_path.resolve() and isinstance(item, str) and
                set((group.name + "/" + item).split("/")).intersection(stage.partitions["final"])):
            raise AssertionError("bundle validation traversed final source metadata")
        return group_getitem(group, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbid_source_quaternion_values)
    monkeypatch.setattr(h5py.Group, "__getitem__", forbid_final_source_groups)
    result = seal_feature_receipt(stage, directory, CONFIG_PATH)
    assert result["cache_sha256"] == original["cache_sha256"]
    assert result["scalers_sha256"] == sha256_file(directory / "scalers.json")
    assert result["experiment_plan_sha256"] == sha256_file(directory / "experiment_plan.json")
    assert result["config_sha256"] == sha256_file(CONFIG_PATH)
    receipt_bytes = (directory / "feature_receipt.json").read_bytes()
    assert seal_feature_receipt(stage, directory, CONFIG_PATH) == result
    assert (directory / "feature_receipt.json").read_bytes() == receipt_bytes
    assert validate_feature_bundle(stage, directory, CONFIG_PATH) == result


@pytest.mark.parametrize("mutation", ["count", "origin", "shape", "dtype", "segment", "final"])
def test_cache_binding_rejects_inconsistent_metadata(feature_bundle, mutation):
    stage, directory = feature_bundle
    with h5py.File(directory / "features.h5", "r+") as cache:
        group = cache["train/synthetic_00/session/0000"]
        if mutation == "count":
            cache.attrs["window_counts_json"] = json.dumps({"train": 466, "development": 155})
        elif mutation == "origin":
            group["origin_index"][1] = 399
        elif mutation in {"shape", "dtype"}:
            del group["features/h1_a0"]
            group.create_dataset("features/h1_a0", shape=(30 if mutation == "shape" else 31, 10),
                                 dtype="f4" if mutation == "shape" else "f8")
        elif mutation == "segment":
            del cache["train/synthetic_00/session/0000"]
        else:
            cache.create_group("final")
        with pytest.raises(ValueError):
            validate_cache_binding(cache, stage)


def test_seal_refuses_changed_original_cache_hash(feature_bundle):
    stage, directory = feature_bundle
    original = unseal(directory)
    with h5py.File(directory / "features.h5", "r+") as cache:
        cache.attrs["extra"] = "changed"
    with pytest.raises(ValueError, match="original feature receipt"):
        seal_feature_receipt(stage, directory, CONFIG_PATH)
    assert json.loads((directory / "feature_receipt.json").read_text()) == original


@pytest.mark.parametrize("mutation", ["partition", "count", "width", "string", "nonfinite", "floor", "ddof"])
def test_seal_rejects_invalid_train_only_scaler(feature_bundle, mutation):
    stage, directory = feature_bundle
    unseal(directory)
    path = directory / "scalers.json"
    scalers = json.loads(path.read_text())
    state = scalers["variants"]["h1_a0"]
    if mutation == "partition":
        state["fit_partition"] = "development"
    elif mutation == "count":
        state["count"] += 1
    elif mutation == "width":
        state["mean"].pop()
    elif mutation == "string":
        state["mean"][0] = "0"
    elif mutation == "nonfinite":
        state["mean"][0] = float("nan")
    elif mutation == "floor":
        state["scale"][0] = 0
    else:
        state["ddof"] = 1
    # Intentionally write a nonfinite JSON token for the rejection case.
    path.write_text(json.dumps(scalers))
    with pytest.raises(ValueError, match="scaler"):
        seal_feature_receipt(stage, directory, CONFIG_PATH)


@pytest.mark.parametrize("artifact", ["scalers.json", "experiment_plan.json"])
def test_sealed_bundle_detects_changed_artifact_bytes(feature_bundle, artifact):
    stage, directory = feature_bundle
    path = directory / artifact
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="mismatch"):
        validate_feature_bundle(stage, directory, CONFIG_PATH)
    with pytest.raises(ValueError, match="mismatch"):
        seal_feature_receipt(stage, directory, CONFIG_PATH)


def test_bundle_rejects_changed_config_before_consumption(feature_bundle):
    stage, directory = feature_bundle
    config_path = directory / "tampered_config.json"
    config = json.loads(CONFIG_PATH.read_text())
    config["stride"] = 20
    write_json(config_path, config)
    with pytest.raises(ValueError, match="config.stride"):
        validate_feature_bundle(stage, directory, config_path)


def test_bundle_rejects_receipt_count_mismatch(feature_bundle):
    stage, directory = feature_bundle
    path = directory / "feature_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["windows"]["train"] += 1
    write_json(path, receipt)
    with pytest.raises(ValueError, match="receipt window counts"):
        validate_feature_bundle(stage, directory, CONFIG_PATH)


def test_receipt_with_partial_seal_cannot_be_resealed(feature_bundle):
    stage, directory = feature_bundle
    path = directory / "feature_receipt.json"
    receipt = json.loads(path.read_text())
    del receipt["schema_version"]
    write_json(path, receipt)
    with pytest.raises(ValueError, match="incomplete sealed"):
        seal_feature_receipt(stage, directory, CONFIG_PATH)


def test_new_feature_writer_refuses_ignored_options(tmp_path):
    source, manifest = locked_fixture(tmp_path)
    stage = LockedDevelopmentStage(source, manifest)
    with pytest.raises(ValueError, match="feature options differ"):
        build_feature_cache(stage, tmp_path / "features.h5", tmp_path / "scalers.json",
                            include_complete_horizon=False, config_path=CONFIG_PATH)
    assert not (tmp_path / "experiment_plan.json").exists()


@pytest.mark.parametrize("command", ["baselines", "neural-baseline"])
def test_cli_validates_bundle_and_prints_only_public_runner_result(feature_bundle, monkeypatch, capsys, command):
    from abc_iueh.cli import main
    stage, directory = feature_bundle
    calls = []

    def run_analytic(received_stage, received_directory, config, *, config_path):
        calls.append((received_stage, received_directory, config_path))
        return {"public_aggregate": True}

    def run_neural(received_stage, cache_path, received_directory, config, *, config_path):
        assert cache_path == directory / "features.h5"
        calls.append((received_stage, received_directory, config_path))
        return {"public_aggregate": True}, directory / "private_errors.h5"

    analytic = types.ModuleType("abc_iueh.experiments")
    analytic.run_analytic = run_analytic
    neural = types.ModuleType("abc_iueh.neural_experiment")
    neural.run_neural = run_neural
    monkeypatch.setitem(sys.modules, "abc_iueh.experiments", analytic)
    monkeypatch.setitem(sys.modules, "abc_iueh.neural_experiment", neural)
    args = [command, "--hdf5", str(stage.hdf5_path), "--split", str(stage.split_path),
            "--config", str(CONFIG_PATH), "--output-dir", str(directory)]
    capsys.readouterr()
    assert main(args) == 0
    output = capsys.readouterr().out
    assert json.loads(output) == {"public_aggregate": True}
    assert "private_errors" not in output
    assert len(calls) == 1 and calls[0][1:] == (directory, str(CONFIG_PATH))
    path = directory / "scalers.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert len(calls) == 1  # No consumer is called after a failed bundle validation.

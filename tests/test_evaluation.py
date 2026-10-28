import h5py
import numpy as np
import pytest

from abc_iueh.evaluation import (cluster_intervals, fit_static_mixture, hard_weights,
                                 predict_endpoint, rank_weights, simplex_grid)
from abc_iueh.experiments import sample_train_calibration
from abc_iueh.geometry import exp_so3, geodesic_deg


def test_actual_blend_loss_differs_from_blended_expert_losses():
    q = np.array([[1, 0, 0, 0]], dtype=np.float32)
    v = np.zeros((1, 4, 3))
    v[0, 1, 2] = np.deg2rad(20) / 20
    target = exp_so3([[0, 0, np.deg2rad(10)]]).astype(np.float32)
    prediction = predict_endpoint(q, v, [0.5, 0.5, 0, 0])
    assert prediction.dtype == np.float32
    assert geodesic_deg(prediction, target)[0] < 1e-5
    individual = [geodesic_deg(predict_endpoint(q, v, np.eye(4)[i]), target)[0] for i in (0, 1)]
    assert np.mean(individual) == pytest.approx(10, abs=1e-5)
    weights, receipt = fit_static_mixture(q, v, target)
    assert receipt["grid_candidates"] == 35
    assert receipt["calibration_mean_geodesic_deg"] < 1e-5
    assert weights[1] == 0.5


def test_router_uses_order_only_and_deterministic_ties():
    scores = np.array([[1., 2., 3., 4.], [0, 0, 0, 0]])
    assert np.array_equal(hard_weights(scores), [[1, 0, 0, 0], [1, 0, 0, 0]])
    assert np.allclose(rank_weights(scores), rank_weights(scores * 7 + 100))
    assert np.allclose(rank_weights(scores).sum(axis=1), 1)
    with pytest.raises(ValueError):
        predict_endpoint(np.ones((1, 4)), np.zeros((1, 4, 3)), [-1, 1, 0, 1])


def test_user_bootstrap_preserves_counts_and_pairs():
    # Means are 1 and 10 with very different recording lengths.
    counts = np.array([100, 1])
    sums = np.array([[100., 200.], [10., 20.]])
    intervals, draws = cluster_intervals(sums, counts, resamples=200, seed=4)
    idx = np.random.default_rng(4).integers(0, 2, size=(200, 2))
    expected = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)[:, None]
    assert np.array_equal(draws, expected)
    assert np.array_equal(draws[:, 1], 2 * draws[:, 0])
    assert not np.isclose((sums[:, 0] / counts).mean(), sums[:, 0].sum() / counts.sum())
    assert np.array_equal(intervals, np.percentile(expected, [2.5, 97.5], axis=0))


def test_calibration_samples_train_only_in_cache_order(tmp_path):
    with h5py.File(tmp_path / "cache.h5", "w") as cache:
        for partition, base in (("train", 0), ("development", 1000)):
            for user, count, offset in (("a", 3, 0), ("b", 5, 3)):
                group = cache.create_group(f"{partition}/{user}/s/0")
                group["origin_index"] = np.arange(count) + 399
                for name, shape in (("current_q", (count, 4)), ("target_q", (count, 4)), ("velocities", (count, 4, 3))):
                    array = np.zeros(shape)
                    array[:, 0] = np.arange(count).reshape((count,) + (1,) * (len(shape) - 2)) + offset + base
                    group[name] = array
        sampled = sample_train_calibration(cache, 5, 27)
        selected = np.sort(np.random.default_rng(27).choice(8, size=5, replace=False))
        assert np.array_equal(sampled["current_q"][:, 0], selected)
        assert sampled["sample_receipt"]["population_windows"] == 8
        assert len(simplex_grid()) == 35


def test_analytic_pipeline_covers_every_origin_without_final_access(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from test_stage import locked_fixture
    from abc_iueh.experiments import run_analytic
    from abc_iueh.stage import LockedDevelopmentStage, build_feature_cache

    source, split = locked_fixture(tmp_path)
    original_getitem = h5py.Dataset.__getitem__

    def guarded_getitem(dataset, item):
        if dataset.name.endswith("/quaternion") and any(f"synthetic_{i:02d}/" in dataset.name for i in range(20, 35)):
            raise AssertionError("final quaternion read")
        return original_getitem(dataset, item)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    stage = LockedDevelopmentStage(source, split)
    build_feature_cache(stage, tmp_path / "features.h5", tmp_path / "scalers.json")
    config = json.loads((Path(__file__).parents[1] / "configs/baseline_features.json").read_text())
    result = run_analytic(stage, tmp_path, config)
    assert result["audit"]["final_quaternion_datasets_read"] == 0
    with h5py.File(tmp_path / "analytic_errors.h5", "r") as errors:
        for partition, count in (("train", 465), ("development", 155)):
            for method, metrics in result["results"][partition]["methods"].items():
                assert errors[partition][method].shape == (count,)
                assert metrics["windows"] == count
                assert metrics["mean_geodesic_deg"] == pytest.approx(errors[partition][method][:].mean())
    with pytest.raises(FileExistsError):
        run_analytic(stage, tmp_path, config)

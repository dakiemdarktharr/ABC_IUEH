"""Synthetic inference checkpoints only: no BOXRR recordings or fitted weights."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from abc_iueh import boxrr_v2_deterministic, boxrr_v2_latent, boxrr_v2_extra
from abc_iueh.boxrr_models import exp_so3

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("inference_v2_tested", HERE/"run_inference_v2.py")
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)
SOURCE = Path(boxrr_v2_deterministic.__file__).parent


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def build(name, config):
    if name in ("energy_v2", "jepa_v2"):
        return boxrr_v2_latent.build_latent_model(name, seed=17, **config)
    if name == "gru_v2":
        return boxrr_v2_extra.build_extra_model(name, seed=17, **config)
    return boxrr_v2_deterministic.build_model(name, seed=17, **config)


def create_package(tmp_path, arm="tcn_v2", width=32):
    study = tmp_path/"study"
    source_root = tmp_path/"deployment"/"src"
    folder = source_root/"abc_iueh"
    folder.mkdir(parents=True)
    for name in adapter.SOURCE_FILES:
        shutil.copyfile(SOURCE/name, folder/name)
    config = {"width": width, "latent_dim": 128, "use_gravity": True}
    sources = {name: adapter._sha(folder/name) for name in adapter.SOURCE_FILES}
    plan = {"schema": "boxrr-round2-experiment-v1", "synthetic_fixture": True, "seed": 17,
            "configs": {arm: config}, "sources": sources,
            "split": {"fit": ["SYNTHETIC_PRIVATE_USER_MUST_NOT_EXPORT"]}}
    study.mkdir()
    (study/"protocol.json").write_text(json.dumps(plan), encoding="utf-8")
    protocol_sha = adapter._sha(study/"protocol.json")
    original = build(arm, config).eval()
    (study/arm).mkdir()
    state = {"binding": {"protocol_sha256": protocol_sha, "model": arm, "phase": "base"},
             "config": config, "model": original.state_dict(), "phase_step": 4,
             "optimizer": {"private_metadata": "OPTIMIZER_MUST_NOT_EXPORT"}, "sampler_rng": {"never": "export"}}
    torch.save(state, study/arm/"best.pt")
    lock = {"protocol_sha256": protocol_sha, "sources": sources,
            "models": {arm: {"config": config, "best_sha256": adapter._sha(study/arm/"best.pt")}}}
    (study/"all_models_locked.json").write_text(json.dumps(lock), encoding="utf-8")
    destination = tmp_path/"deployment"/f"{arm}_w{width}"
    metadata = adapter.export_model(study, arm, destination, source_root)
    return study, destination, source_root, original, metadata


def context():
    increments = torch.randn(420, 3, generator=torch.Generator().manual_seed(42))*.001
    increments += torch.tensor([.002, -.001, .001])
    return exp_so3(increments.cumsum(0)).numpy().astype(np.float32)


@pytest.mark.parametrize("arm", adapter.ARMS)
@pytest.mark.parametrize("width", [32, 64])
def test_export_load_predict_all_fourteen_configurations(tmp_path, arm, width):
    study, destination, sources, original, metadata = create_package(tmp_path, arm, width)
    handle = adapter.load_model(destination)
    observed = context()[:400].copy()
    observed.setflags(write=False)
    before = observed.tobytes()
    result = adapter.predict(handle, observed)
    assert result.shape == (20, 4) and result.dtype == np.float32
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1., atol=2e-6)
    assert observed.tobytes() == before
    assert not handle.model.training and all(not parameter.requires_grad for parameter in handle.model.parameters())
    with torch.inference_mode():
        reference = original(torch.from_numpy(adapter._aligned_unit(observed))[None])[0].numpy()
    distance = np.minimum(np.linalg.norm(result-reference, axis=-1), np.linalg.norm(result+reference, axis=-1))
    assert distance.max() < 5e-5
    trimmed = torch.load(destination/"best.pt", weights_only=True)
    assert "optimizer" not in trimmed and "sampler_rng" not in trimmed and "split" not in trimmed
    assert metadata["synthetic_fixture"] is True
    assert metadata["training_checkpoint_sha256"] == adapter._sha(study/arm/"best.pt")
    assert "SYNTHETIC_PRIVATE_USER" not in (destination/"config.json").read_text(encoding="utf-8")
    assert "OPTIMIZER_MUST_NOT_EXPORT" not in repr(trimmed)


def test_inference_reproducible_sign_invariant_and_stateless(tmp_path):
    _, destination, _, _, _ = create_package(tmp_path, "energy_v2")
    handle = adapter.load_model(destination)
    values = context()
    first = adapter.predict(handle, values[:400])
    flipped = values[:400].copy()
    flipped[::2] *= -1
    np.testing.assert_allclose(first, adapter.predict(handle, flipped), atol=2e-6, rtol=0)
    adapter.predict(handle, values[1:401])
    torch.randn(40)
    np.testing.assert_array_equal(first, adapter.predict(handle, values[:400]))
    combined = np.concatenate((adapter._aligned_unit(values[:400])[-1:], first))
    assert np.all(np.sum(combined[1:]*combined[:-1], axis=-1) >= -1e-7)
    with pytest.raises(ValueError, match="400"):
        adapter.predict(handle, values)  # Never accepts the unseen future20.


@pytest.mark.parametrize("bad", [
    np.zeros((400, 4), dtype=np.float32),
    np.full((400, 4), np.nan, dtype=np.float32),
    np.full((400, 4), np.inf, dtype=np.float32),
    np.full((400, 4), .6, dtype=np.float32),
    np.ones((400, 4), dtype=np.complex64),
    np.ones((400, 4), dtype=np.int32),
    np.zeros((1, 400, 4), dtype=np.float32),
    np.zeros((399, 4), dtype=np.float32),
    np.full((400, 4), 1e300, dtype=np.float64),
])
def test_invalid_inputs_rejected_before_model_forward(bad):
    class Never(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("Invalid input reached the model")
    handle = adapter.InferenceModel(Never(), torch.device("cpu"), {})
    with pytest.raises(ValueError):
        adapter.predict(handle, bad)


def test_tiny_norm_drift_is_repaired_but_material_error_rejected(tmp_path):
    _, destination, _, _, _ = create_package(tmp_path)
    handle = adapter.load_model(destination)
    x = context()[:400]
    expected = adapter.predict(handle, x)
    np.testing.assert_allclose(expected, adapter.predict(handle, x.astype(np.float64)*1.0002), atol=2e-5, rtol=0)
    with pytest.raises(ValueError, match="unit norm"):
        adapter.predict(handle, x*1.02)


def test_source_hash_change_is_rejected_even_after_module_cache_load(tmp_path):
    _, destination, sources, _, _ = create_package(tmp_path)
    adapter.load_model(destination)
    path = sources/"abc_iueh"/"boxrr_v2_latent.py"
    path.write_text(path.read_text(encoding="utf-8")+"\n# synthetic tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        adapter.load_model(destination)


def test_inner_binding_rejected_even_with_updated_file_digest(tmp_path):
    _, destination, _, _, _ = create_package(tmp_path)
    path = destination/"best.pt"
    state = torch.load(path, weights_only=True)
    state["binding"]["model"] = "jepa_v2"
    torch.save(state, path)
    config = json.loads((destination/"config.json").read_bytes())
    config["checkpoint_sha256"] = adapter._sha(path)
    (destination/"config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="internal model"):
        adapter.load_model(destination)


def test_cli_npy_roundtrip_and_input_overwrite_guard(tmp_path):
    _, destination, _, _, _ = create_package(tmp_path, "gru_v2")
    source, output = tmp_path/"context.npy", tmp_path/"prediction.npy"
    np.save(source, context()[:400])
    adapter.main(["--model_dir", str(destination), "--input", str(source), "--output", str(output)])
    result = np.load(output, allow_pickle=False)
    assert result.shape == (20, 4) and result.dtype == np.float32
    with pytest.raises(ValueError, match="overwrite"):
        adapter.main(["--model-dir", str(destination), "--input", str(source), "--output", str(source)])


def test_raw_study_loader_and_export_refuse_to_overwrite(tmp_path):
    study, destination, sources, _, _ = create_package(tmp_path)
    raw = adapter.load_model(study/"tcn_v2", source_root=sources)
    packaged = adapter.load_model(destination)
    np.testing.assert_array_equal(adapter.predict(raw, context()[:400]), adapter.predict(packaged, context()[:400]))
    with pytest.raises(FileExistsError):
        adapter.export_model(study, "tcn_v2", destination, sources)


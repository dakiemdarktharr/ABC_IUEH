"""Offline, stateless Python adapter for the seven BOXRR v2 architectures.

Public interface: load_model(model_dir), predict(handle, float32[400,4]).
Only four hash-verified inference modules and a bound checkpoint are loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import torch

ARMS = ("ordered_v2", "mlp_v2", "tcn_v2", "energy_v2", "jepa_v2", "desp_v2", "gru_v2")
SOURCE_FILES = ("boxrr_models.py", "boxrr_v2_deterministic.py", "boxrr_v2_latent.py", "boxrr_v2_extra.py")
SCHEMA = "abc-boxrr-v2-inference-v1"


def _read(path):
    return json.loads(Path(path).read_bytes())


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _source_directory(root):
    root = Path(root).resolve()
    return root/"abc_iueh" if (root/"abc_iueh").is_dir() else root


def _verify_sources(root, expected):
    directory = _source_directory(root)
    if set(expected) != set(SOURCE_FILES):
        raise ValueError("Exactly the four known inference source hashes are required")
    for name in SOURCE_FILES:
        path = directory/name
        if not _digest(expected[name]) or not path.is_file() or _sha(path) != expected[name]:
            raise ValueError(f"Inference source hash differs: {name}")
    return directory


def _configuration(name, config):
    if name not in ARMS or not isinstance(config, dict):
        raise ValueError("Unknown model/configuration")
    # This deployment adapter is for the two prespecified experiment widths.
    if config.get("width") not in (32, 64) or type(config.get("width")) is not int:
        raise ValueError("Expected experiment width32 or width64")
    if config.get("latent_dim") != 128 or type(config.get("latent_dim")) is not int:
        raise ValueError("Expected experiment latent_dim128")
    if type(config.get("use_gravity")) is not bool:
        raise ValueError("use_gravity must be explicitly recorded")
    if set(config) != {"width", "latent_dim", "use_gravity"}:
        raise ValueError("Unexpected architecture options; review and bind them before export")


def _build(name, config, seed, source_directory, source_hashes):
    # An isolated package prevents an unrelated installed abc_iueh from being
    # imported instead of the already verified packaged source.
    signature = str(source_directory)+json.dumps(source_hashes, sort_keys=True)
    package_name = "_boxrr_v2_"+hashlib.sha256(signature.encode()).hexdigest()[:20]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_directory)]
        package.__package__ = package_name
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        sys.modules[package_name] = package
    if name in ("energy_v2", "jepa_v2"):
        module = importlib.import_module(package_name+".boxrr_v2_latent")
        return module.build_latent_model(name, seed=seed, **config)
    if name == "gru_v2":
        module = importlib.import_module(package_name+".boxrr_v2_extra")
        return module.build_extra_model(name, seed=seed, **config)
    module = importlib.import_module(package_name+".boxrr_v2_deterministic")
    return module.build_model(name, seed=seed, **config)


def _study_metadata(study, arm, source_root):
    study = Path(study)
    if arm not in ARMS:
        raise ValueError("Unknown model")
    plan, lock = _read(study/"protocol.json"), _read(study/"all_models_locked.json")
    protocol_sha = _sha(study/"protocol.json")
    if (plan.get("schema") != "boxrr-round2-experiment-v1"
            or lock.get("protocol_sha256") != protocol_sha
            or lock.get("sources") != plan.get("sources")
            or arm not in lock.get("models", {})):
        raise ValueError("Study protocol/model/source binding differs")
    architecture = plan["configs"][arm]
    _configuration(arm, architecture)
    if lock["models"][arm]["config"] != architecture:
        raise ValueError("Locked architecture differs")
    source_hashes = {name: plan["sources"][name] for name in SOURCE_FILES}
    _verify_sources(source_root, source_hashes)
    checkpoint = study/arm/"best.pt"
    digest = lock["models"][arm]["best_sha256"]
    if not _digest(digest) or _sha(checkpoint) != digest:
        raise ValueError("Selected checkpoint SHA256 differs")
    return {
        "schema": SCHEMA, "source": "BOXRR-23", "synthetic_fixture": bool(plan.get("synthetic_fixture", False)),
        "model": arm, "seed": int(plan["seed"]), "architecture": architecture,
        "protocol_sha256": protocol_sha, "checkpoint_sha256": digest,
        "training_checkpoint_sha256": digest, "source_hashes": source_hashes,
        "input_shape": [400, 4], "output_shape": [20, 4], "quaternion_order": "wxyz",
        "dtype": "float32", "stateless": True,
    }, checkpoint


def _checked_state(checkpoint, metadata):
    if _sha(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = state.get("binding", {})
    if (binding.get("protocol_sha256") != metadata["protocol_sha256"]
            or binding.get("model") != metadata["model"] or binding.get("phase") not in ("base", "refine")
            or state.get("config") != metadata["architecture"]):
        raise ValueError("Checkpoint internal model/protocol/config/phase binding differs")
    if ("training_checkpoint_sha256" in state
            and state["training_checkpoint_sha256"] != metadata["training_checkpoint_sha256"]):
        raise ValueError("Trimmed checkpoint training provenance differs")
    if not isinstance(state.get("model"), dict) or not state["model"]:
        raise ValueError("Checkpoint has no model state")
    if any(not torch.is_tensor(value) or not torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError("Checkpoint contains invalid/nonfinite model tensors")
    return state


@dataclass
class InferenceModel:
    model: torch.nn.Module
    device: torch.device
    metadata: dict


def load_model(model_dir: str, device="cpu", *, source_root=None) -> InferenceModel:
    """Load a sanitized export; raw study arm directories need source_root.

    CPU is the default. No download, fitting, scaler update or user state occurs.
    """
    directory = Path(model_dir).resolve()
    config_path = directory/"config.json"
    if config_path.exists():
        metadata = _read(config_path)
        if (metadata.get("schema") != SCHEMA or metadata.get("source") != "BOXRR-23"
                or metadata.get("input_shape") != [400, 4] or metadata.get("output_shape") != [20, 4]
                or metadata.get("quaternion_order") != "wxyz" or metadata.get("dtype") != "float32"
                or metadata.get("stateless") is not True):
            raise ValueError("Unsupported inference package interface")
        relative = Path(metadata["source_root"])
        if relative.is_absolute():
            raise ValueError("Packaged sources must use a relative path")
        package_root = directory.parent
        resolved = (directory/relative).resolve()
        if not resolved.is_relative_to(package_root):
            raise ValueError("Packaged sources escape the deployment root")
        if source_root is not None and Path(source_root).resolve() != resolved:
            raise ValueError("Do not override the hash-bound packaged source root")
        source_root = resolved
        checkpoint = directory/"best.pt"
    else:
        if source_root is None:
            raise ValueError("Raw study directories require explicit source_root")
        metadata, checkpoint = _study_metadata(directory.parent, directory.name, source_root)
    _configuration(metadata["model"], metadata["architecture"])
    if not _digest(metadata["protocol_sha256"]) or not _digest(metadata["training_checkpoint_sha256"]):
        raise ValueError("Missing protocol/training provenance digest")
    source_directory = _verify_sources(source_root, metadata["source_hashes"])
    state = _checked_state(checkpoint, metadata)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA inference is supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = _build(metadata["model"], metadata["architecture"], int(metadata["seed"]),
                   source_directory, metadata["source_hashes"])
    model.load_state_dict(state["model"], strict=True)
    model.to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    return InferenceModel(model, device, metadata)


def _aligned_unit(sequence, *, anchor=None):
    """Normalize tiny rounding drift and select a continuous quaternion sign."""
    q = np.array(sequence, dtype=np.float64, copy=True)
    norm = np.linalg.norm(q, axis=-1)
    q /= norm[:, None]
    if anchor is None:
        nonzero = np.flatnonzero(q[0] != 0)
        sign = -1. if q[0, nonzero[0]] < 0 else 1.
    else:
        sign = -1. if float(np.dot(anchor, q[0])) < 0 else 1.
    relative_sign = np.where(np.sum(q[1:]*q[:-1], axis=-1) < 0, -1., 1.)
    signs = np.concatenate(([sign], sign*np.cumprod(relative_sign)))
    return np.ascontiguousarray(q*signs[:, None], dtype=np.float32)


def predict(handle: InferenceModel, context: np.ndarray) -> np.ndarray:
    """Observed wxyz[400,4] -> predicted wxyz[20,4], both float32.

    Real floating input is accepted and converted; materially non-unit, zero,
    nonfinite or incorrectly shaped quaternions are rejected. No future input.
    """
    if not isinstance(handle, InferenceModel):
        raise TypeError("Use load_model() to obtain an inference handle")
    values = np.asarray(context)
    if values.shape != (400, 4) or values.dtype.kind != "f":
        raise ValueError("Expected real floating context[400,4] in wxyz order")
    values = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Context contains nonfinite quaternion values")
    scale = np.abs(values).max(-1)
    if np.any(scale < 1e-12):
        raise ValueError("Zero or near-zero quaternion in context")
    # Inputs are supposed to be unit quaternions. This validation also avoids
    # overflow in the ordinary norm for very large finite bad inputs.
    if np.any(scale > 1.001):
        raise ValueError("Context quaternion is not unit norm")
    norm = np.linalg.norm(values, axis=-1)
    if np.any(np.abs(norm-1.) > 1e-3):
        raise ValueError("Context quaternion is not unit norm")
    prepared = _aligned_unit(values)
    handle.model.eval()
    with torch.inference_mode():
        output = handle.model(torch.from_numpy(prepared)[None].to(handle.device))
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 20, 4):
        raise ValueError("Model output must be [1,20,4]")
    predicted = output[0].detach().cpu().numpy()
    if not np.isfinite(predicted).all():
        raise ValueError("Model produced nonfinite predictions")
    if np.max(np.abs(np.linalg.norm(predicted.astype(np.float64), axis=-1)-1.)) > 2e-4:
        raise ValueError("Model output quaternion is not unit norm")
    return _aligned_unit(predicted, anchor=prepared[-1])


def export_model(study_dir, arm, destination, source_root):
    """Write sanitized config + optimizer-free checkpoint using packaged sources.

    source_root must already contain the four original inference modules beneath
    destination.parent. No protocol split, optimizer/RNG state or user ID is copied.
    Existing completed exports are not overwritten.
    """
    destination, source_root = Path(destination).resolve(), Path(source_root).resolve()
    if not source_root.is_relative_to(destination.parent):
        raise ValueError("Copy inference sources inside the deployment root before exporting")
    metadata, checkpoint = _study_metadata(study_dir, arm, source_root)
    state = _checked_state(checkpoint, metadata)
    if (destination/"config.json").exists() or (destination/"best.pt").exists():
        raise FileExistsError("Inference export already exists")
    destination.mkdir(parents=True, exist_ok=True)
    trimmed = {
        "schema": "abc-boxrr-v2-inference-checkpoint-v1",
        "binding": {key: state["binding"][key] for key in ("protocol_sha256", "model", "phase")},
        "config": dict(state["config"]),
        "model": {key: value.detach().cpu().clone() for key, value in state["model"].items()},
        "training_checkpoint_sha256": metadata["training_checkpoint_sha256"],
        "phase_step": int(state["phase_step"]),
    }
    temporary = destination/"best.pt.part"
    torch.save(trimmed, temporary)
    temporary.replace(destination/"best.pt")
    metadata["checkpoint_sha256"] = _sha(destination/"best.pt")
    metadata["source_root"] = Path(os.path.relpath(source_root, destination)).as_posix()
    temporary_config = destination/"config.json.part"
    temporary_config.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    temporary_config.replace(destination/"config.json")
    # Mechanical package verification only; accuracy is never inferred here.
    loaded = load_model(destination)
    return {**metadata, "model_state_tensors": len(loaded.model.state_dict())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", "--model_dir", dest="model_dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Prediction output must not overwrite the input context")
    context = np.load(args.input, allow_pickle=False)
    handle = load_model(args.model_dir, device=args.device, source_root=args.source_root)
    prediction = predict(handle, context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, prediction, allow_pickle=False)


if __name__ == "__main__":
    main()


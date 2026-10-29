"""PyTorch fit-only trainer for the ordered-motion Vast experiments.

No data reader or source-identity claim lives here. A production call requires
the separately reviewed loader's root-authenticated, in-process capability.
The 100-update pilot and the later three-arm screen are separate locked
recipes; a checkpoint can resume only when every recipe/data/code/runtime pin
matches. This code does not grant data, GPU, budget, or experiment authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from .checkpoint import (
        CHECKPOINT_SCHEMA, CheckpointError, CheckpointMismatchError,
        ensure_private_directory, load_checkpoint, optimizer_state_to_device,
        require_checkpoint_match, save_checkpoint, sha256_file,
    )
    from .fit_api import (
        BudgetGate, PilotRecipe, PreparedFitBundle, RootTrainingCapability, ScreenRecipe,
        validate_bundle_shape,
    )
except ImportError:  # python trainer.py
    from checkpoint import (
        CHECKPOINT_SCHEMA, CheckpointError, CheckpointMismatchError,
        ensure_private_directory, load_checkpoint, optimizer_state_to_device,
        require_checkpoint_match, save_checkpoint, sha256_file,
    )
    from fit_api import BudgetGate, PilotRecipe, PreparedFitBundle, RootTrainingCapability, ScreenRecipe, validate_bundle_shape


PILOT_DRAFT_SHA256 = "7a650a3c29c2467de2cb4e2069e97d184b6e908b98b8d53a388b4b9637b9a96f"
THREE_ARM_DRAFT_SHA256 = "93c8165f39c4b347275db86460cbdca583a25a4aea629444e9a47002bf0acd67"
ORDERED_MODEL_SHA256 = "153663b7e5c64a1323f4111a5a310e6551c753ed6e85cb7458d5bef9ff7aa2dc"
RAW_VECTOR_SOURCE_SHA256 = "ddbe6fcd9cb59f112e19e2a4297c453ad3195169eb31615bdb9a5af96fea78dd"
RAW_VECTOR_FACTORY_SHA256 = "dda6c5c4f8c1944d63786ca690440045f523365f024ac6b147139d8f2260a86f"
MIN_FREE_RAM_BYTES = 2 * 1024**3
SCREEN_ARMS = ("ordered_product", "active_order_control", "contemporaneous_raw_vector")
SCREEN_FOLDS = (0, 1, 2)
SCREEN_SEEDS = (20261026, 20261027)
BUDGET_SOFT_USD = 16.0
BUDGET_RESERVE_USD = 4.0
BUDGET_TOTAL_USD = 20.0
CHECKPOINT_MAX_UPDATES = 1000
CHECKPOINT_INTERVAL_SECONDS = 300.0


class TrainingError(RuntimeError):
    """The locked training contract or runtime was not satisfied."""


class BudgetStop(TrainingError):
    """The external root budget controller denied the next paid update."""


def _cgroup_available_bytes(root: Path, membership: str) -> int | None:
    """Apply cgroup v2 or v1 memory limits, including constraining parents."""
    lines = [line.split(":", 2) for line in membership.splitlines() if line.count(":") >= 2]
    v2 = next((parts[2] for parts in lines if parts[1] == ""), None)
    v1 = next((parts[2] for parts in lines if "memory" in parts[1].split(",")), None)
    if v2 is not None and (root / "memory.max").is_file():
        mount, relative, limit_name, usage_name = root, v2, "memory.max", "memory.current"
    elif v1 is not None and (root / "memory" / "memory.limit_in_bytes").is_file():
        mount, relative, limit_name, usage_name = root / "memory", v1, "memory.limit_in_bytes", "memory.usage_in_bytes"
    else:
        return None
    parts = Path(relative.lstrip("/")).parts
    if ".." in parts:
        raise TrainingError("unsafe cgroup membership path")
    current = mount.joinpath(*parts)
    if not (current / limit_name).is_file():
        # A cgroup namespace commonly exposes its own root at the mount.
        current = mount
    available = None
    while True:
        try:
            limit_text = (current / limit_name).read_text(encoding="ascii").strip()
            usage = int((current / usage_name).read_text(encoding="ascii").strip())
            if usage < 0:
                raise ValueError("negative cgroup usage")
            if limit_text != "max":
                limit = int(limit_text)
                if limit <= 0:
                    raise ValueError("invalid cgroup limit")
                remaining = max(0, limit - usage)
                available = remaining if available is None else min(available, remaining)
        except (OSError, ValueError) as exc:
            raise TrainingError("cannot reliably determine cgroup memory availability") from exc
        if current == mount:
            break
        current = current.parent
    return available


def available_physical_ram_bytes() -> int:
    """Read the minimum of host-available and cgroup-available bytes."""
    if not hasattr(os, "sysconf"):
        raise TrainingError("Vast runner requires Linux sysconf RAM accounting")
    try:
        host = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError) as exc:
        raise TrainingError("could not determine available physical RAM") from exc
    if sys.platform.startswith("linux"):
        try:
            membership = Path("/proc/self/cgroup").read_text(encoding="ascii")
            group_available = _cgroup_available_bytes(Path("/sys/fs/cgroup"), membership)
        except OSError as exc:
            raise TrainingError("could not inspect cgroup membership") from exc
        if group_available is not None:
            host = min(host, group_available)
    return host


def require_ram_floor(minimum_bytes: int = MIN_FREE_RAM_BYTES) -> None:
    found = available_physical_ram_bytes()
    if found < minimum_bytes:
        raise TrainingError(f"need >= {minimum_bytes} available RAM bytes; found {found}")


def configure_pre_torch_environment(*, require_cuda: bool) -> None:
    """Set deterministic CUDA workspace before importing PyTorch."""
    if require_cuda and not sys.platform.startswith("linux"):
        raise TrainingError("the Vast launcher requires Linux")
    required = ":4096:8"
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current is not None and current != required:
        raise TrainingError("CUBLAS_WORKSPACE_CONFIG conflicts with the locked deterministic setting")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _checkpoint_due(completed: int, last_completed: int, now: float,
                    last_checkpoint_time: float, total: int) -> bool:
    """Check at an update boundary; a slow update or upload adds latency."""
    return (completed == total or completed - last_completed >= CHECKPOINT_MAX_UPDATES
            or now - last_checkpoint_time >= CHECKPOINT_INTERVAL_SECONDS)


def _primitive_state(value: Any) -> Any:
    """Convert PCG64 state to weights-only-safe Python primitives."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TrainingError("PCG64 state has a non-string key")
        return {key: _primitive_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive_state(item) for item in value]
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        item = value.item()
        if type(item) in (bool, int, float, str):
            return item
    raise TrainingError(f"unsupported sampler state value: {type(value).__name__}")


def _runner_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = ("__init__.py", "checkpoint.py", "fit_api.py", "launcher.py", "lease_clock.py", "raw_vector_adapter.py", "campaign_plan.py", "campaign_executor.py", "requirements-vast.lock", "screen_plan.py", "trainer.py",
             "../data/bundle.py", "../data/ed25519_verifier.py", "../ops_v2/budget_gate.py", "../ops_v2/checkpoint_store.py", "../ops_v2/remote_backup.py",
             "../ops_v2/run_guard.py", "../time_budget/guard.py", "../time_budget/ledger.py",
             "../time_budget/bootstrap.py", "../time_budget/combined.py", "../time_budget/supervisor_linux.py",
             "../time_budget/remote_journal.py")
    hashes = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise TrainingError(f"required runner component is missing or unsafe: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def _runner_sha256() -> str:
    return hashlib.sha256(_canonical_json(_runner_code_hashes())).hexdigest()


def _staging_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root() -> Path:
    return _staging_root().parents[2]


def _model_spec(arm_id: str) -> dict[str, Any]:
    staging = _staging_root()
    if arm_id in ("ordered_product", "active_order_control"):
        return {
            "source_path": staging / "ordered_motion_model_v3" / "ordered_motion_model.py",
            "source_sha256": ORDERED_MODEL_SHA256,
            "factory_sha256": ORDERED_MODEL_SHA256,
            "factory_name": "OrderedMotionProduct" if arm_id == "ordered_product" else "MeansOnlyControl",
            "parameter_count": 2477,
            "factory_kind": "ordered_motion_model_v3",
        }
    if arm_id == "contemporaneous_raw_vector":
        arc_source = _workspace_root() / "ABC_IUEH" / "src" / "abc_iueh" / "arc_model.py"
        adapter = Path(__file__).resolve().parent / "raw_vector_adapter.py"
        return {
            "source_path": arc_source,
            "source_sha256": RAW_VECTOR_SOURCE_SHA256,
            "factory_path": adapter,
            "factory_sha256": RAW_VECTOR_FACTORY_SHA256,
            "common_loss_source_sha256": ORDERED_MODEL_SHA256,
            "factory_name": "ArcModel(vector_angle)",
            "parameter_count": 1683,
            "factory_kind": "pinned_existing_arc_model",
        }
    raise TrainingError(f"arm is not in the locked three-arm screen: {arm_id}")


def _model_binding(arm_id: str) -> tuple[dict[str, Any], str]:
    spec = _model_spec(arm_id)
    for key in ("source_path", "factory_path"):
        if key not in spec:
            continue
        path = Path(spec[key])
        if path.is_symlink() or not path.is_file():
            raise TrainingError(f"model/factory source is missing or unsafe: {path}")
        actual = sha256_file(path)
        expected = spec["source_sha256"] if key == "source_path" else spec["factory_sha256"]
        if actual != expected:
            raise TrainingError(f"{key} digest mismatch: {actual}")
    common_loss_path = _staging_root() / "ordered_motion_model_v3" / "ordered_motion_model.py"
    common_loss_sha = sha256_file(common_loss_path)
    if common_loss_sha != ORDERED_MODEL_SHA256:
        raise TrainingError("common endpoint20 loss source digest mismatch")
    binding = {
        "arm_id": arm_id,
        "factory_kind": spec["factory_kind"],
        "factory_name": spec["factory_name"],
        "factory_sha256": spec["factory_sha256"],
        "model_source_sha256": spec["source_sha256"],
        "common_loss_source_sha256": common_loss_sha,
        "trainable_parameter_count": spec["parameter_count"],
    }
    digest = hashlib.sha256(_canonical_json(binding)).hexdigest()
    return spec, digest


def _model_binding_record(arm_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    common_loss_path = _staging_root() / "ordered_motion_model_v3" / "ordered_motion_model.py"
    common_loss_sha = sha256_file(common_loss_path)
    if common_loss_sha != ORDERED_MODEL_SHA256:
        raise TrainingError("common endpoint20 loss source digest mismatch")
    return {
        "arm_id": arm_id,
        "factory_kind": spec["factory_kind"],
        "factory_name": spec["factory_name"],
        "factory_sha256": spec["factory_sha256"],
        "model_source_sha256": spec["source_sha256"],
        "common_loss_source_sha256": common_loss_sha,
        "trainable_parameter_count": spec["parameter_count"],
    }


def _load_model(arm_id: str, seed: int) -> tuple[Any, Any, Any]:
    spec, _ = _model_binding(arm_id)
    staging = _staging_root()
    if spec["factory_kind"] == "ordered_motion_model_v3":
        path = spec["source_path"]
        module_spec = importlib.util.spec_from_file_location("abc_ordered_motion_model", path)
        if module_spec is None or module_spec.loader is None:
            raise TrainingError("cannot load pinned ordered-motion model source")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        factory = getattr(module, spec["factory_name"])
        model = factory()
        loss_module = module
    else:
        module_spec = importlib.util.spec_from_file_location("abc_raw_vector_adapter", spec["factory_path"])
        if module_spec is None or module_spec.loader is None:
            raise TrainingError("cannot load pinned raw-vector adapter")
        adapter_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(adapter_module)
        if adapter_module.pinned_source_path().resolve(strict=True) != Path(spec["source_path"]).resolve(strict=True):
            raise TrainingError("raw-vector adapter resolved a different source")
        model = adapter_module.create_raw_vector_reference(seed)
        # Common endpoint geodesic loss from the hash-pinned candidate module.
        path = staging / "ordered_motion_model_v3" / "ordered_motion_model.py"
        module_spec = importlib.util.spec_from_file_location("abc_ordered_motion_loss", path)
        if module_spec is None or module_spec.loader is None:
            raise TrainingError("cannot load pinned common geodesic loss")
        loss_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(loss_module)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != spec["parameter_count"]:
        raise TrainingError(f"{arm_id} trainable parameter count mismatch: {count}")
    return model, loss_module, spec


def _runtime_binding(torch: Any, numpy: Any, device: Any) -> dict[str, Any]:
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise TrainingError("Vast runner is pinned to CPython 3.11")
    if numpy.__version__ != "2.4.6" or torch.__version__ != "2.9.1+cu128":
        raise TrainingError("runtime differs from the reviewed project Torch/NumPy pins")
    try:
        h5py_version = package_version("h5py")
    except PackageNotFoundError as exc:
        raise TrainingError("pinned h5py package is missing from the runtime") from exc
    if h5py_version != "3.16.0":
        raise TrainingError("runtime differs from the reviewed h5py pin")
    lock_path = Path(__file__).resolve().parent / "requirements-vast.lock"
    lock_sha = sha256_file(lock_path)
    binding: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os_family": platform.system(),
        "architecture": platform.machine(),
        "numpy": numpy.__version__,
        "h5py": h5py_version,
        # TorchVersion subclasses str but torch.save(weights_only=True) and
        # our checkpoint sanitizer require plain built-in scalar values.
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "device_type": device.type,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dtype": "float32",
        "requirements_lock_sha256": lock_sha,
    }
    if device.type == "cuda":
        binding.update({
            "cudnn": torch.backends.cudnn.version(),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        })
    return binding


def _runtime_sha256(binding: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(binding))).hexdigest()


def _new_sampler(numpy: Any, *, seed: int, fold_id: int) -> tuple[Any, dict[str, Any]]:
    generator = numpy.random.default_rng(numpy.random.SeedSequence([seed, fold_id]))
    if type(generator.bit_generator) is not numpy.random.PCG64:
        raise TrainingError("the locked sampler must be NumPy PCG64")
    return generator, _primitive_state(copy.deepcopy(generator.bit_generator.state))


def _restore_sampler(numpy: Any, state: Mapping[str, Any]) -> Any:
    if state.get("bit_generator") != "PCG64":
        raise TrainingError("checkpoint sampler is not PCG64")
    bit_generator = numpy.random.PCG64()
    try:
        bit_generator.state = dict(state)
    except (TypeError, ValueError) as exc:
        raise TrainingError("checkpoint PCG64 state is malformed") from exc
    return numpy.random.Generator(bit_generator)


def _capture_rng(torch: Any) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu().clone(),
        "torch_cuda_rng_state_all": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def _restore_rng(torch: Any, state: Mapping[str, Any]) -> None:
    # This function is called after model and optimizer reconstruction. Resume
    # does not reseed any generator.
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    cuda_states = state["torch_cuda_rng_state_all"]
    if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
        raise TrainingError("checkpoint CUDA RNG state does not match this runtime")
    if cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _bundle_scope_check(bundle: PreparedFitBundle, recipe: Any) -> None:
    bundle.assert_active()
    # The signed summary is checked in addition to the live authority binding.
    summary = getattr(bundle, "identity_summary", None)
    if not isinstance(summary, Mapping):
        raise PermissionError("fit loader identity summary is missing or malformed")
    if summary.get("fold_id") != recipe.fold_id:
        raise PermissionError("fit bundle fold does not match the requested fit")
    if summary.get("role") != recipe.role:
        raise PermissionError("fit bundle role is not the locked outer_fit10 scope")
    if summary.get("purpose") != recipe.purpose:
        raise PermissionError("fit bundle purpose does not match the requested recipe")
    if summary.get("heldout_or_development_reads", 0) != 0:
        raise PermissionError("fit bundle reports development or held-out reads")
    if summary.get("participant_identifiers_in_receipt", False) is not False:
        raise PermissionError("fit bundle receipt must not expose participant identifiers")
    if type(bundle.total_origins) is not int or bundle.total_origins <= 0:
        raise PermissionError("fit bundle must contain a positive number of verified origins")
    if not isinstance(bundle.identity_sha256, str) or len(bundle.identity_sha256) != 64:
        raise PermissionError("fit bundle must supply a trusted SHA-256 identity binding")


def _load_batch(bundle: PreparedFitBundle, ranks: Any, *, numpy: Any, torch: Any,
                device: Any, batch_size: int) -> tuple[Any, Any]:
    contexts: list[Any] = []
    targets: list[Any] = []
    for rank_value in ranks:
        rank = int(rank_value)
        if not 0 <= rank < bundle.total_origins:
            raise TrainingError("sampler generated an out-of-range fit rank")
        location = bundle.origin_for_rank(rank)
        if (not isinstance(location, (tuple, list)) or len(location) != 2 or
                type(location[0]) is not int or type(location[1]) is not int):
            raise TrainingError("fit loader returned an invalid rank-to-origin mapping")
        context_bytes, target_bytes = bundle.sample_window(location[0], location[1])
        if (not isinstance(context_bytes, bytes) or len(context_bytes) != 400 * 4 * 4 or
                not isinstance(target_bytes, bytes) or len(target_bytes) != 20 * 4 * 4):
            raise TrainingError("fit loader must return FP32 [400,4] and [20,4] byte windows")
        context = numpy.frombuffer(context_bytes, dtype="<f4").reshape(400, 4).copy()
        target = numpy.frombuffer(target_bytes, dtype="<f4").reshape(20, 4).copy()
        if not numpy.isfinite(context).all() or not numpy.isfinite(target).all():
            raise TrainingError("fit batch contains non-finite values")
        if not numpy.allclose(numpy.linalg.norm(context, axis=-1), 1.0, atol=2.0e-3, rtol=0.0):
            raise TrainingError("fit context contains non-unit quaternions")
        if not numpy.allclose(numpy.linalg.norm(target, axis=-1), 1.0, atol=2.0e-3, rtol=0.0):
            raise TrainingError("fit target contains non-unit quaternions")
        contexts.append(context)
        targets.append(target)
    x = numpy.stack(contexts).astype(numpy.float32, copy=False)
    y = numpy.stack(targets).astype(numpy.float32, copy=False)
    if x.shape != (batch_size, 400, 4) or y.shape != (batch_size, 20, 4):
        raise TrainingError("fit batch has the wrong static shape")
    return torch.from_numpy(x).to(device=device), torch.from_numpy(y).to(device=device)


def _extend_draw_hash(previous: str, ranks: Any) -> str:
    rank_bytes = ranks.astype("<i8", copy=False).tobytes()
    return hashlib.sha256(bytes.fromhex(previous) + rank_bytes).hexdigest()


def _optimizer_to_device(optimizer: Any, *, torch: Any, device: Any) -> None:
    optimizer.state = optimizer_state_to_device(optimizer.state, torch=torch, device=device)


def _optimizer_update(model: Any, optimizer: Any, loss_fn: Any,
                     context: Any, target: Any, *, torch: Any) -> float:
    optimizer.zero_grad(set_to_none=True)
    prediction = model(context)
    if tuple(prediction.shape) != tuple(target.shape) or target.shape[-2:] != (20, 4):
        raise TrainingError("model must return [B,20,4] for endpoint20 supervision")
    loss = loss_fn(prediction, target)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise TrainingError("mean endpoint20 geodesic loss is non-finite")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise TrainingError("training produced a non-finite gradient")
    optimizer.step()
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise TrainingError("optimizer produced a non-finite parameter")
    return float(loss.detach().cpu().item())


def _checkpoint_payload(
    *, torch: Any, model: Any, optimizer: Any, recipe: Any, recipe_sha: str,
    model_binding: Mapping[str, Any], model_sha: str, runner_hashes: Mapping[str, str],
    runner_sha: str, runtime_binding: Mapping[str, Any], runtime_sha: str,
    bundle: PreparedFitBundle, completed_updates: int, draw_count: int,
    sampler_state: Mapping[str, Any], sampler_draw_hash: str,
    elapsed_cuda_wall: float, loss_sum: float, loss_count: int, run_id: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "draft_sha256": PILOT_DRAFT_SHA256 if isinstance(recipe, PilotRecipe) else THREE_ARM_DRAFT_SHA256,
        "recipe": recipe.as_dict(),
        "recipe_sha256": recipe_sha,
        "model_binding": dict(model_binding),
        "model_sha256": model_sha,
        "runner_code_hashes": dict(runner_hashes),
        "runner_sha256": runner_sha,
        "runtime_binding": dict(runtime_binding),
        "runtime_sha256": runtime_sha,
        "data_identity_sha256": bundle.identity_sha256,
        "total_origins": int(bundle.total_origins),
        "completed_updates": completed_updates,
        "sampler_cursor_draws": draw_count,
        "sampler_total_draws": recipe.updates * recipe.batch_size,
        "sampler_pcg64_state": _primitive_state(dict(sampler_state)),
        "sampler_draw_hash": sampler_draw_hash,
        **_capture_rng(torch),
        "model_state_dict": {name: value.detach().cpu().clone()
                             for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer_state_to_device(
            optimizer.state_dict(), torch=torch, device="cpu",
        ),
        "elapsed_cuda_wall_seconds": float(elapsed_cuda_wall),
        "train_loss_sum_radians": float(loss_sum),
        "train_loss_count": int(loss_count),
    }


def _make_recipe(mode: str, fold_id: int, training_seed: int, arm_id: str) -> Any:
    if mode == "pilot":
        if (fold_id, training_seed, arm_id) != (0, 20261026, "ordered_product"):
            raise TrainingError("pilot is fixed to fold 0, seed 20261026, ordered_product")
        return PilotRecipe()
    if mode == "screen":
        if fold_id not in (0, 1, 2) or training_seed not in SCREEN_SEEDS or arm_id not in SCREEN_ARMS:
            raise TrainingError("screen fit is outside the three-fold/two-seed/three-arm lock")
        return ScreenRecipe(fold_id=fold_id, training_seed=training_seed, arm_id=arm_id)
    raise TrainingError("mode must be pilot or screen")


def _validate_recipe(recipe: Any) -> None:
    if type(recipe) is PilotRecipe:
        if recipe != PilotRecipe():
            raise TrainingError("pilot recipe is fixed; overrides require a new reviewed lock")
        return
    if type(recipe) is ScreenRecipe:
        if (recipe.fold_id not in SCREEN_FOLDS or recipe.training_seed not in SCREEN_SEEDS or
                recipe.arm_id not in SCREEN_ARMS or recipe != ScreenRecipe(
                    fold_id=recipe.fold_id, training_seed=recipe.training_seed,
                    arm_id=recipe.arm_id,
                )):
            raise TrainingError("screen recipe differs from the fixed three-arm screen")
        return
    raise TrainingError("only the pilot or three-arm screen recipe is accepted")


def _run_fit_impl(
    *, bundle: PreparedFitBundle, root_controller: RootTrainingCapability,
    budget_gate: BudgetGate,
    recipe: Any, checkpoint_directory: Path, resume_checkpoint: Path | None = None,
    resume_checkpoint_sha256: str | None = None, device_name: str = "cuda",
    require_cuda: bool = True,
    resume_receipt_path: Path | None = None, expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Run one locked fit or resume its exact full-state checkpoint."""
    if type(expected_run_id) is not str or not re.fullmatch(r"[0-9a-f]{32}", expected_run_id):
        raise PermissionError("an independently preassigned 32-hex run ID is required")
    require_ram_floor()
    configure_pre_torch_environment(require_cuda=require_cuda)
    _validate_recipe(recipe)
    bundle = validate_bundle_shape(bundle)
    _bundle_scope_check(bundle, recipe)
    recipe_dict = recipe.as_dict()
    recipe_sha = hashlib.sha256(_canonical_json(recipe_dict)).hexdigest()
    spec, model_sha = _model_binding(recipe.arm_id)
    runner_hashes = _runner_code_hashes()
    runner_sha = _runner_sha256()
    if resume_checkpoint is None:
        if resume_checkpoint_sha256 is not None:
            raise CheckpointError("resume digest supplied without a checkpoint path")
        if resume_receipt_path is not None:
            raise CheckpointError("resume receipt supplied without a checkpoint")
    else:
        if not resume_checkpoint_sha256 or sha256_file(resume_checkpoint) != resume_checkpoint_sha256:
            raise CheckpointError("resume checkpoint must match an out-of-band digest")
        if resume_receipt_path is None:
            raise CheckpointError("resume requires a private restored-state receipt path")

    # The loader's opaque capability is authenticated outside this process by
    # the root bootstrap. It binds exact recipe, code/model/data, and resume pin.
    if not isinstance(root_controller, RootTrainingCapability):
        raise PermissionError("an authenticated root training controller is required")
    if not isinstance(budget_gate, BudgetGate):
        raise PermissionError("an authenticated live budget gate is required")
    root_controller.authorize_training(
        recipe=recipe_dict,
        recipe_sha256=recipe_sha,
        data_identity_sha256=bundle.identity_sha256,
        model_sha256=model_sha,
        runner_sha256=runner_sha,
        resume_checkpoint_sha256=resume_checkpoint_sha256,
        completed_updates=None,
    )

    if require_cuda:
        staging = _staging_root()
        if str(staging) not in sys.path:
            sys.path.insert(0, str(staging))
        from vast_training_v1.time_budget.combined import CombinedBudgetGate
        if type(budget_gate) is not CombinedBudgetGate:
            raise PermissionError("CUDA requires the reservation-backed time and USD gates")
        # The watchdog must already be active, before importing/initializing CUDA.
        budget_gate.before_update(remaining_updates=recipe.updates)

    try:
        import numpy as np
        import torch

        if require_cuda:
            if device_name != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise TrainingError("authorized training requires exactly one visible CUDA device")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.init()
            torch.cuda.synchronize(device)
        runtime_binding = _runtime_binding(torch, np, device)
        runtime_sha = _runtime_sha256(runtime_binding)
        model, loss_module, _ = _load_model(recipe.arm_id, recipe.training_seed)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=recipe.learning_rate,
            betas=(recipe.adam_beta1, recipe.adam_beta2),
            eps=recipe.adam_epsilon, weight_decay=recipe.weight_decay,
        )
        checkpoint_root = ensure_private_directory(checkpoint_directory)
        if resume_checkpoint is None and any(checkpoint_root.iterdir()):
            raise CheckpointError("new fit requires an empty private directory; request an authorized resume")

        completed = 0
        draw_count = 0
        loss_sum = 0.0
        loss_count = 0
        elapsed_before = 0.0
        run_id = expected_run_id
        if resume_checkpoint is None:
            random.seed(recipe.training_seed)
            torch.manual_seed(recipe.training_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(recipe.training_seed)
            generator, sampler_state = _new_sampler(
                np, seed=recipe.training_seed, fold_id=recipe.fold_id,
            )
            sampler_draw_hash = "0" * 64
        else:
            state = load_checkpoint(
                resume_checkpoint,
                expected_sha256=resume_checkpoint_sha256,
                torch=torch,
            )
            require_checkpoint_match(
                state,
                recipe_sha256=recipe_sha,
                data_identity_sha256=bundle.identity_sha256,
                model_sha256=model_sha,
                runner_sha256=runner_sha,
                runtime_sha256=runtime_sha,
            )
            if (state.get("recipe") != recipe_dict or
                    state.get("model_binding") != _model_binding_record(recipe.arm_id, spec) or
                    state.get("total_origins") != bundle.total_origins):
                raise CheckpointMismatchError("resume recipe/model/data count differs")
            if state.get("run_id") != expected_run_id:
                raise CheckpointMismatchError("resume checkpoint run ID differs from the preassigned pin")
            completed = state.get("completed_updates")
            draw_count = state.get("sampler_cursor_draws")
            if (type(completed) is not int or not 0 <= completed <= recipe.updates or
                    draw_count != completed * recipe.batch_size or
                    state.get("sampler_total_draws") != recipe.updates * recipe.batch_size):
                raise CheckpointError("resume update and sampler cursors are inconsistent")
            sampler_state = state.get("sampler_pcg64_state")
            if not isinstance(sampler_state, Mapping):
                raise CheckpointError("resume checkpoint is missing PCG64 state")
            generator = _restore_sampler(np, sampler_state)
            sampler_draw_hash = state.get("sampler_draw_hash")
            if not isinstance(sampler_draw_hash, str) or len(sampler_draw_hash) != 64:
                raise CheckpointError("resume draw hash chain is malformed")
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.to(device=device)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            _optimizer_to_device(optimizer, torch=torch, device=device)
            _restore_rng(torch, state)
            loss_sum = float(state.get("train_loss_sum_radians", 0.0))
            loss_count = int(state.get("train_loss_count", 0))
            elapsed_before = float(state.get("elapsed_cuda_wall_seconds", 0.0))
            root_controller.authorize_training(
                recipe=recipe_dict,
                recipe_sha256=recipe_sha,
                data_identity_sha256=bundle.identity_sha256,
                model_sha256=model_sha,
                runner_sha256=runner_sha,
                resume_checkpoint_sha256=resume_checkpoint_sha256,
                completed_updates=completed,
            )
            receipt = {
                "schema": "abc-resume-restored-v1", "resume_verified": True,
                "run_id": run_id, "checkpoint_sha256": resume_checkpoint_sha256,
                "plan_sha256": state["draft_sha256"], "loaded_global_step": completed,
                "optimizer_restored": True, "sampler_restored": True,
                "rng_restored": True, "model_restored": True,
            }
            receipt_path = Path(resume_receipt_path)
            ensure_private_directory(receipt_path.parent)
            with receipt_path.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            root_controller.resume_restored(receipt_path=str(receipt_path),
                                            checkpoint_sha256=resume_checkpoint_sha256,
                                            completed_updates=completed)

        fit_id = f"{recipe.purpose}/fold-{recipe.fold_id}/seed-{recipe.training_seed}/{recipe.arm_id}"
        elapsed_start = time.monotonic()
        digest: str | None = resume_checkpoint_sha256
        latest_checkpoint_path: str | None = None
        latest_metadata_path: str | None = None

        def commit_checkpoint(update_count: int, *, elapsed: float) -> None:
            nonlocal digest, latest_checkpoint_path, latest_metadata_path
            payload = _checkpoint_payload(
                torch=torch, model=model, optimizer=optimizer, recipe=recipe,
                recipe_sha=recipe_sha,
                model_binding=_model_binding_record(recipe.arm_id, spec),
                model_sha=model_sha, runner_hashes=runner_hashes,
                runner_sha=runner_sha, runtime_binding=runtime_binding,
                runtime_sha=runtime_sha, bundle=bundle,
                completed_updates=update_count, draw_count=draw_count,
                sampler_state=generator.bit_generator.state,
                sampler_draw_hash=sampler_draw_hash,
                elapsed_cuda_wall=elapsed, loss_sum=loss_sum,
                loss_count=loss_count, run_id=run_id,
            )
            checkpoint_path, metadata_path, digest_value = save_checkpoint(
                checkpoint_root, payload, torch=torch,
            )
            root_controller.checkpoint_committed(
                checkpoint_sha256=digest_value,
                completed_updates=update_count,
                data_identity_sha256=bundle.identity_sha256,
                recipe_sha256=recipe_sha,
                checkpoint_path=str(checkpoint_path),
                metadata_path=str(metadata_path),
            )
            digest = digest_value
            latest_checkpoint_path = str(checkpoint_path)
            latest_metadata_path = str(metadata_path)

        last_checkpoint_update = completed if resume_checkpoint is not None else 0
        last_checkpoint_time = elapsed_start
        for update_index in range(completed, recipe.updates):
            bundle.assert_active()
            try:
                allowed = budget_gate.before_update(remaining_updates=recipe.updates - update_index)
                if allowed is not True:
                    raise BudgetStop("external budget controller stopped before the next update")
            except Exception:
                if completed > last_checkpoint_update:
                    commit_checkpoint(completed, elapsed=elapsed_before + max(0.0, time.monotonic() - elapsed_start))
                raise
            ranks_np = generator.integers(
                0, bundle.total_origins, size=recipe.batch_size,
                endpoint=False, dtype=np.int64,
            )
            draw_count += recipe.batch_size
            sampler_draw_hash = _extend_draw_hash(sampler_draw_hash, ranks_np)
            context, target = _load_batch(
                bundle, ranks_np, numpy=np, torch=torch, device=device,
                batch_size=recipe.batch_size,
            )
            loss_sum += _optimizer_update(
                model, optimizer, loss_module.endpoint20_geodesic_loss,
                context, target, torch=torch,
            )
            loss_count += 1
            completed = update_index + 1
            sampler_state = _primitive_state(copy.deepcopy(generator.bit_generator.state))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.monotonic()
            elapsed = elapsed_before + max(0.0, now - elapsed_start)
            if _checkpoint_due(completed, last_checkpoint_update, now, last_checkpoint_time, recipe.updates):
                commit_checkpoint(completed, elapsed=elapsed)
                last_checkpoint_update = completed
                last_checkpoint_time = time.monotonic()

        bundle.close()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_total = elapsed_before + max(0.0, time.monotonic() - elapsed_start)
        return {
            "purpose": recipe.purpose,
            "role": recipe.role,
            "fold_id": recipe.fold_id,
            "run_id": run_id,
            "seed": recipe.training_seed,
            "arm": recipe.arm_id,
            "completed_updates": completed,
            "mean_train_loss_radians": loss_sum / max(loss_count, 1),
            "elapsed_cuda_wall_seconds": elapsed_total,
            "checkpoint_sha256": digest,
            "checkpoint_path": latest_checkpoint_path,
            "checkpoint_metadata_path": latest_metadata_path,
            "model_sha256": model_sha,
            "data_identity_sha256": bundle.identity_sha256,
        }
    except BaseException:
        try:
            bundle.abort_and_close()
        except Exception:
            pass
        raise


def run_fit(
    *, bundle: PreparedFitBundle, root_controller: RootTrainingCapability,
    budget_gate: BudgetGate, recipe: Any, checkpoint_directory: Path,
    resume_checkpoint: Path | None = None, resume_checkpoint_sha256: str | None = None,
    device_name: str = "cuda", require_cuda: bool = True,
    resume_receipt_path: Path | None = None, expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Public fail-closed wrapper; close the fit lease on every failure path."""
    try:
        return _run_fit_impl(
            bundle=bundle, root_controller=root_controller, budget_gate=budget_gate,
            recipe=recipe, checkpoint_directory=checkpoint_directory,
            resume_checkpoint=resume_checkpoint,
            resume_checkpoint_sha256=resume_checkpoint_sha256,
            device_name=device_name, require_cuda=require_cuda,
            resume_receipt_path=resume_receipt_path, expected_run_id=expected_run_id,
        )
    except BaseException:
        abort = getattr(bundle, "abort_and_close", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None, *, bundle: Any = None,
         trusted_verifier: Any = None, expected_run_binding: Any = None,
         root_controller: Any = None, budget_gate: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run one authorized ABC train-only fit on Vast")
    parser.add_argument("--mode", choices=("pilot", "screen"), required=True)
    parser.add_argument("--run-id", required=True, help="preassigned 32-hex run ID")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="user-selected private directory; new directories use mode 0700")
    parser.add_argument("--resume", type=Path,
                        help="content-addressed checkpoint with an out-of-band root digest")
    parser.add_argument("--resume-receipt", type=Path,
                        help="new private receipt path for the restored process")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--arm", choices=SCREEN_ARMS, default="ordered_product")
    args = parser.parse_args(argv)
    require_ram_floor()
    configure_pre_torch_environment(require_cuda=True)
    resume_sha = sha256_file(args.resume) if args.resume is not None else None
    recipe = _make_recipe(args.mode, args.fold_id, args.seed, args.arm)
    if bundle is None:
        if (trusted_verifier is None or expected_run_binding is None or
                root_controller is None or budget_gate is None):
            raise PermissionError("only the authenticated root launcher can inject verifier, run binding, controller, and budget gate")
        staging = _staging_root()
        if str(staging) not in sys.path:
            sys.path.insert(0, str(staging))
        from vast_training_v1.data.bundle import open_prepared_train_bundle
        expected_identity_sha256 = getattr(expected_run_binding, "expected_identity_sha256", None)
        if args.resume is not None and not expected_identity_sha256:
            raise PermissionError("resume identity pin must come from the authenticated root run binding")
        bundle = open_prepared_train_bundle(
            args.bundle_dir,
            trusted_verifier=trusted_verifier,
            expected_fold_id=recipe.fold_id,
            expected_run_binding=expected_run_binding,
            expected_identity_sha256=expected_identity_sha256,
        )
    report = run_fit(
        bundle=bundle,
        root_controller=root_controller,
        budget_gate=budget_gate,
        recipe=recipe,
        checkpoint_directory=args.checkpoint_dir,
        resume_checkpoint=args.resume,
        resume_checkpoint_sha256=resume_sha,
        resume_receipt_path=args.resume_receipt, expected_run_id=args.run_id,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

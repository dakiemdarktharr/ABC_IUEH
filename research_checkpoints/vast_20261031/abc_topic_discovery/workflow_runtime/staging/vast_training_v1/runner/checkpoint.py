"""Atomic, private, hash-anchored PyTorch checkpoint storage for Linux/Vast."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable


CHECKPOINT_SCHEMA = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_RE = re.compile(r"^update-[0-9]{6}-[0-9a-f]{64}\.pt$")
_META_FILE_RE = re.compile(r"^update-[0-9]{6}-[0-9a-f]{64}\.meta\.json$")


class CheckpointError(RuntimeError):
    """Base error for checkpoint integrity, compatibility, or persistence."""


class CheckpointIntegrityError(CheckpointError):
    """Checkpoint bytes do not match the root-anchored expected digest."""


class CheckpointMismatchError(CheckpointError):
    """A valid checkpoint belongs to a different recipe or data/model build."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_private_directory(path: Path) -> Path:
    """Create/check a user-owned mode-0700 checkpoint directory; reject links."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    if path.is_symlink():
        raise CheckpointError("checkpoint directory may not be a symlink")
    if not path.exists():
        path.mkdir(parents=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise CheckpointError("checkpoint path must be a real directory")
    info = path.stat(follow_symlinks=False)
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise CheckpointError("checkpoint directory must be owned by the current user")
    if os.name == "posix":
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise CheckpointError("checkpoint directory must have private 0700 permissions")
        os.chmod(path, 0o700)
    return path


def _write_atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path = Path(path)
    if path.exists() and path.is_symlink():
        raise CheckpointError(f"refusing to replace symlink: {path.name}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _cpu_tree(value: Any, torch: Any) -> Any:
    """Keep checkpoint payload compatible with weights_only=True deserialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, dict):
        return {_cpu_tree(key, torch): _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    raise TypeError(f"unsupported checkpoint payload type: {type(value).__name__}")


def save_checkpoint(
    directory: Path,
    state: dict[str, Any],
    *,
    torch: Any,
    serializer: Callable[[Any, Any], None] | None = None,
) -> tuple[Path, Path, str]:
    """Write immutable checkpoint bytes, then atomically advance latest.json.

    The caller must send the resulting digest to its root capability, which
    keeps the out-of-band resume anchor. `latest.json` is a convenience pointer,
    not an authorization source.
    """
    root = ensure_private_directory(directory)
    if type(state) is not dict or state.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("checkpoint state has an unsupported schema")
    safe_state = _cpu_tree(state, torch)
    serializer = serializer or torch.save
    previous: dict[str, str] | None = None
    pointer_path = root / "latest.json"
    if pointer_path.exists():
        old_path, old_sha = latest_checkpoint_path(root)
        old_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        previous = {
            "file": old_path.name,
            "sha256": old_sha,
            "metadata_file": old_pointer["metadata_file"],
            "metadata_sha256": old_pointer["metadata_sha256"],
        }
    fd, temp_name = tempfile.mkstemp(prefix=".checkpoint.", suffix=".tmp", dir=root)
    temp_path = Path(temp_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            serializer(safe_state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(temp_path)
        completed = state.get("completed_updates")
        if type(completed) is not int or not 1 <= completed <= 999_999:
            raise CheckpointError("checkpoint sidecars require at least one completed update")
        final_path = root / f"update-{completed:06d}-{digest}.pt"
        if final_path.exists():
            if final_path.is_symlink() or sha256_file(final_path) != digest:
                raise CheckpointIntegrityError("existing content-addressed checkpoint is corrupt")
            temp_path.unlink()
        else:
            os.replace(temp_path, final_path)
            _fsync_directory(root)
        metadata = _checkpoint_metadata(state, digest, final_path.stat().st_size)
        metadata_path = final_path.with_suffix(".meta.json")
        metadata_bytes = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        _write_atomic_bytes(metadata_path, metadata_bytes)
        metadata_digest = sha256_file(metadata_path)
        pointer = {
            "schema": CHECKPOINT_SCHEMA,
            "file": final_path.name,
            "sha256": digest,
            "metadata_file": metadata_path.name,
            "metadata_sha256": metadata_digest,
            "completed_updates": completed,
            "previous": previous,
        }
        _write_atomic_bytes(
            root / "latest.json",
            json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        _prune_old_checkpoints(root, keep_names={final_path.name, metadata_path.name} |
                               ({previous["file"], previous["metadata_file"]} if previous else set()))
        return final_path, metadata_path, digest
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def latest_checkpoint_path(directory: Path) -> tuple[Path, str]:
    """Read the convenience pointer; digest still must be root-anchored by caller."""
    root = ensure_private_directory(directory)
    pointer_path = root / "latest.json"
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise CheckpointError("latest checkpoint pointer is missing or unsafe")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("latest checkpoint pointer is malformed") from exc
    if type(pointer) is not dict or pointer.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("latest checkpoint pointer has an unsupported schema")
    name, digest = pointer.get("file"), pointer.get("sha256")
    if not isinstance(name, str) or not _FILE_RE.fullmatch(name):
        raise CheckpointError("latest checkpoint pointer contains an unsafe filename")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise CheckpointError("latest checkpoint pointer contains an invalid digest")
    checkpoint_path = root / name
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise CheckpointError("latest checkpoint file is missing or unsafe")
    metadata_name = pointer.get("metadata_file")
    metadata_digest = pointer.get("metadata_sha256")
    if (not isinstance(metadata_name, str) or Path(metadata_name).name != metadata_name or
            metadata_name != checkpoint_path.with_suffix(".meta.json").name or
            not isinstance(metadata_digest, str) or not _SHA256_RE.fullmatch(metadata_digest)):
        raise CheckpointError("latest checkpoint metadata pointer is unsafe")
    metadata_path = root / metadata_name
    if (metadata_path.is_symlink() or not metadata_path.is_file() or
            sha256_file(metadata_path) != metadata_digest):
        raise CheckpointIntegrityError("latest checkpoint metadata bytes do not match pointer")
    return checkpoint_path, digest


def _prune_old_checkpoints(root: Path, *, keep_names: set[str]) -> None:
    """Keep the current and immediately previous generated checkpoint only."""
    for item in root.iterdir():
        if item.name in keep_names or item.is_symlink() or not item.is_file():
            continue
        if _FILE_RE.fullmatch(item.name) or _META_FILE_RE.fullmatch(item.name):
            item.unlink()


def load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    torch: Any,
) -> dict[str, Any]:
    """Verify bytes before deserialization; refuse unanchored/untrusted loads."""
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise CheckpointIntegrityError("an out-of-band root checkpoint digest is required")
    checkpoint_path = Path(path)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise CheckpointError("checkpoint must be a regular non-symlink file")
    if sha256_file(checkpoint_path) != expected_sha256:
        raise CheckpointIntegrityError("checkpoint bytes do not match the root-anchored digest")
    metadata_path = checkpoint_path.with_suffix(".meta.json")
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise CheckpointIntegrityError("checkpoint metadata sidecar is missing or unsafe")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError("checkpoint metadata sidecar is malformed") from exc
    _validate_checkpoint_metadata(metadata, expected_sha256, checkpoint_path.stat().st_size)
    try:
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise CheckpointError("installed PyTorch must support weights_only=True") from exc
    if type(value) is not dict or value.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointError("checkpoint payload has an unsupported schema")
    recipe = value.get("recipe")
    if type(recipe) is not dict:
        raise CheckpointIntegrityError("checkpoint payload recipe is malformed")
    payload_pairs = {
        "run_id": value.get("run_id"),
        "plan_sha256": value.get("draft_sha256"),
        "data_sha256": value.get("data_identity_sha256"),
        "code_sha256": value.get("runner_sha256"),
        "model_sha256": value.get("model_sha256"),
        "fold_id": recipe.get("fold_id"),
        "training_seed": recipe.get("training_seed"),
        "arm": recipe.get("arm_id"),
        "global_step": value.get("completed_updates"),
        "total_steps": recipe.get("updates"),
        "recipe_sha256": value.get("recipe_sha256"),
        "runtime_sha256": value.get("runtime_sha256"),
        "sampler_cursor_draws": value.get("sampler_cursor_draws"),
        "sampler_draw_hash": value.get("sampler_draw_hash"),
    }
    if any(metadata.get(key) != item for key, item in payload_pairs.items()):
        raise CheckpointIntegrityError("checkpoint sidecar does not match serialized state")
    return value


def _checkpoint_metadata(state: dict[str, Any], digest: str, size: int) -> dict[str, Any]:
    required = (
        "run_id", "draft_sha256", "data_identity_sha256", "runner_sha256",
        "model_sha256", "completed_updates", "recipe", "recipe_sha256",
        "sampler_cursor_draws", "sampler_draw_hash", "runtime_sha256",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise CheckpointError("checkpoint state is missing metadata: " + ", ".join(missing))
    recipe = state["recipe"]
    if type(recipe) is not dict:
        raise CheckpointError("checkpoint recipe must be a mapping")
    if not isinstance(state["run_id"], str) or not state["run_id"].strip():
        raise CheckpointError("checkpoint run id is invalid")
    state_components = {
        "model": isinstance(state.get("model_state_dict"), dict),
        "optimizer": isinstance(state.get("optimizer_state_dict"), dict),
        "sampler": isinstance(state.get("sampler_pcg64_state"), dict),
        "python_rng": "python_random_state" in state,
        "numpy_rng": isinstance(state.get("sampler_pcg64_state"), dict),
        "torch_cpu_rng": state.get("torch_cpu_rng_state") is not None,
        "torch_cuda_rng": isinstance(state.get("torch_cuda_rng_state_all"), list),
    }
    if not all(state_components.values()):
        raise CheckpointError("full model/optimizer/sampler/RNG state is incomplete")
    metadata = {
        "schema": "abc-training-checkpoint-v1",
        "run_id": state["run_id"],
        "plan_sha256": state["draft_sha256"],
        "data_sha256": state["data_identity_sha256"],
        "code_sha256": state["runner_sha256"],
        "model_sha256": state["model_sha256"],
        "fold_id": recipe["fold_id"],
        "training_seed": recipe["training_seed"],
        "arm": recipe["arm_id"],
        "global_step": state["completed_updates"],
        "total_steps": recipe["updates"],
        "resume_state_complete": True,
        "state_components": state_components,
        "payload_sha256": digest,
        "payload_size_bytes": size,
        "recipe_sha256": state["recipe_sha256"],
        "runtime_sha256": state["runtime_sha256"],
        "sampler_cursor_draws": state["sampler_cursor_draws"],
        "sampler_draw_hash": state["sampler_draw_hash"],
        "elapsed_cuda_wall_seconds": state.get("elapsed_cuda_wall_seconds", 0.0),
    }
    return metadata


def _validate_checkpoint_metadata(metadata: Any, digest: str, size: int) -> None:
    if type(metadata) is not dict:
        raise CheckpointIntegrityError("checkpoint metadata sidecar root must be an object")
    required = {
        "schema": "abc-training-checkpoint-v1",
        "resume_state_complete": True,
        "payload_sha256": digest,
        "payload_size_bytes": size,
    }
    for key, value in required.items():
        if metadata.get(key) != value:
            raise CheckpointIntegrityError(f"checkpoint metadata {key} does not match payload")
    for key in ("plan_sha256", "data_sha256", "code_sha256", "model_sha256", "recipe_sha256", "runtime_sha256", "sampler_draw_hash"):
        value = metadata.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise CheckpointIntegrityError(f"checkpoint metadata {key} is invalid")
    components = metadata.get("state_components")
    if type(components) is not dict or any(components.get(key) is not True for key in (
        "model", "optimizer", "sampler", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng",
    )):
        raise CheckpointIntegrityError("checkpoint metadata does not declare complete resume state")
    if type(metadata.get("global_step")) is not int or metadata["global_step"] < 1:
        raise CheckpointIntegrityError("checkpoint metadata global_step must be positive")
    if (type(metadata.get("total_steps")) is not int or
            metadata["total_steps"] < metadata["global_step"]):
        raise CheckpointIntegrityError("checkpoint metadata total_steps is inconsistent")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"].strip():
        raise CheckpointIntegrityError("checkpoint metadata run_id is invalid")
    if (isinstance(metadata.get("fold_id"), bool) or type(metadata.get("fold_id")) is not int or
            metadata["fold_id"] < 0):
        raise CheckpointIntegrityError("checkpoint metadata fold_id is invalid")
    if (isinstance(metadata.get("training_seed"), bool) or
            type(metadata.get("training_seed")) is not int):
        raise CheckpointIntegrityError("checkpoint metadata training_seed is invalid")
    if not isinstance(metadata.get("arm"), str) or not metadata["arm"].strip():
        raise CheckpointIntegrityError("checkpoint metadata arm is invalid")


def require_checkpoint_match(
    state: dict[str, Any],
    *,
    recipe_sha256: str,
    data_identity_sha256: str,
    model_sha256: str,
    runner_sha256: str,
    runtime_sha256: str,
) -> None:
    """Reject resume unless every fixed training binding matches exactly."""
    checks = {
        "recipe_sha256": recipe_sha256,
        "data_identity_sha256": data_identity_sha256,
        "model_sha256": model_sha256,
        "runner_sha256": runner_sha256,
        "runtime_sha256": runtime_sha256,
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise CheckpointMismatchError(f"resume rejected: {field} differs")


def optimizer_state_to_device(value: Any, *, torch: Any, device: Any) -> Any:
    """Move tensor leaves in an optimizer checkpoint to the selected device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: optimizer_state_to_device(item, torch=torch, device=device)
                for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(optimizer_state_to_device(item, torch=torch, device=device)
                     for item in value)
    if isinstance(value, list):
        return [optimizer_state_to_device(item, torch=torch, device=device)
                for item in value]
    return value

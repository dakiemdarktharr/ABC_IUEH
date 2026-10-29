"""Runtime contract between the trainer and a separately reviewed fit loader.

This module deliberately contains no data reader. A conforming loader must
authenticate its root-issued, in-process capability before opening data. JSON
manifests can describe an identity, but cannot establish one on their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class PilotRecipe:
    purpose: str = "ordered_motion_exploratory_fit_only_micro_pilot"
    role: str = "outer_fit10"
    fold_id: int = 0
    training_seed: int = 20261026
    arm_id: str = "ordered_product"
    batch_size: int = 128
    updates: int = 100
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    loss: str = "mean_endpoint20_geodesic_radians"

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "role": self.role,
            "fold_id": self.fold_id,
            "training_seed": self.training_seed,
            "arm_id": self.arm_id,
            "batch_size": self.batch_size,
            "updates": self.updates,
            "optimizer": {
                "name": "Adam",
                "learning_rate": self.learning_rate,
                "betas": [self.adam_beta1, self.adam_beta2],
                "epsilon": self.adam_epsilon,
                "weight_decay": self.weight_decay,
            },
            "loss": self.loss,
            "input_shape": [self.batch_size, 400, 4],
            "target_shape": [self.batch_size, 20, 4],
            "data_scope": "train-only outer_fit10; no dev/final",
        }


@dataclass(frozen=True)
class ScreenRecipe:
    purpose: str = "ordered_motion_three_arm_screen"
    role: str = "outer_fit10"
    fold_id: int = 0
    training_seed: int = 20261026
    arm_id: str = "ordered_product"
    batch_size: int = 128
    updates: int = 5000
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    loss: str = "mean_endpoint20_geodesic_radians"

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "role": self.role,
            "fold_id": self.fold_id,
            "training_seed": self.training_seed,
            "arm_id": self.arm_id,
            "batch_size": self.batch_size,
            "updates": self.updates,
            "optimizer": {
                "name": "Adam",
                "learning_rate": self.learning_rate,
                "betas": [self.adam_beta1, self.adam_beta2],
                "epsilon": self.adam_epsilon,
                "weight_decay": self.weight_decay,
            },
            "loss": self.loss,
            "input_shape": [self.batch_size, 400, 4],
            "target_shape": [self.batch_size, 20, 4],
            "data_scope": "train-only outer_fit10; no dev/final",
        }


@runtime_checkable
class RootTrainingCapability(Protocol):
    """Object supplied by the separately authenticated root bootstrap.

    Implementations must validate the active purpose/session/lease and bind
    recipe, model/runner hashes, data identity, resume cursor, and any prior
    checkpoint digest. Implementations must not derive authority from a
    user-editable JSON field. This interface itself is not authentication.
    """

    @property
    def capability_id(self) -> str: ...

    def authorize_training(
        self,
        *,
        recipe: Mapping[str, Any],
        recipe_sha256: str,
        data_identity_sha256: str,
        model_sha256: str,
        runner_sha256: str,
        resume_checkpoint_sha256: str | None,
        completed_updates: int | None,
    ) -> None: ...

    def checkpoint_committed(
        self, *, checkpoint_sha256: str, completed_updates: int,
        data_identity_sha256: str, recipe_sha256: str,
        checkpoint_path: str, metadata_path: str,
    ) -> None:
        """Synchronously verify and back up both files before allowing the next update."""
        ...

    def resume_restored(self, *, receipt_path: str, checkpoint_sha256: str,
                        completed_updates: int) -> None:
        """Verify the restored receipt against independently pinned backup metadata."""
        ...



@runtime_checkable
class BudgetGate(Protocol):
    """Root-injected live estimate gate; it is not provider billing enforcement."""

    @property
    def budget_policy(self) -> Mapping[str, float]: ...

    def before_update(self, *, remaining_updates: int) -> bool:
        """Return true to continue or raise a budget-stop exception."""
        ...


@runtime_checkable
class PreparedFitBundle(Protocol):
    """Indexed same-segment fit-only dataset returned by an authenticated loader.

    `binding` is an opaque data-access lease from the trusted bundle verifier.
    Training authorization and budget control are separate root-injected
    capabilities. Global ranks are stable for this exact fold bundle.
    """

    @property
    def total_origins(self) -> int: ...

    @property
    def identity_sha256(self) -> str: ...

    @property
    def binding(self) -> object: ...

    def assert_active(self) -> None: ...

    def origin_for_rank(self, rank: int) -> tuple[int, int]:
        """Map one global rank to (segment_index, local_origin)."""
        ...

    def sample_window(self, segment_index: int, origin_index: int) -> tuple[bytes, bytes]:
        """Return packed little-endian FP32 [400,4] and [20,4] arrays."""
        ...

    def close(self) -> None: ...

    def abort_and_close(self) -> None: ...


def validate_bundle_shape(bundle: Any) -> PreparedFitBundle:
    """Fail closed for an object that does not implement the reviewed contract."""
    required = (
        "total_origins", "identity_sha256", "binding",
        "assert_active", "origin_for_rank", "sample_window", "close", "abort_and_close",
    )
    missing = [name for name in required if not hasattr(bundle, name)]
    if missing:
        raise TypeError(f"fit loader result is missing fields: {', '.join(missing)}")
    if type(bundle.total_origins) is not int or bundle.total_origins <= 0:
        raise ValueError("fit loader must report a positive integer origin count")
    if not isinstance(bundle.identity_sha256, str) or len(bundle.identity_sha256) != 64:
        raise ValueError("fit loader must supply a SHA-256 identity binding")
    if isinstance(bundle.binding, (Mapping, list, tuple, set, frozenset, str, bytes, int, float, bool)):
        raise PermissionError("fit loader binding must be a root-issued opaque runtime lease")
    if not callable(getattr(bundle.binding, "assert_live", None)):
        raise PermissionError("fit loader binding must expose assert_live()")
    return bundle

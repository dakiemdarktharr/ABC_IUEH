"""Generated-only CPU tests for checkpoint/resume; intentionally not run here."""
from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import checkpoint
import trainer


TEST_RECIPE = {
    "purpose": "generated_test_only",
    "role": "synthetic",
    "fold_id": 0,
    "training_seed": 7301,
    "arm_id": "toy",
    "batch_size": 4,
    "updates": 8,
}
HASHES = {
    "recipe_sha256": hashlib.sha256(b"recipe").hexdigest(),
    "data_identity_sha256": hashlib.sha256(b"generated-data").hexdigest(),
    "model_sha256": hashlib.sha256(b"toy-model").hexdigest(),
    "runner_sha256": hashlib.sha256(b"toy-runner").hexdigest(),
    "runtime_sha256": hashlib.sha256(b"cpu-test-runtime").hexdigest(),
}
PLAN_SHA256 = hashlib.sha256(b"synthetic-plan").hexdigest()
ZERO_DRAW_HASH = "0" * 64


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).unsqueeze(1).expand(-1, 20, -1)


def _new_process(seed: int) -> tuple[ToyModel, torch.optim.Optimizer, np.random.Generator]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = ToyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    sampler = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    return model, optimizer, sampler


def _batch_for_ranks(ranks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(ranks.astype(np.float32) / 257.0).unsqueeze(1)
    x = torch.cat((values, values * 0.3, values * -0.2, torch.ones_like(values)), dim=1)
    offsets = torch.linspace(-0.15, 0.15, 20, dtype=torch.float32).view(1, 20, 1)
    target = x[:, None, :].expand(-1, 20, -1) * 0.7 + offsets
    return x, target


def _one_update(model: ToyModel, optimizer: torch.optim.Optimizer,
                ranks: np.ndarray) -> float:
    x, target = _batch_for_ranks(ranks)
    return trainer._optimizer_update(
        model, optimizer,
        lambda prediction, truth: ((prediction - truth) ** 2).mean(),
        x, target, torch=torch,
    )


def _state(
    model: ToyModel,
    optimizer: torch.optim.Optimizer,
    sampler: np.random.Generator,
    *,
    completed: int,
    draw_count: int,
    draw_hash: str,
    loss_sum: float,
) -> dict:
    sampler_state = trainer._primitive_state(copy.deepcopy(sampler.bit_generator.state))
    return {
        "schema": checkpoint.CHECKPOINT_SCHEMA,
        "run_id": "synthetic-run-0000000000000001",
        "draft_sha256": PLAN_SHA256,
        "recipe": TEST_RECIPE,
        "recipe_sha256": HASHES["recipe_sha256"],
        "model_binding": {"arm_id": "toy"},
        "model_sha256": HASHES["model_sha256"],
        "runner_code_hashes": {"test": HASHES["runner_sha256"]},
        "runner_sha256": HASHES["runner_sha256"],
        "runtime_binding": {"platform": "cpu-generated-test"},
        "runtime_sha256": HASHES["runtime_sha256"],
        "data_identity_sha256": HASHES["data_identity_sha256"],
        "total_origins": 97,
        "completed_updates": completed,
        "sampler_cursor_draws": draw_count,
        "sampler_total_draws": TEST_RECIPE["updates"] * TEST_RECIPE["batch_size"],
        "sampler_pcg64_state": sampler_state,
        "sampler_draw_hash": draw_hash,
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_state_all": [],
        "model_state_dict": {name: value.detach().cpu().clone()
                             for name, value in model.state_dict().items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "elapsed_cuda_wall_seconds": 0.25 * completed,
        "train_loss_sum_radians": loss_sum,
        "train_loss_count": completed,
    }


def _run_segment(model, optimizer, sampler, start, stop, ranks_seen, draw_hash, loss_sum):
    for update in range(start, stop):
        ranks = sampler.integers(0, 97, size=TEST_RECIPE["batch_size"], dtype=np.int64)
        ranks_seen.extend(int(value) for value in ranks)
        draw_hash = trainer._extend_draw_hash(draw_hash, ranks)
        # Advance both process RNGs so the restore assertion covers real state.
        random.random()
        torch.rand(1)
        loss_sum += _one_update(model, optimizer, ranks)
    return draw_hash, loss_sum


def _assert_tree_equal(test: unittest.TestCase, left, right) -> None:
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class CheckpointResumeTests(unittest.TestCase):
    def test_uninterrupted_equals_interrupted_and_resumed_exactly(self) -> None:
        uninterrupted_model, uninterrupted_optimizer, uninterrupted_rng = _new_process(7301)
        uninterrupted_ranks: list[int] = []
        uninterrupted_hash, uninterrupted_loss = _run_segment(
            uninterrupted_model, uninterrupted_optimizer, uninterrupted_rng,
            0, 8, uninterrupted_ranks, ZERO_DRAW_HASH, 0.0,
        )
        uninterrupted_python_state = random.getstate()
        uninterrupted_torch_state = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "private"
            split_model, split_optimizer, split_rng = _new_process(7301)
            split_ranks: list[int] = []
            split_hash, split_loss = _run_segment(
                split_model, split_optimizer, split_rng, 0, 3,
                split_ranks, ZERO_DRAW_HASH, 0.0,
            )
            state = _state(
                split_model, split_optimizer, split_rng, completed=3,
                draw_count=12, draw_hash=split_hash, loss_sum=split_loss,
            )
            payload_path, _metadata_path, payload_sha = checkpoint.save_checkpoint(
                checkpoint_dir, state, torch=torch,
            )

            # Simulate process death and a freshly initialized process. The
            # checkpoint restores global RNG and PCG64 state without a reseed.
            random.seed(9907)
            torch.manual_seed(9907)
            resumed_model = ToyModel()
            resumed_optimizer = torch.optim.Adam(
                resumed_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
            )
            resumed_state = checkpoint.load_checkpoint(
                payload_path, expected_sha256=payload_sha, torch=torch,
            )
            checkpoint.require_checkpoint_match(
                resumed_state,
                recipe_sha256=HASHES["recipe_sha256"],
                data_identity_sha256=HASHES["data_identity_sha256"],
                model_sha256=HASHES["model_sha256"],
                runner_sha256=HASHES["runner_sha256"],
                runtime_sha256=HASHES["runtime_sha256"],
            )
            resumed_model.load_state_dict(resumed_state["model_state_dict"])
            resumed_optimizer.load_state_dict(resumed_state["optimizer_state_dict"])
            resumed_rng = trainer._restore_sampler(np, resumed_state["sampler_pcg64_state"])
            with patch.object(torch.cuda, "device_count", return_value=0):
                trainer._restore_rng(torch, resumed_state)
            resumed_ranks = list(split_ranks)
            resumed_hash = split_hash
            resumed_loss = split_loss
            resumed_hash, resumed_loss = _run_segment(
                resumed_model, resumed_optimizer, resumed_rng, 3, 8,
                resumed_ranks, resumed_hash, resumed_loss,
            )

        self.assertEqual(uninterrupted_ranks, resumed_ranks)
        self.assertEqual(uninterrupted_hash, resumed_hash)
        self.assertEqual(uninterrupted_loss, resumed_loss)
        _assert_tree_equal(self, uninterrupted_model.state_dict(), resumed_model.state_dict())
        _assert_tree_equal(self, uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict())
        self.assertEqual(uninterrupted_rng.bit_generator.state, resumed_rng.bit_generator.state)
        self.assertEqual(uninterrupted_python_state, random.getstate())
        self.assertTrue(torch.equal(uninterrupted_torch_state, torch.get_rng_state()))

    def test_failed_write_keeps_previous_pointer_and_checkpoint(self) -> None:
        model, optimizer, sampler = _new_process(7301)
        ranks = sampler.integers(0, 97, size=4, dtype=np.int64)
        _one_update(model, optimizer, ranks)
        draw_hash = trainer._extend_draw_hash(ZERO_DRAW_HASH, ranks)
        state = _state(model, optimizer, sampler, completed=1, draw_count=4,
                       draw_hash=draw_hash, loss_sum=0.5)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "private"
            first_path, _first_meta, first_sha = checkpoint.save_checkpoint(directory, state, torch=torch)
            pointer_before = (directory / "latest.json").read_bytes()

            def fail_serializer(_state, _stream):
                raise OSError("injected generated-only serializer failure")

            with self.assertRaises(OSError):
                checkpoint.save_checkpoint(directory, state, torch=torch, serializer=fail_serializer)
            self.assertEqual((directory / "latest.json").read_bytes(), pointer_before)
            self.assertEqual(checkpoint.sha256_file(first_path), first_sha)
            latest, latest_sha = checkpoint.latest_checkpoint_path(directory)
            self.assertEqual((latest, latest_sha), (first_path, first_sha))

    def test_tampering_and_recipe_or_dataset_mismatch_are_rejected(self) -> None:
        model, optimizer, sampler = _new_process(7301)
        ranks = sampler.integers(0, 97, size=4, dtype=np.int64)
        _one_update(model, optimizer, ranks)
        state = _state(
            model, optimizer, sampler, completed=1, draw_count=4,
            draw_hash=trainer._extend_draw_hash(ZERO_DRAW_HASH, ranks), loss_sum=0.5,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _meta, digest = checkpoint.save_checkpoint(
                Path(temp_dir) / "private", state, torch=torch,
            )
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaises(checkpoint.CheckpointIntegrityError):
                checkpoint.load_checkpoint(path, expected_sha256=digest, torch=torch)

        for field, changed in (
            ("recipe_sha256", hashlib.sha256(b"changed-recipe").hexdigest()),
            ("data_identity_sha256", hashlib.sha256(b"changed-data").hexdigest()),
        ):
            values = dict(HASHES)
            values[field] = changed
            with self.assertRaises(checkpoint.CheckpointMismatchError):
                checkpoint.require_checkpoint_match(
                    state,
                    recipe_sha256=values["recipe_sha256"],
                    data_identity_sha256=values["data_identity_sha256"],
                    model_sha256=values["model_sha256"],
                    runner_sha256=values["runner_sha256"],
                    runtime_sha256=values["runtime_sha256"],
                )


if __name__ == "__main__":
    unittest.main()

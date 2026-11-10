"""Locked five-model BOXRR experiment with train-only model selection.

The outer 72-user partition is opened only after all five checkpoint locks exist.
This entry point refuses synthetic corpus manifests; tests exercise its components
with generated arrays without changing their provenance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .boxrr_pilot import verify
from .boxrr_source import file_sha256
from .boxrr_train import model_module
from . import boxrr_models as models


SEED = 20261106
PRIMARY_SEED = 20261107
LEGACY_SEED = 20261027
NAMES = ("ordered", "residual_mlp", "tcn", "energy", "jepa")
BASELINES = ("zero_velocity", "cav1", "cav5", "cav20")
SPEED_EDGES = (0.0, 15.0, 45.0, 90.0, float("inf"))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".part")
    torch.save(value, temp)
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_status(output, *, state="running", stage="training", current_arm=None, step=0,
                 completed_arms=None, error=None):
    output = Path(output)
    if completed_arms is None:
        completed_arms = [name for name in NAMES if (output / name / "lock.json").exists()]
    status = {"state": state, "stage": stage, "current_arm": current_arm, "step": step,
              "completed_arms": completed_arms, "unix_time": time.time()}
    if error is not None:
        status["error"] = error
    atomic_json(output / "status.json", status)


def require_official(manifest):
    if manifest.get("source") != "BOXRR-23" or manifest.get("training_allowed") is not True:
        raise ValueError("Synthetic or non-BOXRR data cannot run the actual experiment")
    counts = {p: sum(r["partition"] == p for r in manifest["shards"])
              for p in ("train", "validation")}
    if counts != {"train": 287, "validation": 72}:
        raise ValueError("This experiment requires the frozen 287/72 user cohort")


def split_users(manifest, *, seed=SEED, inner_count=29):
    """Keep the original outer assignment; split only original training users."""
    train = [r["user"] for r in manifest["shards"] if r["partition"] == "train"]
    outer = [r["user"] for r in manifest["shards"] if r["partition"] == "validation"]
    if len(set(train + outer)) != len(train) + len(outer):
        raise ValueError("Duplicate user or partition leakage")
    if not 0 < inner_count < len(train):
        raise ValueError("Inner development must leave fitting users")
    ordered = sorted(train, key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest())
    inner = set(ordered[:inner_count])
    return {"fit": [u for u in train if u not in inner],
            "inner": [u for u in train if u in inner], "outer": outer}


@dataclass
class MemoryUser:
    user: str
    segments: list[np.ndarray]
    cumulative: np.ndarray


class RamCorpus:
    """Immutable full segments; each random draw is uniform among valid starts."""
    def __init__(self, users):
        self.users = list(users)
        if not self.users or any(len(u.cumulative) == 0 for u in self.users):
            raise ValueError("Users must have eligible windows")
        if len({u.user for u in self.users}) != len(self.users):
            raise ValueError("Duplicate RAM user")

    @classmethod
    def load(cls, root, manifest, allowed_users):
        allowed = set(allowed_users)
        loaded = []
        for row in manifest["shards"]:
            if row["user"] not in allowed:
                continue
            segments, lengths = [], []
            with h5py.File(Path(root) / row["path"], "r") as store:
                # Match the historical Batches HDF traversal for legacy replay.
                for session in store["sessions"].values():
                    for segment in session["segments"].values():
                        q = segment["quaternion"]
                        if len(q) >= 420:
                            array = np.asarray(q[:], dtype=np.float32)
                            if array.ndim != 2 or array.shape[1] != 4:
                                raise ValueError("Invalid quaternion shape")
                            array.setflags(write=False)
                            segments.append(array)
                            lengths.append(len(array) - 419)
            loaded.append(MemoryUser(row["user"], segments, np.cumsum(lengths, dtype=np.int64)))
        if {u.user for u in loaded} != allowed:
            raise ValueError("Not all requested users were found")
        return cls(loaded)

    def subset(self, allowed_users):
        allowed = set(allowed_users)
        result = [u for u in self.users if u.user in allowed]
        if {u.user for u in result} != allowed:
            raise ValueError("Requested RAM subset missing users")
        return RamCorpus(result)

    @property
    def bytes(self):
        return sum(s.nbytes for u in self.users for s in u.segments)

    def batch(self, rng, count, user_index=None, *, return_draws=False):
        if count < 1:
            raise ValueError("Batch must be nonempty")
        data = np.empty((count, 420, 4), dtype=np.float32)
        draws = np.empty((count, 3), dtype=np.int64)
        for j in range(count):
            index = int(rng.integers(len(self.users))) if user_index is None else int(user_index)
            user = self.users[index]
            draw = int(rng.integers(int(user.cumulative[-1])))
            segment = int(np.searchsorted(user.cumulative, draw, side="right"))
            start = draw - (int(user.cumulative[segment - 1]) if segment else 0)
            data[j] = user.segments[segment][start:start + 420]
            draws[j] = (index, segment, start)
        result = (torch.from_numpy(data[:, :400]), torch.from_numpy(data[:, 400:]))
        return (*result, draws) if return_draws else result


def bootstrap_mean(values, *, seed=SEED + 80, samples=5000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap needs finite per-user values")
    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(resampled, [0.025, 0.975])]


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def energy_proposal(model, x):
    z, last, velocities = model.encode(x)
    return model.decoder(z, last, velocities)


def inner_score(model, corpus, device, *, windows=256, batch_size=128, seed=SEED + 10,
                return_details=False):
    rng, per_user, proposal_users = np.random.default_rng(seed), [], []
    model.eval()
    with torch.inference_mode():
        for index in range(len(corpus.users)):
            total, proposal_total = 0.0, 0.0
            for start in range(0, windows, batch_size):
                count = min(batch_size, windows - start)
                x, y = corpus.batch(rng, count, index)
                x = x.to(device)
                prediction = model(x)
                error = models.geodesic_radians(prediction[:, -1], y[:, -1].to(device))
                total += float(error.double().sum())
                if getattr(model, "arm", None) == "energy":
                    proposal_error = models.geodesic_radians(energy_proposal(model, x)[:, -1], y[:, -1].to(device))
                    proposal_total += float(proposal_error.double().sum())
            per_user.append(total / windows * 180.0 / math.pi)
            proposal_users.append(proposal_total / windows * 180.0 / math.pi)
    score = float(np.mean(per_user))
    details = {"mean_endpoint20_deg": score}
    if getattr(model, "arm", None) == "energy":
        details["energy_proposal_endpoint20_deg"] = float(np.mean(proposal_users))
    return details if return_details else score


def _source_hashes(baseline_source):
    return {"experiment_sha256": file_sha256(Path(__file__)),
            "models_sha256": file_sha256(Path(models.__file__)),
            "baseline_sha256": file_sha256(baseline_source),
            "loader_sha256": file_sha256(Path(__file__).with_name("boxrr_train.py"))}


def create_protocol(root, manifest, baseline_source, *, steps, batch_size,
                    max_model_seconds, eval_every):
    if steps < 1 or batch_size < 1 or max_model_seconds <= 0 or eval_every < 1:
        raise ValueError("Invalid experiment bounds")
    return {"schema": "boxrr-five-model-v1", "source": "BOXRR-23",
            "manifest_sha256": file_sha256(Path(root) / "manifest.json"),
            "sources": _source_hashes(baseline_source), "models": list(NAMES),
            "split": split_users(manifest), "seed": SEED, "steps": steps,
            "batch_size": batch_size, "max_model_seconds": max_model_seconds,
            "eval_every": eval_every, "inner_windows_per_user": 256,
            "learning_rate": 0.001, "minimum_learning_rate": 0.00005,
            "optimizer": "AdamW", "weight_decay": 0.01, "gradient_clip": 1.0,
            "sampling": "live_uniform_user_then_uniform_eligible_window_with_replacement",
            "checkpoint_selection": "minimum_inner_endpoint20_mean_geodesic_degrees",
            "outer_primary": {"seed": PRIMARY_SEED, "windows_per_user": 1024},
            "outer_legacy": {"seed": LEGACY_SEED, "windows_per_user": 256},
            "metric": "endpoint20_mean_geodesic_degrees_equal_user_weight",
            "bootstrap": {"unit": "user", "draws": 5000, "seed": SEED + 80},
            "horizon_samples": 20, "sample_rate_hz": 100,
            "horizon_ms_official": "unresolved_20_samples_100Hz_100ms_conflict",
            "outer_status": "previously_seen_pilot_validation_not_final_test",
            "model_objectives": {"ordered": "endpoint_geodesic",
                "new_backbones": "endpoint_geodesic_plus_0.2_trajectory_geodesic",
                "energy": "supervised_plus_0.03_contrastive_ranking",
                "jepa": "joint_supervised_EMA_target_latent_L1_0.01_variance_0.01_covariance_0.001"}}


def _binding(protocol_sha, name):
    return {"protocol_sha256": protocol_sha, "model": name, "seed": SEED}


def parameter_counts(model):
    training_only = set()
    if getattr(model, "arm", None) == "jepa":
        training_only.update(id(p) for p in model.target_encoder.parameters())
        training_only.update(id(p) for p in model.predictor.parameters())
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "all_parameters_including_targets": sum(p.numel() for p in model.parameters()),
            "inference_parameters": sum(p.numel() for p in model.parameters() if id(p) not in training_only)}


def _load_checkpoint(path, binding, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state["binding"] != binding:
        raise ValueError("Checkpoint binding differs")
    return state


def train_one(name, baseline, fit, inner, protocol, output, device, *, resume=False):
    directory = output / name
    directory.mkdir(exist_ok=True)
    protocol_sha = file_sha256(output / "protocol.json")
    binding = _binding(protocol_sha, name)
    if (directory / "lock.json").exists():
        lock = read_json(directory / "lock.json")
        if (lock["binding"] != binding or
                file_sha256(directory / "best.pt") != lock["checkpoint_sha256"]):
            raise ValueError("Completed arm lock differs")
        if not resume:
            raise FileExistsError("Completed arm exists; use --resume")
        return lock
    torch.manual_seed(SEED)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    model = models.build_model(name, baseline_module=baseline, seed=SEED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=protocol["weight_decay"])
    rng = np.random.default_rng(SEED)
    step, elapsed, best, best_step = 0, 0.0, float("inf"), -1
    history = {"train": [], "inner": []}
    if (directory / "last.pt").exists():
        if not resume:
            raise FileExistsError("Existing partial arm; use --resume")
        state = _load_checkpoint(directory / "last.pt", binding, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["sampler_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if str(device).startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        step, elapsed = state["step"], state["elapsed_seconds"]
        best, best_step = state["best_inner_deg"], state["best_step"]
        history = state["history"]
    started = time.monotonic()
    write_status(output, current_arm=name, step=step)

    def state_dict():
        _sync(device)
        return {"binding": binding, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "sampler_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
                "step": step, "elapsed_seconds": elapsed + time.monotonic() - started,
                "best_inner_deg": best, "best_step": best_step, "history": history}

    def validate_and_save():
        nonlocal best, best_step
        scores = inner_score(model, inner, device, windows=protocol["inner_windows_per_user"],
                             batch_size=protocol["batch_size"], return_details=True)
        score = scores["mean_endpoint20_deg"]
        if not math.isfinite(score):
            raise ValueError("Nonfinite inner validation")
        history["inner"].append({"step": step, **scores,
                                  "elapsed_seconds": elapsed + time.monotonic() - started})
        if score < best:
            best, best_step = score, step
            atomic_torch(directory / "best.pt", state_dict())
        atomic_torch(directory / "last.pt", state_dict())
        atomic_json(directory / "history.json", history)
        write_status(output, current_arm=name, step=step)
        print(json.dumps({"event": "inner", "model": name, "step": step,
                          "inner_deg": score, "best_deg": best}), flush=True)

    if best_step < 0:
        validate_and_save()
    loss_sum, endpoint_sum, accumulated = 0.0, 0.0, 0
    while step < protocol["steps"] and elapsed + time.monotonic() - started < protocol["max_model_seconds"]:
        model.train()
        x, y = fit.batch(rng, protocol["batch_size"])
        x, y = x.to(device), y.to(device)
        progress = step / max(1, protocol["steps"] - 1)
        lr = protocol["minimum_learning_rate"] + 0.5 * (
            protocol["learning_rate"] - protocol["minimum_learning_rate"]) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = models.training_loss(model, x, y, step, protocol["steps"])
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss: {name} step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"],
                                             error_if_nonfinite=True)
        optimizer.step()
        models.after_optimizer_step(model)
        step += 1
        loss_sum += float(loss.detach())
        endpoint = diagnostics.get("endpoint_radians", diagnostics.get("endpoint_rad", float("nan")))
        endpoint_sum += float(endpoint.detach()) if torch.is_tensor(endpoint) else float(endpoint)
        accumulated += 1
        if step % 100 == 0:
            record = {"step": step, "loss": loss_sum / accumulated,
                      "learning_rate": lr, "gradient_norm": float(norm),
                      "elapsed_seconds": elapsed + time.monotonic() - started,
                      "averaged_minibatches": accumulated}
            if math.isfinite(endpoint_sum):
                record["endpoint20_deg"] = endpoint_sum / accumulated * 180 / math.pi
            for key, value in diagnostics.items():
                numeric = float(value.detach()) if torch.is_tensor(value) else float(value)
                if math.isfinite(numeric):
                    record["last_batch_" + key] = numeric
            history["train"].append(record)
            write_status(output, current_arm=name, step=step)
            loss_sum, endpoint_sum, accumulated = 0.0, 0.0, 0
            print(json.dumps({"event": "train", "model": name, **record}), flush=True)
        if step % protocol["eval_every"] == 0:
            validate_and_save()
    if history["inner"][-1]["step"] != step:
        validate_and_save()
    elapsed_total = elapsed + time.monotonic() - started
    lock = {"binding": binding, "checkpoint_sha256": file_sha256(directory / "best.pt"),
            "last_checkpoint_sha256": file_sha256(directory / "last.pt"),
            "selected_step": best_step, "completed_steps": step, "best_inner_deg": best,
            "elapsed_seconds": elapsed_total, "termination": "steps" if step >= protocol["steps"] else "time_cap",
            **parameter_counts(model),
            "outer_evaluated": False}
    atomic_json(directory / "lock.json", lock)
    write_status(output, current_arm=name, step=step)
    del model, optimizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return lock


def verify_all_locks(output, protocol=None):
    output = Path(output)
    protocol = protocol or read_json(output / "protocol.json")
    receipt = read_json(output / "all_models_locked.json")
    protocol_sha = file_sha256(output / "protocol.json")
    if receipt.get("protocol_sha256") != protocol_sha or set(receipt.get("models", {})) != set(NAMES):
        raise ValueError("All-model lock is incomplete or changed")
    for name in NAMES:
        lock = read_json(output / name / "lock.json")
        if lock != receipt["models"][name] or lock["binding"] != _binding(protocol_sha, name):
            raise ValueError("Individual model lock differs")
        if file_sha256(output / name / "best.pt") != lock["checkpoint_sha256"]:
            raise ValueError("Locked checkpoint bytes differ")
    return receipt


def train_all(root, baseline_source, output, *, device="cuda", steps=20000, batch_size=128,
              max_model_seconds=1200, eval_every=1000, resume=False):
    manifest = verify(root)
    require_official(manifest)
    baseline = model_module(baseline_source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = create_protocol(root, manifest, baseline_source, steps=steps, batch_size=batch_size,
                               max_model_seconds=max_model_seconds, eval_every=eval_every)
    path = output / "protocol.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Experiment exists; use --resume")
        if read_json(path) != protocol:
            raise ValueError("Protocol or source changed on resume")
    else:
        atomic_json(path, protocol)
    write_status(output, stage="loading_training")
    # Both sets are subsets of original train. Outer signals are not loaded here.
    corpus = RamCorpus.load(root, manifest, protocol["split"]["fit"] + protocol["split"]["inner"])
    fit, inner = corpus.subset(protocol["split"]["fit"]), corpus.subset(protocol["split"]["inner"])
    print(json.dumps({"event": "loaded_training_users", "fit": len(fit.users),
                      "inner": len(inner.users), "ram_bytes": corpus.bytes}), flush=True)
    locks = {}
    for name in NAMES:
        locks[name] = train_one(name, baseline, fit, inner, protocol, output, device, resume=resume)
    atomic_json(output / "all_models_locked.json", {
        "protocol_sha256": file_sha256(path), "models": locks,
        "outer_signals_loaded_before_lock": False})
    write_status(output, stage="all_models_locked")
    del corpus, fit, inner
    gc.collect()
    return verify_all_locks(output, protocol)


def baseline_predictions(x):
    q = x[:, -1]
    h = torch.arange(1, 21, dtype=x.dtype, device=x.device)[None, :, None]
    result = {"zero_velocity": q[:, None].expand(-1, 20, -1)}
    for width in (1, 5, 20):
        delta = models.multiply_wxyz(models.conjugate_wxyz(x[:, -width - 1]), q)
        velocity = models.log_so3(delta) / width
        result[f"cav{width}"] = models.multiply_wxyz(q[:, None], models.exp_so3(h * velocity[:, None]))
    return result


def context_speed(x):
    return torch.rad2deg(models.geodesic_radians(x[:, -21:-1], x[:, -20:])).mean(dim=1) * 100.0


def inference_timing(model, context, device, *, warmup=20, repeats=100):
    model.eval()
    x = context[:1].to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        _sync(device)
        timings = []
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            model(x)
            _sync(device)
            timings.append((time.perf_counter() - started) * 1000)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, .95)),
            "batch_size": 1, "warmup": warmup, "repeats": repeats,
            "scope": "Vast_host_PyTorch_not_XR_device", "device": str(device)}


def summarise_errors(errors, speeds):
    summary = {}
    control = errors["ordered"][:, :, -1].mean(axis=1, dtype=np.float64)
    for name, trajectory in errors.items():
        endpoint = trajectory[:, :, -1]
        user_means = endpoint.mean(axis=1, dtype=np.float64)
        bins = []
        for low, high in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
            mask = (speeds >= low) & (speeds < high)
            covered = mask.any(axis=1)
            conditional = [float(endpoint[i][mask[i]].mean()) for i in np.flatnonzero(covered)]
            bins.append({"past_speed_deg_s_low": low,
                         "past_speed_deg_s_high": high if math.isfinite(high) else None,
                         "windows": int(mask.sum()), "users": int(covered.sum()),
                         "mean_deg_equal_covered_user": float(np.mean(conditional)) if conditional else None})
        summary[name] = {"endpoint20_mean_deg": float(user_means.mean()),
                         "user_cluster_ci95_deg": bootstrap_mean(user_means),
                         "per_horizon_mean_deg": trajectory.mean(axis=(0, 1), dtype=np.float64).tolist(),
                         "per_user_endpoint20_deg": user_means.tolist(),
                         "median_window_deg": float(np.median(endpoint)),
                         "p90_window_deg": float(np.quantile(endpoint, .90)),
                         "p95_window_deg": float(np.quantile(endpoint, .95)),
                         "fraction_windows_below_1deg": float((endpoint < 1).mean()),
                         "fraction_users_below_1deg": float((user_means < 1).mean()),
                         "delta_vs_ordered_deg": float((user_means - control).mean()),
                         "paired_user_delta_ci95_deg": bootstrap_mean(user_means - control),
                         "fraction_users_better_than_ordered": float((user_means < control).mean()),
                         "speed_bins": bins}
    return summary


def evaluate_sample(corpus, fitted, *, seed, windows, device, batch_size=128):
    rng = np.random.default_rng(seed)
    names = (*NAMES, *BASELINES, "energy_proposal") if getattr(fitted["energy"], "arm", None) == "energy" else (*NAMES, *BASELINES)
    errors = {name: np.empty((len(corpus.users), windows, 20), dtype=np.float32) for name in names}
    speeds = np.empty((len(corpus.users), windows), dtype=np.float32)
    displacement = np.empty_like(speeds)
    duplicate_counts = []
    sample_hash = hashlib.sha256()
    example = None
    with torch.inference_mode():
        for index, user in enumerate(corpus.users):
            seen = set()
            sample_hash.update(user.user.encode())
            for start in range(0, windows, batch_size):
                count = min(batch_size, windows - start)
                x, y, draws = corpus.batch(rng, count, index, return_draws=True)
                sample_hash.update(draws.astype("<i8", copy=False).tobytes())
                seen.update((int(a), int(b)) for _, a, b in draws)
                if example is None:
                    example = x[:1].clone()
                x, y = x.to(device), y.to(device)
                predictions = {name: model(x) for name, model in fitted.items()}
                predictions.update(baseline_predictions(x))
                if "energy_proposal" in errors:
                    predictions["energy_proposal"] = energy_proposal(fitted["energy"], x)
                for name, prediction in predictions.items():
                    value = torch.rad2deg(models.geodesic_radians(prediction, y))
                    if not torch.isfinite(value).all():
                        raise ValueError("Nonfinite outer errors")
                    errors[name][index, start:start + count] = value.cpu().numpy()
                speeds[index, start:start + count] = context_speed(x).cpu().numpy()
                displacement[index, start:start + count] = torch.rad2deg(
                    models.geodesic_radians(x[:, -1], y[:, -1])).cpu().numpy()
            duplicate_counts.append(windows - len(seen))
    metadata = {"seed": seed, "users": len(corpus.users), "windows_per_user": windows,
                "window_draws": len(corpus.users) * windows,
                "duplicate_draws": int(sum(duplicate_counts)),
                "sample_sha256": sample_hash.hexdigest(),
                "sampling": "uniform_windows_with_replacement_equal_user_weight",
                "user_ids_private": [u.user for u in corpus.users]}
    return errors, speeds, displacement, metadata, example


def evaluate_all(root, baseline_source, output, *, device="cuda"):
    output = Path(output)
    protocol = read_json(output / "protocol.json")
    # This check precedes manifest verification and any outer data access.
    receipt = verify_all_locks(output, protocol)
    write_status(output, stage="loading_outer")
    if _source_hashes(baseline_source) != protocol["sources"]:
        raise ValueError("Evaluation sources differ from locked training")
    if file_sha256(Path(root) / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Evaluation corpus differs")
    manifest = verify(root)
    require_official(manifest)
    if split_users(manifest) != protocol["split"]:
        raise ValueError("Evaluation user split differs")
    baseline = model_module(baseline_source)
    corpus = RamCorpus.load(root, manifest, protocol["split"]["outer"])
    fitted = {}
    for name in NAMES:
        model = models.build_model(name, baseline_module=baseline, seed=SEED).to(device)
        state = _load_checkpoint(output / name / "best.pt", _binding(receipt["protocol_sha256"], name), device)
        model.load_state_dict(state["model"])
        fitted[name] = model.eval()
    report = {"protocol_sha256": receipt["protocol_sha256"], "source": "BOXRR-23",
              "metric": protocol["metric"], "outer_status": protocol["outer_status"],
              "checkpoints": receipt["models"], "samples": {}}
    for label, config in (("primary", protocol["outer_primary"]), ("legacy", protocol["outer_legacy"])):
        write_status(output, stage="evaluating_" + label)
        errors, speeds, displacement, metadata, example = evaluate_sample(
            corpus, fitted, seed=config["seed"], windows=config["windows_per_user"],
            device=device, batch_size=protocol["batch_size"])
        np.savez_compressed(output / f"{label}_errors.npz", **errors,
                            context_speed_deg_s=speeds, future_displacement_deg=displacement)
        report["samples"][label] = {**metadata, "models": summarise_errors(errors, speeds),
                                     "errors_sha256": file_sha256(output / f"{label}_errors.npz")}
        atomic_json(output / f"{label}_metrics.json", report["samples"][label])
        print(json.dumps({"event": "outer_evaluated", "sample": label,
                          "errors": {k: v["endpoint20_mean_deg"] for k, v in
                                     report["samples"][label]["models"].items()}}), flush=True)
    write_status(output, stage="benchmarking")
    report["latency"] = {name: inference_timing(model, example, device) for name, model in fitted.items()}
    report["environment"] = {"torch": torch.__version__, "numpy": np.__version__,
                             "device": str(device),
                             "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None}
    atomic_json(output / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-model-seconds", type=float, default=1200)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.stage in ("train", "all"):
            train_all(args.corpus, args.baseline_source, args.output, device=args.device,
                      steps=args.steps, batch_size=args.batch_size,
                      max_model_seconds=args.max_model_seconds, eval_every=args.eval_every, resume=args.resume)
        if args.stage in ("evaluate", "all"):
            evaluate_all(args.corpus, args.baseline_source, args.output, device=args.device)
        write_status(args.output, state="complete", stage="complete")
    except Exception as exc:
        if args.output.exists():
            write_status(args.output, state="failed", stage="failed", error=type(exc).__name__ + ": " + str(exc))
        raise


if __name__ == "__main__":
    main()

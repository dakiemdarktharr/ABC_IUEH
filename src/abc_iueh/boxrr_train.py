"""Bounded standalone BOXRR pilot trainer, initialized from model source only."""
import argparse
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .boxrr_pilot import verify, SEED
from .boxrr_source import file_sha256

MODEL_SHA256 = "731ba2dde12293b8111d4937f1fb7de593566d017a74ac31513330f8b57dd5ee"


def model_module(path):
    if file_sha256(path) != MODEL_SHA256:
        raise ValueError("Model source differs from the pinned architecture")
    spec = importlib.util.spec_from_file_location("boxrr_architecture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Batches:
    """Uniform users, then uniform eligible windows within that user. No cross-user mixing."""
    def __init__(self, root, manifest, partition, seed):
        self.root, self.rng = Path(root), np.random.default_rng(seed)
        self.users, self.handles = [], OrderedDict()
        for row in manifest["shards"]:
            if row["partition"] != partition:
                continue
            segments = []
            with h5py.File(self.root / row["path"], "r") as store:
                for session in store["sessions"].values():
                    for segment in session["segments"].values():
                        q = segment["quaternion"]
                        if len(q) >= 420:
                            segments.append((q.name, len(q) - 419))
            if segments:
                self.users.append((row["path"], segments, np.cumsum([n for _, n in segments])))
        if not self.users:
            raise ValueError("Partition has no eligible windows")

    def batch(self, count, user_index=None):
        x, y = [], []
        for _ in range(count):
            index = int(self.rng.integers(len(self.users))) if user_index is None else user_index
            path, segments, cumulative = self.users[index]
            draw = int(self.rng.integers(cumulative[-1]))
            segment = int(np.searchsorted(cumulative, draw, side="right"))
            start = draw - (int(cumulative[segment - 1]) if segment else 0)
            if path not in self.handles:
                if len(self.handles) >= 8:
                    self.handles.popitem(last=False)[1].close()
                self.handles[path] = h5py.File(self.root / path, "r")
            self.handles.move_to_end(path)
            q = self.handles[path][segments[segment][0]][start:start + 420]
            x.append(q[:400]); y.append(q[400:])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))

    def close(self):
        for store in self.handles.values():
            store.close()
        self.handles.clear()


def train(root, model_path, output, *, device="cpu", steps=1000, batch_size=128,
          learning_rate=.001, max_seconds=300, resume=False):
    if steps < 1 or batch_size < 1 or max_seconds <= 0 or learning_rate <= 0:
        raise ValueError("Invalid training bounds")
    manifest = verify(root)
    if not manifest["training_allowed"]:
        raise ValueError("Synthetic fixtures cannot be used for this training entry point")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError("Use --resume for this existing run")
    binding = {"manifest_sha256": file_sha256(Path(root) / "manifest.json"),
               "model_sha256": MODEL_SHA256, "seed": SEED, "batch_size": batch_size,
               "learning_rate": learning_rate, "sampling": "uniform_user_then_uniform_window",
               "loss": "endpoint20_geodesic_radians", "device_type": device}
    module = model_module(model_path)
    loader = Batches(root, manifest, "train", SEED)
    started = time.monotonic()
    model = module.OrderedMotionProduct(seed=SEED).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    step, total_seconds = 0, 0.
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Resume binding differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        loader.rng.bit_generator.state = state["sampler_rng"]
        step, total_seconds = state["step"], state["training_seconds"]

    def save():
        temp = output / "checkpoint.part.pt"
        torch.save({"binding": binding, "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "sampler_rng": loader.rng.bit_generator.state,
                    "training_seconds": total_seconds + time.monotonic() - started}, temp)
        temp.replace(checkpoint)

    try:
        model.train()
        while step < steps and time.monotonic() - started < max_seconds:
            x, y = loader.batch(batch_size)
            optimizer.zero_grad(set_to_none=True)
            loss = module.endpoint20_geodesic_loss(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                save()
                print(json.dumps({"step": step, "loss_radians": float(loss.detach()),
                                  "gradient_norm": float(norm)}), flush=True)
        save()
    finally:
        loader.close()
    report = {**binding, "step": step, "checkpoint_sha256": file_sha256(checkpoint),
              "elapsed_seconds": time.monotonic() - started, "validation_evaluated": False}
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(root, model_path, output, *, device="cpu", windows_per_user=256, batch_size=128):
    """Fixed sampled validation; mean endpoint20 geodesic degrees, with ZV/CAV1."""
    if windows_per_user < 1 or batch_size < 1:
        raise ValueError("Invalid validation bounds")
    manifest = verify(root)
    if not manifest["training_allowed"]:
        raise ValueError("Synthetic fixtures cannot be used for this evaluation entry point")
    output = Path(output)
    module = model_module(model_path)
    state = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
    if (state["binding"]["manifest_sha256"] != file_sha256(Path(root) / "manifest.json") or
            state["binding"]["model_sha256"] != MODEL_SHA256):
        raise ValueError("Checkpoint belongs to a different corpus or architecture")
    model = module.OrderedMotionProduct(seed=SEED).to(device)
    model.load_state_dict(state["model"]); model.eval()
    loader = Batches(root, manifest, "validation", SEED + 1)
    user_means = []
    try:
        with torch.no_grad():
            for index in range(len(loader.users)):
                sums = np.zeros(3)
                for start in range(0, windows_per_user, batch_size):
                    count = min(batch_size, windows_per_user - start)
                    x, y = loader.batch(count, user_index=index)
                    x, truth = x.to(device), y[:, -1].to(device)
                    latest = module.multiply_wxyz(module.conjugate_wxyz(x[:, -2]), x[:, -1])
                    cav = module.multiply_wxyz(x[:, -1], module.exp_so3(20 * module.log_so3(latest)))
                    predictions = (model(x)[:, -1], x[:, -1], cav)
                    sums += np.array([float(torch.rad2deg(module.geodesic_radians(p, truth)).sum())
                                      for p in predictions])
                user_means.append(sums / windows_per_user)
    finally:
        loader.close()
    means = np.mean(user_means, axis=0)
    result = {"manifest_sha256": state["binding"]["manifest_sha256"], "checkpoint_step": state["step"],
              "partition": "validation", "sampling": "fixed_uniform_windows_per_user_with_replacement",
              "seed": SEED + 1, "users": len(user_means), "windows_per_user": windows_per_user,
              "metric": "mean_endpoint20_geodesic_degrees_equal_user_weight",
              "model_deg": means[0], "zero_velocity_deg": means[1], "constant_angular_velocity_1_deg": means[2]}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "model-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--windows-per-user", type=int, default=256)
    args = parser.parse_args()
    if args.evaluate:
        result = evaluate(args.corpus, args.model_source, args.output, device=args.device,
                          windows_per_user=args.windows_per_user, batch_size=args.batch_size)
        print(json.dumps(result, indent=2))
        return
    print(json.dumps(train(args.corpus, args.model_source, args.output, device=args.device,
                           steps=args.steps, batch_size=args.batch_size,
                           max_seconds=args.max_seconds, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()

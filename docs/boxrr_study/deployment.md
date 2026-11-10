# Deployment checks: JEPA-TCN

These checks use the trained JEPA-TCN `width=64` private checkpoint and **synthetic inputs only**. They do not measure accuracy, open the final holdout or establish official submission readiness. The graph contains fitted weights and is not committed to this repository.

## ONNX export

On 10 November 2026, a private FP32 opset-17 graph was exported with static `context` float32 `[1,400,4]` and `prediction` float32 `[1,20,4]`. `onnx.checker` passed; ONNX Runtime 1.22.1 CPU matched the PyTorch CPU model on 128 generated contexts (2,560 output quaternions), with maximum sign-invariant full-SO(3) disagreement **0.00001172°**. This bounded synthetic test is not exhaustive and is not parity on a held-out human dataset.

The graph has SHA-256 `8440cb1e637fa7a0181909b486efdcc935cb7cbe46dae548e2b3ce362139b97b` and size 1,225,658 bytes. Its parent inference checkpoint has SHA-256 `3bdfd30b95623b9214d044da3dc6df6f8be51a04e95f500cab205ad43f08ff1a`. Graph bytes can vary with exporter/runtime versions; record the hash of the actual artifact used. The graph contains quaternion and trigonometric operations and two `Einsum` nodes, each with two operands. It has not been checked for operator mapping or CPU fallback on the target XR device. Tracing and constant-folding warnings were emitted; the graph has a fixed input shape.

To export with a privately restored inference bundle:

```bash
python -m pip install -r requirements-onnx.txt
python tools/boxrr_v2/export_onnx.py --model_dir /private/deployment/jepa_v2_w64 --output /private/model_fp32.onnx
```

The exporter verifies the source/checkpoint hashes and runs the ONNX checker plus 12 generated-input CPU parity cases against a 0.002° tolerance. It refuses to overwrite an existing output. A completed challenge package still needs the required model card, offline Linux/CUDA inference wrapper, pinned dependencies, broader parity and clean-environment verification; whether a PyTorch-only package is accepted remains ambiguous in the organizer wording. See [challenge checklist](challenge_checklist.md).

## Arithmetic count

For one stateless batch-one pass, hooks on the executed PyTorch model and direct contraction arithmetic give **19,346,112 MACs** for Conv1d, Linear and two fixed `Einsum` contractions. One MAC means one multiplication plus its accumulation. This is a **partial operator count**, not a verified complete submission count: biases, elementwise arithmetic, normalization, pooling, other reductions and nonlinear/trigonometric operations are excluded. On-device cost and latency cannot be inferred from this number.

| Counted operation | MACs | Working |
|---|---:|---|
| Conv1d | 19,221,504 | `64×12×(128+100) + 5×(128+100)×(64×5+64×128+128×64) + 20×3×64` |
| Linear | 123,456 | `243×64 + 448×128 + 64×32×20 + 128×(60+15)` |
| Two Einsum contractions | 1,152 | `3×2×32×3 + 3×64×3` |

For the two contractions and linear layers, PyTorch profiler independently reported 249,216 FLOPs, equal to twice their 124,608 MACs. Its local CPU profile did not report convolution FLOPs; the convolution count is computed from actual forward-hook output shapes and layer parameters, with the formula above as a separate check. Count convention and omitted operators need review against the intended submission graph.

```bash
python tools/boxrr_v2/count_macs.py --model_dir /private/deployment/jepa_v2_w64
```

## Local timing and target latency

On an AMD Ryzen AI 5 340 Windows machine, ONNX Runtime 1.22.1 CPU provider, batch 1, one intra-op and one inter-op thread, the private graph took **1.242 ms median** and **1.387 ms p95** across 200 `session.run` calls after 40 warm-ups. This is one local run on a synthetic input; it excludes model loading, input sign alignment and file I/O. It is not a target-XR measurement or a guaranteed latency. ST measures the official latency on its own XR-class hardware, according to the [challenge](https://sites.google.com/view/9th-abc-challenge/home/the-challenge).

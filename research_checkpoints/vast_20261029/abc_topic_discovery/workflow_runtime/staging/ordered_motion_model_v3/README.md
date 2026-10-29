# Ordered-motion model v2

Isolated prospective prototype from the 2026-10-29 ordered-motion review. V2
preserves the v1 predictor and adds adversarial numerical/gradient checks plus
a pinned raw-vector reference factory. It does not edit or replace frozen Q1,
change the official protocol, establish novelty, or authorize fitting. There
is no data access or training script here. V1 is preserved separately.

`ordered_motion_model.py` implements normalized active-Hamilton wxyz input
`[B,400,4]` and output `[B,20,4]`. It uses all 399 increments, seven fixed
balanced ordered-area reductions, 30 invariant scalars, and a learned
three-factor SO(3) exponential decoder. All widths are sample counts; 20/100
Hz/100 ms remains unresolved. The geometry tree is Python-unrolled at trace
time into static tensor operations (no ONNX Loop/Scan/If), with 1,009 pair
combines across the seven suffix widths.

| Arm | Trainable parameters | Exact condition |
|---|---:|---|
| `OrderedMotionProduct` | 2,477 | Ordered areas and `Exp(sB1) Exp(s²B2) Exp(s³B3)` |
| `MeansOnlyControl` | 2,477 | Every area replaced by latest increment × suffix mean; associated features recomputed |
| `SingleExponentialControl` | 2,477 | Same features/head and vectors, one `Exp(sB1+s²B2+s³B3)` |
| `DirectSameFeatureControl` | 2,474 | Same 30 scalars + 45 normalized vector coordinates; 75→29→9 head emits three `0.1*tanh` free-vector corrections over the same CAV base and product decoder |
| Pinned raw-vector reference | 1,683 | Existing `ArcModel("vector_angle", seed=...)` factory, pinned by source commit and file digest below |

At endpoint 20, all three powers of `s` equal one. The single-Exp arm therefore
has redundant coefficient rows under endpoint-only loss; equal nominal count
does not imply equal effective endpoint capacity. No interpretation of the
three factors as identified physical motion is supported.

Hidden-layer initialization is deterministic and nonzero; the final layer is
zero initialized, so each arm begins at current-increment constant angular
velocity. The exact endpoint-20 geodesic loss helper is in radians. Training,
seed schedules, source authorization, and outcome claims belong to a separately
reviewed protocol.

`test_ordered_motion_model.py` contains generated-only geometry, equivariance,
sign, gradient, shape, norm, and count checks. V2 adds explicit tests for the
near-π principal-log branch, exact half-turn sign choice, cancellation against
a float64 explicit-pair reference, nearly-collinear paths, and hidden-layer
gradients after one synthetic output-head update. These remain tests, not
measured behavior. **They have not been run.** The
coordinator requested at least 2 GiB available RAM and explicitly deferred all
Torch/ONNX/test imports and execution until scheduling. `export_static_onnx.py`
checks the RAM floor before importing Torch and writes opset 17 with static
shapes; actual exporter/runtime parity and latency remain unverified.

The contemporaneous raw-vector comparison is wired to the existing
`ABC_IUEH/src/abc_iueh/arc_model.py` factory `ArcModel("vector_angle", seed=...)`
at commit `6e7b8a6aaf6d7013cc87ebdbce847d5d13b07837`, file SHA-256
`ddbe6fcd9cb59f112e19e2a4297c453ad3195169eb31615bdb9a5af96fea78dd`, with an
expected 1,683 trainable parameters. `raw_vector_reference.py` checks that
digest before lazily importing the factory and checks the count after
instantiation. Its existing deterministic initializer is reused; no alternate
raw-vector weights or architecture are introduced. Details are recorded in
`raw_vector_reference.json`.

Quaternion math is adapted from the reviewed local ARC pilot geometry in
`ABC_IUEH/src/abc_iueh/arc_model.py` without modifying it. Dependency versions
are inherited from `ABC_IUEH/requirements-gpu-lock.txt` and
`ABC_IUEH/requirements-export-lock.txt`; the CUDA wheel is pinned, while a
CPU-only export setup remains unverified.

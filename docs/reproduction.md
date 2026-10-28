# Reproducing the selector pilot

Updated 2026-10-28. This is a train/development research pilot, not a challenge
submission or final-holdout evaluation.

## Required local artifacts

The public repository contains the implementation, frozen configs and audited
aggregate outputs. It does not contain the source recordings, processed HDF5,
split manifest, feature caches, scalers, per-user metrics, sampled schedules,
checkpoints or exported ONNX graphs. Re-running the measured stages requires
the matching local artifacts created by the preceding pipeline stages. The
code checks their recorded hashes and refuses incompatible or overwritten
experiment bundles. Follow the [data source review](data_sources.md),
[protocol](protocol.md), [baseline feature plan](baseline_feature_plan.md) and
[baseline results](baseline_results_2026-10-28.md) before running them.

The study used the locked user split of 15 train, 5 development and 15 final
users. Only train and development users were opened for selector fitting and
evaluation. The final partition remains sealed. A fresh checkout alone cannot
recreate the reported results because the required source and intermediate
artifacts are private and are intentionally not distributed here.

## Optional dependencies

Start with the base environment and editable package installation in the
[setup instructions](../README.md#setup). For the local CUDA fitting stage,
install the pinned Windows/Linux CUDA 12.8 PyTorch build:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-lock.txt
```

The full-graph portability and CPU ONNX Runtime audit additionally requires:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-export-lock.txt
```

The locks pin PyTorch 2.9.1+cu128 for GPU fitting, ONNX 1.18.0 and ONNX
Runtime 1.22.1 for export/audit, alongside the base NumPy/SciPy/HDF5 versions.
The training command requires CUDA and fails rather than silently falling back
to CPU. CPU-only CI exercises synthetic correctness paths; it does not fit
models or reproduce these real-data outcomes.

To regenerate the aggregate PNG figures, install the optional plotting lock:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-figures-lock.txt
```

It pins Matplotlib 3.11.1. This dependency is not needed for model fitting or
the inference audit.

## Commands

From the repository root in PowerShell, set deterministic thread settings
before starting Python. The CLI sets `CUBLAS_WORKSPACE_CONFIG` before importing
PyTorch; setting it explicitly makes the run environment visible:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

Run the frozen R80 study, the predeclared R128/R256 sensitivity, and the
separate numerical-portability/local-cost audit:

```powershell
.\.venv\Scripts\python.exe -m abc_iueh selectors
.\.venv\Scripts\python.exe -m abc_iueh selector-sensitivity
.\.venv\Scripts\python.exe -m abc_iueh inference-audit
```

Each command defaults to the paths and configurations recorded in its CLI
help and the corresponding file under `configs/`. Existing output directories
are refused; use a new `--output-dir` when creating a separate run. The stages
validate their input hashes and partition restrictions. Do not substitute a
different dataset, split, baseline cache, scaler or run configuration and call
it an exact reproduction.

To run the synthetic software checks, use `python -m pytest -q` after installing
the base dependencies and package. The local suite reported 329 passing tests.
Those tests cover synthetic pipeline, selector, metrics and graph behavior;
they do not establish that a real-data fit completed, that the final partition
was evaluated, or that local latency transfers to an organizer device. CI status
for the published revision is available in [GitHub Actions](https://github.com/dakiemdarktharr/ABC_IUEH/actions).

## Measured train/development results

The selector has 244 trainable parameters and 224 dense selector MACs. Each
fit used 2,000 SGD steps, batch size 128, learning rate 0.001 and fixed seeds
20261026/20261027. The six R80 variants used the same initialization per seed
and identical sampled train-origin batches across variants. R128 and R256
repeated the h1/h20 comparison at the same budget. GPU fitting used CUDA
float32 without autocast or TF32. Reported learned metrics average the two
seeds' error vectors; they are not ensemble predictions.

Lower endpoint-20 mean geodesic error is better:

| Evidence length | h1 mean dev error | h20 mean dev error | h20 − h1 | Paired user-cluster 95% CI | Dev users improved |
|---:|---:|---:|---:|---:|---:|
| R80 | 4.2849° | 4.2943° | +0.009396° | [+0.005227°, +0.013899°] | 0/5 |
| R128 | 4.2901° | 4.2965° | +0.006458° | [+0.003490°, +0.009701°] | 0/5 |
| R256 | 4.2960° | 4.2996° | +0.003575° | [+0.001684°, +0.005676°] | 0/5 |

The primary R80 comparison does not support the claim that h20 feedback-age
evidence improves the selector over h1. The observed difference also fails the
predeclared 3% relative and 0.05° absolute practical-improvement thresholds.
The train-fitted static mixture measured 3.7249° on development, better than
both learned R80 h1 and h20 variants. This mixture used a different fitting
objective and compute procedure, so it is a reference rather than a matched
selector ablation. Inference-audit numerical parity is a portability check,
not an accuracy result.

All development inference is exploratory and conditional on five users. These
results do not establish final-holdout accuracy, formal confirmation or
equivalence, foundation-model capability, competition eligibility, or latency
on the challenge XR target. The official 400-to-20 sample interface is
preserved; the 20-sample / 100 Hz / 100 ms wording remains unresolved, and no
time conversion is inferred here. Alyx-only challenge eligibility is also
unresolved. The paper source and PDF are local for review and have not been
posted as a preprint. JEPA/EBM work has not started, and nothing here implies
that registration or challenge submission occurred.

## Public outputs

- [Selector results and limitations](selector_results_2026-10-28.md)
- [R80 aggregate JSON](results/selector_main_R80.json)
- [R sensitivity aggregate JSON](results/selector_R_sensitivity.json)
- [Inference audit aggregate JSON](results/selector_inference_audit.json)

Private arrays, participant-level outputs, model weights and exported graphs
remain outside the repository.

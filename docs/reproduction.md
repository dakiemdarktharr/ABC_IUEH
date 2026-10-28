# Reproducing the selector and angular-exposure pilots

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

## Angular-exposure pilot

This is a separate, completed exploratory inner-validation experiment. It used
only the existing 15 outer-training users: three user-disjoint folds of ten
fit/five validation users, two seeds and five learned arms, totaling 30 fits.
The old five development users and 15 final users were not evaluated or used
for fitting. Development origin-index integrity metadata was checked; no
development signal or feature values were read. Whole-file hashes are opaque
integrity checks, not final-set evaluation.

The exact measured source is commit
`13c74071156b5b8dfc893f42107ad2cb82ea6921`, with configuration
[`configs/arc_pilot.json`](../configs/arc_pilot.json). Git stores the runner
with LF line endings; the measured Windows worktree had CRLF. A private
receipt binds raw bytes, normalized bytes and the Git blob. Exact raw measured
sources were archived locally before any later changes. After completion and
independent audit, resume guards were strengthened to reject copied or altered
fold/reference artifacts. That separate repair changes no model, loss,
schedule or measured efficacy result; see [review closure](arc_pilot_code_review.md).

With the pinned dependencies, deterministic environment above, and the
matching private HDF5/split/baseline-index artifacts, the measured command was:

```powershell
.\.venv\Scripts\python.exe -u -m abc_iueh.arc_experiment --config configs/arc_pilot.json --output data/arc_pilot/20261028 --device cuda
```

For a new replication use a **new output directory**. An unchanged interrupted
run resumes from its immutable checkpoints; incompatible sources/configuration
are rejected. In particular, do not point the post-repair runner at the old
completed output and describe that as an identical-source rerun. No replication
or additional GPU fit is required to read the public aggregates. The private
source data and artifacts remain prerequisites for reproducing real outcomes.

The primary gives every arm 5,000 Adam updates, batch size 128, learning rate
0.001 and a common fold/seed origin schedule. All arms share encoder
initialization within a cell. Natural loss units differ and are recorded in
the config/plan. The 1,649–1,717 parameter models have nearly matched encoder
MACs; equal updates are not equal elapsed time. Each fold's static mixture
and exposure prior are fit on the same fitting-only 65,536-origin sample,
without using the old all-training scaler or weights.

The predeclared timing-only secondary selects the last saved 100-step
checkpoint below the minimum final per-arm training time in each cell. It
uses the same validation origins and performs no extra fit or accuracy-based
checkpoint selection. Selected training times have maximum/minimum ratios
1.026656 overall and 1.038280 in the worst cell; update/example counts differ.
The corresponding fixed-update time ratios are 2.343 and 2.361. Static fitting
is not matched to neural optimizer compute. Neither analysis establishes equal
FLOPs or equal end-to-end cost on other hardware.

To rebuild the aggregate-only report from the locally held completed summary
and session ledgers, without loading human vectors or fitting:

```powershell
.\.venv\Scripts\python.exe scripts/build_arc_pilot_report.py --help
.\.venv\Scripts\python.exe scripts/build_arc_pilot_report.py
```

The independent audit separately streamed the private errors and verified
fold membership, fitting-only calibration, common schedules, checkpoint
selection, bootstrap arithmetic and source bindings. Both practical-benefit
gates failed. The [results](arc_pilot_results_2026-10-28.md),
[independent audit](arc_pilot_results_audit.md),
[aggregate](results/arc_pilot_20261028.json),
[audit aggregate](results/arc_pilot_independent_audit.json) and
[research decision](arc_pilot_decision_vi.md) preserve positive and negative
findings. User-bootstrap uncertainty is conditional on these fixed CV fits
and two seeds; overlapping windows are not treated as independent users.

Actual GPU: RTX 5050 Laptop, PyTorch 2.9.1+cu128. Training totaled 1,238.49
synchronized seconds; all three CUDA-context sessions totaled 3,914.56
seconds. Peak allocated/reserved CUDA memory of the shared worker was
452.57/486.00 MiB, including data caches and five resident arms. Two host-RAM
guard pauses resumed unchanged; one own-process working-set trim relieved
resident memory. Timings include paging effects. No cloud GPU was rented.

After the guard repair, the local synthetic suite passed 421 tests with
14 existing legacy ONNX deprecation warnings in 71.49 seconds. Run the suite
with the deterministic environment above; a restricted Windows sandbox may
need a new disposable `--basetemp` under `data/`. These tests do not prove
trained pilot export, clean-Linux compatibility, XR latency, foundation-model
transfer, challenge eligibility or a final-holdout result. The old study's
ONNX receipts apply to its selector models, not these new trained models.

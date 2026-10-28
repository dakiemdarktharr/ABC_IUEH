# ABC_IUEH

Reproducible data preparation and train/development experiments for **Forecast Horizon and Feedback Age in Zero-Shot Head-Orientation Prediction**, a proposed study for the 9th ABC Challenge.

The main question is whether mature multi-step prediction errors help select a head-orientation extrapolator after controlling feedback age. Quaternion representation, constant angular velocity, ensemble routing and ordinary neural backbones are inherited components, not novelty claims.

## Current scope

- Source-use audit with a bounded, pinned **WhoIsAlyx** motion downloader.
- Named-column HMD quaternion extraction; native-rate gate; normalization and hemisphere alignment; session filtering; gap-safe offline SLERP; chunked HDF5.
- Deterministic user splits **before** windowing, lazy segment-local 400-to-20 windows, and guarded final-holdout access.
- Causal h1/h20 expert-error features with feedback-age controls, complete-horizon and motion-only features; train-only scalers.
- Zero/constant-angular-velocity, analytic routing/mixture references and a compact NumPy MLP; synthetic correctness tests and user-cluster evaluation.

This is a conditional research pilot, not a submission. Train/development measurements do not establish foundation-model capability, competition eligibility, final-holdout accuracy or target-XR latency. JEPA and EBM are optional later experiments.

**Verified preparation run, 2026-10-28:** 70 source files verified; 67 of 69 sessions retained; 35 eligible users, 134 continuous segments and 8,026,425 quaternion samples. The user split is locked at **15 train / 5 development / 15 final** with seed20261026. HDF5 integrity checks and the initial 49 synthetic tests passed. No model fitting or forecast-error evaluation occurred during preparation. See the [aggregate preparation receipt](docs/data_preparation_summary.json); participant/session manifests and recordings are private.

The subsequent [metadata-integrity update](docs/metadata_integrity.md) pins SHA-256 for the reviewed DVC pointer inventory, removes the SHA-1 computation, and added six regression cases. CI separately checks the reported Semgrep rule.

## Data-use status

| Source | Current use decision |
|---|---|
| WhoIsAlyx, players 42–76 | Conditional use for academic, noncommercial pilot under CC BY-NC-SA 4.0 and applicable local duties. Each session must pass measured quality gates. |
| BOXRR-23 | Disabled until the recipient's advance approval required by its DUA is documented. |
| OpenNEEDS / other candidates | Not enabled without verified access terms, orientation convention and native sampling rate. |

See [source audit](docs/data_sources.md). Collector consent, recipient obligations, technical suitability and competition eligibility are separate. **Alyx-only participation eligibility remains unresolved.** Raw recordings, processed HDF5, user manifests, local attestations and checkpoints stay outside Git.

## Setup

Python 3.11 is the tested interpreter. Create a local environment and install the recorded dependency versions:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. Tests generate synthetic data and do not access human recordings or the network. In restricted Windows sandboxes whose system temp directory disallows rename, create `data/` and pass `--basetemp=data/test_tmp` to pytest; never point basetemp at valuable files because pytest clears that directory.

## Staged pipeline

First inspect metadata without downloading recordings:

```powershell
.\.venv\Scripts\python.exe -m abc_iueh discover
```

The reviewed source commit is pinned. Its inventory contains 70 motion files from 35 potential users, about 5.76 GB in total; these are inventory counts, not qualifying users. Source README, license and DVC config are saved privately under `data/source_evidence/`.

After reviewing the source terms and confirming a noncommercial use:

```powershell
.\.venv\Scripts\python.exe -m abc_iueh fetch --noncommercial --max-files 2 --max-total-mib 256
.\.venv\Scripts\python.exe -m abc_iueh ingest
.\.venv\Scripts\python.exe -m abc_iueh check
```

The default pilot is bounded to two files. It cannot establish the multi-user data gate. A complete reviewed-cohort fetch can be requested explicitly:

```powershell
.\.venv\Scripts\python.exe -m abc_iueh fetch --noncommercial --max-files 70 --max-total-mib 6000
.\.venv\Scripts\python.exe -m abc_iueh ingest
.\.venv\Scripts\python.exe -m abc_iueh check
.\.venv\Scripts\python.exe -m abc_iueh split
```

Downloads stream to temporary files, verify upstream size/MD5 and record SHA-256. Existing verified files are reused. The byte cap counts existing plus selected recordings. No full DVC pull, videos, eye tracking, demographics or physiology are requested. The native motion CSV includes controller fields; only time and named HMD quaternion fields enter the cleaned dataset.

All parts of a source session must be present before it can qualify. Split recordings are grouped into one session for filtering/capping. A reset or duplicate native clock is rejected rather than silently sorted, repaired or joined across a reset.

`split` requires at least 30 eligible users. If the cohort is too small, it fails. For schema/window smoke checks only, use a separate manifest:

```powershell
.\.venv\Scripts\python.exe -m abc_iueh split --smoke --output data/smoke_split.json
```

Smoke manifests have **no training/development/final assignments** and cannot be used for training. Do not evaluate forecast errors on smoke users and later call them an untouched final holdout. Confirmatory manifests refuse overwrite and are bound to the exact HDF5 checksum.

## Train/development baseline and feature stage

The [frozen plan](docs/baseline_feature_plan.md) and
[configuration](configs/baseline_features.json) define this separate stage.
It has no final-partition access option. These commands use an existing locked
source HDF5/split and refuse to overwrite experiment artifacts:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m abc_iueh features
.\.venv\Scripts\python.exe -m abc_iueh baselines
.\.venv\Scripts\python.exe -m abc_iueh neural-baseline
```

The default private run is `data/baseline_runs/20261028_r80/`. Features creates
and seals its plan, cache and train-fitted scalers together. `seal-features`
validates and seals a legacy receipt without regenerating arrays; it rejects
changed artifacts. Baseline consumers independently verify the sealed bundle.
For a new run directory, pass the same `--output-dir` to all three commands.
Set BLAS thread limits before starting Python; the neural reference uses CPU
NumPy and does not require PyTorch or a GPU.

The six feature variants have ten raw columns each. Future endpoint losses are
stored separately as labels. Train-only population scalers are ready for the
later learned selector; analytic routers use raw past errors in degrees.
The 244-parameter learned h1/h20 selector is measured in the subsequent GPU
stage below; it was outside the earlier baseline stage.
Recordings, caches, per-user errors, scalers and weights remain private.
See the [measured aggregate results](docs/baseline_results_2026-10-28.md) and
[independent implementation review](docs/baseline_stage_review.md).

## Selector GPU study and inference audit

The follow-up selector study completed **20 GPU fits**: 12 R80 fits across six
variants and two seeds, then eight h1/h20 sensitivity fits at R128 and R256.
The development results are exploratory and do not meet the predeclared
practical-improvement thresholds. The final 15 users remain sealed. A full
reproduction requires the locally held source HDF5, split, baseline cache and
train-only scalers whose hashes are bound in the configs; these data and model
artifacts are not published.

For the optional CUDA stage, install the pinned GPU build after the base setup
above. The full reproduction commands, dependencies, output scope and measured
results are in [reproduction notes](docs/reproduction.md). The [selector
results](docs/selector_results_2026-10-28.md) include the aggregate metrics,
limits and machine-readable JSON. The separate inference audit requires the
optional ONNX dependencies. It measures portability and local cost, not
challenge-device latency or predictive improvement.

The local synthetic suite reported 329 passing tests; those checks do not
recreate the GPU results. CI status for the published revision is available in
[GitHub Actions](https://github.com/dakiemdarktharr/ABC_IUEH/actions).
The paper source and PDF remain local for review; no preprint has been posted.
JEPA/EBM experiments have not started. Alyx-only challenge eligibility and the
20-sample / 100 Hz / 100 ms specification remain unresolved.

## Protocol and outputs

Read [protocol](docs/protocol.md), [HDF5 schema](docs/schema.md), [implementation review](docs/pipeline_review.md) and [data attribution](docs/attribution.md).

Graph interface remains `float32 context[1,400,4] -> prediction[1,20,4]`, scalar-first wxyz, shift1. The challenge's **20 samples / 100 Hz / 100 ms inconsistency is unresolved**. Configuration preserves 20 outputs and a 100 Hz primary dataset; it does not silently change either number.

Data preparation writes private provenance, quality/rejection statistics and processed data under `data/`. The separate train/development experiment stage fits only on training users. Any published results must distinguish real-data outcomes from synthetic correctness checks.

Upstream data retain their own licenses. A source-code license has not yet been selected; public repository visibility does not relicense the dataset.

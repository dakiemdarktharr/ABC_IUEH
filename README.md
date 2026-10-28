# ABC_IUEH

Reproducible data preparation for **Forecast Horizon and Feedback Age in Zero-Shot Head-Orientation Prediction**, a proposed study for the 9th ABC Challenge.

The main question is whether mature multi-step prediction errors help select a head-orientation extrapolator after controlling feedback age. Quaternion representation, constant angular velocity, ensemble routing and ordinary neural backbones are inherited components, not novelty claims.

## Current scope

- Source-use audit with a bounded, pinned **WhoIsAlyx** motion downloader.
- Named-column HMD quaternion extraction; native-rate gate; normalization and hemisphere alignment; session filtering; gap-safe offline SLERP; chunked HDF5.
- Deterministic user splits **before** windowing, lazy segment-local 400-to-20 windows, and guarded final-holdout access.
- Zero/constant-angular-velocity references and synthetic correctness tests.

This is research infrastructure, not a trained submission. No head-motion forecasting accuracy, foundation-model capability, competition eligibility or target-XR latency has been established. JEPA and EBM are optional later experiments.

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

## Protocol and outputs

Read [protocol](docs/protocol.md), [HDF5 schema](docs/schema.md), [implementation review](docs/pipeline_review.md) and [data attribution](docs/attribution.md).

Graph interface remains `float32 context[1,400,4] -> prediction[1,20,4]`, scalar-first wxyz, shift1. The challenge's **20 samples / 100 Hz / 100 ms inconsistency is unresolved**. Configuration preserves 20 outputs and a 100 Hz primary dataset; it does not silently change either number.

Each pipeline run writes private provenance, quality/rejection statistics and processed data under `data/`. It does not fit model parameters, inspect final forecast errors or establish accuracy. Any published results must distinguish real-data outcomes from synthetic correctness checks.

Upstream data retain their own licenses. A source-code license has not yet been selected; public repository visibility does not relicense the dataset.

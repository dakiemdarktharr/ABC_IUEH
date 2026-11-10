# Head-orientation prediction on BOXRR-23

This project predicts a sequence of head orientations from past quaternion observations. **JEPA-TCN** is the current development candidate. The repository contains its architecture, training and inference code, comparisons, and evaluation protocol.

| | Contract |
|---|---|
| Input | 400 observed unit quaternions, scalar-first `(w, x, y, z)` |
| Output | 20 future unit quaternions in the same order |
| Primary metric | Mean full-SO(3) geodesic error in degrees at output 20 |
| Selected model | `jepa_v2`, `width=64`, `latent_dim=128`, `use_gravity=True` |
| Data | A filtered, user-disjoint subset of official BOXRR-23 |

The organizer specifies 20 outputs at 100 Hz but also describes them as 100 ms. These statements remain inconsistent: our implementation's 20th output is **+200 ms** from the last observation at 100 Hz. We retain the specified 20-output interface; +200 ms is our indexing, not an organizer correction.

## Results

JEPA-TCN was selected using a 256-window-per-user development sample. On a separate 1,024-window-per-user draw from the same 101 development users, it obtained **4.842644°** at output 20, the lowest reported value in this search. This does **not** establish that JEPA-TCN is reliably better than Mixer or that pretraining caused an improvement. This is a sampled, window-weighted development result, not an official test score. The same users were used repeatedly for model selection; the 122-user final holdout remains sealed.

| Model or reference | Endpoint-20 error |
|---|---:|
| JEPA-TCN | **4.842644°** |
| Mixer | 4.848421° |
| GRU | 4.849759° |
| TCN | 4.883348° |
| Hold-last | 7.514382° |
| Constant angular velocity | 8.314382° |

The four neural models shown use `width=64`. [Results and limitations](docs/boxrr_study/results.md) describe sampling, uncertainty, other configurations and the limits of comparison.

## Install and check the code

Python 3.11 and PyTorch 2.9.1 are used for the recorded setup. The following installs a CPU environment on Linux; choose a compatible PyTorch build for CUDA if needed.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-boxrr.txt
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
python -m pytest -q tests tools/boxrr_v2/test_inference_v2.py
```

Tests use generated data and do not measure trained-model accuracy. To instantiate the architecture with **random weights**, see [setup](docs/boxrr_study/setup.md#instantiate-the-selected-architecture).

## Run the trained model

The trained weights and the dataset are **not included in this public repository**. To reproduce the reported model's predictions, obtain the verified private inference bundle through the team, subject to the dataset terms. The bundle includes `jepa_v2_w64/config.json`, `jepa_v2_w64/best.pt`, and the four source files under its `src/` directory. It is distinct from the full training checkpoint.

With that bundle and a NumPy file containing 400 finite, normalized wxyz quaternions:

```bash
python tools/boxrr_v2/run_inference_v2.py \
  --model_dir /private/deployment/jepa_v2_w64 \
  --input /private/context.npy --output /private/prediction.npy
```

The adapter checks the bundled model, configuration and source hashes; it runs offline and writes a float32 `[20,4]` array. It refuses to overwrite an existing output. A private experimental ONNX export now passes a bounded CPU parity check; it is **not** a completed submission or a target-device latency measurement. See [deployment measurements](docs/boxrr_study/deployment.md) for the export, arithmetic-count scope and remaining checks. Detailed bundle layout, setup and training recipe are in [setup](docs/boxrr_study/setup.md).

## Documentation

- [Models and architecture](docs/boxrr_study/models.md): JEPA-TCN, Mixer, GRU, TCN and physical references.
- [Setup and reproduction](docs/boxrr_study/setup.md): dependencies, configuration, private inference and preprocessing.
- [Results](docs/boxrr_study/results.md): development results, uncertainty and limitations.
- [Deployment measurements](docs/boxrr_study/deployment.md): ONNX export, partial MAC count and local timing.
- [Challenge checklist](docs/boxrr_study/challenge_checklist.md): interface, artifact requirements and remaining work.
- [Data sources](docs/data_sources.md) and [protocol](docs/protocol.md): provenance and split rules.

## Project layout

`src/abc_iueh/` contains the BOXRR processing and model code; `tests/` and `tools/boxrr_v2/` contain synthetic checks and the offline inference adapter. `docs/` holds the current study documentation. Recorded identities, trajectories, private manifests, checkpoints and credentials are excluded from Git.

BOXRR-23: Vivek Nair, Wenbo Guo, Rui Wang, James F. O'Brien, Louis Rosenberg and Dawn Song, *Berkeley Open Extended Reality Recording Dataset 2023 (BOXRR-23)*, University of California, Berkeley, 2023, [doi:10.25350/B5NP4V](https://doi.org/10.25350/B5NP4V). Dataset access and use are governed by [Berkeley's terms](https://rdi.berkeley.edu/metaverse/boxrr-23/); repository visibility does not change them. No source-code license has yet been selected for this repository.

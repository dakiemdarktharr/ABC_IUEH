# Setup and reproduction

The public code describes the study; actual recordings, user manifests and fitted weights are private and must be obtained and used under applicable BOXRR terms. A synthetic smoke check verifies installation, not scientific results. The historical `docs/boxrr_pilot.md` refers to the earlier pilot and its original split; do not treat it as the expanded V2 protocol.

## Software setup

Recorded local versions include Python 3.11, NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0 and PyTorch 2.9.1+cu128; the appropriate PyTorch wheel and driver depend on the machine. On Linux, for example:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-boxrr.txt
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps -e .
python -m pytest -q tests/test_boxrr_v2_deterministic.py tests/test_boxrr_v2_latent.py tests/test_boxrr_v2_extra.py tests/test_boxrr_v2_experiment.py tools/boxrr_v2/test_inference_v2.py
```

These tests use generated inputs. For CPU, install a compatible PyTorch 2.9.1 CPU build using the official PyTorch instructions; CUDA is not required to inspect the model. Package version pinning is not equivalent to an independently verified clean-environment reproduction of the original GPU run.

## Instantiate the selected architecture

This creates **randomly initialized weights**, not the trained 4.842644° model:

```python
import torch
from abc_iueh.boxrr_v2_experiment import build_model

model = build_model('jepa_v2', {'width': 64, 'latent_dim': 128, 'use_gravity': True}, seed=20261110)
context = torch.zeros(1, 400, 4, dtype=torch.float32)
context[..., 0] = 1.0  # synthetic identity quaternions
model.eval()
with torch.inference_mode():
    prediction = model(context)
assert prediction.shape == (1, 20, 4) and torch.isfinite(prediction).all()
print(prediction.shape)
```

For real inference, use the verified private export and matching four source files plus `tools/boxrr_v2/run_inference_v2.py`, the hash-bound offline adapter. Training checkpoints and inference exports are distinct artifacts. The adapter is stateless and offline; it is not an ONNX submission or target-device certification. Do not load arbitrary untrusted checkpoints. With a privately restored, verified inference bundle whose relative source layout is intact:

```bash
python tools/boxrr_v2/run_inference_v2.py --model_dir /private/deployment/jepa_v2_w64 --input /private/context.npy --output /private/prediction.npy
```

`context.npy` must contain one finite, unit-normalized `[400,4]` wxyz float array. The command requires the four hash-matching model source files at the location in the export's private `config.json` and will not download weights. It will not overwrite an existing output. The private export is not provided by this repository.

## Recorded V2 experiment recipe

| Setting | Recorded value |
|---|---|
| Data | Official BOXRR-23; 600 fit, 101 development, 122 sealed final users |
| Data grid and windows | 100 Hz, 400 observed and 20 target quaternions, segment-local |
| Configurations | Seven families × `width=32,64`; `latent_dim=128`, `use_gravity=True` |
| Random seed | `20261110`; one seed for the recorded V2 search |
| Training sampling | 50% uniform user + 50% proportional to eligible windows, with replacement |
| Optimizer | AdamW; batch 128, weight decay 0.0001, gradient clip 1 |
| Base | Up to 20,000 updates; initial LR 0.001; cosine schedule and phase time cap |
| JEPA pretraining | 5,000 updates; maximum 600 seconds; EMA decay 0.998 |
| Refinement | Top three per width by development score; up to 40,000 updates for width64, initial LR 0.0002; optimizer reset; 1,800-second cap |
| Loss | Endpoint geodesic + 0.1 × mean trajectory geodesic; JEPA adds 0.02 × representation objective during supervised training |
| Selection / report | 256 / 1,024 windows per development user; selected by window-weighted endpoint20 |

For JEPA-Base, all 5,000 pretrain, 20,000 base and 40,000 refinement updates completed. The selected checkpoint was at refinement step **38,000**, after the base phase; the 1,024-window report score is 4.842644°, while checkpoint selection used a separate smaller draw on the same users. Width32 and width64 used different recorded GPU sessions, and a width64 phase resumed from a checkpoint on different hardware. For exact replay, use the private verified run archive, bound source hashes and matching environment. Re-running the generic CLI will not reproduce the paused/resumed hardware schedule or guarantee identical metrics.

For a **new** authorized experiment with a compatible, privately held corpus, the following illustrates the width64 settings. It starts all seven model families and may require substantial compute; it is not a free or exact replay of the recorded run. `CORPUS` must have the locked manifest and fit/development shards, but no final shards on the training host. Use a fresh output directory. Budget, access terms and any provider lease must be settled separately.

```bash
python -m abc_iueh.boxrr_v2_experiment --stage train \
  --corpus /private/boxrr-expanded --output /private/new-width64-run \
  --device cuda --width 64 --steps 20000 --batch-size 128 \
  --base-seconds 1200 --refine-steps 40000 --refine-seconds 1800
```

After training locks the models, `--stage development` can score the development cohort using the same corpus and output paths. `--stage evaluate` is the separate final path: do not invoke it during selection, and do not treat it as a smoke test.

Original selected full training checkpoint SHA-256: `f76710006a9e642d288f2afa8812ea49da77bff4ed851d76381abeda4934b973`. The trimmed inference export has a different digest: `3bdfd30b95623b9214d044da3dc6df6f8be51a04e95f500cab205ad43f08ff1a`. Private artifacts must not be committed or redistributed with the repo.

## Data processing choices

Source revision: `cschell/boxrr-23` at `af25be3ef76b176fbee0a094e82d97a611f9c950`. XROR axes are read by name and converted to wxyz. Estimate the native rate from median timestamp intervals and retain 85–110 Hz; remove only adjacent duplicate initial timestamps as documented by the adapter and reject interior duplicate/backward clocks. Normalize valid quaternions and align signs. Trim five seconds; use nonoverlapping five-second windows, per-axis 90th−10th percentile Euler sweep and wrapped angular path velocity; aggregate median by window then max by axis. The implementation uses strict `>17°` and `>15°/s`. Cap session length at 1,200 seconds; split gaps over 50 ms; resample each retained segment by offline SLERP at 100 Hz and require at least 1,000 samples per segment. Split by user before training windows are sampled.

Intake screening and resource caps changed prospectively during data preparation, prior to model fitting, and the locked corpus uses both initial and later added users. Consequently this short recipe alone cannot recreate the identical cohort without the private manifest, shard hashes and dated intake policies. Offline bracketing interpolation is not evidence of zero-delay processing of a live sensor.

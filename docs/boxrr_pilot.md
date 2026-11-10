# BOXRR-23 pilot for Vast

**Urgent pilot update, 2026-11-05:** at the user's request, the current cohort
was frozen at 4.434 GiB from 359 official BOXRR-23 users, with 287 train and
72 validation users. This is for preliminary evaluation and does not establish
final-test accuracy. The earlier 6–8 GiB intake target below is superseded for
this release; its provenance is recorded in `pilot_size_exception.json`.

The original plan prepares 6–8 GiB of filtered HDF5 shards and an 80/20
train/validation split by user. It reuses the existing preprocessing and the
reviewed OrderedMotionProduct model source. The current tests use generated
BSON/fpzip/XROR archives, CPU optimizer steps and checkpoint restoration.

The current intake follows the user's explicit 2026-11-05 instruction confirming
access authority and authorizing local filtering and private Drive upload.
A private operator-authorization record binds that instruction to the intake.
It is not an institutional ethics decision and does not claim one was supplied.

## Install and exercise the complete generated path

Python 3.11 is used locally. From the repository or extracted toolkit:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-boxrr.txt
.venv/bin/python -m pip install --no-deps -e .
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m pytest -q tests/test_boxrr.py tests/test_boxrr_pilot.py
```

The model integration case additionally uses PyTorch. The existing Linux GPU
target uses `torch==2.9.1+cu128`. The generated tests run on CPU. Keep installing
and resolving the GPU environment separate until a host is chosen.

## Authorized intake

The private JSON records `dataset=BOXRR-23`, `basis=explicit_user_instruction`,
`instruction`, `scope` and `recorded_on`. It stays outside the release.

```bash
python -m abc_iueh.boxrr_pilot /private/boxrr-pilot --authorization-record /private/operator_authorization.json
python -m abc_iueh.boxrr_pilot /private/boxrr-pilot --verify
```

The source is `cschell/boxrr-23` at revision
`af25be3ef76b176fbee0a094e82d97a611f9c950`. Each archive is checked against its
published size and LFS SHA-256 before decoding. The adapter reads the device's
declared axes, uses seconds from XROR and preserves source coordinates.
Existing filtering applies before the 20-minute session cap. Rates outside
85–110 Hz are rejected. The `>` threshold interpretation, 50 ms gap and tail
window choices remain explicit in the stored config. The official
20 samples / 100 Hz / 100 ms ambiguity remains open.

Real XROR input inspection found repeated initial timestamps (usually several
zero-time frames). The BOXRR adapter removes only adjacent repeats of the
initial timestamp, keeps the first row and records the count. These startup
rows fall inside the subsequent five-second trim. Interior duplicate or
backward timestamps still reject the session. No clock sorting is performed.

Implementation limits are 4 GiB per TAR, 64 MiB per XROR member and 16 million
retained samples per newly processed user. These caps can bias the selected cohort. A fixed
seed determines bucket order in the basic CLI. The current collection first reads
the pinned replay metadata and prioritizes users with at least an hour of broadly
rate-compatible recordings (80–115 Hz in metadata). It screens up to eight complete
sessions in the first 8 MiB of each TAR. A positive motion screen, or an incomplete
first session, triggers full archive download and SHA verification. Retained
sessions still require the narrower 85–110 Hz median-rate gate and all motion
filters. Private `selection.json` fixes the candidate order and screening policy.
This favors high-motion users; it is not a representative population sample.
The 2026-11-05 collection records a prospective efficiency amendment in
`screening_policy.json`: starting at fixed candidate index 2000, two prefix
sessions must pass when at least two complete sessions are available. One
available complete session needs one pass; zero complete sessions still trigger
full inspection. Earlier cached users retain their original one-pass policy.
This first amendment preserved candidate order and all retained-session filters. The
decision used observed intake yield, before any model fitting or validation.
The subsequent `resource_policy_v3.json` amendment preserves the original first
4,300 candidates, inserts 426 eligible official archives between 1 and 2 GiB
in the same metadata/seed order, then resumes the remaining original candidates.
It raises resource bounds only; motion, clock and rate filters are unchanged.
Under v3, the 339 completed immutable shards retained their original 4-million-sample
cap, including ten that reached it. New shards may retain up to 16 million samples.
The isolated expansion diagnostic was initially excluded. These are
implementation decisions made before model fitting, not organizer requirements.
The release includes both dated policy records; the latter supersedes resource
bounds and future candidate order. User-uniform training and equal-user
validation prevent larger shards from receiving greater user weight.
The later v4 resource amendment preserves the first 4,800 v3 candidates and
inserts 119 eligible official archives between 2 and 4 GiB, then resumes the
previous order. Archives above 2 GiB use an exclusive full-archive slot.
It also stages expansions for all ten earlier capped users, using the same
16-million-sample bound. Old versions remain available until each replacement
has passed integrity checks; a separate promotion receipt records any change
to the active corpus. The complete diagnostic may supply one replacement because
it did not hit its diagnostic cap and has fewer than 16 million samples.
Source, filters, split seed and user weighting are unchanged.
The later `screening_policy_v5.json` returns to a one-pass preliminary prefix
screen prospectively from candidate index 5100 of `selection_v4.json`, so that
more valid recordings from the remaining smaller archives can be inspected.
Earlier candidate decisions are unchanged. The full-source checksum and all
retained-session quality filters stay the same; this decision used no model or
validation outcomes. A bounded streaming probe reads no more than the same 8 MiB
and eight complete sessions; its decisions matched the former buffered probe on
three pinned official archives. These are intake choices, not challenge rules.
Quality selection stops after at least 30 users
and 6 GiB have been retained, with an 8 GiB bound. Every accepted user goes
entirely into one partition, assigned by hash order with seed 20261026.
The cohort is an exploratory pilot.

Each completed user gets an immutable shard and source receipt. A failed
intake can continue with the same settings. It rechecks completed users and
downloads the current interrupted TAR again. Raw archives are temporary.
The top-level `manifest.json` is published only after the target sizes and
user count are met. The manifest binds every shard size, SHA and partition,
plus each source TAR's path, published size and SHA for reproducing the cohort.

## Native Python Drive transfer

Use Google credentials with access to the particular files. The client accepts
`GOOGLE_DRIVE_ACCESS_TOKEN` or Application Default Credentials with the Drive
scope. Authentication material stays outside the release and logs. The Drive
folder must have only its owner as a permission entry.

The local uploader refreshes the existing operator login; credentials stay outside
the release. Python handles transfers and checksum readback. A `drive.file` token
from a different OAuth application may require an explicit grant for these files.

```bash
python -m abc_iueh.boxrr_drive upload --folder PRIVATE_FOLDER_ID --corpus /private/boxrr-pilot --toolkit BOXRR23_pilot_toolkit.zip --state /private/drive-upload-state
```

Upload is resumable at file boundaries and uses resumable transfer chunks
inside each file. It preallocates IDs, uploads each HDF5 and the toolkit,
downloads the bytes for SHA verification, then commits the index last.
Preserve the resulting index ID and SHA independently of Vast.

On Vast, install the toolkit, authenticate the download, then run:

```bash
python -m abc_iueh.boxrr_drive fetch --index-id INDEX_ID --index-sha256 RETAINED_SHA256 --destination /workspace/boxrr-pilot
```

Training reads local HDF5 through `abc_iueh.boxrr_pilot.windows(root, "train")`.
It verifies the manifest and yields segment-local `[400,4]` contexts and
`[20,4]` targets. Pass `"validation"` explicitly for evaluation.
Use a single iterator worker until worker sharding is implemented.

Downloads preserve verified completed shards in a `.partial` sibling directory
after interruption. Repeat the same command to resume. The final directory appears
only after all files verify. Training reads local SSD, with no per-batch Drive calls.

## Bounded training entry point

The toolkit contains model source without fitted weights. Install PyTorch for
the chosen target; the reviewed CUDA build is `torch==2.9.1+cu128` and needs a
compatible NVIDIA driver.

```bash
python -m pip install -r requirements-boxrr.txt
python -m pip install --no-deps -e .
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
MODEL=research_checkpoints/vast_20261031/abc_topic_discovery/workflow_runtime/staging/ordered_motion_model_v3/ordered_motion_model.py
python -m abc_iueh.boxrr_train --corpus /workspace/boxrr-pilot --model-source "$MODEL" --output /workspace/cpu-preflight --device cpu --steps 2 --batch-size 4 --max-seconds 120
python -m abc_iueh.boxrr_train --corpus /workspace/boxrr-pilot --model-source "$MODEL" --output /workspace/boxrr-run --device cuda --steps 1000 --batch-size 128 --max-seconds 300
```

The trainer samples users uniformly, then eligible windows uniformly within each
user. It fits only train users. Checkpoints bind model/optimizer/sampler state to
the corpus hash and are written atomically every 100 steps and at the time/step
bound. Add `--resume` with the same corpus and sampling settings to continue.
The time bound stops this process; it does not stop the rented Vast instance.

Validation uses a fixed sample per held-out user and reports endpoint20 mean
geodesic degrees with equal user weights, alongside zero-velocity and one-step
constant-angular-velocity baselines:

```bash
python -m abc_iueh.boxrr_train --corpus /workspace/boxrr-pilot --model-source "$MODEL" --output /workspace/boxrr-run --device cuda --evaluate --windows-per-user 256
```

This is sampled validation, not the organizer's unseen test set.

The CPU smoke test connects this loader to the pinned model, takes one
optimizer step and restores the checkpoint. A long-running BOXRR campaign
still needs a prospectively locked recipe, source-specific campaign bindings,
GPU measurements and the existing budget/stop/recovery controls. The earlier
WhoIsAlyx 18-fit IDs, split and signatures belong to that earlier experiment.

[Official challenge data](https://sites.google.com/view/9th-abc-challenge/home/the-data) ·
[Official mirror](https://huggingface.co/datasets/cschell/boxrr-23) ·
[XROR source](https://github.com/MetaGuard/XROR/blob/b2177253d97066038ab52bf59714f1966ece903e/xror/xror.py).

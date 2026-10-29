# Linux / Vast operations package v2 (staged)

Status: offline draft for review. No instance, account, data, credential, or GPU
was accessed to create this package. The latest prospective plan is the
three-arm draft at
`D:/ChatGPT/abc/abc_topic_discovery/workflow_runtime/staging/ordered_motion_three_arm_draft_20261029/protocol_draft.json`
(SHA-256 `93c8165f39c4b347275db86460cbdca583a25a4aea629444e9a47002bf0acd67`).
It says `DRAFT_NOT_REVIEWED_OR_AUTHORIZED`, `data_access_authorized=false`,
`gpu_authorized=false`, and `main_run_authorized=false`. It prospectively
replaces the four-arm draft only after review and lock; the four-arm history
remains unchanged. Neither the $20 cap nor the three-arm plan authorizes spend.

This package does not include the data or source permit. Keep training code,
model and data in their separately reviewed folders. Use an approved root
launcher for any data access or training. `run_guard.py` is a process watchdog,
not a source gate or cloud billing control.

On the target machine, set `OPS` to the reviewed copy of this helper folder and
keep the training working directory separate:

```bash
export OPS=/path/to/reviewed/vast_training_v1/ops
```

## Project-pinned runtime

- Linux, CPython 3.11, PyTorch `2.9.1+cu128`, NumPy `2.4.6`, SciPy `1.17.1`,
  and h5py `3.16.0`, copied from the project's locks. `cryptography==46.0.3`
  is additionally pinned for Ed25519 verification; its Linux resolver and
  runtime remain untested here.
- One visible RTX 3090, compute capability 8.6. The preflight rejects other
  GPU counts/models and checks RAM/VRAM against operator-supplied bounds.
- The two project lock files are installed together. Preserve the full
  `pip freeze --all` output and its SHA-256 for this environment. No runtime
  install or target Linux/CUDA validation has been done in this task.
- These files pin direct project dependencies, not a complete Linux transitive
  lock with wheel hashes: CUDA/NVIDIA dependencies and installer tooling are
  resolved at install time. Before relying on reproducibility, produce and
  review a complete Linux-specific, hash-pinned lock from the actual target
  environment; verify installation and `pip check` there. Freeze is evidence
  of what was installed, not a substitute for a prevalidated lock.

After separate run authorization, provision a fresh Linux environment with
Python 3.11 and enter a new, empty run-specific working directory; then run
`bash "$OPS/bootstrap_env.sh"`. Review and preserve the generated
`environment-freeze.txt` and its printed SHA-256 with the run's code/model/plan
hashes. Do not update packages mid-run. The setup script installs from the
configured indexes, but no install or network access was performed here.

## Before renting or uploading data

1. Obtain the approved root decision, exact code/data/model hashes, current
   source-use decision, independent review, and a separate GPU/main-run grant.
   Current drafts explicitly do not grant those actions.
2. Copy `budget-template.json` into a new run folder and fill every `null` from
   current provider pricing. Include boot/startup time, failed/retry attempts,
   minimum billing increments, attached and stopped-volume storage through
   expected cleanup, checkpoint copy-out egress, platform fees, and taxes.
   Set `storage.includes_stopped_instance_storage` and
   `egress.includes_checkpoint_copy_out` to `true` only after confirming those
   charges are included in the estimates.
   Include all charges already incurred against the same $20 cap, and classify
   already-paid final storage/export charges separately.
3. Treat **$16 as the soft stop threshold for new training**. Hold **$4** for
   necessary final fees, storage, transfer, or recovery, with a total campaign
   cap of **$20**. The $4 is not discretionary training budget;
   do not use it for extra fits, longer runs, or retries without a new user
   decision. The cap covers the full campaign; do not reset it for a retry or
   resume.
4. Run the offline estimate on the completed plan, then bind the offer and
   campaign ledger into the root controller's per-update gate:

   ```bash
   python3.11 "$OPS/budget_guard.py" budget.json --write-new budget-preflight.json
   ```

   `budget_guard.py` is arithmetic only. `budget_gate.py` denies the next
   optimizer update when quote/ledger inputs are missing or stale, or its
   conservative forecast reaches the $16/$4/$20 limits. The root controller
   must authenticate and bind the operator/coordinator's verified offer
   snapshot and cumulative ledger before calling `BeforeUpdateGate`; JSON
   fields alone are not authority.
5. Manually inspect the selected instance's on-demand/interruptible terms,
   current rate, minimum billing, storage while stopped, egress, fees, and any
   spend-limit coverage in the provider UI. Record a price screenshot/export
   date and the currencies/units used in `price_source`. The enabled tools do
   not include a verified Vast account billing connector, so this package uses
   a fresh root-bound offer and a local running ledger; no provider hard cap is
   claimed. If a live account-cost source becomes available, independently
   validate it and integrate it before relying on posted totals.
6. Only after the separate root grant, upload the exact approved code and data
   through the user's chosen channel. No SSH access is assumed here, and this
   package has no upload/download or API-key handling.

### What the $20 limit can and cannot mean

The forecast models compute with provider minimum-billing increments, attached
and stopped storage, egress, fees/taxes, retries, previous charges, and
separately estimated final storage/export. Those values come from a
coordinator/operator-verified offer and conservative remaining-time bounds.
Pause/stop new training before projected core spend reaches $16, or earlier if
projected final charges threaten the $20 total. The $4 reserve covers necessary
final charges/storage/export only; no discretionary use without a new user
decision. This soft gate is not a provider-enforced hard cap. Billing can be
delayed or include storage, transfer, minimums, failures, fees, or charges
posted later. Check current provider billing and stop the instance manually
before projected cost reaches $20. Stopping the OS or trainer may leave storage
billed; destroying a VM can make its local disk unavailable. No provider
account, spend, or resource was accessed from this package task.

The Q1 wall-time ledger is a separate budget. Existing project guidance
specifies 14,400 CUDA-context seconds with 150 seconds of historical debit and
1,800 seconds reserved; the USD threshold does not change those values. The
current screen's accounting against that ledger must be explicitly resolved in
the root decision. `budget_guard.py` handles USD estimates only, and
`run_guard.py` only limits one process group; neither replaces the shared live
CUDA ledger/watchdog or authorizes starting a run. Descendants that daemonize
into another process group/session are outside its containment guarantee; the
approved launcher must prohibit that or use a separately reviewed cgroup.

On-demand instances generally avoid interruption by spot reclamation but can
still fail or become unavailable; interruptible/spot instances can be reclaimed
with little notice and may have different rates or availability. Neither mode
changes storage/egress costs. Select a mode only after reading the current
provider terms. Store a complete resume checkpoint outside instance-local disk
before relying on any VM lifecycle operation.

## No-data machine smoke checks

These commands generate only tiny synthetic tensors and weights. Run them on
the target before uploading participant data. They do not authorize training.

```bash
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python "$OPS/vast_preflight.py" \
  --expected-peak-ram-gib 1.0 \
  --ram-headroom-gib 2.0 \
  --min-free-vram-gib 2.0
.venv/bin/python "$OPS/cuda_smoke.py" --persistent-dir /path/to/your/persistent/mount
```

Replace the expected peak RAM and VRAM values using a reviewed measured bound
for the actual training code. The example 1/2 GiB values are smoke-test
thresholds only; they do not describe the training workload. `cuda_smoke.py`
checks one RTX 3090, performs a single synthetic CUDA optimizer step, writes a
small generated PyTorch checkpoint to the selected mounted path, reloads it,
and leaves a unique receipt directory. Inspect the path and ensure it is
actually backed by the user's durable storage. A successful local readback
alone does not prove replication or durability after provider loss.

## Budgeted local process wrapper

Only the reviewed root launcher should call this after all gates have passed.
Do not paste a secret-bearing command or environment into a run. It does not
record argv or environment. The wrapper sends SIGINT at the monotonic time
ceiling, waits for a checkpoint-aware shutdown, then escalates to SIGTERM and
SIGKILL after the configured grace intervals. It does not stop billing or the
provider VM. The audit file is metadata-only and must be a new path.

```bash
.venv/bin/python "$OPS/run_guard.py" \
  --max-seconds 600 \
  --audit-new run-001.watchdog.jsonl \
  -- /approved/root/launcher --approved-run-manifest run-001.json
```

SIGINT or SIGTERM sent to the watchdog also starts the same group shutdown.
SIGKILL, watchdog crash, host failure, and detached descendants are outside its
guarantee. The audit explicitly reports detached sessions were not inspected.

The 600 second value is an example only. Derive the real ceiling from the fresh
budget preflight, startup and shutdown allowances, and the coordinator's shared
live ledger. Debit elapsed time on success, failure, startup, I/O, and retries.
Do not launch the trainer directly from this example command.

## Checkpoint and resume contract

The trainer must save a **full** state checkpoint atomically on the instance
disk: write a temporary file, flush/fsync, then atomically rename. Its JSON
metadata sidecar must follow `abc-training-checkpoint-v1` and include all
fields/true state components validated by `checkpoint_store.py`: model,
optimizer, sampler, Python RNG, NumPy RNG, Torch CPU RNG, and Torch CUDA RNG.
It must include run/plan/data/code/model hashes, fold, seed, arm, global step,
and total steps. The draft considered reusing a 100-update pilot prefix, but
the implemented pilot and screen have distinct purpose and bundle identities;
the runner therefore does not permit cross-purpose resume. Resume only an
exact matching verified full-state checkpoint within the same run.

Copy a finalized checkpoint and metadata to a user-selected persistent POSIX
mount (ext4/XFS with hard-link support) without overwriting anything:

```bash
.venv/bin/python "$OPS/checkpoint_store.py" copy \
  ./checkpoint.pt ./checkpoint.meta.json /path/to/your/persistent/mount/run-001
.venv/bin/python "$OPS/checkpoint_store.py" verify \
  /path/to/your/persistent/mount/run-001/checkpoint-<sha256>.manifest.json
```

The copy is published under its SHA-256; it never deletes the instance copy and
refuses different-byte collisions. It checks source stability, destination
bytes and metadata. If the mount lacks hard-link support, the copy fails
safely; use a separately reviewed storage path/tool rather than weakening the
no-overwrite rule. Record the emitted manifest path and hash. Keep the provider
instance until the durable copy has been verified and the next operator has
confirmed the file is readable from the user's storage. The local hash proves
byte integrity, not storage durability, ownership, or future availability.

For an automated post-checkpoint hook, the root launcher must call
`checkpoint_store.backup_checkpoint(payload, sidecar, destination, attestation)`
after every finalized checkpoint and deny further updates when it raises or
the CLI exits nonzero. Equivalent CLI:

```bash
.venv/bin/python "$OPS/checkpoint_store.py" backup \
  ./update-STEP-SHA.pt ./update-STEP-SHA.meta.json \
  /absolute/existing/remote-mount/run-001 /approved/storage-attestation.json
```

The attestation JSON requires `schema` = `abc-external-storage-attestation-v1`,
`external_storage_verified_by_operator` = true, `destination` = the exact
resolved absolute destination, and non-empty `storage_identity` and
`evidence_reference` strings. The hook refuses the same filesystem device and
verifies the copied bytes before returning. A different device and the
attestation are not independent proof of storage outside the VM; verify that
the mount is genuinely remote and readable from another host, and test loss and
recovery before relying on it. The launcher calls this hook synchronously for local-mount backups; it cannot
protect checkpoints that have not finished copying before host loss.

An rclone-backed route can copy a finalized payload and sidecar to a
user-configured remote prefix. It uses `copyto --ignore-existing --immutable`,
downloads each remote object and verifies its SHA-256 and size, then publishes
and verifies the manifest last. For example, from the parent staging directory:

```bash
python -m vast_training_v1.ops_v2.remote_backup backup \
  ./update-STEP-SHA.pt ./update-STEP-SHA.meta.json drive:private/run-001
```

The API is `remote_backup.backup_checkpoint(payload, sidecar, remote_prefix)`;
it returns the remote manifest path and hashes only after readback succeeds.
Failures and conflicting existing bytes raise an error; they are not success.
The launcher can select this route with `--backup-rclone-remote`; it calls the
API synchronously after checkpoint creation. No remote, credential or real
transfer has been configured or tested here. Verify access, storage ownership, independent
recovery and the provider's retention and costs before relying on it.

The prefix must have one exclusive writer. Rclone's general copy interface is
not an atomic conditional create across concurrent writers or external changes;
`--immutable` does not prevent deletion or guarantee backend object locking.
Backend duplicate-name semantics, eventual consistency, post-read modification,
and provider loss remain outside this check. Use provider-side write protection
and a reviewed single-writer procedure. Local temporary storage needs space for
the checkpoint snapshot plus a full readback; it handles opaque bytes without
loading a checkpoint as a model. Readback and egress consume time and bandwidth.
Record the returned manifest SHA independently, and validate it on restore. Do
not interpret a successful copy as proof of future availability or as a data-use
or transfer authorization. Rclone behavior: [copyto](https://rclone.org/commands/rclone_copyto/),
[immutable flag](https://rclone.org/docs/#immutable), and [cat](https://rclone.org/commands/rclone_cat/).

For rclone restore, obtain the manifest and payload SHA-256 pins and expected
run metadata independently from the controller receipt, then use a new private
directory on the next host:

```bash
python -m vast_training_v1.ops_v2.remote_backup restore drive:private/run-001 /private/new-restore \
  --manifest-sha256 MANIFEST_SHA --payload-sha256 PAYLOAD_SHA --expected /private/expected.json
```

The command downloads all three objects, checks both independent digests and
run pins, and prints the local manifest and resume paths. Supply those paths,
the payload pin and `--resume-remote-manifest-sha256 MANIFEST_SHA` to the
launcher; it re-reads remote objects before resume and after model restoration.
The receipt is locally generated within the training process, so preserve its
pins outside that host. It does not by itself prove external authorization.

Before resuming, transfer the frozen expected pin JSON to the instance and run:

```bash
.venv/bin/python "$OPS/checkpoint_store.py" resume-check \
  /persistent/run-001/checkpoint-<sha256>.manifest.json \
  /approved/run-001.expected.json
```

This checks hashes and identity pins without deserializing Torch data. The
trainer must then restore the checkpoint in a clean process and write an
`abc-resume-restored-v1` receipt only after verifying loaded global step and
model, optimizer, sampler, and RNG states. Validate it with:

```bash
.venv/bin/python "$OPS/checkpoint_store.py" verify-restored \
  /persistent/run-001/checkpoint-<sha256>.manifest.json \
  /approved/run-001.expected.json \
  ./resume-restored.json
```

That receipt confirms the trainer's own declared restore checks; it is not an
independent proof that arbitrary trainer code behaved correctly. Never resume
with changed code, data, model, fold, seed, arm, sampler, optimizer, or protocol
pins. A resumed run remains charged to the same $16 training threshold / $20
campaign total, and to the separate Q1 wall-time ledger if applicable.

## Stop, interrupt, and backup procedure

- On a planned stop, ask the trainer to finish the current update and write its
  full-state checkpoint; use the wrapper's SIGINT grace period. Confirm the
  trainer has exited and verify/copy the checkpoint before any instance change.
- On interruptible instances, assume the process can disappear at any moment.
  A frequent checkpoint policy must be locked in the reviewed training
  protocol and its save overhead included in the time/cost projection.
- Use the provider's UI to stop the instance, then confirm its displayed state.
  Do not assume OS shutdown stops billing. Check whether attached storage still
  accrues charges. Check projected and posted charges before resuming.
- Do not destroy the instance or remove any directory as cleanup unless the
  user separately asks for that action after verifying a durable backup.
- For a provider/runtime failure, debit all elapsed and billed time, including
  startup, failed attempts, and storage. Do not silently retry or start a new
  instance if either the USD budget or wall-time ledger would fail. Never treat
  the $4 final-charge reserve as discretionary training budget.

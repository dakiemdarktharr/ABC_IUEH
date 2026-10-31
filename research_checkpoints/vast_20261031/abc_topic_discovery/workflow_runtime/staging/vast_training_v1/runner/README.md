# Current audit policy — 2026-10-31

Use the new dated runtime. The user has superseded the historical budget with
**$4 all-in soft stop / $16 hard ceiling**. Read `../policy.json` and
`../ops_v2/README.md`. The $12 difference cannot fund extra training without a
new user decision. Old v7 runtime/runner/model pins are stale after the audit.

The planner now requires `per_fit_overhead_seconds` to cover initialization,
all checkpoint copies and post-fit remote verification, in addition to a
measured upper bound for seconds/update and the shutdown allowance. The full
5,000-update fit must fit one reservation, and all 18 reservations must fit
the reconciled time ledger. This is a fit-only screen, not a complete research
result: human validation remains disabled in the historical evaluation draft.

# Staged single-fit launcher

This is a single-fit entrypoint for a future Linux/CUDA run. Three fit-only
bundles were subsequently prepared and verified locally; no paid instance,
real training or transfer was started while preparing it. A signed
bundle/key establishes an integrity binding, not eligibility, data-use rights,
transfer approval or provider spending limits. Review the source and the
selected plan before making a run decision. The horizon remains 20 samples;
the official 20-sample / 100 Hz / 100 ms inconsistency is unresolved.

From the parent `staging` directory, use the package name (do not execute the
file directly):

```bash
python -m vast_training_v1.runner.launcher --help
python -m unittest vast_training_v1.runner.test_launcher
```

Provide an independently stored raw Ed25519 public key and key ID; never use a
key supplied by the bundle as its trust anchor. Supply the independently
recorded bundle identity, runner hash, and arm-specific model hash. To inspect
the current hashes without loading Torch or any recordings:

```bash
python -c 'from vast_training_v1.runner import trainer as t; print("runner", t._runner_sha256()); [print(a, t._model_binding(a)[1]) for a in t.SCREEN_ARMS]'
```

Record these values outside the VM alongside the reviewed code and plan. The
runner hash binds the launcher, reader/verifier, budget gate, checkpoint store
and trainer sources; each model hash binds its exact source and loss. The
loader verifies the signed bundle, identity, fold, role and purpose before
training. `pilot` and `screen` are separate purposes and require separate
bundles. The `pilot` accepts fold 0, seed 20261026 and `ordered_product` only;
the fixed `screen` accepts three arms, three folds and two seeds. This launcher
executes **one** selected fit; it does not schedule all 18 fits or select a
winner. It does not evaluate validation5, old development5 or final15.

Before copying the configuration to a training host, generate the 18 run IDs
on the user's machine, save them privately and independently of the host, and
put the same mapping in `run_ids` in the campaign configuration. For example,
from staging, with an existing private destination directory:

```bash
python -c 'import json,os; from vast_training_v1.runner.campaign_plan import generate_run_ids; f=os.fdopen(os.open("/private/run-ids.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"w"); json.dump(generate_run_ids(),f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close()'
```

Retain that file outside the training host. Use the same ID for every resume of
the same fit; do not generate replacement IDs after an interruption. Then fill
every field in a copy of `campaign-template.json` using target Linux paths,
and run:

```bash
python -m vast_training_v1.runner.campaign_plan /path/to/filled-campaign.json /path/to/new-command-plan.json
```

The manifest lists the 18 fit IDs in fixed order and contains no commands.
Generate a separate manifest for one selected fit immediately before its run:

```bash
python -m vast_training_v1.runner.campaign_plan /path/to/filled-campaign.json /path/to/new-fit-command.json --fit fold-0-seed-20261026-ordered_product
```

The command uses the shared time-ledger and anchor paths. It does not freeze an
anchor head: the launcher reads the current anchor under lock and creates a new,
permanent reservation for each process. This command is a generator. For the
separate sequential executor see `campaign_executor.md`; it verifies each
completed fit and backup before proceeding. Between
fit processes, the operator must update the shared ledger with cumulative
lease elapsed time and all campaign charges; otherwise each new process could
underestimate total spend. Do not rerun an existing fit as a fresh run: use
the verified backup and resume procedure below. The executor can reverify and
skip completed fits; it does not automatically retry interrupted fits.

Pass all arguments listed in `--help`, including a fresh, independently
verified `vast-verified-offer-v1` quote, a current `vast-campaign-ledger-v1`
ledger, and a conservative measured upper bound for seconds/update. The ledger
is reread on every update and must include prior campaign costs and a current
lease elapsed time. It must be no more than 60 seconds old at launch. A local
monotonic meter advances the reported elapsed lease time on each update, while
the operator must refresh the campaign ledger at least hourly and verify all
external charges. The quote is reread each update and expires after one hour;
refresh it from a newly verified provider offer before expiry. Missing or stale
entries stop training;
the local forecast cannot enforce provider billing. The $4 all-in soft stop
and $16 all-in hard ceiling are shared across all fits and retries; the $12
headroom is not automatic spending authority. The
separate Q1 wall-time ledger/anchor and Linux supervisor are now integrated;
see `../time_budget/README.md` for required flags, accounting and limitations.
Linux fork/signal tests must pass in CI and on the target host; CPU CI does not establish CUDA readiness.
Runtime/driver compatibility, time bounds and actual Vast pricing have not been
measured or verified here. Existing manifests made with the previous command
generator omit required time-budget arguments and must not be executed.
The campaign config must specify an absolute `.venv/bin/python` path; the
executor checks that it is executable and has a `pyvenv.cfg`. The executor's
boot-bound shared lease clock preserves the elapsed USD estimate between fit
processes and rejects a reboot or regressed ledger. This local clock is not a
provider billing feed or a hard spending cap.

Select either `--backup-dir` (an existing directory on a different filesystem,
with `--storage-attestation` following `ops_v2/README.md`) or a configured
`--backup-rclone-remote drive:private/run-001`. Rclone preflight lists the
configured remote; no credentials are passed in arguments or recorded in
receipts. The remote prefix requires a single exclusive writer and verified
provider-side durability. At update boundaries, checkpoint after at most 1,000
updates or about 5 minutes since the last completed checkpoint, at fit
completion, and when the budget gate denies an update between
checkpoints, the controller synchronously copies and verifies the complete
checkpoint and sidecar before any further optimizer update. Failure stops the
fit. An interrupt or host loss can discard up to 1,000 updates since the last
completed checkpoint, including an update followed by an interrupted backup;
a slow update or upload can extend the 5-minute bound. A different
filesystem plus an operator attestation does not prove off-VM durability.
Verify actual recovery from another host. Output lists verified backup paths
when the fit returns normally; backup manifests also remain on the selected
mount if the process stops early. The controller prints a local checkpoint
receipt after each verified backup; retain the digest and run pins outside the
host. A receipt is a process-local claim, not an independent signature.

For resume, supply the original `--run-id`, `--resume`, a separately pinned `--resume-sha256`, the copied
backup `--resume-manifest`, an independently held `--resume-expected` pin file,
and a new private `--resume-receipt` path. The controller checks the backup
before deserialization, the trainer restores model, optimizer, sampler and RNG
state and writes a receipt, and the controller validates that receipt against
the backup before the next update. A receipt is the trainer's own report, not
independent proof of arbitrary code behavior. Checkpoint/runtime pins must
match, including the same data identity and all code hashes. For rclone, first
restore the three objects with independently retained manifest/payload pins as
shown in `ops_v2/README.md`; add `--resume-remote-manifest-sha256` for the
launcher to reverify them remotely before updates. The runtime compatibility
binding pins OS family, architecture, Python, NumPy, PyTorch, CUDA, cuDNN,
device capability and deterministic settings, not host kernel or VRAM size.
It does not promise bitwise replay on a different host. Code and runtime pin
changes make old checkpoints fail closed; do not silently bypass this check.

## Code-only copy

Build a new local directory with the exact relative tree expected by both the
runner and the raw-vector adapter (from the parent `staging` directory):

```bash
python -m vast_training_v1.assemble_runtime /path/to/new/code-only-directory
```

The explicit allow-list copies code, pins and no-data tests only. It writes a
`code-only-manifest.json` of SHA-256 hashes, includes no recordings, bundles or
weights, and does not upload anything. Inspect the manifest and verify hashes
after transfer; run the launcher from its copied
`abc_topic_discovery/workflow_runtime/staging` directory. Transfer any approved
fit-only bundle separately, with its independently pinned public key and
identity. The assembler does not copy the local-only source adapter or export
tool, nor does it build a validated Linux dependency lock.

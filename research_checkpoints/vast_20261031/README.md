# Vast runtime audited 2026-10-31

Current policy: **$4 all-in soft stop, $16 all-in hard ceiling** for the whole
campaign, including setup, retries, storage, transfers, fees and cleanup. The
$12 headroom is not automatic permission for more training. This replaces the
2026-10-29 budget; the old snapshot and experiment draft remain historical.

Read the [audit](../../docs/vast_readiness_2026-10-31.md) before renting. This
code is prepared for target-host verification. No paid rental, human-data fit
or final-holdout evaluation was performed by the audit. A full 18-fit campaign
is still blocked by target-host evidence and the unfinished human validation
path. CPU/synthetic checks do not establish model quality or CUDA readiness.

The 74 files in `code-only-manifest.json` form a self-contained runtime tree.
Git preserves their exact bytes via `.gitattributes`. The runtime contains no
recordings, signed human-data bundles, credentials, private trust pins or
trained weights. The existing fit-only data and keys remain separately held.

From `abc_topic_discovery/workflow_runtime/staging` in this tree, use a Python
3.11 environment with the recorded numerical dependencies, cryptography and
PyTorch, then run:

```bash
python -m vast_training_v1.run_checks
```

This verifies the manifest, parses the shipped Python, disables CUDA for the
tests and runs generated fixtures. Linux-only process and timer checks run in
CI; Windows skips them explicitly. The production target additionally needs
the pinned CUDA runtime, actual GPU and remote backup/recovery checks.

Read `vast_training_v1/ops_v2/README.md`, `policy.json` and
`runner/campaign_executor.md`. Before a real campaign, fill the null fields in
the templates from measured target throughput, initialization/backup overhead
and a fresh actual provider offer. `per_fit_overhead_seconds` is now required.
The planner checks that 5,000 updates plus all I/O and shutdown fit a reservation;
the executor checks all remaining reservations against the cumulative budget.

Recompute runner/model pins from this exact tree and retain them independently:

```bash
python -c 'from vast_training_v1.runner import trainer as t; print("runner",t._runner_sha256()); [print(a,t._model_binding(a)[1]) for a in t.SCREEN_ARMS]'
```

Old v7 pins and old checkpoints cannot be silently substituted. Corrected
geometric models now use the fit seed for initialization, with candidate/control
paired within a seed. Already signed fit-only bundles need no source re-export
because their data identity is unchanged. Never regenerate run IDs or reset
time/cost history to retry a fit.

Start only through the Linux campaign executor with a verified, mandatory
`--stop-instance-id`. A whole-campaign timer bounds startup, uploads, waits and
verification as well as fitting. SIGTERM/timeout/normal exit request a stop;
an accepted CLI request does not prove billing stopped. Provider-side control
and a separately verified off-host recovery procedure remain necessary.

The official **20 samples / 100 Hz / 100 ms** inconsistency remains unresolved;
the static `400 -> 20`, wxyz, endpoint20 contract is preserved. The old dev5
and final15 remain excluded. BOXRR and Alyx-only competition eligibility retain
their separate source/organizer gates. The 18-fit screen is exploratory and
does not fulfil the original stronger research objective by itself.

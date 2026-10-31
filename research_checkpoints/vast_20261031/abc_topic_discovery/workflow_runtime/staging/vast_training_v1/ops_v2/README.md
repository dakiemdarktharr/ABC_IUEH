# Vast operations — current policy, 2026-10-31

The user sets a **$4 all-in soft stop and $16 all-in hard ceiling** for the
entire campaign. Historical $16/$4/$20 guidance is superseded. The $12 gap is
non-discretionary headroom; there is no automatic permission to spend it.
Setup, retries, resumed fits, storage while stopped, egress, fees and cleanup
all count. A JSON estimate is not provider billing enforcement.

## Before any paid screen

1. Use the new dated runtime and independently retained runner/model/data/key
   pins. Keep recordings, credentials, attestations and weights out of Git.
   Existing signed fit-only bundles need no re-export when only runner pins
   change. Older checkpoint/code pins cannot be silently reused.
2. Read `../policy.json`. Freeze a lawful source-use decision and evaluation
   protocol. BOXRR stays disabled and the old dev5/final15 stay sealed.
   The human validation source in the historical draft is still disabled;
   complete that evaluation path and its review before a full paid screen.
3. On the actual Linux host, provision CPython 3.11, then run
   `bash /absolute/path/ops_v2/bootstrap_env.sh` in a fresh working directory.
   The direct pins are PyTorch 2.9.1+cu128, NumPy 2.4.6, SciPy 1.17.1,
   h5py 3.16.0, cryptography 46.0.3. This is not a wheel-hash transitive lock.
   Preserve `pip check`, the full resolved freeze and its hash. Validate the
   driver and exactly one RTX 3090 CC 8.6 using `vast_preflight.py`.
4. Run `python -m vast_training_v1.run_checks` from the copied staging folder.
   It verifies every manifest hash, parses Python and runs generated CPU tests.
   Linux process-group tests run in a clean interpreter before Torch import.
   `cuda_smoke.py --persistent-dir /private/mount` adds a generated CUDA
   step/local readback on the target. Neither proves off-host recovery.
5. Exercise rclone checkpoint AND time-journal recovery on that host with
   generated fixtures. Keep trust pins independently off-host. Use one writer
   per prefix; Drive/rclone do not supply atomic cross-host locks or CAS.
6. Obtain a fresh actual offer and reconcile every existing campaign charge.
   Complete `budget-template.json`, `offer-template.json` and
   `usd-ledger-template.json`; null/false defaults deny a run. Price increments
   and minimum billable time round upward. The full 18-fit forecast, including
   all process reservations and backup I/O, must remain below $4.
7. Reconcile the original Q1 time ledger and independent anchor: 14,400 seconds
   total, non-export at most 12,600, with the historical 150-second debit
   preserved. A whole-process reservation is permanently charged on every
   attempt. The planner rejects a full fit or 18 reservations that cannot fit.
8. Install and authenticate the official `vastai` CLI on the operator host;
   verify its ability to stop the exact rented instance before training. It is
   deliberately not installed by the numerical-runtime bootstrap. Use the
   executor with mandatory `--stop-instance-id ID`. SIGTERM, timeout and normal
   exit request a provider stop. A missing CLI or rejected request is a failure.

## Checkpoint and recovery

At most 1,000 updates or about five minutes between checkpoint boundaries,
plus the final step and cooperative budget stop. Slow steps/uploads can extend
that interval. Save model, optimizer, sampler and Python/NumPy/Torch CPU/CUDA
RNG state. The synchronous backup publishes payload and metadata, verifies
remote SHA-256/size, then publishes and verifies the manifest last. Any failure
stops updates. Local durable copies require POSIX directory fsync and a verified
external filesystem, not just another path on the same VM.

For rclone use `python -m vast_training_v1.ops_v2.remote_backup --help`.
Restore into a new private directory with independently held payload/manifest
hashes and expected run pins. The launcher checks exact run, plan, data, code,
model, recipe and runtime before deserializing with `weights_only=True`.
Do not reset IDs or ledgers after failure. The executor never automatically
retries an incomplete fit. Every continuation re-verifies completed backups.

## Billing limit

The $16 ceiling is a user constraint. Neither a local gate nor a $16 account
balance establishes a hard provider cap. Vast bills stopped storage, and a
saved payment method can cover negative balances. Check actual provider state
and all remaining commitments; independent off-host control is still required
for host/network failure. No provider-enforced cap has been verified here.
Do not start the full paid screen while these target checks remain unresolved.

Primary sources checked 2026-10-31:
- https://github.com/vast-ai/docs/blob/main/guides/reference/billing.mdx
- https://github.com/vast-ai/docs/blob/main/guides/instances/manage-instances.mdx
- https://rclone.org/commands/rclone_copyto/
- https://rclone.org/docs/#immutable

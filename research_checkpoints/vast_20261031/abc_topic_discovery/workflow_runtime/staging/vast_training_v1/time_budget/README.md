# Q1 CUDA wall-time control — staged, not a run grant

The original `workflow_runtime/budget_ledger.jsonl` is never modified by this
component. Its reviewed 19-row snapshot is pinned by SHA-256
`1a38701d8e884cfa654f7165bef9b67bdbf97ca9a95a06799890aba0d420cb75`
and head `726413b0ef42c3cc01a8f708ee65a6c55ff18cfcd970eb971ad367bf7fc45913`.
It contains one conservative 150-second non-export debit, not an exact CUDA
measurement. A changed original stops initialization and authority creation.
The separate `q1_classical_projection/cuda_wall_ledger.jsonl` is not this history.

`ledger.initialize_ledger` explicitly creates a new file once, without replacing
anything, copying the exact pinned history. No real ledger is initialized by
these tests. `ReservationAuthority` takes an independently stored current file
SHA and head, verifies the prefix and chain under a persistent OS file lock,
and pre-debits whole seconds. `bootstrap.AnchoredReservation` also appends the
new ledger SHA/head to a separately configured append-only anchor, fsyncs it,
and keeps both locks held. An anchor-write failure consumes the reservation and
blocks the next attempt; it does not roll the debit back. A directory difference
is not proof of independent storage or trustworthy authentication. It holds the lock through the supervised process
lifetime; a second cooperating process using the same ledger cannot reserve.
The total cap is 14,400 seconds and the non-export cap 12,600 seconds; export
reservations count toward the total. Each ID is unique, each authority issues at
most one reservation, and failure, interruption, restart and early exit never
refund it. A resumed process requires a new reservation. No refund/delete/reset
API is provided. An interrupted partial write, stale anchor or changed file fails
closed and requires human reconciliation preserving all evidence. Creation and
writes are fsynced; POSIX creation also fsyncs the parent directory.

Initialize the new ledger and independent anchor once only after reconciling
all usage. The Linux launcher now requires these paths, a whole-process wall
reservation, shutdown/signal allowances and externally attested backup storage.
It appends the reservation and anchor before forking. The parent retains the
locks, monitors the deadline, sends SIGINT then TERM/KILL as needed, and checks
that the original process group is empty; the child receives both time and USD
gates before opening the bundle. Grace periods must fit inside the shutdown
allowance. A cooperative SIGINT is checked before each update so the trainer
can checkpoint; an interrupted or stalled operation may still be terminated.
The timer starts on launcher entry and therefore excludes interpreter/import
startup; budget this unmeasured overhead conservatively. A failed process still
spends the whole reservation. Exports are reserved separately; there is not yet
an export-process gate here.

**Incomplete controls / NO-GO:** no protected external anchor store or
authoritative single ledger deployment has been provisioned. The existing
`time_budget/supervisor_linux.py` reuses process-group helpers from
`ops_v2/run_guard.py`, but this fork/signal path has generated-only Linux CI coverage. The target
Linux/CUDA runtime and provider lifecycle remain unverified. The runtime assembler includes this package, but older
snapshots omit it and must not be used with the modified trainer. The file lock is advisory/cooperative;
an actor that bypasses it, chooses another ledger or writes the old ledger, or a
lost host and detached process can invalidate global accounting. A timer thread
can be delayed or die. This is not proof that a process exits before its ceiling,
nor protection against privileged tampering, clock/runtime faults, provider
billing or delayed fees. The separate USD $4 all-in soft stop / $16 hard ceiling ($12 non-discretionary headroom)
forecast remains necessary and is not a provider-enforced spend limit. No real
CUDA, Vast lease, human data or training is authorized by these components.

## Off-host time journal for rclone

When using `--backup-rclone-remote`, the launcher also requires
`--time-journal-remote` (an existing single-writer rclone directory) and
`--time-journal-head` (a separately retained local head file). Initialize a
new empty head only after reconciling the canonical ledger and confirming the
remote journal directory is empty:

```bash
python -m vast_training_v1.time_budget.remote_journal init-head /private/time-journal-head.jsonl
```

Each reservation is permanently debited and anchored locally; before fork the
launcher uploads content-addressed copies of the full ledger and anchor plus a
linked journal manifest, downloads/verifies every SHA, checks for a unique
visible chain head, and appends that head to the local head journal. The
launcher prints `time_journal_published_sha256` before fork. Retain this hash
independently off-host; neither this log nor a same-host file is an independent
anti-rollback authority. A failed upload or crash can leave a local reservation
or remote objects without an updated head; the next launch fails and requires
reconciliation, not a reset or refund.

On a new host, provide the independently retained latest manifest SHA and the
original pinned history, and restore only into new paths:

```bash
python -m vast_training_v1.time_budget.remote_journal recover --remote drive:private/time-journal --expected-head SHA256 --original /private/original-ledger.jsonl --ledger /private/new-ledger.jsonl --anchor /other-private/new-anchor.jsonl --head /private/new-journal-head.jsonl
```

Recovery verifies the visible journal chain, every ledger/anchor digest and
append-only relationship, and rejects a mismatched or ambiguous head. It does
not create the original history. A single cooperating writer must own the
remote prefix across all hosts. Rclone/Drive listing can be eventually
consistent and its generic copy does not provide atomic cross-host locking or
compare-and-swap; deleted/hidden newest entries, a stale independent pin, host
loss or another writer can defeat freshness. Stop and reconcile ambiguous
states; this is not a provider-enforced hard cap. No real Drive access was
performed in testing.

Synthetic stdlib tests:

```bash
python -B -m unittest vast_training_v1.time_budget.test_guard vast_training_v1.time_budget.test_ledger vast_training_v1.time_budget.test_supervisor_linux -v
```

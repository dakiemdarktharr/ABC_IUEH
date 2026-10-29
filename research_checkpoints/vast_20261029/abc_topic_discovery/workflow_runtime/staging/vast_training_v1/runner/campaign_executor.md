# Sequential screen campaign executor

`campaign_executor.py` executes the 18 fixed screen fits in planner order. It
is code, not permission to train, use data, rent a GPU or spend money. Read the
Q1 time-budget and backup instructions and complete the independent launch
and data gates first. The Linux supervisor, real provider billing, and actual
remote backup have not been verified by this synthetic test suite.

Fill the planner configuration with the current reviewed pins, shared CUDA
ledger and anchor paths, reservation and shutdown allowances, USD quote/ledger,
an absolute `.venv/bin/python` path, and either a local external backup
configuration or an rclone remote. Run only
from the assembled runtime's staging directory:

For rclone, replace `backup_dir` and `storage_attestation` with
`backup_rclone_remote`, `time_journal_remote`, and the POSIX path
`time_journal_head`. The head file must be explicitly initialized and
reconciled before launch; see `../time_budget/README.md` for recovery and the
single-writer/freshness limitations.

```bash
/absolute/path/.venv/bin/python -m vast_training_v1.runner.campaign_executor /path/to/campaign.json /private/campaign-state
```

The executor creates a private state directory, holds a cooperative campaign
lock, and writes an exclusive start marker before each launcher. It also
anchors a shared USD lease clock to the current boot and fresh ledger snapshot,
including snapshot age, so successive launcher processes count elapsed lease
time and retain observed operator debit increases. Boot changes, clock or debit
regressions, and clock-head mismatches fail closed. The configured interpreter
must exist, be executable and belong to a directory containing `pyvenv.cfg`.
It runs one launcher at a time; each launcher independently verifies current time/monetary
budget, appends a permanent reservation, and loads only its selected fold. A
successful exit alone does not advance the schedule. The executor checks the
supervisor receipt, 5000 updates, fit/code/data/model pins, final checkpoint,
controller's matching checkpoint receipt, and backup bytes and metadata.
For rclone it restores a temporary verified copy using the local controller's
pre-upload pins and rereads the remote manifest, payload and metadata. Those
pins are process-local consistency evidence, not a signed independent off-host
attestation. The verification itself uses network traffic and storage and may
incur costs; account for it in the separate USD budget. No raw provider output
or credentials are copied into executor receipts.

A nonzero exit, missing/mismatched receipt, backup failure, stale budget,
interruption, or incomplete start stops the campaign. Failed or incomplete fits
are not retried automatically. After a clean pause between fits, the same exact
config may be resumed using `--continue-verified`; all earlier success logs and
backups are reverified before skipping. Missing/altered logs, noncontiguous
receipts or a changed config cause a stop. This does not resume a partially
trained fit; separately reconcile the ledger, checkpoint and locked plan before
any further action. Preserve state directory and external checkpoint evidence.
The operator must still refresh the USD ledger and provider quote as required;
the clock does not query billing. A reboot or an unavailable clock history needs
manual reconciliation and cannot safely be resumed automatically.

Optional `--stop-instance-id 12345` requests `vastai stop instance 12345` on
normal completion, a handled error or keyboard interrupt. The ID must be a
positive decimal. The CLI runs without shell interpolation and provider output
is suppressed. An accepted CLI request does **not** prove the instance is
stopped or billing has ended; confirm provider state separately. Stopped volumes
may still incur charges. A killed executor, host failure, CLI failure or loss of
network can prevent this request, so this is not a hard spend cap or a substitute
for an independent provider-side control.

Synthetic checks (all subprocess/provider operations mocked):

```bash
python -B -m unittest vast_training_v1.runner.test_campaign_executor -v
```

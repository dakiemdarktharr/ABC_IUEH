# Q1 fit-pool lazy loader (v3 draft)

This is an unapproved lazy fit-pool loader with two separately locked future
purposes: a train-only timing prefix and CPU-only standardizer preparation. It
keeps full fit-user quaternion segments in bounded RAM arrays and builds
windows in batches. It does not train models, perform optimizer updates, launch
experiments, expose a CLI, or evaluate outer-validation users. The fixed new
plan and coordinator approval files are absent; production access remains
blocked and Q1 remains **NO-GO**. A production source load also requires a
separate trusted-parent launch capability; local plan/approval JSON alone does
not authorize a source read.

## Fixed boundary

The loader verifies the frozen bounded reader source against its accepted
review pin before importing it:

- reader source SHA-256:
  `2b499f28199f91d0a586fc3611964a16a943c0b206d53eb899ad79a367a32149`
- bounded-reader review SHA-256:
  `8c2e569341412a0bc5e74bbaccd51c3c7b95c8466b057418f4e590a2e76be8dd`

It reuses that reader's metadata-only receipt, origin-map/index, source-path,
split, ownership, and fixed path/inode identity checks. A distinct new
plan and approval are required at these fixed private paths:

- `ABC_IUEH/data/research_controls/q1_fit_loader_v3/fit_pool_plan.json`
- `ABC_IUEH/data/research_controls/q1_fit_loader_v3/fit_pool_approval.json`

The plan must freeze the source and origin receipt chain, split and fold
digests, all three outer folds, role mode, purpose, pipeline and reviewed code
hashes, a complete embedded recipe document and its canonical SHA-256, NumPy
version, sampler versions, standardizer sample, window cap, batch size, logical
read bound, pose-array cap, and loader time cap. Timing-prefix plans also lock
the two exact training seeds and five canonical arms. CPU-preparation plans
instead bind the fixed consumer source hash, allowed operation, CPU/zero-update
policy and CPU resource limits. The approval binds exact plan bytes and all
reviewed artifacts. Validation finishes before the source file is checksummed
or opened. This worker did not create either file.

## Trusted parent launch boundary

Production source loading is currently restricted to
`fit_only_cpu_preparation` / `outer_fit10`. Timing-prefix and inner-diagnostic
plans may still be validated as metadata, but their source-loading scopes stay
blocked pending separate review. The metadata-only
`preflight_approved_fit_pool(fold_id)` API does not require the launch
capability and never opens or checksums source HDF5.

Before a CPU-preparation source load, the coordinating root tool must launch
the process with `ABC_Q1_TRUSTED_CPU_PREP_LAUNCH` set to an exact JSON envelope
binding purpose/role, plan and approval byte hashes, recipe and independent
review hashes, actual loader and consumer code hashes, a unique `launch_id`,
and timezone-aware `expires_utc`. The loader verifies those pins against the
validated fixed documents and current code bytes. Expiry must be future-dated
within 900 seconds. Missing, malformed, forged, expired, wrong-scope, or
code-mismatched envelopes fail before source checksum/open. There is no
filesystem fallback, and the loader does not generate launch envelopes.

At the first production load, the loader creates an exclusive plan-SHA lock
under the ignored `data/research_controls/q1_fit_loader/launch_attempt_locks/`
directory. A process may attempt folds 0, 1, and 2 once each. Before any source
checksum or HDF5 access, each fold attempt also creates an immutable exclusive
`plan_<sha>.fold_<id>.claim.json` file. The claim binds the plan, approval,
recipe, launch and expiry to the parent plan-lock's device/inode/size/mtime and
content hash. The exact normalized UTC expiry is fixed by the first claim;
later folds must present that same expiry, and the parent lock document is
checked against it before each claim. Both plan lock and fold claims are
flushed and fsynced and are never removed or reset. In-process state binds the
lock-root path and directory identity, and verifies the parent lock and earlier
fold claims before adding another claim. Symlinks and Windows
reparse points, including junctions, are rejected along the lock path. The plan
lock remains after a failed attempt, preventing the same plan from being
relaunched by a later process. Tests inject temporary lock directories only.

This environment envelope is a trusted-caller capability, not a cryptographic
signature or OS identity proof. The root tool context must verify reviews and
issue it. The loader binds its values to actual plan, approval, recipe, loader,
and consumer bytes, preventing project-local JSON or changed code from
self-authorizing. It cannot prove which OS process set the environment
variable or protect against a malicious same-OS user who can read arbitrary
files, run commands, or remove the ignored lock. No stronger ACL or
cryptographic boundary is claimed.

Fixed plan/approval parsing rejects duplicate keys and nonfinite numbers at
any nesting level, including exponent overflow, before field validation.

The production timing mode is `outer_fit10`: ten users from the specified
outer-fit set of the 15 train users, including the two selector users used by
the predeclared inner split. The explicitly declared diagnostic mode,
`inner_fit8_diagnostic`, uses the remaining eight inner-fit users and excludes
those selectors. Neither mode can access the fold's outer-validation users,
development users, or final users. The loader accepts only a fold ID already
listed in the fixed plan; `bool` is rejected as an integer. There is no switch
that enables outer validation.

## Memory and window access

Before allocating pose arrays, the loader reads only selected fit-role HDF5
metadata and verifies every selected segment's local hard links, unique object
identity, chunked local storage, shape, and authenticated `N - 419` origin
count. It checks the total float32 quaternion allocation against the approved
cap and the hard 128 MiB limit before the first HDF5 value slice. It then loads
only selected fit-user quaternion segments in contiguous logical slices of at
most 1 MiB into read-only arrays. Timestamp chunks are checked for finite,
strictly increasing 100 Hz grid steps; quaternion chunks are checked for
finite unit norms and chronological hemisphere continuity. Timestamps are
discarded after validation. No window tensor is materialized for the full pool.

Training schedules use `default_rng(SeedSequence([training_seed, fold_id]))`
for the pinned NumPy global-uniform sampler with replacement, with ranks
generated over each fold's canonical fit-origin pool. For a seed and fold, the
same sampled ranks are shared across the five arms so arm comparisons see
identical origins. The fixed plan specifies three folds, training seeds
`[20261026, 20261027]`, canonical arms `[raw_vector, derivative_vector,
same_feature_vector, fixed_weight, learned_weight]`, and a bounded window
count for each schedule. Batches are at most 128 windows; each batch
contains a read-only `[B,400,4]` context and separate read-only `[B,20,4]`
target. `model_inputs` returns context only; `endpoint20` returns the last
target sample. `clone_contexts_for_augmentation()` returns a writable copy.
Origin ranks map to the immutable accepted global index and may not cross
segment boundaries.

For a fit-only standardizer review, the loader creates exactly 65,536 distinct
global origin ranks per fold without replacement using
`default_rng(SeedSequence([20261028, fold_id]))`. It exposes a schedule; it
does not compute fitted statistics.

## Separately locked CPU preparation

`fit_only_cpu_preparation` is a separate plan purpose with role mode
`outer_fit10`. Its schema omits training seeds, arms and training window
schedules, and fixes both training-window resource caps to zero. The only
allowed fitting operation is `population_feature_standardizer`; the plan also
binds `cpu_consumer_source_sha256`, the embedded recipe hash,
`device_policy={device: CPU, cuda_contexts: 0, gpu_updates: 0,
optimizer_updates: 0}`, and a fixed
CPU resource policy (900 seconds, 2 GiB RSS ceiling, and 2 GiB minimum free
memory and disk). The consumer source is checked against its fixed staging
module before source HDF5 bytes are read.

`preflight_approved_fit_pool(fold_id)` validates fixed metadata, plan,
approval, consumer source and deterministic standardizer schedule without
checksumming or opening source HDF5. It returns a frozen, ID-free
`FitPoolPreflight` with purpose, role, CPU/update policy, source and plan hashes,
consumer and recipe hashes, schedule/count hashes, and fixed resource limits.
The consumer SHA is taken from the approved plan and checked against the fixed
module bytes; it is not hardcoded in the loader. The CPU consumer checks that
record and the trusted root launch capability before calling
`load_approved_fit_pool(fold_id)`, which repeats all fixed checks and claims the
exclusive plan attempt lock. A CPU-mode pool exposes no training schedules,
and its context-only batches have
`targets20=None`; requesting `endpoint20` is rejected. This purpose does not
authorize forecast comparisons, outer-validation access, optimizer updates,
GPU work, or Q1_RUN. No CPU-preparation plan or approval has been created.

The fixed sampler and resource-policy limits in code are hard maximums. A
coordinator plan may choose a smaller schedule or byte cap, but cannot raise
the batch cap (128), total pose-array cap (128 MiB), logical read cap (1 MiB),
per-schedule cap (32,768), total 30-schedule cap, or loader-time cap (3,600 s).
Bad folds, roles, seeds, arms, configs, samplers, or caps fail before source
checksum and HDF5 open. Read APIs accept only precomputed immutable schedules
generated from the validated plan; modified or foreign schedules are rejected.

The only source-loading API is `load_approved_fit_pool(fold_id)`. It returns one
fold's read-only pool, with `training_schedules`, `standardizer_schedule`,
`iter_batches(schedule)`, and `safe_aggregate_receipt()`. It takes no path,
role, seed, arm, recipe, sampling indices, or cap from the caller. The pool's
private audit contains dataset paths and must stay in ignored research-control
storage; the safe aggregate contains only counts and hashes. The loader does
not return HDF5 objects and does not call `read_direct` or whole-dataset array
conversion.

For timing-prefix plan authors, the exact top-level plan keys are `schema_version`, `kind`,
`status`, `plan_id`, `purpose`, `role_mode`, `data_bindings`, `sampling`,
`resource_policy`, `recipe`, `recipe_sha256`, and `reviewed_artifacts`. Use
`schema_version=1`, `kind=q1_fit_pool_plan`, and
`status=FROZEN_BEFORE_PREFIX`. Timing-prefix purpose/role pairs are
`train_only_timing_prefix`/`outer_fit10` and
`inner_fit_pool_diagnostic`/`inner_fit8_diagnostic`. CPU preparation uses the
separate purpose `fit_only_cpu_preparation`/`outer_fit10`, status
`FROZEN_BEFORE_CPU_PREPARATION`, and adds exactly
`cpu_consumer_source_sha256`, `fitting_operations_allowed`, `device_policy`,
and `cpu_resource_policy` to the top-level keys. The complete `recipe` object
is canonical JSON whose SHA-256 is `recipe_sha256`.

`data_bindings` must exactly match these fields and the values from the
authenticated fixed metadata: `source_partition`, `source_hdf5_relative_path`,
`source_hdf5_sha256_reference`, `split_sha256`, `fold_digest`, `fold_ids`,
`train_user_count`, `outer_fit_users_per_fold`,
`outer_validation_users_per_fold`, `inner_fit_users_per_fold`,
`inner_selector_users_per_fold`, `origin_inventory_receipt_sha256`,
`origin_map_sha256`, `origin_index_binary_sha256`, `origin_order_sha256`,
`source_path_evidence_sha256`, `fit_segment_value_dataset_names`,
`semantic_run_id`, `semantic_config_sha256`, `semantic_receipt_sha256`,
`arc_config_sha256`, `pipeline_config_sha256`, `ownership_plan_sha256`,
`context_samples`, `horizon_samples`, and `shift`. The loader fills/compares
these values from authenticated metadata; plan authors do not supply pins to
the API.

Timing-prefix `sampling` has exactly: `training_sampler`, `sampler_version`,
`numpy_version`, `fold_ids`, `seed_values`, `arm_ids`,
`windows_per_schedule`, `standardizer_sampler`, `standardizer_seed`,
`standardizer_origins_per_fold`, and `batch_size`. It requires the fixed
sampler, three folds `[0,1,2]`, seeds `[20261026,20261027]`, the five canonical
arm IDs in fixed order, exactly 65,536 standardizer origins per fold, and batch
size 128. CPU-preparation `sampling` instead has exactly `numpy_version`,
`fold_ids`, `standardizer_sampler`, `standardizer_seed`,
`standardizer_origins_per_fold`, and `batch_size`, fixed to seed 20261028 and
65,536 origins. `resource_policy` has exactly `max_pose_array_bytes`,
`max_logical_read_bytes`, `max_batch_size`, `max_windows_per_schedule`,
`max_total_training_windows`, and `max_loader_seconds` with the hard caps above.
CPU-preparation plans set both training-window caps to zero and add exactly
`cpu_resource_policy={max_wall_seconds:900,max_rss_bytes:2147483648,
min_free_memory_bytes:2147483648,min_free_disk_bytes:2147483648}`.
`reviewed_artifacts` must match current hashes for this loader, its tests and
README, the pinned bounded reader and its independent review, and source-path
evidence. The approval has exactly `schema_version`, `kind`, `status`,
`approval_id`, `approved_by_role`, `approved_utc`, `plan_id`, `plan_sha256`,
`recipe_sha256`, `reviewed_artifacts`, and `independent_review_sha256`; it must
use `kind=q1_fit_pool_approval`,
`status=APPROVED_AFTER_INDEPENDENT_REVIEW`, and
`approved_by_role=coordinator`.

The source path and historical byte reference remain
`ABC_IUEH/data/processed/alyx.h5` and
`0986c483c2dc800ff8be1a2a8d2e82d909193f4c3e7c9f7b1eae3e2ea055ea62`. This
authoring task did not open or hash that file, the feature cache, or source
recordings. After approval, the loader verifies the whole-file digest within
the approved time limit and gives h5py that same file handle. The actual source
checksum and schema remain unverified here.

## Synthetic checks

Run with the existing project interpreter; no install, GPU, or Torch is
required:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:TORCH_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\ChatGPT\abc\ABC_IUEH\.venv\Scripts\python.exe' -B `
  'D:\ChatGPT\abc\abc_topic_discovery\workflow_runtime\staging\q1_fit_loader_v3\test_q1_fit_loader.py' -v
```

The tests create only temporary synthetic HDF5. They check plan/approval
bindings, role ownership, pinned helper import order, exact sampler schedules,
65,536 unique standardizer origins, bounded chunk reads, preallocation limits,
lazy context/target indices, read-only arrays, link/file/path denials, gap
checks, augmentation copies, and failure on missing fixed approval before any
source checksum or HDF5 open. Metadata-only checks authenticate fixed origin
metadata while open spies block the actual source HDF5, feature cache, and
recording tree. The production fit-pool path is not run here.

The 20-sample / 100 Hz / 100 ms challenge inconsistency remains unresolved.
Nothing in this draft authorizes human-signal access, model fitting, a Q1 run,
outer validation, or a global Q1 grant.

## V3 legacy-root metadata bridge

V3 is a new fixed-plan namespace. Its plan and approval paths are
`ABC_IUEH/data/research_controls/q1_fit_loader_v3/fit_pool_plan.json` and
`fit_pool_approval.json`; the consumer is fixed under
`staging/q1_cpu_preparation_v3/`. The process-wide attempt lock remains at the
original `data/research_controls/q1_fit_loader/launch_attempt_locks` path.
V1, v2, and the terminal fold-0 claim are preserved; this loader does not
clear or reuse their plan bytes.

The only loading-logic change from the v2 baseline is a strict compatibility
bridge around the pinned reader's root-attribute validator. If both
`source_inventory_file_sha256` and `source_record_inventory_sha256` are
present, the original validator receives the actual HDF5 file and checks both
values exactly. A present-null value, nonmatching value, or partially absent
pair fails closed. Only when both keys are genuinely absent does the bridge
require the caller's already-verified same-file-descriptor source checksum,
the exact fixed HDF5 SHA-256 and path evidence, the root's exact legacy receipt
pointer, fixed inventory and receipt byte identities, the authenticated
70/70 inventory/receipt tuples and session parts, the canonical selected-field
record digest, G1 metadata-bridge evidence, semantic-rebind bindings, and the
fixed split/origin lineage. It then supplies the two derived values through a
read-only overlay to the unchanged pinned root validator. All other HDF5 root
attributes and `config_json` are read from the actual open file. The original
HDF5 object remains in use for all child traversal and value reads.

The fixed legacy receipt is schema v1. The integrated receipt is schema v2;
its byte-level ancestry is recorded at `resealed_from.source_manifest_sha256`
and `resealed_from.source_receipt_sha256`. Its selected-record digest is
`record_inventory_sha256`, repeated in the single `inventory_bindings` entry
alongside the manifest-byte digest. The bridge checks these exact v2 fields,
the schema values, and the 70-row receipt tuples; it does not expect a
top-level `source_manifest_sha256` in the v2 receipt.

In the pinned G1 metadata-bridge document, `signal_open_spy_attempts` and
`human_signal_rows_read` are top-level counters. The loader checks them there;
the receipt-result object separately binds `recording_payloads_read` and the
new-receipt schema version.

The matching receipt tuple includes the metadata `md5` string field. Comparing
that string establishes tuple identity; this code computes no new file or
recording checksum and reads no recording payload. The aggregate receipt
contains only the ID-free mode `direct_root_attributes` or
`legacy_receipt_chain_derived`.

The root's separate metadata-only probe
`workflow_runtime/run_manifests/source_root_lineage_audit_20261029/`
verified both keys are genuinely absent, the exact receipt pointer is present,
the same-FD HDF5 checksum matched the historical reference, and dataset/group
value-read poison counters were zero. The authoring tests here use synthetic
metadata and synthetic HDF5 only. No production HDF5 or recording payload was
opened in this v3 authoring task; a separate root replay and fresh review are
still required before any future source operation.

The metadata-only evidence checker is
`check_lineage_audit_evidence_v2.py`; its output is
`evidence/metadata_lineage_result_v2.json`. It hashes the three fixed audit
JSON files and verifies their statuses, source-hash reference, true attribute
presence bits, config comparison, 70/70 counts, session-part check, and
zero-value-read guards. The addendum also clarifies that the prior tuple audit
compared MD5 metadata strings but did not read or hash source payload bytes.

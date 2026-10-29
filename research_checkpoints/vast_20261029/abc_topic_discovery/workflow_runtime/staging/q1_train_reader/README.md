# Q1 bounded train quaternion reader (draft)

This staging artifact is a fixed-path, coordinator-gated reader for bounded
quaternion windows. It is for read-format and calibration review only; it is
not a complete Q1 training loader. It has no model-fitting, GPU, or
standardization-statistics code. A separate reviewed gate is required for any
fit-user standardization, full train fit, or outer-validation read.

The accepted source path is exactly `ABC_IUEH/data/processed/alyx.h5`, as
recorded in the hash-pinned metadata plan
`data/reset_optimizer_probe/20261028_fixed_train_only/plan_before_results.json`
(SHA-256
`dcb3ebd1e28e8289496235fc1252b40b97a7f233d4f31b867dcccbc41e1e3f77`). The
accepted source-byte reference is
`0986c483c2dc800ff8be1a2a8d2e82d909193f4c3e7c9f7b1eae3e2ea055ea62`. This
authoring and test pass reads no source HDF5, feature cache, recording, or
source CSV bytes and does not hash those payloads. Production execution against
the source HDF5 remains unverified.

## Fixed metadata and ownership

Before any source HDF5 open, the reader calls the pinned train-origin metadata
validator, checks the exact source path evidence, and authenticates the
accepted origin-inventory receipt, private origin map, and little-endian int64
origin index at fixed paths. The private map/index are under ignored
`ABC_IUEH/data/research_controls/`. No caller-supplied paths, hashes, splits,
folds, or modes are accepted. A missing or wrong approval or protocol fails
before HDF5 is opened.

For every frozen outer fold, the ownership is 10 outer-fit users and 5
outer-validation users, all within the 15-user train partition. Within the 10
outer-fit users, SHA-256 rank the UTF-8 strings
`q1-inner-split-v1\0{20261029}\0{outer_fold}\0{user_id}`; the first two ranked
users are the selector set and the remaining eight are inner-fit users. Both
sets are sorted by user ID for canonical iteration. This local implementation
does not import the mutable selector module. A matching synthetic test checks
the shared split contract.

The fixed sampler uses endpoint-inclusive positions
`numpy.linspace(0, N-1, num=min(N, 1024), dtype=numpy.int64)` over each user's
canonical origin order. Selector requests must be exactly 1,024 per selector
user (clipped only when that user's segment contains fewer origins). Inner-fit
prefixes have a fixed upper bound of 1,024 origins per user. A fixed approval
must bind the exact source/receipt/config/fold metadata, this reader, tests,
README and frozen protocol. The approval and protocol files do not exist at
this checkpoint, and the workflow is currently Q1 **NO-GO**. There is no
boolean override or outer-validation role in the reader API.

## Bounded read contract

The only production entrypoint is `iter_approved_quaternion_windows()` in
`q1_train_reader.py`; it accepts no arguments. After a future fixed approval,
it hashes the source bytes through one file handle, then gives that same handle
to h5py. It checks fixed root metadata and only the selected inner-fit or
inner-selector train hierarchy before the first value read. Each approved
origin permits one contiguous 420-row slice: rows 0–399 are returned as a
read-only context, and rows 400–419 as a separate read-only target. The
extractor receives the exact rolling 400-row context. The reader checks finite
unit wxyz quaternions and adjacent hemisphere signs without normalizing or
repairing source values.

The HDF5 guard binds to one FileID and permits only listed train quaternion
paths and exact 420-row ranges. It denies `read_direct`, whole-dataset
conversion, foreign FileIDs, other datasets, unlisted groups, links, VDS, and
external storage. It reads selected timestamps only through shape/type/storage
metadata. It does not enumerate users outside the authorized train rows or
make a global payload check. HDF5 may read/decompress a storage chunk around a
logical slice, while Python receives only the approved selection.

The current endpoint sampler is a bounded format/calibration interface. It
does not yield all 3,330,312 training origins, load full fold-fit segments in
bounded chunks, create batched sliding-window views, or establish numerical
equivalence to the actual source quaternion values. A later full training
loader needs its own reviewed implementation and gate. In particular, this
reader does not claim GPU training readiness.

User, session, segment IDs, maps, and arrays are private. Any future consumer
that persists them must use ignored `data/research_controls/`; only aggregate
identifier-free receipts may leave that private area. The 20-sample / 100 Hz /
100 ms specification conflict remains unresolved, as documented in the
project's protocol notes.

## Synthetic and metadata-only checks

Run with the existing project interpreter (no install, Torch, or GPU needed):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\ChatGPT\abc\ABC_IUEH\.venv\Scripts\python.exe' -B `
  'D:\ChatGPT\abc\abc_topic_discovery\workflow_runtime\staging\q1_train_reader\test_q1_train_reader.py' -v
```

Tests use temporary synthetic HDF5 only for value-window, normalization/sign,
read-boundary, foreign-file, link/storage and denial-spy checks. Metadata-only
checks authenticate the fixed origin receipts/map/index and block any open of
the actual HDF5, feature cache, or recording payloads. They also check that
wrong receipt fields fail before an HDF5 open and that absent protocol or
approval prevents source access. No test runs the production entrypoint on
actual source values or fits a model.

Everything in this folder remains a draft pending independent review. Passing
synthetic tests does not authorize a human Q1 fit, resolve the official timing
ambiguity, or validate the reader against the actual source quaternion values.

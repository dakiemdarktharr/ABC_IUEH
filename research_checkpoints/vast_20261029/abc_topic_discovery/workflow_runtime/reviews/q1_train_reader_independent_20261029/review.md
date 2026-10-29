# Independent Q1 train reader review — 2026-10-29

## Disposition

**Bounded acceptance: synthetic format and fixed-metadata review only.** The frozen reader and metadata boundary behaved as specified in this scope. This is not a Q1 prefix authorization, source-signal validation, complete training loader review, or Q1 run claim. The fixed protocol and coordinator approval are absent, and the production iterator stops before hashing or opening the actual source HDF5.

No source/cache/recording payloads were opened or hashed; no model was trained; no GPU or network was used. The accepted private origin map/index were read only through `validate_metadata_bindings()` at their fixed paths. IDs and map contents were not printed or copied into this report.

## Frozen artifacts reviewed

| Artifact | SHA-256 |
|---|---|
| `staging/q1_train_reader/q1_train_reader.py` | `2b499f28199f91d0a586fc3611964a16a943c0b206d53eb899ad79a367a32149` |
| `staging/q1_train_reader/test_q1_train_reader.py` | `3990aa5b80249de13e0e835138e47d1e66070de24a89906630472012dcb0c208` |
| `staging/q1_train_reader/README.md` | `352837a871097c7b35c91d04a0be94cd6e3acbfd6474d4e898cfb3152a24c2d8` |
| `staging/q1_train_reader/authoring_review_receipt.json` | `77667ee7ef94dd1ab3e1beac8cbc177bb57cfdeeb3550ecb1202dda379a7ce4c` |

All four match the requested frozen pins. No owner bundle files were modified.

## Execution

Interpreter: `D:\ChatGPT\abc\ABC_IUEH\.venv\Scripts\python.exe` (`-B`). Environment: `CUDA_VISIBLE_DEVICES=-1`; OMP, OpenBLAS, MKL, NumExpr, VECLIB, and Torch thread variables set to `1`; `PYTHONDONTWRITEBYTECODE=1`.

Command:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:VECLIB_MAXIMUM_THREADS='1'
$env:TORCH_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\ChatGPT\abc\ABC_IUEH\.venv\Scripts\python.exe' -B `
  'D:\ChatGPT\abc\abc_topic_discovery\workflow_runtime\reviews\q1_train_reader_independent_20261029\run_guarded_suite.py'
```

Result: **13 passed, 0 failures, 0 errors, 0 skipped** (2.852 s). The independent harness added an outer `Path.open`, `builtins.open`, `io.open`, and `os.open` guard for the actual `alyx.h5`, the fixed feature cache, and `data/raw`; its `h5py.File` wrapper allowed only temporary synthetic fixture paths. The run recorded **30 synthetic HDF5 opens and 0 forbidden open attempts**. It also checked the production iterator directly: it failed on the missing fixed approval/protocol before any source-byte hash or HDF5 open. Summary: `actual_source_or_cache_or_raw_opened=false`, `production_entrypoint_blocked_before_source_open=true`.

The runner source is [run_guarded_suite.py](run_guarded_suite.py), SHA-256 `812fe4a03a78c9eff441893df56b321ecbf57bb617f78b6b50bea72cdbd852fe`.

## Review checks and findings

- The metadata-only route authenticated the pinned parent, source-path evidence, actual train-origin receipt, private origin map and little-endian index; it checked the 3,330,312-origin total and 15-user/three-fold metadata without printing identifiers. Its fixed-path preflight checks the actual source as a regular, symlink-free file and checks its inode against protected artifacts. This review did not read source bytes, so the HDF5 byte reference remains unverified here.
- Ownership tests confirm the fixed 10/5 outer split and 8/2 inner split, with SHA-256 ranking seed `20261029`. The selector sampler is endpoint-inclusive `linspace` with exactly 1,024 positions per selector user. The read plan accepts only `inner_fit` and `inner_selector`; no outer-validation role is permitted.
- Synthetic window tests confirm exact shift-one slicing: 420 source rows produce separate read-only float32 `(400, 4)` context and `(20, 4)` target arrays. The guard binds reads to its `FileID`, rejects foreign-file access, wrong/oversized slices and `read_direct`, and records zero denied accesses in the successful fixture pass.
- Synthetic fixtures reject hard-link aliases, soft links, virtual datasets, and external storage before any dataset-value read. The independent external-storage and VDS probes observed zero `Dataset.__getitem__` calls before rejection. The reader only inspects timestamp dtype/shape/storage metadata; it does not read timestamp values. Metadata validation also enforces zero development/final signal reads in the accepted origin receipt. Synthetic denial tests reject development/final hierarchy dereferences.
- The mismatch fixture confirms byte-hash failure precedes h5py open. The live production iterator is gated more strongly: absent fixed protocol/approval currently prevents even that source-byte hash step.

No code defect was found within the bounded format/metadata scope. The source HDF5 layout, actual quaternion contents, and actual source-file-to-origin numerical alignment were not exercised.

## Open gates

1. No fixed frozen prefix protocol or coordinator approval exists. The live production iterator correctly refuses access. This review grants no permission for a human-data prefix.
2. The staged reader is intentionally capped to a bounded prefix (32,768 windows maximum); it does not load all training origins in bounded batches or provide the complete fit/outer-validation/inference integration. Those remain separate implementation and review gates.
3. The actual HDF5 file hash, schema and signal values must be checked only after the required protocol/approval exists and a coordinator-controlled run is authorized. HDF5 may decompress storage chunks internally around an approved logical slice even though Python receives only the selected 420 rows.

**Acceptance boundary:** suitable as a synthetic format/calibration interface and metadata gate. Not ready for actual source reads, Q1 fitting, outer validation, or deployment inference.

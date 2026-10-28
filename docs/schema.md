# Private HDF5 and manifest schema

`data/` is ignored by Git. This schema describes fields; public examples and
tests use synthetic inputs only.

```text
users/<source_user>/sessions/<source_session>/segments/<segment>/
    quaternion  float32 [N,4], unit wxyz, hemisphere aligned
    timestamps  float64 [N], seconds, uniformly spaced at100 Hz
```

Root attributes include schema version, source/license, coordinate convention,
sample rate, unresolved timing statement, offline-interpolation declaration,
effective config JSON and the source-receipt SHA-256. Session attributes preserve
source revision/path provenance and preprocessing statistics.

`alyx_inventory.json` is pinned public metadata saved locally: source commit,
DVC pointers, expected sizes/MD5 and candidate user/session grouping. Its
`payload_downloaded=false` describes discovery time; current transfer state is
in the download receipt, not a mutable flag in the inventory.

`download_receipt.json` records verified transfers with SHA-256 and all expected
motion-file paths of each source session. A partly downloaded session cannot
qualify. Raw CSVs contain controller fields because the source provides a single
motion object; only HMD quaternion/time enter the HDF5.

`quality_report.json` contains all acceptance/rejection reasons and quality
statistics, output counts, config and HDF5 checksum. Participant/session details
are private and must not be committed.

Each HDF5 ingestion stores an immutable source-receipt snapshot next to the HDF5
under `provenance/receipt_<sha256>.json`. The digest is computed from exactly the
bytes parsed, so concurrent later downloads cannot change the provenance of an
already prepared corpus.

`user_split.json` binds disjoint assignments to the HDF5 checksum, seed and split
mode. A smoke manifest has a `smoke` partition and empty train/development/final
partitions with `training_allowed=false`. A confirmatory manifest requires the
minimum qualifying-user count. Never replace a split to obtain favorable errors.

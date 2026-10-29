# Prepared fit10 bundle for Vast

This directory implements the data side of a private Linux Vast train/resume
route. It contains a pure-standard-library exporter/reader contract and
generated-only tests. It contains no participant data, source HDF5, split file,
user IDs, private statistics, or fabricated production receipt.

## Bundle contents and invariants

A successful export writes a fresh, private directory containing exactly:

- `segments.f32le`: concatenated, segment-local quaternion rows as little-endian
  `float32`, order `(w,x,y,z)`.
- `manifest.json`: canonical JSON with exact payload/segment byte hashes,
  root-derived source binding hashes, a root Ed25519 identity attestation, the
  fixed role, and one fold ID.

The exporter creates the directory read-only (`0500`) and files read-only
(`0400`). If a later authorized transfer changes those mode bits, restore them
on Vast before opening; the loader refuses writable bundle files/directories.

The manifest accepts only `role=outer_fit10`, the fixed fit-only pilot
purpose, and one `fold_id` in `0,1,2`. Each bundle is for one fold only. Two
fold bundles may overlap in participants under the frozen OOF design; a run
must open only the bundle for its own fold. It has no fields for outer
validation5, old development5, or sealed final15. The
exporter takes no source pathname and has no operation that copies or packages
the full HDF5. On open, the directory must contain exactly the two allow-listed
files; an extra source file, cache, index, symlink, junction, or reparse point
causes denial.

Every segment declares `sample_count=N`, `origin_count=N-419`, byte offset,
and a digest. The rank map is derived from the signed segment descriptors;
there is no separate indices file. The source adapter checks finite, unit-norm, hemisphere-continuous
quaternions before writing. The reader recomputes the whole payload hash and
checks the manifest/schema before any window is served. A requested origin is
resolved within one segment only. It returns packed `float32` context
`[400,4]` and target `[20,4]` bytes; it never reads a context from one segment
and a target from its neighbor. The challenge's 20-sample / 100 Hz / 100 ms
wording remains unresolved; the bundle preserves the published 400-to-20
sample shapes without converting that horizon to milliseconds.

The payload identity is shared across neural arms. Arm, seed, model, recipe,
runner, checkpoint hash, resume cursor, and run authorization belong to the
opaque root-issued run binding returned by the verifier. For the current
fold-0 micro-pilot, the root run plan must bind the three reviewed arm IDs,
seed `20261026`, and only the first 12,800 PCG64 ranks from
`default_rng(SeedSequence([20261026, 0]))`; all arms share those ranks. This
data module does not infer or authorize that schedule. That binding is
revalidated on every open and through `bundle.assert_active()`. A resume must
pass the checkpoint-authenticated `expected_identity_sha256` and the exact
opaque `expected_run_binding`; the verifier must renew/validate the permit and
confirm the model/runner/recipe/cursor. The run binding's
`expected_identity_sha256` must equal the argument and must be minted from the
byte-verified checkpoint, never copied from a user-editable JSON field.
Manifest JSON alone never authorizes a
source read or training run. The process-local binding object is not
serializable.

## Trainer API

Import from `data.bundle` when the parent staging root is on `sys.path`:

```python
from data.bundle import open_prepared_train_bundle

bundle = open_prepared_train_bundle(
    bundle_dir,
    trusted_verifier=root_injected_verifier,
    expected_fold_id=0,
    expected_run_binding=root_injected_run_binding,
    expected_identity_sha256=checkpoint_bundle_identity,  # resume only
)
try:
    total_origins = bundle.total_origins
    segment_index, local_origin = bundle.origin_for_rank(int64_rank)
    contexts_bytes, targets20_bytes = bundle.sample_window(segment_index, local_origin)
    # Convert the bytes to tensors in the trainer; this module imports no NumPy/Torch.
    bundle.assert_active()
finally:
    bundle.close()
```

`sample_rank(rank)` is a convenience wrapper. `iter_ranked_windows(ranks)` is
lazy and accepts an externally generated sequence of explicit rank integers;
it does not create or store the PCG64 schedule. The trainer owns the exact
seed/fold schedule and must persist its generator/checkpoint state. This keeps
the same rank schedule available to each arm while using a single fit10 bundle.
Call `abort_and_close()` when a run fails; it closes the mapping and calls an
optional `abort()` hook on the opaque root binding.

The export API is
`export_authorized_fit10_bundle(destination, fold_id=0, root=root_injector, purpose=...)`. A
trusted root capability must issue and authenticate a fresh one-use grant
bound to purpose, `outer_fit10`, and that exact fold before its source reader
opens. It then provides only selected fit-segment byte views. The function
streams segments to disk and never materializes sliding windows. It rejects a
grant supplied as JSON/dict, wrong-purpose/role/fold grants, and an expired
grant. The grant's in-process type is a capability only because the root
launcher authenticates and pins the code that supplies it; Python object types
do not authenticate an OS user.

The local CLI is `python -m vast_training_v1.data.export_guard --help` from the parent
`staging` directory. The guard launches `data.export` as a child and polls
available RAM every 250 ms, stopping the child if availability drops below
2 GiB or memory monitoring fails. The adapter also checks its own preflight
and segment boundaries. Polling cannot prevent an instantaneous drop between
samples; incomplete output from an interrupted export must not be uploaded.
Both output parents must already be private. On Windows the export checks the
actual DACL and requires inheritable allow rules for only the current user; Unix
parents must be user-owned with no group/other access. A private parent created
for this project is `workflow_runtime/private_vast_data`; the broader workspace
does not pass this DACL check. The parent check does not protect against a
privileged administrator or someone able to replace its ACL during export.
Running it reads the authorized local source; no real
export was run here. It requires new, separate bundle and private evidence
directories and places the key ID, public key and actual post-read receipt
outside the bundle. Independently preserve the public key and identity; do not
transfer the source evidence or full HDF5 with the train-only bundle.

## Source verification recipe and current dependency

The data exporter does not itself open the local human-data source. Before a
production export, a separately approved root adapter must:

1. Create a new fold-specific decision, plan, approval, independent review,
   trusted launcher capability, and one-use source claim for this exact
   fit-only purpose. Do not reuse the consumed historical V1/V2/V3 claims.
2. Authenticate the exact V3 loader and bounded reader bytes before import.
   Verify the source SHA-256 on the same open file descriptor later passed to
   HDF5, then validate the root/legacy receipt chain, source/split bytes,
   frozen outer-fold membership, and selected fit10 segment metadata before
   reading pose chunks.
3. Select only the requested fold's `outer_fit10`; do not read the fold's
   validation5, old development5, or sealed final15. For every selected
   segment, validate local storage/links, increasing 100 Hz timestamps,
   finite normalized `wxyz` float32 poses, hemisphere alignment, and the exact
   authenticated `N-419` origin count. Retain no user identifiers in the
   exported manifest.
4. Give the exporter a root-issued object whose `assert_live()` is checked
   before every segment. Verify static source hashes before reading; only after
   the stream is fully consumed, finalize and bind the actual read receipt hash.
   Reject any change to the pre-read bindings. The root signs
   `attestation_signing_bytes(bundle_identity_sha256, root_attestation)` with
   its Ed25519 key; do not invent or copy an identity digest into a receipt
   before those bytes exist.
5. On Vast, use a separately authenticated verifier pinned to the root public
   key and the exact train-run plan. The root signs
   `attestation_signing_bytes(bundle_identity_sha256, root_attestation)` from
   `data.bundle`. It verifies the signed manifest, exact payload hash, fold,
   role, current authorization, and the run binding. Store
   the bundle identity digest in the authenticated checkpoint and require it
   on resume. Each fold uses a distinct bundle; never concatenate folds.

`source_adapter_v1` now provides a staged local V3 adapter and a local signing
authority for a fresh purpose-specific export. Its tests use generated inputs;
no real bundle has been created or validated. A signature verifies integrity
under a separately pinned key, not data-use permission or transfer authority.
The earlier V3 CPU-preparation claims were consumed and cannot authorize this
export. A staged single-fit launcher and synchronous backup callback exist;
the real bundle, target Linux/CUDA runtime, provider pricing and budget ledger
still require verification before a paid run.

No network, Vast upload, GPU, CUDA, source file, index, HDF5, or private
statistics were touched for this implementation. Do not upload the full
`ABC_IUEH/data/processed/alyx.h5` or any containing source directory. Transfer
only an independently verified two-file bundle after the user separately
authorizes the required cloud action.

## Generated-only checks

`test_bundle.py` uses two synthetic 420-row quaternion segments and tests the
400/20 window boundary, segment-local `N-419` ranks, wrong-role/fold denial,
symlink and extra-member denial, corrupted-byte denial before the verifier,
opaque binding requirements, and exact bytes/rank mapping after reopen. It
does not import NumPy, h5py, Torch, the historical loader, or real source code.
Generated-only local execution passed 12 tests with one symlink fixture skipped
on this Windows environment. A real Ed25519 signing/verification fixture also
passed using the bundled Python runtime; no human data was accessed.

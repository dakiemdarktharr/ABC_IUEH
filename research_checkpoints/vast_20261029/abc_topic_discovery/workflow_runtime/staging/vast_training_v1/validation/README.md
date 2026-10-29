# Three-arm outer-validation route

The three-arm draft requires the full frozen outer-validation origins within
train15, three folds and two seeds. The prior fit10 reader does not supply this
role. This separate implementation has not yet undergone independent review:
protocol v5 requires review and a prospective lock before an actual validation
read. It has not been run on human data or produced a result.

`source.FixedValidationSegments(fold)` validates the pinned train metadata,
selects the fold's five disjoint validation users and checks each fixed origin
slice is consecutive. **Its iterator is disabled before opening source values:**
the real reader has no independently verified two-stage binding or target gate.
A future implementation must distinguish hashing the whole HDF5 as opaque
bytes, including holdouts, from reading their values.

`evaluator.evaluate_fold` first verifies all 18 byte-hashed checkpoints and
5000-update recipes, code/model pins and one training data identity per fold.
It then scores the six fold predictors on the exact same origins, reporting
endpoint20 geodesic degrees, per-user counts/mean/P95/P99 and pooled-origin
summaries. Users are reported by anonymous fold-local index. Error vectors are
not written. Seed predictions are reported separately, not ensemble averaged.
The metrics are exploratory and do not establish final-holdout performance.
The current output does not implement the separate classical selection,
confidence intervals, event strata, matched-time sensitivity, or final decision.

This CLI cannot perform real evaluation while the source iterator is disabled.
Only after the missing safeguards are implemented and the new reader has been
independently reviewed, copy the completed checkpoints back
to the local machine that already holds the source HDF5 and run the evaluation
there from the parent staging directory. The Vast fit-only bundle is insufficient
to evaluate validation5. Do not upload the full HDF5 or the old-dev/final
holdouts. A validation-only export would need its own reviewed implementation;
none exists here. The manifest must contain 18 records, each
`{"fold_id":0,"arm_id":"ordered_product","training_seed":20261026,
"path":"/private/checkpoint.pt","sha256":"<independently retained hash>"}`,
for all folds/arms/seeds. Checkpoint sidecars must accompany the files. SHA256s
must be retained independently of the checkpoint files; manifest labels alone do
not establish authenticity. Outputs are private and exclusively created.

```bash
python -m vast_training_v1.validation --fold 0 --checkpoints /private/checkpoints.json \
  --fit-manifests /private/fold0/manifest.json /private/fold1/manifest.json /private/fold2/manifest.json \
  --fit-public-keys /private/fold0/public-key.bin /private/fold1/public-key.bin /private/fold2/public-key.bin \
  --fit-key-ids KEY_ID_0 KEY_ID_1 KEY_ID_2 \
  --output /private/validation-fold0.json --device cpu
```

Repeat explicitly for folds 1 and 2 locally. Python, NumPy, PyTorch and the pinned model
and source helper dependencies are required. Runtime, resource use, 18-model
end-to-end execution and the new validation source route have not been tested
on actual data; no cloud operation is performed by these commands themselves.

Each fit manifest signature is checked against its fold-specific public key and
key ID supplied separately, in fold 0, 1, 2 order; local exports generate
separate keys per fold. Each checkpoint's training bundle identity must match its fold's signed
manifest. The legacy evaluator retains per-origin errors in memory for exact quantiles.
`two_stage.py` is a separate synthetic-source draft: it freezes predictions,
checks all records and a retained lock digest before requesting targets, and
uses bounded external sorting and a 2 GiB available-memory check. These checks
do not authenticate callback provenance or prove that its claimed origin map
is the fixed validation map. It is not wired to the CLI or the real reader.
Independent verification of signed checkpoint/code and source/origin pins, a
reviewed target-gated reader, prospective protocol lock and independent review
remain required before actual use. A lock digest must be retained independently;
a digest returned by the same untrusted process is not authorization. CUDA
validation is not integrated with the budget ledger; the CLI restricts CPU.

Additional generated-only draft components are `window_reader.py` (the exact
400-row context and endpoint-20 slice mapping), `hdf5_adapter_draft.py`
(selected validation hierarchy traversal for an already open HDF5 handle), and
`signed_plan.py` (a prospective Ed25519 plan-binding primitive), and
`plan_binding.py` (file-backed measurement of the signed source, split,
protocol, 18 checkpoints, code files and signed fit identities).
`checked_predictor.py` detects changes to model tensors, serializable module
attributes and Torch CPU RNG during inference. These components are not
connected to the disabled reader or CLI. When explicitly called, `plan_binding`
hashes its input files, including the complete source file as opaque bytes; it
does not decode HDF5 values or establish permission. The HDF5 adapter relies on its caller to authenticate the source
on the same file descriptor, check the signed plan, enforce the target lock,
guard memory and obtain independent review. A signature by itself is not that
review or an authorization. These draft components have only generated tests;
no human values have been read.

The sliding contexts overlap: an endpoint for one origin can appear in the
context for a later origin. A two-pass reader alone cannot prevent a stateful
predictor from retaining later contexts; the reviewed prediction route must
enforce the frozen stateless inference contract and bind the code that actually
executes to the independently measured files. The checked predictor is not a
sandbox: it cannot detect concealed changes to process globals, files or network
state, and is not proof of statelessness. The current scorer also does
not implement the protocol's classical reference, cross-user bootstrap,
event diagnostics, or fixed-time comparisons. These remain outstanding and
must not be inferred from the six neural summaries.

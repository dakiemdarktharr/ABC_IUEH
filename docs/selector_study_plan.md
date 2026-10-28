# Prospective selector study plan

Recorded 2026-10-28 before the first real selector fit. This extends the
prospective design in the topic-discovery dossier without changing its primary
contrast, success thresholds, data split or final holdout. Prior train/dev
analytic and 80,572-parameter MLP outcomes are already known and reported.

The six R80 variants use the same 244-parameter 10 -> 16 ReLU -> 4 softmax,
SGD2000 steps, batch128, lr0.001, zero momentum/weight decay, two fixed seeds.
For each seed, clone one initial state and reuse identical global training-origin
draws across every variant. Initialize weights Normal(0,0.02), biases zero,
following the earlier synthetic selector prototype. Do not tune on development
or select checkpoints: retain the final fixed step of every run.

Five learned variants were specified in the original design. The sixth,
h1_a19, is added prospectively as a learned issue-age control before selector
training. It shares issue origins with h20_a0; its mature targets differ.
The primary h20_a0-minus-h1_a0 contrast remains fixed. Report all six variants
and both seeds regardless of outcome. Sensitivity at R128/R256 is predeclared
for h1_a0/h20_a0 with unchanged data/model/steps/batches/seeds.

The expected individual-expert endpoint20 angular loss is a surrogate. Report
actual normalized float32 emitted-blend endpoint20 geodesic error, not the
surrogate as accuracy. Save expected-loss and hard-selection diagnostics to
expose mismatches. Raw/cleaned data, cache arrays, origin schedules, per-user
metrics, scalers and weights stay local; public outputs are aggregate only.

Use the user's RTX5050 with CUDA float32, no autocast/TF32, fixed deterministic
settings. Fail rather than silently train on CPU. Record package/device details,
actual CUDA allocation/kernel use, synchronized fit wall time, peak allocated
and reserved VRAM, data/config/source/scaler hashes, counts and seeds. Resource
measurements on this laptop do not establish latency on the organizers' XR
target. Dense selector MACs224 exclude evidence, experts and geometry.

Train15/development5 remain the only accessible user partitions. Final15 are
excluded from feature construction, targets, metrics, pretraining and fitting.
Whole-file checksums may read opaque bytes for integrity. The existing frozen
baseline bundle is immutable; sensitivity artifacts must be separately bound.

Use paired user-cluster bootstrap2000, window-weighted ratio, fixed seed20261026.
Average the two seed error vectors for the primary summary, not predictions;
also report per-seed spread, user-macro mean and window P95. Compare with the
original train-fitted static mixture on exactly matching origins. Development
inference remains exploratory with five users. No final success/equivalence
claim is permitted, even if a development contrast passes numerical thresholds.

Latest user instruction keeps final sealed through this paper. The resulting
paper is therefore a complete exploratory train/dev pilot, not confirmation of
the preregistered final hypothesis. State this scope change explicitly. The
paper stays local for review; do not submit or post a preprint. JEPA/EBM require
the user's later okay after the main paper is complete.

The 20-sample / 100 Hz / 100 ms conflict and Alyx-only challenge eligibility
remain unresolved. Preserve400 -> 20 static float32 wxyz and sample indices.

# Selector metrics and experiment recording contract

Recorded 2026-10-28 for the prospective [selector study](selector_study_plan.md).
This is a team implementation and reporting contract. It is separate from the
official challenge requirements and from questions still awaiting organizer
clarification. The primary metric follows [protocol.md](protocol.md): emitted
float32 normalized wxyz quaternion geodesic angular error in degrees at output
sample 20. The 20-sample / 100 Hz / 100 ms inconsistency remains unresolved.

## Private evaluator interface

`abc_iueh.selector_metrics.summarize_selector_partition` consumes an already
sealed and validated feature cache, a flat error HDF group, private per-user
sufficient statistics, an optional diagnostic HDF group, and the original
train-fitted static-mixture errors and sufficient statistics. It never accepts
`final`; feature-cache traversal also rejects a final feature partition.

The default six variants are `h1_a0`, `h20_a0`, `h1_a19`, `h20_a20`,
`complete_horizon`, and `motion_only`. Each has two fixed seed datasets
`Selector_<variant>_seed_20261026` and
`Selector_<variant>_seed_20261027`, plus
`Selector_<variant>_two_seed_mean_error`. The last vector must be the exact
rowwise arithmetic mean of the first two error vectors. It is not a prediction
ensemble, a refitted model, or an inference artifact. For R128/R256 sensitivity
in separately bound artifacts, `variants=("h1_a0", "h20_a0")` reuses this
contract; the primary pair and seeds remain fixed. Record R in the run config
and summary provenance; do not mix sensitivity errors into the primary R80 run.

All flat vectors follow the same feature-cache traversal: partition, user,
session, segment, then increasing `origin_index` within the segment. Every
stride1 context is included, starting at origin 399. Each private error and
diagnostic partition group carries `row_order_sha256`, computed by
`cache_row_order_sha256(cache, partition)`. This digest binds traversal order,
private group identities, counts and origin indexes. The immutable sealed cache
checksum binds the corresponding features/labels. A fingerprint attests to the
writer's declared row mapping; it cannot establish semantic correctness if a
writer deliberately assigns the wrong values to correctly labelled rows.

The private JSON schema is
`{partition: {user_sums: {method: {user: sum_error_degrees}}, user_counts:
{user: window_count}}}`. Counts must match the cache exactly. All methods and
the static mixture must cover the same users and windows. The recorder checks
flat-vector shape, local storage, finite [0,180] degree range, exact two-seed
mean arithmetic, and sums within each user's cache blocks. Wrong counts or a
cross-user row permutation fail before reporting. These checks supplement the
runner's source-target/causality tests and feature-cache checksum validation.

The primary run has **3,330,312 train windows from 15 users** and **1,189,027
development windows from 5 users**. These are verification expectations for the
real run, not substitutes for its measured receipt. Final has 15 locked users
and receives no feature, target, model-error, or diagnostic evaluation here.

## Estimands and uncertainty

For method m, let S_um be the sum of endpoint errors for user u and n_u its
window count. Report the window-weighted mean `sum_u S_um / sum_u n_u`,
user-macro mean `mean_u(S_um/n_u)`, and pooled-window P95. P95 describes windows;
overlapping windows are not treated as independent participants. The recorder
also reports a separately named bootstrap CI for the user-macro mean; this is
supplemental and does not replace the predeclared window-weighted estimand.

For each of 2,000 resamples, draw U user indexes with replacement using NumPy
`default_rng(20261026)`, where U is the number of users in this partition. Each
draw includes all of that user's sessions and overlapping windows. Recompute
the window-weighted ratio, including multiplicities. Use the same user draws
for every method and take the 2.5th/97.5th percentiles of paired differences.
This preserves unequal recording lengths and method pairing. A bootstrap of
independent windows, a mean of user means for the primary CI, or independent
resampling of methods violates the contract.

The primary contrast is
`Selector_h20_a0_two_seed_mean_error - Selector_h1_a0_two_seed_mean_error`.
For each user and row, average the two predeclared seed errors before the
primary contrast and bootstrap. Report h20 against issue-age control h1_a19,
target-age control h20_a20, Complete-Horizon, motion-only, and static mixture
using the same pairing. The current learned h1_a19 control is a prospectively
added exploratory control; it must not be relabelled as the separately
preregistered analytic hard-routing second confirmatory contrast. Do not imply
that horizon, target age and issue age are three independent causal factors.

For every variant, separately report the two seed means and their range. User
bootstrap CIs condition on these two fitted runs; they do not include training
set resampling, arbitrary seed variability, or uncertainty from model search.
The exact mean-error vector's P95 can differ from the average of the two seed
P95s; compute it from the mean-error vector itself.

The static mixture is reused from its original train-only fit: simplex step
0.25, 35 candidates and 65,536 calibration windows. It is not fit or selected
using development. Its earlier means (approximately train3.313769 degrees and
dev3.724920 degrees) are provenance cross-checks, not new measurements or values
to hard-code into the new summary. Its arrays and per-user counts/sums must
match the current cache coverage before comparison.

## Decision language

Negative delta means lower error for h20. Absolute improvement is comparator
mean minus h20 mean; relative improvement is that difference divided by the
comparator mean. If the comparator mean is zero, relative improvement is
undefined (`null`) and the relative threshold cannot be met. A user improves
only if that user's two-seed-mean h20 sum is strictly smaller; ties do not count.

The unchanged practical rule is **relative improvement >=3% AND absolute
improvement >=0.05 degrees AND paired user-bootstrap CI upper delta <0 AND
at least60% of users improved**. All four numerical checks are recorded for
development as descriptive illustrations. The final success criterion is
`not_evaluated_final_untouched`, even if all development checks are true. No
development outcome confirms H1, meets the preregistered final criterion, or
justifies opening or reselecting the final split. Training is in-sample;
development inference is exploratory with only five users. The current paper
scope explicitly keeps final sealed.

The predeclared equivalence margin is `max(0.05 degrees, 0.03 * comparator mean)`.
A development CI entirely inside its symmetric margin is labelled only
`CI_inside_margin_conditional_development_only`: a conditional development
illustration, not a formal final equivalence conclusion. Otherwise label the
illustration **inconclusive**. Non-significance alone does not establish
equivalence, and a failed superiority rule does not prove no useful effect.
No multiplicity-adjusted confirmatory claims are made from this development
table. Primary and control comparisons must retain their declared roles.

## Surrogate versus emitted prediction

Private diagnostics have one group per seed method containing
`expected_loss_deg` and `hard_error_deg`, each in the exact same flat row order.
For weights w_k and individual expert endpoint loss L_k, the training surrogate
is `sum_k(w_k * L_k)`. Accuracy is the actual geodesic error after blending
expert velocities, applying the endpoint quaternion rotation, normalizing and
casting the emitted prediction to float32. These are different operations.
`hard_error_deg` measures the endpoint emitted by the argmax-weight expert.

Report mean surrogate, mean actual emitted-blend error, paired mean
`surrogate - emitted_blend`, mean hard-argmax error and paired mean
`hard_argmax - emitted_blend`. The recorder additionally provides a pooled
window Pearson association for each seed. That correlation is a post hoc
descriptive diagnostic, with no independent-window significance test or causal
interpretation. Two-seed diagnostic means are arithmetic averages of seed
means; no correlation is invented by averaging correlations.

Expected individual loss is **not an upper or lower bound** on the nonlinear
emitted blend's loss. A surrogate gap can be positive or negative. Neither a
gap, an oracle advantage nor a correlation establishes that a causal selector
can learn the missing improvement, or that optimization has converged. Keep
future expert losses as labels/diagnostics, separate from past-only inputs.

## Reproducibility and resource receipt

The run receipt and public aggregate must record the following without local
paths, private identities or an unrestricted environment dump:

| Area | Required record |
|---|---|
| Run | Schema/stage, UTC timestamp, configuration SHA-256, code commit/source-file hashes and named baseline code version. Record dirty-source hashes when applicable. |
| Data | Source-HDF5, split, cache, scaler, sealed feature receipt and baseline error/statistics hashes; source adapter/commit; measured windows/users for train and development; final-access audit. |
| Learning | R, six variants or declared sensitivity subset, architecture10->16ReLU->4softmax, both seeds, initialization, identical per-seed initial state and sampled-origin schedule across variants, SGD2,000steps/batch128/lr0.001, momentum0/weight decay0, train-only scaling, fixed last checkpoint and no early stopping. |
| Numeric execution | Python/NumPy/PyTorch/CUDA versions, OS family, GPU model/capability, device assignment, float32/no autocast/no TF32, deterministic settings, evidence of real CUDA allocation and a completed GPU operation. Record versions of the libraries actually used. |
| Time | GPU-synchronized wall seconds around the declared fit scope, including explicit pre/post CUDA synchronization; separate CPU elapsed/wall timer scopes for loading, feature preparation, evaluation/I/O and full stage. `perf_counter` elapsed includes waiting and is not CPU process time. If process CPU time is measured, name it separately. A CUDA event duration and CPU wall duration are different measurements. |
| Memory | Peak CUDA **allocated** and **reserved** bytes, measured after resetting peak counters at a documented scope; total VRAM and any current allocated/reserved values use different field names. |
| Integrity | Private checkpoint/error/diagnostic/summary hashes, matched schedule and row-order fingerprints, source/split unchanged before and after, train/development read counts, zero final quaternion/feature/target/error reads. Hashing opaque source-file bytes is distinct from evaluating final data. |

Selector **244 trainable parameters** is
`10*16 +16 +16*4 +4`. Selector dense matrix **224 MACs per window** is
`10*16 +16*4`; it excludes bias additions, ReLU, softmax, scaling, expert
construction, evidence history, quaternion operations and the400->20 output.
These counts are distinct from estimated whole-graph MACs per streaming step
and from a full operator-cost account. If whole-graph cost is unavailable,
record it as unmeasured with scope; do not report224 as whole-system MACs.
State how any whole-graph estimate counts non-matrix work and streaming reuse.

Laptop CUDA training or evaluator wall time is not a target-XR latency
measurement. The challenge's latency measurement requires its own target
device/runtime/batch1/warm-up/steady-state protocol; do not infer it from MACs,
GPU utilization or a training timer. No XR latency claim is supported here.

## Release boundary

Private HDFs, source/session IDs, per-user sufficient statistics, origin
schedules, features/targets, scalers, checkpoints/model weights, access
attestations, credentials, machine usernames and absolute system paths remain
under ignored local storage. Do not write them into public figures, tables,
paper supplements or JSON. A source/runtime version and a safe device model
are sufficient; an environment dump is not a reproducibility receipt.

The recorder returns only method-level aggregates, counts, interval bounds,
seed lists/spread, aggregate diagnostic means, contrast checks and explicit
scope language. Code/configuration, synthetic tests, aggregate JSON/plots,
references and the reviewed paper may be versioned. The coordinator must
inspect the final assembled public artifact because run metadata is added
outside this recorder. Dataset-use and Alyx-only challenge eligibility caveats
in [data_sources.md](data_sources.md) remain in effect; this study does not
record an organizer clarification or prove registration/submission.

# Fixed angular-exposure GPU pilot — 2026-10-28

Status: written before new-model human-data fits or inner-validation error
values. The user explicitly authorized a new local GPU pilot. This plan and
`configs/arc_pilot.json` define that pilot; the earlier research-reset proposal
with a possible 10,000-step cap is superseded here by **exactly 5,000 steps**.
This is a new exploratory study, not a retrospective change to the old selector
study or its hypothesis. No automatic development/final stage is authorized.

## Question and evidence already known

Does a small conditional exposure distribution yield lower actual endpoint20
angular error than a fold-fitted static blend, direct scalar angle prediction,
and direct angular action-cost regression under controlled data exposure?
Scalar projected-label and direct vector models are mandatory controls. The
old selector's negative development findings and the train-only optimization
diagnostic were known before this design. The new fixed support screen was
also known: a nondeployable best-of-21 training oracle achieved 2.311634° on
65,536 sampled origins, versus 3.322490° for its nested j15 historical static
reference. That establishes geometric headroom only.

No new human-data architecture has demonstrated learnability, improved
generalization or high novelty. A conditional distribution, TCN, quaternion
output and median are inherited ideas. A median is angular-optimal for the
snapped short arc, not general off-axis truth. The documented exact
counterexample remains part of the study's limitations. No likelihood,
calibration, oracle or synthetic result substitutes for actual emitted error.

## Sources, folds and prohibited access

Use the existing conditional academic WhoIsAlyx pilot. BOXRR remains disabled
pending its recipient approval gate; Alyx-only challenge eligibility is
unconfirmed. Preserve source/split/cache hashes in the JSON. The official
20-sample / 100 Hz / 100 ms inconsistency remains unresolved; use sample indices
and the static wxyz float32 400→20 interface.

Only the outer 15 training users supply human signals. Lexically sort their
private IDs, apply `numpy.default_rng(20261028).permutation`, then split into
three five-user folds. Fold IDs are 0,1,2. Each fit uses the other ten users;
all sessions of a person stay together and each person validates once. Save
the private assignment and its digest before fitting. No random-window split.

All fitted statistics, static weights, class priors, diagnostics, pretraining
and calibration use only the ten fitting users of that fold. Fixed token
normalization is the physical-unit factor100, with no fitted scaler. Never
reuse old all15 static weights/population scalers as fold-fitted estimates.
The five old development users were repeatedly observed by earlier studies;
they are **unread in this pilot**, as are all fifteen final users. Reading
deterministic origin-index sequences/shape metadata only for the mandatory
sealed-cache validator is separately logged and is not a signal exception.

Every lazy context and endpoint stays within its existing gap-free segment.
The R80 cache's deterministic origin coverage is retained for pairing; do not
silently add earlier segment origins to one arm. The model consumes only past
context. Candidate labels and costs may use a fitting endpoint as supervision;
they never enter the encoder input. Inner-validation endpoints only score
frozen fits and do not fit priors or normalization.

## Frozen architecture and losses

Use the last342 of400 quaternion observations and341 one-step rotations.
Express increments in the current quaternion frame, take principal SO(3) Log,
and multiply by100. The shared1,360-parameter causal encoder has a3→16 kernel5
stem and depthwise/pointwise residual blocks with kernel5 and dilations4,16,64.
Its receptive field is341 increments. Head parameters give totals1,717 for
NLL/risk,1,649 for scalar and1,683 for vector arms. The maximum parameter gap
is3.96% of the larger model. All weights use NumPy `default_rng(seed)`
Normal(0,.02), module parameter order, float32; all biases are zero. Clone
the exact encoder initial tensors within every fold/seed across all five arms.

`v_world=Log(q_current inverse(q_previous))`; `Q_j=Exp(j v_world)q_current`,
j0…20. Uniform probability is assigned to exactly tied nearest candidates.
This is a quantized best-exposure label, not an observed stopping time.

| Arm | Head and fixed training objective | Emitted decision |
|---|---|---|
| nll21 | Linear16→21; categorical NLL against uniform exact nearest ties | Median, mean and MAP of the **same fitted probabilities** |
| scalar_angle | Linear16→16,ReLU,Linear16→1; actual endpoint atan2-geodesic radians | Continuous exposure `s=20 sigmoid(output)` |
| scalar_pinball | Same scalar head; uniform-tie expectation of `.5 abs(s-j)/20` | Same continuous exposure path |
| risk21 | Linear16→21; MSE of21 actual unsnapped candidate costs in **radians**, averaged over actions | Minimum predicted cost; no probability or energy-density claim |
| vector_angle | Linear16→16,ReLU,Linear16→3; actual endpoint atan2-geodesic radians | Direct anchored tangent correction described below |

Loss units and numerical scales differ by objective. This natural-unit choice
was fixed before fitting and is not matched loss scaling. Do not change it,
the optimizer, an arm or the budget after seeing results. Finite synthetic
gradients at zero and noncommuting rotations are required for the actual angle
and risk paths; an NLL backward pass alone does not verify them.

For scalar/exposure decisions, emit `Exp(min(k,s)v_world)q_current`, k1…20.
Median/MAP/risk use discrete j; scalar and mean-probability decoding use the
continuous arc envelope. The latter are stronger continuous point controls,
not identical discrete action sets. The exact same-discrete-support contrast
is NLL versus risk21.

For the vector control, define
`v_body=vector(conj(q_current)*[0,v_world]*q_current)` and emit
`q_current Exp(k v_body+(k/20)r_body)`. This adds a correction in one anchored
tangent chart; it is generally different from composing a CAV rotation with
a separate residual exponential. Synthetic noncommuting tests verify this
convention. The vector can predict another axis or nonzero motion from a
zero-speed context. Arc-constrained heads cannot.

All arms use deterministic CAV_W1 when `20||v_world||>=pi`. At exactly zero
speed arc heads emit the current quaternion. No extra near-zero threshold is
fit in this pilot. All20 outputs are normalized quaternions. Initial GPU
fitting is distinct from later ONNX/target-XR deployment verification.

## Fixed fitting and baselines

Five arms × three folds × two seeds20261026/20261027 =30 fixed fits. Use
Adam lr.001, betas(.9,.999), eps1e-8, weight_decay0, batch128 and exactly5,000
steps, deterministic float32 CUDA, no autocast/TF32 and no CPU training fallback.
There is no optimizer sweep, early stopping or accuracy-based checkpoint
selection. The primary result always uses step5,000; the secondary uses only
the predeclared training-time rule below. For each
fold/seed sample global fold-fit origins uniformly with replacement using
`default_rng(SeedSequence([training_seed,fold_id]))`; all arms use that exact
serialized schedule. Different heads cannot have identical shapes; record
their initial states and verify the cloned encoder separately.

Fixed train-diagnostic checkpoints are0,1000,2500,5000 on one65,536-origin
fitting-only subset per fold. Sampling is without replacement, RNG seed
20261028+fold_id. They describe optimization; their losses do not choose a checkpoint.
No inner-validation checkpoint sweep. Record all fixed diagnostics, including
inadequate optimization; a weak fit makes interpretation limited, not a
reason to silently add updates.

Fit static weights on a separately identified65,536 fold-fit origins,
without replacement, seed20261028+fold_id: the35-point simplex at increments
.25 over ZV/CAV_W1/W5/W20, minimizing actual emitted endpoint20 angle. Fit the
unconditional nearest-label histogram on exactly the **same static sample**,
uniform exact tie mass, smoothing0. Median/mean/MAP prior decoders share that
single histogram. The sample may coincide with the diagnostic sample under
the fixed seed, which must be recorded rather than portrayed as independent.

Score every predeclared inner-validation origin in batches256, paired across
methods and seeds. Record actual endpoint20 geodesic **degrees**, independently
of training loss. Methods are exactly the15 names in the JSON: three NLL
decoders, four other learned arms, static blend, three priors and ZV/CAV_W1/
W5/W20. Deterministic references have identical error vectors at both seed
keys. DESP/high-order filters remain required for any later strong baseline
paper claim; this bounded five-arm pilot does not claim SOTA.

## Metrics and decision rule

The primary pooled estimate is total angular error divided by total validation
origins across15 users, after arithmetically averaging the two fixed seeds'
**paired endpoint errors**, not quaternion predictions. Report each seed,
three fold means, user-macro mean, pooled P95 of the paired error means and
number of improved users. Millions of overlapping windows are not independent
participants.

Use2,000 whole-user bootstrap draws, seed20261028, the same user draws for all
methods. Recompute error/count ratios and relative improvement in each draw;
never bootstrap windows or divide a CI by a single fixed reference mean.
Intervals condition on these two seeds and fixed cross-validation fits. They
do not integrate independent retraining, fold selection or population/model
uncertainty. Inner folds are exploratory selection folds, not fresh confirmation.
Zero reference means yield undefined relative improvement (`null`), with the
number of invalid ratio draws reported rather than NaN/Infinity.

Primary contrasts: NLL median versus static, scalar_angle and risk21. The new
pilot screen is conjunctive across static, scalar_angle, risk21 **and**
scalar_pinball: at least3% and0.05° pooled improvement, strictly better error
direction for each fixed seed, and improvement in at least two of three fold
means for each comparator. CIs are reported; they are not an additional
retrospective gate. Vector-angle is mandatory: if it is better, report that
the constrained distribution did not establish competitive superiority.
Also report same-probability mean/MAP results and all cheap references, whether
favorable or not. No cherry-picked decoder becomes the primary model.

A threshold pass is an exploratory pilot result, not novelty certification,
challenge eligibility, a foundation-model result or an acceptance promise.
Failure does not prove that uncertainty or exposure information is universally
useless. Do not adapt the search, retune on the old dev users or open final
as a response to this pilot.

## Recorder API and public/private separation

`arc_metrics.summarize_arc_pilot(error_group,user_sums,user_counts,user_folds,
row_order_sha256=private_user_digest_map,resources=optional_records)` consumes
HDF5 `method/str(seed)/user` one-dimensional degree errors. Each dataset binds
to the same per-user origin-order digest. Root attribute is
`row_order_digest(private_user_digest_map)`, SHA256 of sorted compact JSON.
Private independent float64 sums and exact counts must agree with direct
vectors; all errors must be finite in[0,180]. Missing methods, seed/user
coverage, row bindings or unbalanced folds fail before a result is accepted.
The runner additionally verifies source membership/read guards; the metrics
API cannot authenticate human partition membership from an error vector alone.

Metrics compute paired mean-error rows internally and process one method at
a time; they never retain all15 methods' dense errors. Returned data contain
only aggregate counts, anonymous fold IDs, seeds, means/P95, macro means/CIs,
contrasts, gates, resource counts and scope. No person IDs, per-user counts/
errors, session names, private filenames, local paths or model weights go in
the safe aggregate. Per-user HDF5, schedules, fitted states and fold assignments
stay under ignored private data paths. JSON must serialize with `allow_nan=False`.

Optional public resource records contain only arm plus six cells with fold,
seed, synchronized training_seconds, validation_seconds and peak allocated/
reserved bytes. The metrics module whitelists fields; otherwise the runner's
independent safe resource aggregate supplies equivalent evidence. Dense/conv
MAC counts exclude geometry and decision overhead and must be labelled.

## Compute, interruption and readiness gates

Equal updates and sampling exposures are controlled; identical measured wall
compute is **not assumed**. Record per-arm synchronized training time, peak
VRAM, fixed parameter count and full-stateless conv/dense MACs. A strong
same-wall-compute characterization requires maximum/minimum ratio<=1.2 both
for total per-arm training time and each fold/seed cell. If exceeded, qualify
the claim; do not automatically run more fits or a tuning grid. Static and
neural fitting costs are not described as matched.

**Predeclared equal-compute secondary:** retain every immutable100-step state
and its cumulative synchronized training time. Within each fold/seed, set
`T=min(final5000_training_seconds_across_all_five_arms)`. For each arm choose
the last100-step checkpoint whose cumulative training time is<=T. This rule
uses only resource receipts, never a loss or accuracy value, and requires no
extra fitting. Evaluate those frozen states on exactly the same full inner
validation origins, with the identical fold-fitted baselines. Write a separate
error group and metrics summary under an explicit equal-compute-secondary
wrapper. Report actual steps/examples, selected/target times, next checkpoint
gap and rounding. Rounding is down by at most one100-step interval; the models
may consequently see unequal examples. If no checkpoint fits, report the
secondary as inconclusive rather than adding a fit or selecting on efficacy.
The original5,000-step result and conjunctive gate remain primary; a strong
positive paper interpretation must also survive this measured-training-time
secondary, otherwise qualify it. Quantized checkpoint matching is approximate
resource matching, not a claim that all end-to-end pipelines used equal work.

Pilot CUDA work must fit the current5 GPU-hour cap; overall research ceiling
is20 GPU hours. Project runtime from the first100 actual steps and measure
real loading/diagnostic/evaluation separately. Checkpoint every100 steps for
resuming the identical run, including model/optimizer/RNG/schedule state.
Budget interruption reports an incomplete pilot; it never promotes a shorter
checkpoint to the fixed5,000-step result. No paid/cloud compute.

Before real fitting: exact config/source freeze; synthetic model/metric/
runner tests; read-set allowlist and final/dev denial checks; segment-local
source-index fixtures; fold leakage assertions; finite actual loss gradients;
shared encoder/schedule hashes; and independent source review. Only then may
the coordinator execute the authorized GPU job. Afterward independently audit
safe aggregates, counts, numerical identities, hashes/read sets and resource
scope. The new paper, all-window trained export, clean Linux portability,
target-XR latency, competition eligibility and submission remain separate tasks.

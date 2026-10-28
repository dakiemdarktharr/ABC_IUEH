# Independent arc-pilot runner review

Review date: 2026-10-28. Scope: read-only review of the new experiment runner
and its synthetic tests before new human-data model fitting. The reviewer owns
only this report and does not change the runner, model, metrics or frozen
configuration. No human quaternion arrays, targets, caches, fitted artifacts or
participant-level results are opened. No model is fit and no final evaluation
is performed by this reviewer.

Final status: **no blocking error observed in the reviewed fresh-output pilot
path; conditional resume-hardening limitations remain below**. Production
started after the separate reviewer and full-suite checks passed. No source
or configuration was changed by this reviewer after that freeze. This report
does not establish scientific success, convergence, eligibility, registration,
submission or recovery from arbitrary manually altered private artifacts.

## Acceptance contract

The review checks whole-user inner folds inside the 15 outer-training users,
10 fit and 5 validation users per fold; train-only source reads; exact
segment-local origins 399 through N-21; causal context ending at each origin;
fold-fit-only static mixture and exposure prior; shared initialization and
origin schedules; frozen natural-unit losses; actual output20 scoring in
degrees independently of training losses; exact resume/source bindings;
budget accounting; and refusal to overwrite unrelated results.

The coordinator added a pre-efficacy secondary comparison at equal measured
training time: T is the minimum final training time across the five arms in a
fold/seed cell; each arm uses its latest saved 100-step checkpoint at or before
T, chosen using time only. It must reuse the same validation origins and cannot
trigger extra fits or accuracy-based checkpoint selection. This does not replace
the primary fixed-5,000-update comparison. Review will distinguish training,
validation, checkpoint-I/O and total budget clocks.

The five repeatedly observed development users and all 15 final users must
remain unavailable to this pilot. Whole-file integrity hashing is distinguished
from reading a participant quaternion dataset. Inner-validation outcomes remain
exploratory selection-fold evidence, not independent confirmation.

## Partial helper inspection, before final source readiness

The staged flat-window helper constructs N-419 origins for an eligible segment,
from399 throughN-21 inclusive. Its context is origin-399 through origin and its
target is origin+20; concatenated storage offsets do not connect distinct
segments. The fold constructor assigns each of15 users to validation once,
with10 complementary fitting users. Shared schedules sample only fitting-user
origins; static mixture and prior construction index the same fitting subset.
These observations are source inspection, not a completed integration test.

An independent bounded synthetic probe subsequently passed:30 synthetic
segments,480 origins,60 first/last boundary context-target comparisons, balanced
three-fold membership and three repeated schedule/membership checks. It used
only constructed rotations and pure helper calls; no fitting or human source
read occurred. This does not replace the final runner's resume and integration
tests.

The source guard now rejects a non-training iterator before entering the
underlying reader and separately blocks source quaternion values outside the
outer-training users. Historical cache origin-index integrity reads are
distinguished from feature/target reads. The reviewer requested synthetic
checks for guard restoration, before-read rejection and fold-reference
invariance when only validation-user rotations change. At that intermediate
point the final resume, equal-time evaluation and budget paths were still under
implementation; the completed inspection below supersedes that pending status.

## Frozen sources and verification receipts

The final full runner source inspected was `arc_experiment.py`, SHA-256
`db690aebd2bb9ec0a93349b02aff38a853806b964f3aef465729165b5ea9811d`.
Its test source was SHA-256
`157e270785ee5d21ed02794295e6832f92767fa1e786dbef92cd57df577153ad`.
The frozen configuration digest supplied by the author/coordinator was
`1ee9d2fff488ffa6952ce864177003706761b205901afddc69bca6951b26e753`.
The model source had previously been reviewed at SHA-256
`ddbe6fcd9cb59f112e19e2a4297c453ad3195169eb31615bdb9a5af96fea78dd`.

The following evidence has different provenance and is not conflated:

- This reviewer independently inspected the final runner, its tests and the
  relevant inherited source/geometry helpers, and ran the bounded synthetic
  window/fold/schedule probe above.
- This reviewer previously ran 40 model tests successfully, including five
  synthetic CUDA backward paths. That happened before production; no GPU
  test was run concurrently with production fitting.
- The independent model/metrics reviewer reports 67 model-plus-metrics tests
  and 9 runner tests passing. The runner author separately reports 9 synthetic
  runner tests passing in 13.44 seconds. These are attributed receipts, not
  executions performed by this reviewer.
- The coordinator reports the union suite passing 405 tests in 117.49 seconds
  at the frozen runner/configuration, before starting production.
- This reviewer's runner-test attempt returned 4 passes and 5 failures caused by
  access-denied renames in the environment's temporary directory. They occurred
  while creating synthetic fixtures or committing synthetic artifacts and do
  not demonstrate a model/runner logic failure. No real-data fit was attempted.
  The successful separate executions above establish that the directory issue
  did not reproduce in the coordinator's execution context.

## Final source findings

**Read sets and folds.** `train_only_hdf5_guard` refuses non-training iteration
before entering the inherited reader. Value hooks reject source quaternion
datasets outside the outer-training users. Cache value access is limited to
origin-index integrity checks; development indices are not development
features, targets or quaternion signals. The source has no final-partition
fit/evaluation route. The inherited stage verifies source/split binding and
user inventory. Whole-file SHA checks read opaque bytes and are not reported
as semantic final-array access.

Flat storage retains per-segment bounds and gathers 400 past observations plus
a distinct origin+20 target. Candidate angular costs are training labels, not
encoder inputs; tokens depend only on the chosen context. Every outer-training
user validates once; schedules use only 10 fitting users. Initial static/prior
construction uses the fixed 65,536-origin subset of those 10 users. Neither the
old all-15-user static weights nor a population scaler enters a fold fit.

**Controlled fitting and losses.** Each seed's models have matching encoder
digests and a common origin schedule. Useful head-capacity differences are
explicit. Frozen loss units remain NLL, radians for direct angle, normalized
pinball exposure and squared radians for risk regression. Train diagnostics
include CAV_W1/static actual-angle gaps on a fixed fitting subset, without
selecting checkpoints by their outcomes. They are not convergence evidence.

**Metric independence.** Actual float32 output20 wxyz quaternions are copied
to NumPy and scored by the separate geometry implementation in degrees. NLL,
risk and pinball losses are not substituted for that metric. Three NLL
decoders share one fitted probability tensor. Baselines use the same origins;
recordings carry per-user order bindings and counts. The other review owns
the detailed bootstrap/statistical acceptance.

**Resume and serialization.** Immutable zero-padded step files contain all five
model and Adam states, cumulative training clocks, losses, diagnostics,
source/config/schedule bindings and CPU/CUDA RNG states. Repeated schedules
are checked. A resumed unit loads its last serialized prefix; completed
receipts cannot be overwritten. The synthetic test compares final state hashes
between interrupted and uninterrupted fits. Validation binds model states,
reference values and row order, retains a committed prefix when incomplete,
and reopens completed errors read-only. Derived aggregation replays immutable
records after later serialization interruptions. This is clean-interruption
recovery, not a power-loss durability guarantee.

**Time and resource scope.** Production requires actual CUDA; CPU fallback is
rejected. Matmul/cuDNN TF32 are disabled, autocast is rejected, and deterministic
settings are recorded. Per-arm synchronized training clocks exclude shared
loading, diagnostics and checkpoint I/O. Broader session wall time charges
those operations, preparation and validation against the accumulated budget;
resumes charge repeated preparation. Checks pause at serialized training or
committed validation boundaries. Finite work can occur between checks, so this
is not a hard real-time cutoff exactly at 5 hours.

The primary remains fixed 5,000 updates. The secondary selects positive 100-step
prefixes using clocks only, with T the minimum final arm time in each fold/seed
cell. It validates finite increasing clocks, excludes step 0, and reports unused
time and the next-checkpoint gap. It does not use accuracy or trigger extra
fits. Prefix example counts differ. Fixed-update runtime ratios must qualify
any same-compute claim when measured times differ.

## Requests closed before production

The final source includes pre-slice partition refusal, validation-only
perturbation invariance of fold references, CAV/static diagnostic gaps,
positive timing-only prefix selection, rounding metadata, validation-prefix
recovery, completed read-only replay and derived-aggregate replay. Source
inspection and the attributed passing suites support those closures.

## Conditional artifact-binding limitations identified at freeze

Two additional observations were sent while the coordinator was freezing and
starting the separately reviewed production source. No code changed mid-run:

1. At `run_arc_pilot` lines 1086–1091, reloaded references are checked against
   global source/configuration bindings but their recorded fold/calibration
   membership is not independently revalidated. Global bindings are shared
   across folds. Manually copying another fold's reference file into that
   location could pass and import the wrong calibration set. **The initial
   fresh-output path computes the correct fold-specific subset**, and no such
   copy or wrong reference was observed. Human private files were not inspected
   to certify their contents. Future hardening should verify the fold, exact
   deterministic sample/digest and fitting-user membership before reload.
2. At line 1074, an existing private-fold manifest is retained without comparing
   it with recomputed folds/digest. Actual fitting uses recomputed folds, so
   this does not change the reviewed fresh fit. A manually changed manifest
   could nevertheless misdescribe provenance. Future hardening should reject
   mismatching saved manifests before further work.

These are conditional recovery/provenance gaps, not evidence of leakage or
incorrect predictions in the fresh pilot. Their resolution is not certified.
Repair after production starts requires the coordinator's documented
abort/repair/review procedure. This reviewer makes no efficacy-driven source,
configuration, fold, objective or checkpoint change.

## Scientific boundary

Software checks establish neither novelty, learnability, convergence, better
angular error nor challenge eligibility. The median's short-arc/snapped-target
limit and off-arc counterexample remain intact. Inner-validation is exploratory;
final users remain sealed. Paper go/no-go must use actual locked receipts and
all direct controls, not this review status.

## Post-run guard-only repair closure — 2026-10-28

**Status: the two conditional resume/provenance findings above are repaired
in the current source; 25 targeted CPU synthetic tests passed.** This is a
post-run implementation-author closure, distinct from the independent
pre-production review and the coordinator's separate final full-suite check.
No human signal array or measured error vector was opened for this repair.

The coordinator reports that production completed with exit0, all30 model
fits and both metric scopes were independently checked by
`arc_metrics_recorder`, and the actual fold/reference calibration schedules
were verified as correct. The measured source remains commit
`13c74071156b5b8dfc893f42107ad2cb82ea6921`; the exact executed nine modules and
configuration were archived under the private run's `measured_source/`
directory with matching hashes before this repair. This author did not alter
that archive or repeat efficacy fitting/evaluation. The historical working
runner hash used for measurement was
`db690aebd2bb9ec0a93349b02aff38a853806b964f3aef465729165b5ea9811d`.
The measured commit normalizes source line endings separately, as recorded
in the coordinator's provenance receipt. The new hash below must not be
presented as the source executed for those results.

The repair changes artifact reuse guards only:

- An existing private-fold manifest must match a freshly recomputed canonical
  assignment and digest, with exact keys, types, list lengths and values.
  In particular, boolean values cannot pass as integer fold IDs. Validation
  occurs before training source rotations are loaded.
- A reloaded reference must identify the correct fold and contain a sorted,
  unique, in-range, fit-only calibration schedule of the expected size and
  hash. Static weights and prior probabilities must be finite simplexes of
  the correct shapes.
- These local checks are not the ownership proof: the runner deterministically
  rederives the complete static/prior reference from that fold's fitting data
  and compares all stable serialized fields exactly. It uses the verified
  rederived arrays before any completed-unit skip. Copying a payload together
  with its own schedule hash or shared global source/config bindings is
  insufficient. Current reference metadata contains no elapsed timestamps to
  exclude from this comparison.

Fresh-path model initialization, losses, reference-fitting algorithm, origin
sampling, fold assignment and decoder calculations remain unchanged. The
current source hash intentionally differs, so the archived measured run is not
silently relabelled as a run of this later implementation.

The new `tests/test_arc_resume_guards.py` covers valid reference replay,
copied wrong-fold references, held-out calibration origins with an updated
self-hash, alternative fit-only schedules with an updated self-hash, modified
static/prior payloads, fake/recomputed fold digests, duplicated/reordered
memberships, boolean fields, extra keys and global-binding refusal before
reference rederivation. The original runner tests also passed, retaining
exact-state synthetic resume and immutable aggregate replay. Synthetic
integration fixtures use bounded CPU updates only; no human-data fit or GPU
experiment was rerun.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
.\.venv\Scripts\python.exe -m pytest tests/test_arc_resume_guards.py tests/test_arc_experiment.py -q --basetemp data/arc_postrun_guard_tests_20261028
```

Observed result: **25 passed in 10.14 seconds, exit 0**.

Coordinator verification after this repair: the complete local suite passed
**421 tests, 14 existing legacy-ONNX deprecation warnings, in 71.49 seconds,
exit 0**, with the required CUBLAS/thread environment and a new disposable
workspace `--basetemp`. No real-data efficacy fit or final evaluation was
performed by that synthetic test suite.

| Current artifact | Raw-file SHA-256 after repair |
|---|---|
| `src/abc_iueh/arc_experiment.py` |`801881315fd91960f8dbbe2c99f434b40be6234cbdab26767c9e947c5956b735`|
| `tests/test_arc_resume_guards.py` |`54f56b3b18444e2e1cc0b83df2efa305709c1d35d9f1de9046781fc433cf38a8`|
| Unchanged `tests/test_arc_experiment.py` |`157e270785ee5d21ed02794295e6832f92767fa1e786dbef92cd57df577153ad`|
| Unchanged `configs/arc_pilot.json` |`1ee9d2fff488ffa6952ce864177003706761b205901afddc69bca6951b26e753`|
| Unchanged `src/abc_iueh/arc_model.py` |`ddbe6fcd9cb59f112e19e2a4297c453ad3195169eb31615bdb9a5af96fea78dd`|
| Unchanged `src/abc_iueh/arc_metrics.py` |`c7b3a4cf645c8037748e71687bea524c2dba9603f362e2a43e66faecd2ca94b1`|

This closure fixes future artifact reuse. It does not change the measured
pilot's scientific interpretation, establish a new accuracy result, or imply
that the previously identified conditional gaps occurred in the fresh run.

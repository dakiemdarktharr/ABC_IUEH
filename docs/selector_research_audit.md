# Independent selector research and protocol audit

Audit date: 2026-10-28. Scope: the prospective 244-parameter selector study,
its methodological claims, nearest literature, implementation control flow and
synthetic verification. The auditor did not open human quaternion arrays,
feature/target caches, private participant statistics, fitted weights or new
real-data forecasting outcomes, and did not modify Git. Public earlier baseline
documentation was read as required historical context.

## Decision and study status

The design supports a complete **exploratory train/development paper** while
the 15-user final holdout remains sealed. It does not support a confirmation of
the original final hypothesis, formal final equivalence, hidden-test accuracy,
foundation-model capability, SOTA or target-XR latency. A complete manuscript is
not evidence that every experiment in the original 14-day plan was completed.

The latest user instruction supersedes the earlier plan to open final on Day 10.
Keep that chronology explicit: original prospective design; already observed
analytic/80k-MLP development results; prospective selector-stage extension;
fixed selector experiment; exploratory report. Do not rewrite the historical
preregistration to imply the final analysis happened. The original practical
thresholds remain unchanged and **not tested on final**.

An acceptable narrow contribution is a controlled empirical comparison of
matured expert-error horizons and feedback ages in a frozen, compact selector
under the stated Alyx user-disjoint protocol. The result may be positive,
negative or inconclusive; whichever occurs, retain the same question, report
every fixed variant/seed and distinguish effect estimates from speculation.

## Documents and independent boundaries

Read before review: parent/repository AGENTS, ABC_CONTEXT, canonical Obsidian
index/groundtruths/ambiguities/task-interface/evaluation/submission/rules,
repository README, data_sources, protocol, baseline_feature_plan,
baseline_results_2026-10-28, baseline_stage_review, original preregistration,
research_decision/review_response and worker literature/novelty reports.
Reviewed new `configs/selector_study.json` and `docs/selector_study_plan.md`.
The literature review used a source ledger in the local topic-discovery work.
That working file is not part of this public repository.

The literature audit uses primary papers and author/publisher repositories.
It is a bounded update through2026-10-28, not evidence that no other prior
method exists. Source-use decisions in data_sources remain in force: BOXRR is
disabled pending its recipient gate; Alyx-only competition eligibility remains
unconfirmed. Registration status and organizer confirmation were not
independently verified by this auditor. No email or submission was sent by
this auditor.

## Exact method contract

For expert m and current origin t, evidence is

```text
E_m(t;h,a) = (1/R) sum[r=0..R-1]
             d(P_m(t-a-h-r,h), q[t-a-r]), R=80.
v_W(s) = Log_SO3(q[s] inverse(q[s-W])) / W, W in {1,5,20}.
P_W(s,k) = Exp_SO3(k v_W(s)) q[s]; ZV uses v=0.
```

Every evidence target is at or before t. Future endpoint20 expert errors are
training labels only. Complete-Horizon uses the same issue origins `t-20-r`
and averages all k=1..20 matured errors before averaging over R origins.
All methods use identical origins 399..N-21, with observed 400-sample context
ending at t and endpoint target `q[t+20]`, inside one continuous segment.
Replay and inference use world-frame velocities with left-composed increments;
this must not be casually relabelled body-frame velocity.

| Contrast | What is matched | What remains different |
|---|---|---|
| h20,a0 versus h1,a0, primary | Target indices `t-r`, error count R, expert bank, current motion features, model capacity, labels and optimization budget | Forecast issue origins differ by 19 samples; raw score distributions differ |
| h20,a0 versus h1,a19 | Issue origins `t-20-r` | Mature target indices differ by 19 samples; horizon changes |
| h20,a0 versus h20,a20 | Forecast horizon | Target and issue age both change by 20 samples |
| h20,a0 versus Complete-Horizon | Latest issue origins and bank | CH mixes horizons1..20 and target ages0..19 within each origin |
| Evidence versus motion-only | Ten inputs and 244 parameters | Feature meaning differs; motion-only uses ten substantive motion statistics |

Issue age is `h+a`; horizon, target age and issue age are not three independent
factors. These contrasts cannot identify three separate causal effects. Use
“matched target age” or “matched issue age”, not a claim of full causal
disentanglement. At R80 the worst h20,a20,W20 history extends to `t-139`, within
the 400 observations. R128/R256 sensitivities also fit the declared context;
they require separate bound caches, not edits to the frozen R80 bundle.

The six selectors use 10 → 16 ReLU → 4 softmax. Parameter count is
`10*16+16+16*4+4=244`; dense matrix MACs are `10*16+16*4=224`.
The 224 count excludes biases/activations, feature construction, expert replay,
SO(3) geometry and all 20 outputs. It is not total per-streaming-step cost.

For standardized feature x, weights p=softmax(f(x)) and future individual
expert losses ell, training minimizes `mean(sum_m p_m ell_m)` in degrees.
This is a competence-selection surrogate. Inference emits
`Exp(k sum_m p_m v_m) q[t]` for k=1..20, normalized/quantized to float32.
The reported accuracy must score this emitted quaternion at k=20.
Expected expert loss is neither this score nor generally a bound on it: opposed
expert velocities can cancel to a correct blend while both individual experts
have substantial error. Report expected-loss and hard-argmax diagnostics
separately. No Euclidean expert-aggregation regret guarantee is inherited.

## Matching, fitting and leakage audit

The frozen stage specifies 2000 SGD steps, batch 128, lr 0.001, no momentum/weight
decay, seeds 20261026/20261027, fixed final-step checkpoints, no early stopping.
Within a seed, every variant clones the same initial tensors and reuses the
same uniformly sampled global training-origin schedule with replacement.
Normal(0,0.02) weights and zero biases follow the earlier synthetic prototype;
this is a stated stage choice, not an organizer requirement.

Five learned variants existed in the original preregistration. Learned h1,a19
is a sixth, explicitly recorded **before selector fitting**, after earlier
analytic h1,a19 development outcomes were known. Its addition improves the
learned issue-age comparison but is not retrospectively part of the original
five-variant plan. The h20,a0-minus-h1,a0 primary contrast is unchanged.
Predeclared R128/R256 primary-pair sensitivity retains data, model, steps,
batches and seeds; report omission or failure if it is not actually run.

Each variant reuses its existing population mean/std fitted only on training
windows, with a 1e-8 floor, float64 standardization followed by float32 inputs.
No per-user fitting/scaling, development fitting or holdout pretraining is
permitted. Same initialization and batch schedule do not make differently
scaled feature vectors identical; the comparison concerns their usefulness
under this specified training recipe, not all possible learners or convergence.

The reviewed code requires the exact saved stage config and sealed baseline
bundle, source/split/cache/scaler hashes, lexical segment order and complete
stride 1 coverage. Source checks independently recompute current/endpoint20
targets, expert velocities and expert-loss labels for train/development only.
Training loads only training feature/label tensors. All 12 checkpoints are fixed
before the guarded development source reader is reached. No final-partition
option exists in selector fitting/evaluation/metrics entry points.

The final evaluation helper computes only endpoint20 to avoid allocating the
other 19 outputs. An exact float32 parity test on 513 noncommuting synthetic
rows compares it with output20 of the full 20-quaternion wrapper; this passed.
This endpoint-only scoring optimization does not demonstrate exported graph
parity or change the required full output interface.

Whole-file checksum reads are opaque integrity checks, not an empirical
final-data analysis. This audit does not claim that metadata binding alone
proves all feature semantics: the synthetic replay/suffix-invariance checks
provide separate evidence for those semantics.

## Statistical and reporting contract

Primary mean is total endpoint20 degree error divided by total windows.
Bootstrap 2000 draws resample whole users, preserving all their sessions and
overlapping windows; each draw recomputes the ratio and is shared by methods.
Average the two fixed seeds' **error vectors**, then aggregate/bootstrap;
do not describe this as a prediction ensemble. Report each seed and its spread.
User-macro means and pooled-window P95 are separately named statistics.

With five development users, intervals are exploratory and conditional on the
trained models/two fixed seeds. They do not integrate retraining-population or
seed uncertainty. Millions of overlapping windows do not yield millions of
independent participants. Report all secondary contrasts as exploratory
effect/CI diagnostics; do not silently promote them to confirmatory tests or
use many nominal 95% intervals to advertise a discovery.

A development CI containing zero is not proof of no effect. A CI inside the
predeclared margin can be described only as a conditional development
illustration; it is not the final equivalence decision. Even a development
contrast meeting all original numerical thresholds does not establish final
success. Train errors are in-sample, and the prior best-observed development
CAV reference is descriptive rather than an unbiased selected-model estimate.

The metrics consumer checks finite/range/counts, exact paired seed means and
private user sums against flattened cache rows before returning aggregates.
Public outputs must exclude participant IDs, manifests, feature/target/error
vectors, sampled schedules, scalers and weights. Raw source and derived
recordings remain outside Git. Public code/aggregate authorization does not
relicense the source data or establish challenge eligibility.

## Literature implications for the paper

Cerqueira–Torgo explicitly studied IH/CH/FHF/LHB and dynamic versus static
forecast combinations; the published AALTD chapter is 2026, with an earlier
2023 preprint. ADE already predicts expert competence. These sources rule out a
new ensemble principle claim. Our R80 quaternion operationalization must be
credited and defined, not labelled an exact reproduction of their numeric
forecasting algorithms. See the [published chapter](https://link.springer.com/chapter/10.1007/978-3-032-15535-1_9)
and [ADE](https://doi.org/10.1007/s10994-018-05774-y).

Yoon's primary indexed introduction confirms latest-error learned weights, but
full feedback indexing could not be read after bounded retrieval attempts.
Do not contrast our mature 20-step evidence with a claimed one-step-only Yoon method.
HAP already has a horizon-wise CV/learned gate; Illahi already studies relative
pose, classical predictors and preprocessing leakage. Input/task differences
explain comparator limitations, not novelty by themselves. Exact citations,
reading depth and gaps are in the separate source ledger.

Missing DESP/tuned rotation-filter comparison and native-rate sensitivity limit
mechanistic interpretation. Without them, an effect cannot be attributed
uniquely to anticipation rather than derivative bandwidth or 90 → 100 Hz SLERP
smoothing. If not implemented, call these limitations explicitly; do not say
all preregistered controls were completed. Optional JEPA/EBM remains blocked
until the user's later approval after the main paper.

Likewise, the original dossier discusses a wider h×a analytic grid, shuffled
evidence, calibrated softmax, selection regret and event/bin diagnostics.
The six-selector R80 stage alone does not complete that wider list. The paper
must identify which were actually executed; unrun controls remain limitations
or follow-up work, rather than silently becoming successful completed checks.

## Verification receipts and findings

The existing evidence/evaluation contracts passed an independent synthetic run:
**24 passed in 9.80 s**. This covers explicit noncommuting replay, issue-age
matching, future-suffix invariance, quaternion sign/global-frame behavior,
actual float32 labels, substantive motion features, train-only calibration and
paired ratio bootstrap. Synthetic final-quaternion sentinels remained intact.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evidence.py tests/test_evaluation.py -q --basetemp .pytest_cache/research_auditor_20261028_01
```

The first selector-specific independent run gave 53 passed / 2 failed in 12.52 s:

1. `_cuda_runtime('cuda')` passed an index-free `torch.device('cuda')` to
   `torch.cuda.set_device`, which PyTorch 2.9 rejects. This was a real launch
   blocker, not a study-design change. The implementer resolved missing index
   to current device before setting it.
2. A synthetic expected trajectory skipped the production normalization
   sequence and used exact equality, causing a maximum 5.96e-8 float32 component
   discrepancy. The implementer replaced that comparison with a documented
   geodesic tolerance consistent with the float32 contract.

Both fixes were independently verified. A subsequent run gave 55 passed / 1
failure because the implementation source changed while its full synthetic
integration ran; the intended source-stability guard aborted the run. After
the implementer declared source freeze, the complete independent selector
suite passed **58 tests in 16.23 s**, with no failure or skip. This includes
actual CUDA computation, CPU/CUDA one-SGD-update parity, analytic/finite-
difference surrogate gradients, full-wrapper/endpoint parity, exact shared
initialization and origin schedules, sealed-bundle refusal, altered target/
velocity/label refusal, private aggregate hygiene, exact seed means, and the
full fixed 2000-step synthetic runner. Its final sentinels were not read.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -m pytest tests/test_selector.py tests/test_selector_metrics.py tests/test_selector_experiment.py -q --basetemp .pytest_cache/research_auditor_selector_20261028_03
```

The process-local cuBLAS setting enables deterministic dense CUDA operations.
The coordinator reports that the production CLI sets it before importing
Torch; this matches the required initialization order. No unresolved critical
implementation or protocol blocker remains for the authorized R80 train/dev
run at this review point. This is a correctness/authorization boundary review,
not independent verification of real-data efficacy.

Reviewed source fingerprints after the successful run:

| File | SHA-256 |
|---|---|
| `src/abc_iueh/selector.py` | `7ce8a292b562a4b3d07e34bcb83b04b1ca4b80b13087ed5ecb51f72af80fa83f` |
| `src/abc_iueh/selector_experiment.py` | `58604ef6d4522935560726bb3a2cd1d3c76ef199c2936c648a6f88b16a456e24` |
| `src/abc_iueh/selector_metrics.py` | `4f80ea3a25d22a6d8b31e2d7cbcf94ec6d069b3274e36aecd6714580d4eed430` |
| `configs/selector_study.json` | `13f313a391504bae77a0e4bbc6de53d0383349c384a76ef984f46e4d0527e598` |

An initial test fixture also encountered sandbox-TEMP atomic-rename denial;
the fresh workspace-local basetemp resolved it without code changes. This is
an environment finding, not predictive evidence. All audit executions use
synthetic inputs. The reviewer has not independently confirmed real-data
efficacy, full GPU resource figures, end-to-end ONNX parity, clean Linux loading
or XR latency.

## Official recheck and remaining boundaries

The live organizer [Challenge](https://sites.google.com/view/9th-abc-challenge/home/the-challenge),
[Data](https://sites.google.com/view/9th-abc-challenge/home/the-data),
[Participation](https://sites.google.com/view/9th-abc-challenge/home/participation),
[Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts)
and [FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs) were read on
2026-10-28. No organizer clarification resolving 20 samples / 100 Hz / 100 ms was found.
Retain samples, 400 → 20 static float32 wxyz, shift 1 and endpoint20 mean degrees.
Do not silently substitute 10 outputs or 200 Hz.

Dense/ReLU are on the safe-operator list; Softmax is discussed in the export
guidance. Trig/arccos are described as supported but potentially slow. Passing
selector-core training on the RTX 5050 does not establish full-graph export or
organizer-device latency. Packaging, target-compatible graph inspection,
offline inference and model-card cost accounting require their own receipts.

Minimum actual context, threshold equality and the PT/TF-versus-required-ONNX
packaging wording remain unresolved. The manuscript can state its local
400-sample choice and exploratory scope while preserving these unresolved
official statements. It must not advertise an eligible completed challenge
submission from a local paper/code result.

## Second audit wave: evidence-length sensitivity and inference portability

This extension reviews `selector_sensitivity.py`, its frozen configuration,
prospective plan and tests, plus the separate `selector_inference.py`, export
tests, design note and `configs/inference_audit.json`. This auditor continues
to use synthetic fixtures only: no human arrays, real fitted weights or new
individual results were opened. The completed real R80 study was not rerun,
modified or used to select R or an inference checkpoint.

### Sensitivity gate

The R128/R256 child design is coherent with the original declared sensitivity.
Both lengths and only h1,a0/h20,a0 are required. Each child preserves every
parent training/development origin, user/session/segment boundary, current and
endpoint quaternion, expert velocity and future expert-loss label. Those shared
values are checked exactly against the sealed R80 cache. The child schema
rejects final groups, extra variants, external/virtual storage, changed dtypes,
nonfinite values, altered origins and inconsistent counts. Its own validator is
used, rather than incorrectly applying the six-variant R80 validator.

New raw evidence features depend on R; population scalers are fitted separately
for each R/variant using training windows only and are independently recomputed
from the child's training features during validation. Labels remain separate.
The complete parent-main receipt, artifact/checkpoint hashes, fixed schedule
arrays and all 2000 losses per fit are required before any child fitting.
Child training reuses both seeds, 2000 SGD steps, batch 128, learning rate 0.001,
initialization and global origin draws. Actual child schedule/initial-state
hashes must match the completed main run; short synthetic fits cannot be
misreported or sealed as the 2000-step experiment.

Source/split, parent cache/scaler/receipt, main/sensitivity configurations,
prospective plan and implementation files are bound before and after work.
Output directories are exclusive, parent artifacts are read-only, and failed
partial children cannot silently overwrite or reseal evidence. All public
summaries remain aggregate; both R results must be reported without selecting
a favorable R or checkpoint. User-cluster uncertainty and exploratory-only
development interpretation remain the main study's contract.

**Independent sensitivity gate: PASS, all 42 tests.** The combined wave run
finished with 85 passed and 6 failures in 47.25 s; all six failures were in the
separate inference export tests described below, so every sensitivity test
passed. The sensitivity suite includes genuine full 2000-step main and child
SGD fits on synthetic data, exact matched schedules/initialization, immutable
parent byte hashes, independent train-scaler recomputation, artifact and schema
tamper refusal, and final-user sentinels. No unresolved critical sensitivity
implementation/protocol blocker remains for the authorized train/dev run.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -m pytest tests/test_selector_sensitivity.py tests/test_selector_inference.py -q --basetemp .pytest_cache/research_auditor_wave2_20261028_01
```

### Inference contract and numerical gates

The inference module constructs a copied frozen selector and scaler and reads
no checkpoint or recording itself. Forward receives only static float32
`context[1,400,4]`, calculates past-only replay by fixed gather indices, blends
world velocities and returns all 20 normalized wxyz predictions. It has no
cross-call/session state or inference-time fitting. Individual-horizon, CH and
motion-only indices match the offline definitions. The maximum supported
delayed history remains within context index 0; prospective R128/R256 pair
inference is also representable.

The graph computes geometry, scaler constants and standardization in float32.
It therefore approximates the experiment's float64 geometry and scaling with
intermediate float32 rounding. Claim numerical agreement within recorded
angular tolerances, not bit identity of raw features or equivalence at 1e-12.
Near-zero velocity cosine differences can be larger than output-angle drift;
all such observations must be reported rather than hidden by float64 export.

The prospective `inference_audit.json` fixes seed 20261026 for every model
variant, without development checkpoint selection. Allowed real portability
sampling is 26 complete contexts per train/development user, uniformly without
replacement within user, seed 20261028. Final users are excluded. The numerical
gates are maximum synthetic and maximum sampled-real full-trajectory drift of
0.02 degrees, plus absolute sampled mean endpoint20 score change of 0.01 degrees.
These govern portability and must not replace the research success thresholds.
Sampling and actual trained-weight/real-context measurement belong to the
coordinator's later separate run; this auditor has not performed them.

Export explicitly requests opset 17, fixed names/shapes, float32, no dynamic
axes or external weight file. Actual ONNX checking/shape inference and CPU
Runtime execution are tested. The graph inspector rejects Loop/If/Scan,
NonZero/TopK, custom domains, non-float32 floating types and sequence types;
tests also verify no ArgMax, static intermediate shapes and the two dense
layers/one softmax. Static operation inventories are an arithmetic proxy before
provider optimization, not organizer XR latency or whole-graph dense MACs.

An exact-half-turn alternating context exposed a genuine export regression:
legacy atan2 lowering mishandled a negative-zero canonical scalar. The
independent combined run loaded the pre-fix module and reproduced six failed
export cases; h1,a0 maximum trajectory drift was about 46.00093 degrees. The
author changed the already nonnegative canonical scalar to its absolute value,
forcing positive zero before atan2. This preserves the intended principal-log
definition and avoids changing the model or numerical threshold. Independent
post-fix graph verification is recorded below once the inference source is
frozen. The expanded standard suite asserts the 0.02-degree gate across all six
variants, three seeds and five stationary/tiny/ordinary/wide rotation scales,
as well as exact half turns.

### Final second-wave verification

After the inference author froze the signed-zero fix, the auditor ran a fresh
process and independent temporary export directory. **All 49 inference tests
passed in 14.44 s, with no skip.** Twelve warnings describe the deliberately
selected legacy ONNX exporter and its logging API deprecation; they are not
checker/runtime failures. All six variants were exported at opset 17 and
executed in CPU ONNX Runtime. Their names, static input/output/intermediate
shapes, float32 floating types, absent forbidden/custom/control-flow operators,
finite unit outputs and full-trajectory numerical gates passed inspection.

The expanded suite checks all 90 combinations of six variants, three seeds and
five stationary/tiny/ordinary/wide synthetic rotation scales against the
offline reference, plus the six exact-half-turn export cases. The 0.02-degree
maximum trajectory threshold passed for every case. The earlier signed-zero
failure is resolved, rather than omitted or accommodated by changing a gate.
The design note separately reports the author's observed maxima; this
independent receipt establishes the assertions on the frozen code, and does
not label the author's exact maxima as independently re-estimated measurements.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.\.venv\Scripts\python.exe -m pytest tests/test_selector_inference.py -q --basetemp .pytest_cache/research_auditor_inference_20261028_02
```

No unresolved critical source/protocol blocker remains for the coordinator's
authorized train/development sensitivity execution or numerical portability
audit. Actual trained-checkpoint and sampled-real trajectory/score drift remain
unmeasured by this reviewer. They must pass the frozen 0.02-degree maximum and
0.01-degree absolute sampled mean endpoint-error change gates, reported
separately for train/development, before claiming the portable graph preserves
the measured predictor sufficiently. Real weights, scalers, sample schedules
and context/target arrays remain private. Local CPU/RTX cost receipts cannot
establish organizer XR latency or clean-Linux submission readiness.

Second-wave source fingerprints verified after testing:

| File | SHA-256 |
|---|---|
| `src/abc_iueh/selector_sensitivity.py` | `1351b8aad432a203fc3470e49d2858544e86812b6e9c5ce57fa797e2bb858071` |
| `tests/test_selector_sensitivity.py` | `53c52c9e9f07e3eed203328890cb4d9f1e59588914427b85f4eab0077ff3e108` |
| `configs/selector_sensitivity.json` | `42a7c5af74c96735a12e36ec6a1e30fbf70f92364ac9f3c2507ae0fcde47f27b` |
| `src/abc_iueh/selector_inference.py` | `103f41babfa32114d2f4c3389a1fb79b3656896498164f4532b5fa97c91b09a7` |
| `tests/test_selector_inference.py` | `da51eaae234218f82c1d95a37777426e0eafd84674641759563fcfc74566e2d9` |
| `configs/inference_audit.json` | `b1a19605be10d1bba298cba698e5692bbcdcd638e3084e5a2c672275138e542b` |

Only this research-audit document was changed by the auditor in the second
wave. Synthetic test artifacts remain in ignored temporary directories; no
human data, real weights, private individual outcomes or Git state were read
or changed by the audit.

# Full selector inference graph

Implementation and synthetic validation date: 2026-10-28. This document records
an engineering implementation, separate from official challenge requirements
and from real-data model results. The canonical requirements remain in the
Obsidian ABC notes. The20-sample /100 Hz /100 ms inconsistency is unresolved;
this implementation preserves sample indices and the400 ->20 interface.

## Interface and frozen inputs

`FullSelectorInference(selector_state_dict, scaler, variant="h20_a0",
evidence_length=80)` receives one fixed244-parameter selector state dictionary
and its matching, already fitted training scaler. Extract `checkpoint["state_dict"]`
after a trusted local `torch.load(path, map_location="cpu", weights_only=True)`;
the module itself never loads a checkpoint or opens a recording. Pass one
variant's scaler, with `fit_partition="train"`, `ddof=0`, `epsilon=1e-8`, and
ten finite `mean`/`scale` values. Constructor checks do not fit or update it.

`forward(context)` accepts only float32 `[1,400,4]`, scalar-first wxyz, and returns
float32 `[1,20,4]`. Inputs must satisfy the official finite, nonzero, unit and
hemisphere-aligned quaternion precondition. Eager execution checks shape/dtype;
tensor values are not copied to the host or validated with data-dependent Python
branches during forward. It has no persistent session state. Every replayed
forecast and mature error comes from the supplied400 observations, so a new
session naturally replaces all history. No per-user gradient update occurs.

The constructor copies and freezes the four learned tensors. Train scalers are
constants, rounded to float32 for export. Every forward operation stays on the
input device. NumPy is used only during construction/explicit offline reference
and graph inspection; forward uses tensor operations and literal constants.
`raw_features(context)` and `selector_weights(context)` expose diagnostic paths
that share the full forward implementation. They are not extra exported I/O.

## Causal feature construction

Let context index399 be the current origin, R the evidence length, and bandwidth
W be1,5,20. World-frame expert velocity is
`Log(q[j] * inverse(q[j-W])) / W`; ZV contributes the zero vector. Log chooses
the principal rotation and uses the first nonzero vector component positive at
an exact half turn. Explicit nested `Where` expressions implement that tie;
there is no ArgMax-based expert selection.

For individual-horizon evidence `(h,a)`, replay origins are
`399-a-h-r`, targets are `399-a-r`, and `r=0..R-1`. Each of the four experts
emits a normalized quaternion at horizon h. Their shortest geodesic errors in
degrees are averaged over R. Supported variants are h1_a0, h20_a0, h1_a19 and
h20_a20. R80 is primary; the same constructor supports prospective R128/R256
h1/h20 sensitivities. R1..340 is accepted, matching the existing feature builder;
h20_a20,R340 reaches context index0 exactly, including its W20 lookback.

Complete-Horizon uses origins `399-20-r` and every horizon k1..20 at each origin,
with target `origin+k`. A constant gather-index tensor and broadcasted
`[1,R,20,4,3]` velocity/horizon computation evaluate the whole replay at once.
There is no horizon recurrence or unrolled20-step dependency. Individual-horizon
variants use the same layout with one horizon. Four mean error columns precede
the three current speeds and three pairwise velocity cosines.

Motion-only uses those six current motion columns, three norms of pairwise
velocity differences, and `||Log(q[399]*inverse(q[299]))||/100`. Cosine products
use the existing1e-12 denominator floor. All ten columns are standardized by the
supplied frozen training statistics, then passed through10 ->16 ReLU ->4
Softmax. The four weights blend world angular velocities. Output k is
`normalize(Exp(k * blended_velocity) * q[399])`, k1..20. This is the emitted
prediction whose endpoint accuracy must be measured; the training surrogate
remains a separate expected individual-expert loss.

## Numerical contract

The offline experiment normalizes and calculates geometry in float64, rounds
each replayed expert quaternion to float32 before its float64 geodesic error,
stores raw features as float32, and applies scaler arithmetic in float64 before
rounding again. A float32-only portable graph cannot reproduce all of those
roundings bit for bit. It uses the same mathematical definitions, but computes
geometry, mature errors and normalization with float32 tensors and constants.
Neither this implementation nor its tests claims equality at1e-12.

Quaternion normalization uses a scale-normalized norm. Log and geodesic distance
use atan2 of vector norm and scalar; Exp uses the analytic zero-angle limit.
Grouping the Hamilton product's cross-product terms preserves exact vector
cancellation for a stationary normalized quaternion in eager float32 arithmetic.
The canonical log scalar also passes through Abs to turn signed negative zero
into positive zero. An exact-half-turn regression exposed that legacy ONNX
atan2 lowering otherwise divides by negative zero and returns the wrong sign;
this correction preserves the principal-log definition. The exported graph
still needs provider-specific numerical checks.

Before real checkpoint/export measurements, the coordinator fixed these
engineering gates: maximum synthetic output quaternion angular difference from
the offline reference <=0.02 degrees, maximum sampled real trajectory angular
difference <=0.02 degrees, and absolute change in the sampled mean endpoint20
error <=0.01 degrees separately for training and development. The frozen
`configs/inference_audit.json` specifies26 contexts per allowed user and the
sampling seed before these measurements. These are numerical portability
checks, not model-selection or accuracy-success thresholds. A failure must be
reported without modifying a fitted selector or its training loss to pass it.
The original study's practical improvement thresholds are unchanged.

`reference_from_segment(...)` calls the existing float64 feature builder and
full20-output blend helper. Its segment must include400 context observations
and20 targets at the requested origin because the builder also returns labels;
those targets do not enter the features, weights or outputs. This helper does
not authorize access to any partition. Synthetic tests perturb future suffixes
and unused context prefixes, check independent calls leave no state, and check
quaternion sign invariance and world-frame equivariance.

The following is **synthetic evidence only**, from fabricated stronger selector
weights (the deterministic initializer's two weight matrices multiplied by10,
fixed output biases), fabricated train-tagged nontrivial scalers, six R80
variants, three seeds118/228/918 and increment scales0,1e-10,1e-7,0.018,1.8 radians
per component:90 cases, each with20 outputs. A separate exact-half-turn case is
also in the test suite. These are neither real fitted weights nor real motion.

| Observed maximum over the90 cases | Value |
|---|---:|
| Eager float32 versus offline output angle |0.000369802 degrees |
| ONNX Runtime CPU versus offline output angle |0.002302892 degrees |
| ONNX Runtime CPU versus eager output angle |0.002400859 degrees |
| ONNX Runtime output norm deviation |1.192093e-7 |
| Raw feature absolute difference, any column |0.002882914 |

The largest raw-feature difference occurred in a near-stationary cosine column
at increment scale1e-7, with output drift about1.08e-5 degrees for that case.
The largest ONNX output drift occurred for the deliberately wide-angle
Complete-Horizon case, seed918. Raw feature parity therefore must not be
described as uniformly near1e-6, even though all output-angle gates passed.
The eager/ONNX differences are measured honestly rather than erased by a
float64 export. Real trained-checkpoint and allowed train/development parity,
including the effect on actual endpoint20 error, remain a separate coordinator
measurement. Final-holdout data were not accessed for this implementation.
The additional alternating exact-half-turn context passed all six variants;
its maximum ONNX/offline trajectory difference was0.001464681 degrees.

## Export, operator inventory and cost

`export_full_selector_onnx(model, path, example_context=None)` uses the installed
legacy `torch.onnx.export` path explicitly (`dynamo=False`), opset17, no dynamic
axes, and no external weight file. Existing paths are refused. The exporter
runs the ONNX checker and static shape inference, verifies exact names/shapes
and float32 graph I/O, and rejects the listed control-flow/selection/custom
operators, non-float32 floating tensors and sequence types. `inspect_onnx(path)`
can independently reproduce the inventory from an existing graph.

With local PyTorch2.9.1+cu128, ONNX1.18.0 and ONNX Runtime1.22.1, synthetic R80
exports contained only FLOAT, INT64 and BOOL tensors; all inferred intermediate
shapes were static. The individual-horizon and Complete-Horizon graphs each
had605 nodes; motion-only had530. Their synthetic-weight file sizes were56,218,
68,455 and45,725 bytes respectively. Real artifact bytes/hashes must be recorded
from the exported real checkpoint. Constant indices make CH larger even though
its graph node count equals the individual-horizon count.

Observed operator types across the six graphs were Abs, Add, And, Atan, Clip,
Concat, Constant, Cos, Div, Equal, Gather, Gemm, Greater, Less, Mul, Neg, Not,
Or, ReduceMax, ReduceMean, ReduceSum, Relu, Reshape, Sin, Slice, Softmax, Sqrt,
Sub, Unsqueeze and Where. The exporter lowered atan2 to Atan/Div/Where with
quadrant logic. There were no Loop/If/Scan/NonZero/TopK, custom domains or ArgMax.
Some operators lie outside the organizers' explicitly named safe examples;
observing successful CPU ONNX execution does not prove target-XR portability
or latency. Trigonometry is specifically a potential deployment cost in the
canonical submission note. No target-XR or CUDA ONNX Runtime measurement is
claimed; this local ONNX Runtime installation exposes CPU execution only.

Each streaming step recomputes one full stateless400-to20 graph. The selector
has244 learned parameters and224 dense MACs (`10*16+16*4`), excluding biases,
Softmax, evidence, experts, SO(3), indexing and normalization. The following
mathematical counts describe the actual exported static graph before provider
optimization. They are not a total MAC estimate or hardware instruction count.

| R80 cost component per streaming step | Individual h1/h20 variants | Complete-Horizon | Motion-only |
|---|---:|---:|---:|
| Selector dense MACs |224|224|224|
| Elementwise multiplications outside Gemm |28,726|387,446|2,449|
| Elementwise Add/Sub plus reduction additions plus Gemm biases |19,304|267,064|1,744|
| Elementwise divisions plus ReduceMean divisions |15,676|185,916|3,603|
| Square roots |2,629|34,549|475|
| Sine evaluations |340|6,420|20|
| Cosine evaluations |340|6,420|20|
| Atan evaluations |563|6,643|4|
| ReduceMax comparisons |5,169|64,449|1,332|
| Four-way Softmax nodes |1|1|1|

For an elementwise operator, the count is its inferred output tensor's element
count, summed over nodes. ReduceSum/Mean addition count is input elements minus
output elements; ReduceMean also adds one division per output. ReduceMax uses
input elements minus output elements comparisons. Gemm bias additions total20.
The inventory separately exposes Abs/Neg/ReLU and all operator-output element
counts, including Clip, Boolean comparisons, Where and memory/shape operations.
These additional operations and Softmax's internal implementation are excluded
from the arithmetic rows above. Provider fusion, vectorization, constant-folding
and exact kernel implementations may change actual work. Measure full-graph
warmup/synchronized runtime on the device being reported; do not substitute the
224-MAC core for full streaming-step cost.

## Reproduction boundary

Run `python -m pytest -q tests/test_selector_inference.py` in the repository's
environment. These tests create only synthetic arrays and temporary ONNX files;
ONNX/Runtime checks skip when optional dependencies are unavailable. They do
not open any HDF5, baseline cache, participant manifest or real checkpoint.
The verified local run passed49 tests, including all six ONNX/Runtime variants;
the legacy exporter emitted its expected deprecation warnings.
The source implementation, test file and this design note were the only
implementation-owned files changed. Existing stage/selector/evidence/geometry
code and running experiment inputs were left unchanged.

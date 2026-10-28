# Fixed inference measurement protocol

Recorded2026-10-28 before the real inference audit. The frozen
`configs/inference_audit.json` and the source hashes saved in the audit plan
define this engineering check. Its purpose is numerical portability and local
full-graph cost, not model selection or a new accuracy experiment. It preserves
the unresolved20-sample /100Hz /100ms specification and static float32
`context[1,400,4] -> prediction[1,20,4]` wxyz interface.

The implementation is `abc_iueh.inference_audit.run_inference_audit(stage,
baseline_dir, main_dir, output_dir, config, config_path=...)`. Use
`LockedDevelopmentStage` with the existing locked source/split. The normal
private output is `data/inference_runs/20261028_fp32_audit/`. Existing output
directories are refused. The recorder requires the completed R128/R256 study
summary before timing, so selector training/evaluation does not contend with
the audit. This does not make the working desktop a controlled laboratory
benchmark; residual desktop load, clocks, temperature and power management can
still affect measurements.

## Binding and checkpoint choice

Before sampling or exporting, validate the sealed original baseline feature
bundle and the main selector receipt, plan, summary, all declared artifacts and
checkpoint checksums. Compare source/split/cache/scaler bindings and the main
implementation hashes. Load only trusted local checkpoint files, with
`torch.load(..., map_location="cpu", weights_only=True)`. Require the saved
metadata and exactly four finite float32 tensors of shapes16x10,16,4x16,4,
totalling244 parameters. Every variant uses seed20261026, the first predeclared
seed, at its fixed2,000th SGD step. No development ranking chooses a seed,
variant, checkpoint or export option.

Save the fixed config, its SHA-256, the main/baseline bindings and current
inference source hashes in `experiment_plan.json` before reading real sample
values or evaluating/exporting graphs. Recheck those artifacts, source files,
config and data/split integrity after all measurements. Record a private
failure trace if an exception prevents completion. Numerical gate failures
produce a completed summary with failure status; do not modify gates, trained
weights or loss to turn such a result into a pass.

## Balanced real-context sample

Use NumPy `default_rng(20261028)`, partitions train then development, users in
sorted private ID order. For each user concatenate complete segment-origin
indexes in sorted session/segment order, draw26 without replacement uniformly
from that user's complete origins, then sort the selected global indexes before
mapping them back to segments. Sampling is balanced per user; it is not a
uniform draw over all study windows. There are390 contexts from15 training
users and130 from5 development users,520 total. The underlying populations
remain3,330,312 training and1,189,027 development windows.

Save private user/session/segment/origin identities and a SHA-256 of the
canonical sample schedule. Every variant and execution provider receives the
same sampled contexts. Public summaries contain partition counts and the
schedule fingerprint, never identities or individual-context/per-user metrics.

Quaternion values enter only through `LockedDevelopmentStage.iter_segments`
for train/development. For selected source origin t, copy exactly
`q[t-399:t+21]`:400 context observations followed by20 target observations.
The inference graph sees only its first400. The final row is the output20
metric target. The existing offline `reference_from_segment` calls the
float64 feature builder, float64-then-float32 scaler arithmetic, frozen
SelectorMLP and float64 geometry/final-float32 blend helper. The helper requires
the suffix because the builder also returns labels; future observations never
enter features, selector weights or emitted output. No final quaternion,
feature, target, prediction or error value is requested. Opaque whole-file
checksum reads are solely integrity checks.

## Numerical comparison and fixed gates

For all six R80 variants export the actual first-seed `FullSelectorInference`
graph with `export_full_selector_onnx`, opset17, no dynamic axes, no external
weights or float64 tensors. Keep the ONNX files private. Run eager PyTorch CPU
and ONNX Runtime CPU on every fixed context. If CUDA is available, also run the
same frozen eager graph on CUDA with resident float32 input; report it as a
separate provider. Do not claim CUDA ONNX Runtime execution when the installed
provider is CPU-only. Record actual library/provider/device versions, TF32 and
autocast state, thread count and peak allocated/reserved CUDA memory.

For each variant/provider/partition, report the maximum and mean geodesic
angular difference over all20 output quaternions, maximum output norm
deviation, finite-output status, actual sampled mean endpoint20 error and its
signed/absolute change from the offline reference on the exact same targets.
These sampled scores describe numerical changes on a balanced subset; they do
not replace the main all-window accuracy table. ORT-versus-eager CPU maximum
trajectory drift is an additional descriptive diagnostic.

The frozen real-data gates apply separately to training and development:

- Maximum sampled trajectory drift from the offline reference <=0.02degrees.
- Absolute change in that partition's sampled mean endpoint20 error
  <=0.01degrees.

The recorder also applies the already fixed0.02degree synthetic trajectory
gate to each actual checkpoint/provider, using four deterministic synthetic
420-observation segments: a stationary trajectory; random walks with component
increment scales0.018 and1.8radians, each seeded20261028 and hemisphere-aligned;
and alternating identity/exact-half-turn quaternions. These are separate from
the independently validated stronger-weight synthetic suite described in
`selector_inference_design.md`; neither test set is real-world accuracy data.

Changing float32 arithmetic to match the reference is outside this audit.
Report any failure and its scope. Passing these numerical gates establishes
only observed local numerical portability under the measured providers. It
does not establish predictive improvement, equivalence of the research
hypotheses, final-holdout performance or universal device portability.

## Local latency and graph cost

Use the first scheduled training context as the fixed resident timing input
for every variant. All provider/model construction, export, reference
calculation, HDF reading and host-to-device input transfer occur outside the
timed scope. Warm up50 forwards, then measure500 forwards, batch1, full stateless
400-to20 computation. Save individual timing samples privately and report
median/P95/P99 milliseconds publicly. ONNX Runtime uses only
CPUExecutionProvider, sequential execution, intra/inter-op threads1; eager CPU
uses PyTorch threads1. CPU timings use `perf_counter_ns` wall elapsed and
include framework dispatch/output allocation.

CUDA timing uses an already resident input/model, CUDA events around each full
forward, and completion synchronization for every repetition. Also record a
separate synchronized CPU wall duration. Event time is elapsed time on the GPU
stream around the complete eager operation sequence, not a sum of isolated
kernel durations; it can include gaps while Python schedules work. Host file
I/O, context transfer and output copies are excluded from this timing. The
separate CPU wall timer includes dispatch and waiting. These are local RTX5050
desktop measurements, not ONNX CUDA measurements or organizer-target XR
latency. Do not combine or substitute the provider clocks.

Record actual graph bytes and SHA-256, checker/shape-inference results,
operators, tensor types, node counts, inferred operator-output element counts,
scalar arithmetic and reduction additions/divisions/comparisons using the
export helper's inspector. Keep the algorithm workload dimensions. The244
trainable parameters and224 selector dense MACs are distinct from the full
stateless graph: evidence replay, normalization, quaternion operations,
trigonometry, Softmax and indexing have additional cost. Do not turn nonlinear
operations into a fabricated total MAC count. Static graph inventory is
mathematical work before provider optimization, not measured hardware
instructions or a substitute for runtime. Operators outside official safe
examples remain a target-runtime portability question.

## Artifacts and release

`summary.json` and `receipt.json` contain aggregate results, integrity evidence
and explicit no-final/no-model-selection scope. The run directory also retains
private ONNX graphs, sample identities, per-context differences and timing
samples. Graphs contain learned weights and are excluded from public Git.
Public extraction must contain no raw quaternions, features, target arrays,
weights/scalers, user/session IDs, per-user metrics, usernames, absolute paths
or environment dumps. Source and artifact fingerprints use safe basenames.
The coordinator reviews any extraction into repository docs or the paper.

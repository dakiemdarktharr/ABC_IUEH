# Data and evaluation protocol

Protocol date: 2026-10-28. This implementation prepares a conditional pilot for
**Forecast Horizon and Feedback Age in Zero-Shot Head-Orientation Prediction**.
It does not change the predeclared research question or report model efficacy.

## Official requirements

Primary references: [The Data](https://sites.google.com/view/9th-abc-challenge/home/the-data),
[The Challenge](https://sites.google.com/view/9th-abc-challenge/home/the-challenge),
[FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs),
[Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts).

Preserve scalar-first wxyz, float32 static graph shapes [1,400,4] -> [1,20,4],
strict causality and shift 1. Only output 20 mean geodesic degrees is the primary
accuracy statistic; no per-user gradient updates at inference. State, if added,
must derive from the past and reset per session. This pipeline is stateless.

The 20-sample  / 100 Hz  / 100 ms contradiction and minimum actual context remain
unresolved. Use all 400 context samples for the initial neural baseline. Do not
replace 20 outputs with 10 or change the clock to 200 Hz to reconcile the text.

## Source and preprocessing sequence

1. Audit source-owner terms, access gates and native data schema. Only the pinned
   WhoIsAlyx later cohort is enabled. BOXRR is blocked pending its recipient
   approval gate. An alternative pilot does not establish competition eligibility.
2. Discover all source motion pointers. Fetch only selected motion objects under
   explicit byte/file limits; verify DVC MD5, exact size and a local SHA-256.
3. Require every part of a source session. Extract elapsed `delta_time_ms /1000`
   and HMD quaternion **by field name**. Preserve source-native OpenVR frame;
   no unverified handedness conversion is applied.
4. Reject nonfinite, duplicate or backward timestamps; do not sort them. Estimate
   native rate from median original timestamp differences, before dropping bad
   quaternion rows, trimming or interpolation. Keep 85–110 Hz.
5. Drop nonfinite/near-zero quaternion rows, normalize and align hemispheres.
   Trim the first 5 seconds anchored to the original clock. Missing/invalid rows
   expose real gaps rather than being replaced by invented orientations.
6. Apply session motion filtering to the **whole trimmed recording**, before the
  20-minute cap. Five-second nonoverlapping windows, at least 10 valid samples:
   Euler xyz 90th-minus-10th-percentile sweep; shortest wrapped angular differences
   for velocity. Take median per axis across windows, then max across axes for
   each statistic. Sweep and velocity maxima may come from different axes.
7. Keep sweep > 17 degrees and velocity > 15 degrees/s under the tutorial interpretation.
   Record `>=` sensitivity and boundary equality; the official comparison conflict
   is not resolved by this choice. The filter statistics are not model metrics.
8. Cap at 20 minutes after trimming, split gaps>50 ms, and SLERP independently
   inside each retained segment onto a 100 Hz grid. Keep segments >= 1000 samples.
   This is **offline bracketing interpolation**; it is not a claim about a
   zero-delay native sensor preprocessing system.
9. Write chunked HDF5 without materializing overlapping windows.

## Explicit team choices and limitations

The 50 ms gap, 1000-sample floor, 20-minute cap, norm epsilon 1e-8, SciPy active
extrinsic `xyz` convention and tail handling are implementation choices. They
are recorded in configuration and HDF5 metadata, not organizer clarifications.
Sweep is not unwrapped; only velocity differences are wrapped. Gimbal-lock
warnings and dropped rows are counted. Velocity pairs spanning large gaps are
excluded. Full filter windows use 5 seconds as denominator; partial tail windows
use last-valid minus first-valid time. `tail_window_policy=drop` is available for
a documented sensitivity run, not retrospective outcome selection.

Multi-file sessions share one filter/cap. If their clocks reset when concatenated,
reject that session; do not infer a missing time offset. Other complete sessions
from the same user may still qualify. A successful transfer alone does not make
a session or user eligible.

## User split, holdout and windowing

After eligibility is measured, hash-sort user IDs with seed 20261026. Final count
is max(12,round(0.43*N)), development count 5, training is the remainder. Require
N >= 30. For 35 users this gives 15 train / 5 development / 15 final. Keep every session
of a user in the same partition. Lock the manifest before inspecting forecast
errors; it refuses overwrite and includes the exact source-HDF5 SHA-256.

No development/final users in pretraining, scaler fitting or model fitting.
Data-quality filtering is allowed before this split; it must not use model-error
outcomes. The pilot check validates shape, norms, sign continuity, sample rate
and available window counts; it does not score predictions on potential holdout
users. Smoke-only manifests explicitly prohibit training.

A context starts at index s and covers s..s+399. Targets cover s+400..s+419;
endpoint20 is s+419. Stride1 windows stay inside one segment. Exposed NumPy views
are read-only; make a copy before augmentation. CAV estimates use past samples
only. Metrics do not treat overlapping windows as independent participants.

The main study will compare ZV, CAV bandwidths 1/5/20, fixed/equal mixtures,
individual/complete-horizon evidence and learned selectors. The contrast of
h20,a0 against h1,a0 uses the same 244-parameter selector and matched data/compute.
The predeclared practical threshold is >= 3% relative AND 0.05 degrees absolute,
with 95% paired user-cluster CI upper delta < 0 and improvement in >= 60% of users.
These thresholds and the split cannot be changed after viewing final results.
Final holdout forecast evaluation requires the later locked model/configuration;
it is outside the current data-pipeline stage.

## Optional later work

JEPA and EBM remain conditional experiments after the main baselines/ablations
and export. Pretraining must respect the same user split. They do not justify
changing the main question or consuming time reserved for paper/export repairs.

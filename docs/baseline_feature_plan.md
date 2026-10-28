# Frozen train/development baseline and feature stage

Plan recorded before examining baseline errors, 2026-10-28. Configuration is
`configs/baseline_features.json`. This stage implements the existing research
question; it does not change its confirmatory hypothesis or final holdout.

Only the locked train15 and development5 users may have quaternion datasets read
for this stage. Final15 users are excluded from prediction, target extraction,
feature construction, pretraining, scaler fitting and model evaluation. Whole-file
checksum validation does not interpret final samples. Record split/HDF5 hashes
before and after the run and reject partition overlap or a changed data binding.

## Causal signals

For unit wxyz rotations q, fixed experts are ZV and CAV bandwidths W=1,5,20:

```text
v_W(t) = Log_SO3(q[t] * inverse(q[t-W])) / W
P_W(t,k) = Exp_SO3(k*v_W(t)) * q[t]
E_m(t;h,a) = mean_r=0..79 d(P_m(t-a-h-r,h), q[t-a-r])
```

All score targets are <=t. Origins scored for h20 are20 samples older than their
targets. Export h1a0, h20a0, h1a19 (matched issue age) and h20a20. Complete-Horizon
scores average horizons1..20 for origins t-20-r, whose targets have all arrived.
For every segment, usable current origins are399..N-21, so all methods use the
same stride1 contexts and endpoint20 targets at t+20. Scores are computed with
prefix sums but must match explicit past-only replay and suffix-perturbation tests.

Each evidence variant exports10 raw features: four expert errors, three current
speed norms and three pairwise velocity cosines. Speeds use radians/sample;
errors use degrees. Cosines use a fixed1e-12 denominator floor and clipping.
Motion-only exports10 features: speeds3, cosines3, pairwise velocity-difference
norms3 and a100-sample net speed1. Fit feature population mean/std on train windows
only, use a1e-8 std floor, and apply unchanged to development. No per-user scaling.
Per-expert future endpoint20 errors are supervision/evaluation labels, never inputs.

## Baselines and fitting

Report each fixed expert, equal velocity mixture, a static fitted velocity mixture,
hard routing for each evidence variant, Complete-Horizon routing, and rank-weighted
h1/h20 routing (temperature1). Deterministic expert order resolves ties. Report
the lowest development error among the predeclared CV bandwidths as a descriptive,
optimistic reference. This does not select or promote a model/checkpoint for final
evaluation, and is not an unbiased test result. Hindsight best-expert error is an
oracle diagnostic and cannot be submitted as a causal predictor.

Fit the static mixture using a step0.25 simplex grid (35 candidates) on up to65536
uniformly sampled training windows with the fixed seed. Minimize the actual
emitted endpoint20 angle. No development data in fitting its weights.

The neural reference is1197->64ReLU->60,80572 parameters: normalized adjacent
relative-rotation vector components from400 past quaternions,20 relative quaternion
outputs composed with q[t]. Its three-axis input scaling is fitted on train
adjacent increments only. Offline Adam, lr0.001,2000 steps,batch128,two seeds as
configured. Optimize sign-aligned quaternion MSE across20 outputs and report
actual endpoint20 mean geodesic degrees. Fixed steps, no development checkpoint
selection in this stage. This is a reference baseline, not the244-parameter
h1/h20 selector, which will be trained in a later stage.

## Reporting and artifact boundaries

Primary descriptive statistic: window-weighted mean endpoint20 geodesic degrees.
Also report user-macro mean, P95 and2000 user-cluster bootstrap confidence
intervals. Preserve all windows of a sampled user; do not bootstrap overlapping
windows independently. Development has only five users: comparisons and CIs here
are exploratory, not confirmation of the main learned-selector hypothesis.

Keep feature arrays, per-user metrics, labels, sampling IDs, scalers and model
weights local under ignored directories. Publish only code, configuration and
checked aggregate results. No final model errors or feature arrays are permitted
in this stage. The 20 samples /100 Hz /100 ms discrepancy remains unresolved.

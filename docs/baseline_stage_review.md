# Independent baseline-stage implementation review

Review date: 2026-10-28. Scope: the frozen train/development baseline and causal
feature stage. This review examines source code and synthetic checks; it does
not inspect human quaternion arrays, private participant statistics or measured
forecasting outcomes, and makes no claim about predictive efficacy.

The review read the canonical ABC index, groundtruths and ambiguities, the local
context and repository instructions, the data-use/protocol documents, and the
frozen baseline plan/configuration. Official constraints remain separate from
team choices. In particular, the 20-sample / 100 Hz / 100 ms inconsistency has
no recorded organizer resolution, and an Alyx pilot does not establish challenge
eligibility.

## Reviewed contracts

- `evidence.py` implements
  `E_m(t;h,a) = mean(r=0..79) d(P_m(t-a-h-r,h), q[t-a-r])`.
  Every error target is at or before the current origin. `h1_a19` and `h20_a0`
  share forecast issue origins; their mature targets differ. Complete-Horizon
  averages horizons 1..20 at the same origins `t-20-r`.
- All methods share origins `399..N-21`, 400 observed samples ending at `t`,
  and the endpoint20 target `q[t+20]`, within one continuous segment. Expert
  forecasts are normalized and emitted as float32 before angular scoring.
  Future expert errors remain separate labels rather than input features.
- The static velocity mixture searches all 35 step-0.25 simplex candidates on
  up to 65,536 uniformly sampled training origins. Its objective scores the
  actual blended endpoint rotation, rather than blending individual losses.
  Hard/rank routers use only past errors with deterministic expert ordering.
- `RelativeMLP` has 80,572 parameters and the 1197 -> 64 ReLU -> 60 architecture.
  It consumes short-arc adjacent body-frame increments from all 400 observations,
  emits 20 normalized float32 wxyz rotations, and trains with sign-aligned
  quaternion MSE. Analytic gradients include both normalization stages.
- The neural runner fits the three-axis scaler using each adjacent increment
  of training segments once, samples global training origins uniformly with
  replacement, and performs the two frozen 2,000-step Adam fits. Development
  quaternions are loaded after both fits; no development checkpoint selection
  or gradient update occurs. The reported two-seed mean averages errors, and
  is not presented as a prediction ensemble.
- Reported primary means use total angular error divided by total windows.
  The 2,000 bootstrap replicates resample whole users and recompute that ratio;
  compared methods share each user draw. User-macro mean and window P95 are
  separate descriptive statistics. Five development users support exploratory
  comparisons only.
- `LockedDevelopmentStage` exposes train/development quaternion readers only.
  Split/source hashes are checked before and after work. Whole-file hashing
  reads bytes without interpreting final samples; feature validation traverses
  only train/development source segment metadata and cache origins.

## Review findings and resolutions

1. The initial cache check bound only source/split hashes and user membership.
   The completed guard also validates the exact frozen configuration, expert
   order, all variant/dataset shapes and dtypes, full stride1 origins, source
   segment coverage, window counts, train-only scaler metadata, and local HDF5
   storage. A sealed receipt binds cache, scaler, plan and configuration bytes;
   altered sealed artifacts cannot be silently resealed.
2. The initial `development_selected_cav` wording could imply model selection
   forbidden by repository instructions. The descriptive reference is now
   explicitly the lowest observed development error among predeclared CAV
   references. It does not select a checkpoint or final model or feed fitting.
3. Enforcing the complete frozen bundle initially invalidated a legacy synthetic
   cache test that requested `complete_horizon=False`. The test must exercise
   the full frozen plan; it now does so and passes. Abbreviated feature
   construction remains covered at the pure segment-feature level.
4. The initial neural consumer checked cache metadata without checking sealed
   artifact bytes. Both production experiment APIs now enforce the exact frozen
   configuration and sealed bundle before consuming data. The neural consumer
   also requires its cache path to resolve to the run directory's `features.h5`;
   the CLI supplies that same path and validates before dispatch.

## Verification and limits

Synthetic checks compare prefix evidence with explicit replay, perturb future
suffixes, test quaternion-sign/frame behavior, compare emitted expert errors,
check input immutability, finite-difference MLP gradients, Adam learning and
checkpoint continuation, and enforce final-quaternion read sentinels. An
independent SciPy rotation replay using noncommuting random rotations agreed
with all four age/horizon variants, Complete-Horizon over 80 origins, and the
four emitted expert endpoint errors within float32 arithmetic tolerance.

After the guard and runner changes stabilized, an independent targeted run
passed **103 tests in 54.05 seconds**, with no failure:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_evidence.py tests/test_evaluation.py tests/test_neural.py tests/test_stage.py tests/test_neural_experiment.py --basetemp=data/baseline_review_test_tmp_final
```

The integration tests include sentinel final users, refusal of altered sealed
artifacts/configurations, stride1/target alignment, train-only scaler fitting,
and CLI refusal before consumer dispatch on a changed bundle. Runner control
flow uses synthetic test doubles and shortened fits; separate MLP tests exercise
the actual model, gradients, Adam and checkpoint continuation. This reviewer
did not independently execute or inspect the full real-data training/evaluation
run. No unresolved implementation blocker was found within these reviewed
contracts and synthetic checks.

This review does not independently audit real-data outcome files, source
licenses beyond the recorded audit, registration/submission state, foundation
model capability, target-device latency, ONNX export, or the later 244-parameter
learned selector. The fixed 2,000-step MLP run has no hyperparameter tuning or
convergence guarantee. No final-holdout accuracy is permitted in this stage.

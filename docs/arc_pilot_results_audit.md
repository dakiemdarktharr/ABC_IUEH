# Independent arc-pilot results audit

Audit completed: **2026-10-28**. Status: **mechanical recomputation passed**. The audit independently recomputed both completed inner-validation scopes from the private endpoint-error vectors. It did not read source quaternion values, development signals/features, or final-set signals; it did not fit a model or invoke GPU kernels.

## Frozen bindings and access boundary

The completed-run configuration digest matched the pre-run freeze. The raw runner, model, metrics, and supporting source hashes matched the summary bindings, and the manifest matched the frozen configuration and bindings. The source-data and split digests in the run receipts matched the predeclared digests; the coordinator's post-run unchanged flags were true. The reported 20-sample / 100 Hz / 100 ms inconsistency remains unresolved.

The run used 15 outer-training users, assigned once each to three five-user validation folds; each fit used the other ten. I reproduced the fold assignment from the frozen seed and split metadata. Each fold's 65,536-origin static/prior calibration sample matched the deterministic draw from that fold's fitting users. Static weights used the declared 35-point grid, and the exposure prior used the same fitting-only sample. All six saved 5,000-by-128 training schedules reproduced exactly from their fixed fold/seed rules and contained no validation origins.

Both scopes covered the same 3,330,312 validation origins. Across 15 methods, two seeds and 15 users, the audit checked row-order bindings, direct per-user sums/counts, finite errors in [0,180] degrees, deterministic-reference equality across seeds, and the arithmetic mean of paired seed errors. The independently recomputed means, percentiles, fold/seed statistics, bootstrap intervals, contrasts and gates matched the stored aggregate summaries.

The run receipts record 52 train-segment reads across the 15 training users and zero development-segment or final-array reads. Development origin-index integrity metadata was read, but no development signal or feature values were read. The auditor read only private validation errors and related metadata; it read zero source quaternion values, development values and final values. No participant-level values or identifiers are included here.

## Primary result: fixed updates

All five learned arms received 5,000 Adam updates of batch size 128 in each of three folds and two seeds. The primary endpoint is mean geodesic error in degrees at output sample 20. Lower is better.

| Method | Pooled mean (°) | Pooled P95 (°) |
|---|---:|---:|
| NLL median | 2.87759 | 9.73999 |
| Direct scalar angle | 2.88301 | 9.75504 |
| Scalar pinball | 2.90343 | 9.90550 |
| Angular-risk regression | 2.89134 | 9.73756 |
| Direct vector angle | 2.52865 | 8.18547 |
| Fold-fitted static mixture | 3.31377 | 11.71599 |

NLL median improved on the static mixture by 0.43618° (13.16%; paired whole-user 95% CI for NLL-minus-static delta: −0.48190° to −0.38864°). It did not clear the predeclared conjunctive screen against all four mandatory comparators:

| Comparator | NLL-median improvement (°) | Relative | Both seeds better? | Folds improved | Comparator screen |
|---|---:|---:|---|---:|---|
| Static mixture | 0.43618 | 13.16% | Yes | 3/3 | Pass |
| Direct scalar angle | 0.00542 | 0.19% | No | 2/3 | Fail |
| Angular-risk regression | 0.01375 | 0.48% | Yes | 2/3 | Fail |
| Scalar pinball | 0.02584 | 0.89% | Yes | 2/3 | Fail |

The screen requires at least 0.05° and 3% pooled improvement against every comparator, the same direction in both seeds, and improvement in at least two folds. The NLL-median primary screen therefore **failed**. The scalar-angle contrast also changes direction across seeds. Confidence intervals are reported evidence, not an extra threshold.

The predeclared direct-vector control was lower-error than NLL median by 0.34893° (2.52865° vs 2.87759°). This limits a claim that the constrained exposure-distribution head is competitive with the direct vector control.

## Equal-training-time secondary

The timing-only secondary selected each arm's latest 100-step checkpoint at or below the minimum final training time in its fold/seed cell. I independently scanned all 51 checkpoints in each of the six cells, verified monotonic clocks, reproduced the selected steps from time alone, and checked selected model-state hashes. The selected states reused the same full validation origins; example counts differed because checkpoint steps differed.

| Method | Pooled mean (°) |
|---|---:|
| NLL median | 2.87685 |
| Direct scalar angle | 2.97387 |
| Scalar pinball | 2.90459 |
| Angular-risk regression | 2.89134 |
| Direct vector angle | 2.58029 |
| Fold-fitted static mixture | 3.31377 |

NLL median cleared the comparator screen against static mixture and direct scalar angle in this secondary. It did not clear the required effect-size screen against angular-risk regression or scalar pinball, so the conjunctive secondary screen also **failed**. The direct vector control remained lower-error than NLL median.

## Descriptive post-hoc vector comparison

The following vector-versus-static contrast is post hoc and does not replace the failed primary screen. The signed delta is vector minus static; negative values favor vector.

| Scope | Delta (°) | Improvement vs static | Paired whole-user 95% CI for delta | Seed deltas (°) | Fold deltas (°) | Users improved |
|---|---:|---:|---:|---|---|---:|
| Fixed updates | −0.78511 | 23.69% | [−0.88215, −0.68421] | [−0.79533, −0.77490] | [−0.86226, −0.86372, −0.64489] | 15/15 |
| Equal training time | −0.73348 | 22.13% | [−0.82834, −0.63284] | [−0.72796, −0.73899] | [−0.81272, −0.82717, −0.57776] | 15/15 |

For fixed updates, the paired bootstrap 95% interval for relative improvement is 20.42%–26.49%; for equal training time it is 18.90%–24.83%. These intervals condition on the fixed two seeds and three fitted cross-validation models. They do not quantify independent retraining or new-population uncertainty.

## Resources and interpretation

The six fit receipts contain 30 completed arm fits at the fixed 5,000 updates. For this **fixed-update primary**, training-time ratios were 2.343 maximum-to-minimum across arms in total and 2.361 in the worst fold/seed cell. Both exceed the 1.2 rule for a strong same-wall-compute characterization. The primary controls match updates, batch size and sampled examples; their measured synchronized training times do not match.

The separate timing-only secondary did perform an approximate time-matched comparison: its recorded maximum-to-minimum ratios were 1.026656 overall and 1.038280 in the worst fold/seed cell. Checkpoint selection is quantized to fixed 100-step intervals, so this is approximate training-time matching with potentially unequal examples, not exact FLOP equality or a claim that every end-to-end operation used equal compute. Static fitting is not compute-matched to neural fitting.

The three CUDA-session ledgers total 3,914.56 seconds (1.0874 CUDA-context hours), below the five-hour pilot cap. The first two sessions paused at the host-memory guard; the final session resumed from the saved run state and completed without model, optimizer, source, or threshold changes. The memory-trim receipt records the pilot process working set falling from about 1.889 GB to 31 MB. Training clocks include the resulting paging effects, so they are backend and host-resource measurements, not equal FLOPs.

The bounded HDF5 recomputation peaked at 186,216,448 bytes of working set before loading Torch for checkpoint-state verification. The separate CPU-only checkpoint phase brought the process peak to 514,523,136 bytes; no dense error arrays remained resident during that phase and no GPU kernels ran. The independent script and aggregate audit receipt are retained separately from this report; per-user vectors and assignments remain private.

## Paper decision boundary

This pilot does not support a positive superiority claim for the constrained exposure-distribution primary model under its predeclared gate. The equal-time analysis reaches the same overall decision, and the vector control is clearly lower-error within these fixed folds. A paper based on these results should either report the exploratory comparison as a negative/qualified result or defer a positive architecture claim until a new, independently specified evaluation. The post-hoc vector result is a lead for future work, not confirmation.

Results remain conditional exploratory evidence on one Alyx corpus, not challenge eligibility, a foundation-model result, final-holdout evidence, or an acceptance prediction. BOXRR remains disabled pending its separate approval gate, and Alyx-only eligibility remains unresolved. No final-set data, challenge submission, or paper submission was performed by this audit.

# Angular-exposure GPU pilot — actual results, 2026-10-28

User-disjoint inner validation inside the existing 15 outer-training users only: 10 fit / 5 validation per fold, 3 folds, 2 seeds, 30 fits.
The previously used 5 development and 15 final users were not evaluated. These numbers are not directly comparable to the old five-user development means.

Primary: 5,000 Adam updates, batch size 128, same fold-fit sampling schedule and cloned encoder initialization per fold/seed.
Secondary: timing-only selection of saved 100-step prefixes, no additional fits; training examples differ in that comparison.

| Method | Fixed 5,000 mean (degrees) | Approximate time-matched mean (degrees) | Fixed 5,000 P95 (degrees) |
|---|---:|---:|---:|
| Static mixture | 3.313769 | 3.313769 | 11.715987 |
| CAV_W1 | 3.569728 | 3.569728 | 12.086124 |
| CAV_W5 | 4.105021 | 4.105021 | 13.953203 |
| CAV_W20 | 5.601990 | 5.601990 | 19.163842 |
| ZV | 7.057532 | 7.057532 | 26.063285 |
| NLL21 median (primary) | 2.877588 | 2.876846 | 9.739988 |
| NLL21 mean | 2.913884 | 2.919945 | 9.953341 |
| NLL21 MAP | 3.101420 | 3.097223 | 9.882149 |
| Direct scalar-angle | 2.883012 | 2.973867 | 9.755040 |
| Scalar-pinball | 2.903430 | 2.904588 | 9.905504 |
| Angular-risk21 | 2.891339 | 2.891339 | 9.737561 |
| Direct vector-angle | 2.528655 | 2.580291 | 8.185467 |
| prior_median | 3.285228 | 3.285228 | 11.448575 |
| prior_mean | 3.379853 | 3.379853 | 12.117484 |
| prior_map | 3.569728 | 3.569728 | 12.086161 |

## Predeclared NLL-median contrasts

Negative delta means NLL-median has lower error. CIs are paired whole-user bootstrap intervals conditional on two seeds and fixed CV fits.

| Comparison | Fixed 5,000 delta (degrees) [95% CI] | Users improved | Time-matched delta (degrees) [95% CI] |
|---|---:|---:|---:|
| NLL median − Static mixture | -0.436181 [-0.481902,-0.388640] | 15/15 | -0.436924 [-0.480801,-0.391126] |
| NLL median − Direct scalar-angle | -0.005425 [-0.012786,+0.002140] | 9/15 | -0.097021 [-0.111103,-0.084742] |
| NLL median − Angular-risk21 | -0.013751 [-0.025202,-0.001116] | 13/15 | -0.014493 [-0.024132,-0.004382] |
| NLL median − Scalar-pinball | -0.025842 [-0.039111,-0.011644] | 10/15 | -0.027742 [-0.039314,-0.015751] |
| NLL median − Direct vector-angle | +0.348933 [+0.291989,+0.402973] | 0/15 | +0.296554 [+0.237240,+0.351435] |

Fixed-update conjunctive gate: **FAIL**. Time-matched gate: **FAIL**.
The gate requires at least 3% and 0.05° improvement against each of static/scalar-angle/risk21/pinball, both seed directions and at least 2/3 folds. It is not a novelty or publication-acceptance certificate.

## Measured resources

Device: NVIDIA GeForce RTX 5050 Laptop GPU; Torch 2.9.1+cu128.
All 30 fits synchronized training: 1238.49 s. GPU-context session wall including preparation/validation/I/O/resource pauses: 3914.56 s (1.087 h).
Shared worker peak CUDA allocated/reserved: 452.57/486.00 MiB. This includes caches and all resident arms.
Maximum/minimum fixed-update training-time ratio: 2.343; per-cell maximum 2.361.
Approximate time-matched ratios: 1.026656 overall and 1.038280 in the worst cell; checkpoints have 100-step granularity.
Equal updates are not equal time; the separate timed-prefix analysis was fixed before any efficacy result. Static calibration is a separate cheap baseline, not a neural-compute-matched fit.

## Interpretation limits

A closest-label median is optimal for snapped short-arc labels, not arbitrary off-axis rotations. Direct scalar and actual angular-risk controls are mandatory for this reason.
Single conditional Alyx corpus; inner folds are exploratory selection-fold evidence. No foundation breadth, final confirmation, target-XR latency or high-novelty architecture claim is established.
BOXRR remains disabled; Alyx-only challenge eligibility and the 20 samples / 100 Hz / 100 ms specification dispute remain unresolved.
Per-seed/fold estimates, full user-cluster uncertainty, compute selections and source bindings are in the aggregate JSON. Human vectors, participant maps and weights remain private.

Measured code commit: `13c74071156b5b8dfc893f42107ad2cb82ea6921`; summary SHA-256 `b5ebbdc1ffe6251ec7a28b8228e9b346a5022a97322e85d0d3f3f640debcd650`.
See [fixed plan](arc_pilot_plan.md), [code review](arc_pilot_code_review.md), [independent result audit](arc_pilot_results_audit.md), and [aggregate](results/arc_pilot_20261028.json).

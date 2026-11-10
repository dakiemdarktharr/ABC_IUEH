# Data and evaluation protocol

The current study uses the official BOXRR-23 corpus and a user-disjoint split: **600 fit / 101 observed development / 122 reserved final**. Split users before drawing windows. Development users are excluded from fitting and pretraining, but were repeatedly used for model and checkpoint selection. Final users remain sealed through selection and were not used in the reported results.

Inputs are 400 wxyz unit quaternions; targets are the next 20 samples within the same gap-free segment. The primary metric is the mean full SO(3) geodesic angle in degrees at predicted sample 20. The report uses 1,024 sampled windows per development user and weights each user's mean by its eligible window population. A user-balanced mean is reported separately. Overlapping windows are not independent participants.

Native-rate filtering, normalization, hemisphere alignment, trimming, session motion filters and 100 Hz offline SLERP are described in [setup](boxrr_study/setup.md#data-processing-choices). Our strict `>17°` and `>15°/s` thresholds are implementation choices: the organizer also uses `≥` in its summary. The 20-sample / 100 Hz / 100 ms inconsistency is unresolved. Do not alter the output count or describe the implementation's +200 ms indexing as an organizer clarification.

The measured constant-angular-velocity baseline and neural comparators are in [results](boxrr_study/results.md). Before any final evaluation, lock the model, checkpoint, preprocessing, baselines and metric. The private final holdout is not the organizer's unreleased test set.

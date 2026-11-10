# BOXRR head-orientation study: reader's guide

This is a research record for the 9th ABC Challenge. It gives a paper author and reviewer the question, implementation choices, measurements and missing evidence in one place. It is **not a paper**, a final-test report or an official submission package.

## Task in plain language

Given up to 400 previously observed head orientations represented by unit quaternions, predict the next 20. The system receives no future poses when it predicts. The challenge measures the mean full-orientation geodesic error at the **20th output**, in degrees. It also considers parameter count, target-device latency and innovation. See the organizer's [challenge](https://sites.google.com/view/9th-abc-challenge/home/the-challenge), [data guidance](https://sites.google.com/view/9th-abc-challenge/home/the-data) and [FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs).

The organizer describes the stream as 100 Hz, the output as 20 samples, and the horizon as 100 ms. Those statements conflict: in our indexing, 20 successive 10 ms intervals are **200 ms**. We keep the required 20 outputs and report +200 ms as our arithmetic, not as an organizer correction. The latest source check for this record was 10 November 2026; no explicit resolution was found in the pages reviewed.

## Dataset and split

The current campaign uses official BOXRR-23 only. We selected and filtered a subset rather than the entire dataset. The frozen expanded corpus has **600 fitting users, 101 development users and 122 reserved final users**. The development group combines 29 previous inner and 72 previously observed outer users, so it has repeatedly influenced model choices. The final users remained sealed through the recorded searches. User identities, recordings, derived motion data, private manifests and checkpoints are not in this repository.

Sessions were checked for native rate, motion and data quality; quaternions were normalized, sign-aligned and resampled using offline SLERP onto a 100 Hz grid. The first five seconds were trimmed, long gaps split and windows kept inside segments. We used a strict `>17°` sweep and `>15°/s` velocity interpretation; the organizer page also uses ≥ elsewhere. Screening and resource caps favor certain users, so the resulting sample is not representative of all BOXRR or the organizer's unseen test set. Full details and implementation choices: [setup](setup.md).

Dataset citation required by the challenge: Vivek Nair, Wenbo Guo, Rui Wang, James F. O'Brien, Louis Rosenberg and Dawn Song, *Berkeley Open Extended Reality Recording Dataset 2023 (BOXRR-23)*, University of California, Berkeley, 2023, [doi:10.25350/B5NP4V](https://doi.org/10.25350/B5NP4V). Dataset access is subject to the [Berkeley terms](https://rdi.berkeley.edu/metaverse/boxrr-23/), separate from this repository's code.

## Decision at the current evidence stage

**JEPA-TCN** has the lowest observed endpoint error in the expanded development search: **4.842644°**. Keep **Mixer** as the principal neural comparator and backup, and **GRU** as a recurrent comparator. Include **constant angular velocity**, which the organizer requires, and **zero velocity** as a simple reference. TCN is the temporal-convolution comparator. See [model definitions](models.md) and [results](results.md).

The result is a single-seed, development-selected estimate, not evidence of final-test generalization or a confirmed improvement due to JEPA pretraining. The official test comes from outside BOXRR. The final holdout has not been used for these choices. A prior pilot's lower numerical error came from a different cohort and averaging method and is not directly comparable.

## Guide to the repository

- [Model names, architecture and comparison roles](models.md).
- [Environment, exact experiment settings and safe reproduction](setup.md).
- [Measured tables, confidence intervals and claims to avoid](results.md).
- [Organizer requirements and remaining submission evidence](challenge_checklist.md).
- `src/abc_iueh/boxrr_v2_deterministic.py`, `boxrr_v2_latent.py`, `boxrr_v2_extra.py` and `boxrr_v2_experiment.py` are the V2 model and experiment implementations. Internal identifiers are kept unchanged to match the recorded checkpoints.

The older README sections and legacy study files are historical. They do not describe the active dataset or justify reusing their fitted artifacts.

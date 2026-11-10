# Results and interpretation

All results here are development measurements, not official test results. The main statistic is the full SO(3) geodesic angle at predicted sample 20, averaged with eligible-window population weights. In the V2 report, 1,024 starts per user were sampled with replacement from 101 development users: 103,424 draws, 102,904 unique. Checkpoints were selected on 256 draws per user from the same development users. This is not an independent confirmation set.

| Model | Recorded ID | width=32 | width=64 |
|---|---|---:|---:|
| Ordered physics hybrid | `ordered_v2` | 5.105993° | 5.091874° |
| Mixer | `mlp_v2` | 4.868064° | 4.848421° |
| TCN | `tcn_v2` | 4.890514° | 4.883348° |
| Energy hybrid | `energy_v2` | 4.936355° | 4.888846° |
| JEPA-TCN | `jepa_v2` | 4.938041° | **4.842644°** |
| DESP-inspired hybrid | `desp_v2` | 5.012470° | 5.009055° |
| GRU | `gru_v2` | 4.873700° | 4.849759° |
| Hold-last | `zero_velocity` | — | 7.514382° |
| Constant angular velocity, one interval | `cav1` | — | 8.314382° |

For JEPA-TCN with `width=64`, the user-bootstrap 95% interval is **[4.506233°, 5.180676°]**. Its equal-user mean is 4.432382°, a secondary statistic with a different weighting, not the primary score. The endpoint-10 result is 2.422722° at +100 ms in this implementation; it does not replace endpoint 20. No V2 configuration achieved a primary mean below 3°.

JEPA-TCN minus Mixer (both `width=64`) is −0.005777° with user-paired 95% bootstrap interval [−0.012325°, +0.000365°]. This does not establish a stable advantage. Neither that interval nor the model interval accounts for repeated model selection or training-seed variability. V2 is a budgeted search, not a controlled causal comparison: JEPA ran 5,000 pretraining and 60,000 supervised updates, while TCN (`width=64`) ran 20,000 supervised updates. Widths also ran under different hardware and time constraints.

## Later and earlier studies

- An MLP adapter scored 4.847903° versus its control 4.848421° (paired delta −0.000518°, interval [−0.001066°, +0.000057°]); its predefined 0.05° practical gate was not met. Do not promote it as a demonstrated improvement.
- V5 used the same 101 development users and report seed, but a different training sampler and checkpoint selection. Its scratch TCN control scored 4.860721°, direct model 5.289343°, IMM 5.607351°, and CDE 6.093112°. The reported V5 candidate-winner field selects only among the latter three, excluding the better TCN control. Training budgets were time capped and unmatched.
- The earlier BOXRR pilot's TCN 4.671606° and JEPA 4.672673° used 72 outer users and an equal-user mean. They cannot be ranked directly against the expanded sample above. The older TCN rescored on the expanded development sample was 5.023458°.
- Synthetic tests verify software, not accuracy.

## Interpretation boundaries

No results here establish official test accuracy, a score under 2° at sample 20, state-of-the-art status, or a benefit caused by pretraining. More robust constant-velocity filtering, additional training seeds, independent locked evaluation, deployment cost and actual target-device performance remain open. The dataset selection and offline interpolation further limit generalization and native-sensor latency claims.

Source provenance: expanded run protocol SHA-256 `b4d6e33f9a5125ab944408b5569a53818ab368bc66e411f9b5c3df770765beee`; frozen corpus manifest SHA-256 `3e0d7fba1a28bf09a03c5d4f00d773b9583fe309df4417a28380882f5b680293`. These are provenance identifiers, not published participant manifests. Full error arrays, checkpoints and split identities are private.

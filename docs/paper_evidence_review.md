# Independent paper evidence review

Review date: 2026-10-28. Final status: **complete exploratory local draft with
clear limitations; all R1–R6 findings resolved.** No essential evidence or claim
blocker remains. This document contains aggregate
research evidence only. It does not certify submission, eligibility, ethics
approval, or target-XR deployment.

## Final decision and closure

The revised scientific source is complete for the scoped exploratory pilot.
This is not a prediction of publication acceptance, an ethics determination or
a statement of competition compliance. The initial findings below are retained
as review history and superseded by this closure.

| Finding | Final resolution |
|---|---|
| R1: final count | Corrected to 15 final, with 15 train and 5 development; modelling uses 20 people, not all 35 prepared users. |
| R2: interpretation | Abstract names the fixed recipe, all-12 single-expert argmax result and absent convergence evidence. Protocol records replacement-draw exposure; discussion reports the full-train constant-reference gap. No general null about mature-error information is claimed. |
| R3: original wider scope | Unrun wider analytic grid and native-clock sensitivity are disclosed. Public report/audit preserve the wider unexecuted controls and earlier analytic results without claiming the original full study was completed. |
| R4: chronology | Method states the sixth learned h1_a19 control was recorded before selector fitting after earlier analytic/neural-reference outcomes were known. |
| R5: reproduction | Exact CH origin/target averaging, future endpoint labels excluded from inputs, and static actual-angle fitting objective/budget are now explicit. |
| R6: inference status | All manuscript status statements report six passing local FP32 audits and preserve the unmeasured XR-target boundary. Actual graph/runtime/parity tables are included. |

The updated README and [reproduction recipe](reproduction.md) give pinned
optional GPU/export dependencies, all three stage commands and the honest
private-artifact boundary. The reviewer reread the final source/recipe/report
and four public aggregate JSONs. Each public JSON exactly matches its allowed
private aggregate original; a recursive value scan found no absolute filesystem
paths or email strings. No human array, participant-level statistic or model
weight was opened, no extra experiment was run and no Git operation was performed.

Independently recomputed all-variant numerical maxima are 0.0001384226619324089
degrees real trajectory drift, 0.000002916191664148471 degrees absolute partition
mean endpoint-error change, 1.7090764781180212e-7 output norm deviation and
0.00029693403177878593 degrees synthetic actual-checkpoint trajectory drift.
Resource and compact arithmetic tables agree with the receipt; the latter
correctly separates 20 Gemm bias additions and 224 dense selector MACs.
No total hardware MAC count, all-window FP32 accuracy, CUDA-ORT, clean-Linux
or XR-target result is inferred.

The reviewer inspected six rendered pages with current tables/figures and
corrected relative-index caption; no obvious clipping or overflow was visible.
The coordinator subsequently reports the final short prose fixes compiled and
rendered in six pages with no overfull, undefined-reference or missing-glyph
warnings. That last compile is a coordinator receipt. The reported 329-test
local suite was not rerun by this paper reviewer; exact-publication-commit CI
remains a separate release check.

The repaired AI assistance disclosure identifies the workflow, sections and
assistance level, consistent with independently opened
[IEEE Open guidance](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/),
without claiming a human audit or submission already occurred. Research novelty
remains modest and application-specific. No further experiment or final access
is needed for this draft. JEPA/EBM still require later human authorization.

## First-wave judgement (historical)

The evidence supports an exploratory train/development case study of a fixed
244-parameter selector recipe. It does not establish a new ensemble method,
general absence of useful long-horizon information, a converged learner
comparison, final equivalence or competitive challenge accuracy. The
application-specific novelty is modest; the defensible contribution is a
carefully scoped negative finding with causal replay and surrogate/output
diagnostics. The first-wave revisions below were resolved in the final check
above.

The reviewer read the prospective research/stage plans, protocol, data-source
audit, metrics/inference contracts, source ledger, manuscript sources and
aggregate experiment summaries. No human quaternions, private participant
statistics, arrays, model weights or final measurements were opened. No
additional experiment or Git operation was performed. Final PDF layout was not
certified because the available render preceded source/figure changes.

## Verified aggregate evidence

| Development comparison | Mean difference h20−h1, degrees | Conditional 95% user-cluster CI | Users improved |
|---|---:|---:|---:|
| R80 | +0.00939631 | [+0.00522733,+0.01389938] | 0/5 |
| R128 | +0.00645787 | [+0.00348986,+0.00970125] | 0/5 |
| R256 | +0.00357548 | [+0.00168440,+0.00567634] | 0/5 |

R80 actual endpoint means are 4.28487743 degrees for h1_a0,
4.29427374 for h20_a0, 4.23982259 for motion-only and 3.72492040 for the
train-fitted static mixture. Means average the two fixed-seed **error vectors**;
they are not prediction ensembles. Five user clusters and two fitted seeds
limit population inference despite 1,189,027 overlapping development windows.
The final 15 users remain sealed; modelling uses 15 train and 5 development users,
20 people in total, from a prepared 35-user cohort.

All 12 R80 fits choose CAV_W1 by hard argmax on 100% of train/development
windows. Soft weights can still vary. Each fit sees 256,000 sampled training
origins with replacement from 3,330,312 origins. The h20 full-train expected
individual-expert loss is 4.13784 degrees versus 3.56973 for constant CAV_W1.
This and the absence of a convergence check limit interpretation to the fixed
recipe. On development, h20 expected loss 4.66565 is below the static mixture's
diagnostic 4.89434, while actual h20 error 4.29427 exceeds static 3.72492.
Competence-selection loss and emitted-blend loss therefore rank these two
predictors differently; neither loss is a universal bound on the other.

The R80 CUDA receipt records 143.6193 seconds for the full measured stage,
peak allocated 928,942,592 bytes and reserved 954,204,160 bytes, actual CUDA
tensors/operation, deterministic float32 and disabled TF32/autocast. These are
local training resources. Main and sensitivity receipts record zero final
quaternion reads and unchanged source/split bindings.

The separately completed inference aggregate reports PASS for all six first-
predeclared-seed R80 graphs, using 390 train and 130 development contexts,
26 per accessible user. It reports static float32 opset 17 export, CPU ONNX
Runtime and separate eager CPU/CUDA paths. h20_a0 CPU ORT median is 0.4220 ms;
eager CUDA event median is 4.4448 ms and synchronized wall median 4.4674 ms.
These different clock scopes exclude file I/O and input transfer. They establish
only observed local numerical portability/cost; they are not target-XR,
CUDA-ORT, clean-Linux or live-sensor measurements, nor new all-window accuracy.

## First-wave manuscript findings (resolved above)

| ID | Finding | Acceptance condition |
|---|---|---|
| R1 | The reviewed conclusion incorrectly says five final users. | Correct to 15 final; retain 5 development and 20 modelling users. |
| R2 | Abstract omits single-expert argmax collapse and nonconvergence limitation. | Make the fixed-recipe negative finding explicit; retain that soft weights need not be constant. Include sampled exposure/full-train constant-reference gap in text or a linked supplement. |
| R3 | Original broader analytic h×a/native-clock and other controls are not transparently reconciled with current scope. | List completed selectors/R controls, earlier analytic references and unrun analytic-grid/native-90-Hz/calibration/permutation/event/filter controls. Never claim the full original confirmatory plan was completed. |
| R4 | “Prospective” learned h1_a19 extension lacks its chronology. | State it was recorded before selector fitting after earlier analytic/MLP development outcomes were known; do not conflate it with the original analytic confirmatory contrast. |
| R5 | CH indexing, future labels and static fitting objective/budget need precise definitions. | Define CH origins t−20−r and targets origin+k, k=1…20; define future endpoint losses as training labels only. Describe the static 35-point simplex and 65,536-train-window actual-angle fit separately from SGD's expected-loss objective. |
| R6 | Export-pending statements are stale now that an inference aggregate exists. | Update all manuscript/README status statements after engineering review accepts that receipt; preserve no target-XR/clean-Linux/CUDA-ORT claim and do not replace main accuracy with sampled diagnostics. |

Smaller reproducibility edits: label h20_a20 as changing feedback ages at fixed
h; specify world-frame radians/sample for velocities; report separate seed
means/ranges; state bootstrap RNG/percentile construction in the public recipe;
link code/config/aggregate evidence with fixed source versions; name the pinned
WhoIsAlyx source/data license separately from the paper license. Repository
README needs selector/sensitivity/inference reproduction commands. Recompile
and render the final 4–6 IEEE pages after all edits; confirm figures/equations,
citations, legibility and no overflow.

## Citation spot check and novelty boundary

Independently opened primary sources on 2026-10-28:

- [Published IH/CH chapter](https://link.springer.com/chapter/10.1007/978-3-032-15535-1_9): Cerqueira/Torgo, 2026, LNCS 16255, 134–148. Its [2023 preprint methods](https://arxiv.org/html/2306.14563v1) define IH/CH/FHF/LHB. These are inherited concepts, not new weighting principles here.
- [Arbitrage of forecasting experts](https://link.springer.com/article/10.1007/s10994-018-05774-y) and [author paper](https://web.cs.dal.ca/~ltorgo/publication/2019_ctpss19/2019_CTPSS19.pdf): four authors, Machine Learning 108:913–944 (2019), online 4 December 2018, learned expert-loss arbitration. The narrow attribution is accurate.
- [WhoIsAlyx primary paper](https://www.frontiersin.org/journals/virtual-reality/articles/10.3389/frvir.2023.1272234/full): verified author list, 10 November 2023, DOI 10.3389/frvir.2023.1272234. Collection-study approval and its 71-person headline do not establish recipient approval or this pilot's 35-user count.
- [Yoon institutional record](https://pure.korea.ac.kr/en/publications/deep-6-dof-head-motion-prediction-for-latency-in-lightweight-augm/): title/ICCE 2022/DOI confirmed. Its author-order discrepancy with the numbered author PDF remains recorded in the source ledger; exact feedback indexing remains unverified. The manuscript appropriately avoids one-step-only claims about it.

No broad first-method, SOTA, foundation-model, real-time, final-test or completed
submission claim was found. The title is acceptable if zero-shot is explicitly
limited to frozen inference on unseen users. The unresolved 20-sample/100 Hz/100 ms
interface contradiction, data-license/ethics distinction and Alyx-only
eligibility boundary are preserved. The first-wave source/layout follow-ups
were closed as recorded in the final decision above; exact-commit publication
checks remain the coordinator's separate responsibility.

# Data sources and use gates

Audit date: **2026-10-28**. Scope: head-orientation forecasting for academic, noncommercial research. This is a source and engineering audit, not legal advice or a determination of an institution's review obligations. Dataset-owner permission, collection-study approval, recipient approval, technical suitability, and challenge eligibility are separate questions.

## Decision matrix

| Source | Published data terms and access | Technical fit | Current decision |
|---|---|---|---|
| Who Is Alyx? — later cohort | Owner-published CC BY-NC-SA 4.0; research-publication consent stated in the repository. No separate recipient-IRB or individual access-application requirement was found in the reviewed owner README/license. | HMD quaternion and timestamps; later recordings have nominal 90 Hz. Session quality and eligible-user count require measurement. | **Conditional GO for a bounded noncommercial pilot** under the published terms and applicable local duties. Full study remains conditional on quality and split gates. |
| BOXRR-23 | CC BY-NC-SA 4.0 on the owner page **plus a DUA requiring advance independent recipient ethics approval**. | Intended challenge corpus; select sessions in the required native-rate band. | **NO-GO for download/use while required approval evidence is absent.** A public mirror does not close this gate. |
| OpenNEEDS | Primary paper describes access by author permission; an explicit reusable data license was not verified. | Nominal 90 Hz, full unit-quaternion head orientation; suitable in principle. | **NO-GO in the current pipeline** until permission and applicable data terms are documented. |
| Comprehensive VR Dataset, TAM DataHub | Public record and article found; a dataset license was not verified from the record text returned in this audit. | Full Euler rotations, but native HMD recording rate and conversion convention remain unverified; only 14 participants reported. | **NO-GO as a ready substitute.** Do not infer its data license or rate from the article license or device specifications. |

The matrix records decisions for this pipeline. It does not claim that a source is prohibited for every possible use. No raw recordings were downloaded as part of this audit.

## Challenge requirements and unresolved eligibility

The official [Data page](https://sites.google.com/view/9th-abc-challenge/home/the-data) expects a BOXRR download/filter/HDF5 pipeline: retain sessions whose native rate is 85–110 Hz, sanitize to head quaternion/timestamp, normalize, handle within-session gaps with SLERP, and use a uniform 100 Hz grid. Split by user before window creation. It also directs teams to comply with Berkeley's terms and forbids redistribution of raw or derived challenge data outside the team.

The [FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs) allows additional data that a team may lawfully use and requires disclosure in the paper. This does **not** explicitly confirm that an Alyx-only corpus replaces the expected BOXRR pipeline. **Alyx-only competition eligibility is unresolved.** Pilot results must be labelled as Alyx research results, without claiming an eligible challenge submission.

Keep the published **20-output-sample** interface. The specification also calls this 100 ms at 100 Hz, an unresolved inconsistency; do not change output length or sampling rate to reconcile it. The Data page's summary uses `>=` motion thresholds while its tutorial uses `>`; preserve the chosen interpretation in configuration and report boundary sensitivity. No organizer clarification is recorded here.

## Who Is Alyx? — selected pilot source

Use the owner repository pinned to commit [`5da510121f99fc8a6cc51e570770ca93e5745646`](https://github.com/cschell/who-is-alyx/tree/5da510121f99fc8a6cc51e570770ca93e5745646). Read the case-sensitive [`Readme.md`](https://github.com/cschell/who-is-alyx/blob/5da510121f99fc8a6cc51e570770ca93e5745646/Readme.md) and [`License.md`](https://github.com/cschell/who-is-alyx/blob/5da510121f99fc8a6cc51e570770ca93e5745646/License.md).

The README states written consent to publish pseudonymized data for research. Its rate table describes **35 potential 90 Hz users**: 34 with two sessions and one with a single session. The recording-rate defect affects players 1–41 and was fixed from player 42; restrict candidates to **42–76**, then measure every session. Do not upsample early low-rate recordings to make them eligible. Candidate users are not a post-filter survivor count. The README notes interruptions, including split recording files, so files and sessions must not be counted interchangeably.

Use original motion recordings, not the provided 15 fps interpolation. Retain only HMD orientation, time, and private grouping identifiers. The adapter must select `hmd_rot_w`, `hmd_rot_x`, `hmd_rot_y`, and `hmd_rot_z` **by column name**, verify time units and coordinate convention, and test known rotations before generating windows. Do not infer physical CSV column order from a paper's quaternion notation. The source supplies full orientation rather than merely a viewing-direction vector. Preserve session boundaries and avoid connecting across interruptions.

The data owner applies **CC BY-NC-SA 4.0**. Its [legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) permits noncommercial use within the licensed rights; sharing requires attribution, source/license notices, and modification notices. Sharing adapted material invokes ShareAlike. Privacy and other rights are outside the license grant. This project keeps raw recordings and cleaned datasets out of the public repository; that is a project handling choice, not a claim that this license categorically prohibits redistribution. A software license for this repository does not relicense the source data. The treatment of trained weights is not determined by this audit.

The primary paper is Rack, Fernando, Yalcin, Hotho, and Latoschik, **“Who is Alyx? A new behavioral biometric dataset for user identification in XR,”** published **2023-11-10**, [DOI 10.3389/frvir.2023.1272234](https://doi.org/10.3389/frvir.2023.1272234). Its ethics statement reports approval of the collection study and participant consent. The article's CC BY license is distinct from the repository's data license. Collection-study approval does not establish a receiving institution's own determination; that remains outside this source audit. The reviewed publisher/repository text does not impose the BOXRR-specific recipient approval clause on Alyx.

Attribution should cite that paper and link the pinned repository and CC BY-NC-SA 4.0. Preserve the creator notices supplied with the source: the README and license file list different creator-name sets. Do not silently replace either notice. Describe preprocessing changes in the paper/model documentation.

## BOXRR-23 — gated source

The [Berkeley owner page](https://rdi.berkeley.edu/metaverse/boxrr-23/) identifies BOXRR-23, dataset DOI [10.25350/B5NP4V](https://doi.org/10.25350/B5NP4V), and links its [Ethical Dataset Use Agreement](https://rdi.berkeley.edu/metaverse/boxrr-23/dua.pdf). The complete 17-clause DUA was read in this audit.

Clause 9 requires research using the dataset to be independently reviewed and approved **in advance** by the IRB or equivalent board governing the recipient organization. Clause 8 distinguishes Berkeley's own review from recipient responsibilities. Clause 4 restricts further distribution without written consent; clause 7 carries restrictions to agents/subcontractors with access. Clauses 10–11 prohibit identification/contact and sensitive-information inference. Clause 5 requires acknowledgement/citation.

Until the required recipient approval is documented, keep the BOXRR path disabled. Academic or noncommercial purpose, challenge registration, collector approval, and HTTP availability are not evidence that clause 9 has been satisfied. Record an actual approval's scope and date before changing this decision. Do not upload BOXRR raw or derived data to the public repository.

## Bounded alternatives

**OpenNEEDS:** Emery, Zannoli, Warren, Xiao, and Talathi, ETRA 2021, published **2021-05-25**, [DOI 10.1145/3448018.3457996](https://doi.org/10.1145/3448018.3457996). The [primary methods](https://dl.acm.org/doi/fullHtml/10.1145/3448018.3457996) report 44 voluntary participants, 90 Hz head/hand recording, and head unit quaternions in `x,y,z,w` order. Participants consented to research/commercial use. The conclusion nevertheless directs readers to request dataset permission from the authors. Consent is not a recipient license, and “open-source” in an abstract does not supply missing terms. No request was sent; no data license or access grant was verified. This source is technically promising but access-gated.

**Comprehensive VR Dataset:** Kreß, Bremmer, and Lappe, [TAM DataHub record, version 2](https://tam-datahub.online.uni-marburg.de/entities/dataset/3d9dfb1b-c73a-4274-afb0-ef238d4ba93c), **2024-10-18**, dataset [DOI 10.60834/tam-datahub-6.2](https://doi.org/10.60834/tam-datahub-6.2). Its [Data in Brief article](https://doi.org/10.1016/j.dib.2024.111187), available **2024-11-29**, describes timestamped pitch/yaw/roll, 14 participants, consent, and resampled 26.32/52.63 Hz releases. Display refresh of 90 Hz and eye tracking of 120 Hz do not establish native HMD recording rate. The article is CC BY; the dataset record's extracted text did not expose a data license. Euler-to-quaternion conversion also requires a verified rotation convention. These gaps remain unresolved; no data were fetched.

The bounded search did not verify a second immediately usable dataset satisfying **both** full-orientation/native-85–110-Hz requirements and clear data-use terms. This is a search result, not proof that none exists.

## Pilot acceptance and public-release boundary

For the Alyx pilot, record source commit, file checksums, native timestamp rate, valid-quaternion fraction, gaps, interruptions, trim/filter statistics, and rejection reasons. Apply the native-rate gate before resampling. Count qualifying users only after session checks; preserve each user's sessions within one split. A successful download or parser run does not establish study readiness.

Keep raw files, cleaned HDF5, participant/session manifests, cache files, and checkpoints containing source samples out of version control. Public examples should be synthetic. Publish code, configurations without participant details, source citations, and aggregate audit/results only after checking their contents. These are project handling rules; they do not substitute for a dataset license or required approval.

See [the search and reading log](data_audit_search_log.md) for exact sources, scope read, retrieval limits, and negative findings.

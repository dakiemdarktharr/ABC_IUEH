# Data-source audit search and reading log

Audit date: **2026-10-28**. Purpose: select a source for a noncommercial head-orientation pilot while preserving separate access, licensing, ethics, technical, and challenge-eligibility decisions. Four dataset candidates were examined. No raw recording was downloaded, no access request/contact was sent, and no account or public repository was modified by this audit.

## Primary sources read

| Source and stable identifier | Scope read | Result/limit |
|---|---|---|
| [ABC — The Data](https://sites.google.com/view/9th-abc-challenge/home/the-data), live page accessed on audit date | Complete extracted text: filtering, preprocessing, validation, data use/citation. | Governs the challenge-side pipeline; disputed threshold equality remains. |
| [ABC — FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs), live page accessed on audit date | Complete extracted text; relevant data, task, submission, and paper entries checked. | Additional lawful datasets permitted; no explicit Alyx-only replacement approval found. |
| [ABC — Submitting Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts), live page accessed on audit date | Complete extracted text; interface, training-data documentation, offline inference. Embedded interface image was not newly inspected in this audit. | The source audit does not certify an exported artifact. |
| [BOXRR owner page](https://rdi.berkeley.edu/metaverse/boxrr-23/), dataset [DOI 10.25350/B5NP4V](https://doi.org/10.25350/B5NP4V) | Dataset description, owner license metadata, ethics/restriction links. | Owner terms take precedence over assumptions based on public availability. |
| [BOXRR DUA](https://rdi.berkeley.edu/metaverse/boxrr-23/dua.pdf), undated PDF as served on audit date | All 17 clauses read from fresh extracted PDF text and compared with an earlier text extract. PDF layout/signatures were not newly visually inspected. | Explicit recipient approval gate; see `data_sources.md`. No claim that access is approved. |
| [WhoIsAlyx Readme.md](https://github.com/cschell/who-is-alyx/blob/5da510121f99fc8a6cc51e570770ca93e5745646/Readme.md), commit `5da510121f99fc8a6cc51e570770ca93e5745646` | Entire file: rate table, setup, motion description, known issues, citation, consent/license. | Content was also read via its [raw text URL](https://raw.githubusercontent.com/cschell/who-is-alyx/5da510121f99fc8a6cc51e570770ca93e5745646/Readme.md); no recording body accessed. |
| [WhoIsAlyx License.md](https://github.com/cschell/who-is-alyx/blob/5da510121f99fc8a6cc51e570770ca93e5745646/License.md), same commit | Entire file, independently of README. | Direct data-owner license evidence. Creator notices differ between the files; preserve both. |
| [WhoIsAlyx paper](https://www.frontiersin.org/journals/virtual-reality/articles/10.3389/frvir.2023.1272234/full), DOI `10.3389/frvir.2023.1272234`, published 2023-11-10 | Metadata, dataset/setup, ethics, encoding/preprocessing, availability and copyright; full-text extraction available, identification benchmark was not reproduced. | Separates collection approval and article CC BY from data permission. |
| [WhoIsAlyx archive v2.1](https://zenodo.org/records/8379914), DOI `10.5281/zenodo.8379914` | Record metadata/version/license label, not archive contents. | Record labels the license “Other (Attribution)”; do not substitute that generic label for owner `License.md`. Live README is pinned by commit rather than assumed identical to v2.1. |
| [CC BY-NC-SA 4.0 legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) | Full extracted legal code, with focus on §§1–4, scope/other rights, attribution and ShareAlike. | Used to summarize the named license; not an institution-specific legal determination. |
| [OpenNEEDS primary paper](https://dl.acm.org/doi/fullHtml/10.1145/3448018.3457996), DOI `10.1145/3448018.3457996`, 2021-05-25 | Metadata, §2.1 capture/consent, §3 sensors, conclusion/access instructions. | Nominal technical fit verified; dataset license and recipient access permission not verified. No Dropbox recording inspected. |
| [TAM DataHub owner record](https://tam-datahub.online.uni-marburg.de/entities/dataset/3d9dfb1b-c73a-4274-afb0-ef238d4ba93c), DOI `10.60834/tam-datahub-6.2`, version 2, 2024-10-18 | Extracted description, archive listing and version history. | Returned text did not expose a data license. No archive fetched; absence from extracted text does not prove absence from every delivery artifact. |
| [Comprehensive VR primary article](https://doi.org/10.1016/j.dib.2024.111187), 2024-11-29; [full-text mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC11699299/) | Data description/table, collection/device sections, ethics, data availability and article license. | Native HMD clock and exact Euler convention remain unresolved; no session-level inspection. |

## Bounded discovery process

| Search objective/query | Primary follow-up | Outcome |
|---|---|---|
| Owner sources for WhoIsAlyx terms, consent and later-rate cohort | Pinned README/license and official paper above | Selected conditional pilot source. Did not infer survivor counts from repository size. |
| `OpenNEEDs dataset official license 90 Hz quaternion`; then `OpenNEEDS dataset license data download 2020 Emery github` | Original ACM paper and its access statement | Found a permission route, not a documented current access grant or clear data license. Stopped without contacting authors. |
| `VR head orientation quaternion dataset 90 Hz license CC BY dataset` | Followed the primary Comprehensive VR article and owner TAM record as the fourth candidate | Search also returned aggregation/calibration and unrelated gaze results. They were not promoted to audited alternatives. |
| `"tam-datahub-6.2" "license"`; then exact DOI with owner-record objective | TAM version-2 record | Did not verify the missing rate/license requirements. Stopped at four candidates rather than expanding indefinitely. |

Read-only web search/fetch tools supplied discovery and primary text. Exa's initial GitHub HTML fetch timed out; raw **documentation** endpoints succeeded. Some Google Sites fetches timed out, and the first guessed URLs omitted `/home/`; the correct official URLs in the table subsequently returned readable text through the web reader. An initial search call failed schema validation and returned no research evidence; the corrected call succeeded. A failed fetch was never treated as source evidence.

## What was not established

- Recipient approval for BOXRR, or an exception/replacement of its explicit DUA requirement.
- A receiving institution's independent ethics determination for Alyx or any alternative; owner permission and collection consent do not answer that institutional question.
- Challenge eligibility of training only on Alyx, an organizer clarification of the horizon/threshold inconsistencies, or submission/registration status.
- Measured native rates, quaternion layout correctness, consent withdrawals, usable duration, post-filter users, or unseen-user performance for downloaded data. Those require separate ingestion and experiment receipts.
- Permission to relicense source data as repository software, or a legal conclusion about trained-weight licensing.

The actionable decisions and handling rules are in [data_sources.md](data_sources.md). Re-audit source terms before adding another corpus or releasing dataset-derived artifacts.

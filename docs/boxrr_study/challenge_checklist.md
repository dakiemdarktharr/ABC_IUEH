# Challenge requirements and evidence checklist

Checked against organizer pages on **10 November 2026**. This is a working evidence map, not proof of registration, eligibility, submission or completion. Source pages can change; check them again before submitting.

## Official requirements

- **Paper:** 4–6 pages in IEEE format or 8–10 pages in IJABC format; describe method and preprocessing, evaluation including a constant-velocity comparator, strengths and limits, total trainable parameters, MACs per streaming step, and BOXRR-23 citation. At least one team member must present. [Participation](https://sites.google.com/view/9th-abc-challenge/home/participation), [FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs).
- **Model interface:** static graph `context` float32 `[1,400,4]` to `prediction` float32 `[1,20,4]`, wxyz; Python wrapper `[400,4]` to `[20,4]`. Strictly causal, no more than 400 context samples, no inference-time network access or per-user fitting. If state is used, disclose static float32 state tensors and reset at session boundaries. [Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts), [FAQ](https://sites.google.com/view/9th-abc-challenge/home/faqs).
- **Archive:** a single `TEAMNAME_headpose.zip`, at most 500 MB, with `model_fp32.onnx`, `run_inference.py`, pinned `requirements.txt`, `model_card.md` and `README.md`; optional int8 ONNX and weights. The same page and FAQ also say PyTorch/TensorFlow are accepted and ONNX is preferred/deployed; this wording is not fully aligned with the archive's “required” ONNX item, so preserve the discrepancy and resolve it with the organizers. [Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts).
- **Model card:** architecture, trainable parameter count, MACs and working, actual context length, state or explicit statelessness, compression/quantization, data and preprocessing, and known failures. [Artifacts](https://sites.google.com/view/9th-abc-challenge/home/submitting-artifacts).
- **Scoring:** accuracy 40%, trainable parameter count 10%, target XR latency 30%, innovation 20%; MACs are a proxy, not measured target latency. [Challenge](https://sites.google.com/view/9th-abc-challenge/home/the-challenge).
- **Registration and sharing:** organizer participation page says registration was mandatory before 27 October 2026, and prohibits redistribution of raw or derived dataset outside the team. Whether this team registered is not established by the files reviewed. [Participation](https://sites.google.com/view/9th-abc-challenge/home/participation), [Data](https://sites.google.com/view/9th-abc-challenge/home/the-data).

## Available versus still needed

| Evidence | Status |
|---|---|
| BOXRR-only data process, V2 architectures, recipe and development metrics | Documented; exact cohort and weights are private |
| Required constant-velocity comparison | Measured `cav1`; a stronger filtered baseline remains useful |
| Selected candidate | JEPA-TCN on repeatedly used development users; not confirmed final performance |
| Model count | Training-object and inference-path counts documented; verify the delivered artifact |
| MACs per streaming step | **Not yet measured or cross-checked** |
| Target XR latency and operator mapping | **Not established**; do not substitute local CPU timing |
| Selected V2 ONNX, numerical parity, clean Linux/CUDA package validation | **Not established**; Python adapter is not a completed submission |
| Final evaluation | **Not performed**; lock candidate, checkpoint, preprocessing, baseline and metric before any final access |
| Paper | **Not written as part of this preparation** |
| Registration and organization-specific data approvals | **Unknown**; do not infer from access or an unchecked list |

The 20 samples / 100 Hz / 100 ms wording and filter `>` versus `≥` are unresolved source conflicts. Record any organizer clarification with date and source before changing this record.

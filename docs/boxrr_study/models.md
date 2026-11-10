# Models, names and comparison roles

These short names are **reading aliases**, not renamed classes or new claims of novelty. “Base” means the project's configuration `width=64`; it does not mean 64-bit precision, 64 layers, 64 participants, or 64 output samples. The tested `width=32` configuration can be called “Small”. The original code names and checkpoint bindings remain unchanged. A name alone is not enough to reconstruct a model: retain its code version, `width`, `latent_dim`, `use_gravity`, checkpoint and preprocessing.

| Reading name | Original experiment identifier | What it does | Role |
|---|---|---|---|
| **JEPA-Base** | `jepa_v2`, width 64 | Temporal convolutional predictor with training-only joint-embedding objective and EMA teacher | Selected candidate |
| **Mixer-Base** | `mlp_v2`, width 64 | MLP-Mixer-style temporal/channel encoder, physics decoder and linear skip | Main neural comparator / backup |
| **GRU-Base** | `gru_v2`, width 64 | Recurrent encoder over ordered pose patches, with recent-motion bypass | Recurrent comparator / backup |
| **TCN-Base** | `tcn_v2`, width 64 | Causal temporal convolution encoder, physics decoder and horizon head | Convolutional comparator |
| **Hold-last** | `zero_velocity` | Repeat the latest observed orientation | Parameter-free reference |
| **Constant angular velocity** | `cav1` | Extrapolate the last observed rotation increment | Required baseline |

These neural comparators are project-specific hybrids, not unmodified canonical MLP, GRU or TCN baselines and not exact reproductions of external papers. They share features and a physics-based decoder; architecture differences are not isolated pure backbone ablations. `desp_v2` is a project adaptation, not a faithful reproduction of published DESP. A baseline may be ordinary and useful without establishing that it is the strongest possible version.

## Shared input and decoder

Each model consumes `[batch, 400, 4]` wxyz quaternions and returns `[batch, 20, 4]`. Features include 12-channel pose/increment/change tokens, multi-scale motion summaries and an optional gravity-direction feature; the tested models use `use_gravity=True`. These are computed from the observed context. The shared decoder mixes **15** causal kinematic trajectories and adds a learned signed tangent-space residual, converts it back to a quaternion and normalizes it. Its trajectories include hold-last, multiple constant-velocity estimates, damped motion, fitted acceleration and smoothing-based estimates. Inference has no per-user fitting or state carried across calls.

## JEPA-Base

The online encoder uses causal depthwise temporal convolution on recent high-resolution and coarser longer-history features. It produces a latent vector of size **128**; convolution channels are **64**. A physics-mixture decoder and horizon head produce the output trajectory. During training, a predictor learns representations for four future patches against an exponential-moving-average teacher (`ema_decay=0.998`); future target poses are used only for training the teacher/loss, never as inference input. Representation regularizers and motion descriptors are also training-only. The inference forward path does not use the teacher, patch predictor or descriptor head. See `JointEmbeddingV2`, `TemporalV2` and `training_loss` in the source.

## Mixer-Base, GRU-Base and TCN-Base

- **Mixer-Base:** a 64-channel token projection followed by two mixer blocks, mixing across time and channels; a 128-dimensional latent feeds the shared decoder, with a linear skip from observed increments.
- **GRU-Base:** 400 samples become 50 ordered patches of eight. A two-layer, one-direction GRU processes them. Here `width=64` sets its hidden dimension to `max(32, 2×64) = 128`, not 64. Its recent-increment bypass and summary join the recurrent outputs before the 128-dimensional latent and shared decoder.
- **TCN-Base:** 64 convolution channels with causal depthwise blocks, dilations 1, 2, 4, 8 and 16 on short and coarser paths, then a 128-dimensional latent and shared decoder. JEPA-Base builds on this temporal forecasting family; the two runs did not receive matched pretraining and supervised budgets.

## Later V5 candidates

**SO3-Direct** (`direct`) predicts a rotation displacement using relative 6D/increment features, causal convolution, a full-context readout and a learned lag branch. **NeuralCDE** (`cde`) uses 32 recent poses, a learned past-only second-order Savitzky–Golay tangent control and midpoint integration; it is a PyTorch adaptation, not an exact reproduction of a published implementation. **IMM-Residual** (`imm`) uses 40 recent poses and combines constant-velocity/constant-acceleration filtering, evidence-based mixing and a neural residual. Their code is in the `boxrr_v5_*` modules. V5 also trained a scratch TCN control on the same V5 sampler. These candidates did not outperform that control on the reported development sample; their unequal time caps preclude a matched-compute conclusion.

## Parameter accounting

The recorded JEPA training object had **291,156 trainable** parameters and **385,748 total** including the frozen teacher. The recorded inference-path accounting is **255,566**, excluding teacher and training-only modules. These answer different questions: the organizer checks the delivered artifact's trainable parameter count, so verify and document the actual exported graph's count before submission rather than substituting the training-object count. Mixer-Base has 203,211 and GRU-Base 308,235 trainable parameters in the recorded runs; TCN-Base has 255,566. MACs and actual target XR latency remain unverified.

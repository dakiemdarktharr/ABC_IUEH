# Prospective evidence-length sensitivity

Recorded 2026-10-28 before the first real R128/R256 selector fit. This implements
the R128/R256 sensitivity already declared in the topic-discovery
preregistration and the main selector study plan. The immutable R80 main study
remains the primary analysis. R128 and R256 are both reported; neither R nor a
checkpoint is selected from development outcomes.

Each child contains only h1_a0 and h20_a0, ten raw causal features each. The
existing feature helper computes these with complete-horizon construction
disabled. Shared current/endpoint quaternions, world velocities, expert-loss
labels and stride-one origin indices must equal the original R80 cache exactly.
Every train/development user, session and segment and every window is retained.
Source quaternion values can enter only through the locked train/development
reader. Final users remain sealed, including feature construction and metrics.

Population mean and standard deviation are fitted independently for each R and
variant, on training windows only, using float64 arithmetic, ddof zero and
epsilon 1e-8, followed by float32 normalized features. Labels remain separate
future individual-expert endpoint20 angular losses in raw degrees. The model
and loss remain the main study's 244-parameter 10 -> 16 ReLU -> 4 softmax and
expected individual-expert angular-loss surrogate.

Training uses both fixed seeds 20261026/20261027, 2000 SGD steps, batch128,
learning rate0.001, zero momentum and weight decay, the same initialization and
the same sampled global train origins as the completed R80 main run. Actual
schedule and initialization hashes are checked against that main run per seed.
No early stopping or development checkpoint selection is implemented. CUDA
float32 is required, with deterministic algorithms, autocast disabled and TF32
disabled. CPU execution exists only in lower-level synthetic correctness tests.
The completed main receipt, its complete artifact/checkpoint hashes and all
2000 loss values per fit are checked before child fitting begins.

Accuracy is the actual normalized float32 emitted blend's endpoint20 geodesic
error, not the surrogate. Both seeds and their arithmetic mean of error vectors
are reported. Expected-loss and hard-argmax diagnostics are saved separately.
Paired user-cluster bootstrap2000, seed20261026, window-weighted ratio, user-macro
mean and pooled-window P95 remain unchanged. The original train-fitted static
mixture is compared on exactly matching origins. Thresholds remain 3% relative
and0.05 degrees absolute, CI upper delta below zero and60% of users improved;
all development inference is exploratory with five users, and these numerical
checks cannot establish final success or formal final equivalence.

Each R receives a separate cache, scaler file, execution plan and sealed receipt
bound to source HDF5, split, original R80 cache/scalers/receipt, main config,
sensitivity config, this plan and implementation source hashes. The child
validator requires exactly two variants, local nonvirtual datasets, exact
shapes/dtypes/counts/origins, finite values, unchanged shared parent arrays and
train-only scaler moments. It does not apply the six-variant R80 cache validator
to a child. Existing output directories are refused; changed artifacts cannot
be resealed. Failed partial directories remain evidence of an incomplete run.

The run rechecks source/config/parent/plan bindings after feature construction,
after each fit/evaluation child and after both children. It records actual
resource use, full private training losses, private origin schedules, private
per-user/error arrays, checkpoint hashes, actual training exposure and aggregate
metrics. Public summaries contain no participant identities; caches, scalers,
per-user statistics, sampled origins and checkpoints stay in ignored local
storage. This plan records intended execution, not evidence that a run occurred.

Official 400 -> 20 float32 wxyz and sample indices are preserved. The challenge's
20-sample /100 Hz /100 ms contradiction and Alyx-only eligibility remain
unresolved. The paper is an exploratory train/development pilot with final
sealed, remains local for review, and does not imply registration or submission.
JEPA/EBM require the user's later approval after the main paper.

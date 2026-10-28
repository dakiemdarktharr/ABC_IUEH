"""Emitted endpoint20 metrics and user-cluster uncertainty, never window CIs."""
import itertools
import numpy as np

from .geometry import exp_so3, geodesic_deg, multiply_wxyz, normalize_wxyz

EXPERT_NAMES = ("ZV", "CAV_W1", "CAV_W5", "CAV_W20")


def predict_endpoint(current_q, velocities, weights):
    q, v, w = np.asarray(current_q), np.asarray(velocities), np.asarray(weights, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or v.shape != (len(q), 4, 3):
        raise ValueError("expected current[N,4] and expert velocities[N,4,3]")
    q = normalize_wxyz(q)
    if not np.isfinite(v).all():
        raise ValueError("expert velocities must be finite")
    if w.shape == (4,):
        w = np.broadcast_to(w, (len(q), 4))
    if w.shape != (len(q), 4) or not np.isfinite(w).all() or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1):
        raise ValueError("mixture weights must be simplex rows")
    velocity = np.sum(v * w[..., None], axis=1)
    return normalize_wxyz(multiply_wxyz(exp_so3(20 * velocity), q)).astype(np.float32)


def hard_weights(scores):
    s = np.asarray(scores)
    if s.ndim != 2 or s.shape[1] != 4 or not np.isfinite(s).all():
        raise ValueError("routing requires four finite past-error scores")
    return np.eye(4)[np.argmin(s, axis=1)]


def rank_weights(scores, temperature=1.0):
    s = np.asarray(scores)
    if s.ndim != 2 or s.shape[1] != 4 or not np.isfinite(s).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid rank routing inputs")
    order = np.argsort(s, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(4)[None, :], axis=1)
    weights = np.exp(-rank / temperature)
    return weights / weights.sum(axis=1, keepdims=True)


def analytic_errors(data, static_weights, rank_temperature=1.0):
    losses = np.asarray(data["expert_loss"], dtype=np.float64)
    result = {name: losses[:, i] for i, name in enumerate(EXPERT_NAMES)}
    q, target, velocities = data["current_q"], data["target_q"], data["velocities"]
    for name, weights in (("equal_mixture", np.full(4, 0.25)), ("static_mixture", static_weights)):
        result[name] = geodesic_deg(predict_endpoint(q, velocities, weights), target)
    for name, features in data["features"].items():
        if name == "motion_only":
            continue
        scores = features[:, :4]
        result["hard_" + name] = geodesic_deg(predict_endpoint(q, velocities, hard_weights(scores)), target)
        if name in {"h1_a0", "h20_a0"}:
            result["rank_" + name] = geodesic_deg(predict_endpoint(q, velocities, rank_weights(scores, rank_temperature)), target)
    result["oracle_bank_hindsight"] = losses.min(axis=1)
    if any(not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 180 + 1e-6) for x in result.values()):
        raise ValueError("invalid angular errors")
    return result


def simplex_grid(step=0.25):
    divisions = round(1 / step)
    if divisions < 1 or not np.isclose(divisions * step, 1):
        raise ValueError("simplex step must divide one")
    return np.array([row for row in itertools.product(range(divisions + 1), repeat=4)
                     if sum(row) == divisions], dtype=np.float64) / divisions


def fit_static_mixture(current, velocities, target, step=0.25):
    grid = simplex_grid(step)
    errors = np.array([geodesic_deg(predict_endpoint(current, velocities, w), target).mean() for w in grid])
    index = int(np.argmin(errors))
    return grid[index], {"fit_partition": "train", "grid_candidates": len(grid),
                         "calibration_windows": len(current), "calibration_mean_geodesic_deg": float(errors[index]),
                         "weights": grid[index].tolist(), "objective": "actual_emitted_endpoint20_angle"}


def cluster_intervals(user_sums, user_counts, *, resamples=2000, seed=20261026):
    sums, counts = np.asarray(user_sums, dtype=np.float64), np.asarray(user_counts, dtype=np.float64)
    if sums.ndim == 1:
        sums = sums[:, None]
    if (sums.ndim != 2 or not len(sums) or counts.shape != (len(sums),) or np.any(counts <= 0)
            or not np.isfinite(counts).all() or not np.isfinite(sums).all() or resamples < 1):
        raise ValueError("invalid user-cluster sufficient statistics")
    indexes = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[indexes].sum(axis=1) / counts[indexes].sum(axis=1)[:, None]
    return np.percentile(means, [2.5, 97.5], axis=0), means


def summarize_methods(error_group, user_sums, user_counts, *, resamples=2000, seed=20261026):
    methods, users = sorted(user_sums), sorted(user_counts)
    counts = np.array([user_counts[u] for u in users], dtype=np.float64)
    sums = np.array([[user_sums[m][u] for m in methods] for u in users], dtype=np.float64)
    intervals, bootstrap = cluster_intervals(sums, counts, resamples=resamples, seed=seed)
    result = {}
    for i, method in enumerate(methods):
        errors = error_group[method][:]
        result[method] = {"mean_geodesic_deg": float(sums[:, i].sum() / counts.sum()),
                          "user_macro_mean_deg": float(np.mean(sums[:, i] / counts)),
                          "p95_geodesic_deg": float(np.percentile(errors, 95)),
                          "user_cluster_95_CI_mean_deg": intervals[:, i].tolist(),
                          "users": len(users), "windows": int(counts.sum())}
    contrasts = {}
    for comparator in ("hard_h1_a0", "hard_h1_a19", "hard_complete_horizon"):
        if "hard_h20_a0" in methods and comparator in methods:
            left, right = methods.index("hard_h20_a0"), methods.index(comparator)
            delta = sums[:, left] - sums[:, right]
            draws = bootstrap[:, left] - bootstrap[:, right]
            contrasts[f"hard_h20_a0_minus_{comparator}"] = {
                "mean_delta_deg": float(delta.sum() / counts.sum()),
                "user_cluster_95_CI_delta_deg": np.percentile(draws, [2.5, 97.5]).tolist(),
                "users_improved": int(np.count_nonzero(delta < 0)), "users": len(users),
                "interpretation": "exploratory_analytic_router_not_confirmatory_learned_selector"}
    return result, contrasts

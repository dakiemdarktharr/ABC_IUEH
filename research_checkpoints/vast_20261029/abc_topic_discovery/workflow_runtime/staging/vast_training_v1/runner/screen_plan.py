"""Frozen code representation of the draft three-arm fit schedule."""
from __future__ import annotations

from itertools import product


ARMS = ("ordered_product", "active_order_control", "contemporaneous_raw_vector")
FOLDS = (0, 1, 2)
SEEDS = (20261026, 20261027)
BATCH_SIZE = 128
UPDATES_PER_FIT = 5000
DRAW_COUNT_PER_FIT = BATCH_SIZE * UPDATES_PER_FIT
TOTAL_FITS = len(FOLDS) * len(SEEDS) * len(ARMS)
TOTAL_UPDATES = TOTAL_FITS * UPDATES_PER_FIT


def screen_fits() -> tuple[dict[str, int | str], ...]:
    """Return one immutable-order 18-fit schedule; no outcomes are consulted."""
    fits = tuple(
        {"fold_id": fold_id, "training_seed": seed, "arm_id": arm}
        for fold_id, seed, arm in product(FOLDS, SEEDS, ARMS)
    )
    if len(fits) != 18 or len({(x["fold_id"], x["training_seed"], x["arm_id"]) for x in fits}) != 18:
        raise AssertionError("three-arm screen schedule is not exactly 18 unique fits")
    return fits

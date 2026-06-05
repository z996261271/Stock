"""Capacity and lot-size helpers for execution simulation."""

from __future__ import annotations

import numpy as np


def capacity_fill_notional_from_arrays(
    *,
    prices: np.ndarray,
    price_valid: np.ndarray,
    amounts: np.ndarray,
    amount_valid: np.ndarray,
    desired_notional: np.ndarray,
    executable: np.ndarray,
    side: str,
    lot_size: int,
    capacity_pct_of_amount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip desired notional by traded-value capacity while preserving hard blocks."""
    desired = np.asarray(desired_notional, dtype=np.float64)
    if len(desired) == 0:
        empty_float = np.asarray([], dtype=np.float64)
        empty_bool = np.asarray([], dtype=bool)
        return empty_float, empty_bool, empty_bool
    lot_value = np.asarray(prices, dtype=np.float64) * max(int(lot_size), 1)
    cap_value = np.asarray(amounts, dtype=np.float64) * max(float(capacity_pct_of_amount), 0.0)
    lot_ok = desired >= lot_value if side == "buy" else desired > 0
    can_consider = (
        np.asarray(executable, dtype=bool)
        & np.asarray(price_valid, dtype=bool)
        & np.asarray(amount_valid, dtype=bool)
        & np.isfinite(desired)
        & (desired > 0)
        & np.isfinite(cap_value)
        & (cap_value > 0)
        & np.isfinite(lot_value)
        & (lot_value > 0)
        & lot_ok
    )
    filled = np.zeros(len(desired), dtype=np.float64)
    clipped = np.minimum(desired, cap_value)
    if side == "buy":
        lots = np.zeros(len(desired), dtype=np.float64)
        lot_valid = np.isfinite(lot_value) & (lot_value > 0)
        lots[lot_valid] = np.floor(clipped[lot_valid] / lot_value[lot_valid]) * lot_value[lot_valid]
    else:
        lots = clipped
    filled[can_consider] = lots[can_consider]
    filled[can_consider] = np.minimum(filled[can_consider], desired[can_consider])
    filled_mask = filled > 0
    partial_mask = filled_mask & (filled < desired)
    return filled, filled_mask, partial_mask

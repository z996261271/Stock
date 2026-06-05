"""Pure rebalance helpers used by execution simulation."""

from __future__ import annotations

import numpy as np


def normalize_current_weights(current_symbols: np.ndarray, current_weights: np.ndarray | None) -> np.ndarray:
    """Return finite positive current weights matching current_symbols length."""
    if current_weights is None or len(current_weights) != len(current_symbols):
        return (
            np.full(len(current_symbols), 1.0 / len(current_symbols), dtype=np.float64)
            if len(current_symbols)
            else np.asarray([], dtype=np.float64)
        )
    values = np.asarray(current_weights, dtype=np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, 0.0)


def target_rebalance_weight(
    target_symbols: np.ndarray,
    target_weight_override: float | None,
    max_position_weight: float,
) -> float:
    """Return the desired per-position target weight after single-name cap."""
    target_weight = (
        float(target_weight_override)
        if target_weight_override is not None
        else (1.0 / len(target_symbols) if len(target_symbols) else 0.0)
    )
    if max_position_weight > 0:
        target_weight = min(target_weight, float(max_position_weight))
    return float(target_weight)

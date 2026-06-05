"""Concentration and exposure primitives."""

from __future__ import annotations

import numpy as np


UNKNOWN_INDUSTRY = "unknown"


def industry_exposure_map(
    symbols: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Aggregate positive portfolio weights by industry label."""
    if len(symbols) == 0:
        return {}
    if labels is None or len(labels) != len(symbols):
        labels = np.full(len(symbols), UNKNOWN_INDUSTRY, dtype=object)
    exposures: dict[str, float] = {}
    for label, weight in zip(labels, weights, strict=False):
        value = float(weight)
        if not np.isfinite(value) or value <= 0:
            continue
        label_text = str(label or UNKNOWN_INDUSTRY)
        exposures[label_text] = exposures.get(label_text, 0.0) + value
    return exposures


def max_industry_exposure(
    symbols: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray | None = None,
) -> float:
    """Return the largest aggregate industry weight."""
    exposures = industry_exposure_map(symbols, weights, labels)
    return float(max(exposures.values())) if exposures else 0.0


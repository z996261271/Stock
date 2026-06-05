"""Compact day-data access helpers for backtest simulation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DayDataLike(Protocol):
    symbols: np.ndarray
    entry_open: np.ndarray
    close: np.ndarray


def values_for_symbols(day: DayDataLike, symbols: np.ndarray, field: str) -> tuple[np.ndarray, np.ndarray]:
    if len(symbols) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=bool)
    source = getattr(day, field)
    positions = np.searchsorted(day.symbols, symbols)
    valid = positions < len(day.symbols)
    matched = np.zeros(len(symbols), dtype=bool)
    matched[valid] = day.symbols[positions[valid]] == symbols[valid]
    prices = np.full(len(symbols), np.nan, dtype=np.float32)
    prices[matched] = source[positions[matched]]
    matched &= np.isfinite(prices) & (prices > 0)
    return prices, matched


def open_for_symbols(day: DayDataLike, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(symbols) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=bool)
    return values_for_symbols(day, symbols, "entry_open")


def bool_for_symbols(day: DayDataLike, symbols: np.ndarray, field: str) -> np.ndarray:
    if len(symbols) == 0:
        return np.asarray([], dtype=bool)
    source = getattr(day, field)
    positions = np.searchsorted(day.symbols, symbols)
    valid = positions < len(day.symbols)
    matched = np.zeros(len(symbols), dtype=bool)
    matched[valid] = day.symbols[positions[valid]] == symbols[valid]
    output = np.zeros(len(symbols), dtype=bool)
    output[matched] = source[positions[matched]]
    return output


def close_for_symbols(day: DayDataLike, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(symbols) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=bool)
    positions = np.searchsorted(day.symbols, symbols)
    valid = positions < len(day.symbols)
    matched = np.zeros(len(symbols), dtype=bool)
    matched[valid] = day.symbols[positions[valid]] == symbols[valid]
    prices = np.full(len(symbols), np.nan, dtype=np.float32)
    prices[matched] = day.close[positions[matched]]
    matched &= np.isfinite(prices) & (prices > 0)
    return prices, matched


def weighted_symbol_return(ratios: np.ndarray, weights: np.ndarray | None) -> float:
    """Return mean or weighted return from per-symbol price ratios."""
    if len(ratios) == 0:
        return 0.0
    if weights is None or len(weights) != len(ratios):
        return float(np.mean(ratios) - 1.0)
    clean_weights = np.asarray(weights, dtype=np.float64)
    clean_weights = np.where(np.isfinite(clean_weights) & (clean_weights > 0), clean_weights, 0.0)
    invested_weight = float(np.sum(clean_weights))
    if invested_weight <= 0:
        return 0.0
    return float(np.sum(clean_weights * (ratios.astype(np.float64) - 1.0)))


def open_to_open_return(
    day: DayDataLike,
    symbols: np.ndarray,
    prev_open: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    prices, valid = open_for_symbols(day, symbols)
    ratios = np.ones(len(symbols), dtype=np.float32)
    good = valid & np.isfinite(prev_open) & (prev_open > 0)
    ratios[good] = prices[good] / prev_open[good]
    updated = prev_open.copy()
    updated[valid] = prices[valid]
    return weighted_symbol_return(ratios, weights), updated


def return_since_entry(
    day: DayDataLike,
    symbols: np.ndarray,
    entry_open: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    prices, valid = close_for_symbols(day, symbols)
    ratios = np.ones(len(symbols), dtype=np.float32)
    good = valid & np.isfinite(entry_open) & (entry_open > 0)
    ratios[good] = prices[good] / entry_open[good]
    return weighted_symbol_return(ratios, weights)

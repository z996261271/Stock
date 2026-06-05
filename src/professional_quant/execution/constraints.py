"""Execution constraints shared by CLI scripts and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


GROWTH_BOARD_PREFIXES = ("300", "301", "688", "689")


def infer_board_label(symbol: str) -> str:
    """Infer the broad exchange board from an A-share symbol prefix."""
    normalized = str(symbol).strip()
    if normalized.startswith(("688", "689")):
        return "科创板"
    if normalized.startswith(("300", "301")):
        return "创业板"
    return "主板"


def is_growth_board(symbol: str, board: str | None = None) -> bool:
    """Return true for ChiNext/STAR symbols or matching board labels."""
    board_text = str(board or "")
    if "创业" in board_text or "科创" in board_text or "STAR" in board_text.upper() or "CHINEXT" in board_text.upper():
        return True
    return str(symbol).strip().startswith(GROWTH_BOARD_PREFIXES)


def limit_rate_for_symbol(symbol: str, board: str | None = None, is_st: bool = False) -> float:
    """Return the daily price-limit band for the symbol state."""
    if is_st:
        return 0.05
    if is_growth_board(symbol, board):
        return 0.20
    return 0.10


def status_bool(series: pd.Series, default: bool = False) -> pd.Series:
    """Normalize nullable integer status flags to booleans."""
    return pd.to_numeric(series, errors="coerce").fillna(int(default)).astype(int).astype(bool)


def locked_limit_masks(
    *,
    prev_close: np.ndarray,
    entry_open: np.ndarray,
    entry_high: np.ndarray,
    entry_low: np.ndarray,
    limit_rate: np.ndarray,
    entry_suspended: np.ndarray | None = None,
    limit_epsilon: float = 0.002,
    block_limit_trades: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return buyable/sellable masks after suspension and locked-limit checks."""
    prev = np.asarray(prev_close, dtype=float)
    open_ = np.asarray(entry_open, dtype=float)
    high = np.asarray(entry_high, dtype=float)
    low = np.asarray(entry_low, dtype=float)
    rates = np.asarray(limit_rate, dtype=float)
    upper = prev * (1.0 + rates)
    lower = prev * (1.0 - rates)
    valid = np.isfinite(open_) & (open_ > 0)
    if entry_suspended is not None:
        valid &= ~np.asarray(entry_suspended, dtype=bool)
    buyable = valid.copy()
    sellable = valid.copy()
    if block_limit_trades:
        limit_up_locked = valid & np.isfinite(low) & (open_ >= upper * (1.0 - limit_epsilon)) & (
            low >= upper * (1.0 - limit_epsilon)
        )
        limit_down_locked = valid & np.isfinite(high) & (open_ <= lower * (1.0 + limit_epsilon)) & (
            high <= lower * (1.0 + limit_epsilon)
        )
        buyable &= ~limit_up_locked
        sellable &= ~limit_down_locked
    return buyable, sellable


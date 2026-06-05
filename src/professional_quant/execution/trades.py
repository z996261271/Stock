"""Trade event and cost helpers for execution simulation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def trade_status(desired_notional: float, filled_notional: float) -> str:
    """Classify a requested trade from its desired and filled notional."""
    if filled_notional <= 0:
        return "blocked"
    if filled_notional < desired_notional:
        return "partial"
    return "filled"


def trade_block_reason_from_fields(
    *,
    side: str,
    desired_notional: float,
    filled_notional: float,
    price_valid: bool,
    symbol_present: bool,
    suspended: bool,
    executable: bool,
    amount_valid: bool,
    price: float,
    amount: float,
    lot_size: int,
    capacity_pct_of_amount: float,
) -> str:
    """Return the report reason for filled, partial, and blocked trades."""
    if not price_valid:
        return "missing_or_invalid_open"
    if not symbol_present:
        return "symbol_not_in_entry_universe"
    if suspended:
        return "suspended_block"
    if not executable:
        return "limit_up_block" if side == "buy" else "limit_down_block"
    if not amount_valid or not np.isfinite(amount) or amount <= 0:
        return "missing_or_invalid_amount"
    lot_value = float(price) * max(int(lot_size), 1)
    cap_value = float(amount) * max(float(capacity_pct_of_amount), 0.0)
    if side == "buy" and (cap_value < lot_value or desired_notional < lot_value):
        return "lot_size_block"
    if filled_notional < desired_notional:
        return "capacity_partial"
    return "filled"


def trade_event_row(
    *,
    signal_date: Any,
    entry_date: Any,
    symbol: str,
    side: str,
    desired_notional: float,
    filled_notional: float,
    weight_before: float,
    weight_after: float,
    reason: str,
    entry_open: float | None,
    entry_amount: float | None,
) -> dict[str, Any]:
    """Build the stable report row for one attempted trade."""
    return {
        "signal_date": _date_text(signal_date),
        "entry_date": _date_text(entry_date),
        "symbol": symbol,
        "side": side,
        "status": trade_status(desired_notional, filled_notional),
        "reason": reason,
        "desired_notional": float(desired_notional),
        "filled_notional": float(filled_notional),
        "unfilled_notional": float(max(desired_notional - filled_notional, 0.0)),
        "weight_before": float(weight_before),
        "weight_after": float(weight_after),
        "entry_open": entry_open,
        "entry_amount": entry_amount,
    }


def side_cost_fraction_from_amounts(
    *,
    amounts: np.ndarray,
    valid: np.ndarray,
    notionals: np.ndarray,
    equity_cash: float,
    side: str,
    buy_cost: float,
    sell_cost: float,
    slippage_bps: float,
    impact_bps_per_pct_amount: float,
) -> float:
    """Return transaction cost as a fraction of portfolio cash equity."""
    if len(notionals) == 0 or equity_cash <= 0:
        return 0.0
    amounts_array = np.asarray(amounts, dtype=np.float64)
    notionals_array = np.asarray(notionals, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(notionals_array) & (notionals_array > 0)
    if not bool(valid_mask.any()):
        return 0.0
    traded = notionals_array[valid_mask]
    amount_pct = traded / amounts_array[valid_mask]
    impact_bps = float(impact_bps_per_pct_amount) * amount_pct * 100.0
    fee = float(buy_cost) if side == "buy" else float(sell_cost)
    rate = fee + (float(slippage_bps) + impact_bps) / 10_000.0
    return float(np.sum(traded * rate) / equity_cash)


def _date_text(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime | date):
        return value.strftime("%Y-%m-%d")
    return str(value)

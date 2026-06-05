"""Walk-forward result assembly and metrics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_walkforward_result(
    *,
    returns: np.ndarray,
    active: np.ndarray,
    trades: np.ndarray,
    equity_rows: list[dict[str, Any]],
    pick_rows: list[dict[str, Any]],
    trade_event_rows: list[dict[str, Any]],
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build scaled artifacts and formal metrics from one walk-forward replay."""
    equity_df = pd.DataFrame(equity_rows)
    picks_df = pd.DataFrame(pick_rows)
    trades_df = pd.DataFrame(trade_event_rows)
    if equity_df.empty:
        raise RuntimeError("dynamic walk-forward produced no rows")
    equity_df["equity"] = equity_df["equity"] * initial_cash
    first_date = pd.Timestamp(equity_df["entry_date"].iloc[0])
    last_date = pd.Timestamp(equity_df["entry_date"].iloc[-1])
    total_return = float(equity_df["equity"].iloc[-1] / initial_cash - 1.0)
    elapsed_days = max((last_date - first_date).days, 1)
    annual_return = (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0 if total_return > -1 else np.nan
    metrics = {
        "initial_cash": initial_cash,
        "start_date": first_date.strftime("%Y-%m-%d"),
        "end_date": last_date.strftime("%Y-%m-%d"),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": total_return,
        "annual_return": float(annual_return),
        "max_drawdown": float(equity_df["drawdown"].min()),
        "periods": int(len(equity_df)),
        "active_period_rate": float(np.mean(active)),
        "trade_period_rate": float(np.mean(trades)),
        "trade_count": int(np.sum(trades)),
        "executed_trade_count": int(equity_df["trade_count"].sum()),
        "blocked_buy_count": int(equity_df["blocked_buy_count"].sum()),
        "blocked_sell_count": int(equity_df["blocked_sell_count"].sum()),
        "partial_buy_count": int(equity_df["partial_buy_count"].sum()),
        "partial_sell_count": int(equity_df["partial_sell_count"].sum()),
        "industry_blocked_count": int(equity_df["industry_blocked_count"].sum())
        if "industry_blocked_count" in equity_df
        else 0,
        "turnover_blocked_count": int(equity_df["turnover_blocked_count"].sum()),
        "turnover_value": float(equity_df["turnover_value"].sum()),
        "unfilled_buy_value": float(equity_df["unfilled_buy_value"].sum()),
        "unfilled_sell_value": float(equity_df["unfilled_sell_value"].sum()),
        "turnover_blocked_value": float(equity_df["turnover_blocked_value"].sum()),
        "max_period_turnover_pct": float(equity_df["turnover_pct"].max()),
        "avg_invested_weight": float(equity_df["invested_weight"].mean()),
        "avg_cash_weight": float(equity_df["cash_weight"].mean()),
        "max_position_weight_observed": float(equity_df["max_position_weight"].max()),
        "max_industry_weight_observed": float(equity_df["max_industry_weight"].max())
        if "max_industry_weight" in equity_df
        else 0.0,
        "trade_log_rows": int(len(trades_df)),
        "trade_log_status_counts": trades_df["status"].value_counts().to_dict() if "status" in trades_df else {},
        "trade_log_reason_counts": trades_df["reason"].value_counts().to_dict() if "reason" in trades_df else {},
        "portfolio_risk_off_rate": float(equity_df["portfolio_risk_off"].mean())
        if "portfolio_risk_off" in equity_df
        else 0.0,
        "avg_period_return": float(np.mean(returns)),
        "period_return_std": float(np.std(returns)),
        "positive_period_rate": float(np.mean(returns > 0)),
        "pick_count": int(len(picks_df)),
        "date_validation": {
            "signal_before_entry": bool(
                (pd.to_datetime(equity_df["signal_date"]) < pd.to_datetime(equity_df["entry_date"])).all()
            )
        },
    }
    return equity_df, picks_df, trades_df, metrics

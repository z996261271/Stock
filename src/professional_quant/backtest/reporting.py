"""Backtest reporting metrics shared by formal runners and tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def finite_float_or_none(value: Any) -> float | None:
    """Return a JSON-safe finite float, or None for missing/non-finite values."""
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def annualized_return(total_return: float, first_date: pd.Timestamp, last_date: pd.Timestamp) -> float | None:
    if total_return <= -1.0:
        return None
    elapsed_days = max((last_date - first_date).days, 1)
    return finite_float_or_none((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)


def return_curve(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    return np.cumprod(1.0 + values)


def professional_performance_metrics(equity_df: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    """Return professional risk/return metrics for a dynamic report."""
    if equity_df.empty:
        return {}
    entry_dates = pd.to_datetime(equity_df["entry_date"])
    first_date = pd.Timestamp(entry_dates.iloc[0])
    last_date = pd.Timestamp(entry_dates.iloc[-1])
    final_equity = float(equity_df["equity"].iloc[-1])
    total_return = final_equity / initial_cash - 1.0
    annual_return = annualized_return(total_return, first_date, last_date)
    returns = pd.to_numeric(equity_df["period_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    curve = return_curve(returns)
    drawdown = curve / np.maximum.accumulate(curve) - 1.0
    max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0
    annual_volatility = float(np.std(returns, ddof=0) * np.sqrt(252.0)) if len(returns) else 0.0
    downside_returns = returns[returns < 0]
    downside_volatility = (
        float(np.std(downside_returns, ddof=0) * np.sqrt(252.0)) if len(downside_returns) else 0.0
    )
    sharpe = None
    if annual_return is not None and annual_volatility > 0:
        sharpe = annual_return / annual_volatility
    sortino = None
    if annual_return is not None and downside_volatility > 0:
        sortino = annual_return / downside_volatility
    calmar = None
    if annual_return is not None and max_drawdown < 0:
        calmar = annual_return / abs(max_drawdown)
    return {
        "risk_free_rate": 0.0,
        "periods_per_year_assumption": 252,
        "annual_return": annual_return,
        "annual_volatility": finite_float_or_none(annual_volatility),
        "downside_volatility": finite_float_or_none(downside_volatility),
        "sharpe": finite_float_or_none(sharpe),
        "sortino": finite_float_or_none(sortino),
        "calmar": finite_float_or_none(calmar),
        "max_drawdown": finite_float_or_none(max_drawdown),
        "best_period_return": finite_float_or_none(np.max(returns)) if len(returns) else None,
        "worst_period_return": finite_float_or_none(np.min(returns)) if len(returns) else None,
        "positive_period_rate": finite_float_or_none(np.mean(returns > 0)) if len(returns) else None,
        "skew": finite_float_or_none(pd.Series(returns).skew()) if len(returns) >= 3 else None,
        "kurtosis": finite_float_or_none(pd.Series(returns).kurt()) if len(returns) >= 4 else None,
    }


def relative_performance_metrics(equity_df: pd.DataFrame, benchmark: dict[str, Any]) -> dict[str, Any]:
    """Return excess-return and alpha/beta stats against a daily benchmark series."""
    if equity_df.empty:
        return {}
    benchmark_returns = benchmark.get("daily_returns")
    if not benchmark_returns:
        return {}
    strategy = equity_df[["entry_date", "period_return"]].copy()
    strategy["date"] = pd.to_datetime(strategy["entry_date"])
    strategy["strategy_return"] = pd.to_numeric(strategy["period_return"], errors="coerce").fillna(0.0)
    benchmark_frame = pd.DataFrame(benchmark_returns)
    if benchmark_frame.empty or not {"date", "return"}.issubset(benchmark_frame.columns):
        return {}
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame["date"])
    benchmark_frame["benchmark_return"] = pd.to_numeric(benchmark_frame["return"], errors="coerce").fillna(0.0)
    merged = strategy.merge(benchmark_frame[["date", "benchmark_return"]], on="date", how="inner")
    if len(merged) < 2:
        return {}
    strategy_returns = merged["strategy_return"].to_numpy(dtype=np.float64)
    benchmark_values = merged["benchmark_return"].to_numpy(dtype=np.float64)
    excess = strategy_returns - benchmark_values
    benchmark_variance = float(np.var(benchmark_values, ddof=0))
    beta = None
    alpha_period = None
    if benchmark_variance > 0:
        beta = float(np.cov(strategy_returns, benchmark_values, ddof=0)[0, 1] / benchmark_variance)
        alpha_period = float(np.mean(strategy_returns) - beta * np.mean(benchmark_values))
    excess_std = float(np.std(excess, ddof=0))
    return {
        "benchmark_name": benchmark.get("name"),
        "matched_periods": int(len(merged)),
        "total_excess_return": finite_float_or_none(float(np.prod(1.0 + excess) - 1.0)),
        "avg_period_excess_return": finite_float_or_none(float(np.mean(excess))),
        "tracking_error": finite_float_or_none(float(excess_std * np.sqrt(252.0))),
        "information_ratio": finite_float_or_none(
            float(np.mean(excess) * 252.0 / (excess_std * np.sqrt(252.0))) if excess_std > 0 else None
        ),
        "beta": finite_float_or_none(beta),
        "alpha_period": finite_float_or_none(alpha_period),
        "alpha_annualized": finite_float_or_none(alpha_period * 252.0 if alpha_period is not None else None),
        "correlation": finite_float_or_none(float(np.corrcoef(strategy_returns, benchmark_values)[0, 1])),
    }


def period_return_breakdown(equity_df: pd.DataFrame, frequency: str) -> list[dict[str, Any]]:
    """Build annual or monthly return/risk rows from the equity curve."""
    if equity_df.empty:
        return []
    if frequency not in {"Y", "M"}:
        raise ValueError(f"unsupported breakdown frequency: {frequency}")
    frame = equity_df.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["period_key"] = frame["entry_date"].dt.to_period(frequency).astype(str)
    rows: list[dict[str, Any]] = []
    for period_key, group in frame.groupby("period_key", sort=True):
        group = group.sort_values("entry_date")
        returns = pd.to_numeric(group["period_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        curve = return_curve(returns)
        if len(curve) == 0:
            continue
        drawdown = curve / np.maximum.accumulate(curve) - 1.0
        first_date = pd.Timestamp(group["entry_date"].iloc[0])
        last_date = pd.Timestamp(group["entry_date"].iloc[-1])
        total_return = float(curve[-1] - 1.0)
        row = {
            "period": str(period_key),
            "start_date": first_date.strftime("%Y-%m-%d"),
            "end_date": last_date.strftime("%Y-%m-%d"),
            "total_return": finite_float_or_none(total_return),
            "max_drawdown": finite_float_or_none(np.min(drawdown)),
            "periods": int(len(group)),
            "active_period_rate": finite_float_or_none(group["active"].mean()) if "active" in group else None,
            "positive_period_rate": finite_float_or_none(np.mean(returns > 0)),
            "rebalance_periods": int(group["trade"].sum()) if "trade" in group else 0,
            "executed_trade_count": int(group["trade_count"].sum()) if "trade_count" in group else 0,
            "blocked_buy_count": int(group["blocked_buy_count"].sum()) if "blocked_buy_count" in group else 0,
            "blocked_sell_count": int(group["blocked_sell_count"].sum()) if "blocked_sell_count" in group else 0,
            "partial_buy_count": int(group["partial_buy_count"].sum()) if "partial_buy_count" in group else 0,
            "partial_sell_count": int(group["partial_sell_count"].sum()) if "partial_sell_count" in group else 0,
            "turnover_blocked_count": int(group["turnover_blocked_count"].sum())
            if "turnover_blocked_count" in group
            else 0,
            "turnover_value": float(group["turnover_value"].sum()) if "turnover_value" in group else 0.0,
            "unfilled_buy_value": float(group["unfilled_buy_value"].sum()) if "unfilled_buy_value" in group else 0.0,
            "unfilled_sell_value": float(group["unfilled_sell_value"].sum()) if "unfilled_sell_value" in group else 0.0,
            "turnover_blocked_value": float(group["turnover_blocked_value"].sum())
            if "turnover_blocked_value" in group
            else 0.0,
            "max_period_turnover_pct": finite_float_or_none(group["turnover_pct"].max())
            if "turnover_pct" in group
            else None,
            "avg_invested_weight": finite_float_or_none(group["invested_weight"].mean())
            if "invested_weight" in group
            else None,
            "avg_cash_weight": finite_float_or_none(group["cash_weight"].mean()) if "cash_weight" in group else None,
            "max_position_weight_observed": finite_float_or_none(group["max_position_weight"].max())
            if "max_position_weight" in group
            else None,
        }
        if frequency == "Y":
            row["annual_return"] = annualized_return(total_return, first_date, last_date)
        rows.append(row)
    return rows


def compute_equal_weight_benchmark(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    name: str,
) -> dict[str, Any]:
    """Compute a close-to-close equal-weight benchmark for the loaded raw-price universe."""
    required = {"symbol", "trade_date", "raw_close"}
    if df.empty or not required.issubset(df.columns):
        return {}
    temp = df[["symbol", "trade_date", "raw_close"]].copy()
    temp["trade_date"] = pd.to_datetime(temp["trade_date"])
    temp["raw_close"] = pd.to_numeric(temp["raw_close"], errors="coerce")
    temp = temp.dropna(subset=["symbol", "trade_date", "raw_close"])
    temp = temp[(temp["trade_date"] <= end_date)].sort_values(["symbol", "trade_date"])
    if temp.empty:
        return {}
    temp["ret"] = temp.groupby("symbol")["raw_close"].pct_change()
    temp = temp[(temp["trade_date"] >= start_date) & (temp["trade_date"] <= end_date)]
    daily = temp.groupby("trade_date")["ret"].mean().dropna()
    if daily.empty:
        return {}
    curve = return_curve(daily.to_numpy(dtype=np.float64))
    drawdown = curve / np.maximum.accumulate(curve) - 1.0
    total_return = float(curve[-1] - 1.0)
    first_date = pd.Timestamp(daily.index[0])
    last_date = pd.Timestamp(daily.index[-1])
    annual_volatility = float(np.std(daily.to_numpy(dtype=np.float64), ddof=0) * np.sqrt(252.0))
    annual_return = annualized_return(total_return, first_date, last_date)
    return {
        "name": name,
        "method": "daily close-to-close equal-weight average of loaded raw-price symbols",
        "start_date": first_date.strftime("%Y-%m-%d"),
        "end_date": last_date.strftime("%Y-%m-%d"),
        "symbols": int(temp["symbol"].nunique()),
        "periods": int(len(daily)),
        "total_return": finite_float_or_none(total_return),
        "annual_return": annual_return,
        "annual_volatility": finite_float_or_none(annual_volatility),
        "max_drawdown": finite_float_or_none(np.min(drawdown)),
        "positive_period_rate": finite_float_or_none(np.mean(daily.to_numpy(dtype=np.float64) > 0)),
        "daily_returns": [
            {"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "return": finite_float_or_none(value)}
            for date, value in daily.items()
        ],
    }


_finite_float_or_none = finite_float_or_none
_annualized_return = annualized_return
_return_curve = return_curve

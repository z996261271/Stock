"""Formal strategy attribution and capacity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from professional_quant.backtest.reporting import annualized_return, finite_float_or_none, return_curve


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_optional(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_formal_attribution_report(
    *,
    metrics: dict[str, Any],
    equity: pd.DataFrame | None = None,
    picks: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    capacity: pd.DataFrame | None = None,
    benchmark_name: str = "main_board_equal_weight_raw_close",
    min_capacity_annual_return: float = 0.0,
    sensitivity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report explaining formal strategy returns versus the main-board benchmark."""
    equity = equity if equity is not None else pd.DataFrame()
    picks = picks if picks is not None else pd.DataFrame()
    trades = trades if trades is not None else pd.DataFrame()
    capacity = capacity if capacity is not None else pd.DataFrame()

    benchmark = benchmark_gap(metrics, benchmark_name)
    yearly = yearly_attribution(metrics, equity, benchmark_name)
    cash = cash_exposure(equity)
    friction = trade_friction(trades)
    industries = industry_attribution(equity, picks)
    holdings = holding_attribution(picks, trades)
    capacity_diag = capacity_diagnostics(capacity, min_annual_return=min_capacity_annual_return)
    report = {
        "schema_version": "formal_attribution.v1",
        "strategy": metrics.get("config", {}).get("strategy"),
        "is_formal_valid": metrics.get("is_formal_valid"),
        "benchmark_gap": benchmark,
        "yearly_attribution": yearly,
        "industry_attribution": industries,
        "holding_attribution": holdings,
        "cash_exposure": cash,
        "trade_friction": friction,
        "capacity_and_untradable": capacity_diag,
        "sensitivity_matrix": sensitivity_report or {},
        "risk_budget": metrics.get("risk_budget", {}),
        "findings": underperformance_findings(benchmark, yearly, cash, friction, industries, capacity_diag),
        "source_fields": {
            "metrics": [
                "professional_metrics.relative_to_benchmarks",
                "annual_breakdown",
                "benchmarks",
                "risk_budget",
            ],
            "equity": sorted(equity.columns.tolist()) if not equity.empty else [],
            "picks": sorted(picks.columns.tolist()) if not picks.empty else [],
            "trades": sorted(trades.columns.tolist()) if not trades.empty else [],
            "capacity_stress": sorted(capacity.columns.tolist()) if not capacity.empty else [],
        },
    }
    return report


def benchmark_gap(metrics: dict[str, Any], benchmark_name: str) -> dict[str, Any]:
    professional = metrics.get("professional_metrics", {})
    relative = professional.get("relative_to_benchmarks", {}).get(benchmark_name, {})
    benchmark = metrics.get("benchmarks", {}).get(benchmark_name, {})
    return {
        "benchmark_name": benchmark_name,
        "strategy_total_return": _float(metrics.get("total_return")),
        "strategy_annual_return": _float(metrics.get("annual_return")),
        "strategy_max_drawdown": _float(metrics.get("max_drawdown")),
        "benchmark_total_return": _float(benchmark.get("total_return")),
        "benchmark_annual_return": _float(benchmark.get("annual_return")),
        "benchmark_max_drawdown": _float(benchmark.get("max_drawdown")),
        "matched_periods": _int(relative.get("matched_periods")),
        "total_excess_return": _float(relative.get("total_excess_return")),
        "avg_period_excess_return": _float(relative.get("avg_period_excess_return")),
        "tracking_error": _float(relative.get("tracking_error")),
        "information_ratio": _float(relative.get("information_ratio")),
        "beta": _float(relative.get("beta")),
        "alpha_annualized": _float(relative.get("alpha_annualized")),
        "correlation": _float(relative.get("correlation")),
    }


def yearly_attribution(metrics: dict[str, Any], equity: pd.DataFrame, benchmark_name: str) -> list[dict[str, Any]]:
    strategy_rows = {str(row.get("period")): dict(row) for row in metrics.get("annual_breakdown", [])}
    benchmark_rows = _benchmark_yearly_rows(metrics.get("benchmarks", {}).get(benchmark_name, {}))
    equity_by_year = _equity_yearly_friction(equity)
    rows: list[dict[str, Any]] = []
    for year in sorted(set(strategy_rows) | set(benchmark_rows) | set(equity_by_year)):
        strategy = strategy_rows.get(year, {})
        benchmark = benchmark_rows.get(year, {})
        strategy_total = _float(strategy.get("total_return"))
        benchmark_total = _float(benchmark.get("total_return"))
        rows.append(
            {
                "year": year,
                "strategy_total_return": strategy_total,
                "strategy_annual_return": _float(strategy.get("annual_return")),
                "strategy_max_drawdown": _float(strategy.get("max_drawdown")),
                "benchmark_total_return": benchmark_total,
                "benchmark_annual_return": _float(benchmark.get("annual_return")),
                "benchmark_max_drawdown": _float(benchmark.get("max_drawdown")),
                "geometric_excess_return": _geometric_excess(strategy_total, benchmark_total),
                "periods": _int(strategy.get("periods")),
                "rebalance_periods": _int(strategy.get("rebalance_periods")),
                **equity_by_year.get(year, {}),
            }
        )
    return rows


def _benchmark_yearly_rows(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    daily_returns = benchmark.get("daily_returns") or []
    frame = pd.DataFrame(daily_returns)
    if frame.empty or not {"date", "return"}.issubset(frame.columns):
        return {}
    frame["date"] = pd.to_datetime(frame["date"])
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce").fillna(0.0)
    frame["year"] = frame["date"].dt.year.astype(str)
    rows: dict[str, dict[str, Any]] = {}
    for year, group in frame.groupby("year", sort=True):
        returns = group["return"].to_numpy(dtype=np.float64)
        curve = return_curve(returns)
        if not len(curve):
            continue
        drawdown = curve / np.maximum.accumulate(curve) - 1.0
        total_return = float(curve[-1] - 1.0)
        first_date = pd.Timestamp(group["date"].iloc[0])
        last_date = pd.Timestamp(group["date"].iloc[-1])
        rows[str(year)] = {
            "total_return": finite_float_or_none(total_return),
            "annual_return": annualized_return(total_return, first_date, last_date),
            "max_drawdown": finite_float_or_none(np.min(drawdown)),
            "periods": int(len(group)),
        }
    return rows


def _equity_yearly_friction(equity: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if equity.empty or "entry_date" not in equity:
        return {}
    frame = equity.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    frame["year"] = frame["entry_date"].dt.year.astype(str)
    rows: dict[str, dict[str, Any]] = {}
    for year, group in frame.groupby("year", sort=True):
        rows[str(year)] = {
            "avg_cash_weight": _mean(group, "cash_weight"),
            "max_cash_weight": _max(group, "cash_weight"),
            "avg_invested_weight": _mean(group, "invested_weight"),
            "max_position_weight_observed": _max(group, "max_position_weight"),
            "max_industry_weight_observed": _max(group, "max_industry_weight"),
            "blocked_buy_count": _sum_int(group, "blocked_buy_count"),
            "blocked_sell_count": _sum_int(group, "blocked_sell_count"),
            "partial_buy_count": _sum_int(group, "partial_buy_count"),
            "partial_sell_count": _sum_int(group, "partial_sell_count"),
            "unfilled_buy_value": _sum_float(group, "unfilled_buy_value"),
            "unfilled_sell_value": _sum_float(group, "unfilled_sell_value"),
            "turnover_blocked_value": _sum_float(group, "turnover_blocked_value"),
        }
    return rows


def industry_attribution(equity: pd.DataFrame, picks: pd.DataFrame) -> dict[str, Any]:
    exposure_rows: list[dict[str, Any]] = []
    if not picks.empty and {"industry_label", "weight"}.issubset(picks.columns):
        frame = picks.copy()
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
        if "score" in frame:
            frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        grouped = frame.groupby("industry_label", dropna=False)
        for label, group in grouped:
            exposure_rows.append(
                {
                    "industry_label": str(label),
                    "pick_rows": int(len(group)),
                    "unique_symbols": int(group["symbol"].nunique()) if "symbol" in group else None,
                    "signal_dates": int(group["signal_date"].nunique()) if "signal_date" in group else None,
                    "avg_pick_weight": _float(group["weight"].mean()),
                    "max_pick_weight": _float(group["weight"].max()),
                    "avg_score": _float(group["score"].mean()) if "score" in group else None,
                }
            )
        exposure_rows.sort(key=lambda row: (row["max_pick_weight"] or 0.0, row["pick_rows"]), reverse=True)

    top_industry_rows: list[dict[str, Any]] = []
    if not equity.empty and {"top_industry", "period_return"}.issubset(equity.columns):
        frame = equity.copy()
        frame["period_return"] = pd.to_numeric(frame["period_return"], errors="coerce").fillna(0.0)
        grouped = frame.groupby("top_industry", dropna=False)
        for label, group in grouped:
            returns = group["period_return"].to_numpy(dtype=np.float64)
            total = float(np.prod(1.0 + returns) - 1.0) if len(returns) else None
            top_industry_rows.append(
                {
                    "top_industry": str(label),
                    "periods": int(len(group)),
                    "total_return_when_top": _float(total),
                    "avg_period_return": _float(group["period_return"].mean()),
                    "avg_cash_weight": _mean(group, "cash_weight"),
                    "max_industry_weight": _max(group, "max_industry_weight"),
                }
            )
        top_industry_rows.sort(key=lambda row: row["periods"], reverse=True)

    return {
        "exposure_by_pick_industry": exposure_rows[:20],
        "top_industry_period_returns": top_industry_rows[:20],
        "note": (
            "Pick-industry rows measure selected exposure. Top-industry rows group period returns by the largest "
            "industry exposure in the portfolio; they are explanatory, not exact Brinson attribution."
        ),
    }


def holding_attribution(picks: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not picks.empty and "symbol" in picks.columns:
        frame = picks.copy()
        frame["weight"] = pd.to_numeric(frame.get("weight"), errors="coerce").fillna(0.0)
        if "score" in frame:
            frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        grouped = frame.groupby("symbol", dropna=False)
        for symbol, group in grouped:
            rows.append(
                {
                    "symbol": str(symbol),
                    "industry_label": _first_text(group, "industry_label"),
                    "pick_rows": int(len(group)),
                    "signal_dates": int(group["signal_date"].nunique()) if "signal_date" in group else None,
                    "first_signal_date": _min_text(group, "signal_date"),
                    "last_signal_date": _max_text(group, "signal_date"),
                    "avg_weight": _float(group["weight"].mean()),
                    "max_weight": _float(group["weight"].max()),
                    "avg_score": _float(group["score"].mean()) if "score" in group else None,
                }
            )
    trade_rows = _trade_by_symbol(trades)
    by_symbol = {row["symbol"]: row for row in rows}
    for symbol, trade_row in trade_rows.items():
        if symbol in by_symbol:
            by_symbol[symbol].update(trade_row)
        else:
            by_symbol[symbol] = {"symbol": symbol, **trade_row}
    combined = list(by_symbol.values())
    combined.sort(
        key=lambda row: (
            float(row.get("unfilled_notional") or 0.0),
            int(row.get("pick_rows") or 0),
            float(row.get("max_weight") or 0.0),
        ),
        reverse=True,
    )
    return {
        "top_holdings_and_friction": combined[:30],
        "note": "Rows are ranked by unfilled notional first, then selection frequency and max selected weight.",
    }


def _trade_by_symbol(trades: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades.empty or "symbol" not in trades.columns:
        return {}
    frame = trades.copy()
    for column in ("desired_notional", "filled_notional", "unfilled_notional"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    rows: dict[str, dict[str, Any]] = {}
    for symbol, group in frame.groupby("symbol", dropna=False):
        rows[str(symbol)] = {
            "trade_orders": int(len(group)),
            "status_counts": _value_counts(group, "status"),
            "reason_counts": _value_counts(group, "reason"),
            "desired_notional": _sum_float(group, "desired_notional"),
            "filled_notional": _sum_float(group, "filled_notional"),
            "unfilled_notional": _sum_float(group, "unfilled_notional"),
        }
    return rows


def cash_exposure(equity: pd.DataFrame) -> dict[str, Any]:
    if equity.empty:
        return {"status": "missing_equity"}
    rows: dict[str, Any] = {
        "status": "available",
        "periods": int(len(equity)),
        "avg_cash_weight": _mean(equity, "cash_weight"),
        "median_cash_weight": _median(equity, "cash_weight"),
        "max_cash_weight": _max(equity, "cash_weight"),
        "periods_cash_over_20pct": _count_over(equity, "cash_weight", 0.20),
        "periods_cash_over_50pct": _count_over(equity, "cash_weight", 0.50),
        "avg_invested_weight": _mean(equity, "invested_weight"),
        "median_invested_weight": _median(equity, "invested_weight"),
        "max_position_weight_observed": _max(equity, "max_position_weight"),
        "max_industry_weight_observed": _max(equity, "max_industry_weight"),
        "avg_positions": _mean(equity, "positions"),
        "portfolio_risk_off_rate": _mean_bool(equity, "portfolio_risk_off"),
    }
    rows["cash_over_20pct_rate"] = _ratio(rows["periods_cash_over_20pct"], rows["periods"])
    rows["cash_over_50pct_rate"] = _ratio(rows["periods_cash_over_50pct"], rows["periods"])
    return rows


def trade_friction(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"status": "missing_trades"}
    frame = trades.copy()
    for column in ("desired_notional", "filled_notional", "unfilled_notional"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    total_orders = int(len(frame))
    status = frame.get("status", pd.Series(dtype=object)).astype(str)
    blocked_orders = int((status == "blocked").sum())
    partial_orders = int((status == "partial").sum())
    filled_orders = int((status == "filled").sum())
    desired = _sum_float(frame, "desired_notional") or 0.0
    unfilled = _sum_float(frame, "unfilled_notional") or 0.0
    return {
        "status": "available",
        "orders": total_orders,
        "status_counts": _value_counts(frame, "status"),
        "reason_counts": _value_counts(frame, "reason"),
        "blocked_orders": blocked_orders,
        "partial_orders": partial_orders,
        "filled_orders": filled_orders,
        "blocked_order_ratio": _ratio(blocked_orders, total_orders),
        "partial_order_ratio": _ratio(partial_orders, total_orders),
        "untradable_signal_ratio": _ratio(blocked_orders + partial_orders, total_orders),
        "desired_notional": _float(desired),
        "filled_notional": _sum_float(frame, "filled_notional"),
        "unfilled_notional": _float(unfilled),
        "unfilled_notional_ratio": _ratio(unfilled, desired),
        "by_reason": _trade_friction_group(frame, "reason"),
        "by_side": _trade_friction_group(frame, "side"),
        "top_unfilled_symbols": _top_unfilled_symbols(frame),
    }


def _trade_friction_group(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame:
        return []
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False):
        status = group.get("status", pd.Series(dtype=object)).astype(str)
        desired = _sum_float(group, "desired_notional") or 0.0
        unfilled = _sum_float(group, "unfilled_notional") or 0.0
        rows.append(
            {
                column: str(value),
                "orders": int(len(group)),
                "blocked_orders": int((status == "blocked").sum()),
                "partial_orders": int((status == "partial").sum()),
                "desired_notional": _float(desired),
                "filled_notional": _sum_float(group, "filled_notional"),
                "unfilled_notional": _float(unfilled),
                "unfilled_notional_ratio": _ratio(unfilled, desired),
            }
        )
    rows.sort(key=lambda row: (row["unfilled_notional"] or 0.0, row["orders"]), reverse=True)
    return rows


def _top_unfilled_symbols(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "symbol" not in frame or "unfilled_notional" not in frame:
        return []
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", dropna=False):
        unfilled = _sum_float(group, "unfilled_notional") or 0.0
        if unfilled <= 0:
            continue
        rows.append(
            {
                "symbol": str(symbol),
                "orders": int(len(group)),
                "unfilled_notional": _float(unfilled),
                "reason_counts": _value_counts(group, "reason"),
            }
        )
    rows.sort(key=lambda row: row["unfilled_notional"] or 0.0, reverse=True)
    return rows[:20]


def capacity_diagnostics(capacity: pd.DataFrame, min_annual_return: float = 0.0) -> dict[str, Any]:
    if capacity.empty:
        return {"status": "missing_capacity_stress"}
    required = {"initial_cash", "annual_return", "max_drawdown"}
    if not required.issubset(capacity.columns):
        return {"status": "missing_required_columns", "columns": sorted(capacity.columns)}
    frame = capacity.copy()
    numeric_columns = [
        "initial_cash",
        "annual_return",
        "max_drawdown",
        "avg_cash_weight",
        "unfilled_buy_value",
        "unfilled_sell_value",
        "turnover_blocked_value",
        "blocked_buy_count",
        "blocked_sell_count",
        "partial_buy_count",
        "partial_sell_count",
    ]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["unfilled_total_value"] = (
        frame.get("unfilled_buy_value", 0.0)
        + frame.get("unfilled_sell_value", 0.0)
        + frame.get("turnover_blocked_value", 0.0)
    )
    current = _current_capacity_row(frame)
    rows: list[dict[str, Any]] = []
    for initial_cash, group in frame.groupby("initial_cash", dropna=False):
        annual = pd.to_numeric(group["annual_return"], errors="coerce").dropna()
        drawdown = pd.to_numeric(group["max_drawdown"], errors="coerce").dropna()
        annual_min = _float(annual.min()) if not annual.empty else None
        rows.append(
            {
                "initial_cash": _float(initial_cash),
                "rows": int(len(group)),
                "annual_return_min": annual_min,
                "annual_return_median": _float(annual.median()) if not annual.empty else None,
                "annual_return_max": _float(annual.max()) if not annual.empty else None,
                "max_drawdown_worst": _float(drawdown.min()) if not drawdown.empty else None,
                "avg_cash_weight_max": _max(group, "avg_cash_weight"),
                "unfilled_total_value_max": _max(group, "unfilled_total_value"),
                "blocked_buy_count_max": _max(group, "blocked_buy_count"),
                "blocked_sell_count_max": _max(group, "blocked_sell_count"),
                "partial_buy_count_max": _max(group, "partial_buy_count"),
                "partial_sell_count_max": _max(group, "partial_sell_count"),
                "passes_non_negative_worst_case": bool(annual_min is not None and annual_min >= 0.0),
                "passes_min_annual_return_worst_case": bool(
                    annual_min is not None and annual_min >= min_annual_return
                ),
            }
        )
    rows.sort(key=lambda row: row["initial_cash"] or 0.0)
    non_negative = [row["initial_cash"] for row in rows if row["passes_non_negative_worst_case"]]
    threshold = [row["initial_cash"] for row in rows if row["passes_min_annual_return_worst_case"]]
    return {
        "status": "available",
        "min_annual_return_threshold": _float(min_annual_return),
        "current_setting": current,
        "capital_by_initial_cash": rows,
        "capital_limit_non_negative_worst_case": _float(max(non_negative)) if non_negative else None,
        "capital_limit_min_annual_return_worst_case": _float(max(threshold)) if threshold else None,
        "stress_rows": int(len(frame)),
        "risk_note": (
            "Capital limits are inferred only from tested initial_cash grid points. A blank limit means every tested "
            "cash level failed the stated worst-case threshold."
        ),
    }


def _current_capacity_row(frame: pd.DataFrame) -> dict[str, Any]:
    if "is_current_setting" in frame:
        mask = frame["is_current_setting"].astype(str).str.lower().isin({"1", "true", "yes"})
        selected = frame[mask]
        row = selected.iloc[0] if not selected.empty else frame.iloc[0]
    else:
        row = frame.iloc[0]
    return {
        "initial_cash": _float(row.get("initial_cash")),
        "capacity_pct_of_amount": _float(row.get("capacity_pct_of_amount")),
        "slippage_bps": _float(row.get("slippage_bps")),
        "annual_return": _float(row.get("annual_return")),
        "max_drawdown": _float(row.get("max_drawdown")),
        "avg_cash_weight": _float(row.get("avg_cash_weight")),
        "unfilled_total_value": _float(row.get("unfilled_total_value")),
    }


def underperformance_findings(
    benchmark: dict[str, Any],
    yearly: list[dict[str, Any]],
    cash: dict[str, Any],
    friction: dict[str, Any],
    industries: dict[str, Any],
    capacity: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    excess = benchmark.get("total_excess_return")
    if excess is not None and excess < 0:
        findings.append(
            {
                "type": "benchmark_underperformance",
                "severity": "high",
                "evidence": {
                    "benchmark": benchmark.get("benchmark_name"),
                    "total_excess_return": excess,
                    "strategy_total_return": benchmark.get("strategy_total_return"),
                    "benchmark_total_return": benchmark.get("benchmark_total_return"),
                },
            }
        )
    weak_years = [row for row in yearly if (row.get("geometric_excess_return") or 0.0) < 0.0]
    if weak_years:
        findings.append(
            {
                "type": "weak_relative_years",
                "severity": "medium",
                "evidence": [
                    {
                        "year": row.get("year"),
                        "geometric_excess_return": row.get("geometric_excess_return"),
                        "avg_cash_weight": row.get("avg_cash_weight"),
                    }
                    for row in weak_years
                ],
            }
        )
    if (cash.get("avg_cash_weight") or 0.0) >= 0.20:
        findings.append(
            {
                "type": "cash_drag",
                "severity": "medium",
                "evidence": {
                    "avg_cash_weight": cash.get("avg_cash_weight"),
                    "cash_over_20pct_rate": cash.get("cash_over_20pct_rate"),
                },
            }
        )
    if (friction.get("untradable_signal_ratio") or 0.0) >= 0.20:
        findings.append(
            {
                "type": "execution_friction",
                "severity": "high",
                "evidence": {
                    "untradable_signal_ratio": friction.get("untradable_signal_ratio"),
                    "unfilled_notional_ratio": friction.get("unfilled_notional_ratio"),
                    "reason_counts": friction.get("reason_counts"),
                },
            }
        )
    if (cash.get("max_industry_weight_observed") or 0.0) >= 0.35:
        findings.append(
            {
                "type": "industry_concentration",
                "severity": "medium",
                "evidence": {
                    "max_industry_weight_observed": cash.get("max_industry_weight_observed"),
                    "top_pick_industries": industries.get("exposure_by_pick_industry", [])[:5],
                },
            }
        )
    if capacity.get("status") == "available" and capacity.get("capital_limit_non_negative_worst_case") is None:
        findings.append(
            {
                "type": "capacity_limit_not_supported",
                "severity": "high",
                "evidence": {"min_annual_return_threshold": 0.0},
            }
        )
    return findings


def _float(value: Any) -> float | None:
    try:
        return finite_float_or_none(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_value = _float(numerator)
    denominator_value = _float(denominator)
    if numerator_value is None or denominator_value in (None, 0.0):
        return None
    return _float(numerator_value / denominator_value)


def _geometric_excess(strategy_total: float | None, benchmark_total: float | None) -> float | None:
    if strategy_total is None or benchmark_total is None or benchmark_total <= -1.0:
        return None
    return _float((1.0 + strategy_total) / (1.0 + benchmark_total) - 1.0)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    values = _series(frame, column)
    return _float(values.mean()) if not values.empty else None


def _median(frame: pd.DataFrame, column: str) -> float | None:
    values = _series(frame, column)
    return _float(values.median()) if not values.empty else None


def _max(frame: pd.DataFrame, column: str) -> float | None:
    values = _series(frame, column)
    return _float(values.max()) if not values.empty else None


def _sum_float(frame: pd.DataFrame, column: str) -> float | None:
    values = _series(frame, column)
    return _float(values.sum()) if not values.empty else None


def _sum_int(frame: pd.DataFrame, column: str) -> int:
    values = _series(frame, column)
    return int(values.sum()) if not values.empty else 0


def _count_over(frame: pd.DataFrame, column: str, threshold: float) -> int:
    values = _series(frame, column)
    return int((values > threshold).sum()) if not values.empty else 0


def _mean_bool(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    normalized = frame[column].astype(str).str.lower().isin({"1", "true", "yes"})
    return _float(normalized.mean()) if len(normalized) else None


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype(str).value_counts(dropna=False).items()}


def _first_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = frame[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else None


def _min_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = frame[column].dropna().astype(str)
    return values.min() if not values.empty else None


def _max_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = frame[column].dropna().astype(str)
    return values.max() if not values.empty else None

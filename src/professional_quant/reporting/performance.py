"""Pyfolio-style performance tear sheet from formal backtest artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from professional_quant.backtest.reporting import (
    finite_float_or_none,
    period_return_breakdown,
    professional_performance_metrics,
    relative_performance_metrics,
    return_curve,
)


def build_performance_report(
    *,
    metrics: dict[str, Any] | None = None,
    equity: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    benchmark_name: str = "main_board_equal_weight_raw_close",
    initial_cash: float | None = None,
) -> dict[str, Any]:
    """Build a stable JSON report for return, risk, drawdown, and trade execution diagnostics."""
    metrics = metrics or {}
    equity = equity if equity is not None else pd.DataFrame()
    trades = trades if trades is not None else pd.DataFrame()
    resolved_initial_cash = float(initial_cash or metrics.get("initial_cash") or _infer_initial_cash(equity))
    monthly = metrics.get("monthly_breakdown") or period_return_breakdown(equity, "M")
    annual = metrics.get("annual_breakdown") or period_return_breakdown(equity, "Y")
    risk = metrics.get("professional_metrics") or professional_performance_metrics(equity, resolved_initial_cash)
    benchmark = metrics.get("benchmarks", {}).get(benchmark_name, {})
    relative = risk.get("relative_to_benchmarks", {}).get(benchmark_name)
    if relative is None:
        relative = relative_performance_metrics(equity, benchmark)
    report = {
        "schema_version": "performance_tearsheet.v1",
        "summary": summary(metrics, equity, resolved_initial_cash),
        "risk_return": risk_return_section(metrics, risk),
        "return_breakdowns": {
            "annual": annual,
            "monthly": monthly,
            "worst_years": worst_periods(annual, 5),
            "worst_months": worst_periods(monthly, 10),
            "consecutive_losing_months": consecutive_losing_periods(monthly),
        },
        "drawdowns": drawdown_intervals(equity),
        "trade_statistics": trade_statistics(trades),
        "benchmark": {
            "name": benchmark_name,
            "summary": benchmark_summary(benchmark),
            "relative": relative,
        },
        "execution_context": execution_context(equity, trades),
        "source_fields": {
            "metrics": sorted(metrics.keys()),
            "equity": sorted(equity.columns.tolist()) if not equity.empty else [],
            "trades": sorted(trades.columns.tolist()) if not trades.empty else [],
        },
    }
    return report


def summary(metrics: dict[str, Any], equity: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    if equity.empty:
        return {
            "initial_cash": finite_float_or_none(initial_cash),
            "final_equity": finite_float_or_none(metrics.get("final_equity")),
            "total_return": finite_float_or_none(metrics.get("total_return")),
        }
    frame = _prepared_equity(equity)
    first_date = pd.Timestamp(frame["entry_date"].iloc[0]).strftime("%Y-%m-%d")
    last_date = pd.Timestamp(frame["entry_date"].iloc[-1]).strftime("%Y-%m-%d")
    final_equity = float(frame["equity"].iloc[-1]) if "equity" in frame else metrics.get("final_equity")
    total_return = final_equity / initial_cash - 1.0 if initial_cash else metrics.get("total_return")
    return {
        "initial_cash": finite_float_or_none(initial_cash),
        "final_equity": finite_float_or_none(final_equity),
        "total_return": finite_float_or_none(total_return),
        "start_date": first_date,
        "end_date": last_date,
        "periods": int(len(frame)),
        "active_period_rate": _mean_or_metric(frame, metrics, "active", "active_period_rate"),
        "positive_period_rate": _positive_rate(frame),
    }


def risk_return_section(metrics: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    return {
        "annual_return": finite_float_or_none(risk.get("annual_return", metrics.get("annual_return"))),
        "annual_volatility": finite_float_or_none(risk.get("annual_volatility")),
        "downside_volatility": finite_float_or_none(risk.get("downside_volatility")),
        "sharpe": finite_float_or_none(risk.get("sharpe")),
        "sortino": finite_float_or_none(risk.get("sortino")),
        "calmar": finite_float_or_none(risk.get("calmar")),
        "max_drawdown": finite_float_or_none(risk.get("max_drawdown", metrics.get("max_drawdown"))),
        "best_period_return": finite_float_or_none(risk.get("best_period_return")),
        "worst_period_return": finite_float_or_none(risk.get("worst_period_return")),
        "skew": finite_float_or_none(risk.get("skew")),
        "kurtosis": finite_float_or_none(risk.get("kurtosis")),
    }


def worst_periods(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        [dict(row) for row in rows if row.get("total_return") is not None],
        key=lambda row: float(row["total_return"]),
    )
    return ordered[:limit]


def consecutive_losing_periods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("total_return")
        if value is not None and float(value) < 0:
            current.append(dict(row))
            if len(current) > len(best):
                best = list(current)
        else:
            current = []
    return {
        "max_streak": int(len(best)),
        "periods": [row.get("period") for row in best],
        "total_return": finite_float_or_none(np.prod([1.0 + float(row["total_return"]) for row in best]) - 1.0)
        if best
        else None,
    }


def drawdown_intervals(equity: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if equity.empty or not {"entry_date", "period_return"}.issubset(equity.columns):
        return []
    frame = _prepared_equity(equity)
    returns = pd.to_numeric(frame["period_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    curve = return_curve(returns)
    if len(curve) == 0:
        return []
    peaks = np.maximum.accumulate(curve)
    drawdowns = curve / peaks - 1.0
    dates = pd.to_datetime(frame["entry_date"]).reset_index(drop=True)
    intervals: list[dict[str, Any]] = []
    index = 0
    while index < len(drawdowns):
        if drawdowns[index] >= 0:
            index += 1
            continue
        start = index
        while start > 0 and drawdowns[start - 1] < 0:
            start -= 1
        trough = index
        while index + 1 < len(drawdowns) and drawdowns[index + 1] < 0:
            index += 1
            if drawdowns[index] < drawdowns[trough]:
                trough = index
        recovery = index + 1 if index + 1 < len(drawdowns) and drawdowns[index + 1] >= 0 else None
        intervals.append(
            {
                "start_date": dates.iloc[start].strftime("%Y-%m-%d"),
                "trough_date": dates.iloc[trough].strftime("%Y-%m-%d"),
                "recovery_date": dates.iloc[recovery].strftime("%Y-%m-%d") if recovery is not None else None,
                "max_drawdown": finite_float_or_none(drawdowns[trough]),
                "duration_periods": int((recovery if recovery is not None else index) - start + 1),
                "recovered": recovery is not None,
            }
        )
        index += 1
    intervals.sort(key=lambda row: abs(row["max_drawdown"] or 0.0), reverse=True)
    return intervals[:limit]


def trade_statistics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"rows": 0, "pnl_available": False}
    frame = trades.copy()
    stats: dict[str, Any] = {
        "rows": int(len(frame)),
        "pnl_available": any(col in frame.columns for col in ("pnl", "return", "trade_return")),
        "status_counts": _value_counts(frame, "status"),
        "side_counts": _value_counts(frame, "side"),
        "reason_counts": _value_counts(frame, "reason"),
    }
    for column in ("desired_notional", "filled_notional", "unfilled_notional"):
        if column in frame:
            stats[column] = finite_float_or_none(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())
    desired = stats.get("desired_notional")
    filled = stats.get("filled_notional")
    if desired and desired > 0 and filled is not None:
        stats["fill_ratio"] = finite_float_or_none(float(filled) / float(desired))
    return stats


def benchmark_summary(benchmark: dict[str, Any]) -> dict[str, Any]:
    keys = ("total_return", "annual_return", "max_drawdown", "annual_volatility", "symbols", "periods")
    return {key: benchmark.get(key) for key in keys if key in benchmark}


def execution_context(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if not equity.empty:
        frame = equity.copy()
        for column in (
            "trade_count",
            "blocked_buy_count",
            "blocked_sell_count",
            "partial_buy_count",
            "partial_sell_count",
            "turnover_value",
            "unfilled_buy_value",
            "unfilled_sell_value",
        ):
            if column in frame:
                context[column] = finite_float_or_none(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())
        for column in ("cash_weight", "invested_weight", "turnover_pct"):
            if column in frame:
                context[f"avg_{column}"] = finite_float_or_none(pd.to_numeric(frame[column], errors="coerce").mean())
    if not trades.empty and {"status", "symbol"}.issubset(trades.columns):
        blocked = trades[trades["status"].astype(str).isin({"blocked", "partial"})]
        context["symbols_with_friction"] = int(blocked["symbol"].astype(str).nunique())
    return context


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown companion for the JSON tear sheet."""
    summary_row = report.get("summary", {})
    risk = report.get("risk_return", {})
    benchmark = report.get("benchmark", {})
    relative = benchmark.get("relative") or {}
    lines = [
        "# Performance Tear Sheet",
        "",
        "## Summary",
        _kv_line("Period", f"{summary_row.get('start_date')} to {summary_row.get('end_date')}"),
        _kv_line("Total return", _pct(summary_row.get("total_return"))),
        _kv_line("Final equity", _number(summary_row.get("final_equity"))),
        "",
        "## Risk Return",
        _kv_line("Annual return", _pct(risk.get("annual_return"))),
        _kv_line("Max drawdown", _pct(risk.get("max_drawdown"))),
        _kv_line("Sharpe", _number(risk.get("sharpe"))),
        _kv_line("Sortino", _number(risk.get("sortino"))),
        _kv_line("Calmar", _number(risk.get("calmar"))),
        "",
        "## Benchmark",
        _kv_line("Benchmark", benchmark.get("name")),
        _kv_line("Total excess return", _pct(relative.get("total_excess_return"))),
        _kv_line("Information ratio", _number(relative.get("information_ratio"))),
        _kv_line("Beta", _number(relative.get("beta"))),
        "",
        "## Worst Months",
    ]
    for row in report.get("return_breakdowns", {}).get("worst_months", [])[:5]:
        lines.append(f"- {row.get('period')}: {_pct(row.get('total_return'))}")
    lines.extend(["", "## Largest Drawdowns"])
    for row in report.get("drawdowns", [])[:5]:
        lines.append(
            "- "
            f"{row.get('start_date')} -> {row.get('trough_date')}: "
            f"{_pct(row.get('max_drawdown'))}, recovered={row.get('recovered')}"
        )
    return "\n".join(lines) + "\n"


def _prepared_equity(equity: pd.DataFrame) -> pd.DataFrame:
    frame = equity.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"])
    return frame.sort_values("entry_date")


def _infer_initial_cash(equity: pd.DataFrame) -> float:
    if equity.empty or not {"equity", "period_return"}.issubset(equity.columns):
        return 1.0
    first_equity = float(pd.to_numeric(equity["equity"], errors="coerce").iloc[0])
    first_return = float(pd.to_numeric(equity["period_return"], errors="coerce").fillna(0.0).iloc[0])
    return first_equity / (1.0 + first_return) if first_return > -1.0 else first_equity


def _mean_or_metric(frame: pd.DataFrame, metrics: dict[str, Any], column: str, metric_key: str) -> float | None:
    if column in frame:
        return finite_float_or_none(pd.to_numeric(frame[column], errors="coerce").mean())
    return finite_float_or_none(metrics.get(metric_key))


def _positive_rate(frame: pd.DataFrame) -> float | None:
    if "period_return" not in frame:
        return None
    returns = pd.to_numeric(frame["period_return"], errors="coerce").dropna()
    return finite_float_or_none((returns > 0).mean()) if len(returns) else None


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype(str).value_counts().sort_index().items()}


def _kv_line(key: str, value: Any) -> str:
    return f"- **{key}:** {value}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.4f}"

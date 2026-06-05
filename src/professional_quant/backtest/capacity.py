"""Capacity and slippage stress-report helpers."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def capacity_stress_plan(config: Any) -> dict[str, Any]:
    """Record the capacity/slippage grid that formal reports should replay."""
    initial_cash = float(_get(config, "initial_cash"))
    pct = float(_get(config, "capacity_pct_of_amount"))
    slippage = float(_get(config, "slippage_bps"))
    impact = float(_get(config, "impact_bps_per_pct_amount"))
    cash_values = sorted({initial_cash, initial_cash * 5.0, initial_cash * 10.0})
    pct_values = sorted({max(pct / 2.0, 0.0001), pct, pct * 2.0})
    slippage_values = sorted({max(slippage / 2.0, 0.0), slippage, slippage * 2.0})
    return {
        "current": {
            "initial_cash": initial_cash,
            "capacity_pct_of_amount": pct,
            "slippage_bps": slippage,
            "impact_bps_per_pct_amount": impact,
            "capacity_equity_mode": _get(config, "capacity_equity_mode"),
        },
        "recommended_grid": {
            "initial_cash": cash_values,
            "capacity_pct_of_amount": pct_values,
            "slippage_bps": slippage_values,
            "impact_bps_per_pct_amount": [impact],
        },
        "status": "grid_declared_not_replayed",
        "next_step": "rerun the same frozen strategy across this grid and compare annual_return/max_drawdown/blocked trades",
    }


def build_capacity_stress_row(
    *,
    initial_cash: float,
    capacity_pct: float,
    slippage_bps: float,
    impact_bps: float,
    capacity_equity_mode: str,
    current: Mapping[str, Any],
    stress_metrics: Mapping[str, Any],
    picks: pd.DataFrame,
    trade_log: pd.DataFrame,
) -> dict[str, Any]:
    """Build one JSON/CSV-safe row for capacity_stress.csv."""
    return {
        "initial_cash": float(initial_cash),
        "capacity_pct_of_amount": float(capacity_pct),
        "slippage_bps": float(slippage_bps),
        "impact_bps_per_pct_amount": float(impact_bps),
        "capacity_equity_mode": capacity_equity_mode,
        "is_current_setting": _is_current_setting(
            initial_cash=float(initial_cash),
            capacity_pct=float(capacity_pct),
            slippage_bps=float(slippage_bps),
            impact_bps=float(impact_bps),
            current=current,
        ),
        "periods": stress_metrics.get("periods"),
        "total_return": stress_metrics.get("total_return"),
        "annual_return": stress_metrics.get("annual_return"),
        "max_drawdown": stress_metrics.get("max_drawdown"),
        "final_equity": stress_metrics.get("final_equity"),
        "executed_trade_count": stress_metrics.get("executed_trade_count"),
        "blocked_buy_count": stress_metrics.get("blocked_buy_count"),
        "blocked_sell_count": stress_metrics.get("blocked_sell_count"),
        "partial_buy_count": stress_metrics.get("partial_buy_count"),
        "partial_sell_count": stress_metrics.get("partial_sell_count"),
        "turnover_blocked_count": stress_metrics.get("turnover_blocked_count"),
        "turnover_value": stress_metrics.get("turnover_value"),
        "unfilled_buy_value": stress_metrics.get("unfilled_buy_value"),
        "unfilled_sell_value": stress_metrics.get("unfilled_sell_value"),
        "turnover_blocked_value": stress_metrics.get("turnover_blocked_value"),
        "avg_cash_weight": stress_metrics.get("avg_cash_weight"),
        "max_position_weight_observed": stress_metrics.get("max_position_weight_observed"),
        "max_industry_weight_observed": stress_metrics.get("max_industry_weight_observed"),
        "industry_blocked_count": stress_metrics.get("industry_blocked_count"),
        "max_period_turnover_pct": stress_metrics.get("max_period_turnover_pct"),
        "pick_count": int(len(picks)),
        "trade_log_rows": int(len(trade_log)),
    }


def mark_capacity_stress_replayed(plan: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    """Return replay metadata after the capacity grid has been evaluated."""
    updated = dict(plan)
    updated["status"] = "grid_replayed"
    updated["rows"] = int(len(rows))
    updated["result_columns"] = list(rows.columns)
    updated["next_step"] = "inspect capacity_stress.csv for return/drawdown degradation and blocked/partial trade pressure"
    return updated


def _is_current_setting(
    *,
    initial_cash: float,
    capacity_pct: float,
    slippage_bps: float,
    impact_bps: float,
    current: Mapping[str, Any],
) -> bool:
    return bool(
        initial_cash == float(current["initial_cash"])
        and capacity_pct == float(current["capacity_pct_of_amount"])
        and slippage_bps == float(current["slippage_bps"])
        and impact_bps == float(current["impact_bps_per_pct_amount"])
    )


def _get(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)

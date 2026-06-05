"""Report-ready portfolio risk budget summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd

from professional_quant.backtest.reporting import finite_float_or_none


def risk_budget_report(
    metrics: dict[str, Any],
    equity_df: pd.DataFrame,
    picks_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize main portfolio risk sources in report-ready form."""
    industry_exposure_rows: list[dict[str, Any]] = []
    if not picks_df.empty and {"industry_label", "weight"}.issubset(picks_df.columns):
        exposure = (
            picks_df.assign(weight=pd.to_numeric(picks_df["weight"], errors="coerce").fillna(0.0))
            .groupby("industry_label", dropna=False)["weight"]
            .agg(["mean", "max", "count"])
            .sort_values("max", ascending=False)
        )
        industry_exposure_rows = [
            {
                "industry_label": str(label),
                "avg_pick_weight": finite_float_or_none(row["mean"]),
                "max_pick_weight": finite_float_or_none(row["max"]),
                "pick_rows": int(row["count"]),
            }
            for label, row in exposure.head(10).iterrows()
        ]

    risk_sources = [
        {
            "name": "market_drawdown",
            "measure": "max_drawdown",
            "value": finite_float_or_none(metrics.get("max_drawdown")),
            "control": "portfolio_stop_loss",
            "limit": config.get("portfolio_stop_loss"),
        },
        {
            "name": "single_name_concentration",
            "measure": "max_position_weight_observed",
            "value": finite_float_or_none(metrics.get("max_position_weight_observed")),
            "control": "max_position_weight",
            "limit": config.get("max_position_weight"),
        },
        {
            "name": "industry_concentration",
            "measure": "max_industry_weight_observed",
            "value": finite_float_or_none(metrics.get("max_industry_weight_observed")),
            "control": "max_industry_weight",
            "limit": config.get("max_industry_weight"),
        },
        {
            "name": "liquidity_capacity",
            "measure": "unfilled_notional",
            "value": finite_float_or_none(
                float(metrics.get("unfilled_buy_value", 0.0)) + float(metrics.get("unfilled_sell_value", 0.0))
            ),
            "control": "capacity_pct_of_amount",
            "limit": config.get("capacity_pct_of_amount"),
        },
        {
            "name": "turnover_pressure",
            "measure": "max_period_turnover_pct",
            "value": finite_float_or_none(metrics.get("max_period_turnover_pct")),
            "control": "max_turnover_pct",
            "limit": config.get("max_turnover_pct"),
        },
        {
            "name": "execution_blocks",
            "measure": "blocked_and_partial_trades",
            "value": int(metrics.get("blocked_buy_count", 0))
            + int(metrics.get("blocked_sell_count", 0))
            + int(metrics.get("partial_buy_count", 0))
            + int(metrics.get("partial_sell_count", 0)),
            "control": "limit_suspend_capacity_rules",
            "limit": "logged",
        },
    ]
    return {
        "risk_sources": risk_sources,
        "industry_exposure_top": industry_exposure_rows,
        "portfolio_risk_off_rate": finite_float_or_none(metrics.get("portfolio_risk_off_rate")),
        "avg_cash_weight": finite_float_or_none(metrics.get("avg_cash_weight")),
        "avg_invested_weight": finite_float_or_none(metrics.get("avg_invested_weight")),
        "constraint_notes": [
            "Single-name, industry, turnover, and capacity controls are measured on rebalance periods.",
            "Industry exposure requires local symbol-industry metadata; missing labels are reported as unknown.",
        ],
    }

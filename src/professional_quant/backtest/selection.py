"""Training-window scoring helpers for dynamic backtest selection."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def metrics_from_returns(
    returns: np.ndarray,
    active: np.ndarray,
    trades: np.ndarray,
    entry_dates: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Compute training-window return/risk metrics from precomputed spec series."""
    period_returns = returns[mask]
    if len(period_returns) == 0:
        return {}
    period_returns = period_returns.astype(np.float64)
    if not np.isfinite(period_returns).all() or np.any(period_returns <= -1.0):
        return {}
    equity_curve = np.cumprod(1.0 + period_returns)
    if not np.isfinite(equity_curve).all():
        return {}
    equity = float(equity_curve[-1])
    if not np.isfinite(equity) or equity <= 0:
        return {}
    peaks = np.maximum.accumulate(equity_curve)
    drawdown = float(np.min(equity_curve / peaks - 1.0))
    used_dates = entry_dates[mask]
    elapsed_days = max((pd.Timestamp(used_dates[-1]) - pd.Timestamp(used_dates[0])).days, 1)
    annual_return = equity ** (365.25 / elapsed_days) - 1.0
    if not np.isfinite(annual_return):
        return {}
    return {
        "total_return": float(equity - 1.0),
        "final_equity": equity,
        "annual_return": float(annual_return),
        "max_drawdown": drawdown,
        "avg_period_return": float(np.mean(period_returns)),
        "period_return_std": float(np.std(period_returns)),
        "positive_period_rate": float(np.mean(period_returns > 0)),
        "active_period_rate": float(np.mean(active[mask])),
        "trade_period_rate": float(np.mean(trades[mask])),
        "periods": int(len(period_returns)),
        "trades": int(np.sum(trades[mask])),
    }


def training_score(row: dict[str, Any], profile: str) -> float:
    """Score one training result under the configured selection profile."""
    annual = row["annual_return"]
    drawdown = abs(min(row["max_drawdown"], 0.0))
    active_rate = row["active_period_rate"]
    positive_rate = row["positive_period_rate"]
    trade_rate = row["trade_period_rate"]
    period_std = row["period_return_std"]
    if not np.isfinite(annual) or not np.isfinite(period_std):
        return -np.inf
    if profile == "robust":
        if active_rate < 0.20 or positive_rate < 0.42 or drawdown > 0.35 or trade_rate > 0.80:
            return -np.inf
        return float(annual - 1.10 * drawdown + 0.10 * positive_rate - 0.05 * period_std - 0.03 * trade_rate)
    if profile == "balanced":
        if active_rate < 0.15 or positive_rate < 0.38 or drawdown > 0.55 or trade_rate > 0.90:
            return -np.inf
        return float(annual - 0.55 * drawdown + 0.06 * positive_rate - 0.02 * trade_rate)
    if profile == "aggressive":
        if active_rate < 0.10 or positive_rate < 0.35 or drawdown > 0.80:
            return -np.inf
        return float(annual - 0.18 * drawdown + 0.03 * positive_rate)
    if profile == "return40":
        if active_rate < 0.08 or positive_rate < 0.34 or drawdown > 0.90 or trade_rate > 0.95:
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(annual + 0.35 * excess_return - 0.20 * target_gap - 0.08 * drawdown + 0.02 * positive_rate)
    if profile == "stable40":
        min_sub_annual = float(row.get("subperiod_min_annual_return", annual))
        worst_sub_drawdown = abs(min(float(row.get("subperiod_worst_drawdown", row["max_drawdown"])), 0.0))
        min_sub_positive = float(row.get("subperiod_min_positive_period_rate", positive_rate))
        subperiod_count = int(row.get("subperiod_count", 0) or 0)
        if active_rate < 0.12 or positive_rate < 0.38 or drawdown > 0.70 or trade_rate > 0.90:
            return -np.inf
        if subperiod_count >= 2 and (min_sub_annual < -0.02 or min_sub_positive < 0.34 or worst_sub_drawdown > 0.78):
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(
            0.70 * annual
            + 0.30 * min_sub_annual
            + 0.25 * excess_return
            - 0.18 * target_gap
            - 0.22 * drawdown
            - 0.12 * worst_sub_drawdown
            + 0.04 * positive_rate
            - 0.03 * trade_rate
        )
    if profile == "stable40q":
        min_sub_annual = float(row.get("subperiod_min_annual_return", annual))
        worst_sub_drawdown = abs(min(float(row.get("subperiod_worst_drawdown", row["max_drawdown"])), 0.0))
        min_sub_positive = float(row.get("subperiod_min_positive_period_rate", positive_rate))
        max_sub_trade = float(row.get("subperiod_max_trade_period_rate", trade_rate))
        subperiod_count = int(row.get("subperiod_count", 0) or 0)
        if active_rate < 0.12 or positive_rate < 0.38 or drawdown > 0.70 or trade_rate > 0.90:
            return -np.inf
        if subperiod_count >= 4 and (
            min_sub_annual < -0.04
            or min_sub_positive < 0.33
            or worst_sub_drawdown > 0.68
            or max_sub_trade > 0.88
        ):
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(
            0.55 * annual
            + 0.45 * min_sub_annual
            + 0.25 * excess_return
            - 0.22 * target_gap
            - 0.28 * drawdown
            - 0.20 * worst_sub_drawdown
            + 0.05 * positive_rate
            - 0.04 * trade_rate
        )
    if profile == "stable40y":
        year_count = int(row.get("year_count", 0) or 0)
        year_positive_rate = float(row.get("year_positive_rate", positive_rate))
        year_min_annual = float(row.get("year_min_annual_return", annual))
        year_median_annual = float(row.get("year_median_annual_return", annual))
        year_max_annual = float(row.get("year_max_annual_return", annual))
        year_worst_drawdown = abs(min(float(row.get("year_worst_drawdown", row["max_drawdown"])), 0.0))
        year_max_trade = float(row.get("year_max_trade_period_rate", trade_rate))
        if annual < 0.25 or active_rate < 0.12 or positive_rate < 0.38 or drawdown > 0.75 or trade_rate > 0.90:
            return -np.inf
        if year_count >= 8 and (
            year_positive_rate < 0.58
            or year_min_annual < -0.35
            or year_median_annual < 0.08
            or year_max_annual < 0.40
            or year_worst_drawdown > 0.58
            or year_max_trade > 0.88
        ):
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(
            0.45 * annual
            + 0.35 * year_median_annual
            + 0.20 * year_min_annual
            + 0.25 * excess_return
            - 0.20 * target_gap
            - 0.24 * drawdown
            - 0.18 * year_worst_drawdown
            + 0.08 * year_positive_rate
            - 0.04 * year_max_trade
        )
    if profile == "durable40":
        min_sub_annual = float(row.get("subperiod_min_annual_return", annual))
        max_sub_annual = float(row.get("subperiod_max_annual_return", annual))
        worst_sub_drawdown = abs(min(float(row.get("subperiod_worst_drawdown", row["max_drawdown"])), 0.0))
        min_sub_positive = float(row.get("subperiod_min_positive_period_rate", positive_rate))
        max_sub_trade = float(row.get("subperiod_max_trade_period_rate", trade_rate))
        subperiod_count = int(row.get("subperiod_count", 0) or 0)
        if active_rate < 0.50 or positive_rate < 0.44 or drawdown > 0.58 or trade_rate > 0.55:
            return -np.inf
        if annual < 0.22 or period_std > 0.035:
            return -np.inf
        if subperiod_count >= 2 and (
            min_sub_annual < 0.08
            or min_sub_positive < 0.45
            or worst_sub_drawdown > 0.48
            or max_sub_trade > 0.55
        ):
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(
            0.38 * annual
            + 0.40 * min_sub_annual
            + 0.08 * max_sub_annual
            + 0.20 * excess_return
            - 0.16 * target_gap
            - 0.18 * drawdown
            - 0.22 * worst_sub_drawdown
            + 0.12 * positive_rate
            + 0.10 * min_sub_positive
            - 0.06 * max_sub_trade
        )
    if profile == "recent40":
        min_sub_annual = float(row.get("subperiod_min_annual_return", annual))
        first_sub_annual = float(row.get("subperiod_first_annual_return", annual))
        last_sub_annual = float(row.get("subperiod_last_annual_return", annual))
        last_sub_drawdown = abs(min(float(row.get("subperiod_last_drawdown", row["max_drawdown"])), 0.0))
        min_sub_positive = float(row.get("subperiod_min_positive_period_rate", positive_rate))
        max_sub_trade = float(row.get("subperiod_max_trade_period_rate", trade_rate))
        subperiod_count = int(row.get("subperiod_count", 0) or 0)
        if active_rate < 0.50 or positive_rate < 0.44 or drawdown > 0.62 or trade_rate > 0.60:
            return -np.inf
        if annual < 0.22 or period_std > 0.038:
            return -np.inf
        if subperiod_count >= 2 and (
            first_sub_annual < 0.00
            or last_sub_annual < 0.16
            or min_sub_annual < 0.05
            or last_sub_drawdown > 0.48
            or min_sub_positive < 0.44
            or max_sub_trade > 0.60
        ):
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(
            0.28 * annual
            + 0.42 * last_sub_annual
            + 0.20 * min_sub_annual
            + 0.15 * excess_return
            - 0.12 * target_gap
            - 0.16 * drawdown
            - 0.20 * last_sub_drawdown
            + 0.10 * positive_rate
            + 0.08 * min_sub_positive
            - 0.05 * max_sub_trade
        )
    if profile == "holdout40":
        min_sub_annual = float(row.get("subperiod_min_annual_return", annual))
        median_sub_annual = float(row.get("subperiod_median_annual_return", annual))
        last_sub_annual = float(row.get("subperiod_last_annual_return", annual))
        worst_sub_drawdown = abs(min(float(row.get("subperiod_worst_drawdown", row["max_drawdown"])), 0.0))
        last_sub_drawdown = abs(min(float(row.get("subperiod_last_drawdown", row["max_drawdown"])), 0.0))
        min_sub_positive = float(row.get("subperiod_min_positive_period_rate", positive_rate))
        max_sub_trade = float(row.get("subperiod_max_trade_period_rate", trade_rate))
        subperiod_count = int(row.get("subperiod_count", 0) or 0)
        if active_rate < 0.45 or positive_rate < 0.42 or drawdown > 0.68 or trade_rate > 0.65:
            return -np.inf
        if annual < 0.20 or period_std > 0.040:
            return -np.inf
        if subperiod_count >= 4 and (
            min_sub_annual < -0.05
            or median_sub_annual < 0.08
            or last_sub_annual < 0.18
            or worst_sub_drawdown > 0.58
            or last_sub_drawdown > 0.48
            or min_sub_positive < 0.40
            or max_sub_trade > 0.65
        ):
            return -np.inf
        excess_return = max(annual - 0.40, 0.0)
        target_gap = max(0.40 - annual, 0.0)
        return float(
            0.24 * annual
            + 0.36 * last_sub_annual
            + 0.22 * median_sub_annual
            + 0.12 * min_sub_annual
            + 0.14 * excess_return
            - 0.12 * target_gap
            - 0.14 * drawdown
            - 0.16 * worst_sub_drawdown
            - 0.10 * last_sub_drawdown
            + 0.08 * positive_rate
            - 0.05 * max_sub_trade
        )
    raise ValueError(profile)


def spec_signature(spec: Any) -> tuple[Any, ...]:
    """Return the stable identity tuple used to cache validation metrics."""
    return (
        spec.formula.name,
        tuple(sorted((str(name), float(weight)) for name, weight in spec.formula.weights.items())),
        spec.market_filter,
        spec.top_n,
        spec.min_amount,
        spec.min_price,
        spec.trend_filter,
        spec.min_hold_days,
        spec.max_hold_days,
        spec.replace_count,
        spec.stop_loss,
    )


def spec_to_row(spec: Any) -> dict[str, Any]:
    """Serialize a dynamic spec to diagnostics columns."""
    return {
        "formula": spec.formula.name,
        "weights": json.dumps(spec.formula.weights, ensure_ascii=False),
        "market_filter": spec.market_filter,
        "top_n": spec.top_n,
        "min_amount": spec.min_amount,
        "min_price": spec.min_price,
        "trend_filter": spec.trend_filter,
        "min_hold_days": spec.min_hold_days,
        "max_hold_days": spec.max_hold_days,
        "replace_count": spec.replace_count,
        "stop_loss": spec.stop_loss,
    }


def choose_yearly_specs_from_series(
    *,
    series_list: list[Any],
    signal_dates: np.ndarray,
    entry_dates: np.ndarray,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_years: int,
    min_train_periods: int,
    keep_top: int,
    score_profile: str,
    freeze_selection_date: pd.Timestamp | None = None,
    validation_metrics_func: Callable[[Any], dict[str, Any]] | None = None,
) -> tuple[dict[int, Any], pd.DataFrame]:
    """Choose yearly specs from precomputed dynamic series and return diagnostics."""
    yearly_specs: dict[int, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    validation_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def validation_metrics_for(row: dict[str, Any]) -> dict[str, Any]:
        spec = row.get("_spec")
        if spec is None or validation_metrics_func is None:
            return {}
        key = spec_signature(spec)
        metrics = validation_cache.get(key)
        if metrics is None:
            metrics = validation_metrics_func(spec)
            if metrics:
                metrics = dict(metrics)
                metrics["wf_score"] = training_score(metrics, score_profile)
            validation_cache[key] = metrics
        return {f"validation_{key}": value for key, value in metrics.items()}

    frozen_spec: Any | None = None
    frozen_train_start_text: str | None = None
    frozen_train_end_text: str | None = None
    if freeze_selection_date is not None:
        freeze_train_end = freeze_selection_date
        freeze_train_start = freeze_train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
        frozen_train_start_text = freeze_train_start.strftime("%Y-%m-%d")
        frozen_train_end_text = freeze_train_end.strftime("%Y-%m-%d")
        freeze_mask = (signal_dates >= freeze_train_start.to_datetime64()) & (
            signal_dates <= freeze_train_end.to_datetime64()
        )
        if int(np.sum(freeze_mask)) < min_train_periods:
            diagnostics.append(
                {
                    "year": None,
                    "status": "frozen_selection_skipped_insufficient_training_periods",
                    "train_start": frozen_train_start_text,
                    "train_end": frozen_train_end_text,
                    "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d"),
                }
            )
        else:
            frozen_results = ranked_series_results(series_list, entry_dates, freeze_mask, score_profile)
            if frozen_results:
                frozen_row = frozen_results[0]
                frozen_spec = frozen_row["_spec"]
                diagnostics.append(
                    {
                        "year": None,
                        "status": "frozen_selected",
                        "train_start": frozen_train_start_text,
                        "train_end": frozen_train_end_text,
                        "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d"),
                        **public_row(frozen_row),
                        **validation_metrics_for(frozen_row),
                    }
                )
                for rank, row in enumerate(frozen_results[:keep_top], start=1):
                    diagnostics.append(
                        {
                            "year": None,
                            "status": "frozen_train_candidate",
                            "selected_rank": rank,
                            "train_start": frozen_train_start_text,
                            "train_end": frozen_train_end_text,
                            "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d"),
                            **public_row(row),
                            **validation_metrics_for(row),
                        }
                    )
            else:
                diagnostics.append(
                    {
                        "year": None,
                        "status": "frozen_selection_skipped_no_valid_training_result",
                        "train_start": frozen_train_start_text,
                        "train_end": frozen_train_end_text,
                        "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d"),
                    }
                )

    for year in range(start_date.year, end_date.year + 1):
        test_start = max(start_date, pd.Timestamp(year=year, month=1, day=1))
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
        if freeze_selection_date is not None and test_start > freeze_selection_date and frozen_spec is not None:
            yearly_specs[year] = frozen_spec
            diagnostics.append(
                {
                    "year": year,
                    "status": "selected_frozen",
                    "train_start": frozen_train_start_text,
                    "train_end": frozen_train_end_text,
                    "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d"),
                    **spec_to_row(frozen_spec),
                }
            )
            continue
        train_mask = (signal_dates >= train_start.to_datetime64()) & (signal_dates <= train_end.to_datetime64())
        if int(np.sum(train_mask)) < min_train_periods:
            diagnostics.append(
                {
                    "year": year,
                    "status": "skipped_insufficient_training_periods",
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                }
            )
            continue

        results = ranked_series_results(series_list, entry_dates, train_mask, score_profile)
        if not results:
            diagnostics.append(
                {
                    "year": year,
                    "status": "skipped_no_valid_training_result",
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                }
            )
            continue

        selected = results[0]
        yearly_specs[year] = selected["_spec"]
        diagnostics.append(
            {
                "year": year,
                "status": "selected",
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                **public_row(selected),
            }
        )
        for rank, row in enumerate(results[:keep_top], start=1):
            diagnostics.append(
                {
                    "year": year,
                    "status": "train_candidate",
                    "selected_rank": rank,
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    **public_row(row),
                }
            )

    return yearly_specs, pd.DataFrame(diagnostics)


def ranked_series_results(
    series_list: list[Any],
    entry_dates: np.ndarray,
    train_mask: np.ndarray,
    score_profile: str,
) -> list[dict[str, Any]]:
    """Rank precomputed spec series for one training mask."""
    results: list[dict[str, Any]] = []
    subperiod_parts = 4 if score_profile in {"stable40q", "holdout40"} else 2
    entry_years = pd.DatetimeIndex(entry_dates).year.to_numpy() if score_profile == "stable40y" else None
    for series in series_list:
        row = metrics_from_returns(series.returns, series.active, series.trades, entry_dates, train_mask)
        if not row:
            continue
        row.update(
            training_subperiod_metrics(
                series.returns,
                series.active,
                series.trades,
                entry_dates,
                train_mask,
                parts=subperiod_parts,
            )
        )
        if score_profile == "stable40y":
            row.update(
                training_calendar_year_metrics(
                    series.returns,
                    series.active,
                    series.trades,
                    entry_dates,
                    train_mask,
                    entry_years=entry_years,
                )
            )
        row["wf_score"] = training_score(row, score_profile)
        if np.isfinite(row["wf_score"]):
            row.update(spec_to_row(series.spec))
            row["_spec"] = series.spec
            row["_series"] = series
            results.append(row)
    results.sort(key=lambda row: row["wf_score"], reverse=True)
    return results


def training_subperiod_metrics(
    returns: np.ndarray,
    active: np.ndarray,
    trades: np.ndarray,
    entry_dates: np.ndarray,
    train_mask: np.ndarray,
    parts: int = 2,
) -> dict[str, Any]:
    """Summarize stability across equal-sized slices inside one training window."""
    indices = np.flatnonzero(train_mask)
    if parts < 2 or len(indices) < parts:
        return {"subperiod_count": 0}
    annual_returns: list[float] = []
    drawdowns: list[float] = []
    positive_rates: list[float] = []
    active_rates: list[float] = []
    trade_rates: list[float] = []
    for split_indices in np.array_split(indices, parts):
        if len(split_indices) == 0:
            continue
        sub_mask = np.zeros(len(train_mask), dtype=bool)
        sub_mask[split_indices] = True
        metrics = metrics_from_returns(returns, active, trades, entry_dates, sub_mask)
        if not metrics:
            continue
        annual_returns.append(float(metrics["annual_return"]))
        drawdowns.append(float(metrics["max_drawdown"]))
        positive_rates.append(float(metrics["positive_period_rate"]))
        active_rates.append(float(metrics["active_period_rate"]))
        trade_rates.append(float(metrics["trade_period_rate"]))
    if not annual_returns:
        return {"subperiod_count": 0}
    return {
        "subperiod_count": int(len(annual_returns)),
        "subperiod_first_annual_return": float(annual_returns[0]),
        "subperiod_last_annual_return": float(annual_returns[-1]),
        "subperiod_min_annual_return": float(min(annual_returns)),
        "subperiod_median_annual_return": float(np.median(np.asarray(annual_returns, dtype=np.float64))),
        "subperiod_max_annual_return": float(max(annual_returns)),
        "subperiod_first_drawdown": float(drawdowns[0]),
        "subperiod_last_drawdown": float(drawdowns[-1]),
        "subperiod_worst_drawdown": float(min(drawdowns)),
        "subperiod_first_positive_period_rate": float(positive_rates[0]),
        "subperiod_last_positive_period_rate": float(positive_rates[-1]),
        "subperiod_min_positive_period_rate": float(min(positive_rates)),
        "subperiod_first_active_period_rate": float(active_rates[0]),
        "subperiod_last_active_period_rate": float(active_rates[-1]),
        "subperiod_min_active_period_rate": float(min(active_rates)),
        "subperiod_first_trade_period_rate": float(trade_rates[0]),
        "subperiod_last_trade_period_rate": float(trade_rates[-1]),
        "subperiod_max_trade_period_rate": float(max(trade_rates)),
    }


def training_calendar_year_metrics(
    returns: np.ndarray,
    active: np.ndarray,
    trades: np.ndarray,
    entry_dates: np.ndarray,
    train_mask: np.ndarray,
    entry_years: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize training stability by calendar year inside the training window."""
    indices = np.flatnonzero(train_mask)
    if len(indices) == 0:
        return {"year_count": 0}
    if entry_years is None:
        entry_years = pd.DatetimeIndex(entry_dates).year.to_numpy()
    years = entry_years[indices]
    annual_returns: list[float] = []
    drawdowns: list[float] = []
    positive_rates: list[float] = []
    active_rates: list[float] = []
    trade_rates: list[float] = []
    for year in sorted(set(int(value) for value in years)):
        year_mask = train_mask & (entry_years == year)
        metrics = metrics_from_returns(returns, active, trades, entry_dates, year_mask)
        if not metrics:
            continue
        annual_returns.append(float(metrics["annual_return"]))
        drawdowns.append(float(metrics["max_drawdown"]))
        positive_rates.append(float(metrics["positive_period_rate"]))
        active_rates.append(float(metrics["active_period_rate"]))
        trade_rates.append(float(metrics["trade_period_rate"]))
    if not annual_returns:
        return {"year_count": 0}
    annual_array = np.asarray(annual_returns, dtype=np.float64)
    return {
        "year_count": int(len(annual_returns)),
        "year_positive_count": int(np.sum(annual_array > 0.0)),
        "year_positive_rate": float(np.mean(annual_array > 0.0)),
        "year_min_annual_return": float(np.min(annual_array)),
        "year_median_annual_return": float(np.median(annual_array)),
        "year_max_annual_return": float(np.max(annual_array)),
        "year_worst_drawdown": float(min(drawdowns)),
        "year_min_positive_period_rate": float(min(positive_rates)),
        "year_min_active_period_rate": float(min(active_rates)),
        "year_max_trade_period_rate": float(max(trade_rates)),
    }


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove private selection fields from diagnostics rows."""
    return {key: value for key, value in row.items() if not str(key).startswith("_")}

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
    for series in series_list:
        row = metrics_from_returns(series.returns, series.active, series.trades, entry_dates, train_mask)
        if not row:
            continue
        row["wf_score"] = training_score(row, score_profile)
        if np.isfinite(row["wf_score"]):
            row.update(spec_to_row(series.spec))
            row["_spec"] = series.spec
            row["_series"] = series
            results.append(row)
    results.sort(key=lambda row: row["wf_score"], reverse=True)
    return results


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove private selection fields from diagnostics rows."""
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


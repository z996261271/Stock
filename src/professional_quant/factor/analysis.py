"""Alphalens-style factor diagnostics built from formal picks artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from professional_quant.backtest.reporting import finite_float_or_none

RETURN_CANDIDATES = (
    "forward_return",
    "period_return",
    "next_return",
    "future_return",
    "return",
)


def build_factor_report(
    picks: pd.DataFrame,
    *,
    equity: pd.DataFrame | None = None,
    score_col: str = "score",
    return_col: str | None = None,
    quantiles: int = 5,
) -> dict[str, Any]:
    """Build a JSON-safe factor report from picks and optional equity data."""
    equity = equity if equity is not None else pd.DataFrame()
    prepared = prepare_picks(picks, score_col=score_col, return_col=return_col, quantiles=quantiles)
    selected_return_col = prepared.attrs.get("return_col")
    has_forward_returns = selected_return_col is not None
    report = {
        "schema_version": "factor_tearsheet.v1",
        "status": "complete" if has_forward_returns else "missing_forward_returns",
        "score_column": score_col,
        "return_column": selected_return_col,
        "quantiles": int(max(1, quantiles)),
        "summary": factor_summary(prepared, score_col=score_col, return_col=selected_return_col),
        "information_coefficient": information_coefficient(prepared, score_col, selected_return_col),
        "quantile_returns": quantile_returns(prepared, selected_return_col),
        "top_bottom_spread": top_bottom_spread(prepared, selected_return_col),
        "quantile_turnover": quantile_turnover(prepared),
        "rank_autocorrelation": rank_autocorrelation(prepared, score_col),
        "industry_summary": industry_summary(prepared, selected_return_col),
        "equity_context": equity_context(equity),
        "warnings": warnings_for_report(prepared, score_col, selected_return_col),
        "source_fields": {
            "picks": sorted(picks.columns.tolist()) if not picks.empty else [],
            "equity": sorted(equity.columns.tolist()) if not equity.empty else [],
        },
    }
    return report


def prepare_picks(
    picks: pd.DataFrame,
    *,
    score_col: str,
    return_col: str | None,
    quantiles: int,
) -> pd.DataFrame:
    if picks.empty:
        frame = pd.DataFrame()
        frame.attrs["return_col"] = None
        return frame
    if "signal_date" not in picks.columns:
        raise ValueError("picks must contain signal_date")
    if "symbol" not in picks.columns:
        raise ValueError("picks must contain symbol")
    if score_col not in picks.columns:
        raise ValueError(f"picks must contain score column: {score_col}")

    frame = picks.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    selected_return_col = select_return_col(frame, return_col)
    if selected_return_col:
        frame[selected_return_col] = pd.to_numeric(frame[selected_return_col], errors="coerce")
    if "weight" in frame:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["signal_date", "symbol", score_col]).sort_values(["signal_date", score_col])
    frame["factor_quantile"] = assign_quantiles(frame, score_col, quantiles)
    frame.attrs["return_col"] = selected_return_col
    return frame


def select_return_col(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"requested return column not found: {requested}")
        return requested
    for candidate in RETURN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    return None


def assign_quantiles(frame: pd.DataFrame, score_col: str, quantiles: int) -> pd.Series:
    bucket_count = int(max(1, quantiles))
    if frame.empty:
        return pd.Series(dtype="Int64")
    pieces: list[pd.Series] = []
    for _, group in frame.groupby("signal_date", sort=True):
        ranks = group[score_col].rank(method="first")
        usable = min(bucket_count, int(len(group)))
        if usable <= 1:
            labels = pd.Series(1, index=group.index, dtype="int64")
        else:
            labels = pd.qcut(ranks, q=usable, labels=False, duplicates="drop") + 1
            labels = labels.astype("int64")
        pieces.append(labels)
    return pd.concat(pieces).sort_index().astype("Int64")


def factor_summary(frame: pd.DataFrame, *, score_col: str, return_col: str | None) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "signal_dates": 0,
            "unique_symbols": 0,
            "has_forward_returns": False,
        }
    first_signal = pd.Timestamp(frame["signal_date"].min()).strftime("%Y-%m-%d")
    last_signal = pd.Timestamp(frame["signal_date"].max()).strftime("%Y-%m-%d")
    per_date = frame.groupby("signal_date")["symbol"].nunique()
    summary = {
        "rows": int(len(frame)),
        "signal_dates": int(frame["signal_date"].nunique()),
        "unique_symbols": int(frame["symbol"].nunique()),
        "first_signal_date": first_signal,
        "last_signal_date": last_signal,
        "avg_names_per_signal": finite_float_or_none(per_date.mean()),
        "min_names_per_signal": int(per_date.min()) if len(per_date) else None,
        "max_names_per_signal": int(per_date.max()) if len(per_date) else None,
        "score_mean": finite_float_or_none(frame[score_col].mean()),
        "score_std": finite_float_or_none(frame[score_col].std(ddof=0)),
        "has_forward_returns": return_col is not None,
    }
    if "weight" in frame:
        summary["avg_weight"] = finite_float_or_none(frame["weight"].mean())
        summary["max_weight"] = finite_float_or_none(frame["weight"].max())
    return summary


def information_coefficient(frame: pd.DataFrame, score_col: str, return_col: str | None) -> dict[str, Any]:
    if frame.empty or return_col is None:
        return {"status": "missing_forward_returns", "daily": [], "summary": {}}
    rows: list[dict[str, Any]] = []
    for signal_date, group in frame.groupby("signal_date", sort=True):
        valid = group[[score_col, return_col]].dropna()
        if len(valid) < 2:
            ic = None
        else:
            ic = valid[score_col].rank(method="average").corr(valid[return_col].rank(method="average"))
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "rank_ic": finite_float_or_none(ic),
                "names": int(len(valid)),
            }
        )
    values = [row["rank_ic"] for row in rows if row["rank_ic"] is not None]
    summary = {
        "periods": int(len(values)),
        "mean_rank_ic": finite_float_or_none(np.mean(values)) if values else None,
        "std_rank_ic": finite_float_or_none(np.std(values, ddof=0)) if values else None,
        "positive_rate": finite_float_or_none(np.mean(np.asarray(values) > 0)) if values else None,
    }
    return {"status": "complete", "daily": rows, "summary": summary}


def quantile_returns(frame: pd.DataFrame, return_col: str | None) -> dict[str, Any]:
    if frame.empty or return_col is None:
        return {"status": "missing_forward_returns", "rows": []}
    rows: list[dict[str, Any]] = []
    grouped = frame.dropna(subset=[return_col]).groupby("factor_quantile", sort=True)
    for quantile, group in grouped:
        returns = group[return_col].to_numpy(dtype=np.float64)
        rows.append(
            {
                "quantile": int(quantile),
                "rows": int(len(group)),
                "mean_return": finite_float_or_none(np.mean(returns)),
                "median_return": finite_float_or_none(np.median(returns)),
                "positive_rate": finite_float_or_none(np.mean(returns > 0)),
            }
        )
    return {"status": "complete", "rows": rows}


def top_bottom_spread(frame: pd.DataFrame, return_col: str | None) -> dict[str, Any]:
    if frame.empty or return_col is None:
        return {"status": "missing_forward_returns", "daily": [], "summary": {}}
    daily: list[dict[str, Any]] = []
    for signal_date, group in frame.dropna(subset=[return_col]).groupby("signal_date", sort=True):
        top = group["factor_quantile"].max()
        bottom = group["factor_quantile"].min()
        top_return = group.loc[group["factor_quantile"] == top, return_col].mean()
        bottom_return = group.loc[group["factor_quantile"] == bottom, return_col].mean()
        daily.append(
            {
                "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "top_quantile": int(top),
                "bottom_quantile": int(bottom),
                "top_return": finite_float_or_none(top_return),
                "bottom_return": finite_float_or_none(bottom_return),
                "spread": finite_float_or_none(top_return - bottom_return),
            }
        )
    spreads = [row["spread"] for row in daily if row["spread"] is not None]
    return {
        "status": "complete",
        "daily": daily,
        "summary": {
            "periods": int(len(spreads)),
            "mean_spread": finite_float_or_none(np.mean(spreads)) if spreads else None,
            "positive_rate": finite_float_or_none(np.mean(np.asarray(spreads) > 0)) if spreads else None,
        },
    }


def quantile_turnover(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "factor_quantile" not in frame:
        return {"status": "empty", "rows": []}
    rows: list[dict[str, Any]] = []
    previous: dict[int, set[str]] = {}
    for signal_date, group in frame.groupby("signal_date", sort=True):
        current = {
            int(quantile): set(values["symbol"].astype(str))
            for quantile, values in group.groupby("factor_quantile", sort=True)
        }
        for quantile, symbols in current.items():
            prior = previous.get(quantile)
            if prior is None:
                turnover = None
            elif not symbols and not prior:
                turnover = 0.0
            else:
                turnover = 1.0 - len(symbols & prior) / max(len(symbols | prior), 1)
            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                    "quantile": quantile,
                    "names": int(len(symbols)),
                    "turnover": finite_float_or_none(turnover),
                }
            )
        previous = current
    values = [row["turnover"] for row in rows if row["turnover"] is not None]
    return {
        "status": "complete",
        "rows": rows,
        "summary": {
            "mean_turnover": finite_float_or_none(np.mean(values)) if values else None,
            "max_turnover": finite_float_or_none(np.max(values)) if values else None,
        },
    }


def rank_autocorrelation(frame: pd.DataFrame, score_col: str) -> dict[str, Any]:
    if frame.empty:
        return {"status": "empty", "rows": [], "summary": {}}
    rows: list[dict[str, Any]] = []
    previous_date: pd.Timestamp | None = None
    previous_ranks: pd.Series | None = None
    for signal_date, group in frame.groupby("signal_date", sort=True):
        current = group.set_index("symbol")[score_col].rank(method="average")
        correlation = None
        overlap = 0
        if previous_ranks is not None:
            common = current.index.intersection(previous_ranks.index)
            overlap = int(len(common))
            if overlap >= 2:
                correlation = current.loc[common].corr(previous_ranks.loc[common])
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date).strftime("%Y-%m-%d"),
                "previous_signal_date": previous_date.strftime("%Y-%m-%d") if previous_date is not None else None,
                "overlap": overlap,
                "rank_autocorrelation": finite_float_or_none(correlation),
            }
        )
        previous_date = pd.Timestamp(signal_date)
        previous_ranks = current
    values = [row["rank_autocorrelation"] for row in rows if row["rank_autocorrelation"] is not None]
    return {
        "status": "complete",
        "rows": rows,
        "summary": {
            "mean_rank_autocorrelation": finite_float_or_none(np.mean(values)) if values else None,
            "min_rank_autocorrelation": finite_float_or_none(np.min(values)) if values else None,
        },
    }


def industry_summary(frame: pd.DataFrame, return_col: str | None) -> dict[str, Any]:
    if frame.empty or "industry_label" not in frame:
        return {"status": "missing_industry", "rows": []}
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("industry_label", dropna=False, sort=True):
        row = {
            "industry_label": str(label),
            "rows": int(len(group)),
            "signal_dates": int(group["signal_date"].nunique()),
            "unique_symbols": int(group["symbol"].nunique()),
            "avg_score": finite_float_or_none(group["score"].mean()) if "score" in group else None,
            "avg_quantile": finite_float_or_none(group["factor_quantile"].mean()),
        }
        if "weight" in group:
            row["avg_weight"] = finite_float_or_none(group["weight"].mean())
            row["max_weight"] = finite_float_or_none(group["weight"].max())
        if return_col is not None:
            returns = group[return_col].dropna().to_numpy(dtype=np.float64)
            row["mean_return"] = finite_float_or_none(np.mean(returns)) if len(returns) else None
            row["positive_rate"] = finite_float_or_none(np.mean(returns > 0)) if len(returns) else None
        rows.append(row)
    rows.sort(key=lambda item: (item.get("rows") or 0, item.get("max_weight") or 0.0), reverse=True)
    return {"status": "complete", "rows": rows}


def equity_context(equity: pd.DataFrame) -> dict[str, Any]:
    if equity.empty:
        return {}
    context = {"rows": int(len(equity))}
    if "period_return" in equity:
        returns = pd.to_numeric(equity["period_return"], errors="coerce").dropna()
        context["period_return_mean"] = finite_float_or_none(returns.mean()) if len(returns) else None
        context["period_return_positive_rate"] = finite_float_or_none((returns > 0).mean()) if len(returns) else None
    if "cash_weight" in equity:
        cash = pd.to_numeric(equity["cash_weight"], errors="coerce").dropna()
        context["avg_cash_weight"] = finite_float_or_none(cash.mean()) if len(cash) else None
    if "trade" in equity:
        context["rebalance_periods"] = int(pd.to_numeric(equity["trade"], errors="coerce").fillna(0).sum())
    return context


def warnings_for_report(frame: pd.DataFrame, score_col: str, return_col: str | None) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if frame.empty:
        warnings.append({"type": "empty_picks", "message": "picks file has no usable rows"})
        return warnings
    if return_col is None:
        warnings.append(
            {
                "type": "missing_forward_returns",
                "message": "picks do not include per-symbol forward returns; IC and quantile returns are unavailable",
            }
        )
    per_date = frame.groupby("signal_date")["symbol"].nunique()
    if len(per_date) and per_date.min() < 2:
        warnings.append(
            {
                "type": "thin_cross_section",
                "message": "some signal dates have fewer than two symbols, limiting rank diagnostics",
            }
        )
    if frame[score_col].nunique(dropna=True) < 2:
        warnings.append({"type": "constant_score", "message": "score column has fewer than two distinct values"})
    return warnings

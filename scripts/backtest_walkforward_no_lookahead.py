#!/usr/bin/env python3
"""Walk-forward A-share daily-bar strategy research without look-ahead bias.

The script intentionally separates three dates:
- signal_date: factors are computed from the signal close and earlier bars only.
- entry_date: the first trading day after signal_date, using next open.
- exit_date: the first trading day after the next signal_date, using next open.

Strategy parameters are selected year-by-year from completed prior-year results.
That avoids choosing one best full-sample parameter set and replaying it over the
same period.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_data_quality import adjustment_coverage, require_factor_adjust_coverage
from quant_universe import board_scope_sql
from run_manifest import collect_manifest, write_manifest

FACTOR_CACHE_VERSION = "v5"

FEATURES = [
    "mom_252_21_r",
    "mom_126_21_r",
    "mom_63_r",
    "mom_21_r",
    "mom_10_r",
    "mom_5_r",
    "rev_20_r",
    "rev_10_r",
    "rev_5_r",
    "near_high_252_r",
    "near_high_126_r",
    "near_high_63_r",
    "lowvol_21_r",
    "lowvol_63_r",
    "highvol_21_r",
    "highvol_63_r",
    "maxret_5_r",
    "lowmax_5_r",
    "lowmax_21_r",
    "panic_21_r",
    "liq_21_r",
    "lowliq_21_r",
    "surge_5_21_r",
    "surge_21_63_r",
    "trend_63_126_r",
    "ma_stack_r",
    "time_series_r",
    "rsrs_18_r",
    "range_contract_r",
    "efficiency_21_r",
    "money_strength_21_r",
    "gap_5_r",
    "amplitude_21_r",
    "drawdown_63_r",
    "turnover_21_r",
    "low_turnover_21_r",
    "turnover_contract_21_63_r",
    "amount_contract_21_63_r",
    "vol_contract_21_63_r",
    "low_downvol_63_r",
    "low_beta_63_r",
    "low_idio_vol_63_r",
    "deep_drawdown_63_r",
]

G_DAY_SETS: dict[str, list["DayData"]] = {}
G_MARKET: dict[str, set[pd.Timestamp]] = {}
G_SERIES_DATES: dict[str, dict[str, np.ndarray]] = {}
G_ROUND_TRIP_COST = 0.0011
G_SCORE_PROFILE = "robust"


@dataclass(frozen=True)
class Formula:
    name: str
    weights: dict[str, float]


@dataclass(frozen=True)
class Spec:
    frequency: str
    formula: Formula
    market_filter: str
    top_n: int
    min_amount: float
    min_price: float
    trend_filter: str


@dataclass
class DayData:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbols: np.ndarray
    names: np.ndarray
    signal_close: np.ndarray
    entry_open: np.ndarray
    features: np.ndarray
    amount21: np.ndarray
    close: np.ndarray
    ret: np.ndarray
    trend_masks: dict[str, np.ndarray]


@dataclass
class CandidateSeries:
    spec: Spec
    returns: np.ndarray
    active: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward no-lookahead A-share strategy scan.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--min-train-periods", type=int, default=80)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--keep-top", type=int, default=100)
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--frequency", choices=["all", "D", "W", "10D", "20D", "M"], default="all")
    parser.add_argument("--top-n", default="5", help="position count, e.g. 5 or all for 1/2/3/5")
    parser.add_argument(
        "--score-profile",
        choices=["robust", "balanced", "aggressive"],
        default="robust",
        help="yearly training selector: robust controls drawdown, aggressive prioritizes annual return",
    )
    parser.add_argument(
        "--formula-set",
        choices=["base", "expanded"],
        default="expanded",
        help="candidate formula pool: base uses the original stable formulas, expanded includes short-cycle attack formulas",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--board-scope",
        choices=["main", "all"],
        default="main",
        help="stock universe: main keeps Shanghai/Shenzhen main-board prefixes only",
    )
    parser.add_argument(
        "--factor-adjust",
        choices=["raw", "qfq", "hfq"],
        default="hfq",
        help="price adjustment used for factor history; execution always uses raw prices",
    )
    parser.add_argument(
        "--strict-factor-adjust",
        action="store_true",
        help="require factor-adjust rows instead of falling back to raw where adjusted rows are missing",
    )
    return parser.parse_args()


def parse_date(value: str) -> pd.Timestamp:
    normalized = value.strip().replace("-", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError("date must be YYYY-MM-DD or YYYYMMDD")
    return pd.Timestamp(datetime.strptime(normalized, "%Y%m%d").date())


def load_data(
    db: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    board_scope: str = "main",
    factor_adjust: str = "hfq",
    allow_factor_fallback: bool = True,
) -> pd.DataFrame:
    lookback_start = start_date - pd.Timedelta(days=460)
    board_clause, board_params = board_scope_sql(board_scope, "d")
    strict_clause = "" if allow_factor_fallback or factor_adjust == "raw" else " and f.symbol is not null"
    query = f"""
        select d.symbol, coalesce(s.name, d.symbol) as name, d.trade_date,
               d.open as raw_open, d.high as raw_high, d.low as raw_low, d.close as raw_close,
               d.volume as raw_volume, d.amount, d.turnover, d.pct_chg as raw_pct_chg,
               coalesce(f.open, d.open) as open,
               coalesce(f.high, d.high) as high,
               coalesce(f.low, d.low) as low,
               coalesce(f.close, d.close) as close,
               case when f.symbol is not null then ? else 'raw' end as factor_adjust_used
        from daily_bars d
        left join symbols s on s.symbol = d.symbol
        left join daily_bars f
          on f.symbol = d.symbol
         and f.trade_date = d.trade_date
         and f.adjust = ?
        where d.adjust = 'raw'
          and d.trade_date >= ?
          and d.trade_date <= ?
          and d.open is not null
          and d.high is not null
          and d.low is not null
          and d.close is not null
          and d.amount is not null
          {board_clause}
          {strict_clause}
    """
    with sqlite3.connect(db) as conn:
        coverage = adjustment_coverage(conn, lookback_start, end_date, board_scope, factor_adjust)
        if not allow_factor_fallback:
            coverage = require_factor_adjust_coverage(conn, lookback_start, end_date, board_scope, factor_adjust)
        df = pd.read_sql_query(
            query,
            conn,
            params=(
                factor_adjust,
                factor_adjust,
                lookback_start.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                *board_params,
            ),
        )
    if df.empty:
        raise RuntimeError("no daily bars loaded")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"])
    for column in [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "amount",
        "turnover",
        "raw_pct_chg",
        "open",
        "high",
        "low",
        "close",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    used = sorted(str(value) for value in df["factor_adjust_used"].dropna().unique())
    if factor_adjust != "raw" and coverage["missing_raw_rows"] > 0:
        print(
            f"warning: {factor_adjust} factor coverage incomplete; using raw fallback where missing "
            f"(board_scope={board_scope}, raw_rows={coverage['raw_rows']}, "
            f"adjusted_rows={coverage['adjusted_rows']}, missing_raw_rows={coverage['missing_raw_rows']}, "
            f"coverage_ratio={coverage['coverage_ratio']:.6f})",
            flush=True,
        )
    out = df.dropna(subset=["raw_open", "raw_high", "raw_low", "raw_close", "open", "high", "low", "close", "amount"])
    out.attrs["factor_adjust_coverage"] = coverage
    out.attrs["factor_adjust_used"] = used
    return out


def load_or_build_factors(
    db: Path,
    cache_dir: Path,
    research_start: pd.Timestamp,
    end_date: pd.Timestamp,
    use_cache: bool,
    board_scope: str = "main",
    factor_adjust: str = "hfq",
    allow_factor_fallback: bool = True,
) -> pd.DataFrame:
    coverage_start = research_start - pd.Timedelta(days=460)
    with sqlite3.connect(db) as conn:
        if allow_factor_fallback:
            adjustment_coverage(conn, coverage_start, end_date, board_scope, factor_adjust)
        else:
            require_factor_adjust_coverage(conn, coverage_start, end_date, board_scope, factor_adjust)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_tag = "fallback" if allow_factor_fallback else "strict"
    cache_stem = (
        f"walkforward_factors_{FACTOR_CACHE_VERSION}_{board_scope}_{factor_adjust}_{fallback_tag}"
        f"_{research_start:%Y%m%d}_{end_date:%Y%m%d}"
    )
    feather_file = cache_dir / f"{cache_stem}.feather"
    pickle_file = cache_dir / f"{cache_stem}.pkl"
    if use_cache and feather_file.exists():
        print(f"loading factor cache {feather_file}", flush=True)
        return pd.read_feather(feather_file)
    if use_cache and pickle_file.exists():
        print(f"loading factor cache {pickle_file}", flush=True)
        return pd.read_pickle(pickle_file)
    df = add_factors(load_data(db, research_start, end_date, board_scope, factor_adjust, allow_factor_fallback))
    if use_cache:
        try:
            temp_file = feather_file.with_suffix(".tmp.feather")
            df.reset_index(drop=True).to_feather(temp_file)
            temp_file.replace(feather_file)
            print(f"wrote factor cache {feather_file}", flush=True)
        except Exception as exc:
            temp_file = pickle_file.with_suffix(".tmp.pkl")
            df.to_pickle(temp_file)
            temp_file.replace(pickle_file)
            print(f"wrote pickle factor cache {pickle_file} because feather failed: {exc}", flush=True)
    return df


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby("symbol", group_keys=False)
    df["ret_1"] = group["close"].pct_change()
    df["prev_close"] = group["close"].shift(1)
    df["raw_prev_close"] = group["raw_close"].shift(1)
    df["gap_1"] = df["raw_open"] / df["raw_prev_close"] - 1.0
    intraday_range = (df["high"] / df["low"].replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
    df["mom_252_21"] = group["close"].shift(21) / group["close"].shift(252) - 1.0
    df["mom_126_21"] = group["close"].shift(21) / group["close"].shift(126) - 1.0
    df["mom_63"] = group["close"].pct_change(63)
    df["mom_21"] = group["close"].pct_change(21)
    df["mom_10"] = group["close"].pct_change(10)
    df["mom_5"] = group["close"].pct_change(5)
    df["rev_20"] = -df["mom_21"]
    df["rev_10"] = -df["mom_10"]
    df["rev_5"] = -group["close"].pct_change(5)
    for window in [21, 63, 126, 252]:
        df[f"ma_{window}"] = group["close"].rolling(window).mean().reset_index(level=0, drop=True)
    for window in [63, 126, 252]:
        high = group["close"].rolling(window).max().reset_index(level=0, drop=True)
        df[f"near_high_{window}"] = df["close"] / high
    df["vol_63"] = group["ret_1"].rolling(63).std().reset_index(level=0, drop=True)
    df["vol_21"] = group["ret_1"].rolling(21).std().reset_index(level=0, drop=True)
    mkt_ret_1 = df.groupby("trade_date")["ret_1"].transform("mean")
    mkt_var_63 = mkt_ret_1.groupby(df["symbol"]).rolling(63).var().reset_index(level=0, drop=True)
    cov_63 = (
        (df["ret_1"] * mkt_ret_1).groupby(df["symbol"]).rolling(63).mean().reset_index(level=0, drop=True)
        - group["ret_1"].rolling(63).mean().reset_index(level=0, drop=True)
        * mkt_ret_1.groupby(df["symbol"]).rolling(63).mean().reset_index(level=0, drop=True)
    )
    df["beta_63"] = cov_63 / mkt_var_63.replace(0, np.nan)
    df["idio_ret_1"] = df["ret_1"] - df["beta_63"] * mkt_ret_1
    df["idio_vol_63"] = df["idio_ret_1"].groupby(df["symbol"]).rolling(63).std().reset_index(level=0, drop=True)
    down_ret_1 = df["ret_1"].where(df["ret_1"] < 0, 0.0)
    df["down_vol_63"] = down_ret_1.groupby(df["symbol"]).rolling(63).std().reset_index(level=0, drop=True)
    df["max_ret_21"] = group["ret_1"].rolling(21).max().reset_index(level=0, drop=True)
    df["max_ret_5"] = group["ret_1"].rolling(5).max().reset_index(level=0, drop=True)
    df["min_ret_21"] = group["ret_1"].rolling(21).min().reset_index(level=0, drop=True)
    df["panic_21"] = -df["min_ret_21"]
    df["range_21"] = intraday_range.groupby(df["symbol"]).rolling(21).mean().reset_index(level=0, drop=True)
    df["range_63"] = intraday_range.groupby(df["symbol"]).rolling(63).mean().reset_index(level=0, drop=True)
    df["range_contract"] = -(df["range_21"] / df["range_63"] - 1.0)
    abs_ret_21 = group["ret_1"].rolling(21).apply(lambda values: np.abs(values).sum(), raw=True).reset_index(level=0, drop=True)
    df["efficiency_21"] = df["mom_21"].abs() / abs_ret_21.replace(0, np.nan)
    up_amount = df["amount"].where(df["ret_1"] > 0, 0.0)
    down_amount = df["amount"].where(df["ret_1"] < 0, 0.0)
    up_amount_21 = up_amount.groupby(df["symbol"]).rolling(21).sum().reset_index(level=0, drop=True)
    down_amount_21 = down_amount.groupby(df["symbol"]).rolling(21).sum().reset_index(level=0, drop=True)
    df["money_strength_21"] = up_amount_21 / (up_amount_21 + down_amount_21).replace(0, np.nan)
    df["gap_5"] = group["gap_1"].rolling(5).mean().reset_index(level=0, drop=True)
    df["amplitude_21"] = df["range_21"]
    df["amount_ma_5"] = group["amount"].rolling(5).mean().reset_index(level=0, drop=True)
    df["amount_ma_21"] = group["amount"].rolling(21).mean().reset_index(level=0, drop=True)
    df["amount_ma_63"] = group["amount"].rolling(63).mean().reset_index(level=0, drop=True)
    df["surge_5_21"] = df["amount_ma_5"] / df["amount_ma_21"]
    df["surge_21_63"] = df["amount_ma_21"] / df["amount_ma_63"]
    df["turnover_ma_21"] = group["turnover"].rolling(21).mean().reset_index(level=0, drop=True)
    df["turnover_ma_63"] = group["turnover"].rolling(63).mean().reset_index(level=0, drop=True)
    df["turnover_contract_21_63"] = -(df["turnover_ma_21"] / df["turnover_ma_63"].replace(0, np.nan) - 1.0)
    df["amount_contract_21_63"] = -(df["amount_ma_21"] / df["amount_ma_63"].replace(0, np.nan) - 1.0)
    df["vol_contract_21_63"] = -(df["vol_21"] / df["vol_63"].replace(0, np.nan) - 1.0)
    df["trend_63_126"] = df["ma_63"] / df["ma_126"] - 1.0
    high_63 = group["close"].rolling(63).max().reset_index(level=0, drop=True)
    df["drawdown_63"] = df["close"] / high_63 - 1.0
    high_mean_18 = group["high"].rolling(18).mean().reset_index(level=0, drop=True)
    low_mean_18 = group["low"].rolling(18).mean().reset_index(level=0, drop=True)
    df["rsrs_18"] = high_mean_18 / low_mean_18.replace(0, np.nan) - 1.0
    df["ma_stack"] = (
        (df["close"] > df["ma_21"]).astype(float)
        + (df["ma_21"] > df["ma_63"]).astype(float)
        + (df["ma_63"] > df["ma_126"]).astype(float)
    )
    df["time_series"] = (
        (df["mom_252_21"] > 0).astype(float)
        + (df["mom_126_21"] > 0).astype(float)
        + (df["mom_63"] > 0).astype(float)
        + (df["close"] > df["ma_126"]).astype(float)
    )
    return df


def formulas(formula_set: str = "expanded") -> list[Formula]:
    base = [
        Formula(
            "academic_momentum",
            {"mom_252_21_r": 0.45, "mom_126_21_r": 0.25, "near_high_252_r": 0.15, "liq_21_r": 0.15},
        ),
        Formula(
            "industry_proxy_breakout",
            {"near_high_252_r": 0.35, "mom_126_21_r": 0.25, "mom_63_r": 0.20, "surge_5_21_r": 0.20},
        ),
        Formula(
            "low_volume_winner",
            {"mom_252_21_r": 0.35, "mom_126_21_r": 0.25, "lowliq_21_r": 0.20, "lowvol_63_r": 0.20},
        ),
        Formula(
            "time_series_quality",
            {"time_series_r": 0.30, "trend_63_126_r": 0.25, "near_high_126_r": 0.20, "lowmax_21_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "pullback_in_uptrend",
            {"mom_126_21_r": 0.30, "time_series_r": 0.25, "rev_5_r": 0.20, "trend_63_126_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "volume_breakout",
            {"surge_5_21_r": 0.30, "near_high_126_r": 0.25, "mom_21_r": 0.20, "mom_63_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "anti_lottery_momentum",
            {"mom_126_21_r": 0.30, "lowmax_21_r": 0.25, "lowvol_63_r": 0.20, "trend_63_126_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "high_beta_attack",
            {"mom_126_21_r": 0.30, "mom_63_r": 0.25, "highvol_21_r": 0.20, "surge_5_21_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "rsrs_breakout",
            {"rsrs_18_r": 0.30, "near_high_126_r": 0.25, "mom_63_r": 0.20, "money_strength_21_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "squeeze_breakout",
            {"range_contract_r": 0.30, "near_high_126_r": 0.25, "surge_5_21_r": 0.20, "mom_21_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "efficient_trend",
            {"efficiency_21_r": 0.30, "trend_63_126_r": 0.25, "mom_126_21_r": 0.20, "lowvol_63_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "money_flow_leader",
            {"money_strength_21_r": 0.30, "mom_63_r": 0.25, "near_high_126_r": 0.20, "surge_5_21_r": 0.15, "liq_21_r": 0.10},
        ),
        Formula(
            "gap_strength",
            {"gap_5_r": 0.25, "mom_21_r": 0.25, "money_strength_21_r": 0.20, "near_high_126_r": 0.20, "liq_21_r": 0.10},
        ),
        Formula(
            "calm_money_trend",
            {"money_strength_21_r": 0.25, "range_contract_r": 0.25, "efficiency_21_r": 0.20, "trend_63_126_r": 0.20, "liq_21_r": 0.10},
        ),
        Formula(
            "pullback_money",
            {"drawdown_63_r": 0.25, "money_strength_21_r": 0.25, "mom_126_21_r": 0.20, "range_contract_r": 0.15, "liq_21_r": 0.15},
        ),
        Formula(
            "low_turnover_trend",
            {"mom_126_21_r": 0.30, "low_turnover_21_r": 0.25, "lowvol_63_r": 0.20, "near_high_126_r": 0.15, "liq_21_r": 0.10},
        ),
    ]
    expanded = [
        Formula(
            "short_momentum_surge",
            {"mom_21_r": 0.35, "mom_10_r": 0.25, "surge_5_21_r": 0.25, "liq_21_r": 0.15},
        ),
        Formula(
            "micro_squeeze",
            {"mom_21_r": 0.25, "rev_5_r": 0.20, "surge_5_21_r": 0.25, "lowliq_21_r": 0.15, "range_contract_r": 0.15},
        ),
        Formula(
            "panic_rebound",
            {"rev_20_r": 0.30, "rev_10_r": 0.25, "panic_21_r": 0.20, "surge_5_21_r": 0.15, "mom_63_r": 0.10},
        ),
        Formula(
            "volume_acceleration",
            {"surge_5_21_r": 0.30, "surge_21_63_r": 0.20, "near_high_63_r": 0.20, "mom_10_r": 0.20, "liq_21_r": 0.10},
        ),
        Formula(
            "limit_strength",
            {"mom_21_r": 0.25, "maxret_5_r": 0.25, "highvol_21_r": 0.20, "surge_5_21_r": 0.20, "liq_21_r": 0.10},
        ),
        Formula(
            "low_noise_breakout",
            {"near_high_63_r": 0.25, "mom_21_r": 0.20, "lowvol_21_r": 0.20, "lowmax_5_r": 0.20, "liq_21_r": 0.15},
        ),
        Formula(
            "quiet_accumulation",
            {
                "low_turnover_21_r": 0.25,
                "money_strength_21_r": 0.20,
                "near_high_126_r": 0.20,
                "lowvol_63_r": 0.20,
                "mom_63_r": 0.15,
            },
        ),
        Formula(
            "low_turnover_pullback",
            {
                "low_turnover_21_r": 0.25,
                "mom_126_21_r": 0.25,
                "time_series_r": 0.20,
                "rev_5_r": 0.15,
                "lowvol_63_r": 0.15,
            },
        ),
        Formula(
            "low_turnover_pullback_calm",
            {
                "low_turnover_21_r": 0.30,
                "mom_126_21_r": 0.20,
                "time_series_r": 0.20,
                "lowvol_63_r": 0.20,
                "rev_5_r": 0.10,
            },
        ),
        Formula(
            "low_turnover_pullback_fast",
            {
                "mom_126_21_r": 0.25,
                "rev_5_r": 0.25,
                "time_series_r": 0.20,
                "low_turnover_21_r": 0.20,
                "lowvol_63_r": 0.10,
            },
        ),
        Formula(
            "low_turnover_pullback_trend",
            {
                "mom_126_21_r": 0.30,
                "time_series_r": 0.25,
                "low_turnover_21_r": 0.20,
                "rev_5_r": 0.15,
                "lowvol_63_r": 0.10,
            },
        ),
        Formula(
            "squeeze_trend_quality",
            {
                "range_contract_r": 0.25,
                "near_high_126_r": 0.25,
                "mom_126_21_r": 0.20,
                "lowvol_63_r": 0.20,
                "low_turnover_21_r": 0.10,
            },
        ),
        Formula(
            "pullback_defensive_trend",
            {
                "mom_126_21_r": 0.25,
                "time_series_r": 0.25,
                "rev_5_r": 0.20,
                "lowvol_63_r": 0.20,
                "low_turnover_21_r": 0.10,
            },
        ),
        Formula(
            "industry_neutral_reversal",
            {
                "ind_rev_5_r": 0.30,
                "ind_low_turnover_21_r": 0.25,
                "ind_lowvol_63_r": 0.20,
                "ind_lowmax_21_r": 0.15,
                "liq_21_r": 0.10,
            },
        ),
        Formula(
            "industry_neutral_lowvol_pullback",
            {
                "ind_low_turnover_21_r": 0.30,
                "ind_lowvol_63_r": 0.25,
                "ind_lowmax_21_r": 0.20,
                "ind_rev_5_r": 0.15,
                "liq_21_r": 0.10,
            },
        ),
        Formula(
            "industry_neutral_defensive_trend",
            {
                "ind_mom_126_21_r": 0.25,
                "time_series_r": 0.25,
                "ind_rev_5_r": 0.20,
                "ind_lowvol_63_r": 0.20,
                "ind_low_turnover_21_r": 0.10,
            },
        ),
        Formula(
            "low_beta_pullback_trend",
            {
                "mom_126_21_r": 0.25,
                "time_series_r": 0.20,
                "rev_5_r": 0.20,
                "low_beta_63_r": 0.15,
                "low_downvol_63_r": 0.10,
                "low_turnover_21_r": 0.10,
            },
        ),
        Formula(
            "dryup_reacceleration",
            {
                "amount_contract_21_63_r": 0.25,
                "turnover_contract_21_63_r": 0.20,
                "money_strength_21_r": 0.20,
                "mom_21_r": 0.15,
                "low_idio_vol_63_r": 0.10,
                "rev_5_r": 0.10,
            },
        ),
        Formula(
            "vol_compression_breakout",
            {
                "vol_contract_21_63_r": 0.25,
                "range_contract_r": 0.20,
                "near_high_63_r": 0.20,
                "mom_10_r": 0.15,
                "surge_5_21_r": 0.10,
                "liq_21_r": 0.10,
            },
        ),
        Formula(
            "drawdown_repair_quality",
            {
                "deep_drawdown_63_r": 0.20,
                "rev_10_r": 0.20,
                "money_strength_21_r": 0.20,
                "low_downvol_63_r": 0.15,
                "low_beta_63_r": 0.15,
                "liq_21_r": 0.10,
            },
        ),
    ]
    if formula_set == "base":
        return base
    if formula_set == "expanded":
        return base + expanded
    raise ValueError(f"unknown formula set: {formula_set}")


def signal_dates(all_dates: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> list[pd.Timestamp]:
    dates = pd.Series(all_dates[(all_dates >= start_date) & (all_dates <= end_date)])
    if dates.empty:
        return []
    if frequency == "W":
        return list(dates.groupby(dates.dt.to_period("W-FRI")).max())
    if frequency == "M":
        return list(dates.groupby(dates.dt.to_period("M")).max())
    if frequency == "D":
        return list(dates)
    if frequency == "10D":
        return list(dates.iloc[::10])
    if frequency == "20D":
        return list(dates.iloc[::20])
    raise ValueError(frequency)


def add_rank_columns(signal_df: pd.DataFrame) -> pd.DataFrame:
    signal_df = signal_df.copy()
    grouped = signal_df.groupby("trade_date", group_keys=False)
    rank_map = {
        "mom_252_21_r": ("mom_252_21", True),
        "mom_126_21_r": ("mom_126_21", True),
        "mom_63_r": ("mom_63", True),
        "mom_21_r": ("mom_21", True),
        "mom_10_r": ("mom_10", True),
        "mom_5_r": ("mom_5", True),
        "rev_20_r": ("rev_20", True),
        "rev_10_r": ("rev_10", True),
        "rev_5_r": ("rev_5", True),
        "near_high_252_r": ("near_high_252", True),
        "near_high_126_r": ("near_high_126", True),
        "near_high_63_r": ("near_high_63", True),
        "lowvol_21_r": ("vol_21", False),
        "lowvol_63_r": ("vol_63", False),
        "highvol_21_r": ("vol_21", True),
        "highvol_63_r": ("vol_63", True),
        "maxret_5_r": ("max_ret_5", True),
        "lowmax_5_r": ("max_ret_5", False),
        "lowmax_21_r": ("max_ret_21", False),
        "panic_21_r": ("panic_21", True),
        "liq_21_r": ("amount_ma_21", True),
        "lowliq_21_r": ("amount_ma_21", False),
        "surge_5_21_r": ("surge_5_21", True),
        "surge_21_63_r": ("surge_21_63", True),
        "trend_63_126_r": ("trend_63_126", True),
        "ma_stack_r": ("ma_stack", True),
        "time_series_r": ("time_series", True),
        "rsrs_18_r": ("rsrs_18", True),
        "range_contract_r": ("range_contract", True),
        "efficiency_21_r": ("efficiency_21", True),
        "money_strength_21_r": ("money_strength_21", True),
        "gap_5_r": ("gap_5", True),
        "amplitude_21_r": ("amplitude_21", True),
        "drawdown_63_r": ("drawdown_63", True),
        "turnover_21_r": ("turnover_ma_21", True),
        "low_turnover_21_r": ("turnover_ma_21", False),
        "turnover_contract_21_63_r": ("turnover_contract_21_63", True),
        "amount_contract_21_63_r": ("amount_contract_21_63", True),
        "vol_contract_21_63_r": ("vol_contract_21_63", True),
        "low_downvol_63_r": ("down_vol_63", False),
        "low_beta_63_r": ("beta_63", False),
        "low_idio_vol_63_r": ("idio_vol_63", False),
        "deep_drawdown_63_r": ("drawdown_63", False),
    }
    for rank_name, (column, ascending) in rank_map.items():
        signal_df[rank_name] = grouped[column].rank(pct=True, ascending=ascending)
    return signal_df


def build_day_data(
    df: pd.DataFrame,
        open_price: pd.DataFrame,
        close_price: pd.DataFrame,
    dates: list[pd.Timestamp],
    all_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> list[DayData]:
    signal_df = add_rank_columns(df[df["trade_date"].isin(set(dates))].copy())
    output: list[DayData] = []
    for index, signal_date in enumerate(dates):
        future = all_dates[all_dates > signal_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        if index + 1 < len(dates):
            next_future = all_dates[all_dates > dates[index + 1]]
            if len(next_future) == 0:
                continue
            exit_date = next_future[0]
            exit_prices = open_price.loc[exit_date]
        else:
            exit_date = min(end_date, all_dates[-1])
            exit_prices = close_price.loc[exit_date]
        group = signal_df[signal_df["trade_date"] == signal_date].copy().sort_values("symbol")
        if group.empty:
            continue
        symbols = group["symbol"].astype(str).to_numpy()
        entry = open_price.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        exit_ = exit_prices.reindex(symbols).to_numpy(dtype=float)
        period_ret = exit_ / entry - 1.0
        output.append(
            DayData(
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                symbols=symbols,
                names=group["name"].astype(str).to_numpy(),
                signal_close=group["raw_close"].to_numpy(dtype=float),
                entry_open=entry.astype(np.float32),
                features=group[FEATURES].to_numpy(dtype=np.float32).T,
                amount21=group["amount_ma_21"].to_numpy(dtype=np.float32),
                close=group["raw_close"].to_numpy(dtype=np.float32),
                ret=period_ret.astype(np.float32),
                trend_masks={
                    "none": np.ones(len(group), dtype=bool),
                    "ma126": (group["close"] > group["ma_126"]).to_numpy(dtype=bool),
                    "stack2": (group["ma_stack"] >= 2).to_numpy(dtype=bool),
                    "stack3": (group["ma_stack"] >= 3).to_numpy(dtype=bool),
                    "ts3": (group["time_series"] >= 3).to_numpy(dtype=bool),
                    "ts4": (group["time_series"] >= 4).to_numpy(dtype=bool),
                },
            )
        )
    return output


def market_states(close_price: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    daily = close_price.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    curve = (1.0 + daily).cumprod()
    ma21 = curve.rolling(21).mean()
    ma63 = curve.rolling(63).mean()
    ma126 = curve.rolling(126).mean()
    ret21 = curve.pct_change(21)
    ret63 = curve.pct_change(63)
    dates = curve.index
    return {
        "none": set(dates),
        "ma21": set(dates[curve > ma21]),
        "ma63": set(dates[curve > ma63]),
        "ma126": set(dates[curve > ma126]),
        "ma21_63": set(dates[(curve > ma21) & (ma21 > ma63)]),
        "ret21_pos": set(dates[ret21 > 0]),
        "ret63_pos": set(dates[ret63 > 0]),
        "ret21_63_pos": set(dates[(ret21 > 0) & (ret63 > 0)]),
        "ma21_ret21": set(dates[(curve > ma21) & (ret21 > 0)]),
        "ma63_ret63": set(dates[(curve > ma63) & (ret63 > 0)]),
        "risk_on": set(dates[(curve > ma21) & (ma21 > ma63) & (ret21 > 0)]),
    }


def parse_top_n_values(value: str) -> list[int]:
    if str(value).lower() == "all":
        return [1, 2, 3, 5]
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--top-n must be positive or all")
    return [parsed]


def specs_for_frequency(frequency: str, top_n_values: list[int], formula_set: str) -> list[Spec]:
    specs: list[Spec] = []
    frequencies = ["W", "10D", "20D", "M"] if frequency == "all" else [frequency]
    for freq in frequencies:
        for formula in formulas(formula_set):
            for market_filter in [
                "none",
                "ma21",
                "ma63",
                "ma126",
                "ma21_63",
                "ret21_pos",
                "ret63_pos",
                "ret21_63_pos",
                "ma21_ret21",
                "ma63_ret63",
                "risk_on",
            ]:
                for top_n in top_n_values:
                    for min_amount in [20_000_000, 50_000_000, 100_000_000]:
                        for min_price in [3.0, 5.0, 10.0]:
                            for trend_filter in ["none", "ma126", "stack2", "stack3", "ts3", "ts4"]:
                                specs.append(Spec(freq, formula, market_filter, top_n, min_amount, min_price, trend_filter))
    return specs


def requested_frequencies(frequency: str) -> list[str]:
    return ["W", "10D", "20D", "M"] if frequency == "all" else [frequency]


def scan_candidate(days: list[DayData], spec: Spec, allowed_market_dates: set[pd.Timestamp], round_trip_cost: float) -> dict:
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    weights = np.zeros(len(FEATURES), dtype=float)
    for name, weight in spec.formula.weights.items():
        weights[feature_index[name]] = weight

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    period_returns = []
    active_periods = 0
    for day in days:
        if day.signal_date not in allowed_market_dates:
            period_returns.append(0.0)
            continue
        score = weights @ day.features
        mask = (
            np.isfinite(score)
            & np.isfinite(day.ret)
            & (day.amount21 >= spec.min_amount)
            & (day.close >= spec.min_price)
            & day.trend_masks[spec.trend_filter]
        )
        if int(mask.sum()) < spec.top_n:
            period_returns.append(0.0)
            continue
        positions = np.flatnonzero(mask)
        masked_scores = score[positions]
        top_positions = positions[np.argpartition(masked_scores, -spec.top_n)[-spec.top_n:]]
        net_ret = float(np.nanmean(day.ret[top_positions]) - round_trip_cost)
        active_periods += 1
        equity *= 1.0 + net_ret
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        period_returns.append(net_ret)
    if not period_returns:
        return {}
    elapsed_days = max((days[-1].exit_date - days[0].entry_date).days, 1)
    annual_return = equity ** (365.25 / elapsed_days) - 1.0 if equity > 0 else np.nan
    returns = np.asarray(period_returns, dtype=float)
    return {
        "frequency": spec.frequency,
        "formula": spec.formula.name,
        "weights": spec.formula.weights,
        "market_filter": spec.market_filter,
        "top_n": spec.top_n,
        "min_amount": spec.min_amount,
        "min_price": spec.min_price,
        "trend_filter": spec.trend_filter,
        "start_date": days[0].entry_date.strftime("%Y-%m-%d"),
        "end_date": days[-1].exit_date.strftime("%Y-%m-%d"),
        "total_return": float(equity - 1.0),
        "final_equity": float(equity),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "avg_period_return": float(np.nanmean(returns)),
        "period_return_std": float(np.nanstd(returns)),
        "positive_period_rate": float(np.mean(returns > 0)),
        "periods": int(len(returns)),
        "active_periods": int(active_periods),
    }


def build_candidate_series(
    days: list[DayData],
    spec: Spec,
    allowed_market_dates: set[pd.Timestamp],
    round_trip_cost: float,
) -> CandidateSeries:
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    weights = np.zeros(len(FEATURES), dtype=float)
    for name, weight in spec.formula.weights.items():
        weights[feature_index[name]] = weight

    returns = np.zeros(len(days), dtype=np.float64)
    active = np.zeros(len(days), dtype=bool)
    for index, day in enumerate(days):
        if day.signal_date not in allowed_market_dates:
            continue
        score = weights @ day.features
        mask = (
            np.isfinite(score)
            & np.isfinite(day.ret)
            & (day.amount21 >= spec.min_amount)
            & (day.close >= spec.min_price)
            & day.trend_masks[spec.trend_filter]
        )
        if int(mask.sum()) < spec.top_n:
            continue
        positions = np.flatnonzero(mask)
        masked_scores = score[positions]
        top_positions = positions[np.argpartition(masked_scores, -spec.top_n)[-spec.top_n:]]
        returns[index] = float(np.nanmean(day.ret[top_positions]) - round_trip_cost)
        active[index] = True
    return CandidateSeries(spec=spec, returns=returns, active=active)


def metrics_from_returns(series: CandidateSeries, mask: np.ndarray) -> dict:
    dates = G_SERIES_DATES[series.spec.frequency]
    returns = series.returns[mask]
    active = series.active[mask]
    if len(returns) == 0:
        return {}
    equity_curve = np.cumprod(1.0 + returns)
    equity = float(equity_curve[-1])
    peaks = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.min(equity_curve / peaks - 1.0))
    entry_dates = dates["entry_dates"][mask]
    exit_dates = dates["exit_dates"][mask]
    elapsed_days = max(
        (pd.Timestamp(exit_dates[-1]) - pd.Timestamp(entry_dates[0])).days,
        1,
    )
    annual_return = equity ** (365.25 / elapsed_days) - 1.0 if equity > 0 else np.nan
    return {
        "frequency": series.spec.frequency,
        "formula": series.spec.formula.name,
        "weights": series.spec.formula.weights,
        "market_filter": series.spec.market_filter,
        "top_n": series.spec.top_n,
        "min_amount": series.spec.min_amount,
        "min_price": series.spec.min_price,
        "trend_filter": series.spec.trend_filter,
        "start_date": pd.Timestamp(entry_dates[0]).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(exit_dates[-1]).strftime("%Y-%m-%d"),
        "total_return": float(equity - 1.0),
        "final_equity": equity,
        "annual_return": float(annual_return),
        "max_drawdown": max_drawdown,
        "avg_period_return": float(np.nanmean(returns)),
        "period_return_std": float(np.nanstd(returns)),
        "positive_period_rate": float(np.mean(returns > 0)),
        "periods": int(len(returns)),
        "active_periods": int(np.sum(active)),
    }


def build_candidate_series_for_spec(spec: Spec) -> CandidateSeries | None:
    days = G_DAY_SETS[spec.frequency]
    if not days:
        return None
    return build_candidate_series(days, spec, G_MARKET[spec.market_filter], G_ROUND_TRIP_COST)


def evaluate_spec(spec: Spec) -> dict | None:
    metrics = scan_candidate(
        G_DAY_SETS[spec.frequency],
        spec,
        G_MARKET[spec.market_filter],
        G_ROUND_TRIP_COST,
    )
    return metrics or None


def training_score(row: dict) -> float:
    annual = row["annual_return"]
    drawdown = abs(min(row["max_drawdown"], 0.0))
    active_rate = row["active_periods"] / max(row["periods"], 1)
    positive_rate = row["positive_period_rate"]
    period_std = row["period_return_std"]
    if not np.isfinite(annual) or not np.isfinite(period_std):
        return -np.inf
    if G_SCORE_PROFILE == "robust":
        if active_rate < 0.40 or positive_rate < 0.30 or drawdown > 0.35:
            return -np.inf
        return float(annual - 1.25 * drawdown + 0.15 * positive_rate - 0.10 * period_std)
    if G_SCORE_PROFILE == "balanced":
        if active_rate < 0.25 or positive_rate < 0.25 or drawdown > 0.55:
            return -np.inf
        return float(annual - 0.70 * drawdown + 0.10 * positive_rate - 0.05 * period_std)
    if G_SCORE_PROFILE == "aggressive":
        if active_rate < 0.15 or positive_rate < 0.18 or drawdown > 0.85:
            return -np.inf
        return float(annual - 0.20 * drawdown + 0.05 * positive_rate)
    raise ValueError(f"unknown score profile: {G_SCORE_PROFILE}")


def spec_key(row: dict) -> tuple:
    return (
        row["frequency"],
        row["formula"],
        row["market_filter"],
        int(row["top_n"]),
        float(row["min_amount"]),
        float(row["min_price"]),
        row["trend_filter"],
    )


def build_spec_lookup(specs: list[Spec]) -> dict[tuple, Spec]:
    return {
        (
            spec.frequency,
            spec.formula.name,
            spec.market_filter,
            spec.top_n,
            spec.min_amount,
            spec.min_price,
            spec.trend_filter,
        ): spec
        for spec in specs
    }


def filter_days(days: list[DayData], start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[DayData]:
    return [day for day in days if start_date <= day.signal_date <= end_date]


def choose_specs_by_year(
    specs: list[Spec],
    day_sets: dict[str, list[DayData]],
    market: dict[str, set[pd.Timestamp]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_years: int,
    min_train_periods: int,
    workers: int,
    chunksize: int,
    keep_top: int,
) -> tuple[dict[int, Spec], pd.DataFrame]:
    global G_DAY_SETS, G_MARKET, G_SERIES_DATES
    years = range(start_date.year, end_date.year + 1)
    yearly_specs: dict[int, Spec] = {}
    diagnostics: list[dict] = []
    context = mp.get_context("fork")
    G_DAY_SETS = day_sets
    G_MARKET = market
    G_SERIES_DATES = {
        frequency: {
            "signal_dates": np.asarray([day.signal_date.to_datetime64() for day in days], dtype="datetime64[ns]"),
            "entry_dates": np.asarray([day.entry_date.to_datetime64() for day in days], dtype="datetime64[ns]"),
            "exit_dates": np.asarray([day.exit_date.to_datetime64() for day in days], dtype="datetime64[ns]"),
        }
        for frequency, days in day_sets.items()
    }
    candidate_series: list[CandidateSeries] = []
    with ProcessPoolExecutor(max_workers=max(workers, 1), mp_context=context) as executor:
        for series in executor.map(build_candidate_series_for_spec, specs, chunksize=max(chunksize, 1)):
            if series is not None:
                candidate_series.append(series)
    series_spec_lookup = {
        (
            series.spec.frequency,
            series.spec.formula.name,
            series.spec.market_filter,
            series.spec.top_n,
            series.spec.min_amount,
            series.spec.min_price,
            series.spec.trend_filter,
        ): series.spec
        for series in candidate_series
    }

    for year in years:
        test_start = max(start_date, pd.Timestamp(year=year, month=1, day=1))
        test_end = min(end_date, pd.Timestamp(year=year, month=12, day=31))
        train_end = test_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
        results: list[dict] = []
        train_start64 = train_start.to_datetime64()
        train_end64 = train_end.to_datetime64()
        for series in candidate_series:
            signal_dates_ = G_SERIES_DATES[series.spec.frequency]["signal_dates"]
            train_mask = (signal_dates_ >= train_start64) & (signal_dates_ <= train_end64)
            if int(np.sum(train_mask)) < min_train_periods:
                continue
            row = metrics_from_returns(series, train_mask)
            if row:
                row["wf_score"] = training_score(row)
                results.append(row)
        results = [row for row in results if np.isfinite(row["wf_score"])]
        results.sort(key=lambda row: row["wf_score"], reverse=True)
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
        yearly_specs[year] = series_spec_lookup[spec_key(selected)]
        diagnostics.append(
            {
                "year": year,
                "status": "selected",
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "selected_rank": 1,
                **{key: selected[key] for key in selected if key not in {"weights"}},
                "weights": json.dumps(selected["weights"], ensure_ascii=False),
            }
        )
        for rank, row in enumerate(results[:keep_top], start=1):
            diagnostics.append(
                {
                    "year": year,
                    "status": "train_candidate",
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "selected_rank": rank,
                    **{key: row[key] for key in row if key not in {"weights"}},
                    "weights": json.dumps(row["weights"], ensure_ascii=False),
                }
            )
    return yearly_specs, pd.DataFrame(diagnostics)


def run_walkforward(
    day_sets: dict[str, list[DayData]],
    market: dict[str, set[pd.Timestamp]],
    yearly_specs: dict[int, Spec],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float,
    round_trip_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    equity = 1.0
    peak = 1.0
    equity_rows: list[dict] = []
    pick_rows: list[dict] = []
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    for year in range(start_date.year, end_date.year + 1):
        spec = yearly_specs.get(year)
        if spec is None:
            continue
        weights = np.zeros(len(FEATURES), dtype=float)
        for name, weight in spec.formula.weights.items():
            weights[feature_index[name]] = weight
        for day in day_sets[spec.frequency]:
            if not (start_date <= day.signal_date <= end_date and day.signal_date.year == year):
                continue
            if day.signal_date not in market[spec.market_filter]:
                period_ret = 0.0
                selected_positions: np.ndarray = np.asarray([], dtype=int)
            else:
                score = weights @ day.features
                mask = (
                    np.isfinite(score)
                    & np.isfinite(day.ret)
                    & (day.amount21 >= spec.min_amount)
                    & (day.close >= spec.min_price)
                    & day.trend_masks[spec.trend_filter]
                )
                if int(mask.sum()) >= spec.top_n:
                    positions = np.flatnonzero(mask)
                    masked_scores = score[positions]
                    selected_positions = positions[np.argpartition(masked_scores, -spec.top_n)[-spec.top_n:]]
                    selected_positions = selected_positions[np.argsort(score[selected_positions])[::-1]]
                    period_ret = float(np.nanmean(day.ret[selected_positions]) - round_trip_cost)
                else:
                    period_ret = 0.0
                    selected_positions = np.asarray([], dtype=int)
            equity *= 1.0 + period_ret
            peak = max(peak, equity)
            equity_rows.append(
                {
                    "signal_date": day.signal_date.strftime("%Y-%m-%d"),
                    "entry_date": day.entry_date.strftime("%Y-%m-%d"),
                    "exit_date": day.exit_date.strftime("%Y-%m-%d"),
                    "year": year,
                    "equity": float(equity * initial_cash),
                    "period_return": period_ret,
                    "drawdown": float(equity / peak - 1.0),
                    "formula": spec.formula.name,
                    "market_filter": spec.market_filter,
                    "top_n": spec.top_n,
                    "min_amount": spec.min_amount,
                    "min_price": spec.min_price,
                    "trend_filter": spec.trend_filter,
                    "positions": int(len(selected_positions)),
                }
            )
            score = weights @ day.features
            for position in selected_positions:
                pick_rows.append(
                    {
                        "signal_date": day.signal_date.strftime("%Y-%m-%d"),
                        "entry_date": day.entry_date.strftime("%Y-%m-%d"),
                        "exit_date": day.exit_date.strftime("%Y-%m-%d"),
                        "symbol": day.symbols[position],
                        "name": day.names[position],
                        "signal_close": float(day.signal_close[position]),
                        "period_return": float(day.ret[position] - round_trip_cost),
                        "score": float(score[position]),
                        "formula": spec.formula.name,
                    }
                )
    equity_df = pd.DataFrame(equity_rows)
    picks_df = pd.DataFrame(pick_rows)
    if equity_df.empty:
        raise RuntimeError("walk-forward produced no equity rows")
    total_return = equity_df["equity"].iloc[-1] / initial_cash - 1.0
    first_date = pd.Timestamp(equity_df["entry_date"].iloc[0])
    last_date = pd.Timestamp(equity_df["exit_date"].iloc[-1])
    days = max((last_date - first_date).days, 1)
    annual_return = (1.0 + total_return) ** (365.25 / days) - 1.0 if total_return > -1 else np.nan
    returns = equity_df["period_return"].to_numpy(dtype=float)
    max_drawdown = float(equity_df["drawdown"].min())
    active_rate = float((equity_df["positions"] > 0).mean())
    metrics = {
        "initial_cash": initial_cash,
        "start_date": first_date.strftime("%Y-%m-%d"),
        "end_date": last_date.strftime("%Y-%m-%d"),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": max_drawdown,
        "periods": int(len(equity_df)),
        "active_period_rate": active_rate,
        "avg_period_return": float(np.mean(returns)),
        "period_return_std": float(np.std(returns)),
        "positive_period_rate": float(np.mean(returns > 0)),
        "trade_pick_count": int(len(picks_df)),
    }
    return equity_df, picks_df, metrics


def main() -> int:
    global G_DAY_SETS, G_MARKET, G_ROUND_TRIP_COST, G_SCORE_PROFILE
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        db_end = pd.Timestamp(
            conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()[0]
        )
    start_date = parse_date(args.start_date)
    end_date = min(parse_date(args.end_date), db_end) if args.end_date else db_end
    G_ROUND_TRIP_COST = args.round_trip_cost
    G_SCORE_PROFILE = args.score_profile

    research_start = start_date - pd.DateOffset(years=args.train_years) + pd.Timedelta(days=1)
    coverage_start = research_start - pd.Timedelta(days=460)
    with sqlite3.connect(args.db) as conn:
        factor_adjust_coverage = adjustment_coverage(
            conn,
            coverage_start,
            end_date,
            args.board_scope,
            args.factor_adjust,
        )
    print(
        f"loading {research_start.date()} -> {end_date.date()} "
        f"(walk-forward test starts {start_date.date()})",
        flush=True,
    )
    df = load_or_build_factors(
        args.db,
        args.cache_dir,
        research_start,
        end_date,
        use_cache=not args.no_cache,
        board_scope=args.board_scope,
        factor_adjust=args.factor_adjust,
        allow_factor_fallback=not args.strict_factor_adjust,
    )
    all_dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
    raw_open_price = df.pivot(index="trade_date", columns="symbol", values="raw_open").sort_index()
    raw_close_price = df.pivot(index="trade_date", columns="symbol", values="raw_close").sort_index()
    factor_close_price = df.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    market = market_states(factor_close_price)
    frequencies = requested_frequencies(args.frequency)
    day_sets = {
        frequency: build_day_data(
            df,
            raw_open_price,
            raw_close_price,
            signal_dates(all_dates, research_start, end_date, frequency),
            all_dates,
            end_date,
        )
        for frequency in frequencies
    }
    print({key: len(value) for key, value in day_sets.items()}, flush=True)

    top_n_values = parse_top_n_values(args.top_n)
    specs = specs_for_frequency(args.frequency, top_n_values, args.formula_set)
    print(
        f"walk-forward selecting among {len(specs)} specs, frequency={args.frequency}, "
        f"top_n={top_n_values}, workers={args.workers}",
        flush=True,
    )
    yearly_specs, diagnostics = choose_specs_by_year(
        specs=specs,
        day_sets=day_sets,
        market=market,
        start_date=start_date,
        end_date=end_date,
        train_years=args.train_years,
        min_train_periods=args.min_train_periods,
        workers=args.workers,
        chunksize=args.chunksize,
        keep_top=args.keep_top,
    )
    equity, picks, metrics = run_walkforward(
        day_sets,
        market,
        yearly_specs,
        start_date,
        end_date,
        args.initial_cash,
        args.round_trip_cost,
    )
    metrics["config"] = {
        "strategy": "walkforward_no_lookahead_momentum_breakout",
        "frequency": args.frequency,
        "top_n": top_n_values,
        "train_years": args.train_years,
        "min_train_periods": args.min_train_periods,
        "round_trip_cost": args.round_trip_cost,
        "score_profile": args.score_profile,
        "formula_set": args.formula_set,
        "board_scope": args.board_scope,
        "factor_adjust": args.factor_adjust,
        "factor_adjust_used": sorted(str(value) for value in df["factor_adjust_used"].dropna().unique()),
        "factor_adjust_coverage": factor_adjust_coverage,
        "factor_adjust_fallback": not args.strict_factor_adjust,
        "cache_dir": str(args.cache_dir),
        "cache_enabled": not args.no_cache,
        "selection_rule": "each calendar year uses only prior completed training years",
        "execution_rule": "signal close factors; buy next trading day open; sell next signal's next trading day open",
        "bias_controls": [
            "main-board stock universe by symbol prefix" if args.board_scope == "main" else "all loaded A-share symbols",
            "no current-name ST/delist filtering",
            "all rolling features use signal-date and earlier bars",
            "yearly parameter selection does not use the test year",
            "raw prices for execution; configured adjusted prices for factors with optional raw fallback",
        ],
    }
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    top_tag = "top" + "-".join(str(value) for value in top_n_values)
    prefix = args.output_dir / (
        f"walkforward_no_lookahead_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        f"_{args.frequency}_train{args.train_years}y_{top_tag}_{args.score_profile}_{args.formula_set}"
        f"_{args.board_scope}_{args.factor_adjust}"
    )
    equity.to_csv(prefix.with_suffix(".equity.csv"), index=False)
    picks.to_csv(prefix.with_suffix(".picks.csv"), index=False)
    diagnostics.to_csv(prefix.with_suffix(".diagnostics.csv"), index=False)
    metrics_path = prefix.with_suffix(".metrics.json")
    manifest_path = prefix.with_suffix(".manifest.json")
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    write_manifest(
        manifest_path,
        collect_manifest(
            args.db,
            sys.argv,
            [
                Path(__file__),
                Path(__file__).with_name("quant_universe.py"),
                Path(__file__).with_name("run_manifest.py"),
            ],
            {
                "equity": prefix.with_suffix(".equity.csv"),
                "picks": prefix.with_suffix(".picks.csv"),
                "diagnostics": prefix.with_suffix(".diagnostics.csv"),
                "metrics": metrics_path,
                "manifest": manifest_path,
            },
            {
                "strategy": metrics["config"]["strategy"],
                "start_date": metrics["start_date"],
                "end_date": metrics["end_date"],
                "factor_adjust": args.factor_adjust,
                "strict_factor_adjust": args.strict_factor_adjust,
            },
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

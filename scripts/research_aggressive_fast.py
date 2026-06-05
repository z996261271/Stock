#!/usr/bin/env python3
"""Fast in-sample mining for aggressive A-share daily-bar strategies."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_aggressive_strategies import add_research_factors, load_data


FEATURES = [
    "ret240_r",
    "ret120_r",
    "ret60_r",
    "ret40_r",
    "ret20_r",
    "ret10_r",
    "ret5_r",
    "rev20_r",
    "rev10_r",
    "rev5_r",
    "near240_r",
    "near120_r",
    "near60_r",
    "lowvol60_r",
    "lowvol20_r",
    "highvol60_r",
    "liq20_r",
    "lowliq20_r",
    "surge5_r",
    "surge20_r",
    "maxret20_r",
    "lowmax20_r",
    "trend_r",
    "stack_r",
]


@dataclass(frozen=True)
class Formula:
    name: str
    weights: dict[str, float]


@dataclass
class DayData:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbols: np.ndarray
    names: np.ndarray
    signal_close: np.ndarray
    features: np.ndarray
    amount20: np.ndarray
    close: np.ndarray
    ret: np.ndarray
    trend_masks: dict[str, np.ndarray]


G_DAY_SETS: dict[str, list[DayData]] = {}
G_MARKET: dict[str, set[pd.Timestamp]] = {}
G_ROUND_TRIP_COST = 0.0011


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast aggressive strategy mining.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--start-date", default="2021-02-01")
    parser.add_argument("--end-date")
    parser.add_argument("--target-return", type=float, default=10.0)
    parser.add_argument(
        "--target-annual-return",
        type=float,
        help="stop when annualized return reaches this value, e.g. 0.4 for 40%",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument("--keep-top", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--chunksize", type=int, default=64)
    return parser.parse_args()


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value.replace("-", ""))


def formulas() -> list[Formula]:
    return [
        Formula("balanced_attack", {"ret120_r": 0.55, "lowvol60_r": 0.25, "liq20_r": 0.10, "near240_r": 0.10}),
        Formula("pure_long_mom", {"ret120_r": 0.70, "ret60_r": 0.20, "liq20_r": 0.10}),
        Formula("late_stage_leader", {"ret240_r": 0.35, "ret120_r": 0.25, "near240_r": 0.25, "liq20_r": 0.15}),
        Formula("breakout_240", {"near240_r": 0.45, "ret60_r": 0.25, "ret20_r": 0.20, "liq20_r": 0.10}),
        Formula("breakout_surge", {"near120_r": 0.30, "ret20_r": 0.25, "surge5_r": 0.25, "liq20_r": 0.20}),
        Formula("trend_quality", {"trend_r": 0.35, "ret60_r": 0.30, "near120_r": 0.20, "liq20_r": 0.15}),
        Formula("strong_pullback", {"ret120_r": 0.35, "ret60_r": 0.25, "rev10_r": 0.25, "liq20_r": 0.15}),
        Formula("hot_reversal", {"ret60_r": 0.30, "ret20_r": 0.25, "rev5_r": 0.25, "surge5_r": 0.20}),
        Formula("limit_chaser", {"ret20_r": 0.30, "maxret20_r": 0.30, "surge5_r": 0.25, "liq20_r": 0.15}),
        Formula("high_beta_mom", {"ret120_r": 0.35, "ret60_r": 0.25, "highvol60_r": 0.25, "liq20_r": 0.15}),
        Formula("low_float_proxy", {"ret60_r": 0.30, "ret20_r": 0.25, "lowliq20_r": 0.25, "near120_r": 0.20}),
        Formula("steady_breakout", {"near240_r": 0.30, "ret120_r": 0.25, "lowmax20_r": 0.20, "lowvol60_r": 0.15, "liq20_r": 0.10}),
        Formula("short_mom", {"ret20_r": 0.45, "ret10_r": 0.25, "surge5_r": 0.20, "liq20_r": 0.10}),
        Formula("one_month_reversal", {"rev20_r": 0.45, "ret120_r": 0.30, "lowvol20_r": 0.15, "liq20_r": 0.10}),
        Formula("new_high_low_vol", {"near120_r": 0.35, "lowvol20_r": 0.25, "ret60_r": 0.25, "liq20_r": 0.15}),
        Formula("micro_squeeze", {"ret20_r": 0.30, "rev5_r": 0.25, "surge5_r": 0.25, "lowliq20_r": 0.20}),
        Formula("panic_rebound", {"rev20_r": 0.35, "rev10_r": 0.25, "surge5_r": 0.20, "ret60_r": 0.20}),
        Formula("volume_breakout", {"surge5_r": 0.35, "near60_r": 0.25, "ret20_r": 0.25, "liq20_r": 0.15}),
    ]


def get_signal_dates(all_dates: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> list[pd.Timestamp]:
    dates = pd.Series(all_dates[(all_dates >= start_date) & (all_dates <= end_date)])
    if frequency == "M":
        return list(dates.groupby(dates.dt.to_period("M")).max())
    if frequency == "20D":
        return list(dates.iloc[::20])
    if frequency == "10D":
        return list(dates.iloc[::10])
    if frequency == "W":
        return list(dates.groupby(dates.dt.to_period("W-FRI")).max())
    raise ValueError(frequency)


def add_rank_columns(signal_df: pd.DataFrame) -> pd.DataFrame:
    signal_df = signal_df.copy()
    grouped = signal_df.groupby("trade_date", group_keys=False)
    rank_map = {
        "ret240_r": ("ret_240", True),
        "ret120_r": ("ret_120", True),
        "ret60_r": ("ret_60", True),
        "ret40_r": ("ret_40", True),
        "ret20_r": ("ret_20", True),
        "ret10_r": ("ret_10", True),
        "ret5_r": ("ret_5", True),
        "rev20_r": ("ret_20", False),
        "rev10_r": ("ret_10", False),
        "rev5_r": ("ret_5", False),
        "near240_r": ("near_high_240", True),
        "near120_r": ("near_high_120", True),
        "near60_r": ("near_high_60", True),
        "lowvol60_r": ("vol_60", False),
        "lowvol20_r": ("vol_20", False),
        "highvol60_r": ("vol_60", True),
        "liq20_r": ("amount_ma_20", True),
        "lowliq20_r": ("amount_ma_20", False),
        "surge5_r": ("amount_surge_5_20", True),
        "surge20_r": ("amount_surge_20_60", True),
        "maxret20_r": ("max_ret_20", True),
        "lowmax20_r": ("max_ret_20", False),
        "trend_r": ("trend_quality", True),
        "stack_r": ("ma_stack", True),
    }
    for rank_name, (column, ascending) in rank_map.items():
        signal_df[rank_name] = grouped[column].rank(pct=True, ascending=ascending)
    return signal_df


def build_day_data(
    df: pd.DataFrame,
    open_price: pd.DataFrame,
    close_price: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    all_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> list[DayData]:
    signal_df = df[df["trade_date"].isin(set(signal_dates))].copy()
    signal_df = add_rank_columns(signal_df)
    feature_columns = FEATURES
    output = []
    for index, signal_date in enumerate(signal_dates):
        future = all_dates[all_dates > signal_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        if index + 1 < len(signal_dates):
            next_future = all_dates[all_dates > signal_dates[index + 1]]
            exit_date = next_future[0] if len(next_future) else all_dates[-1]
            exit_prices = open_price.loc[exit_date]
        else:
            exit_date = min(end_date, all_dates[-1])
            exit_prices = close_price.loc[exit_date]
        group = signal_df[signal_df["trade_date"] == signal_date].copy()
        if group.empty:
            continue
        symbols = group["symbol"].astype(str).to_numpy()
        entry = open_price.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        exit_ = exit_prices.reindex(symbols).to_numpy(dtype=float)
        period_ret = exit_ / entry - 1.0
        features = group[feature_columns].to_numpy(dtype=float).T
        close = group["close"].to_numpy(dtype=float)
        amount20 = group["amount_ma_20"].to_numpy(dtype=float)
        masks = {
            "none": np.ones(len(group), dtype=bool),
            "ma20": (group["close"] > group["ma_20"]).to_numpy(dtype=bool),
            "ma60": (group["close"] > group["ma_60"]).to_numpy(dtype=bool),
            "ma120": (group["close"] > group["ma_120"]).to_numpy(dtype=bool),
            "stack": (group["ma_stack"] >= 3).to_numpy(dtype=bool),
        }
        output.append(
            DayData(
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                symbols=symbols,
                names=group["name"].astype(str).to_numpy(),
                signal_close=close,
                features=features,
                amount20=amount20,
                close=close,
                ret=period_ret,
                trend_masks=masks,
            )
        )
    return output


def market_states(close_price: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    daily = close_price.pct_change().mean(axis=1).fillna(0)
    curve = (1.0 + daily).cumprod()
    ma20 = curve.rolling(20).mean()
    ma60 = curve.rolling(60).mean()
    ma120 = curve.rolling(120).mean()
    ret20 = curve.pct_change(20)
    ret60 = curve.pct_change(60)
    dates = curve.index
    return {
        "none": set(dates),
        "ma20": set(dates[curve > ma20]),
        "ma60": set(dates[curve > ma60]),
        "ma120": set(dates[curve > ma120]),
        "ma20_60": set(dates[(curve > ma20) & (ma20 > ma60)]),
        "ret20_pos": set(dates[ret20 > 0]),
        "ret60_pos": set(dates[ret60 > 0]),
    }


def scan_candidate(
    days: list[DayData],
    formula: Formula,
    top_n: int,
    min_amount: float,
    min_price: float,
    trend_filter: str,
    allowed_market_dates: set[pd.Timestamp],
    round_trip_cost: float,
    collect_rows: bool = False,
) -> tuple[dict, list[dict]]:
    feature_index = {name: index for index, name in enumerate(FEATURES)}
    weights = np.zeros(len(FEATURES), dtype=float)
    for name, weight in formula.weights.items():
        weights[feature_index[name]] = weight

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    period_returns = []
    selected_rows = []
    for day in days:
        if day.signal_date not in allowed_market_dates:
            period_returns.append(0.0)
            continue
        score = weights @ day.features
        mask = (
            np.isfinite(score)
            & np.isfinite(day.ret)
            & (day.amount20 >= min_amount)
            & (day.close >= min_price)
            & day.trend_masks[trend_filter]
        )
        available = int(mask.sum())
        if available < top_n:
            period_returns.append(0.0)
            continue
        masked_positions = np.flatnonzero(mask)
        masked_scores = score[masked_positions]
        top_positions = masked_positions[np.argpartition(masked_scores, -top_n)[-top_n:]]
        top_positions = top_positions[np.argsort(score[top_positions])[::-1]]
        gross_ret = float(np.nanmean(day.ret[top_positions]))
        net_ret = gross_ret - round_trip_cost
        equity *= 1.0 + net_ret
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        period_returns.append(net_ret)
        if collect_rows:
            for position in top_positions:
                selected_rows.append(
                    {
                        "signal_date": day.signal_date.strftime("%Y-%m-%d"),
                        "entry_date": day.entry_date.strftime("%Y-%m-%d"),
                        "exit_date": day.exit_date.strftime("%Y-%m-%d"),
                        "symbol": day.symbols[position],
                        "name": day.names[position],
                        "signal_close": float(day.signal_close[position]),
                        "period_return": float(day.ret[position] - round_trip_cost),
                        "score": float(score[position]),
                    }
                )
    if not period_returns:
        return {}, []
    returns = np.asarray(period_returns, dtype=float)
    total_return = equity - 1.0
    first_date = days[0].entry_date
    last_date = days[-1].exit_date
    elapsed_days = max((last_date - first_date).days, 1)
    annual_return = equity ** (365.25 / elapsed_days) - 1.0 if equity > 0 else np.nan
    annualized_periods = 252 / 20
    annual_vol = float(np.nanstd(returns) * np.sqrt(annualized_periods))
    return {
        "start_date": first_date.strftime("%Y-%m-%d"),
        "end_date": last_date.strftime("%Y-%m-%d"),
        "total_return": float(total_return),
        "final_equity": float(equity),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "avg_period_return": float(np.nanmean(returns)),
        "period_return_std": float(np.nanstd(returns)),
        "positive_period_rate": float(np.mean(returns > 0)),
        "periods": int(len(returns)),
        "annual_volatility_like": annual_vol,
    }, selected_rows


def candidate_specs() -> list[tuple[str, Formula, str, int, float, float, str]]:
    specs = []
    for frequency in ["M", "20D", "10D", "W"]:
        for formula in formulas():
            for market_filter in ["none", "ma20", "ma60", "ma120", "ma20_60", "ret20_pos", "ret60_pos"]:
                for top_n in [1, 2, 3, 5]:
                    for min_amount in [20_000_000, 50_000_000, 100_000_000]:
                        for min_price in [3.0, 5.0, 10.0, 20.0]:
                            for trend_filter in ["none", "ma20", "ma60", "ma120", "stack"]:
                                specs.append((frequency, formula, market_filter, top_n, min_amount, min_price, trend_filter))
    return specs


def evaluate_spec(
    spec: tuple[str, Formula, str, int, float, float, str],
) -> dict | None:
    frequency, formula, market_filter, top_n, min_amount, min_price, trend_filter = spec
    metrics, _ = scan_candidate(
        G_DAY_SETS[frequency],
        formula,
        top_n,
        min_amount,
        min_price,
        trend_filter,
        G_MARKET[market_filter],
        G_ROUND_TRIP_COST,
        collect_rows=False,
    )
    if not metrics:
        return None
    return {
        "frequency": frequency,
        "formula": formula.name,
        "weights": formula.weights,
        "market_filter": market_filter,
        "top_n": top_n,
        "min_amount": min_amount,
        "min_price": min_price,
        "trend_filter": trend_filter,
        **metrics,
    }


def selected_rows_for_best(
    best: dict,
    day_sets: dict[str, list[DayData]],
    market: dict[str, set[pd.Timestamp]],
    round_trip_cost: float,
) -> list[dict]:
    formula_by_name = {formula.name: formula for formula in formulas()}
    metrics, rows = scan_candidate(
        day_sets[best["frequency"]],
        formula_by_name[best["formula"]],
        int(best["top_n"]),
        float(best["min_amount"]),
        float(best["min_price"]),
        str(best["trend_filter"]),
        market[str(best["market_filter"])],
        round_trip_cost,
        collect_rows=True,
    )
    if metrics:
        best.update(metrics)
    return rows


def main() -> int:
    global G_DAY_SETS, G_MARKET, G_ROUND_TRIP_COST
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        db_end = pd.Timestamp(
            conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()[0]
        )
    start_date = parse_date(args.start_date)
    end_date = min(parse_date(args.end_date), db_end) if args.end_date else db_end

    print(f"loading {start_date.date()} -> {end_date.date()}", flush=True)
    df = add_research_factors(load_data(args.db, start_date, end_date))
    all_dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
    open_price = df.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    close_price = df.pivot(index="trade_date", columns="symbol", values="close").sort_index().ffill()
    market = market_states(close_price)

    day_sets = {
        frequency: build_day_data(
            df,
            open_price,
            close_price,
            get_signal_dates(all_dates, start_date, end_date, frequency),
            all_dates,
            end_date,
        )
        for frequency in ["M", "20D", "10D", "W"]
    }
    print({key: len(value) for key, value in day_sets.items()}, flush=True)
    G_DAY_SETS = day_sets
    G_MARKET = market
    G_ROUND_TRIP_COST = args.round_trip_cost

    results = []
    best_key = None
    scanned = 0
    specs = candidate_specs()
    print(f"candidate specs={len(specs)} {args.executor}_workers={args.workers}", flush=True)

    if args.executor == "process":
        context = mp.get_context("fork")
        executor_factory = lambda: ProcessPoolExecutor(  # noqa: E731 - compact factory for two executors.
            max_workers=max(args.workers, 1),
            mp_context=context,
        )
    else:
        executor_factory = lambda: ThreadPoolExecutor(max_workers=max(args.workers, 1))  # noqa: E731

    with executor_factory() as executor:
        for row in executor.map(evaluate_spec, specs, chunksize=max(args.chunksize, 1)):
            scanned += 1
            if row is None:
                continue
            results.append(row)
            if best_key is None or row["annual_return"] > best_key["annual_return"]:
                best_key = row
            if scanned % 1000 == 0 and best_key:
                print(
                    f"scanned {scanned}/{len(specs)}, best ret={best_key['total_return']:.2%} "
                    f"annual={best_key['annual_return']:.2%} {best_key['formula']} {best_key['frequency']}",
                    flush=True,
                )
            if best_key and is_target_hit(best_key, args):
                print(
                    f"target hit after {scanned}: "
                    f"ret={best_key['total_return']:.2%} annual={best_key['annual_return']:.2%}",
                    flush=True,
                )
                break

    if not results or best_key is None:
        raise RuntimeError("no strategy result")
    results = sorted(results, key=lambda item: item["annual_return"], reverse=True)
    best_key = results[0]
    best_rows = selected_rows_for_best(best_key, day_sets, market, args.round_trip_cost)
    best_key["warning"] = "in-sample mined result; high overfitting risk"
    best_key["start_date"] = start_date.strftime("%Y-%m-%d")
    best_key["end_date"] = end_date.strftime("%Y-%m-%d")
    best_key["generated_at"] = datetime.now().isoformat(timespec="seconds")
    best_key["scanned"] = scanned
    best_key["target_return"] = args.target_return
    best_key["target_annual_return"] = args.target_annual_return

    prefix = args.output_dir / (
        f"aggressive_fast_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        f"_{best_key['formula']}_{best_key['frequency']}_top{best_key['top_n']}"
    )
    with prefix.with_suffix(".metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(best_key, fh, ensure_ascii=False, indent=2)
    pd.DataFrame(results[: args.keep_top]).to_csv(prefix.with_suffix(".scan.csv"), index=False)
    pd.DataFrame(best_rows).to_csv(prefix.with_suffix(".picks.csv"), index=False)
    print(json.dumps(best_key, ensure_ascii=False, indent=2), flush=True)
    return 0


def is_target_hit(row: dict, args: argparse.Namespace) -> bool:
    if args.target_annual_return is not None:
        return row["annual_return"] >= args.target_annual_return
    return row["total_return"] >= args.target_return


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Mine aggressive long-only daily-bar strategies on the local A-share database.

This is a research tool, not a production strategy generator. It deliberately
searches many in-sample combinations, then replays the best candidate with the
same next-open execution engine used by the regular backtest.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_momentum_lowvol import BacktestConfig, threshold_tag


@dataclass(frozen=True)
class Formula:
    name: str
    weights: dict[str, float]


@dataclass(frozen=True)
class Candidate:
    formula: Formula
    frequency: str
    top_n: int
    min_amount: float
    min_price: float
    trend_filter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan aggressive A-share daily-bar strategies.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--start-date", default="2021-02-01")
    parser.add_argument("--end-date")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--buy-cost", type=float, default=0.0003)
    parser.add_argument("--sell-cost", type=float, default=0.0008)
    parser.add_argument("--target-return", type=float, default=10.0)
    parser.add_argument("--keep-top", type=int, default=30)
    return parser.parse_args()


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value.replace("-", ""))


def load_data(db: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    lookback_start = start_date - pd.Timedelta(days=420)
    query = """
        select d.symbol, s.name, d.trade_date, d.open, d.high, d.low,
               d.close, d.amount, d.turnover
        from daily_bars d
        join symbols s on s.symbol = d.symbol
        where d.adjust = 'raw'
          and d.trade_date >= ?
          and d.trade_date <= ?
          and d.open is not null
          and d.high is not null
          and d.low is not null
          and d.close is not null
          and d.amount is not null
          and s.name not like '%ST%'
          and s.name not like '%退%'
    """
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(
            query,
            conn,
            params=(lookback_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )
    if df.empty:
        raise RuntimeError("no bars loaded")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"])
    for column in ["open", "high", "low", "close", "amount", "turnover"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "amount"])


def add_research_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby("symbol", group_keys=False)
    df["ret_1"] = group["close"].pct_change(1)
    for window in [2, 3, 5, 10, 20, 40, 60, 120, 180, 240]:
        df[f"ret_{window}"] = group["close"].pct_change(window)
    for window in [5, 10, 20, 60, 120, 240]:
        df[f"ma_{window}"] = group["close"].rolling(window).mean().reset_index(level=0, drop=True)
    for window in [20, 60, 120, 240]:
        df[f"high_{window}"] = group["close"].rolling(window).max().reset_index(level=0, drop=True)
        df[f"near_high_{window}"] = df["close"] / df[f"high_{window}"]
        df[f"vol_{window}"] = group["ret_1"].rolling(window).std().reset_index(level=0, drop=True)
    df["max_ret_5"] = group["ret_1"].rolling(5).max().reset_index(level=0, drop=True)
    df["max_ret_20"] = group["ret_1"].rolling(20).max().reset_index(level=0, drop=True)
    df["min_ret_20"] = group["ret_1"].rolling(20).min().reset_index(level=0, drop=True)
    for window in [5, 20, 60]:
        df[f"amount_ma_{window}"] = group["amount"].rolling(window).mean().reset_index(level=0, drop=True)
    df["amount_surge_5_20"] = df["amount_ma_5"] / df["amount_ma_20"]
    df["amount_surge_20_60"] = df["amount_ma_20"] / df["amount_ma_60"]
    df["turnover_ma_20"] = group["turnover"].rolling(20).mean().reset_index(level=0, drop=True)
    df["trend_quality"] = df["ma_60"] / df["ma_120"] - 1.0
    df["ma_stack"] = (
        (df["close"] > df["ma_20"]).astype(float)
        + (df["ma_20"] > df["ma_60"]).astype(float)
        + (df["ma_60"] > df["ma_120"]).astype(float)
    )
    df["drawdown_60"] = df["close"] / df["high_60"] - 1.0
    df["drawdown_120"] = df["close"] / df["high_120"] - 1.0
    return df


def formulas() -> list[Formula]:
    return [
        Formula("balanced_attack", {"ret120_r": 0.55, "lowvol60_r": 0.25, "liq20_r": 0.10, "near240_r": 0.10}),
        Formula("pure_long_mom", {"ret120_r": 0.70, "ret60_r": 0.20, "liq20_r": 0.10}),
        Formula("breakout_240", {"near240_r": 0.45, "ret60_r": 0.25, "ret20_r": 0.20, "liq20_r": 0.10}),
        Formula("breakout_surge", {"near120_r": 0.30, "ret20_r": 0.25, "surge5_r": 0.25, "liq20_r": 0.20}),
        Formula("trend_quality", {"trend_r": 0.35, "ret60_r": 0.30, "near120_r": 0.20, "liq20_r": 0.15}),
        Formula("strong_pullback", {"ret120_r": 0.35, "ret60_r": 0.25, "rev10_r": 0.25, "liq20_r": 0.15}),
        Formula("hot_reversal", {"ret60_r": 0.30, "ret20_r": 0.25, "rev5_r": 0.25, "surge5_r": 0.20}),
        Formula("limit_chaser", {"ret20_r": 0.30, "maxret20_r": 0.30, "surge5_r": 0.25, "liq20_r": 0.15}),
        Formula("high_beta_mom", {"ret120_r": 0.35, "ret60_r": 0.25, "highvol60_r": 0.25, "liq20_r": 0.15}),
        Formula("low_float_proxy", {"ret60_r": 0.30, "ret20_r": 0.25, "lowliq20_r": 0.25, "near120_r": 0.20}),
        Formula("steady_breakout", {"near240_r": 0.30, "ret120_r": 0.25, "lowmax20_r": 0.20, "lowvol60_r": 0.15, "liq20_r": 0.10}),
        Formula("late_stage_leader", {"ret240_r": 0.35, "ret120_r": 0.25, "near240_r": 0.25, "liq20_r": 0.15}),
        Formula("short_mom", {"ret20_r": 0.45, "ret10_r": 0.25, "surge5_r": 0.20, "liq20_r": 0.10}),
        Formula("one_month_reversal", {"rev20_r": 0.45, "ret120_r": 0.30, "lowvol20_r": 0.15, "liq20_r": 0.10}),
        Formula("new_high_low_vol", {"near120_r": 0.35, "lowvol20_r": 0.25, "ret60_r": 0.25, "liq20_r": 0.15}),
    ]


def signal_dates(all_dates: pd.DatetimeIndex, start_date: pd.Timestamp, end_date: pd.Timestamp, frequency: str) -> list[pd.Timestamp]:
    dates = pd.Series(all_dates[(all_dates >= start_date) & (all_dates <= end_date)])
    if frequency == "M":
        return list(dates.groupby(dates.dt.to_period("M")).max())
    if frequency == "W":
        return list(dates.groupby(dates.dt.to_period("W-FRI")).max())
    if frequency == "10D":
        return list(dates.iloc[::10])
    if frequency == "20D":
        return list(dates.iloc[::20])
    raise ValueError(f"unknown frequency: {frequency}")


def build_ranked_signal_frame(df: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    signal_df = df[df["trade_date"].isin(set(dates))].copy()
    rank_sources = {
        "ret240_r": "ret_240",
        "ret120_r": "ret_120",
        "ret60_r": "ret_60",
        "ret40_r": "ret_40",
        "ret20_r": "ret_20",
        "ret10_r": "ret_10",
        "rev20_r": ("ret_20", False),
        "rev10_r": ("ret_10", False),
        "rev5_r": ("ret_5", False),
        "near240_r": "near_high_240",
        "near120_r": "near_high_120",
        "near60_r": "near_high_60",
        "lowvol60_r": ("vol_60", False),
        "lowvol20_r": ("vol_20", False),
        "highvol60_r": "vol_60",
        "liq20_r": "amount_ma_20",
        "lowliq20_r": ("amount_ma_20", False),
        "surge5_r": "amount_surge_5_20",
        "surge20_r": "amount_surge_20_60",
        "maxret20_r": "max_ret_20",
        "lowmax20_r": ("max_ret_20", False),
        "trend_r": "trend_quality",
        "stack_r": "ma_stack",
    }
    grouped = signal_df.groupby("trade_date", group_keys=False)
    for rank_name, source in rank_sources.items():
        ascending = True
        column = source
        if isinstance(source, tuple):
            column, ascending = source
        if ascending:
            signal_df[rank_name] = grouped[column].rank(pct=True)
        else:
            signal_df[rank_name] = grouped[column].rank(pct=True, ascending=False)
    return signal_df


def trend_mask(frame: pd.DataFrame, trend_filter: str) -> pd.Series:
    if trend_filter == "none":
        return pd.Series(True, index=frame.index)
    if trend_filter == "ma20":
        return frame["close"] > frame["ma_20"]
    if trend_filter == "ma60":
        return frame["close"] > frame["ma_60"]
    if trend_filter == "ma120":
        return frame["close"] > frame["ma_120"]
    if trend_filter == "stack":
        return frame["ma_stack"] >= 3
    raise ValueError(f"unknown trend filter: {trend_filter}")


def select_picks(signal_frame: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    required = list(candidate.formula.weights)
    base = signal_frame[
        (signal_frame["close"] >= candidate.min_price)
        & (signal_frame["amount_ma_20"] >= candidate.min_amount)
        & trend_mask(signal_frame, candidate.trend_filter)
    ].dropna(subset=required + ["open", "close", "amount_ma_20"])
    if base.empty:
        return pd.DataFrame()
    base = base.copy()
    base["score"] = sum(base[column] * weight for column, weight in candidate.formula.weights.items())
    picks = (
        base.sort_values(["trade_date", "score"], ascending=[True, False])
        .groupby("trade_date", group_keys=False)
        .head(candidate.top_n)
    )
    return picks.rename(columns={"trade_date": "signal_date", "close": "signal_close"})


def fast_period_backtest(
    picks: pd.DataFrame,
    open_price: pd.DataFrame,
    close_price: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float,
    round_trip_cost: float,
) -> dict:
    if picks.empty:
        return {}
    all_dates = close_price.loc[(close_price.index >= start_date) & (close_price.index <= end_date)].index
    signal_to_symbols = {
        pd.Timestamp(date): group["symbol"].tolist()
        for date, group in picks.groupby("signal_date")
    }
    signal_list = sorted(signal_to_symbols)
    equity = initial_cash
    equity_rows = []
    for index, signal_date in enumerate(signal_list):
        future = all_dates[all_dates > signal_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        if index + 1 < len(signal_list):
            exit_future = all_dates[all_dates > signal_list[index + 1]]
            exit_date = exit_future[0] if len(exit_future) else end_date
            exit_matrix = open_price
        else:
            exit_date = all_dates[-1]
            exit_matrix = close_price
        returns = []
        for symbol in signal_to_symbols[signal_date]:
            if symbol not in open_price.columns or symbol not in exit_matrix.columns:
                continue
            entry = open_price.at[entry_date, symbol] if entry_date in open_price.index else np.nan
            exit_ = exit_matrix.at[exit_date, symbol] if exit_date in exit_matrix.index else np.nan
            if pd.notna(entry) and pd.notna(exit_) and entry > 0:
                returns.append(exit_ / entry - 1.0 - round_trip_cost)
        if not returns:
            continue
        equity *= 1.0 + float(np.mean(returns))
        equity_rows.append({"date": exit_date, "equity": equity})
    if not equity_rows:
        return {}
    curve = pd.DataFrame(equity_rows).drop_duplicates("date", keep="last").sort_values("date")
    curve["ret"] = curve["equity"].pct_change().fillna(0)
    drawdown = curve["equity"] / curve["equity"].cummax() - 1
    total_return = curve["equity"].iloc[-1] / initial_cash - 1
    days = (curve["date"].iloc[-1] - curve["date"].iloc[0]).days
    annual_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 and total_return > -1 else np.nan
    annual_vol = curve["ret"].std(ddof=0) * np.sqrt(252)
    return {
        "start_date": curve["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": curve["date"].iloc[-1].strftime("%Y-%m-%d"),
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "max_drawdown": float(drawdown.min()),
        "periods": int(len(curve)),
    }


def replay_with_daily_engine(
    df: pd.DataFrame,
    picks: pd.DataFrame,
    args: argparse.Namespace,
    candidate: Candidate,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = BacktestConfig(
        db=args.db,
        output_dir=args.output_dir,
        years=0,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        top_n=candidate.top_n,
        min_amount=candidate.min_amount,
        min_price=candidate.min_price,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        initial_cash=args.initial_cash,
        score_mode=candidate.formula.name,
    )
    replay_picks = picks[
        [
            "signal_date",
            "symbol",
            "name",
            "signal_close",
            "score",
            "ret_120",
            "ret_60",
            "ret_20",
            "near_high_240",
            "amount_ma_20",
        ]
    ].copy()
    equity, trades, metrics = run_daily_engine(df, replay_picks, start_date, end_date, config)
    return equity, trades, metrics


def run_daily_engine(
    df: pd.DataFrame,
    picks: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from backtest_momentum_lowvol import run_backtest

    equity, trades, metrics = run_backtest(df, picks, start_date, end_date, config)
    return equity, trades, metrics


def candidate_grid() -> list[Candidate]:
    result = []
    for formula in formulas():
        for frequency in ["M", "20D", "10D", "W"]:
            for top_n in [1, 2, 3, 5]:
                for min_amount in [20_000_000, 50_000_000, 100_000_000]:
                    for min_price in [3.0, 5.0, 10.0, 20.0]:
                        for trend_filter in ["none", "ma20", "ma60", "ma120", "stack"]:
                            result.append(
                                Candidate(
                                    formula=formula,
                                    frequency=frequency,
                                    top_n=top_n,
                                    min_amount=min_amount,
                                    min_price=min_price,
                                    trend_filter=trend_filter,
                                )
                            )
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        db_end = pd.Timestamp(
            conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()[0]
        )
    start_date = parse_date(args.start_date)
    end_date = min(parse_date(args.end_date), db_end) if args.end_date else db_end
    df = add_research_factors(load_data(args.db, start_date, end_date))
    all_dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
    open_price = df.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    close_price = df.pivot(index="trade_date", columns="symbol", values="close").sort_index().ffill()

    frames_by_frequency = {
        frequency: build_ranked_signal_frame(df, signal_dates(all_dates, start_date, end_date, frequency))
        for frequency in ["M", "20D", "10D", "W"]
    }

    results = []
    best_picks = None
    best_candidate = None
    for scanned, candidate in enumerate(candidate_grid(), 1):
        picks = select_picks(frames_by_frequency[candidate.frequency], candidate)
        rough = fast_period_backtest(
            picks=picks,
            open_price=open_price,
            close_price=close_price,
            start_date=start_date,
            end_date=end_date,
            initial_cash=args.initial_cash,
            round_trip_cost=args.buy_cost + args.sell_cost,
        )
        if not rough:
            continue
        result = {
            "formula": candidate.formula.name,
            "frequency": candidate.frequency,
            "top_n": candidate.top_n,
            "min_amount": candidate.min_amount,
            "min_price": candidate.min_price,
            "trend_filter": candidate.trend_filter,
            **rough,
        }
        results.append(result)
        if best_candidate is None or result["total_return"] > results[-2]["total_return"] if len(results) > 1 else True:
            best_candidate = candidate
            best_picks = picks

        if scanned % 500 == 0:
            current_best = max(results, key=lambda row: row["total_return"])
            print(
                f"scanned={scanned} best={current_best['total_return']:.2%} "
                f"{current_best['formula']} {current_best['frequency']} top{current_best['top_n']}"
            )
        if rough["total_return"] >= args.target_return:
            print(f"target hit in rough scan at {rough['total_return']:.2%}")
            best_candidate = candidate
            best_picks = picks
            break

    if not results or best_candidate is None or best_picks is None:
        raise RuntimeError("no candidate produced a result")

    results = sorted(results, key=lambda row: row["total_return"], reverse=True)
    top_results = results[: args.keep_top]
    replay_candidate = best_candidate
    if results[0]["formula"] != best_candidate.formula.name:
        lookup = {
            (c.formula.name, c.frequency, c.top_n, c.min_amount, c.min_price, c.trend_filter): c
            for c in candidate_grid()
        }
        top = results[0]
        replay_candidate = lookup[
            (top["formula"], top["frequency"], top["top_n"], top["min_amount"], top["min_price"], top["trend_filter"])
        ]
        best_picks = select_picks(frames_by_frequency[replay_candidate.frequency], replay_candidate)

    equity, trades, metrics = replay_with_daily_engine(
        df=df,
        picks=best_picks,
        args=args,
        candidate=replay_candidate,
        start_date=start_date,
        end_date=end_date,
    )
    metrics["config"] = {
        "strategy": "aggressive_factor_mining",
        "formula": replay_candidate.formula.name,
        "weights": replay_candidate.formula.weights,
        "frequency": replay_candidate.frequency,
        "top_n": replay_candidate.top_n,
        "min_amount": replay_candidate.min_amount,
        "min_price": replay_candidate.min_price,
        "trend_filter": replay_candidate.trend_filter,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "warning": "in-sample mined result; high overfitting risk",
    }
    metrics["rough_scan_top"] = top_results
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    prefix = args.output_dir / (
        f"aggressive_mined_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        f"_{replay_candidate.formula.name}_{replay_candidate.frequency}"
        f"_top{replay_candidate.top_n}_amt{threshold_tag(replay_candidate.min_amount)}"
        f"_p{threshold_tag(replay_candidate.min_price)}_{replay_candidate.trend_filter}"
    )
    equity.to_csv(prefix.with_suffix(".equity.csv"), index=False)
    trades.to_csv(prefix.with_suffix(".trades.csv"), index=False)
    best_picks.to_csv(prefix.with_suffix(".picks.csv"), index=False)
    pd.DataFrame(top_results).to_csv(prefix.with_suffix(".scan.csv"), index=False)
    with prefix.with_suffix(".metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

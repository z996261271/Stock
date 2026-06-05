#!/usr/bin/env python3
"""Backtest an aggressive sector-momentum + stock-momentum top-N strategy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest_momentum_lowvol import (
    BacktestConfig,
    add_factors,
    compute_equal_weight_benchmark,
    load_data,
    month_end_signal_dates,
    run_backtest,
    threshold_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest aggressive THS industry momentum plus stock momentum."
    )
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-boards", type=int, default=8)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--min-corr", type=float, default=0.25)
    parser.add_argument("--min-amount", type=float, default=50_000_000)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--buy-cost", type=float, default=0.0003)
    parser.add_argument("--sell-cost", type=float, default=0.0008)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument(
        "--score-mode",
        choices=["sector_attack_v1", "sector_breakout_v1", "sector_surge_v1"],
        default="sector_attack_v1",
    )
    return parser.parse_args()


def load_industry_bars(
    db: Path, lookback_start: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        query = """
            select board_code, board_name, trade_date, open, close, amount
            from industry_daily_bars
            where trade_date >= ?
              and trade_date <= ?
              and close is not null
        """
        boards = pd.read_sql_query(
            query,
            conn,
            params=(lookback_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )
    if boards.empty:
        raise RuntimeError(
            "industry_daily_bars is empty; run scripts/fetch_ths_industry_daily.py first"
        )
    boards["trade_date"] = pd.to_datetime(boards["trade_date"])
    for column in ["open", "close", "amount"]:
        boards[column] = pd.to_numeric(boards[column], errors="coerce")
    boards = boards.dropna(subset=["close"])
    return boards.sort_values(["board_code", "trade_date"])


def add_attack_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = add_factors(df)
    group = df.groupby("symbol", group_keys=False)
    df["ret_60"] = group["close"].pct_change(60)
    df["high_252"] = group["close"].rolling(252).max().reset_index(level=0, drop=True)
    df["near_high_252"] = df["close"] / df["high_252"]
    df["amount_ma60"] = group["amount"].rolling(60).mean().reset_index(level=0, drop=True)
    df["volume_surge"] = df["amount_ma20"] / df["amount_ma60"]
    df["ma20"] = group["close"].rolling(20).mean().reset_index(level=0, drop=True)
    df["trend_stack"] = (
        (df["close"] > df["ma20"]).astype(float)
        + (df["ma20"] > df["ma60"]).astype(float)
        + (df["ma60"] > df["ma120"]).astype(float)
    )
    return df


def add_board_factors(boards: pd.DataFrame) -> pd.DataFrame:
    boards = boards.copy()
    group = boards.groupby("board_code", group_keys=False)
    boards["board_ret_1d"] = group["close"].pct_change()
    boards["board_ret_20"] = group["close"].pct_change(20)
    boards["board_ret_60"] = group["close"].pct_change(60)
    boards["board_mom_120_20"] = group["close"].shift(20) / group["close"].shift(120) - 1.0
    boards["board_vol_60"] = group["board_ret_1d"].rolling(60).std().reset_index(level=0, drop=True)
    boards["board_amount_ma20"] = group["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    return boards


def build_signal_table(
    df: pd.DataFrame,
    boards: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    config: BacktestConfig,
    top_boards: int,
    corr_window: int,
    min_corr: float,
    score_mode: str,
) -> pd.DataFrame:
    picks = []
    signal_df = df[df["trade_date"].isin(set(signal_dates))].copy()
    signal_df = signal_df[
        (signal_df["mom_120_20"].notna())
        & (signal_df["ret_60"].notna())
        & (signal_df["near_high_252"].notna())
        & (signal_df["vol_60"].notna())
        & (signal_df["amount_ma20"] >= config.min_amount)
        & (signal_df["volume_surge"].notna())
        & (signal_df["close"] > signal_df["ma120"])
        & (signal_df["close"] > config.min_price)
    ]
    stock_returns = df.pivot(index="trade_date", columns="symbol", values="ret_1d").sort_index()
    board_returns = boards.pivot(index="trade_date", columns="board_code", values="board_ret_1d").sort_index()
    board_name_map = (
        boards[["board_code", "board_name"]].drop_duplicates().set_index("board_code")["board_name"].to_dict()
    )

    for signal_date, group in signal_df.groupby("trade_date"):
        board_slice = boards[boards["trade_date"] == signal_date].copy()
        board_slice = board_slice[
            board_slice["board_ret_60"].notna()
            & board_slice["board_mom_120_20"].notna()
            & board_slice["board_vol_60"].notna()
        ]
        if board_slice.empty:
            continue
        board_slice["board_ret60_rank"] = board_slice["board_ret_60"].rank(pct=True)
        board_slice["board_mom_rank"] = board_slice["board_mom_120_20"].rank(pct=True)
        board_slice["board_ret20_rank"] = board_slice["board_ret_20"].rank(pct=True)
        board_slice["board_liq_rank"] = board_slice["board_amount_ma20"].rank(pct=True)
        board_slice["board_score"] = (
            board_slice["board_ret60_rank"] * 0.45
            + board_slice["board_mom_rank"] * 0.35
            + board_slice["board_ret20_rank"] * 0.10
            + board_slice["board_liq_rank"] * 0.10
        )
        hot_boards = board_slice.nlargest(top_boards, "board_score").copy()
        hot_codes = hot_boards["board_code"].tolist()
        if not hot_codes:
            continue

        stock_window = stock_returns.loc[stock_returns.index < signal_date].tail(corr_window)
        board_window = board_returns.reindex(stock_window.index)[hot_codes]
        if len(stock_window) < corr_window * 0.8 or board_window.empty:
            continue

        corr_frame = pd.DataFrame(index=stock_window.columns)
        for board_code in hot_codes:
            corr_frame[board_code] = stock_window.corrwith(board_window[board_code])
        corr_frame = corr_frame.dropna(how="all")
        if corr_frame.empty:
            continue
        best_board = corr_frame.idxmax(axis=1)
        best_corr = corr_frame.max(axis=1)

        group = group.copy()
        group["board_code"] = group["symbol"].map(best_board)
        group["board_corr"] = group["symbol"].map(best_corr)
        group = group[group["board_corr"] >= min_corr]
        if group.empty:
            continue

        board_scores = hot_boards.set_index("board_code")["board_score"]
        group["board_name"] = group["board_code"].map(board_name_map)
        group["board_score"] = group["board_code"].map(board_scores)
        group["stock_mom_rank"] = group["mom_120_20"].rank(pct=True)
        group["ret60_rank"] = group["ret_60"].rank(pct=True)
        group["near_high_rank"] = group["near_high_252"].rank(pct=True)
        group["liq_rank"] = group["amount_ma20"].rank(pct=True)
        group["surge_rank"] = group["volume_surge"].rank(pct=True)
        group["low_vol_rank"] = (-group["vol_60"]).rank(pct=True)
        group["trend_rank"] = group["trend_stack"].rank(pct=True)
        group["score"] = score_series(group, score_mode)
        selected = group.nlargest(config.top_n, "score")
        picks.append(
            selected[
                [
                    "trade_date",
                    "symbol",
                    "name",
                    "board_code",
                    "board_name",
                    "board_corr",
                    "close",
                    "mom_120_20",
                    "ret_60",
                    "near_high_252",
                    "vol_60",
                    "amount_ma20",
                    "volume_surge",
                    "board_score",
                    "score",
                ]
            ].rename(columns={"trade_date": "signal_date", "close": "signal_close"})
        )

    if not picks:
        raise RuntimeError("strategy produced no monthly picks")
    return pd.concat(picks, ignore_index=True)


def score_series(group: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "sector_attack_v1":
        return (
            group["board_score"] * 0.35
            + group["stock_mom_rank"] * 0.25
            + group["near_high_rank"] * 0.15
            + group["ret60_rank"] * 0.10
            + group["liq_rank"] * 0.10
            + group["trend_rank"] * 0.05
        )
    if mode == "sector_breakout_v1":
        return (
            group["board_score"] * 0.30
            + group["near_high_rank"] * 0.25
            + group["stock_mom_rank"] * 0.20
            + group["ret60_rank"] * 0.15
            + group["liq_rank"] * 0.10
        )
    if mode == "sector_surge_v1":
        return (
            group["board_score"] * 0.30
            + group["stock_mom_rank"] * 0.25
            + group["surge_rank"] * 0.20
            + group["ret60_rank"] * 0.15
            + group["liq_rank"] * 0.10
        )
    raise ValueError(f"unknown score mode: {mode}")


def main() -> int:
    args = parse_args()
    config = BacktestConfig(
        db=args.db,
        output_dir=args.output_dir,
        years=args.years,
        start_date=None,
        end_date=None,
        top_n=args.top_n,
        min_amount=args.min_amount,
        min_price=args.min_price,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        initial_cash=args.initial_cash,
        score_mode=args.score_mode,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    df, start_date, end_date = load_data(config)
    df = add_attack_factors(df)
    lookback_start = start_date - pd.Timedelta(days=380)
    boards = add_board_factors(load_industry_bars(config.db, lookback_start, end_date))
    signal_dates = month_end_signal_dates(df, start_date, end_date)
    picks = build_signal_table(
        df=df,
        boards=boards,
        signal_dates=signal_dates,
        config=config,
        top_boards=args.top_boards,
        corr_window=args.corr_window,
        min_corr=args.min_corr,
        score_mode=args.score_mode,
    )
    equity, trades, metrics = run_backtest(df, picks, start_date, end_date, config)
    metrics["benchmark"] = compute_equal_weight_benchmark(df, start_date, end_date)
    metrics["config"] = {
        "strategy": "ths_sector_momentum_stock_attack",
        "score_mode": args.score_mode,
        "top_n": config.top_n,
        "top_boards": args.top_boards,
        "corr_window": args.corr_window,
        "min_corr": args.min_corr,
        "min_amount": config.min_amount,
        "min_price": config.min_price,
        "buy_cost": config.buy_cost,
        "sell_cost": config.sell_cost,
        "years": config.years,
        "signal_rule": "month-end sector/stock factors, next trading day open execution",
    }
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    prefix = config.output_dir / (
        f"sector_attack_3y_{args.score_mode}_top{config.top_n}"
        f"_boards{args.top_boards}_corr{str(args.min_corr).replace('.', 'p')}"
        f"_amt{threshold_tag(config.min_amount)}_p{threshold_tag(config.min_price)}"
    )
    equity.to_csv(prefix.with_suffix(".equity.csv"), index=False)
    trades.to_csv(prefix.with_suffix(".trades.csv"), index=False)
    picks.to_csv(prefix.with_suffix(".picks.csv"), index=False)
    with prefix.with_suffix(".metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

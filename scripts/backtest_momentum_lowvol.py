#!/usr/bin/env python3
"""Backtest a simple A-share daily-bar momentum/low-volatility rotation strategy."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    db: Path
    output_dir: Path
    years: int
    start_date: Optional[str]
    end_date: Optional[str]
    top_n: int
    min_amount: float
    min_price: float
    buy_cost: float
    sell_cost: float
    initial_cash: float
    score_mode: str


def parse_args() -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Backtest monthly momentum + low-volatility rotation on local SQLite daily bars."
    )
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--start-date", help="inclusive YYYY-MM-DD/YYYMMDD backtest start")
    parser.add_argument("--end-date", help="inclusive YYYY-MM-DD/YYYMMDD backtest end")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-amount", type=float, default=50_000_000)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--buy-cost", type=float, default=0.0003)
    parser.add_argument("--sell-cost", type=float, default=0.0008)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument(
        "--score-mode",
        choices=[
            "balanced",
            "pure_momentum",
            "momentum_liquidity",
            "momentum_highvol",
            "lowvol_momentum",
            "residual_lowvol",
            "anti_lottery_momentum",
            "china_daily_v1",
        ],
        default="lowvol_momentum",
    )
    args = parser.parse_args()
    return BacktestConfig(
        db=args.db,
        output_dir=args.output_dir,
        years=args.years,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        min_amount=args.min_amount,
        min_price=args.min_price,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        initial_cash=args.initial_cash,
        score_mode=args.score_mode,
    )


def parse_date_arg(value: str) -> pd.Timestamp:
    normalized = value.strip().replace("-", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError("date must be YYYY-MM-DD or YYYYMMDD")
    return pd.Timestamp(datetime.strptime(normalized, "%Y%m%d").date())


def load_data(config: BacktestConfig) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    with sqlite3.connect(config.db) as conn:
        db_max_date = pd.Timestamp(
            conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()[0]
        )
        end_date = parse_date_arg(config.end_date) if config.end_date else db_max_date
        end_date = min(end_date, db_max_date)
        start_date = parse_date_arg(config.start_date) if config.start_date else end_date - pd.DateOffset(years=config.years)
        lookback_start = start_date - pd.Timedelta(days=260)

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
        df = pd.read_sql_query(
            query,
            conn,
            params=(lookback_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )

    if df.empty:
        raise RuntimeError("no daily bars loaded for backtest")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"])
    for col in ["open", "high", "low", "close", "amount", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "amount"])
    return df, start_date.normalize(), end_date.normalize()


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    group = df.groupby("symbol", group_keys=False)
    df["ret_1d"] = group["close"].pct_change()
    df["market_ret"] = df.groupby("trade_date")["ret_1d"].transform("mean")

    group = df.groupby("symbol", group_keys=False)
    ret_x_market = df["ret_1d"] * df["market_ret"]
    market_sq = df["market_ret"] ** 2
    mean_ret = group["ret_1d"].rolling(120).mean().reset_index(level=0, drop=True)
    mean_market = group["market_ret"].rolling(120).mean().reset_index(level=0, drop=True)
    mean_ret_x_market = ret_x_market.groupby(df["symbol"]).rolling(120).mean().reset_index(level=0, drop=True)
    mean_market_sq = market_sq.groupby(df["symbol"]).rolling(120).mean().reset_index(level=0, drop=True)
    cov_market = mean_ret_x_market - mean_ret * mean_market
    var_market = mean_market_sq - mean_market**2
    df["beta_120"] = cov_market / var_market.replace(0, np.nan)
    df["resid_ret_1d"] = df["ret_1d"] - df["beta_120"] * df["market_ret"]

    group = df.groupby("symbol", group_keys=False)
    df["mom_120_20"] = group["close"].shift(20) / group["close"].shift(120) - 1.0
    df["resid_mom_120_20"] = group["resid_ret_1d"].shift(20).rolling(100).sum().reset_index(level=0, drop=True)
    df["ret_20"] = group["close"].pct_change(20)
    df["vol_60"] = group["ret_1d"].rolling(60).std().reset_index(level=0, drop=True)
    df["max_ret_20"] = group["ret_1d"].rolling(20).max().reset_index(level=0, drop=True)
    df["ma120"] = group["close"].rolling(120).mean().reset_index(level=0, drop=True)
    df["ma60"] = group["close"].rolling(60).mean().reset_index(level=0, drop=True)
    df["trend_quality"] = df["ma60"] / df["ma120"] - 1.0
    df["drawdown_120"] = df["close"] / group["close"].rolling(120).max().reset_index(level=0, drop=True) - 1.0
    df["amount_ma20"] = group["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    df["turnover_ma20"] = group["turnover"].rolling(20).mean().reset_index(level=0, drop=True)
    return df


def month_end_signal_dates(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[pd.Timestamp]:
    dates = pd.Series(sorted(df.loc[df["trade_date"].between(start_date, end_date), "trade_date"].unique()))
    if dates.empty:
        return []
    return list(dates.groupby(dates.dt.to_period("M")).max())


def build_signal_table(
    df: pd.DataFrame,
    signal_dates: list[pd.Timestamp],
    config: BacktestConfig,
) -> pd.DataFrame:
    picks = []
    signal_set = set(signal_dates)
    signal_df = df[df["trade_date"].isin(signal_set)].copy()
    signal_df = signal_df[
        (signal_df["mom_120_20"].notna())
        & (signal_df["resid_mom_120_20"].notna())
        & (signal_df["vol_60"].notna())
        & (signal_df["max_ret_20"].notna())
        & (signal_df["trend_quality"].notna())
        & (signal_df["drawdown_120"].notna())
        & (signal_df["amount_ma20"] >= config.min_amount)
        & (signal_df["close"] > signal_df["ma120"])
        & (signal_df["close"] > config.min_price)
    ]

    for _signal_date, group in signal_df.groupby("trade_date"):
        group = group.copy()
        if group.empty:
            continue
        group["mom_rank"] = group["mom_120_20"].rank(pct=True)
        group["low_vol_rank"] = (-group["vol_60"]).rank(pct=True)
        group["high_vol_rank"] = group["vol_60"].rank(pct=True)
        group["liq_rank"] = group["amount_ma20"].rank(pct=True)
        group["resid_mom_rank"] = group["resid_mom_120_20"].rank(pct=True)
        group["low_max_ret_rank"] = (-group["max_ret_20"]).rank(pct=True)
        group["trend_rank"] = group["trend_quality"].rank(pct=True)
        group["drawdown_rank"] = group["drawdown_120"].rank(pct=True)
        group["low_turnover_rank"] = (-group["turnover_ma20"]).rank(pct=True)
        group["score"] = score_series(group, config.score_mode)
        selected = group.nlargest(config.top_n, "score")
        picks.append(
            selected[
                [
                    "trade_date",
                    "symbol",
                    "name",
                    "close",
                    "mom_120_20",
                    "resid_mom_120_20",
                    "vol_60",
                    "max_ret_20",
                    "trend_quality",
                    "drawdown_120",
                    "amount_ma20",
                    "turnover_ma20",
                    "score",
                ]
            ].rename(columns={"trade_date": "signal_date", "close": "signal_close"})
        )

    if not picks:
        raise RuntimeError("strategy produced no monthly picks")
    return pd.concat(picks, ignore_index=True)


def score_series(group: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "balanced":
        return group["mom_rank"] * 0.60 + group["low_vol_rank"] * 0.30 + group["liq_rank"] * 0.10
    if mode == "pure_momentum":
        return group["mom_rank"]
    if mode == "momentum_liquidity":
        return group["mom_rank"] * 0.80 + group["liq_rank"] * 0.20
    if mode == "momentum_highvol":
        return group["mom_rank"] * 0.70 + group["high_vol_rank"] * 0.20 + group["liq_rank"] * 0.10
    if mode == "lowvol_momentum":
        return group["low_vol_rank"] * 0.70 + group["mom_rank"] * 0.20 + group["liq_rank"] * 0.10
    if mode == "residual_lowvol":
        return (
            group["low_vol_rank"] * 0.40
            + group["resid_mom_rank"] * 0.30
            + group["low_max_ret_rank"] * 0.15
            + group["liq_rank"] * 0.10
            + group["trend_rank"] * 0.05
        )
    if mode == "anti_lottery_momentum":
        return (
            group["mom_rank"] * 0.35
            + group["low_max_ret_rank"] * 0.20
            + group["low_vol_rank"] * 0.20
            + group["liq_rank"] * 0.15
            + group["low_turnover_rank"] * 0.10
        )
    if mode == "china_daily_v1":
        return (
            group["resid_mom_rank"] * 0.30
            + group["low_vol_rank"] * 0.25
            + group["low_max_ret_rank"] * 0.15
            + group["trend_rank"] * 0.15
            + group["liq_rank"] * 0.10
            + group["drawdown_rank"] * 0.05
        )
    raise ValueError(f"unknown score mode: {mode}")


def run_backtest(
    df: pd.DataFrame,
    picks: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    price = df.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    mark_price = price.ffill()
    open_price = df.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    all_dates = price.loc[(price.index >= start_date) & (price.index <= end_date)].index
    signal_dates = sorted(pd.to_datetime(picks["signal_date"].unique()))

    cash = config.initial_cash
    positions: dict[str, float] = {}
    equity_rows = []
    trade_rows = []
    current_weight_symbols: set[str] = set()

    signal_to_symbols = {
        signal_date: group["symbol"].tolist()
        for signal_date, group in picks.groupby("signal_date")
    }

    for signal_date in signal_dates:
        future_dates = all_dates[all_dates > signal_date]
        if len(future_dates) == 0:
            continue
        exec_date = future_dates[0]
        target_symbols = [
            symbol for symbol in signal_to_symbols[signal_date] if symbol in open_price.columns
        ]
        target_symbols = [
            symbol for symbol in target_symbols if pd.notna(open_price.at[exec_date, symbol])
        ]
        if not target_symbols:
            continue

        portfolio_value = cash + sum(
            shares * mark_price.at[signal_date, symbol]
            for symbol, shares in positions.items()
            if symbol in mark_price.columns and pd.notna(mark_price.at[signal_date, symbol])
        )

        # Sell names no longer selected at next open.
        for symbol in list(positions):
            if symbol in target_symbols or pd.isna(open_price.at[exec_date, symbol]):
                continue
            shares = positions.pop(symbol)
            proceeds = shares * open_price.at[exec_date, symbol] * (1 - config.sell_cost)
            cash += proceeds
            trade_rows.append(
                {
                    "date": exec_date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "side": "sell",
                    "price": float(open_price.at[exec_date, symbol]),
                    "shares": float(shares),
                    "cash_after": float(cash),
                }
            )

        target_value = portfolio_value / len(target_symbols)
        for symbol in target_symbols:
            open_px = open_price.at[exec_date, symbol]
            current_value = positions.get(symbol, 0.0) * open_px
            diff_value = target_value - current_value
            if abs(diff_value) < portfolio_value * 0.002:
                continue
            if diff_value > 0:
                spend = min(diff_value, cash)
                shares = spend * (1 - config.buy_cost) / open_px
                if shares <= 0:
                    continue
                positions[symbol] = positions.get(symbol, 0.0) + shares
                cash -= spend
                side = "buy"
            else:
                shares = min(positions.get(symbol, 0.0), -diff_value / open_px)
                if shares <= 0:
                    continue
                positions[symbol] = positions.get(symbol, 0.0) - shares
                cash += shares * open_px * (1 - config.sell_cost)
                side = "trim"
            trade_rows.append(
                {
                    "date": exec_date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "side": side,
                    "price": float(open_px),
                    "shares": float(shares),
                    "cash_after": float(cash),
                }
            )
        positions = {symbol: shares for symbol, shares in positions.items() if shares > 1e-10}
        current_weight_symbols = set(target_symbols)

        next_signal_dates = [d for d in signal_dates if d > signal_date]
        next_signal_date = next_signal_dates[0] if next_signal_dates else end_date
        mark_dates = all_dates[(all_dates >= exec_date) & (all_dates <= next_signal_date)]
        for mark_date in mark_dates:
            market_value = 0.0
            stale_symbols = 0
            for symbol, shares in positions.items():
                if symbol in mark_price.columns and pd.notna(mark_price.at[mark_date, symbol]):
                    market_value += shares * mark_price.at[mark_date, symbol]
                else:
                    stale_symbols += 1
            equity = cash + market_value
            equity_rows.append(
                {
                    "date": mark_date,
                    "equity": equity,
                    "cash": cash,
                    "positions": len(positions),
                    "target_positions": len(current_weight_symbols),
                    "stale_symbols": stale_symbols,
                }
            )

    equity = pd.DataFrame(equity_rows).drop_duplicates(subset=["date"], keep="last")
    equity = equity.sort_values("date")
    trades = pd.DataFrame(trade_rows)
    metrics = compute_metrics(equity, trades, config)
    return equity, trades, metrics


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, config: BacktestConfig) -> dict:
    if equity.empty:
        raise RuntimeError("equity curve is empty")
    equity = equity.copy()
    equity["ret"] = equity["equity"].pct_change().fillna(0.0)
    total_return = equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1.0
    days = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days
    annual_return = (1.0 + total_return) ** (365.25 / days) - 1.0 if days > 0 else np.nan
    annual_vol = equity["ret"].std(ddof=0) * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol and not np.isnan(annual_vol) else np.nan
    drawdown = equity["equity"] / equity["equity"].cummax() - 1.0
    max_drawdown = drawdown.min()
    monthly = equity.set_index("date")["equity"].resample("ME").last().pct_change().dropna()
    positive_months = (monthly > 0).sum()
    return {
        "initial_cash": config.initial_cash,
        "start_date": equity["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": equity["date"].iloc[-1].strftime("%Y-%m-%d"),
        "final_equity": float(equity["equity"].iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe_like": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "trading_days": int(len(equity)),
        "months": int(len(monthly)),
        "positive_months": int(positive_months),
        "positive_month_rate": float(positive_months / len(monthly)) if len(monthly) else np.nan,
        "trade_count": int(len(trades)),
    }


def compute_equal_weight_benchmark(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
    temp = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)].copy()
    temp["ret"] = temp.groupby("symbol")["close"].pct_change()
    daily = temp.groupby("trade_date")["ret"].mean().dropna()
    if daily.empty:
        return {}
    curve = (1.0 + daily).cumprod()
    total_return = curve.iloc[-1] - 1.0
    days = (curve.index[-1] - curve.index[0]).days
    annual_return = (1.0 + total_return) ** (365.25 / days) - 1.0 if days > 0 else np.nan
    annual_vol = daily.std(ddof=0) * np.sqrt(252)
    drawdown = curve / curve.cummax() - 1.0
    return {
        "name": "all_loaded_symbols_equal_weight_close_to_close",
        "start_date": curve.index[0].strftime("%Y-%m-%d"),
        "end_date": curve.index[-1].strftime("%Y-%m-%d"),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "max_drawdown": float(drawdown.min()),
    }


def main() -> int:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    df, start_date, end_date = load_data(config)
    df = add_factors(df)
    signal_dates = month_end_signal_dates(df, start_date, end_date)
    picks = build_signal_table(df, signal_dates, config)
    equity, trades, metrics = run_backtest(df, picks, start_date, end_date, config)
    metrics["benchmark"] = compute_equal_weight_benchmark(df, start_date, end_date)
    metrics["config"] = {
        "strategy": "monthly_momentum_lowvol_liquidity",
        "score_mode": config.score_mode,
        "top_n": config.top_n,
        "min_amount": config.min_amount,
        "min_price": config.min_price,
        "buy_cost": config.buy_cost,
        "sell_cost": config.sell_cost,
        "years": config.years,
        "requested_start_date": config.start_date,
        "requested_end_date": config.end_date,
        "signal_rule": "month-end close factors, next trading day open execution",
        "score": score_description(config.score_mode),
    }
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    prefix = config.output_dir / (
        f"momentum_lowvol_{date_tag(start_date)}_{date_tag(end_date)}_{config.score_mode}_top{config.top_n}"
        f"_amt{threshold_tag(config.min_amount)}_p{threshold_tag(config.min_price)}"
    )
    equity.to_csv(prefix.with_suffix(".equity.csv"), index=False)
    trades.to_csv(prefix.with_suffix(".trades.csv"), index=False)
    picks.to_csv(prefix.with_suffix(".picks.csv"), index=False)
    with prefix.with_suffix(".metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def score_description(mode: str) -> str:
    descriptions = {
        "balanced": "0.60 momentum_120_20 + 0.30 low_vol_60 + 0.10 liquidity_20",
        "pure_momentum": "1.00 momentum_120_20",
        "momentum_liquidity": "0.80 momentum_120_20 + 0.20 liquidity_20",
        "momentum_highvol": "0.70 momentum_120_20 + 0.20 high_vol_60 + 0.10 liquidity_20",
        "lowvol_momentum": "0.70 low_vol_60 + 0.20 momentum_120_20 + 0.10 liquidity_20",
        "residual_lowvol": "0.40 low_vol_60 + 0.30 residual_momentum_120_20 + 0.15 low_MAX_20 + 0.10 liquidity_20 + 0.05 trend_quality",
        "anti_lottery_momentum": "0.35 momentum_120_20 + 0.20 low_MAX_20 + 0.20 low_vol_60 + 0.15 liquidity_20 + 0.10 low_turnover_20",
        "china_daily_v1": "0.30 residual_momentum_120_20 + 0.25 low_vol_60 + 0.15 low_MAX_20 + 0.15 trend_quality + 0.10 liquidity_20 + 0.05 drawdown_120",
    }
    return descriptions[mode]


def threshold_tag(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    if abs(value) >= 1_000_000 and value % 1_000_000 == 0:
        return f"{int(value / 1_000_000)}m"
    return text


def date_tag(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


if __name__ == "__main__":
    raise SystemExit(main())

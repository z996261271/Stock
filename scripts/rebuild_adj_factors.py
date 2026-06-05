#!/usr/bin/env python3
"""Rebuild adj_factors from daily_bars raw/qfq/hfq rows."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from fetch_akshare_daily import connect, init_db, normalize_symbol, refresh_adj_factors_from_daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive forward/backward factors from stored daily_bars.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--symbols", help="comma-separated symbols; default rebuilds all adjusted symbols")
    parser.add_argument("--start-date", help="optional YYYY-MM-DD lower bound")
    parser.add_argument("--end-date", help="optional YYYY-MM-DD upper bound")
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    return parser.parse_args()


def _date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def adjusted_symbol_ranges(
    conn: sqlite3.Connection,
    symbols: list[str] | None,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str, str]]:
    where = ["adjust in ('qfq', 'hfq')"]
    params: list[str] = []
    if symbols:
        where.append(f"symbol in ({', '.join('?' for _ in symbols)})")
        params.extend(symbols)
    if start_date:
        where.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("trade_date <= ?")
        params.append(end_date)
    return conn.execute(
        f"""
        select symbol, min(trade_date), max(trade_date)
        from daily_bars
        where {' and '.join(where)}
        group by symbol
        order by symbol
        """,
        params,
    ).fetchall()


def run(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    init_db(conn)
    symbols = None
    if args.symbols:
        symbols = [normalize_symbol(item) for item in args.symbols.split(",") if item.strip()]
    ranges = adjusted_symbol_ranges(conn, symbols, args.start_date, args.end_date)
    touched = 0
    for symbol, min_date, max_date in ranges:
        start = _date(args.start_date or min_date)
        end = _date(args.end_date or max_date)
        touched += max(refresh_adj_factors_from_daily(conn, symbol, start, end), 0)
    summary = {
        "db": str(args.db),
        "symbols": len(ranges),
        "adj_factor_rows_touched": touched,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    conn.close()
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

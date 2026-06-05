#!/usr/bin/env python3
"""Fetch market-level A-share valuation history into SQLite."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

from fetch_akshare_daily import connect
from quant_schema import ensure_quant_schema

SOURCE = "akshare.legulegu.a_share_pe_pb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch all-A market PE/PB valuation history.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def _date_filter(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"])
    if start_date:
        out = out[out["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["trade_date"] <= pd.Timestamp(end_date)]
    out["trade_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    return out


def fetch_market_valuation(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    pe = ak.stock_a_ttm_lyr().rename(
        columns={
            "date": "trade_date",
            "middlePETTM": "middle_pe_ttm",
            "averagePETTM": "average_pe_ttm",
            "middlePELYR": "middle_pe_lyr",
            "averagePELYR": "average_pe_lyr",
            "close": "pe_close",
        }
    )
    pb = ak.stock_a_all_pb().rename(
        columns={
            "date": "trade_date",
            "middlePB": "middle_pb",
            "equalWeightAveragePB": "average_pb",
            "close": "pb_close",
        }
    )
    columns_pe = ["trade_date", "middle_pe_ttm", "average_pe_ttm", "middle_pe_lyr", "average_pe_lyr", "pe_close"]
    columns_pb = ["trade_date", "middle_pb", "average_pb", "pb_close"]
    frame = pd.merge(pe[columns_pe], pb[columns_pb], on="trade_date", how="outer")
    frame = _date_filter(frame, start_date, end_date)
    numeric = [column for column in frame.columns if column != "trade_date"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["source"] = SOURCE
    frame["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return frame.sort_values("trade_date")


def write_market_valuation(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    ensure_quant_schema(conn)
    rows = frame[
        [
            "trade_date",
            "middle_pe_ttm",
            "average_pe_ttm",
            "middle_pe_lyr",
            "average_pe_lyr",
            "middle_pb",
            "average_pb",
            "pe_close",
            "pb_close",
            "source",
            "fetched_at",
        ]
    ].where(pd.notna(frame), None)
    conn.executemany(
        """
        INSERT OR REPLACE INTO market_valuation_daily (
            trade_date, middle_pe_ttm, average_pe_ttm, middle_pe_lyr, average_pe_lyr,
            middle_pb, average_pb, pe_close, pb_close, source, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows.itertuples(index=False, name=None),
    )
    conn.commit()
    return int(len(rows))


def main() -> int:
    args = parse_args()
    frame = fetch_market_valuation(args.start_date, args.end_date)
    with connect(args.db) as conn:
        rows = write_market_valuation(conn, frame)
    print(
        {
            "db": str(args.db),
            "rows": rows,
            "start_date": frame["trade_date"].min() if not frame.empty else None,
            "end_date": frame["trade_date"].max() if not frame.empty else None,
            "source": SOURCE,
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

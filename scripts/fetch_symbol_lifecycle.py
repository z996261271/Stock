#!/usr/bin/env python3
"""Fetch current/listed and delisted A-share lifecycle metadata into SQLite."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd

from fetch_akshare_daily import connect, init_db, normalize_symbol


LIFECYCLE_SOURCE = "akshare.exchange_stock_lists"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate symbol_lifecycle with list/delist metadata.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--no-delisted", action="store_true", help="skip delisted-company endpoints")
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    return parser.parse_args()


def _date_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"-", "--", "nan", "None"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def board_for_symbol(symbol: str, fallback: str | None = None) -> str:
    if fallback:
        return str(fallback).strip()
    if symbol.startswith("688"):
        return "科创板"
    if symbol.startswith(("300", "301")):
        return "创业板"
    if symbol.startswith(("4", "8", "9")):
        return "北交所"
    return "主板"


def market_for_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return "sh"
    if symbol.startswith(("4", "8")):
        return "bj"
    return "sz"


def _record(
    symbol: Any,
    name: Any,
    list_date: Any,
    delist_date: Any = None,
    board: str | None = None,
    market: str | None = None,
    source: str = LIFECYCLE_SOURCE,
) -> dict[str, Any]:
    normalized = normalize_symbol(symbol)
    return {
        "symbol": normalized,
        "name": "" if name is None or pd.isna(name) else str(name).strip(),
        "list_date": _date_or_none(list_date),
        "delist_date": _date_or_none(delist_date),
        "board": board_for_symbol(normalized, board),
        "market": market or market_for_symbol(normalized),
        "source": source,
    }


def fetch_current_lifecycle() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for board_name, market, symbol_arg in [
        ("主板", "sh", "主板A股"),
        ("科创板", "sh", "科创板"),
    ]:
        try:
            df = ak.stock_info_sh_name_code(symbol=symbol_arg)
            for row in df.itertuples(index=False):
                records.append(
                    _record(
                        getattr(row, "证券代码"),
                        getattr(row, "证券简称"),
                        getattr(row, "上市日期"),
                        board=board_name,
                        market=market,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - keep other exchanges usable.
            warnings.append(f"stock_info_sh_name_code({symbol_arg}) failed: {exc}")

    try:
        df = ak.stock_info_sz_name_code(symbol="A股列表")
        for row in df.itertuples(index=False):
            records.append(
                _record(
                    getattr(row, "A股代码"),
                    getattr(row, "A股简称"),
                    getattr(row, "A股上市日期"),
                    board=getattr(row, "板块", None),
                    market="sz",
                )
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stock_info_sz_name_code(A股列表) failed: {exc}")

    try:
        df = ak.stock_info_bj_name_code()
        for row in df.itertuples(index=False):
            records.append(
                _record(
                    getattr(row, "证券代码"),
                    getattr(row, "证券简称"),
                    getattr(row, "上市日期"),
                    board="北交所",
                    market="bj",
                )
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stock_info_bj_name_code failed: {exc}")

    return records, warnings


def fetch_delisted_lifecycle() -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        df = ak.stock_info_sh_delist(symbol="全部")
        for row in df.itertuples(index=False):
            records.append(
                _record(
                    getattr(row, "公司代码"),
                    getattr(row, "公司简称"),
                    getattr(row, "上市日期", None),
                    getattr(row, "终止上市日期", None)
                    if hasattr(row, "终止上市日期")
                    else getattr(row, "暂停上市日期", None),
                    board=board_for_symbol(normalize_symbol(getattr(row, "公司代码"))),
                    market="sh",
                    source="akshare.stock_info_sh_delist",
                )
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stock_info_sh_delist(全部) failed: {exc}")

    try:
        df = ak.stock_info_sz_delist(symbol="终止上市公司")
        for row in df.itertuples(index=False):
            records.append(
                _record(
                    getattr(row, "证券代码"),
                    getattr(row, "证券简称"),
                    getattr(row, "上市日期", None),
                    getattr(row, "终止上市日期", None),
                    board=board_for_symbol(normalize_symbol(getattr(row, "证券代码"))),
                    market="sz",
                    source="akshare.stock_info_sz_delist",
                )
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stock_info_sz_delist(终止上市公司) failed: {exc}")

    return records, warnings


def save_lifecycle(conn: sqlite3.Connection, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows_by_symbol: dict[str, dict[str, Any]] = {}
    for record in records:
        existing = rows_by_symbol.get(record["symbol"])
        if existing is None:
            rows_by_symbol[record["symbol"]] = record
            continue
        existing["name"] = record["name"] or existing["name"]
        existing["list_date"] = existing["list_date"] or record["list_date"]
        existing["delist_date"] = record["delist_date"] or existing["delist_date"]
        existing["board"] = record["board"] or existing["board"]
        existing["market"] = record["market"] or existing["market"]
        if record["source"] not in existing["source"].split("+"):
            existing["source"] = f"{existing['source']}+{record['source']}"

    rows = [
        (
            item["symbol"],
            item["name"],
            item["list_date"],
            item["delist_date"],
            item["board"],
            item["market"],
            item["source"],
            fetched_at,
        )
        for item in rows_by_symbol.values()
    ]
    conn.executemany(
        """
        INSERT INTO symbol_lifecycle (
            symbol, name, list_date, delist_date, board, market, source, fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            name = COALESCE(NULLIF(excluded.name, ''), symbol_lifecycle.name),
            list_date = COALESCE(excluded.list_date, symbol_lifecycle.list_date),
            delist_date = COALESCE(excluded.delist_date, symbol_lifecycle.delist_date),
            board = COALESCE(excluded.board, symbol_lifecycle.board),
            market = COALESCE(excluded.market, symbol_lifecycle.market),
            source = excluded.source,
            fetched_at = excluded.fetched_at
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def run(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    init_db(conn)

    current_records, current_warnings = fetch_current_lifecycle()
    delisted_records: list[dict[str, Any]] = []
    delisted_warnings: list[str] = []
    if not args.no_delisted:
        delisted_records, delisted_warnings = fetch_delisted_lifecycle()

    saved = save_lifecycle(conn, [*current_records, *delisted_records])
    summary = {
        "db": str(args.db),
        "current_records": len(current_records),
        "delisted_records": len(delisted_records),
        "saved_symbols": saved,
        "warnings": [*current_warnings, *delisted_warnings],
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    conn.close()
    return 1 if not current_records else 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

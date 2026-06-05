#!/usr/bin/env python3
"""Fetch Eastmoney industry constituents into the canonical symbol_industries table."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from professional_quant.data.industry import ensure_symbol_industry_schema, normalize_symbol, upsert_symbol_industries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Eastmoney stock-to-industry mappings.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--board-retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--min-unique-symbols", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_em_industry_boards(retries: int = 5, sleep: float = 0.5) -> pd.DataFrame:
    last_error: Exception | None = None
    boards = pd.DataFrame()
    for attempt in range(max(retries, 1)):
        try:
            boards = ak.stock_board_industry_name_em()
            break
        except Exception as exc:  # noqa: BLE001 - vendor endpoint can fail transiently.
            last_error = exc
            time.sleep(max(sleep, 0.0) + attempt)
    required = {"板块名称", "板块代码"}
    if boards.empty or not required.issubset(boards.columns):
        raise RuntimeError(f"Eastmoney industry board listing unavailable: {last_error}")
    return boards[["板块名称", "板块代码"]].dropna().drop_duplicates("板块代码").reset_index(drop=True)


def fetch_constituents(board_name: str, board_code: str, retries: int, sleep: float) -> tuple[pd.DataFrame, str | None]:
    last_error: Exception | None = None
    symbol = board_code or board_name
    for attempt in range(max(retries, 1)):
        try:
            frame = ak.stock_board_industry_cons_em(symbol=symbol)
            if frame.empty:
                return frame, "empty"
            return frame, None
        except Exception as exc:  # noqa: BLE001 - vendor endpoint can fail transiently.
            last_error = exc
            time.sleep(max(sleep, 0.0) + attempt)
    return pd.DataFrame(), str(last_error)


def build_symbol_industry_rows(boards: pd.DataFrame, constituents: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], int]:
    """Convert board constituents into one deterministic industry row per stock symbol."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    fetched_at = datetime.now().isoformat(timespec="seconds")
    for board in boards.to_dict("records"):
        board_name = str(board["板块名称"])
        board_code = str(board["板块代码"])
        frame = constituents.get(board_code, pd.DataFrame())
        if frame.empty or "代码" not in frame.columns:
            continue
        for record in frame.to_dict("records"):
            symbol = normalize_symbol(record.get("代码"))
            if not symbol:
                continue
            if symbol in seen:
                duplicates += 1
                continue
            seen.add(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "industry_name": board_name,
                    "industry_code": board_code,
                    "industry_level": "eastmoney_industry_board",
                    "provider": "eastmoney",
                    "source": "akshare.stock_board_industry_cons_em",
                    "fetched_at": fetched_at,
                }
            )
    return rows, duplicates


def fetch_all_constituents(boards: pd.DataFrame, workers: int, retries: int, sleep: float) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    output: dict[str, pd.DataFrame] = {}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {
            executor.submit(fetch_constituents, str(row["板块名称"]), str(row["板块代码"]), retries, sleep): row
            for row in boards.to_dict("records")
        }
        for future in as_completed(futures):
            row = futures[future]
            board_name = str(row["板块名称"])
            board_code = str(row["板块代码"])
            frame, error = future.result()
            output[board_code] = frame
            if error:
                errors.append({"industry_name": board_name, "industry_code": board_code, "error": error})
            print(f"{board_code} {board_name}: {len(frame)} rows" + (f" ({error})" if error else ""), flush=True)
    return output, errors


def main() -> int:
    args = parse_args()
    boards = load_em_industry_boards(args.board_retries, args.sleep)
    constituents, errors = fetch_all_constituents(boards, args.workers, args.retries, args.sleep)
    rows, duplicates = build_symbol_industry_rows(boards, constituents)
    output: dict[str, Any] = {
        "db": str(args.db),
        "provider": "eastmoney",
        "boards": int(len(boards)),
        "constituent_rows": int(sum(len(frame) for frame in constituents.values())),
        "unique_symbols": int(len(rows)),
        "duplicate_symbol_rows": int(duplicates),
        "errors": errors,
        "dry_run": args.dry_run,
    }
    if args.min_unique_symbols:
        output["min_unique_symbols"] = int(args.min_unique_symbols)
        output["meets_min_unique_symbols"] = len(rows) >= args.min_unique_symbols
    if not args.dry_run:
        with sqlite3.connect(args.db) as conn:
            ensure_symbol_industry_schema(conn)
            output["rows_upserted"] = upsert_symbol_industries(conn, rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not rows:
        return 2
    if args.min_unique_symbols and len(rows) < args.min_unique_symbols:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

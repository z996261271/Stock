#!/usr/bin/env python3
"""Fetch historical ST/suspension status into symbol_status_daily via BaoStock."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from fetch_akshare_daily import (
    baostock_symbol,
    call_with_timeout,
    connect,
    ensure_baostock,
    format_iso,
    init_db,
    logout_baostock,
    normalize_symbol,
    parse_yyyymmdd,
    reset_baostock,
)
from quant_universe import is_main_board_symbol


STATUS_FIELDS = "date,code,tradestatus,isST"
STATUS_SOURCE = "baostock.query_history_k_data_plus.status"
STATUS_COLUMNS = [
    "symbol",
    "trade_date",
    "is_st",
    "is_suspended",
    "board",
    "source",
    "fetched_at",
]


def board_for_symbol(symbol: str) -> str:
    if symbol.startswith("688"):
        return "科创板"
    if symbol.startswith(("300", "301")):
        return "创业板"
    if symbol.startswith(("4", "8", "9")):
        return "北交所"
    if is_main_board_symbol(symbol):
        return "主板"
    return "其他"


def existing_min_date(conn: sqlite3.Connection, symbol: str) -> date | None:
    row = conn.execute(
        "select min(trade_date) from symbol_status_daily where symbol = ?",
        (symbol,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return datetime.strptime(row[0], "%Y-%m-%d").date()


def existing_max_date(conn: sqlite3.Connection, symbol: str) -> date | None:
    row = conn.execute(
        "select max(trade_date) from symbol_status_daily where symbol = ?",
        (symbol,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return datetime.strptime(row[0], "%Y-%m-%d").date()


def select_symbols(conn: sqlite3.Connection, requested: str | None, max_symbols: int | None, board_scope: str) -> list[str]:
    if requested:
        symbols = sorted(dict.fromkeys(normalize_symbol(item) for item in requested.split(",") if item.strip()))
    else:
        lifecycle_exists = conn.execute(
            "select 1 from sqlite_master where type='table' and name='symbol_lifecycle'"
        ).fetchone()
        if lifecycle_exists:
            symbols = [
                row[0]
                for row in conn.execute(
                    """
                    select symbol
                    from symbol_lifecycle
                    where symbol is not null
                    order by symbol
                    """
                )
            ]
        else:
            symbols = []
        if not symbols:
            symbols = [row[0] for row in conn.execute("select symbol from symbols order by symbol")]

    if board_scope == "main":
        symbols = [symbol for symbol in symbols if is_main_board_symbol(symbol)]
    elif board_scope != "all":
        raise ValueError(f"unknown board scope: {board_scope}")
    if max_symbols:
        symbols = symbols[:max_symbols]
    return symbols


def build_jobs(
    conn: sqlite3.Connection,
    symbols: list[str],
    start_date: date,
    end_date: date,
    full_refresh: bool,
    progress_every: int,
) -> tuple[list[tuple[int, int, str, date, date]], int]:
    total_jobs = len(symbols)
    jobs: list[tuple[int, int, str, date, date]] = []
    skipped = 0
    for job_index, symbol in enumerate(symbols, start=1):
        if full_refresh:
            ranges = [(start_date, end_date)]
        else:
            local_min = existing_min_date(conn, symbol)
            local_max = existing_max_date(conn, symbol)
            if not local_min or not local_max:
                ranges = [(start_date, end_date)]
            else:
                ranges = []
                if local_min > start_date:
                    ranges.append((start_date, min(end_date, local_min - timedelta(days=1))))
                if local_max < end_date:
                    ranges.append((max(start_date, local_max + timedelta(days=1)), end_date))

        valid_ranges = [(start, end) for start, end in ranges if start <= end]
        if not valid_ranges:
            skipped += 1
            if should_print_progress(job_index, total_jobs, progress_every):
                print(f"[{job_index}/{total_jobs}] {symbol}: status up to date")
            continue
        for start, end in valid_ranges:
            jobs.append((job_index, total_jobs, symbol, start, end))
    return jobs, skipped


def fetch_status_baostock(symbol: str, start: date, end: date, query_timeout: int) -> pd.DataFrame:
    def query() -> pd.DataFrame:
        bs = ensure_baostock()
        rs = bs.query_history_k_data_plus(
            baostock_symbol(symbol),
            STATUS_FIELDS,
            start_date=format_iso(start),
            end_date=format_iso(end),
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock status query failed for {symbol}: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=rs.fields)

    return call_with_timeout(query_timeout, query)


def normalize_status(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    out = pd.DataFrame()
    out["symbol"] = df["code"].map(normalize_symbol) if "code" in df.columns else symbol
    out["trade_date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    is_st_raw = df["isST"] if "isST" in df.columns else pd.Series(0, index=df.index)
    status_raw = df["tradestatus"] if "tradestatus" in df.columns else pd.Series(1, index=df.index)
    out["is_st"] = pd.to_numeric(is_st_raw, errors="coerce").fillna(0).astype(int)
    trading = pd.to_numeric(status_raw, errors="coerce")
    out["is_suspended"] = (trading.fillna(1).astype(int) == 0).astype(int)
    out["board"] = out["symbol"].map(board_for_symbol)
    out["source"] = STATUS_SOURCE
    out["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return out[STATUS_COLUMNS]


def fetch_job(job: tuple[int, int, str, date, date], query_timeout: int) -> tuple[tuple[int, int, str, date, date], pd.DataFrame]:
    _job_index, _total_jobs, symbol, start, end = job
    try:
        raw = fetch_status_baostock(symbol, start, end, query_timeout)
        return job, normalize_status(raw, symbol)
    except Exception:
        reset_baostock()
        raise


def save_status(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
    placeholders = ", ".join("?" for _ in STATUS_COLUMNS)
    update_columns = [column for column in STATUS_COLUMNS if column not in {"symbol", "trade_date"}]
    update_sql = ", ".join(f"{column} = excluded.{column}" for column in update_columns)
    conn.executemany(
        f"""
        INSERT INTO symbol_status_daily ({", ".join(STATUS_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(symbol, trade_date) DO UPDATE SET
            {update_sql}
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def run_jobs_sequential(
    conn: sqlite3.Connection,
    jobs: list[tuple[int, int, str, date, date]],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    success = 0
    failed = 0
    rows_total = 0
    for job in jobs:
        job_index, total_jobs, symbol, start, end = job
        try:
            _job, normalized = fetch_job(job, args.query_timeout)
            rows = save_status(conn, normalized)
            success += 1
            rows_total += rows
            if should_print_progress(job_index, total_jobs, args.progress_every):
                print(f"[{job_index}/{total_jobs}] {symbol} {format_iso(start)}..{format_iso(end)}: saved {rows} status rows")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{job_index}/{total_jobs}] {symbol}: status failed: {exc}", file=sys.stderr)
            if args.stop_on_error:
                break
        time.sleep(args.sleep)
    return success, failed, rows_total


def run_jobs_parallel(
    conn: sqlite3.Connection,
    jobs: list[tuple[int, int, str, date, date]],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    success = 0
    failed = 0
    rows_total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_job, job, args.query_timeout): job for job in jobs}
        for future in as_completed(futures):
            job_index, total_jobs, symbol, start, end = futures[future]
            try:
                _job, normalized = future.result()
                rows = save_status(conn, normalized)
                success += 1
                rows_total += rows
                if should_print_progress(job_index, total_jobs, args.progress_every):
                    print(f"[{job_index}/{total_jobs}] {symbol} {format_iso(start)}..{format_iso(end)}: saved {rows} status rows")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[{job_index}/{total_jobs}] {symbol}: status failed: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    for pending in futures:
                        pending.cancel()
                    break
    return success, failed, rows_total


def should_print_progress(job_index: int, total_jobs: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return job_index == 1 or job_index == total_jobs or job_index % progress_every == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate symbol_status_daily with historical ST/suspension flags.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--start-date", type=parse_yyyymmdd, default=parse_yyyymmdd("20060101"))
    parser.add_argument("--end-date", type=parse_yyyymmdd, default=date.today())
    parser.add_argument("--symbols", help="comma-separated symbols; default uses symbol_lifecycle then symbols")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--board-scope", choices=["main", "all"], default="all")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--query-timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.start_date > args.end_date:
        raise ValueError("start-date must be <= end-date")
    conn = connect(args.db)
    init_db(conn)
    symbols = select_symbols(conn, args.symbols, args.max_symbols, args.board_scope)
    if not symbols:
        raise RuntimeError("no symbols selected")
    jobs, skipped = build_jobs(conn, symbols, args.start_date, args.end_date, args.full_refresh, args.progress_every)
    print(
        f"selected {len(symbols)} symbols, jobs={len(jobs)}, skipped={skipped}, "
        f"range={format_iso(args.start_date)}..{format_iso(args.end_date)}, db={args.db}"
    )
    if args.workers <= 1:
        success, failed, rows_total = run_jobs_sequential(conn, jobs, args)
    else:
        success, failed, rows_total = run_jobs_parallel(conn, jobs, args)
    logout_baostock()
    summary = {
        "db": str(args.db),
        "symbols": len(symbols),
        "jobs": len(jobs),
        "skipped": skipped,
        "success": success,
        "failed": failed,
        "rows_saved": rows_total,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    conn.close()
    return 1 if failed else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch A-share daily bars from AKShare into a local SQLite database."""

from __future__ import annotations

import argparse
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

from quant_schema import ensure_quant_schema

EASTMONEY_SOURCE = "eastmoney.push2his.kline"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
AKSHARE_SOURCE = "akshare.stock_zh_a_hist"
TX_SOURCE = "akshare.stock_zh_a_hist_tx"
BAOSTOCK_SOURCE = "baostock.query_history_k_data_plus"
BAOSTOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)
_BAOSTOCK = None
_BAOSTOCK_LOGGED_IN = False


class FetchTimeout(TimeoutError):
    """Raised when one vendor request exceeds the configured timeout."""

COLUMN_MAP = {
    "日期": "trade_date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "chg",
    "换手率": "turnover",
}

BAR_COLUMNS = [
    "symbol",
    "trade_date",
    "adjust",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover",
    "source",
    "fetched_at",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover",
]


def parse_yyyymmdd(value: str) -> date:
    normalized = value.replace("-", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise argparse.ArgumentTypeError("date must be YYYYMMDD or YYYY-MM-DD")
    return datetime.strptime(normalized, "%Y%m%d").date()


def format_yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def format_iso(value: date) -> str:
    return value.isoformat()


def normalize_symbol(value: object) -> str:
    return str(value).strip().split(".")[-1].zfill(6)


def adjust_values(adjust: str) -> list[str]:
    if adjust == "both":
        return ["raw", "hfq"]
    if adjust == "all":
        return ["raw", "qfq", "hfq"]
    return [adjust]


def ak_adjust_value(adjust: str) -> str:
    return "" if adjust == "raw" else adjust


def eastmoney_adjust_value(adjust: str) -> str:
    values = {
        "raw": "0",
        "qfq": "1",
        "hfq": "2",
    }
    return values[adjust]


def baostock_adjust_value(adjust: str) -> str:
    values = {
        "raw": "3",
        "qfq": "2",
        "hfq": "1",
    }
    return values[adjust]


def baostock_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"sh.{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj.{symbol}"
    return f"sz.{symbol}"


def eastmoney_secid(symbol: str) -> str:
    market = "1" if symbol.startswith(("5", "6", "9")) else "0"
    return f"{market}.{symbol}"


def tx_symbol(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def ensure_baostock():
    global _BAOSTOCK, _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK is None:
        import baostock as bs  # type: ignore[import-not-found]

        _BAOSTOCK = bs
    if not _BAOSTOCK_LOGGED_IN:
        login_result = _BAOSTOCK.login()
        if login_result.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")
        _BAOSTOCK_LOGGED_IN = True
    return _BAOSTOCK


def logout_baostock() -> None:
    global _BAOSTOCK_LOGGED_IN
    if _BAOSTOCK is not None and _BAOSTOCK_LOGGED_IN:
        _BAOSTOCK.logout()
        _BAOSTOCK_LOGGED_IN = False


def reset_baostock() -> None:
    global _BAOSTOCK_LOGGED_IN
    try:
        import baostock.common.context as bs_context  # type: ignore[import-not-found]

        default_socket = getattr(bs_context, "default_socket", None)
        if default_socket is not None:
            default_socket.close()
            bs_context.default_socket = None
    except Exception:  # noqa: BLE001 - best-effort reset after a stuck vendor call.
        pass
    if _BAOSTOCK is not None and _BAOSTOCK_LOGGED_IN:
        try:
            _BAOSTOCK.logout()
        except Exception:  # noqa: BLE001 - best-effort reset after a stuck vendor call.
            pass
    _BAOSTOCK_LOGGED_IN = False


def call_with_timeout(timeout_seconds: int, func):
    if timeout_seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return func()

    def handle_timeout(_signum, _frame):
        raise FetchTimeout(f"vendor request exceeded {timeout_seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            fetched_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            adjust TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            amplitude REAL,
            pct_chg REAL,
            chg REAL,
            turnover REAL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date, adjust)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_bars_date
            ON daily_bars (trade_date);

        CREATE INDEX IF NOT EXISTS idx_daily_bars_symbol_date
            ON daily_bars (symbol, trade_date);

        CREATE INDEX IF NOT EXISTS idx_daily_bars_adjust_date_symbol
            ON daily_bars (adjust, trade_date, symbol);

        CREATE TABLE IF NOT EXISTS fetch_status (
            symbol TEXT NOT NULL,
            adjust TEXT NOT NULL,
            requested_start TEXT NOT NULL,
            requested_end TEXT NOT NULL,
            last_status TEXT NOT NULL,
            rows_fetched INTEGER NOT NULL,
            source_used TEXT,
            message TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (symbol, adjust)
        );

        CREATE VIEW IF NOT EXISTS adjusted_daily_bars AS
        SELECT
            symbol,
            trade_date,
            adjust AS adjust_type,
            open,
            high,
            low,
            close,
            volume,
            amount,
            source,
            fetched_at
        FROM daily_bars;
        """
    )
    ensure_column(conn, "fetch_status", "source_used", "TEXT")
    ensure_quant_schema(conn)
    conn.commit()


def ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    """Add a nullable column to an existing table if a local DB predates it."""
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def normalize_symbol_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("AKShare returned an empty A-share symbol list")

    code_col = next((col for col in ["代码", "code", "证券代码", "A股代码"] if col in df.columns), None)
    name_col = next((col for col in ["名称", "name", "证券简称", "A股简称"] if col in df.columns), None)
    if not code_col:
        raise RuntimeError(f"unexpected symbol-list columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["symbol"] = df[code_col].map(normalize_symbol)
    out["name"] = df[name_col].astype(str) if name_col else ""
    out = out.drop_duplicates(subset=["symbol"]).sort_values("symbol")
    return out


def fetch_symbols_once() -> pd.DataFrame:
    last_exc: Exception | None = None
    for func_name in ["stock_zh_a_spot_em", "stock_info_a_code_name"]:
        try:
            df = getattr(ak, func_name)()
            symbols = normalize_symbol_frame(df)
            print(f"symbol list source: akshare.{func_name}")
            return symbols
        except Exception as exc:  # noqa: BLE001 - try the next free AKShare endpoint.
            last_exc = exc
            print(f"warning: akshare.{func_name} failed: {exc}", file=sys.stderr)
    assert last_exc is not None
    raise last_exc


def fetch_symbols(retries: int, retry_sleep: float) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return fetch_symbols_once()
        except Exception as exc:  # noqa: BLE001 - retry vendor/network failures.
            last_exc = exc
            if attempt <= retries:
                time.sleep(retry_sleep * attempt)
    assert last_exc is not None
    raise last_exc


def save_symbols(conn: sqlite3.Connection, symbols: pd.DataFrame) -> None:
    fetched_at = datetime.now().isoformat(timespec="seconds")
    rows = [(row.symbol, row.name, fetched_at) for row in symbols.itertuples(index=False)]
    conn.executemany(
        """
        INSERT INTO symbols (symbol, name, fetched_at)
        VALUES (?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            name = excluded.name,
            fetched_at = excluded.fetched_at
        """,
        rows,
    )
    conn.commit()


def max_trade_date(conn: sqlite3.Connection, symbol: str, adjust: str) -> date | None:
    row = conn.execute(
        """
        SELECT MAX(trade_date)
        FROM daily_bars
        WHERE symbol = ? AND adjust = ?
        """,
        (symbol, adjust),
    ).fetchone()
    if not row or not row[0]:
        return None
    return datetime.strptime(row[0], "%Y-%m-%d").date()


def min_trade_date(conn: sqlite3.Connection, symbol: str, adjust: str) -> date | None:
    row = conn.execute(
        """
        SELECT MIN(trade_date)
        FROM daily_bars
        WHERE symbol = ? AND adjust = ?
        """,
        (symbol, adjust),
    ).fetchone()
    if not row or not row[0]:
        return None
    return datetime.strptime(row[0], "%Y-%m-%d").date()


def raw_trade_range(
    conn: sqlite3.Connection,
    symbol: str,
    start_date: date,
    end_date: date,
) -> tuple[date, date] | None:
    row = conn.execute(
        """
        SELECT MIN(trade_date), MAX(trade_date)
        FROM daily_bars
        WHERE symbol = ?
          AND adjust = 'raw'
          AND trade_date >= ?
          AND trade_date <= ?
        """,
        (symbol, format_iso(start_date), format_iso(end_date)),
    ).fetchone()
    if not row or not row[0] or not row[1]:
        return None
    return (
        datetime.strptime(row[0], "%Y-%m-%d").date(),
        datetime.strptime(row[1], "%Y-%m-%d").date(),
    )


def fetch_daily_akshare(symbol: str, start: date, end: date, adjust: str, timeout: float) -> pd.DataFrame:
    kwargs = {
        "symbol": symbol,
        "period": "daily",
        "start_date": format_yyyymmdd(start),
        "end_date": format_yyyymmdd(end),
        "adjust": ak_adjust_value(adjust),
    }
    try:
        return ak.stock_zh_a_hist(**kwargs, timeout=timeout)
    except TypeError as exc:
        if "timeout" not in str(exc):
            raise
        return ak.stock_zh_a_hist(**kwargs)


def fetch_daily_eastmoney(symbol: str, start: date, end: date, adjust: str, timeout: float) -> pd.DataFrame:
    params = {
        "secid": eastmoney_secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": eastmoney_adjust_value(adjust),
        "beg": format_yyyymmdd(start),
        "end": format_yyyymmdd(end),
    }
    response = requests.get(
        EASTMONEY_KLINE_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("rc") != 0:
        raise RuntimeError(f"Eastmoney rc={payload.get('rc')} for {symbol}: {payload}")

    data = payload.get("data") or {}
    klines = data.get("klines") or []
    rows = [line.split(",") for line in klines]
    return pd.DataFrame(
        rows,
        columns=[
            "trade_date",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "chg",
            "turnover",
        ],
    )


def fetch_daily_tx(symbol: str, start: date, end: date, adjust: str, timeout: float) -> pd.DataFrame:
    return ak.stock_zh_a_hist_tx(
        symbol=tx_symbol(symbol),
        start_date=format_yyyymmdd(start),
        end_date=format_yyyymmdd(end),
        adjust=ak_adjust_value(adjust),
        timeout=timeout,
    )


def fetch_daily_baostock(symbol: str, start: date, end: date, adjust: str) -> pd.DataFrame:
    bs = ensure_baostock()
    rs = bs.query_history_k_data_plus(
        baostock_symbol(symbol),
        BAOSTOCK_FIELDS,
        start_date=format_iso(start),
        end_date=format_iso(end),
        frequency="d",
        adjustflag=baostock_adjust_value(adjust),
    )
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query failed for {symbol}: {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def normalize_daily(df: pd.DataFrame, symbol: str, adjust: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    if "日期" in df.columns:
        out = df.rename(columns=COLUMN_MAP).copy()
    elif source == EASTMONEY_SOURCE:
        out = normalize_eastmoney_daily(df, symbol)
    elif source == TX_SOURCE:
        out = normalize_tx_daily(df, symbol)
    elif "date" in df.columns:
        out = normalize_baostock_daily(df)
    else:
        raise RuntimeError(f"unexpected daily columns for {symbol}: {list(df.columns)}")

    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date.astype(str)

    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            out[col] = None
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["adjust"] = adjust
    out["source"] = source
    out["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return out[BAR_COLUMNS]


def normalize_eastmoney_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = symbol
    return out


def normalize_baostock_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["trade_date"] = df["date"]
    out["symbol"] = df["code"].map(normalize_symbol)
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    out["volume"] = df["volume"]
    out["amount"] = df["amount"]
    out["turnover"] = df["turn"]
    out["pct_chg"] = df["pctChg"]
    out["amplitude"] = None
    if "preclose" in df.columns:
        out["chg"] = pd.to_numeric(df["close"], errors="coerce") - pd.to_numeric(
            df["preclose"], errors="coerce"
        )
    else:
        out["chg"] = None
    return out


def normalize_tx_daily(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["trade_date"] = df["date"]
    out["symbol"] = symbol
    out["open"] = df["open"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["close"] = df["close"]
    # AKShare's Tencent endpoint exposes this column as "amount", but values
    # match share/lot volume scale better than traded-value scale.
    out["volume"] = df["amount"]
    out["amount"] = None
    out["turnover"] = None
    out["pct_chg"] = None
    out["amplitude"] = None
    out["chg"] = None
    return out


def save_daily(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
    placeholders = ", ".join(["?"] * len(BAR_COLUMNS))
    update_columns = [col for col in BAR_COLUMNS if col not in {"symbol", "trade_date", "adjust"}]
    update_sql = ", ".join(f"{col} = excluded.{col}" for col in update_columns)

    conn.executemany(
        f"""
        INSERT INTO daily_bars ({", ".join(BAR_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(symbol, trade_date, adjust) DO UPDATE SET
            {update_sql}
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def refresh_adj_factors_from_daily(
    conn: sqlite3.Connection,
    symbol: str,
    start: date,
    end: date,
) -> int:
    """Derive qfq/hfq adjustment factors from locally stored raw/adjusted bars."""
    fetched_at = datetime.now().isoformat(timespec="seconds")
    cursor = conn.execute(
        """
        INSERT INTO adj_factors (
            symbol, trade_date, adj_factor, forward_factor, backward_factor, source, fetched_at
        )
        SELECT
            r.symbol,
            r.trade_date,
            COALESCE(h.close / NULLIF(r.close, 0), q.close / NULLIF(r.close, 0)) AS adj_factor,
            q.close / NULLIF(r.close, 0) AS forward_factor,
            h.close / NULLIF(r.close, 0) AS backward_factor,
            TRIM(
                COALESCE(q.source, '') ||
                CASE WHEN q.source IS NOT NULL AND h.source IS NOT NULL THEN '+' ELSE '' END ||
                COALESCE(h.source, '')
            ) AS source,
            ? AS fetched_at
        FROM daily_bars r
        LEFT JOIN daily_bars q
          ON q.symbol = r.symbol
         AND q.trade_date = r.trade_date
         AND q.adjust = 'qfq'
        LEFT JOIN daily_bars h
          ON h.symbol = r.symbol
         AND h.trade_date = r.trade_date
         AND h.adjust = 'hfq'
        WHERE r.symbol = ?
          AND r.adjust = 'raw'
          AND r.trade_date >= ?
          AND r.trade_date <= ?
          AND r.close IS NOT NULL
          AND r.close != 0
          AND (q.close IS NOT NULL OR h.close IS NOT NULL)
        ON CONFLICT(symbol, trade_date) DO UPDATE SET
            adj_factor = COALESCE(excluded.adj_factor, adj_factors.adj_factor),
            forward_factor = COALESCE(excluded.forward_factor, adj_factors.forward_factor),
            backward_factor = COALESCE(excluded.backward_factor, adj_factors.backward_factor),
            source = CASE
                WHEN excluded.source IS NULL OR excluded.source = '' THEN adj_factors.source
                ELSE excluded.source
            END,
            fetched_at = excluded.fetched_at
        """,
        (fetched_at, symbol, format_iso(start), format_iso(end)),
    )
    conn.commit()
    return cursor.rowcount if cursor.rowcount is not None else 0


def update_status(
    conn: sqlite3.Connection,
    symbol: str,
    adjust: str,
    start: date,
    end: date,
    status: str,
    rows: int,
    source_used: str | None = None,
    message: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO fetch_status (
            symbol, adjust, requested_start, requested_end,
            last_status, rows_fetched, source_used, message, fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, adjust) DO UPDATE SET
            requested_start = excluded.requested_start,
            requested_end = excluded.requested_end,
            last_status = excluded.last_status,
            rows_fetched = excluded.rows_fetched,
            source_used = excluded.source_used,
            message = excluded.message,
            fetched_at = excluded.fetched_at
        """,
        (
            symbol,
            adjust,
            format_iso(start),
            format_iso(end),
            status,
            rows,
            source_used,
            message[:1000],
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def requested_symbol_frame(requested: str) -> pd.DataFrame:
    symbols = [normalize_symbol(item) for item in requested.split(",") if item.strip()]
    if not symbols:
        raise RuntimeError("--symbols was provided but no valid symbols were parsed")
    return pd.DataFrame({"symbol": sorted(dict.fromkeys(symbols)), "name": ""})


def select_symbols(symbol_frame: pd.DataFrame, max_symbols: int | None) -> list[str]:
    symbols = symbol_frame["symbol"].tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]
    return symbols


def fetch_with_retries(
    symbol: str,
    start: date,
    end: date,
    adjust: str,
    provider: str,
    retries: int,
    timeout: float,
    query_timeout: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, str]:
    last_exc: Exception | None = None
    providers = ["em", "akshare", "baostock", "tx"] if provider == "auto" else [provider]
    for attempt in range(1, retries + 2):
        for current_provider in providers:
            try:
                if current_provider == "em":
                    return call_with_timeout(
                        query_timeout,
                        lambda: fetch_daily_eastmoney(symbol, start, end, adjust, timeout),
                    ), EASTMONEY_SOURCE
                if current_provider == "tx":
                    return call_with_timeout(
                        query_timeout,
                        lambda: fetch_daily_tx(symbol, start, end, adjust, timeout),
                    ), TX_SOURCE
                if current_provider == "akshare":
                    return call_with_timeout(
                        query_timeout,
                        lambda: fetch_daily_akshare(symbol, start, end, adjust, timeout),
                    ), AKSHARE_SOURCE
                if current_provider == "baostock":
                    return call_with_timeout(
                        query_timeout,
                        lambda: fetch_daily_baostock(symbol, start, end, adjust),
                    ), BAOSTOCK_SOURCE
                raise ValueError(f"unknown provider: {current_provider}")
            except Exception as exc:  # noqa: BLE001 - keep batch fetch running across vendor errors.
                last_exc = exc
                if current_provider == "baostock":
                    reset_baostock()
                if provider == "auto":
                    print(
                        f"warning: {symbol} {adjust} {current_provider} failed, trying next provider: {exc}",
                        file=sys.stderr,
                    )
        if attempt <= retries:
            time.sleep(retry_sleep * attempt)
    assert last_exc is not None
    raise last_exc


def build_jobs(
    conn: sqlite3.Connection,
    symbols: list[str],
    adjusts: list[str],
    start_date: date,
    end_date: date,
    full_refresh: bool,
    progress_every: int,
) -> tuple[list[tuple[int, int, str, str, date, date]], int]:
    total_jobs = len(symbols) * len(adjusts)
    jobs = []
    skipped = 0

    job_index = 0
    for symbol in symbols:
        for adjust in adjusts:
            job_index += 1
            effective_start = start_date
            effective_end = end_date
            if adjust in {"qfq", "hfq"}:
                raw_range = raw_trade_range(conn, symbol, start_date, end_date)
                if raw_range is None:
                    skipped += 1
                    if should_print_progress(job_index, total_jobs, progress_every):
                        print(f"[{job_index}/{total_jobs}] {symbol} {adjust}: no raw rows in requested range")
                    continue
                effective_start, effective_end = raw_range

            if full_refresh:
                ranges = [(effective_start, effective_end)]
            else:
                existing_min = min_trade_date(conn, symbol, adjust)
                existing_max = max_trade_date(conn, symbol, adjust)
                if not existing_min or not existing_max:
                    ranges = [(effective_start, effective_end)]
                else:
                    ranges = []
                    if existing_min > effective_start:
                        ranges.append((effective_start, min(effective_end, existing_min - timedelta(days=1))))
                    if existing_max < effective_end:
                        ranges.append((max(effective_start, existing_max + timedelta(days=1)), effective_end))

            valid_ranges = [
                (requested_start, requested_end)
                for requested_start, requested_end in ranges
                if requested_start <= requested_end
            ]
            if not valid_ranges:
                skipped += 1
                if should_print_progress(job_index, total_jobs, progress_every):
                    print(f"[{job_index}/{total_jobs}] {symbol} {adjust}: up to date")
                continue

            for requested_start, requested_end in valid_ranges:
                jobs.append((job_index, total_jobs, symbol, adjust, requested_start, requested_end))

    return jobs, skipped


def fetch_job(
    job: tuple[int, int, str, str, date, date],
    args: argparse.Namespace,
) -> tuple[tuple[int, int, str, str, date, date], pd.DataFrame, str]:
    _job_index, _total_jobs, symbol, adjust, requested_start, requested_end = job
    raw, source = fetch_with_retries(
        symbol=symbol,
        start=requested_start,
        end=requested_end,
        adjust=adjust,
        provider=args.provider,
        retries=args.retries,
        timeout=args.timeout,
        query_timeout=args.query_timeout,
        retry_sleep=args.retry_sleep,
    )
    normalized = normalize_daily(raw, symbol, adjust, source)
    return job, normalized, source


def run_jobs(
    conn: sqlite3.Connection,
    jobs: list[tuple[int, int, str, str, date, date]],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    if args.workers <= 1:
        return run_jobs_sequential(conn, jobs, args)
    if args.provider == "auto":
        raise ValueError("--workers > 1 is only supported with an explicit provider")
    if args.provider == "baostock":
        return run_jobs_parallel(conn, jobs, args, executor_cls=ProcessPoolExecutor)
    return run_jobs_parallel(conn, jobs, args, executor_cls=ThreadPoolExecutor)


def run_jobs_sequential(
    conn: sqlite3.Connection,
    jobs: list[tuple[int, int, str, str, date, date]],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    success = 0
    failed = 0
    rows_total = 0

    for job in jobs:
        job_index, total_jobs, symbol, adjust, requested_start, requested_end = job
        try:
            _job, normalized, source = fetch_job(job, args)
            rows = save_daily(conn, normalized)
            source_used = source or (normalized["source"].dropna().iloc[0] if not normalized.empty else None)
            if adjust in {"qfq", "hfq"} or args.adjust in {"both", "all"}:
                refresh_adj_factors_from_daily(conn, symbol, requested_start, requested_end)
            update_status(
                conn,
                symbol,
                adjust,
                requested_start,
                requested_end,
                "ok",
                rows,
                source_used,
                f"source={source_used}" if source_used else "",
            )
            success += 1
            rows_total += rows
            if should_print_progress(job_index, total_jobs, args.progress_every):
                print(
                    f"[{job_index}/{total_jobs}] {symbol} {adjust} "
                    f"{format_iso(requested_start)}..{format_iso(requested_end)}: "
                    f"saved {rows} rows source={source_used or 'none'}"
                )
        except Exception as exc:  # noqa: BLE001 - record and continue batch.
            failed += 1
            update_status(conn, symbol, adjust, requested_start, requested_end, "failed", 0, None, str(exc))
            print(f"[{job_index}/{total_jobs}] {symbol} {adjust}: failed: {exc}", file=sys.stderr)
            if args.stop_on_error:
                break
        time.sleep(args.sleep)

    return success, failed, rows_total


def run_jobs_parallel(
    conn: sqlite3.Connection,
    jobs: list[tuple[int, int, str, str, date, date]],
    args: argparse.Namespace,
    executor_cls,
) -> tuple[int, int, int]:
    success = 0
    failed = 0
    rows_total = 0

    with executor_cls(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            job_index, total_jobs, symbol, adjust, requested_start, requested_end = futures[future]
            try:
                _job, normalized, source = future.result()
                rows = save_daily(conn, normalized)
                source_used = source or (normalized["source"].dropna().iloc[0] if not normalized.empty else None)
                if adjust in {"qfq", "hfq"} or args.adjust in {"both", "all"}:
                    refresh_adj_factors_from_daily(conn, symbol, requested_start, requested_end)
                update_status(
                    conn,
                    symbol,
                    adjust,
                    requested_start,
                    requested_end,
                    "ok",
                    rows,
                    source_used,
                    f"source={source_used}" if source_used else "",
                )
                success += 1
                rows_total += rows
                if should_print_progress(job_index, total_jobs, args.progress_every):
                    print(
                        f"[{job_index}/{total_jobs}] {symbol} {adjust} "
                        f"{format_iso(requested_start)}..{format_iso(requested_end)}: "
                        f"saved {rows} rows source={source_used or 'none'}"
                    )
            except Exception as exc:  # noqa: BLE001 - record and continue batch.
                failed += 1
                update_status(conn, symbol, adjust, requested_start, requested_end, "failed", 0, None, str(exc))
                print(f"[{job_index}/{total_jobs}] {symbol} {adjust}: failed: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    for pending in futures:
                        pending.cancel()
                    break

    return success, failed, rows_total


def run(args: argparse.Namespace) -> int:
    start_date = args.start_date
    end_date = args.end_date
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")

    conn = connect(args.db)
    init_db(conn)

    if args.symbols:
        print("using explicitly requested symbols; skipping full A-share symbol-list fetch...")
        symbol_frame = requested_symbol_frame(args.symbols)
    else:
        print("fetching A-share symbol list from AKShare...")
        symbol_frame = fetch_symbols(args.retries, args.retry_sleep)
    save_symbols(conn, symbol_frame)
    symbols = select_symbols(symbol_frame, args.max_symbols)
    if not symbols:
        raise RuntimeError("no symbols selected")

    adjusts = adjust_values(args.adjust)
    print(
        f"selected {len(symbols)} symbols, adjusts={','.join(adjusts)}, "
        f"range={format_iso(start_date)}..{format_iso(end_date)}, "
        f"provider={args.provider}, workers={args.workers}, db={args.db}"
    )

    jobs, skipped = build_jobs(
        conn=conn,
        symbols=symbols,
        adjusts=adjusts,
        start_date=start_date,
        end_date=end_date,
        full_refresh=args.full_refresh,
        progress_every=args.progress_every,
    )

    success, failed, rows_total = run_jobs(conn, jobs, args)

    logout_baostock()
    conn.close()
    print(
        f"done: success={success}, skipped={skipped}, failed={failed}, rows_saved={rows_total}, db={args.db}"
    )
    return 1 if failed else 0


def should_print_progress(job_index: int, total_jobs: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return job_index == 1 or job_index == total_jobs or job_index % progress_every == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch free A-share daily K-line history from AKShare into SQLite."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/stock_daily.sqlite3"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_yyyymmdd,
        default=parse_yyyymmdd("19900101"),
        help="Fetch start date, YYYYMMDD or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_yyyymmdd,
        default=date.today(),
        help="Fetch end date, YYYYMMDD or YYYY-MM-DD.",
    )
    parser.add_argument(
        "--adjust",
        choices=["raw", "qfq", "hfq", "both", "all"],
        default="raw",
        help="Price adjustment mode. both=raw+hfq; all=raw+qfq+hfq.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "em", "tx", "akshare", "baostock"],
        default="auto",
        help="Daily-bar provider. auto tries Eastmoney, Tencent, BaoStock, then AKShare.",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols to fetch, e.g. 000001,600519. Default: all current A-shares.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        help="Limit number of symbols for smoke tests.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Fetch from start-date even when local rows already exist.",
    )
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between API calls.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per symbol/adjust.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Base sleep seconds between retries.")
    parser.add_argument("--timeout", type=float, default=15.0, help="AKShare request timeout seconds when supported.")
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=120,
        help="Hard timeout seconds for one symbol/adjust vendor request. Use 0 to disable.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop batch on first failed symbol.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent fetch workers. Use with --provider em, tx, or akshare; SQLite writes stay single-threaded.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print one success/skip progress line every N jobs; failures are always printed. Use 0 for quiet.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

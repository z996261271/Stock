#!/usr/bin/env python3
"""Fetch THS industry index daily bars into the local SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd
import requests
from py_mini_racer import py_mini_racer

from akshare.stock_feature.stock_board_industry_ths import _get_file_content_ths
from akshare.utils import demjson


@dataclass(frozen=True)
class FetchConfig:
    db: Path
    start_date: str
    end_date: str
    workers: int
    retries: int
    timeout: int
    sleep: float


def parse_args() -> FetchConfig:
    parser = argparse.ArgumentParser(description="Fetch THS industry index daily bars.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()
    return FetchConfig(
        db=args.db,
        start_date=args.start_date,
        end_date=args.end_date,
        workers=args.workers,
        retries=args.retries,
        timeout=args.timeout,
        sleep=args.sleep,
    )


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS industry_boards (
            board_code TEXT PRIMARY KEY,
            board_name TEXT NOT NULL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS industry_daily_bars (
            board_code TEXT NOT NULL,
            board_name TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (board_code, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_industry_daily_bars_date
            ON industry_daily_bars (trade_date);
        """
    )


def ths_cookie_v() -> str:
    js_code = py_mini_racer.MiniRacer()
    js_code.eval(_get_file_content_ths("ths.js"))
    return js_code.call("v")


def fetch_year(code: str, year: int, timeout: int, retries: int) -> pd.DataFrame:
    url = f"https://d.10jqka.com.cn/v4/line/bk_{code}/01/{year}.js"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            v_code = ths_cookie_v()
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36"
                ),
                "Referer": "http://q.10jqka.com.cn",
                "Host": "d.10jqka.com.cn",
                "Cookie": f"v={v_code}",
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            text = response.text
            if "data" not in text:
                return pd.DataFrame()
            payload = demjson.decode(text[text.find("{") : -1])
            if not payload.get("data"):
                return pd.DataFrame()
            frame = pd.DataFrame(payload["data"].split(";"))[0].str.split(",", expand=True)
            frame = frame.iloc[:, :7]
            frame.columns = ["trade_date", "open", "high", "low", "close", "volume", "amount"]
            return frame
        except Exception as exc:  # noqa: BLE001 - vendor endpoint is unstable.
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to fetch {code} {year}: {last_error}")


def fetch_board(row: dict, config: FetchConfig) -> tuple[dict, pd.DataFrame, str | None]:
    board_name = str(row["name"])
    board_code = str(row["code"])
    begin_year = int(config.start_date[:4])
    end_year = int(config.end_date[:4])
    frames = []
    try:
        for year in range(begin_year, end_year + 1):
            frames.append(fetch_year(board_code, year, config.timeout, config.retries))
            if config.sleep:
                time.sleep(config.sleep)
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if data.empty:
            return row, data, "empty"
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data = data.dropna(subset=["trade_date"])
        start = pd.to_datetime(config.start_date)
        end = pd.to_datetime(config.end_date)
        data = data[data["trade_date"].between(start, end)]
        for column in ["open", "high", "low", "close", "volume", "amount"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["close"])
        data["board_code"] = board_code
        data["board_name"] = board_name
        return row, data, None
    except Exception as exc:  # noqa: BLE001 - keep other boards moving.
        return row, pd.DataFrame(), str(exc)


def upsert(conn: sqlite3.Connection, boards: pd.DataFrame, bars: pd.DataFrame) -> None:
    fetched_at = datetime.now().isoformat(timespec="seconds")
    board_rows = [
        (str(row.code), str(row.name), "ths.industry", fetched_at)
        for row in boards.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO industry_boards(board_code, board_name, source, fetched_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(board_code) DO UPDATE SET
            board_name=excluded.board_name,
            source=excluded.source,
            fetched_at=excluded.fetched_at
        """,
        board_rows,
    )
    if not bars.empty:
        rows = [
            (
                row.board_code,
                row.board_name,
                row.trade_date.strftime("%Y-%m-%d"),
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
                row.amount,
                "ths.industry.index",
                fetched_at,
            )
            for row in bars.itertuples(index=False)
        ]
        conn.executemany(
            """
            INSERT INTO industry_daily_bars(
                board_code, board_name, trade_date, open, high, low, close,
                volume, amount, source, fetched_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(board_code, trade_date) DO UPDATE SET
                board_name=excluded.board_name,
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                amount=excluded.amount,
                source=excluded.source,
                fetched_at=excluded.fetched_at
            """,
            rows,
        )
    conn.commit()


def main() -> int:
    config = parse_args()
    boards = ak.stock_board_industry_name_ths()
    boards["code"] = boards["code"].astype(str)
    all_bars = []
    errors = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [
            executor.submit(fetch_board, row._asdict(), config)
            for row in boards.itertuples(index=False)
        ]
        for index, future in enumerate(as_completed(futures), 1):
            board, bars, error = future.result()
            name = board["name"]
            code = board["code"]
            if error:
                errors.append((code, name, error))
                print(f"[{index}/{len(futures)}] {code} {name}: {error}")
            else:
                all_bars.append(bars)
                print(f"[{index}/{len(futures)}] {code} {name}: {len(bars)} rows")

    bars = pd.concat(all_bars, ignore_index=True) if all_bars else pd.DataFrame()
    with connect(config.db) as conn:
        init_db(conn)
        upsert(conn, boards, bars)

    print(
        {
            "boards": int(len(boards)),
            "bars": int(len(bars)),
            "errors": len(errors),
            "db": str(config.db),
        }
    )
    return 0 if len(bars) else 1


if __name__ == "__main__":
    raise SystemExit(main())

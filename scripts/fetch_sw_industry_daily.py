#!/usr/bin/env python3
"""Fetch SW industry index daily bars into the local SQLite database."""

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


@dataclass(frozen=True)
class FetchConfig:
    db: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    levels: list[str]
    workers: int
    retries: int
    sleep: float


def parse_ymd(value: str) -> pd.Timestamp:
    normalized = value.strip().replace("-", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise ValueError("date must be YYYY-MM-DD or YYYYMMDD")
    return pd.Timestamp(datetime.strptime(normalized, "%Y%m%d").date())


def parse_args() -> FetchConfig:
    parser = argparse.ArgumentParser(description="Fetch SW industry index daily bars.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--start-date", default="20060105")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument(
        "--levels",
        default="一级行业",
        help="comma separated SW levels, e.g. 一级行业 or 一级行业,二级行业",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()
    return FetchConfig(
        db=args.db,
        start_date=parse_ymd(args.start_date),
        end_date=parse_ymd(args.end_date),
        levels=[level.strip() for level in args.levels.split(",") if level.strip()],
        workers=max(args.workers, 1),
        retries=max(args.retries, 1),
        sleep=max(args.sleep, 0.0),
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
        CREATE INDEX IF NOT EXISTS idx_industry_daily_bars_source_date
            ON industry_daily_bars (source, trade_date);
        """
    )


def load_boards(levels: list[str]) -> pd.DataFrame:
    frames = []
    for level in levels:
        boards = ak.index_realtime_sw(symbol=level)
        if boards.empty:
            continue
        boards = boards[["指数代码", "指数名称"]].rename(
            columns={"指数代码": "board_code", "指数名称": "board_name"}
        )
        boards["board_code"] = boards["board_code"].astype(str)
        boards["board_name"] = boards["board_name"].astype(str)
        boards["level"] = level
        frames.append(boards)
    if not frames:
        raise RuntimeError("no SW industry boards loaded")
    return pd.concat(frames, ignore_index=True).drop_duplicates("board_code")


def fetch_board(row: dict, config: FetchConfig) -> tuple[dict, pd.DataFrame, str | None]:
    code = str(row["board_code"])
    name = str(row["board_name"])
    last_error: Exception | None = None
    for attempt in range(config.retries):
        try:
            data = ak.index_hist_sw(symbol=code, period="day")
            if data.empty:
                return row, data, "empty"
            data = data.rename(
                columns={
                    "日期": "trade_date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
            data = data.dropna(subset=["trade_date"])
            data = data[data["trade_date"].between(config.start_date, config.end_date)]
            for column in ["open", "high", "low", "close", "volume", "amount"]:
                data[column] = pd.to_numeric(data[column], errors="coerce")
            data = data.dropna(subset=["close"])
            data["board_code"] = code
            data["board_name"] = name
            data["level"] = row["level"]
            return row, data, None
        except Exception as exc:  # noqa: BLE001 - vendor endpoint is unstable.
            last_error = exc
            time.sleep(1 + attempt)
    return row, pd.DataFrame(), str(last_error)


def upsert(conn: sqlite3.Connection, boards: pd.DataFrame, bars: pd.DataFrame) -> None:
    fetched_at = datetime.now().isoformat(timespec="seconds")
    board_rows = [
        (
            str(row.board_code),
            str(row.board_name),
            f"sw.industry.{row.level}",
            fetched_at,
        )
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
    if bars.empty:
        conn.commit()
        return

    rows = []
    for row in bars.to_dict("records"):
        rows.append(
            (
                row["board_code"],
                row["board_name"],
                row["trade_date"].strftime("%Y-%m-%d"),
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
                row["amount"],
                f"sw.industry.index.{row['level']}",
                fetched_at,
            )
        )
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
    boards = load_boards(config.levels)
    all_bars = []
    errors = []
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(fetch_board, row._asdict(), config) for row in boards.itertuples(index=False)]
        for index, future in enumerate(as_completed(futures), 1):
            board, bars, error = future.result()
            code = board["board_code"]
            name = board["board_name"]
            if error:
                errors.append((code, name, error))
                print(f"[{index}/{len(futures)}] {code} {name}: {error}", flush=True)
            else:
                all_bars.append(bars)
                print(f"[{index}/{len(futures)}] {code} {name}: {len(bars)} rows", flush=True)
            if config.sleep:
                time.sleep(config.sleep)

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
            "start_date": config.start_date.strftime("%Y-%m-%d"),
            "end_date": config.end_date.strftime("%Y-%m-%d"),
            "levels": config.levels,
        },
        flush=True,
    )
    return 0 if len(bars) else 1


if __name__ == "__main__":
    raise SystemExit(main())

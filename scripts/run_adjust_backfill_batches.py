#!/usr/bin/env python3
"""Run qfq/hfq backfill in resumable symbol batches."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROGRESS_RE = re.compile(r"^\[(?P<done>\d+)/(?P<total>\d+)\]\s+(?P<symbol>\S+)\s+(?P<adjust>\S+)\s+(?P<message>.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for adjusted daily-bar backfill.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--adjust", choices=["qfq", "hfq"], required=True)
    parser.add_argument("--provider", choices=["em", "akshare", "baostock", "tx"], default="baostock")
    parser.add_argument("--start-date", default="20060101")
    parser.add_argument("--end-date", default="20260529")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--query-timeout", type=int, default=180)
    parser.add_argument("--max-batches", type=int, help="limit batches for smoke tests")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="child fetch progress interval; 1 refreshes every symbol",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iso_date(value: str) -> str:
    value = value.replace("-", "")
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def remaining_symbols(db: Path, adjust: str, start_date: str, end_date: str) -> list[str]:
    start_iso = iso_date(start_date)
    end_iso = iso_date(end_date)
    with sqlite3.connect(db) as conn:
        return [
            row[0]
            for row in conn.execute(
                """
                with raw_coverage as (
                    select
                        symbol,
                        min(trade_date) as raw_min,
                        max(trade_date) as raw_max,
                        count(*) as raw_count
                    from daily_bars
                    where adjust = 'raw'
                      and trade_date >= ?
                      and trade_date <= ?
                    group by symbol
                ),
                adjusted_coverage as (
                    select
                        symbol,
                        min(trade_date) as adjusted_min,
                        max(trade_date) as adjusted_max,
                        count(*) as adjusted_count
                    from daily_bars
                    where adjust = ?
                      and trade_date >= ?
                      and trade_date <= ?
                    group by symbol
                )
                select r.symbol
                from raw_coverage r
                left join adjusted_coverage a on a.symbol = r.symbol
                where coalesce(a.adjusted_count, 0) < r.raw_count
                   or a.adjusted_min > r.raw_min
                   or a.adjusted_max < r.raw_max
                order by r.symbol
                """,
                (start_iso, end_iso, adjust, start_iso, end_iso),
            )
        ]


def render_live_progress(
    *,
    batch_index: int,
    batch_count: int,
    previous_symbols: int,
    total_symbols: int,
    child_done: int,
    child_total: int,
    symbol: str,
    adjust: str,
    message: str,
) -> None:
    overall_done = min(total_symbols, previous_symbols + child_done)
    percent = (overall_done / total_symbols * 100) if total_symbols else 100.0
    status = (
        f"batch {batch_index}/{batch_count} | "
        f"symbols {overall_done}/{total_symbols} ({percent:5.1f}%) | "
        f"current {child_done}/{child_total} {symbol} {adjust} {message}"
    )
    sys.stdout.write("\r\033[K" + status[:220])
    sys.stdout.flush()


def run_child_with_live_progress(
    command: list[str],
    batch_log: Path,
    *,
    batch_index: int,
    batch_count: int,
    previous_symbols: int,
    total_symbols: int,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with batch_log.open("w", encoding="utf-8") as log_fh:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_fh.write(line)
            log_fh.flush()
            clean = line.rstrip()
            match = PROGRESS_RE.match(clean)
            if match:
                render_live_progress(
                    batch_index=batch_index,
                    batch_count=batch_count,
                    previous_symbols=previous_symbols,
                    total_symbols=total_symbols,
                    child_done=int(match.group("done")),
                    child_total=int(match.group("total")),
                    symbol=match.group("symbol"),
                    adjust=match.group("adjust"),
                    message=match.group("message"),
                )
            elif "failed:" in clean or clean.startswith("done:"):
                sys.stdout.write("\r\033[K" + clean[:220] + "\n")
                sys.stdout.flush()
        return process.wait()


def run(args: argparse.Namespace) -> int:
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    summary_path = args.logs_dir / f"adjust_backfill_{args.adjust}_{timestamp}.jsonl"
    symbols = remaining_symbols(args.db, args.adjust, args.start_date, args.end_date)
    batches = list(chunks(symbols, args.batch_size))
    if args.max_batches:
        batches = batches[: args.max_batches]
    header = {
        "event": "start",
        "adjust": args.adjust,
        "provider": args.provider,
        "remaining_symbols": len(symbols),
        "planned_batches": len(batches),
        "batch_size": args.batch_size,
        "db": str(args.db),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(header, ensure_ascii=False))
    summary_path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        return 0

    failures = 0
    completed_symbols = 0
    for index, batch in enumerate(batches, start=1):
        batch_log = args.logs_dir / f"adjust_backfill_{args.adjust}_{timestamp}_batch{index:04d}.log"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "fetch_akshare_daily.py"),
            "--db",
            str(args.db),
            "--provider",
            args.provider,
            "--symbols",
            ",".join(batch),
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--adjust",
            args.adjust,
            "--workers",
            str(args.workers),
            "--retries",
            str(args.retries),
            "--retry-sleep",
            str(args.retry_sleep),
            "--query-timeout",
            str(args.query_timeout),
            "--progress-every",
            str(args.progress_every),
        ]
        started_at = datetime.now()
        print(
            f"starting batch {index}/{len(batches)}: {batch[0]}..{batch[-1]} "
            f"({len(batch)} symbols), log={batch_log}",
            flush=True,
        )
        returncode = run_child_with_live_progress(
            command,
            batch_log,
            batch_index=index,
            batch_count=len(batches),
            previous_symbols=completed_symbols,
            total_symbols=len(symbols),
        )
        print()
        event = {
            "event": "batch",
            "batch": index,
            "batches": len(batches),
            "symbols": len(batch),
            "first_symbol": batch[0],
            "last_symbol": batch[-1],
            "returncode": returncode,
            "seconds": round((datetime.now() - started_at).total_seconds(), 2),
            "log": str(batch_log),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        completed_symbols += len(batch)
        if returncode != 0:
            failures += 1
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event, ensure_ascii=False), flush=True)
    done = {
        "event": "done",
        "adjust": args.adjust,
        "batches": len(batches),
        "failed_batches": failures,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "summary": str(summary_path),
    }
    with summary_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(done, ensure_ascii=False) + "\n")
    print(json.dumps(done, ensure_ascii=False))
    return 1 if failures else 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

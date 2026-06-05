#!/usr/bin/env python3
"""Run the local quality gate for research-framework changes."""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COMPILE_TARGETS = sorted(
    str(path.relative_to(ROOT))
    for folder in ("scripts", "tests", "src")
    for path in (ROOT / folder).rglob("*.py")
)
TEST_TARGETS = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py"))


def run_command(command: list[str], *, label: str) -> None:
    print(f"[quality] {label}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def insert_bar(
    conn: sqlite3.Connection,
    symbol: str,
    trade_date: str,
    adjust: str,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> None:
    conn.execute(
        """
        insert into daily_bars (
            symbol, trade_date, adjust, open, high, low, close, volume, amount,
            amplitude, pct_chg, chg, turnover, source, fetched_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            trade_date,
            adjust,
            open_,
            high,
            low,
            close,
            1_000_000,
            100_000_000,
            0.0,
            0.0,
            0.0,
            1.0,
            "quality_gate",
            "2020-01-03T18:00:00",
        ),
    )


def build_quality_smoke_db(db: Path) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            create table daily_bars (
                symbol text not null,
                trade_date text not null,
                adjust text not null,
                open real, high real, low real, close real, volume real, amount real,
                amplitude real, pct_chg real, chg real, turnover real,
                source text not null, fetched_at text not null,
                primary key(symbol, trade_date, adjust)
            )
            """
        )
        conn.execute("create table symbols (symbol text primary key, name text)")
        conn.execute("insert into symbols values ('000001', 'quality-smoke')")
        for adjust, multiplier in (("raw", 1.0), ("qfq", 1.1), ("hfq", 1.2)):
            insert_bar(conn, "000001", "2020-01-02", adjust, 10 * multiplier, 11 * multiplier, 9 * multiplier, 10.5 * multiplier)
            insert_bar(conn, "000001", "2020-01-03", adjust, 10.5 * multiplier, 12 * multiplier, 10 * multiplier, 11 * multiplier)
        conn.commit()
    finally:
        conn.close()

    run_command(
        [
            PYTHON,
            "scripts/data_quality_report.py",
            "--db",
            str(db),
            "--init-schema",
            "--start-date",
            "2020-01-02",
            "--end-date",
            "2020-01-03",
            "--board-scope",
            "main",
        ],
        label="schema bootstrap",
    )

    conn = sqlite3.connect(db)
    try:
        conn.execute(
            """
            insert into adj_factors (
                symbol, trade_date, adj_factor, forward_factor, backward_factor, source, fetched_at
            )
            values
                ('000001', '2020-01-02', 1.0, 1.0, 1.0, 'quality_gate', '2020-01-03T18:00:00'),
                ('000001', '2020-01-03', 1.0, 1.0, 1.0, 'quality_gate', '2020-01-03T18:00:00')
            """
        )
        conn.execute(
            """
            insert into symbol_lifecycle (symbol, name, list_date, board, market, source, fetched_at)
            values ('000001', 'quality-smoke', '1991-04-03', 'main', 'SZ', 'quality_gate', '2020-01-03T18:00:00')
            """
        )
        conn.execute(
            """
            insert into symbol_status_daily (symbol, trade_date, is_st, is_suspended, board, source, fetched_at)
            values
                ('000001', '2020-01-02', 0, 0, 'main', 'quality_gate', '2020-01-03T18:00:00'),
                ('000001', '2020-01-03', 0, 0, 'main', 'quality_gate', '2020-01-03T18:00:00')
            """
        )
        conn.execute(
            """
            insert into symbol_industries (
                symbol, industry_name, industry_code, industry_level, provider, source, fetched_at
            )
            values ('000001', '银行', '801780', '一级行业', 'quality_gate', 'quality_gate', '2020-01-03T18:00:00')
            """
        )
        conn.commit()
    finally:
        conn.close()


def run_data_quality_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="stock-quality-") as directory:
        db = Path(directory) / "quality.sqlite3"
        build_quality_smoke_db(db)
        run_command(
            [
                PYTHON,
                "scripts/data_quality_report.py",
                "--db",
                str(db),
                "--start-date",
                "2020-01-02",
                "--end-date",
                "2020-01-03",
                "--board-scope",
                "main",
                "--fail-on-red-flag",
            ],
            label="data quality smoke",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local quality checks without touching the production database.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="only run compile checks and the data-quality smoke gate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_command([PYTHON, "-m", "py_compile", *COMPILE_TARGETS], label="compile")
    if importlib.util.find_spec("ruff") is not None:
        run_command([PYTHON, "-m", "ruff", "check", "scripts", "src", "tests"], label="ruff")
    else:
        print("[quality] ruff: skipped; module is not installed in this environment", flush=True)
    if not args.skip_tests:
        for target in TEST_TARGETS:
            run_command([PYTHON, target], label=f"test {target}")
    run_data_quality_smoke()
    print("[quality] passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

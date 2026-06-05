#!/usr/bin/env python3
"""Generate a data-credibility report for the local A-share research database."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_data_quality import build_quality_report
from quant_schema import CANONICAL_ADJUSTS, ensure_quant_schema_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JSON data quality report for stock_daily.sqlite3.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--board-scope", choices=["main", "all"], default="main")
    parser.add_argument(
        "--required-adjust",
        action="append",
        choices=CANONICAL_ADJUSTS,
        help="required adjustment stream; repeatable. Defaults to raw/qfq/hfq.",
    )
    parser.add_argument("--output", type=Path, help="write report JSON to this path")
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="create missing reference/state tables before reporting (does not backfill data)",
    )
    parser.add_argument(
        "--fail-on-red-flag",
        action="store_true",
        help="exit non-zero when red_flags is not empty",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.init_schema:
        ensure_quant_schema_path(args.db)

    if args.end_date:
        end_date = pd.Timestamp(args.end_date)
    else:
        end_date = pd.Timestamp(datetime.now().date())
    report = build_quality_report(
        args.db,
        pd.Timestamp(args.start_date),
        end_date,
        args.board_scope,
        tuple(args.required_adjust or CANONICAL_ADJUSTS),
    )
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 2 if args.fail_on_red_flag and report.get("red_flags") else 0


if __name__ == "__main__":
    raise SystemExit(main())

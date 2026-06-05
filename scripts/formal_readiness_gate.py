#!/usr/bin/env python3
"""Hard gate for publishing formal backtest artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_data_quality import build_quality_report  # noqa: E402
from quant_schema import CANONICAL_ADJUSTS, ensure_quant_schema_path  # noqa: E402
from validate_formal_reports import build_report as build_formal_report_validation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail unless data quality and formal report artifacts are publishable.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--board-scope", choices=["main", "all"], default="main")
    parser.add_argument(
        "--required-adjust",
        action="append",
        choices=CANONICAL_ADJUSTS,
        help="required adjustment stream; repeatable. Defaults to raw/qfq/hfq.",
    )
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/formal"))
    parser.add_argument("--allow-empty-reports", action="store_true")
    parser.add_argument("--quarantine-invalid-reports", action="store_true")
    parser.add_argument("--init-schema", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_readiness_report(
    *,
    db: Path,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    board_scope: str,
    required_adjusts: tuple[str, ...],
    reports_dir: Path,
    allow_empty_reports: bool = False,
    quarantine_invalid_reports: bool = False,
) -> dict[str, Any]:
    data_quality = build_quality_report(
        db,
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
        board_scope,
        required_adjusts,
    )
    formal_reports = build_formal_report_validation(
        reports_dir,
        require_formal_valid=True,
        allow_empty=allow_empty_reports,
        quarantine_invalid=quarantine_invalid_reports,
    )
    blockers: list[str] = []
    for flag in data_quality.get("red_flags", []):
        blockers.append(f"data_quality:{flag}")
    if not formal_reports["is_valid"]:
        blockers.append(
            f"formal_reports:issue_files={formal_reports['issue_files']}:metrics_files={formal_reports['metrics_files']}"
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "is_ready": not blockers,
        "blockers": blockers,
        "data_quality": data_quality,
        "formal_reports": formal_reports,
        "publish_contract": {
            "required_adjusts": list(required_adjusts),
            "required_reference_coverage": ["symbol_lifecycle", "symbol_status_daily", "symbol_industries"],
            "required_report_fields": "validate_formal_reports.REQUIRED_METRIC_FIELDS",
        },
    }


def main() -> int:
    args = parse_args()
    if args.init_schema:
        ensure_quant_schema_path(args.db)
    report = build_readiness_report(
        db=args.db,
        start_date=args.start_date,
        end_date=args.end_date,
        board_scope=args.board_scope,
        required_adjusts=tuple(args.required_adjust or CANONICAL_ADJUSTS),
        reports_dir=args.reports_dir,
        allow_empty_reports=args.allow_empty_reports,
        quarantine_invalid_reports=args.quarantine_invalid_reports,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["is_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

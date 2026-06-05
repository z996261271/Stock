#!/usr/bin/env python3
"""Import stock-to-industry mappings into the canonical local table."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from professional_quant.data.industry import (  # noqa: E402
    ensure_symbol_industry_schema,
    load_industry_rows,
    upsert_symbol_industries,
)
from professional_quant.data.providers.base import provider_success  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import symbol industry mappings from CSV/JSON.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--input", type=Path, required=True, help="CSV/JSON with symbol and industry_name/industry columns")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--source", default="manual_import")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_industry_rows(args.input, args.provider, args.source)
    output = {
        "db": str(args.db),
        "input": str(args.input),
        "provider": args.provider,
        "source": args.source,
        "rows_loaded": len(rows),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        output["sample"] = rows[:5]
        result_rows = len(rows)
    else:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(args.db) as conn:
            ensure_symbol_industry_schema(conn)
            result_rows = upsert_symbol_industries(conn, rows)
            output["rows_upserted"] = result_rows
    output["provider_result"] = provider_success(
        provider=args.provider,
        dataset="symbol_industries",
        rows=result_rows,
        start_date=None,
        end_date=None,
        metadata={
            "source": args.source,
            "input": str(args.input),
            "dry_run": bool(args.dry_run),
            "rows_loaded": len(rows),
        },
        maturity="stable" if args.provider == "local" else "beta",
    ).to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

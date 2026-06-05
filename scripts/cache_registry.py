#!/usr/bin/env python3
"""Build a lightweight registry for local cache files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from professional_quant.data.cache_registry import scan_cache_dir, write_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan cache files and write cache_registry.v1 JSON.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--script", type=Path, default=Path("scripts/backtest_dynamic_rebalance.py"))
    parser.add_argument("--output", type=Path, default=Path("data/cache/cache_registry.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = scan_cache_dir(args.cache_dir, db=args.db, script_path=args.script)
    registry = write_registry(entries, args.output)
    print(json.dumps(registry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

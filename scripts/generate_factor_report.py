#!/usr/bin/env python3
"""Generate an alphalens-style factor report from formal picks artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from professional_quant.backtest.attribution import read_csv_optional  # noqa: E402
from professional_quant.factor.analysis import build_factor_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build factor IC, quantile, turnover, and industry diagnostics.")
    parser.add_argument("--picks", type=Path, required=True)
    parser.add_argument("--equity", type=Path)
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--return-col")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_factor_report(
        read_csv_optional(args.picks),
        equity=read_csv_optional(args.equity),
        score_col=args.score_col,
        return_col=args.return_col,
        quantiles=args.quantiles,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

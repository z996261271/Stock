#!/usr/bin/env python3
"""Build an attribution report for a formal dynamic backtest family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from professional_quant.backtest.attribution import (  # noqa: E402
    build_formal_attribution_report,
    load_json,
    read_csv_optional,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain formal strategy return drivers and execution friction.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--equity", type=Path)
    parser.add_argument("--picks", type=Path)
    parser.add_argument("--trades", type=Path)
    parser.add_argument("--capacity-stress", type=Path)
    parser.add_argument("--sensitivity-report", type=Path)
    parser.add_argument("--benchmark", default="main_board_equal_weight_raw_close")
    parser.add_argument(
        "--min-capacity-annual-return",
        type=float,
        default=0.0,
        help="annual-return threshold used for the conservative capital-limit grid point",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def infer_family_paths(metrics: Path, args: argparse.Namespace) -> dict[str, Path | None]:
    prefix = _metrics_prefix(metrics)
    return {
        "equity": args.equity or prefix.with_suffix(".equity.csv"),
        "picks": args.picks or prefix.with_suffix(".picks.csv"),
        "trades": args.trades or prefix.with_suffix(".trades.csv"),
        "capacity": args.capacity_stress or prefix.with_suffix(".capacity_stress.csv"),
    }


def _metrics_prefix(metrics: Path) -> Path:
    name = metrics.name
    if name.endswith(".metrics.json"):
        return metrics.with_name(name.removesuffix(".metrics.json"))
    return metrics.with_suffix("")


def _optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return load_json(path)


def main() -> int:
    args = parse_args()
    paths = infer_family_paths(args.metrics, args)
    metrics = load_json(args.metrics)
    report = build_formal_attribution_report(
        metrics=metrics,
        equity=read_csv_optional(paths["equity"]),
        picks=read_csv_optional(paths["picks"]),
        trades=read_csv_optional(paths["trades"]),
        capacity=read_csv_optional(paths["capacity"]),
        benchmark_name=args.benchmark,
        min_capacity_annual_return=args.min_capacity_annual_return,
        sensitivity_report=_optional_json(args.sensitivity_report),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

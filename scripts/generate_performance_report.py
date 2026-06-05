#!/usr/bin/env python3
"""Generate JSON and Markdown performance tear sheets from formal artifacts."""

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

from professional_quant.backtest.attribution import load_json, read_csv_optional  # noqa: E402
from professional_quant.reporting.performance import build_performance_report, render_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pyfolio-style performance tear sheet artifacts.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--equity", type=Path)
    parser.add_argument("--trades", type=Path)
    parser.add_argument("--benchmark", default="main_board_equal_weight_raw_close")
    parser.add_argument("--initial-cash", type=float)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def infer_family_paths(metrics: Path, args: argparse.Namespace) -> dict[str, Path | None]:
    prefix = _metrics_prefix(metrics)
    return {
        "equity": args.equity or prefix.with_suffix(".equity.csv"),
        "trades": args.trades or prefix.with_suffix(".trades.csv"),
    }


def _metrics_prefix(metrics: Path) -> Path:
    name = metrics.name
    if name.endswith(".metrics.json"):
        return metrics.with_name(name.removesuffix(".metrics.json"))
    return metrics.with_suffix("")


def main() -> int:
    args = parse_args()
    paths = infer_family_paths(args.metrics, args)
    report = build_performance_report(
        metrics=load_json(args.metrics),
        equity=read_csv_optional(paths["equity"]),
        trades=read_csv_optional(paths["trades"]),
        benchmark_name=args.benchmark,
        initial_cash=args.initial_cash,
    )
    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    markdown_text = render_markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_text, encoding="utf-8")
    print(json_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

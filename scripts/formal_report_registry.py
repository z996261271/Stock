#!/usr/bin/env python3
"""Build a release registry for formal report artifacts."""

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

from professional_quant.reporting.registry import build_formal_release_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind formal report artifacts to stable release IDs.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/formal"))
    parser.add_argument("--release-id", help="optional fixed release id for every discovered report family")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = build_formal_release_registry(
        args.reports_dir,
        release_id=args.release_id,
        include_invalid=args.include_invalid,
    )
    text = json.dumps(registry, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if all(row["is_publishable"] for row in registry["releases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())

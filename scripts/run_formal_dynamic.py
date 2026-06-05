#!/usr/bin/env python3
"""Run the formal dynamic backtest from a versioned JSON config."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
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
from professional_quant.risk.defaults import apply_formal_risk_defaults_to_mapping  # noqa: E402

METADATA_ARG_MAP = {
    "split_policy": "--split-policy-json",
    "frozen_config": "--frozen-config-json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal dynamic backtest wrapper.")
    parser.add_argument("--config", type=Path, default=Path("configs/formal_dynamic_default.json"))
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--dry-run", action="store_true", help="print command without executing")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("factor_adjust") == "raw":
        raise ValueError("formal config must use qfq or hfq factor_adjust, not raw")
    if not data.get("strict_factor_adjust", False):
        raise ValueError("formal config must set strict_factor_adjust=true")
    if not data.get("formal", False):
        raise ValueError("formal config must set formal=true")
    if not isinstance(data.get("split_policy"), dict):
        raise ValueError("formal config must include split_policy object")
    if not isinstance(data.get("frozen_config"), dict):
        raise ValueError("formal config must include frozen_config object")
    if not data["frozen_config"].get("frozen", False):
        raise ValueError("formal config frozen_config.frozen must be true")
    if not data.get("freeze_selection_date"):
        frozen_date = data["frozen_config"].get("freeze_selection_date")
        if frozen_date:
            data["freeze_selection_date"] = frozen_date
    if not data.get("freeze_selection_date"):
        raise ValueError("formal config must include freeze_selection_date")
    apply_formal_risk_defaults_to_mapping(data)
    return data


def config_to_args(config: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if key in METADATA_ARG_MAP:
            args.extend([METADATA_ARG_MAP[key], json.dumps(value, ensure_ascii=False, sort_keys=True)])
            continue
        flag = f"--{key.replace('_', '-')}"
        if key == "formal_required_adjusts" and isinstance(value, list):
            args.extend([flag, ",".join(str(item) for item in value)])
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, (dict, list)):
            raise ValueError(f"unsupported structured config key for CLI: {key}")
        if value is None:
            continue
        args.extend([flag, str(value)])
    return args


def _db_max_raw_date(db: Path) -> pd.Timestamp | None:
    if not db.exists():
        return None
    with sqlite3.connect(db) as conn:
        row = conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()
    return pd.Timestamp(row[0]) if row and row[0] else None


def formal_quality_window(config: dict[str, Any], db: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = pd.Timestamp(config["start_date"])
    train_years = int(config.get("train_years", 4))
    research_start = start_date - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
    coverage_start = research_start - pd.Timedelta(days=460)
    db_end = _db_max_raw_date(db)
    if config.get("end_date"):
        requested_end = pd.Timestamp(config["end_date"])
        end_date = min(requested_end, db_end) if db_end is not None else requested_end
    elif db_end is not None:
        end_date = db_end
    else:
        end_date = start_date
    return coverage_start, end_date


def run_formal_quality_gate(config: dict[str, Any], db: Path) -> dict[str, Any]:
    start_date, end_date = formal_quality_window(config, db)
    required_adjusts = config.get("formal_required_adjusts", ["raw", "qfq", "hfq"])
    if isinstance(required_adjusts, str):
        required_adjusts = [item.strip() for item in required_adjusts.split(",") if item.strip()]
    report = build_quality_report(
        db,
        start_date,
        end_date,
        config.get("board_scope", "main"),
        tuple(required_adjusts),
    )
    if report.get("red_flags"):
        raise RuntimeError("formal data quality gate failed: " + "; ".join(report["red_flags"][:20]))
    return report


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "backtest_dynamic_rebalance.py"),
        "--db",
        str(args.db),
        *config_to_args(config),
    ]
    if args.dry_run:
        print(json.dumps({"command": command}, ensure_ascii=False, indent=2))
        return 0
    run_formal_quality_gate(config, args.db)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

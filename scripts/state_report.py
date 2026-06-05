#!/usr/bin/env python3
"""Report current signal/position/alert/paper-trade state."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_schema import ensure_quant_schema
from quant_state import state_counts
from professional_quant.paper.observation import (  # noqa: E402
    DEFAULT_OBSERVATION_DAYS,
    observation_audit,
    parse_bool_int,
    parse_config_json,
    rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper-trading state tables.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--strategy", help="optional strategy filter")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--observation-days",
        type=int,
        default=DEFAULT_OBSERVATION_DAYS,
        help="audit the latest N distinct signal-run dates for paper-observation readiness; default is 60",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_report(
    db: Path,
    strategy: str | None,
    limit: int,
    observation_days: int = DEFAULT_OBSERVATION_DAYS,
) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)
        report: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "database": str(db),
            "strategy": strategy,
            "counts": state_counts(conn, strategy),
            "paper_observation": observation_audit(conn, strategy, observation_days),
        }
        strategy_where = "where strategy = ?" if strategy else ""
        params = (strategy,) if strategy else ()
        report["latest_signals"] = rows(
            conn,
            f"""
            select signal_id, strategy, symbol, signal_date, action, score, weight, reason
            from signals
            {strategy_where}
            order by signal_date desc, created_at desc
            limit ?
            """,
            (*params, limit),
        )
        latest_runs = rows(
            conn,
            f"""
            select run_id, strategy, signal_date, status, config_json, message, started_at, finished_at
            from signal_runs
            {strategy_where}
            order by signal_date desc, started_at desc
            limit ?
            """,
            (*params, limit),
        )
        for row in latest_runs:
            config = parse_config_json(row.pop("config_json", None))
            row["manual_confirmation_required"] = config.get("manual_confirmation_required")
            row["manual_confirmed_at"] = config.get("manual_confirmed_at")
            row["data_delay_days"] = config.get("data_delay_days")
            row["latest_raw_trade_date"] = config.get("latest_raw_trade_date")
        report["latest_signal_runs"] = latest_runs
        latest_paper_runs = rows(
            conn,
            f"""
            select run_id, strategy, signal_date, entry_date, status, picks_path, observation_days,
                   data_fresh, manual_confirmation_required, manual_confirmed_at, manifest_json,
                   message, started_at, finished_at, updated_at
            from paper_run_registry
            {strategy_where}
            order by signal_date desc, updated_at desc
            limit ?
            """,
            (*params, limit),
        )
        for row in latest_paper_runs:
            row["data_fresh"] = parse_bool_int(row.get("data_fresh"))
            row["manual_confirmation_required"] = parse_bool_int(row.get("manual_confirmation_required"))
            row["manifest"] = parse_config_json(row.pop("manifest_json", None))
        report["latest_paper_runs"] = latest_paper_runs
        report["latest_positions"] = rows(
            conn,
            f"""
            select strategy, symbol, as_of_date, quantity, avg_cost, market_value, cash
            from positions
            {strategy_where}
            order by as_of_date desc, updated_at desc
            limit ?
            """,
            (*params, limit),
        )
        report["latest_alerts"] = rows(
            conn,
            f"""
            select alert_id, strategy, alert_date, severity, title, acknowledged_at
            from alerts
            {strategy_where}
            order by alert_date desc, created_at desc
            limit ?
            """,
            (*params, limit),
        )
        report["latest_paper_trades"] = rows(
            conn,
            f"""
            select trade_id, strategy, symbol, trade_date, side, quantity, price, amount, status
            from paper_trades
            {strategy_where}
            order by trade_date desc, created_at desc
            limit ?
            """,
            (*params, limit),
        )
        return report


def main() -> int:
    args = parse_args()
    report = build_report(args.db, args.strategy, args.limit, args.observation_days)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

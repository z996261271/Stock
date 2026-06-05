#!/usr/bin/env python3
"""Idempotent daily paper-trading pipeline with freshness and failure recording."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from datetime import date
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_daily_from_picks import build_rows_from_picks, latest_picks_file  # noqa: E402
from generate_daily_signals import STRATEGY, build_plan, write_plan  # noqa: E402
from professional_quant.core.context import RunContext  # noqa: E402
from professional_quant.core.events import (  # noqa: E402
    ALERT_DISPATCHED,
    DATA_READY,
    ORDER_PLAN_BUILT,
    PAPER_RUN_RECORDED,
    SIGNAL_GENERATED,
)
from quant_schema import ensure_quant_schema  # noqa: E402
from quant_state import record_alert, record_alert_attempt, record_signal_run, stable_id  # noqa: E402
from quant_state import record_paper_run  # noqa: E402
from state_report import DEFAULT_OBSERVATION_DAYS, build_report as build_state_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily checked paper-trading workflow from formal picks.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--picks", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/formal"))
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--signal-date")
    parser.add_argument("--entry-date")
    parser.add_argument("--run-id")
    parser.add_argument("--allow-stale-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state-report-output", type=Path)
    parser.add_argument("--manual-confirmed-at", help="operator confirmation timestamp for this paper run")
    parser.add_argument(
        "--observation-days",
        type=int,
        default=DEFAULT_OBSERVATION_DAYS,
        help="state-report lookback trading days; default is the latest 60",
    )
    return parser.parse_args()


def latest_raw_trade_date(db: Path) -> str | None:
    if not db.exists():
        return None
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)
        row = conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()
    return str(row[0]) if row and row[0] else None


def freshness_report(db: Path, signal_date: str) -> dict[str, Any]:
    latest_raw = latest_raw_trade_date(db)
    is_fresh = latest_raw is not None and latest_raw >= signal_date
    return {
        "latest_raw_trade_date": latest_raw,
        "required_signal_date": signal_date,
        "data_delay_days": data_delay_days(latest_raw, signal_date),
        "is_fresh": is_fresh,
        "blocker": None if is_fresh else f"raw data stale: latest={latest_raw}, required>={signal_date}",
    }


def data_delay_days(latest_raw: str | None, signal_date: str) -> int | None:
    if not latest_raw:
        return None
    latest = date.fromisoformat(latest_raw)
    required = date.fromisoformat(signal_date)
    return max((required - latest).days, 0)


def record_pipeline_failure(db: Path, strategy: str, signal_date: str, run_id: str, message: str) -> None:
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)
        record_signal_run(
            conn,
            strategy=strategy,
            signal_date=signal_date,
            status="failed",
            message=message,
            run_id=run_id,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        alert_id = record_alert(
            conn,
            strategy=strategy,
            alert_date=signal_date,
            severity="error",
            title="daily paper pipeline failed",
            message=message,
            payload={"run_id": run_id},
        )
        record_alert_attempt(conn, run_id=run_id, alert_id=alert_id, channel="state_db", status="failed", message=message)


def record_pipeline_registry(
    db: Path,
    *,
    run_id: str,
    strategy: str,
    signal_date: str,
    entry_date: str | None,
    status: str,
    picks_path: Path,
    observation_days: int,
    freshness: dict[str, Any],
    manual_confirmed_at: str | None,
    manifest: dict[str, Any],
    message: str | None = None,
) -> None:
    with sqlite3.connect(db) as conn:
        record_paper_run(
            conn,
            run_id=run_id,
            strategy=strategy,
            signal_date=signal_date,
            entry_date=entry_date,
            status=status,
            picks_path=str(picks_path),
            observation_days=observation_days,
            data_fresh=bool(freshness.get("is_fresh")),
            manual_confirmation_required=True,
            manual_confirmed_at=manual_confirmed_at,
            manifest=manifest,
            message=message,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )


def pipeline_manifest(
    *,
    run_id: str,
    strategy: str,
    picks_path: Path,
    plan: dict[str, Any],
    freshness: dict[str, Any],
    observation_days: int,
    allow_stale_data: bool,
    manual_confirmed_at: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "paper_pipeline_manifest.v1",
        "run_id": run_id,
        "strategy": strategy,
        "picks": str(picks_path),
        "signal_date": plan.get("signal_date"),
        "entry_date": plan.get("entry_date"),
        "observation_days": int(observation_days),
        "allow_stale_data": bool(allow_stale_data),
        "freshness": freshness,
        "manual_confirmation": {
            "required": True,
            "confirmed_at": manual_confirmed_at,
        },
        "plan_counts": {
            "signals": int(len(plan.get("signals", []))),
            "positions": int(len(plan.get("positions", []))),
            "paper_trades": int(len(plan.get("paper_trades", []))),
        },
    }


def build_pipeline_output(
    *,
    db: Path,
    picks_path: Path,
    strategy: str,
    cash: float,
    signal_date: str | None,
    entry_date: str | None,
    run_id: str | None,
    dry_run: bool,
    allow_stale_data: bool,
    observation_days: int = DEFAULT_OBSERVATION_DAYS,
    manual_confirmed_at: str | None = None,
) -> tuple[dict[str, Any], int]:
    rows, selected_signal_date, selected_entry_date = build_rows_from_picks(
        picks_path,
        db,
        strategy,
        signal_date,
        entry_date,
    )
    resolved_run_id = run_id or stable_id("run", strategy, selected_signal_date, str(picks_path))
    freshness = freshness_report(db, selected_signal_date)
    plan = build_plan(rows, strategy, selected_signal_date, selected_entry_date, cash)
    context = RunContext(
        run_id=resolved_run_id,
        strategy=strategy,
        mode="paper",
        metadata={
            "picks": str(picks_path),
            "observation_days": int(observation_days),
            "allow_stale_data": bool(allow_stale_data),
        },
    )
    if freshness["is_fresh"]:
        context.emit(DATA_READY, freshness)
    context.emit(SIGNAL_GENERATED, {"signals": int(len(plan.get("signals", [])))})
    context.emit(ORDER_PLAN_BUILT, {"paper_trades": int(len(plan.get("paper_trades", [])))})
    manifest = pipeline_manifest(
        run_id=resolved_run_id,
        strategy=strategy,
        picks_path=picks_path,
        plan=plan,
        freshness=freshness,
        observation_days=observation_days,
        allow_stale_data=allow_stale_data,
        manual_confirmed_at=manual_confirmed_at,
    )
    manifest["run_context"] = context.manifest()
    output: dict[str, Any] = {
        "run_id": resolved_run_id,
        "strategy": strategy,
        "picks": str(picks_path),
        "dry_run": dry_run,
        "allow_stale_data": allow_stale_data,
        "observation_days": int(observation_days),
        "manual_confirmation": {
            "required": True,
            "confirmed_at": manual_confirmed_at,
        },
        "freshness": freshness,
        "pipeline_manifest": manifest,
        "run_context": context.manifest(),
        "plan": plan,
    }
    if not freshness["is_fresh"] and not allow_stale_data:
        message = str(freshness["blocker"])
        output["status"] = "failed"
        output["message"] = message
        context.emit(ALERT_DISPATCHED, {"status": "failed", "message": message})
        context.emit(PAPER_RUN_RECORDED, {"status": "failed"})
        manifest["run_context"] = context.manifest()
        output["run_context"] = context.manifest()
        if not dry_run:
            record_pipeline_failure(db, strategy, selected_signal_date, resolved_run_id, message)
            record_pipeline_registry(
                db,
                run_id=resolved_run_id,
                strategy=strategy,
                signal_date=selected_signal_date,
                entry_date=selected_entry_date,
                status="failed",
                picks_path=picks_path,
                observation_days=observation_days,
                freshness=freshness,
                manual_confirmed_at=manual_confirmed_at,
                manifest=manifest,
                message=message,
            )
        return output, 2
    if dry_run:
        output["status"] = "planned"
        output["run_context"] = context.manifest()
        output["pipeline_manifest"]["run_context"] = context.manifest()
        return output, 0
    output["counts"] = write_plan(
        db,
        strategy,
        resolved_run_id,
        plan,
        run_config={
            "manual_confirmation_required": True,
            "manual_confirmed_at": manual_confirmed_at,
            "data_delay_days": freshness["data_delay_days"],
            "latest_raw_trade_date": freshness["latest_raw_trade_date"],
            "source_picks": str(picks_path),
        },
    )
    context.emit(PAPER_RUN_RECORDED, {"status": "finished"})
    manifest["run_context"] = context.manifest()
    record_pipeline_registry(
        db,
        run_id=resolved_run_id,
        strategy=strategy,
        signal_date=selected_signal_date,
        entry_date=selected_entry_date,
        status="finished",
        picks_path=picks_path,
        observation_days=observation_days,
        freshness=freshness,
        manual_confirmed_at=manual_confirmed_at,
        manifest=manifest,
        message="daily paper pipeline finished",
    )
    output["state_report"] = build_state_report(db, strategy, limit=5, observation_days=observation_days)
    output["status"] = "finished"
    output["pipeline_manifest"] = manifest
    output["run_context"] = context.manifest()
    return output, 0


def main() -> int:
    args = parse_args()
    picks_path = args.picks or latest_picks_file(args.reports_dir)
    try:
        output, status = build_pipeline_output(
            db=args.db,
            picks_path=picks_path,
            strategy=args.strategy,
            cash=args.cash,
            signal_date=args.signal_date,
            entry_date=args.entry_date,
            run_id=args.run_id,
            dry_run=args.dry_run,
            allow_stale_data=args.allow_stale_data,
            observation_days=args.observation_days,
            manual_confirmed_at=args.manual_confirmed_at,
        )
    except Exception as exc:
        selected_signal_date = args.signal_date or "unknown"
        resolved_run_id = args.run_id or stable_id("run", args.strategy, selected_signal_date, str(picks_path))
        if not args.dry_run and selected_signal_date != "unknown":
            record_pipeline_failure(args.db, args.strategy, selected_signal_date, resolved_run_id, str(exc))
        output = {"run_id": resolved_run_id, "strategy": args.strategy, "status": "failed", "message": str(exc)}
        status = 2
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.state_report_output and output.get("state_report"):
        args.state_report_output.parent.mkdir(parents=True, exist_ok=True)
        args.state_report_output.write_text(json.dumps(output["state_report"], ensure_ascii=False, indent=2) + "\n")
    print(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

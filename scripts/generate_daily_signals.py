#!/usr/bin/env python3
"""Generate idempotent daily paper-trading signals from a prepared signal file."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from professional_quant.paper.planning import build_paper_plan  # noqa: E402

from quant_schema import ensure_quant_schema  # noqa: E402
from quant_state import (  # noqa: E402
    record_alert,
    record_alert_attempt,
    record_paper_trade,
    record_position,
    record_signal,
    record_signal_run,
    stable_id,
    state_counts,
)  # noqa: E402


STRATEGY = "dynamic_daily_checked_rebalance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-trading signals and state rows.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--signals", type=Path, required=True, help="CSV or JSON signal file")
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--entry-date", help="planned next-open/paper-trade date; defaults to signal_date")
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--run-id")
    parser.add_argument("--manual-confirmed-at", help="operator confirmation timestamp for this paper run")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_signal_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("signals", [])
        if not isinstance(data, list):
            raise ValueError("JSON signal file must be a list or an object with a signals list")
        return [dict(row) for row in data]
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def normalize_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("symbol", "")).strip()
    if not symbol:
        raise ValueError(f"signal row missing symbol: {row}")
    action = str(row.get("action", row.get("side", "buy"))).strip().lower()
    if action not in {"buy", "sell", "hold"}:
        raise ValueError(f"unsupported action for {symbol}: {action}")
    weight = row.get("weight")
    score = row.get("score")
    price = row.get("price", row.get("entry_open"))
    return {
        "symbol": symbol,
        "action": action,
        "score": float(score) if pd.notna(score) else None,
        "weight": float(weight) if pd.notna(weight) else None,
        "price": float(price) if pd.notna(price) else None,
        "reason": str(row.get("reason", row.get("formula", "daily_signal"))),
        "payload": {key: _json_safe(value) for key, value in row.items()},
    }


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def build_plan(rows: list[dict[str, Any]], strategy: str, signal_date: str, entry_date: str, cash: float) -> dict[str, Any]:
    return build_paper_plan(
        raw_rows=rows,
        strategy=strategy,
        signal_date=signal_date,
        entry_date=entry_date,
        cash=cash,
        signal_id_fn=lambda item_strategy, symbol, item_signal_date, action: stable_id(
            "sig",
            "",
            item_strategy,
            symbol,
            item_signal_date,
            action,
        ),
)


def write_plan(
    db: Path,
    strategy: str,
    run_id: str,
    plan: dict[str, Any],
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    config = {
        "entry_date": plan["entry_date"],
        "cash": plan["cash"],
        "manual_confirmation_required": True,
        "manual_confirmed_at": None,
    }
    if run_config:
        config.update(dict(run_config))
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)
        record_signal_run(
            conn,
            strategy=strategy,
            signal_date=plan["signal_date"],
            status="started",
            config=config,
            run_id=run_id,
        )
        for row in plan["signals"]:
            record_signal(
                conn,
                strategy=strategy,
                symbol=row["symbol"],
                signal_date=plan["signal_date"],
                action=row["action"],
                score=row["score"],
                weight=row["weight"],
                reason=row["reason"],
                payload=row["payload"],
                signal_id=row["signal_id"],
                run_id=run_id,
            )
        for row in plan["positions"]:
            record_position(
                conn,
                strategy=strategy,
                symbol=row["symbol"],
                as_of_date=plan["entry_date"],
                quantity=row["quantity"],
                avg_cost=row["avg_cost"],
                market_value=row["market_value"],
                cash=row["cash"],
                payload=row["payload"],
            )
        for row in plan["paper_trades"]:
            record_paper_trade(
                conn,
                strategy=strategy,
                symbol=row["symbol"],
                signal_id=row["signal_id"],
                trade_date=plan["entry_date"],
                side=row["side"],
                quantity=row["quantity"],
                price=row["price"],
                amount=row["amount"],
                status=row["status"],
                reason=row["reason"],
                payload=row.get("payload"),
            )
        alert_id = record_alert(
            conn,
            strategy=strategy,
            alert_date=plan["signal_date"],
            severity="info",
            title="daily paper signals generated",
            message=f"{len(plan['signals'])} signals, {len(plan['paper_trades'])} planned paper trades",
            payload={"run_id": run_id},
        )
        record_alert_attempt(conn, run_id=run_id, alert_id=alert_id, channel="state_db", status="recorded")
        record_signal_run(
            conn,
            strategy=strategy,
            signal_date=plan["signal_date"],
            status="finished",
            config=config,
            message="daily paper signals recorded",
            run_id=run_id,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        return state_counts(conn, strategy)


def main() -> int:
    args = parse_args()
    entry_date = args.entry_date or args.signal_date
    rows = load_signal_rows(args.signals)
    plan = build_plan(rows, args.strategy, args.signal_date, entry_date, args.cash)
    run_id = args.run_id or stable_id("run", args.strategy, args.signal_date)
    output = {"run_id": run_id, "strategy": args.strategy, "dry_run": args.dry_run, "plan": plan}
    if not args.dry_run:
        output["counts"] = write_plan(
            args.db,
            args.strategy,
            run_id,
            plan,
            run_config={"manual_confirmed_at": args.manual_confirmed_at},
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

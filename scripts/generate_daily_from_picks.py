#!/usr/bin/env python3
"""Generate paper-trading signals from the latest dynamic backtest picks artifact."""

from __future__ import annotations

import argparse
import json
import sqlite3
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

from generate_daily_signals import STRATEGY, build_plan, write_plan  # noqa: E402
from quant_schema import ensure_quant_schema  # noqa: E402
from quant_state import stable_id  # noqa: E402
from professional_quant.data.industry import normalize_symbol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate buy/sell/hold paper signals from latest picks.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--picks", type=Path, help="specific .picks.csv file")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/formal"))
    parser.add_argument("--strategy", default=STRATEGY)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--signal-date", help="use a specific signal_date from the picks file; defaults to latest")
    parser.add_argument("--entry-date", help="override entry_date; defaults to picks entry_date or next signal date")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def latest_picks_file(reports_dir: Path) -> Path:
    candidates = sorted(reports_dir.rglob("*.picks.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no .picks.csv files found under {reports_dir}")
    return candidates[0]


def latest_positions(db: Path, strategy: str) -> dict[str, dict[str, Any]]:
    if not db.exists():
        return {}
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select p.*
            from positions p
            join (
                select symbol, max(as_of_date) as max_date
                from positions
                where strategy = ?
                group by symbol
            ) latest on latest.symbol = p.symbol and latest.max_date = p.as_of_date
            where p.strategy = ? and p.quantity > 0
            """,
            (strategy, strategy),
        ).fetchall()
    return {str(row["symbol"]): dict(row) for row in rows}


def build_rows_from_picks(
    picks_path: Path,
    db: Path,
    strategy: str,
    signal_date: str | None = None,
    entry_date: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    picks = pd.read_csv(picks_path)
    required = {"signal_date", "symbol"}
    if not required.issubset(picks.columns):
        raise ValueError(f"picks file must include {sorted(required)}: {picks_path}")
    picks["signal_date"] = pd.to_datetime(picks["signal_date"]).dt.strftime("%Y-%m-%d")
    selected_signal_date = signal_date or str(picks["signal_date"].max())
    current = picks[picks["signal_date"] == selected_signal_date].copy()
    if current.empty:
        raise ValueError(f"no picks for signal_date={selected_signal_date} in {picks_path}")
    if entry_date is None:
        if "entry_date" in current.columns and current["entry_date"].notna().any():
            selected_entry_date = str(pd.to_datetime(current["entry_date"].dropna().iloc[0]).strftime("%Y-%m-%d"))
        else:
            selected_entry_date = selected_signal_date
    else:
        selected_entry_date = entry_date

    previous = latest_positions(db, strategy)
    current["symbol"] = current["symbol"].map(normalize_symbol)
    target_symbols = set(current["symbol"].astype(str))
    rows: list[dict[str, Any]] = []
    for record in current.sort_values(["weight", "score"], ascending=[False, False], na_position="last").to_dict("records"):
        symbol = normalize_symbol(record["symbol"])
        weight = _float_or_none(record.get("weight"))
        action = "hold" if symbol in previous else "buy"
        rows.append(
            {
                "symbol": symbol,
                "action": action,
                "score": _float_or_none(record.get("score")),
                "weight": weight,
                "entry_open": _float_or_none(record.get("entry_open", record.get("price"))),
                "reason": str(record.get("formula", record.get("reason", "latest_dynamic_pick"))),
                "source_picks": str(picks_path),
                "source_signal_date": selected_signal_date,
            }
        )
    for symbol, position in sorted(previous.items()):
        if symbol in target_symbols:
            continue
        rows.append(
            {
                "symbol": symbol,
                "action": "sell",
                "score": None,
                "weight": 0.0,
                "entry_open": _float_or_none(position.get("avg_cost")),
                "quantity": _float_or_none(position.get("quantity")) or 0.0,
                "reason": "not_in_latest_target_picks",
                "source_picks": str(picks_path),
                "source_signal_date": selected_signal_date,
            }
        )
    return rows, selected_signal_date, selected_entry_date


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def main() -> int:
    args = parse_args()
    picks_path = args.picks or latest_picks_file(args.reports_dir)
    rows, signal_date, entry_date = build_rows_from_picks(
        picks_path,
        args.db,
        args.strategy,
        args.signal_date,
        args.entry_date,
    )
    plan = build_plan(rows, args.strategy, signal_date, entry_date, args.cash)
    run_id = args.run_id or stable_id("run", args.strategy, signal_date, str(picks_path))
    output = {
        "run_id": run_id,
        "strategy": args.strategy,
        "picks": str(picks_path),
        "dry_run": args.dry_run,
        "plan": plan,
    }
    if not args.dry_run:
        output["counts"] = write_plan(args.db, args.strategy, run_id, plan)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

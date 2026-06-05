#!/usr/bin/env python3
"""Build nested-validation and sensitivity summaries from formal report artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize professional validation evidence from report artifacts.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--capacity-stress", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_validation_summary(metrics: dict[str, Any], diagnostics: pd.DataFrame | None) -> dict[str, Any]:
    split = metrics.get("split_policy", {})
    multiple = metrics.get("multiple_testing", {})
    rows: list[dict[str, Any]] = []
    if diagnostics is not None and not diagnostics.empty and {"year", "status"}.issubset(diagnostics.columns):
        selected = diagnostics[diagnostics["status"].astype(str).isin(["selected", "selected_frozen", "frozen_selected"])]
        for _, row in selected.sort_values("year").iterrows():
            rows.append(
                {
                    "year": int(row["year"]) if pd.notna(row.get("year")) else None,
                    "status": str(row.get("status")),
                    "train_start": row.get("train_start"),
                    "train_end": row.get("train_end"),
                    "formula": row.get("formula"),
                    "annual_return": _float_or_none(row.get("annual_return")),
                    "max_drawdown": _float_or_none(row.get("max_drawdown")),
                }
            )
    return {
        "status": "available" if rows else "missing_diagnostics",
        "split_policy": split,
        "outer_rows": rows,
        "scan_size": {
            "specs_evaluated": multiple.get("specs_evaluated"),
            "selected_rows": multiple.get("selected_rows"),
            "train_candidate_rows_written": multiple.get("train_candidate_rows_written"),
        },
        "anti_leakage_note": "Outer year rows must be evaluated after prior-period selection; final test results must not reselection-tune.",
    }


def sensitivity_summary(capacity: pd.DataFrame | None) -> dict[str, Any]:
    if capacity is None or capacity.empty:
        return {"status": "missing_capacity_stress"}
    required = {"initial_cash", "capacity_pct_of_amount", "slippage_bps", "annual_return", "max_drawdown"}
    if not required.issubset(capacity.columns):
        return {"status": "missing_required_columns", "columns": sorted(capacity.columns)}
    current = capacity[capacity.get("is_current_setting", False) == True]  # noqa: E712
    base = current.iloc[0] if not current.empty else capacity.iloc[0]
    rows: list[dict[str, Any]] = []
    for _, row in capacity.iterrows():
        rows.append(
            {
                "initial_cash": _float_or_none(row.get("initial_cash")),
                "capacity_pct_of_amount": _float_or_none(row.get("capacity_pct_of_amount")),
                "slippage_bps": _float_or_none(row.get("slippage_bps")),
                "annual_return": _float_or_none(row.get("annual_return")),
                "max_drawdown": _float_or_none(row.get("max_drawdown")),
                "annual_return_delta_vs_base": _float_or_none(
                    _float_or_zero(row.get("annual_return")) - _float_or_zero(base.get("annual_return"))
                ),
                "max_drawdown_delta_vs_base": _float_or_none(
                    _float_or_zero(row.get("max_drawdown")) - _float_or_zero(base.get("max_drawdown"))
                ),
            }
        )
    annual_returns = pd.to_numeric(capacity["annual_return"], errors="coerce").dropna()
    drawdowns = pd.to_numeric(capacity["max_drawdown"], errors="coerce").dropna()
    return {
        "status": "available",
        "rows": rows,
        "annual_return_min": _float_or_none(annual_returns.min()) if not annual_returns.empty else None,
        "annual_return_max": _float_or_none(annual_returns.max()) if not annual_returns.empty else None,
        "max_drawdown_worst": _float_or_none(drawdowns.min()) if not drawdowns.empty else None,
        "risk_note": "This is a capacity/slippage sensitivity summary; top_n/hold/stop-loss sensitivity requires additional frozen replays.",
    }


def _float_or_zero(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else 0.0


def _float_or_none(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def build_report(metrics_path: Path, diagnostics_path: Path | None, capacity_path: Path | None) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path and diagnostics_path.exists() else None
    capacity = pd.read_csv(capacity_path) if capacity_path and capacity_path.exists() else None
    return {
        "metrics": str(metrics_path),
        "strategy": metrics.get("config", {}).get("strategy"),
        "is_formal_valid": metrics.get("is_formal_valid"),
        "nested_validation": nested_validation_summary(metrics, diagnostics),
        "sensitivity": sensitivity_summary(capacity),
        "multiple_testing": metrics.get("multiple_testing", {}),
        "risk_budget": metrics.get("risk_budget", {}),
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.metrics, args.diagnostics, args.capacity_stress)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

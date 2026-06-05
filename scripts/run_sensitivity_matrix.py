#!/usr/bin/env python3
"""Run or plan frozen formal replays across a bounded sensitivity matrix."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_formal_dynamic import config_to_args, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay frozen formal config over sensitivity variants.")
    parser.add_argument("--config", type=Path, default=Path("configs/formal_dynamic_default.json"))
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/formal/sensitivity"))
    parser.add_argument("--execute", action="store_true", help="run the backtests; default only writes the matrix plan")
    parser.add_argument("--workers", type=int, help="override workers for every variant")
    parser.add_argument("--timeout", type=int, default=0, help="optional per-run timeout seconds")
    return parser.parse_args()


def _variant_values(config: dict[str, Any]) -> dict[str, list[Any]]:
    top_n = int(config.get("top_n", 3))
    min_hold = int(config.get("min_hold_days", 2)) if "min_hold_days" in config else 2
    max_hold = int(config.get("max_hold_days", 10)) if "max_hold_days" in config else 10
    stop_loss = config.get("stop_loss")
    capacity = float(config.get("capacity_pct_of_amount", 0.02))
    slippage = float(config.get("slippage_bps", 5.0))
    market_filter_enabled = bool(config.get("industry_source", ""))
    return {
        "top_n": sorted({max(top_n - 1, 1), top_n, top_n + 1}),
        "min_hold_days": sorted({max(min_hold - 1, 1), min_hold, min_hold + 1}),
        "max_hold_days": sorted({max(max_hold - 5, min_hold), max_hold, max_hold + 5}),
        "stop_loss": _sorted_optional_float_values(stop_loss),
        "industry_source": [config.get("industry_source", ""), ""] if market_filter_enabled else [""],
        "capacity_pct_of_amount": sorted({max(capacity / 2.0, 0.0001), capacity, capacity * 2.0}),
        "slippage_bps": sorted({max(slippage / 2.0, 0.0), slippage, slippage * 2.0}),
    }


def _sorted_optional_float_values(value: Any) -> list[Any]:
    if value is None:
        return ["none", 0.10, 0.20]
    base = float(value)
    values = {"none", max(base / 2.0, 0.01), base, min(base * 2.0, 0.50)}
    return sorted(values, key=lambda item: -1.0 if item == "none" else float(item))


def build_matrix(config: dict[str, Any], output_dir: Path, worker_override: int | None = None) -> list[dict[str, Any]]:
    """Build one-at-a-time sensitivity variants around the frozen base config."""
    values = _variant_values(config)
    base = copy.deepcopy(config)
    base["output_dir"] = str(output_dir / "base")
    if worker_override is not None:
        base["workers"] = worker_override
    rows = [
        {
            "variant_id": "base",
            "dimension": "base",
            "value": "base",
            "config": base,
        }
    ]
    for dimension, candidates in values.items():
        base_value = config.get(dimension)
        for value in candidates:
            if _same_value(value, base_value):
                continue
            variant = copy.deepcopy(config)
            variant[dimension] = value
            if worker_override is not None:
                variant["workers"] = worker_override
            value_label = "none" if value is None else str(value).replace("/", "_").replace(" ", "_")
            variant_id = f"{dimension}_{value_label}"
            variant["output_dir"] = str(output_dir / variant_id)
            rows.append(
                {
                    "variant_id": variant_id,
                    "dimension": dimension,
                    "value": value,
                    "config": variant,
                }
            )
    return rows


def matrix_plan_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": row["variant_id"],
            "dimension": row["dimension"],
            "value": row["value"],
            "output_dir": row["config"].get("output_dir"),
            "top_n": row["config"].get("top_n"),
            "min_hold_days": row["config"].get("min_hold_days"),
            "max_hold_days": row["config"].get("max_hold_days"),
            "stop_loss": row["config"].get("stop_loss"),
            "industry_source": row["config"].get("industry_source"),
            "capacity_pct_of_amount": row["config"].get("capacity_pct_of_amount"),
            "slippage_bps": row["config"].get("slippage_bps"),
        }
        for row in matrix
    ]


def execute_variant(db: Path, row: dict[str, Any], timeout: int = 0) -> dict[str, Any]:
    output_dir = Path(str(row["config"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "backtest_dynamic_rebalance.py"),
        "--db",
        str(db),
        *config_to_args(row["config"]),
    ]
    started = datetime.now().isoformat(timespec="seconds")
    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout if timeout > 0 else None,
    )
    metrics_files = sorted(output_dir.glob("*.metrics.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    metrics: dict[str, Any] = {}
    if metrics_files:
        metrics = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    return {
        "variant_id": row["variant_id"],
        "dimension": row["dimension"],
        "value": row["value"],
        "returncode": result.returncode,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "metrics_path": str(metrics_files[0]) if metrics_files else None,
        "annual_return": metrics.get("annual_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "total_return": metrics.get("total_return"),
        "final_equity": metrics.get("final_equity"),
        "is_formal_valid": metrics.get("is_formal_valid"),
        "stderr_tail": result.stderr[-2000:],
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(results)
    if frame.empty:
        return {"status": "empty"}
    planned_only = "returncode" in frame and frame["returncode"].isna().all()
    if "returncode" in frame:
        completed = frame[frame["returncode"] == 0]
    else:
        completed = frame
    annual = pd.to_numeric(
        completed["annual_return"] if "annual_return" in completed else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()
    drawdown = pd.to_numeric(
        completed["max_drawdown"] if "max_drawdown" in completed else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()
    return {
        "status": "planned" if planned_only else ("executed" if "returncode" in frame else "planned"),
        "variants": int(len(frame)),
        "completed": int(len(completed)),
        "failed": 0 if planned_only else int(len(frame) - len(completed)),
        "annual_return_min": float(annual.min()) if not annual.empty else None,
        "annual_return_max": float(annual.max()) if not annual.empty else None,
        "max_drawdown_worst": float(drawdown.min()) if not drawdown.empty else None,
        "risk_note": "One-at-a-time frozen replays cover top_n, hold days, stop-loss, market filter, capacity, and slippage sensitivity.",
    }


def _same_value(left: Any, right: Any) -> bool:
    if left is None and right in (None, ""):
        return True
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(config, args.output_dir, args.workers)
    plan_rows = matrix_plan_rows(matrix)
    plan_path = args.output_dir / "sensitivity_matrix.plan.csv"
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)
    config_path = args.output_dir / "sensitivity_matrix.configs.json"
    config_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    if args.execute:
        results = [execute_variant(args.db, row, args.timeout) for row in matrix]
    else:
        results = [
            {
                "variant_id": row["variant_id"],
                "dimension": row["dimension"],
                "value": row["value"],
                "returncode": None,
                "metrics_path": None,
            }
            for row in matrix
        ]
    results_path = args.output_dir / "sensitivity_matrix.results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    report = {
        "config": str(args.config),
        "db": str(args.db),
        "execute": args.execute,
        "plan_csv": str(plan_path),
        "configs_json": str(config_path),
        "results_csv": str(results_path),
        "summary": summarize_results(results),
    }
    report_path = args.output_dir / "sensitivity_matrix.report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if not args.execute or report["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

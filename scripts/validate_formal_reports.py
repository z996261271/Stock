#!/usr/bin/env python3
"""Validate formal report artifacts for required professional fields."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Any


REQUIRED_METRIC_FIELDS = (
    "is_formal_valid",
    "data_quality_red_flags",
    "status_coverage",
    "split_policy",
    "frozen_config",
    "professional_metrics",
    "annual_breakdown",
    "monthly_breakdown",
    "benchmarks",
    "multiple_testing",
    "capacity_stress",
    "risk_budget",
    "config",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal .metrics.json/.manifest.json pairs.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/formal"))
    parser.add_argument("--require-formal-valid", action="store_true")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="treat an empty active formal directory as valid after old artifacts are quarantined",
    )
    parser.add_argument(
        "--quarantine-invalid",
        action="store_true",
        help="move invalid report artifact families under reports-dir/_invalid/<timestamp>/",
    )
    parser.add_argument("--quarantine-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_metrics(path: Path, require_formal_valid: bool = False) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_METRIC_FIELDS if field not in metrics]
    manifest = _manifest_for_metrics(path)
    issues: list[str] = []
    if missing:
        issues.append("missing_fields:" + ",".join(missing))
    if not manifest.exists():
        issues.append("manifest_missing")
    if require_formal_valid and metrics.get("is_formal_valid") is not True:
        issues.append("not_formal_valid")
    config = metrics.get("config", {})
    industry = config.get("industry_coverage", {})
    if industry and config.get("execution", {}).get("max_industry_weight", 0) and not industry.get("is_real_industry_mapping"):
        issues.append("industry_cap_without_real_mapping")
    return {
        "metrics": str(path),
        "manifest": str(manifest) if manifest.exists() else None,
        "is_formal_valid": metrics.get("is_formal_valid"),
        "issues": issues,
    }


def _manifest_for_metrics(path: Path) -> Path:
    name = path.name
    if name.endswith(".metrics.json"):
        return path.with_name(name.removesuffix(".metrics.json") + ".manifest.json")
    return path.with_suffix(".manifest.json")


def _active_metrics_files(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(reports_dir.rglob("*.metrics.json")):
        relative_parts = path.relative_to(reports_dir).parts
        if any(part.startswith("_invalid") for part in relative_parts):
            continue
        files.append(path)
    return files


def _artifact_family(metrics_path: Path) -> list[Path]:
    if metrics_path.name.endswith(".metrics.json"):
        prefix = metrics_path.name.removesuffix(".metrics.json")
    else:
        prefix = metrics_path.stem
    return sorted(path for path in metrics_path.parent.glob(f"{prefix}.*") if path.is_file())


def _unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}.{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to find unique quarantine target for {path}")


def quarantine_report_family(metrics_path: Path, reports_dir: Path, quarantine_dir: Path | None = None) -> list[str]:
    """Move every artifact belonging to one report prefix into quarantine."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    if metrics_path.name.endswith(".metrics.json"):
        prefix = metrics_path.name.removesuffix(".metrics.json")
    else:
        prefix = metrics_path.stem
    destination = (quarantine_dir or reports_dir / "_invalid") / stamp / prefix
    destination.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for artifact in _artifact_family(metrics_path):
        target = _unique_target(destination / artifact.name)
        shutil.move(str(artifact), str(target))
        moved.append(str(target))
    return moved


def build_report(
    reports_dir: Path,
    require_formal_valid: bool = False,
    allow_empty: bool = False,
    quarantine_invalid: bool = False,
    quarantine_dir: Path | None = None,
) -> dict[str, Any]:
    files = _active_metrics_files(reports_dir)
    rows = [validate_metrics(path, require_formal_valid) for path in files]
    issues = [row for row in rows if row["issues"]]
    quarantined_rows: list[dict[str, Any]] = []
    if quarantine_invalid and issues:
        moved_sources: set[str] = set()
        for row in issues:
            metrics = Path(row["metrics"])
            if str(metrics) in moved_sources or not metrics.exists():
                continue
            moved = quarantine_report_family(metrics, reports_dir, quarantine_dir)
            moved_sources.update(str(path) for path in _artifact_family(metrics))
            quarantined_rows.append({**row, "quarantined_files": moved})
        files = _active_metrics_files(reports_dir)
        rows = [validate_metrics(path, require_formal_valid) for path in files]
        issues = [row for row in rows if row["issues"]]
    return {
        "reports_dir": str(reports_dir),
        "metrics_files": int(len(files)),
        "issue_files": int(len(issues)),
        "quarantined_issue_files": int(len(quarantined_rows)),
        "quarantined_files": int(sum(len(row["quarantined_files"]) for row in quarantined_rows)),
        "is_valid": bool((files or allow_empty) and not issues),
        "rows": rows,
        "quarantined_rows": quarantined_rows,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        args.reports_dir,
        args.require_formal_valid,
        allow_empty=args.allow_empty,
        quarantine_invalid=args.quarantine_invalid,
        quarantine_dir=args.quarantine_dir,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["is_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

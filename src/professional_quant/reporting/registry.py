"""Release registry for formal report artifact families."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ARTIFACT_SUFFIXES = {
    "metrics": ".metrics.json",
    "manifest": ".manifest.json",
    "data_quality": ".data_quality.json",
    "picks": ".picks.csv",
    "trades": ".trades.csv",
    "equity": ".equity.csv",
    "diagnostics": ".diagnostics.csv",
    "capacity_stress": ".capacity_stress.csv",
    "attribution": ".attribution.json",
    "factor_report": ".factor_report.json",
    "performance_report": ".performance_report.json",
    "performance_markdown": ".performance_report.md",
}

REQUIRED_RELEASE_ARTIFACTS = ("metrics", "manifest", "data_quality", "picks", "trades")


def build_formal_release_registry(
    reports_dir: Path,
    *,
    release_id: str | None = None,
    include_invalid: bool = False,
) -> dict[str, Any]:
    metrics_files = _metrics_files(reports_dir, include_invalid=include_invalid)
    rows = [registry_row(path, release_id=release_id) for path in metrics_files]
    return {
        "schema_version": "formal_release_registry.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reports_dir": str(reports_dir),
        "release_count": int(len(rows)),
        "required_artifacts": list(REQUIRED_RELEASE_ARTIFACTS),
        "releases": rows,
    }


def registry_row(metrics_path: Path, *, release_id: str | None = None) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    prefix = _prefix_from_metrics(metrics_path)
    artifacts = artifact_map(prefix)
    metrics_artifact = artifacts.get("metrics", {})
    resolved_release_id = release_id or stable_release_id(prefix.name, str(metrics_artifact.get("sha256", "")))
    missing_required = [name for name in REQUIRED_RELEASE_ARTIFACTS if name not in artifacts]
    return {
        "release_id": resolved_release_id,
        "prefix": str(prefix),
        "metrics": str(metrics_path),
        "is_publishable": bool(metrics.get("is_formal_valid") is True and not missing_required),
        "missing_required_artifacts": missing_required,
        "is_formal_valid": metrics.get("is_formal_valid"),
        "data_quality_red_flags": metrics.get("data_quality_red_flags", []),
        "strategy": metrics.get("config", {}).get("strategy"),
        "generated_at": metrics.get("generated_at"),
        "split_policy": metrics.get("split_policy", {}),
        "frozen_config": metrics.get("frozen_config", {}),
        "execution_config": metrics.get("config", {}).get("execution", {}),
        "performance": {
            "total_return": metrics.get("total_return"),
            "annual_return": metrics.get("annual_return"),
            "max_drawdown": metrics.get("max_drawdown"),
        },
        "artifacts": artifacts,
    }


def artifact_map(prefix: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, suffix in ARTIFACT_SUFFIXES.items():
        path = prefix.with_suffix(suffix)
        if path.exists():
            artifacts[name] = artifact_descriptor(path)
    return artifacts


def artifact_descriptor(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": int(len(data)),
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
    }


def stable_release_id(prefix_name: str, metrics_sha: str) -> str:
    digest = hashlib.sha256(f"{prefix_name}|{metrics_sha}".encode("utf-8")).hexdigest()[:12]
    date_token = _date_token(prefix_name)
    return f"rel_{date_token}_{digest}" if date_token else f"rel_{digest}"


def _date_token(prefix_name: str) -> str | None:
    parts = prefix_name.split("_")
    if len(parts) >= 3 and parts[0] == "dynamic" and parts[1] == "rebalance":
        return f"{parts[2]}_{parts[3]}" if len(parts) >= 4 else parts[2]
    return None


def _metrics_files(reports_dir: Path, *, include_invalid: bool) -> list[Path]:
    if not reports_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(reports_dir.rglob("*.metrics.json")):
        relative_parts = path.relative_to(reports_dir).parts
        if not include_invalid and any(part.startswith("_invalid") for part in relative_parts):
            continue
        files.append(path)
    return files


def _prefix_from_metrics(metrics_path: Path) -> Path:
    name = metrics_path.name
    if name.endswith(".metrics.json"):
        return metrics_path.with_name(name.removesuffix(".metrics.json"))
    return metrics_path.with_suffix("")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

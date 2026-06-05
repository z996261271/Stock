#!/usr/bin/env python3
"""Reproducibility manifest helpers for research/backtest reports."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import sqlite3
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_schema import REFERENCE_TABLES, STATE_TABLES


MANIFEST_SCHEMA_VERSION = 1


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_registry_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": file_sha256(path),
    }
    if not path.exists():
        return summary
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary["error"] = f"invalid_json:{exc}"
        return summary
    summary.update(
        {
            "schema_version": registry.get("schema_version"),
            "generated_at": registry.get("generated_at"),
            "entry_count": registry.get("entry_count"),
        }
    )
    entries = registry.get("entries")
    if isinstance(entries, list):
        summary["latest_cache_end_date"] = max(
            (entry.get("end_date") for entry in entries if isinstance(entry, dict) and entry.get("end_date")),
            default=None,
        )
        summary["factor_adjusts"] = sorted(
            {
                str(entry.get("factor_adjust"))
                for entry in entries
                if isinstance(entry, dict) and entry.get("factor_adjust")
            }
        )
    return summary


def sqlite_rows(conn: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(query).fetchall()]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not table_exists(conn, table_name):
        return False
    return any(str(row[1]) == column_name for row in conn.execute(f"PRAGMA table_info({table_name})"))


def database_summary(db: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(db),
        "exists": db.exists(),
        "size_bytes": db.stat().st_size if db.exists() else 0,
    }
    if not db.exists():
        return summary

    with sqlite3.connect(db) as conn:
        summary["schema_tables"] = {
            "reference": {name: table_exists(conn, name) for name in REFERENCE_TABLES},
            "state": {name: table_exists(conn, name) for name in STATE_TABLES},
        }
        if table_exists(conn, "daily_bars"):
            summary["daily_bars"] = sqlite_rows(
                conn,
                """
                select adjust, count(*) as rows, count(distinct symbol) as symbols,
                       min(trade_date) as min_trade_date, max(trade_date) as max_trade_date
                from daily_bars
                group by adjust
                order by adjust
                """,
            )
        if table_exists(conn, "industry_daily_bars"):
            summary["industry_daily_bars"] = sqlite_rows(
                conn,
                """
                select source, count(*) as rows, count(distinct board_code) as boards,
                       min(trade_date) as min_trade_date, max(trade_date) as max_trade_date
                from industry_daily_bars
                group by source
                order by source
                """,
            )
        if table_exists(conn, "symbols"):
            summary["symbols"] = sqlite_rows(
                conn,
                "select count(*) as rows, min(fetched_at) as min_fetched_at, max(fetched_at) as max_fetched_at from symbols",
            )[0]
        if table_exists(conn, "fetch_status"):
            fetch_source_expr = (
                "coalesce(source_used, '')"
                if column_exists(conn, "fetch_status", "source_used")
                else "''"
            )
            summary["fetch_status"] = sqlite_rows(
                conn,
                f"""
                select adjust, last_status,
                       {fetch_source_expr} as source_used,
                       count(*) as rows, sum(rows_fetched) as rows_fetched,
                       max(fetched_at) as max_fetched_at
                from fetch_status
                group by adjust, last_status, source_used
                order by adjust, last_status, source_used
                """,
            )
            summary["fetch_failures"] = sqlite_rows(
                conn,
                """
                select symbol, adjust, requested_start, requested_end, message, fetched_at
                from fetch_status
                where last_status != 'ok'
                order by fetched_at desc
                limit 20
                """,
            )
        if table_exists(conn, "adj_factors"):
            summary["adj_factors"] = sqlite_rows(
                conn,
                """
                select count(*) as rows, count(distinct symbol) as symbols,
                       min(trade_date) as min_trade_date, max(trade_date) as max_trade_date
                from adj_factors
                """,
            )[0]
        if table_exists(conn, "symbol_lifecycle"):
            summary["symbol_lifecycle"] = sqlite_rows(
                conn,
                """
                select count(*) as rows, count(distinct symbol) as symbols,
                       sum(case when list_date is not null then 1 else 0 end) as with_list_date,
                       sum(case when delist_date is not null then 1 else 0 end) as with_delist_date
                from symbol_lifecycle
                """,
            )[0]
        if table_exists(conn, "symbol_status_daily"):
            summary["symbol_status_daily"] = sqlite_rows(
                conn,
                """
                select count(*) as rows, count(distinct symbol) as symbols,
                       min(trade_date) as min_trade_date, max(trade_date) as max_trade_date,
                       sum(case when is_st = 1 then 1 else 0 end) as st_rows,
                       sum(case when is_suspended = 1 then 1 else 0 end) as suspended_rows
                from symbol_status_daily
                """,
            )[0]
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def collect_manifest(
    db: Path,
    argv: Iterable[str],
    script_paths: Iterable[Path],
    outputs: Mapping[str, Path | str],
    extra: Mapping[str, Any] | None = None,
    cache_registry_path: Path | None = Path("data/cache/cache_registry.json"),
) -> dict[str, Any]:
    argv_list = [str(item) for item in argv]
    scripts = []
    for path in script_paths:
        script_path = Path(path)
        scripts.append(
            {
                "path": str(script_path),
                "exists": script_path.exists(),
                "sha256": file_sha256(script_path),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": argv_list,
        "command_line": " ".join(shlex.quote(item) for item in argv_list),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "akshare": package_version("akshare"),
            "baostock": package_version("baostock"),
        },
        "scripts": scripts,
        "database": database_summary(db),
        "cache_registry": cache_registry_summary(cache_registry_path) if cache_registry_path is not None else None,
        "outputs": _json_safe(outputs),
        "extra": _json_safe(extra or {}),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

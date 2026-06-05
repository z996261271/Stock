"""Cache registry helpers for local factor/data bundle files."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


CACHE_NAME_RE = re.compile(
    r"(?P<cache_key>walkforward_factors(?:_v\d+)?)"
    r"(?:_(?P<board_scope>main|all))?"
    r"(?:_(?P<factor_adjust>raw|qfq|hfq))?"
    r"(?:_(?P<mode>strict|fallback))?"
    r"_(?P<start_date>\d{8})_(?P<end_date>\d{8})"
)


@dataclass(frozen=True)
class CacheRegistryEntry:
    cache_key: str
    file_path: str
    bytes: int
    modified_at: str
    sha256_head: str
    factor_adjust: str | None = None
    board_scope: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    mode: str | None = None
    source_tables: list[str] = field(default_factory=list)
    source_table_max_dates: dict[str, str | None] = field(default_factory=dict)
    freshness_status: str = "unknown"
    stale_reasons: list[str] = field(default_factory=list)
    script_hash: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    invalidated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scan_cache_dir(
    cache_dir: Path,
    *,
    db: Path | None = None,
    source_tables: list[str] | None = None,
    script_path: Path | None = None,
) -> list[CacheRegistryEntry]:
    """Return registry entries for local cache files without loading cache payloads."""
    if not cache_dir.exists():
        return []
    source_tables = source_tables or ["daily_bars", "symbol_status_daily", "symbol_lifecycle", "symbol_industries"]
    max_dates = source_table_max_dates(db, source_tables) if db else {}
    script_hash = file_sha256(script_path) if script_path and script_path.exists() else None
    entries: list[CacheRegistryEntry] = []
    for path in sorted(cache_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "cache_registry.json":
            continue
        parsed = parse_cache_filename(path)
        stat = path.stat()
        end_date = _date_token(parsed.get("end_date"))
        stale_reasons = cache_stale_reasons(end_date, max_dates)
        entries.append(
            CacheRegistryEntry(
                cache_key=parsed.get("cache_key") or path.stem,
                file_path=str(path),
                bytes=int(stat.st_size),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                sha256_head=file_sha256(path, limit_bytes=1024 * 1024),
                factor_adjust=parsed.get("factor_adjust"),
                board_scope=parsed.get("board_scope"),
                start_date=_date_token(parsed.get("start_date")),
                end_date=end_date,
                mode=parsed.get("mode"),
                source_tables=source_tables,
                source_table_max_dates=max_dates,
                freshness_status=_freshness_status(end_date, stale_reasons),
                stale_reasons=stale_reasons,
                script_hash=script_hash,
            )
        )
    return entries


def parse_cache_filename(path: Path) -> dict[str, str | None]:
    match = CACHE_NAME_RE.search(path.stem)
    return match.groupdict() if match else {}


def source_table_max_dates(db: Path | None, source_tables: list[str]) -> dict[str, str | None]:
    if db is None or not db.exists():
        return {table: None for table in source_tables}
    result: dict[str, str | None] = {}
    with sqlite3.connect(db) as conn:
        existing = {
            row[0]
            for row in conn.execute("select name from sqlite_master where type = 'table'").fetchall()
        }
        for table in source_tables:
            if table not in existing:
                result[table] = None
                continue
            column = _date_column(table)
            try:
                row = conn.execute(f"select max({column}) from {table}").fetchone()
            except sqlite3.Error:
                result[table] = None
            else:
                result[table] = str(row[0]) if row and row[0] else None
    return result


def cache_stale_reasons(cache_end_date: str | None, source_max_dates: dict[str, str | None]) -> list[str]:
    """Return source-table reasons that make a dated cache older than local trade data."""
    if not cache_end_date:
        return []
    cache_date = _date_prefix(cache_end_date)
    if not cache_date:
        return []
    reasons: list[str] = []
    for table, max_date in sorted(source_max_dates.items()):
        if _date_column(table) != "trade_date":
            continue
        source_date = _date_prefix(max_date)
        if source_date and source_date > cache_date:
            reasons.append(f"{table}:source_max_date={source_date}>cache_end_date={cache_date}")
    return reasons


def write_registry(entries: list[CacheRegistryEntry], output: Path) -> dict[str, Any]:
    registry = {
        "schema_version": "cache_registry.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entry_count": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return registry


def file_sha256(path: Path, *, limit_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if limit_bytes is None:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        else:
            digest.update(handle.read(limit_bytes))
    return digest.hexdigest()


def _date_token(value: str | None) -> str | None:
    if not value or len(value) != 8:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _date_prefix(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value)
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return _date_token(text[:8]) if len(text) >= 8 and text[:8].isdigit() else None


def _freshness_status(cache_end_date: str | None, stale_reasons: list[str]) -> str:
    if stale_reasons:
        return "stale"
    return "current" if cache_end_date else "unknown"


def _date_column(table: str) -> str:
    if table in {"symbol_lifecycle", "symbol_industries"}:
        return "fetched_at"
    return "trade_date"

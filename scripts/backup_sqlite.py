#!/usr/bin/env python3
"""Create a consistent SQLite backup plus a small manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Back up a SQLite database with sqlite3.Connection.backup().")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("backups"))
    parser.add_argument("--tag", default="manual")
    return parser.parse_args()


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = [
        row[0]
        for row in conn.execute(
            "select name from sqlite_master where type = 'table' and name not like 'sqlite_%' order by name"
        )
    ]
    counts: dict[str, int] = {}
    for table in tables:
        counts[table] = int(conn.execute(f'select count(*) from "{table}"').fetchone()[0])
    return counts


def backup_sqlite(db: Path, output_dir: Path, tag: str) -> dict[str, Any]:
    if not db.exists():
        raise FileNotFoundError(db)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in tag).strip("-") or "manual"
    backup_path = output_dir / f"{db.stem}_{safe_tag}_{timestamp}.sqlite3"
    manifest_path = backup_path.with_suffix(".manifest.json")

    with sqlite3.connect(db) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
        source_counts = table_counts(source)
        target_counts = table_counts(target)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(db),
        "backup": str(backup_path),
        "tag": tag,
        "source_size_bytes": db.stat().st_size,
        "backup_size_bytes": backup_path.stat().st_size,
        "backup_sha256": file_sha256(backup_path),
        "source_table_counts": source_counts,
        "backup_table_counts": target_counts,
        "counts_match": source_counts == target_counts,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = backup_sqlite(args.db, args.output_dir, args.tag)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

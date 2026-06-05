"""Stock-to-industry metadata helpers."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from professional_quant.risk.exposure import UNKNOWN_INDUSTRY


INDUSTRY_TABLE = "symbol_industries"
INDUSTRY_COLUMNS = (
    "symbol",
    "industry_name",
    "industry_code",
    "industry_level",
    "provider",
    "source",
    "fetched_at",
)


def normalize_symbol(value: Any) -> str:
    """Normalize a stock code to a six-character A-share symbol when possible."""
    if value is None:
        return ""
    text = str(value).strip()
    if "." in text and len(text.split(".")[0]) >= 6:
        text = text.split(".")[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    if 0 < len(digits) < 6:
        return digits.zfill(6)
    return text


def ensure_symbol_industry_schema(conn: sqlite3.Connection) -> None:
    """Create the canonical stock-industry mapping table."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbol_industries (
            symbol TEXT PRIMARY KEY,
            industry_name TEXT NOT NULL,
            industry_code TEXT,
            industry_level TEXT,
            provider TEXT NOT NULL DEFAULT 'local',
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_symbol_industries_industry
            ON symbol_industries (industry_name);
        """
    )


def load_industry_rows(path: Path, provider: str, source: str) -> list[dict[str, Any]]:
    """Load stock-industry rows from CSV or JSON."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("industries", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError("industry JSON must be a list or an object with an industries list")
    else:
        rows = pd.read_csv(path).to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        symbol = normalize_symbol(row.get("symbol", row.get("代码", row.get("证券代码"))))
        industry_name = str(
            row.get(
                "industry_name",
                row.get("industry", row.get("sector", row.get("行业", row.get("板块名称", "")))),
            )
            or ""
        ).strip()
        if not symbol or not industry_name:
            continue
        normalized.append(
            {
                "symbol": symbol,
                "industry_name": industry_name,
                "industry_code": _empty_to_none(row.get("industry_code", row.get("板块代码", row.get("code")))),
                "industry_level": str(row.get("industry_level", row.get("level", "")) or "").strip() or None,
                "provider": str(row.get("provider", provider) or provider),
                "source": str(row.get("source", source) or source),
                "fetched_at": str(row.get("fetched_at") or datetime.now().isoformat(timespec="seconds")),
            }
        )
    return normalized


def upsert_symbol_industries(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    """Upsert normalized stock-industry rows and return the inserted row count."""
    ensure_symbol_industry_schema(conn)
    payload = [
        (
            row["symbol"],
            row["industry_name"],
            row.get("industry_code"),
            row.get("industry_level"),
            row.get("provider") or "local",
            row.get("source") or "unknown",
            row.get("fetched_at") or datetime.now().isoformat(timespec="seconds"),
        )
        for row in rows
        if row.get("symbol") and row.get("industry_name")
    ]
    if not payload:
        return 0
    conn.executemany(
        """
        INSERT INTO symbol_industries (
            symbol, industry_name, industry_code, industry_level, provider, source, fetched_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            industry_name=excluded.industry_name,
            industry_code=excluded.industry_code,
            industry_level=excluded.industry_level,
            provider=excluded.provider,
            source=excluded.source,
            fetched_at=excluded.fetched_at
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def sqlite_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}


def load_symbol_industry_map(db: Path) -> dict[str, str]:
    """Return a symbol -> real industry label map from local metadata."""
    candidates = [
        ("symbol_industries", ("industry_name", "industry_code")),
        ("symbol_industry", ("industry_name", "industry_code", "board_name", "board_code")),
        ("symbols", ("industry_name", "industry", "sector")),
    ]
    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute("select name from sqlite_master where type = 'table'")
        }
        for table_name, columns in candidates:
            if table_name not in tables:
                continue
            available = sqlite_columns(conn, table_name)
            label_column = next((column for column in columns if column in available), None)
            if not label_column or "symbol" not in available:
                continue
            rows = conn.execute(
                f"""
                select symbol, {label_column}
                from {table_name}
                where {label_column} is not null
                  and trim(cast({label_column} as text)) != ''
                """
            ).fetchall()
            mapping = {normalize_symbol(symbol): str(label) for symbol, label in rows if normalize_symbol(symbol)}
            if mapping:
                return mapping
    return {}


def apply_industry_labels(df: pd.DataFrame, industry_map: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach an industry_label column and return coverage metadata."""
    frame = df.copy()
    had_industry_column = "industry_label" in frame.columns
    if had_industry_column:
        labels = frame["industry_label"].where(frame["industry_label"].notna(), UNKNOWN_INDUSTRY).astype(str)
        source = "data_column"
    elif industry_map:
        labels = frame["symbol"].astype(str).map(industry_map).fillna(UNKNOWN_INDUSTRY).astype(str)
        source = "symbol_industries"
    else:
        labels = pd.Series(UNKNOWN_INDUSTRY, index=frame.index, dtype=str)
        source = "missing"
    labels = labels.replace({"": UNKNOWN_INDUSTRY, "nan": UNKNOWN_INDUSTRY, "None": UNKNOWN_INDUSTRY})
    frame["industry_label"] = labels
    known_mask = frame["industry_label"] != UNKNOWN_INDUSTRY
    symbols = frame["symbol"].astype(str)
    known_symbols = symbols[known_mask].nunique()
    total_symbols = symbols.nunique()
    return frame, {
        "source": source,
        "symbols": int(total_symbols),
        "known_symbols": int(known_symbols),
        "missing_symbols": int(max(total_symbols - known_symbols, 0)),
        "coverage_ratio": float(known_symbols / total_symbols) if total_symbols else 0.0,
        "unique_industries": int(frame.loc[known_mask, "industry_label"].nunique()),
        "is_real_industry_mapping": bool(source == "symbol_industries"),
    }


def industry_coverage(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope_sql: str,
    board_params: tuple[Any, ...],
) -> dict[str, Any]:
    """Return symbol-level coverage of the canonical industry mapping table."""
    start_text = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if not _table_exists(conn, INDUSTRY_TABLE):
        expected = _expected_symbols(conn, start_text, end_text, board_scope_sql, board_params)
        return {
            "requested_start": start_text,
            "requested_end": end_text,
            "expected_symbols": expected,
            "covered_symbols": 0,
            "missing_symbols": expected,
            "coverage_ratio": 0.0,
            "table": INDUSTRY_TABLE,
        }
    row = conn.execute(
        f"""
        with expected as (
            select distinct d.symbol
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and d.open is not null
              and d.high is not null
              and d.low is not null
              and d.close is not null
              and d.amount is not null
              {board_scope_sql}
        )
        select count(*) as expected_symbols,
               sum(case when i.symbol is not null then 1 else 0 end) as covered_symbols
        from expected
        left join symbol_industries i on i.symbol = expected.symbol
        """,
        (start_text, end_text, *board_params),
    ).fetchone()
    expected = int(row[0] or 0)
    covered = int(row[1] or 0)
    return {
        "requested_start": start_text,
        "requested_end": end_text,
        "expected_symbols": expected,
        "covered_symbols": covered,
        "missing_symbols": max(expected - covered, 0),
        "coverage_ratio": float(covered / expected) if expected else 0.0,
        "table": INDUSTRY_TABLE,
    }


def _expected_symbols(
    conn: sqlite3.Connection,
    start_text: str,
    end_text: str,
    board_scope_sql: str,
    board_params: tuple[Any, ...],
) -> int:
    row = conn.execute(
        f"""
        select count(distinct d.symbol)
        from daily_bars d
        where d.adjust = 'raw'
          and d.trade_date >= ?
          and d.trade_date <= ?
          and d.open is not null
          and d.high is not null
          and d.low is not null
          and d.close is not null
          and d.amount is not null
          {board_scope_sql}
        """,
        (start_text, end_text, *board_params),
    ).fetchone()
    return int(row[0] or 0)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table_name,),
    ).fetchone() is not None


def _empty_to_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None

#!/usr/bin/env python3
"""Canonical SQLite schema helpers for data, research, and paper-trading state."""

from __future__ import annotations

import sqlite3
from pathlib import Path


CANONICAL_ADJUSTS = ("raw", "qfq", "hfq")

REFERENCE_TABLES = (
    "adj_factors",
    "symbol_lifecycle",
    "symbol_status_daily",
    "symbol_industries",
)

STATE_TABLES = (
    "signal_runs",
    "paper_run_registry",
    "signals",
    "positions",
    "alerts",
    "alert_attempts",
    "paper_trades",
)


def ensure_quant_schema(conn: sqlite3.Connection) -> None:
    """Create the canonical auxiliary schema without mutating existing rows."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS adj_factors (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            adj_factor REAL,
            forward_factor REAL,
            backward_factor REAL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (symbol, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_adj_factors_date
            ON adj_factors (trade_date);

        CREATE TABLE IF NOT EXISTS symbol_lifecycle (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            list_date TEXT,
            delist_date TEXT,
            board TEXT,
            market TEXT,
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_symbol_lifecycle_board
            ON symbol_lifecycle (board);

        CREATE TABLE IF NOT EXISTS symbol_status_daily (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            is_st INTEGER,
            is_suspended INTEGER,
            board TEXT,
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_symbol_status_daily_date
            ON symbol_status_daily (trade_date);

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

        CREATE TABLE IF NOT EXISTS market_valuation_daily (
            trade_date TEXT PRIMARY KEY,
            middle_pe_ttm REAL,
            average_pe_ttm REAL,
            middle_pe_lyr REAL,
            average_pe_lyr REAL,
            middle_pb REAL,
            average_pb REAL,
            pe_close REAL,
            pb_close REAL,
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS symbol_valuation_daily (
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            pe_ttm REAL,
            pe_static REAL,
            pb REAL,
            pcf REAL,
            total_market_cap REAL,
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_symbol_valuation_daily_date
            ON symbol_valuation_daily (trade_date);

        CREATE TABLE IF NOT EXISTS symbol_financial_indicator (
            symbol TEXT NOT NULL,
            report_date TEXT NOT NULL,
            notice_date TEXT NOT NULL,
            update_date TEXT,
            report_type TEXT,
            report_year INTEGER,
            roe REAL,
            roic REAL,
            gross_margin REAL,
            net_margin REAL,
            asset_return REAL,
            debt_asset_ratio REAL,
            revenue_growth_yoy REAL,
            profit_growth_yoy REAL,
            deduct_profit_growth_yoy REAL,
            operating_cashflow_to_revenue REAL,
            source TEXT NOT NULL DEFAULT 'unknown',
            fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, report_date, notice_date)
        );

        CREATE INDEX IF NOT EXISTS idx_symbol_financial_indicator_notice
            ON symbol_financial_indicator (notice_date);

        CREATE TABLE IF NOT EXISTS signals (
            signal_id TEXT PRIMARY KEY,
            run_id TEXT,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            action TEXT NOT NULL,
            score REAL,
            weight REAL,
            reason TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_signals_strategy_date
            ON signals (strategy, signal_date);

        CREATE TABLE IF NOT EXISTS signal_runs (
            run_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT,
            message TEXT,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_signal_runs_strategy_date
            ON signal_runs (strategy, signal_date);

        CREATE TABLE IF NOT EXISTS paper_run_registry (
            run_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            entry_date TEXT,
            status TEXT NOT NULL,
            picks_path TEXT,
            observation_days INTEGER,
            data_fresh INTEGER,
            manual_confirmation_required INTEGER,
            manual_confirmed_at TEXT,
            manifest_json TEXT,
            message TEXT,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_paper_run_registry_strategy_date
            ON paper_run_registry (strategy, signal_date);

        CREATE TABLE IF NOT EXISTS positions (
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            quantity REAL NOT NULL,
            avg_cost REAL,
            market_value REAL,
            cash REAL,
            payload_json TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (strategy, symbol, as_of_date)
        );

        CREATE INDEX IF NOT EXISTS idx_positions_strategy_date
            ON positions (strategy, as_of_date);

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            alert_date TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            message TEXT,
            payload_json TEXT,
            acknowledged_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_strategy_date
            ON alerts (strategy, alert_date);

        CREATE TABLE IF NOT EXISTS alert_attempts (
            attempt_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            alert_id TEXT NOT NULL DEFAULT '',
            channel TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 1,
            message TEXT,
            payload_json TEXT,
            attempted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (run_id, channel, alert_id)
        );

        CREATE INDEX IF NOT EXISTS idx_alert_attempts_run_id
            ON alert_attempts (run_id);

        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_id TEXT,
            trade_date TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL,
            amount REAL,
            status TEXT NOT NULL,
            reason TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_paper_trades_strategy_date
            ON paper_trades (strategy, trade_date);
        """
    )
    _ensure_column(conn, "adj_factors", "forward_factor", "REAL")
    _ensure_column(conn, "adj_factors", "backward_factor", "REAL")
    _ensure_column(conn, "signals", "run_id", "TEXT")
    if table_exists(conn, "signals"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_run_id ON signals (run_id)")
    conn.commit()


def ensure_quant_schema_path(db: Path) -> None:
    """Open *db* and create the canonical auxiliary schema."""
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    """Add a nullable column to an existing table if an older DB lacks it."""
    if not table_exists(conn, table_name):
        return
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

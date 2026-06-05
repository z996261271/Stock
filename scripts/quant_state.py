#!/usr/bin/env python3
"""Helpers for durable signal, position, alert, and paper-trade state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Mapping

from quant_schema import ensure_quant_schema


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_text(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def stable_id(prefix: str, *parts: Any) -> str:
    """Return a stable short ID for idempotent state upserts."""
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def record_signal_run(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    signal_date: str,
    status: str = "started",
    config: Mapping[str, Any] | None = None,
    message: str | None = None,
    run_id: str | None = None,
    finished_at: str | None = None,
) -> str:
    """Insert or update one idempotent signal-generation run."""
    ensure_quant_schema(conn)
    run_id = run_id or stable_id("run", strategy, signal_date)
    conn.execute(
        """
        INSERT INTO signal_runs (
            run_id, strategy, signal_date, status, config_json, message, started_at, finished_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            strategy = excluded.strategy,
            signal_date = excluded.signal_date,
            status = excluded.status,
            config_json = excluded.config_json,
            message = excluded.message,
            finished_at = excluded.finished_at
        """,
        (run_id, strategy, signal_date, status, _json_text(config), message, _now_text(), finished_at),
    )
    conn.commit()
    return run_id


def record_paper_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    strategy: str,
    signal_date: str,
    entry_date: str | None,
    status: str,
    picks_path: str | None = None,
    observation_days: int | None = None,
    data_fresh: bool | None = None,
    manual_confirmation_required: bool | None = None,
    manual_confirmed_at: str | None = None,
    manifest: Mapping[str, Any] | None = None,
    message: str | None = None,
    finished_at: str | None = None,
) -> str:
    """Upsert one paper pipeline run registry row."""
    ensure_quant_schema(conn)
    conn.execute(
        """
        INSERT INTO paper_run_registry (
            run_id, strategy, signal_date, entry_date, status, picks_path, observation_days,
            data_fresh, manual_confirmation_required, manual_confirmed_at, manifest_json,
            message, started_at, finished_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            strategy = excluded.strategy,
            signal_date = excluded.signal_date,
            entry_date = excluded.entry_date,
            status = excluded.status,
            picks_path = excluded.picks_path,
            observation_days = excluded.observation_days,
            data_fresh = excluded.data_fresh,
            manual_confirmation_required = excluded.manual_confirmation_required,
            manual_confirmed_at = excluded.manual_confirmed_at,
            manifest_json = excluded.manifest_json,
            message = excluded.message,
            finished_at = excluded.finished_at,
            updated_at = excluded.updated_at
        """,
        (
            run_id,
            strategy,
            signal_date,
            entry_date,
            status,
            picks_path,
            observation_days,
            _bool_int(data_fresh),
            _bool_int(manual_confirmation_required),
            manual_confirmed_at,
            _json_text(manifest),
            message,
            _now_text(),
            finished_at,
            _now_text(),
        ),
    )
    conn.commit()
    return run_id


def record_signal(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    symbol: str,
    signal_date: str,
    action: str,
    score: float | None = None,
    weight: float | None = None,
    reason: str | None = None,
    payload: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Insert or replace one strategy signal and return its ID."""
    ensure_quant_schema(conn)
    signal_id = signal_id or stable_id("sig", run_id or "", strategy, symbol, signal_date, action)
    conn.execute(
        """
        INSERT INTO signals (
            signal_id, run_id, strategy, symbol, signal_date, action, score, weight, reason, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(signal_id) DO UPDATE SET
            run_id = excluded.run_id,
            strategy = excluded.strategy,
            symbol = excluded.symbol,
            signal_date = excluded.signal_date,
            action = excluded.action,
            score = excluded.score,
            weight = excluded.weight,
            reason = excluded.reason,
            payload_json = excluded.payload_json
        """,
        (
            signal_id,
            run_id,
            strategy,
            symbol,
            signal_date,
            action,
            score,
            weight,
            reason,
            _json_text(payload),
            _now_text(),
        ),
    )
    conn.commit()
    return signal_id


def record_position(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    symbol: str,
    as_of_date: str,
    quantity: float,
    avg_cost: float | None = None,
    market_value: float | None = None,
    cash: float | None = None,
    payload: Mapping[str, Any] | None = None,
) -> None:
    """Upsert one position snapshot."""
    ensure_quant_schema(conn)
    conn.execute(
        """
        INSERT INTO positions (
            strategy, symbol, as_of_date, quantity, avg_cost, market_value, cash, payload_json, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy, symbol, as_of_date) DO UPDATE SET
            quantity = excluded.quantity,
            avg_cost = excluded.avg_cost,
            market_value = excluded.market_value,
            cash = excluded.cash,
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
        """,
        (strategy, symbol, as_of_date, quantity, avg_cost, market_value, cash, _json_text(payload), _now_text()),
    )
    conn.commit()


def record_alert(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    alert_date: str,
    severity: str,
    title: str,
    message: str | None = None,
    payload: Mapping[str, Any] | None = None,
    alert_id: str | None = None,
) -> str:
    """Insert or replace one alert and return its ID."""
    ensure_quant_schema(conn)
    alert_id = alert_id or stable_id("alert", strategy, alert_date, severity, title)
    conn.execute(
        """
        INSERT INTO alerts (
            alert_id, strategy, alert_date, severity, title, message, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(alert_id) DO UPDATE SET
            strategy = excluded.strategy,
            alert_date = excluded.alert_date,
            severity = excluded.severity,
            title = excluded.title,
            message = excluded.message,
            payload_json = excluded.payload_json
        """,
        (alert_id, strategy, alert_date, severity, title, message, _json_text(payload), _now_text()),
    )
    conn.commit()
    return alert_id


def record_alert_attempt(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    channel: str,
    status: str,
    alert_id: str | None = None,
    message: str | None = None,
    payload: Mapping[str, Any] | None = None,
    attempt_id: str | None = None,
) -> str:
    """Upsert one alert delivery attempt and increment retries for the same run/channel/alert."""
    ensure_quant_schema(conn)
    normalized_alert_id = alert_id or ""
    attempt_id = attempt_id or stable_id("attempt", run_id, channel, normalized_alert_id)
    conn.execute(
        """
        INSERT INTO alert_attempts (
            attempt_id, run_id, alert_id, channel, status, attempt_count, message, payload_json, attempted_at
        )
        VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(run_id, channel, alert_id) DO UPDATE SET
            status = excluded.status,
            attempt_count = alert_attempts.attempt_count + 1,
            message = excluded.message,
            payload_json = excluded.payload_json,
            attempted_at = excluded.attempted_at
        """,
        (attempt_id, run_id, normalized_alert_id, channel, status, message, _json_text(payload), _now_text()),
    )
    conn.commit()
    return attempt_id


def record_paper_trade(
    conn: sqlite3.Connection,
    *,
    strategy: str,
    symbol: str,
    trade_date: str,
    side: str,
    quantity: float,
    price: float | None = None,
    amount: float | None = None,
    status: str = "planned",
    signal_id: str | None = None,
    reason: str | None = None,
    payload: Mapping[str, Any] | None = None,
    trade_id: str | None = None,
) -> str:
    """Insert or replace one paper-trade record and return its ID."""
    ensure_quant_schema(conn)
    trade_id = trade_id or stable_id("ptrade", strategy, symbol, trade_date, side, signal_id or "")
    conn.execute(
        """
        INSERT INTO paper_trades (
            trade_id, strategy, symbol, signal_id, trade_date, side, quantity, price, amount,
            status, reason, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_id) DO UPDATE SET
            strategy = excluded.strategy,
            symbol = excluded.symbol,
            signal_id = excluded.signal_id,
            trade_date = excluded.trade_date,
            side = excluded.side,
            quantity = excluded.quantity,
            price = excluded.price,
            amount = excluded.amount,
            status = excluded.status,
            reason = excluded.reason,
            payload_json = excluded.payload_json
        """,
        (
            trade_id,
            strategy,
            symbol,
            signal_id,
            trade_date,
            side,
            quantity,
            price,
            amount,
            status,
            reason,
            _json_text(payload),
            _now_text(),
        ),
    )
    conn.commit()
    return trade_id


def state_counts(conn: sqlite3.Connection, strategy: str | None = None) -> dict[str, int]:
    """Return state row counts, optionally scoped to a strategy."""
    ensure_quant_schema(conn)
    tables = (
        "signal_runs",
        "paper_run_registry",
        "signals",
        "positions",
        "alerts",
        "alert_attempts",
        "paper_trades",
    )
    counts: dict[str, int] = {}
    for table in tables:
        if strategy and table == "alert_attempts":
            counts[table] = int(
                conn.execute(
                    """
                    select count(*)
                    from alert_attempts a
                    join signal_runs r on r.run_id = a.run_id
                    where r.strategy = ?
                    """,
                    (strategy,),
                ).fetchone()[0]
            )
        elif strategy:
            counts[table] = int(
                conn.execute(f"select count(*) from {table} where strategy = ?", (strategy,)).fetchone()[0]
            )
        else:
            counts[table] = int(conn.execute(f"select count(*) from {table}").fetchone()[0])
    return counts


def _bool_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0

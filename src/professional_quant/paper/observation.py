"""Paper-trading observation window audit helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


DEFAULT_OBSERVATION_DAYS = 60

RAW_TRADING_DAY_QUERY = """
select distinct trade_date
from daily_bars
where adjust = 'raw'
  and trade_date <= ?
order by trade_date desc
limit ?
"""


def rows(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def parse_config_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {"_invalid_config_json": True}
    return loaded if isinstance(loaded, dict) else {}


def parse_bool_int(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def expected_trading_dates(conn: sqlite3.Connection, anchor_date: str | None, days: int) -> list[str]:
    """Return latest raw trading dates ending at anchor_date when market data is available."""
    if not anchor_date or not table_exists(conn, "daily_bars"):
        return []
    return [
        str(row["trade_date"])
        for row in rows(
            conn,
            RAW_TRADING_DAY_QUERY,
            (anchor_date, max(days, 1)),
        )
    ]


def observation_audit(conn: sqlite3.Connection, strategy: str | None, days: int) -> dict[str, Any]:
    """Audit paper-observation continuity and duplicate/failed delivery risk."""
    where = "where strategy = ?" if strategy else ""
    params = (strategy,) if strategy else ()
    run_dates = rows(
        conn,
        f"""
        select signal_date,
               count(*) as runs,
               sum(case when status = 'finished' then 1 else 0 end) as finished_runs,
               sum(case when status != 'finished' then 1 else 0 end) as unfinished_runs
        from signal_runs
        {where}
        group by signal_date
        order by signal_date desc
        limit ?
        """,
        (*params, max(days, 1)),
    )
    covered_dates = [row["signal_date"] for row in run_dates]
    latest_signal_date = covered_dates[0] if covered_dates else None
    expected_dates = expected_trading_dates(conn, latest_signal_date, days)
    covered_set = set(covered_dates)
    missing_expected_dates = [date for date in expected_dates if date not in covered_set]
    duplicate_attempts = rows(
        conn,
        f"""
        select a.run_id, a.channel, a.alert_id, a.attempt_count, a.status
        from alert_attempts a
        join signal_runs r on r.run_id = a.run_id
        {"where r.strategy = ?" if strategy else ""}
          {"and" if strategy else "where"} a.attempt_count > 1
        order by a.attempted_at desc
        limit 20
        """,
        params,
    )
    failed_runs = rows(
        conn,
        f"""
        select run_id, strategy, signal_date, status, message
        from signal_runs
        {"where strategy = ? and status != 'finished'" if strategy else "where status != 'finished'"}
        order by signal_date desc
        limit 20
        """,
        params,
    )
    dates_with_no_signals = rows(
        conn,
        f"""
        select r.signal_date, r.run_id
        from signal_runs r
        left join signals s on s.run_id = r.run_id
        {"where r.strategy = ?" if strategy else ""}
        group by r.run_id
        having count(s.signal_id) = 0
        order by r.signal_date desc
        limit 20
        """,
        params,
    )
    run_detail_rows = rows(
        conn,
        f"""
        select run_id, strategy, signal_date, status, config_json, started_at, finished_at
        from signal_runs
        {where}
        order by signal_date desc
        limit ?
        """,
        (*params, max(days, 1)),
    )
    manual_confirmation_missing: list[dict[str, Any]] = []
    delayed_runs: list[dict[str, Any]] = []
    for row in run_detail_rows:
        config = parse_config_json(row.get("config_json"))
        if config.get("manual_confirmation_required") is True and not config.get("manual_confirmed_at"):
            manual_confirmation_missing.append(
                {
                    "run_id": row.get("run_id"),
                    "strategy": row.get("strategy"),
                    "signal_date": row.get("signal_date"),
                    "status": row.get("status"),
                }
            )
        delay = config.get("data_delay_days")
        try:
            delay_days = int(float(delay)) if delay is not None else 0
        except (TypeError, ValueError):
            delay_days = 0
        if delay_days > 0:
            delayed_runs.append(
                {
                    "run_id": row.get("run_id"),
                    "strategy": row.get("strategy"),
                    "signal_date": row.get("signal_date"),
                    "data_delay_days": delay_days,
                    "latest_raw_trade_date": config.get("latest_raw_trade_date"),
                }
            )
    observed = len(covered_dates)
    has_consecutive_evidence = bool(expected_dates)
    readiness_dates = expected_dates if expected_dates else covered_dates
    window_60 = set(readiness_dates[:60])
    window_90 = set(readiness_dates[:90])
    manual_confirmation_missing_60 = [row for row in manual_confirmation_missing if row.get("signal_date") in window_60]
    manual_confirmation_missing_90 = [row for row in manual_confirmation_missing if row.get("signal_date") in window_90]
    delayed_runs_60 = [row for row in delayed_runs if row.get("signal_date") in window_60]
    delayed_runs_90 = [row for row in delayed_runs if row.get("signal_date") in window_90]
    missing_expected_60 = [date for date in expected_dates[:60] if date not in covered_set]
    missing_expected_90 = [date for date in expected_dates[:90] if date not in covered_set]
    observed_expected_days = len(expected_dates) - len(missing_expected_dates)
    ready_60 = (
        observed >= 60
        and not failed_runs
        and not dates_with_no_signals
        and not manual_confirmation_missing_60
        and not delayed_runs_60
    )
    ready_90 = (
        observed >= 90
        and not failed_runs
        and not dates_with_no_signals
        and not manual_confirmation_missing_90
        and not delayed_runs_90
    )
    if has_consecutive_evidence:
        ready_60 = (
            len(expected_dates) >= 60
            and len(expected_dates[:60]) - len(missing_expected_60) >= 60
            and not missing_expected_60
            and not failed_runs
            and not dates_with_no_signals
            and not manual_confirmation_missing_60
            and not delayed_runs_60
        )
        ready_90 = (
            len(expected_dates) >= 90
            and len(expected_dates[:90]) - len(missing_expected_90) >= 90
            and not missing_expected_90
            and not failed_runs
            and not dates_with_no_signals
            and not manual_confirmation_missing_90
            and not delayed_runs_90
        )
    return {
        "requested_days": int(days),
        "observed_distinct_signal_dates": int(observed),
        "oldest_signal_date": covered_dates[-1] if covered_dates else None,
        "latest_signal_date": latest_signal_date,
        "expected_trading_dates_available": has_consecutive_evidence,
        "expected_trading_dates": expected_dates,
        "missing_expected_signal_dates": missing_expected_dates,
        "missing_expected_signal_dates_60d": missing_expected_60,
        "missing_expected_signal_dates_90d": missing_expected_90,
        "observed_expected_trading_dates": int(observed_expected_days),
        "is_60_day_ready": bool(ready_60),
        "is_90_day_ready": bool(ready_90),
        "run_date_rows": run_dates,
        "failed_or_unfinished_runs": failed_runs,
        "runs_without_signals": dates_with_no_signals,
        "duplicate_or_retry_attempts": duplicate_attempts,
        "manual_confirmation_missing": manual_confirmation_missing,
        "manual_confirmation_missing_60d": manual_confirmation_missing_60,
        "manual_confirmation_missing_90d": manual_confirmation_missing_90,
        "delayed_data_runs": delayed_runs,
        "delayed_data_runs_60d": delayed_runs_60,
        "delayed_data_runs_90d": delayed_runs_90,
        "audit_note": (
            "Default paper reporting audits the latest 60 raw trading dates; 90-day readiness is still available with "
            "--observation-days 90. Readiness requires finished runs with signals, no duplicate/retry pressure, no "
            "failed runs, no data-delay runs, and completed manual confirmation on consecutive raw trading dates when "
            "daily_bars is available; without market dates it falls back to distinct stored signal dates."
        ),
    }

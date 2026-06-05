#!/usr/bin/env python3
"""Data quality primitives shared by research and reporting scripts."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_schema import CANONICAL_ADJUSTS, REFERENCE_TABLES, STATE_TABLES, table_exists  # noqa: E402
from quant_universe import board_scope_sql  # noqa: E402
from professional_quant.data.industry import industry_coverage  # noqa: E402


REQUIRED_RAW_COLUMNS = ("open", "high", "low", "close", "amount")
FORMAL_STATUS_COVERAGE_MIN = 0.995
FORMAL_INDUSTRY_COVERAGE_MIN = 0.995


def _date_text(value: pd.Timestamp | str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def adjustment_coverage(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
    factor_adjust: str,
    include_distinct: bool = False,
) -> dict[str, Any]:
    """Return row-level coverage of *factor_adjust* against usable raw bars."""
    if factor_adjust not in CANONICAL_ADJUSTS:
        raise ValueError(f"unknown factor_adjust: {factor_adjust}")

    board_clause, board_params = board_scope_sql(board_scope, "d")
    raw_not_null = " and ".join(f"d.{column} is not null" for column in REQUIRED_RAW_COLUMNS)
    start_text = _date_text(start_date)
    end_text = _date_text(end_date)
    if factor_adjust == "raw":
        raw_select = """
                count(*) as raw_rows,
                min(d.trade_date) as min_trade_date,
                max(d.trade_date) as max_trade_date
        """
        if include_distinct:
            raw_select += """,
                count(distinct d.symbol) as raw_symbols,
                count(distinct d.trade_date) as raw_trade_dates
            """
        raw_row = conn.execute(
            f"""
            select
                {raw_select}
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and {raw_not_null}
              {board_clause}
            """,
            (start_text, end_text, *board_params),
        ).fetchone()
        raw_rows = int(raw_row[0] or 0)
        raw_symbols = int(raw_row[3] or 0) if include_distinct else None
        raw_trade_dates = int(raw_row[4] or 0) if include_distinct else None
        return {
            "factor_adjust": factor_adjust,
            "board_scope": board_scope,
            "requested_start": start_text,
            "requested_end": end_text,
            "raw_rows": raw_rows,
            "raw_symbols": raw_symbols,
            "raw_trade_dates": raw_trade_dates,
            "min_trade_date": raw_row[1],
            "max_trade_date": raw_row[2],
            "adjusted_rows": raw_rows,
            "adjusted_symbols": raw_symbols,
            "missing_raw_rows": 0,
            "missing_symbols": 0,
            "missing_trade_dates": 0,
            "coverage_ratio": 1.0 if raw_rows else 0.0,
        }

    raw_select = """
            count(*) as raw_rows,
            min(d.trade_date) as min_trade_date,
            max(d.trade_date) as max_trade_date
    """
    if include_distinct:
        raw_select += """,
            count(distinct d.symbol) as raw_symbols,
            count(distinct d.trade_date) as raw_trade_dates
        """
    raw_row = conn.execute(
        f"""
        select
            {raw_select}
        from daily_bars d
        where d.adjust = 'raw'
          and d.trade_date >= ?
          and d.trade_date <= ?
          and {raw_not_null}
          {board_clause}
        """,
        (start_text, end_text, *board_params),
    ).fetchone()
    raw_rows = int(raw_row[0] or 0)
    raw_min_date = raw_row[1]
    raw_max_date = raw_row[2]
    raw_symbols = int(raw_row[3] or 0) if include_distinct else None
    raw_trade_dates = int(raw_row[4] or 0) if include_distinct else None
    adjusted_not_null = " and ".join(f"f.{column} is not null" for column in ("open", "high", "low", "close"))
    adjust_clause, adjust_params = board_scope_sql(board_scope, "f")
    adjusted_select = """
            count(*) as adjusted_total_rows,
            min(f.trade_date) as min_trade_date,
            max(f.trade_date) as max_trade_date
    """
    if include_distinct:
        adjusted_select += ", count(distinct f.symbol) as adjusted_total_symbols"
    adjusted_total_row = conn.execute(
        f"""
        select
            {adjusted_select}
        from daily_bars f
        where f.adjust = ?
          and f.trade_date >= ?
          and f.trade_date <= ?
          and {adjusted_not_null}
          {adjust_clause}
        """,
        (factor_adjust, start_text, end_text, *adjust_params),
    ).fetchone()
    adjusted_total_rows = int(adjusted_total_row[0] or 0)
    adjusted_min_date = adjusted_total_row[1]
    adjusted_max_date = adjusted_total_row[2]
    if (
        not include_distinct
        and raw_rows > 0
        and adjusted_total_rows >= raw_rows
        and adjusted_min_date is not None
        and adjusted_max_date is not None
        and str(adjusted_min_date) <= str(raw_min_date)
        and str(adjusted_max_date) >= str(raw_max_date)
    ):
        return {
            "factor_adjust": factor_adjust,
            "board_scope": board_scope,
            "requested_start": start_text,
            "requested_end": end_text,
            "raw_rows": raw_rows,
            "raw_symbols": raw_symbols,
            "raw_trade_dates": raw_trade_dates,
            "min_trade_date": raw_min_date,
            "max_trade_date": raw_max_date,
            "adjusted_rows": raw_rows,
            "adjusted_symbols": raw_symbols if include_distinct else None,
            "missing_raw_rows": 0,
            "missing_symbols": 0,
            "missing_trade_dates": 0,
            "coverage_ratio": 1.0,
            "coverage_check": "count_range_fast_path",
        }

    joined_select = """
            count(*) as raw_rows,
            min(d.trade_date) as min_trade_date,
            max(d.trade_date) as max_trade_date,
            count(f.symbol) as adjusted_rows
    """
    if include_distinct:
        joined_select += """,
            count(distinct d.symbol) as raw_symbols,
            count(distinct d.trade_date) as raw_trade_dates,
            count(distinct f.symbol) as adjusted_symbols,
            count(distinct case when f.symbol is null then d.symbol end) as missing_symbols,
            count(distinct case when f.symbol is null then d.trade_date end) as missing_trade_dates
        """
    joined_row = conn.execute(
        f"""
        select {joined_select}
        from daily_bars d
        left join daily_bars f
          on f.symbol = d.symbol
         and f.trade_date = d.trade_date
         and f.adjust = ?
         and {adjusted_not_null}
        where d.adjust = 'raw'
          and d.trade_date >= ?
          and d.trade_date <= ?
          and {raw_not_null}
          {board_clause}
        """,
        (factor_adjust, start_text, end_text, *board_params),
    ).fetchone()
    raw_rows = int(joined_row[0] or 0)
    adjusted_rows = int(joined_row[3] or 0)
    raw_symbols = int(joined_row[4] or 0) if include_distinct else None
    raw_trade_dates = int(joined_row[5] or 0) if include_distinct else None
    adjusted_symbols = int(joined_row[6] or 0) if include_distinct else None
    missing_symbols = int(joined_row[7] or 0) if include_distinct else None
    missing_trade_dates = int(joined_row[8] or 0) if include_distinct else None
    missing_rows = max(raw_rows - adjusted_rows, 0)
    return {
        "factor_adjust": factor_adjust,
        "board_scope": board_scope,
        "requested_start": start_text,
        "requested_end": end_text,
        "raw_rows": raw_rows,
        "raw_symbols": raw_symbols,
        "raw_trade_dates": raw_trade_dates,
        "min_trade_date": joined_row[1],
        "max_trade_date": joined_row[2],
        "adjusted_rows": adjusted_rows,
        "adjusted_symbols": adjusted_symbols,
        "missing_raw_rows": missing_rows,
        "missing_symbols": missing_symbols,
        "missing_trade_dates": missing_trade_dates,
        "coverage_ratio": float(adjusted_rows / raw_rows) if raw_rows else 0.0,
    }


def adjustment_missing_samples(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
    factor_adjust: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return the largest symbol-level gaps for an adjusted stream."""
    if factor_adjust == "raw":
        return []
    board_clause, board_params = board_scope_sql(board_scope, "d")
    raw_not_null = " and ".join(f"d.{column} is not null" for column in REQUIRED_RAW_COLUMNS)
    adjusted_total = conn.execute(
        f"""
        select count(*)
        from daily_bars f
        where f.adjust = ?
          and f.trade_date >= ?
          and f.trade_date <= ?
          and f.open is not null
          and f.high is not null
          and f.low is not null
          and f.close is not null
          {board_clause.replace('d.', 'f.')}
        """,
        (factor_adjust, _date_text(start_date), _date_text(end_date), *board_params),
    ).fetchone()[0]
    if int(adjusted_total or 0) <= 0:
        conn.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in conn.execute(
                f"""
                select d.symbol,
                       count(*) as missing_raw_rows,
                       min(d.trade_date) as min_missing_trade_date,
                       max(d.trade_date) as max_missing_trade_date
                from daily_bars d
                where d.adjust = 'raw'
                  and d.trade_date >= ?
                  and d.trade_date <= ?
                  and {raw_not_null}
                  {board_clause}
                group by d.symbol
                order by missing_raw_rows desc, d.symbol
                limit ?
                """,
                (_date_text(start_date), _date_text(end_date), *board_params, limit),
            )
        ]
    conn.row_factory = sqlite3.Row
    return [
        dict(row)
        for row in conn.execute(
            f"""
            select d.symbol,
                   count(*) as missing_raw_rows,
                   min(d.trade_date) as min_missing_trade_date,
                   max(d.trade_date) as max_missing_trade_date
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and {raw_not_null}
              {board_clause}
              and not exists (
                  select 1
                  from daily_bars f
                  where f.symbol = d.symbol
                    and f.trade_date = d.trade_date
                    and f.adjust = ?
                    and f.open is not null
                    and f.high is not null
                    and f.low is not null
                    and f.close is not null
              )
            group by d.symbol
            order by missing_raw_rows desc, d.symbol
            limit ?
            """,
            (_date_text(start_date), _date_text(end_date), *board_params, factor_adjust, limit),
        )
    ]


def require_factor_adjust_coverage(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
    factor_adjust: str,
) -> dict[str, Any]:
    """Require complete adjusted-factor coverage; raise with evidence otherwise."""
    coverage = adjustment_coverage(conn, start_date, end_date, board_scope, factor_adjust, include_distinct=True)
    if coverage["raw_rows"] <= 0:
        raise RuntimeError(
            "strict factor-adjust validation failed: no usable raw bars "
            f"for board_scope={board_scope} {coverage['requested_start']}->{coverage['requested_end']}"
        )
    if factor_adjust != "raw" and coverage["missing_raw_rows"] > 0:
        raise RuntimeError(
            "strict factor-adjust validation failed: "
            f"factor_adjust={factor_adjust} board_scope={board_scope} "
            f"{coverage['requested_start']}->{coverage['requested_end']} "
            f"raw_rows={coverage['raw_rows']} adjusted_rows={coverage['adjusted_rows']} "
            f"missing_raw_rows={coverage['missing_raw_rows']} "
            f"missing_symbols={coverage['missing_symbols']} "
            f"coverage_ratio={coverage['coverage_ratio']:.6f}"
        )
    return coverage


def status_coverage(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
) -> dict[str, Any]:
    """Return symbol-date coverage of historical ST/suspension status rows."""
    start_text = _date_text(start_date)
    end_text = _date_text(end_date)
    board_clause, board_params = board_scope_sql(board_scope, "d")
    raw_not_null = " and ".join(f"d.{column} is not null" for column in REQUIRED_RAW_COLUMNS)
    if not table_exists(conn, "symbol_status_daily"):
        raw_row = conn.execute(
            f"""
            select count(*) as expected_rows,
                   count(distinct d.symbol) as expected_symbols,
                   count(distinct d.trade_date) as expected_trade_dates,
                   min(d.trade_date) as min_trade_date,
                   max(d.trade_date) as max_trade_date
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and {raw_not_null}
              {board_clause}
            """,
            (start_text, end_text, *board_params),
        ).fetchone()
        expected_rows = int(raw_row[0] or 0)
        return {
            "board_scope": board_scope,
            "requested_start": start_text,
            "requested_end": end_text,
            "expected_raw_rows": expected_rows,
            "covered_raw_rows": 0,
            "missing_raw_rows": expected_rows,
            "expected_symbols": int(raw_row[1] or 0),
            "covered_symbols": 0,
            "missing_symbols": int(raw_row[1] or 0),
            "expected_trade_dates": int(raw_row[2] or 0),
            "min_trade_date": raw_row[3],
            "max_trade_date": raw_row[4],
            "coverage_ratio": 0.0,
        }

    row = conn.execute(
        f"""
        select count(*) as expected_rows,
               count(s.symbol) as covered_rows,
               count(distinct d.symbol) as expected_symbols,
               count(distinct s.symbol) as covered_symbols,
               count(distinct d.trade_date) as expected_trade_dates,
               min(d.trade_date) as min_trade_date,
               max(d.trade_date) as max_trade_date
        from daily_bars d
        left join symbol_status_daily s
          on s.symbol = d.symbol
         and s.trade_date = d.trade_date
        where d.adjust = 'raw'
          and d.trade_date >= ?
          and d.trade_date <= ?
          and {raw_not_null}
          {board_clause}
        """,
        (start_text, end_text, *board_params),
    ).fetchone()
    expected_rows = int(row[0] or 0)
    covered_rows = int(row[1] or 0)
    expected_symbols = int(row[2] or 0)
    covered_symbols = int(row[3] or 0)
    return {
        "board_scope": board_scope,
        "requested_start": start_text,
        "requested_end": end_text,
        "expected_raw_rows": expected_rows,
        "covered_raw_rows": covered_rows,
        "missing_raw_rows": max(expected_rows - covered_rows, 0),
        "expected_symbols": expected_symbols,
        "covered_symbols": covered_symbols,
        "missing_symbols": max(expected_symbols - covered_symbols, 0),
        "expected_trade_dates": int(row[4] or 0),
        "min_trade_date": row[5],
        "max_trade_date": row[6],
        "coverage_ratio": float(covered_rows / expected_rows) if expected_rows else 0.0,
    }


def lifecycle_coverage(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
) -> dict[str, Any]:
    """Return symbol-level lifecycle metadata coverage for the requested raw universe."""
    start_text = _date_text(start_date)
    end_text = _date_text(end_date)
    board_clause, board_params = board_scope_sql(board_scope, "d")
    raw_not_null = " and ".join(f"d.{column} is not null" for column in REQUIRED_RAW_COLUMNS)
    if not table_exists(conn, "symbol_lifecycle"):
        row = conn.execute(
            f"""
            select count(distinct d.symbol)
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and {raw_not_null}
              {board_clause}
            """,
            (start_text, end_text, *board_params),
        ).fetchone()
        expected = int(row[0] or 0)
        return {
            "board_scope": board_scope,
            "requested_start": start_text,
            "requested_end": end_text,
            "expected_symbols": expected,
            "covered_symbols": 0,
            "missing_symbols": expected,
            "coverage_ratio": 0.0,
        }

    row = conn.execute(
        f"""
        with expected as (
            select distinct d.symbol
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and {raw_not_null}
              {board_clause}
        )
        select count(*) as expected_symbols,
               sum(
                   case
                       when exists (
                           select 1
                           from symbol_lifecycle l
                           where l.symbol = expected.symbol
                       )
                       then 1 else 0
                   end
               ) as covered_symbols
        from expected
        """,
        (start_text, end_text, *board_params),
    ).fetchone()
    expected = int(row[0] or 0)
    covered = int(row[1] or 0)
    return {
        "board_scope": board_scope,
        "requested_start": start_text,
        "requested_end": end_text,
        "expected_symbols": expected,
        "covered_symbols": covered,
        "missing_symbols": max(expected - covered, 0),
        "coverage_ratio": float(covered / expected) if expected else 0.0,
    }


def symbol_industry_coverage(
    conn: sqlite3.Connection,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
) -> dict[str, Any]:
    """Return stock-industry mapping coverage for the requested raw universe."""
    board_clause, board_params = board_scope_sql(board_scope, "d")
    report = industry_coverage(conn, start_date, end_date, board_clause, tuple(board_params))
    report["board_scope"] = board_scope
    return report


def latest_trade_dates(
    conn: sqlite3.Connection,
    end_date: pd.Timestamp | str,
    board_scope: str,
    adjusts: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return latest available trade_date per adjust in the requested board scope."""
    end_text = _date_text(end_date)
    output: dict[str, Any] = {}
    for adjust in adjusts:
        board_clause, board_params = board_scope_sql(board_scope, "d")
        row = conn.execute(
            f"""
            select max(d.trade_date)
            from daily_bars d
            where d.adjust = ?
              and d.trade_date <= ?
              {board_clause}
            """,
            (adjust, end_text, *board_params),
        ).fetchone()
        output[str(adjust)] = row[0]
    return output


def schema_presence(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return presence of P0 reference/state tables."""
    reference = {name: table_exists(conn, name) for name in REFERENCE_TABLES}
    state = {name: table_exists(conn, name) for name in STATE_TABLES}
    return {
        "reference_tables": reference,
        "state_tables": state,
        "missing_reference_tables": [name for name, exists in reference.items() if not exists],
        "missing_state_tables": [name for name, exists in state.items() if not exists],
    }


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    if not table_exists(conn, table_name):
        return False
    return any(str(row[1]) == column_name for row in conn.execute(f"PRAGMA table_info({table_name})"))


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    report["is_formal_ready"] = len(report.get("red_flags", [])) == 0
    return report


def build_quality_report(
    db: Path,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    board_scope: str,
    required_adjusts: list[str] | tuple[str, ...] = CANONICAL_ADJUSTS,
) -> dict[str, Any]:
    """Build a JSON-safe database quality report for the requested slice."""
    report: dict[str, Any] = {
        "database": {
            "path": str(db),
            "exists": db.exists(),
            "size_bytes": db.stat().st_size if db.exists() else 0,
        },
        "requested": {
            "start_date": _date_text(start_date),
            "end_date": _date_text(end_date),
            "board_scope": board_scope,
            "required_adjusts": list(required_adjusts),
        },
        "red_flags": [],
    }
    if not db.exists():
        report["red_flags"].append("database_missing")
        return _finalize_report(report)

    with sqlite3.connect(db) as conn:
        if not table_exists(conn, "daily_bars"):
            report["red_flags"].append("daily_bars_missing")
            report["schema"] = schema_presence(conn)
            return _finalize_report(report)

        report["schema"] = schema_presence(conn)
        for missing in report["schema"]["missing_reference_tables"]:
            report["red_flags"].append(f"reference_table_missing:{missing}")
        for missing in report["schema"]["missing_state_tables"]:
            report["red_flags"].append(f"state_table_missing:{missing}")

        coverage = [
            adjustment_coverage(conn, start_date, end_date, board_scope, adjust, include_distinct=False)
            for adjust in required_adjusts
        ]
        report["adjustment_coverage"] = coverage
        report["adjustment_missing_samples"] = {
            item["factor_adjust"]: adjustment_missing_samples(
                conn,
                start_date,
                end_date,
                board_scope,
                item["factor_adjust"],
            )
            for item in coverage
            if item["factor_adjust"] != "raw" and item["missing_raw_rows"] > 0
        }
        for item in coverage:
            if item["raw_rows"] <= 0:
                report["red_flags"].append(f"raw_coverage_empty:{item['board_scope']}")
            if item["factor_adjust"] != "raw" and item["missing_raw_rows"] > 0:
                report["red_flags"].append(
                    f"adjust_missing:{item['factor_adjust']}:missing_raw_rows={item['missing_raw_rows']}"
                )
        if table_exists(conn, "daily_bars"):
            conn.row_factory = sqlite3.Row
            report["latest_trade_dates"] = latest_trade_dates(conn, end_date, board_scope, required_adjusts)
            for adjust, latest in report["latest_trade_dates"].items():
                if latest is None:
                    report["red_flags"].append(f"latest_adjust_missing:{adjust}")
                elif str(latest) < _date_text(end_date):
                    report["red_flags"].append(
                        f"latest_adjust_before_requested_end:{adjust}:{latest}<{_date_text(end_date)}"
                    )
            report["daily_bars_by_adjust"] = [
                dict(row)
                for row in conn.execute(
                    """
                    select adjust, count(*) as rows,
                           min(trade_date) as min_trade_date, max(trade_date) as max_trade_date
                    from daily_bars
                    group by adjust
                    order by adjust
                    """
                )
            ]
        if table_exists(conn, "fetch_status"):
            conn.row_factory = sqlite3.Row
            source_expr = "coalesce(source_used, '')" if column_exists(conn, "fetch_status", "source_used") else "''"
            report["fetch_status_summary"] = [
                dict(row)
                for row in conn.execute(
                    f"""
                    select adjust, last_status, {source_expr} as source_used,
                           count(*) as rows,
                           sum(rows_fetched) as rows_fetched,
                           max(fetched_at) as max_fetched_at
                    from fetch_status
                    group by adjust, last_status, {source_expr}
                    order by adjust, last_status, source_used
                    """
                )
            ]
            report["fetch_failure_samples"] = [
                dict(row)
                for row in conn.execute(
                    """
                    select symbol, adjust, requested_start, requested_end, message, fetched_at
                    from fetch_status
                    where last_status != 'ok'
                    order by fetched_at desc
                    limit 20
                    """
                )
            ]
        if table_exists(conn, "adj_factors"):
            report["adj_factors"] = conn.execute(
                """
                select count(*) as rows, count(distinct symbol) as symbols,
                       min(trade_date) as min_trade_date, max(trade_date) as max_trade_date
                from adj_factors
                """
            ).fetchone()
            report["adj_factors"] = dict(report["adj_factors"])
            if report["adj_factors"]["rows"] <= 0 and any(adjust != "raw" for adjust in required_adjusts):
                report["red_flags"].append("reference_data_empty:adj_factors")
        if table_exists(conn, "symbol_lifecycle"):
            report["symbol_lifecycle"] = dict(
                conn.execute(
                    """
                    select count(*) as rows, count(distinct symbol) as symbols,
                           sum(case when list_date is not null then 1 else 0 end) as with_list_date,
                           sum(case when delist_date is not null then 1 else 0 end) as with_delist_date
                    from symbol_lifecycle
                    """
                ).fetchone()
            )
            if report["symbol_lifecycle"]["rows"] <= 0:
                report["red_flags"].append("reference_data_empty:symbol_lifecycle")
            report["symbol_lifecycle"]["coverage"] = lifecycle_coverage(conn, start_date, end_date, board_scope)
            if report["symbol_lifecycle"]["coverage"]["missing_symbols"] > 0:
                report["red_flags"].append(
                    "lifecycle_symbols_incomplete:"
                    f"symbols={report['symbol_lifecycle']['coverage']['covered_symbols']}/"
                    f"{report['symbol_lifecycle']['coverage']['expected_symbols']}"
                )
        if table_exists(conn, "symbol_status_daily"):
            report["symbol_status_daily"] = dict(
                conn.execute(
                    """
                    select count(*) as rows, count(distinct symbol) as symbols,
                           min(trade_date) as min_trade_date,
                           max(trade_date) as max_trade_date,
                           sum(case when is_st = 1 then 1 else 0 end) as st_rows,
                           sum(case when is_suspended = 1 then 1 else 0 end) as suspended_rows
                    from symbol_status_daily
                    """
                ).fetchone()
            )
            if report["symbol_status_daily"]["rows"] <= 0:
                report["red_flags"].append("reference_data_empty:symbol_status_daily")
            else:
                row_coverage = status_coverage(conn, start_date, end_date, board_scope)
                report["symbol_status_daily"]["row_coverage"] = row_coverage
                if row_coverage["expected_raw_rows"] > 0 and row_coverage["coverage_ratio"] < FORMAL_STATUS_COVERAGE_MIN:
                    report["red_flags"].append(
                        "status_rows_incomplete:"
                        f"rows={row_coverage['covered_raw_rows']}/{row_coverage['expected_raw_rows']} "
                        f"ratio={row_coverage['coverage_ratio']:.6f}"
                    )
                expected_symbols = int(row_coverage["expected_symbols"])
                covered_symbols = int(row_coverage["covered_symbols"])
                missing_symbols = max(expected_symbols - covered_symbols, 0)
                report["symbol_status_daily"]["expected_symbols_with_raw"] = expected_symbols
                report["symbol_status_daily"]["covered_symbols_with_status"] = covered_symbols
                report["symbol_status_daily"]["missing_symbols_with_status"] = missing_symbols
                if missing_symbols > 0:
                    report["red_flags"].append(
                        "status_symbols_incomplete:"
                        f"symbols={covered_symbols}/{expected_symbols}"
                    )
        if table_exists(conn, "symbol_industries"):
            report["symbol_industries"] = dict(
                conn.execute(
                    """
                    select count(*) as rows, count(distinct symbol) as symbols,
                           count(distinct industry_name) as industries,
                           min(fetched_at) as min_fetched_at,
                           max(fetched_at) as max_fetched_at
                    from symbol_industries
                    """
                ).fetchone()
            )
            if report["symbol_industries"]["rows"] <= 0:
                report["red_flags"].append("reference_data_empty:symbol_industries")
            report["symbol_industries"]["coverage"] = symbol_industry_coverage(
                conn,
                start_date,
                end_date,
                board_scope,
            )
            coverage_ratio = float(report["symbol_industries"]["coverage"]["coverage_ratio"])
            if (
                report["symbol_industries"]["coverage"]["expected_symbols"] > 0
                and coverage_ratio < FORMAL_INDUSTRY_COVERAGE_MIN
            ):
                report["red_flags"].append(
                    "industry_symbols_incomplete:"
                    f"symbols={report['symbol_industries']['coverage']['covered_symbols']}/"
                    f"{report['symbol_industries']['coverage']['expected_symbols']} "
                    f"ratio={coverage_ratio:.6f}"
                )
    return _finalize_report(report)

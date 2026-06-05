#!/usr/bin/env python3
"""Fetch point-in-time A-share financial indicators into SQLite."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fetch_akshare_daily import connect  # noqa: E402
from fetch_symbol_valuation import load_symbols_file, normalize_valuation_symbol  # noqa: E402
from professional_quant.data.providers.base import provider_failure, provider_success  # noqa: E402
from quant_schema import ensure_quant_schema  # noqa: E402

SOURCE = "akshare.stock_financial_analysis_indicator_em"

FIELD_MAP = {
    "ROEJQ": "roe",
    "ROIC": "roic",
    "XSMLL": "gross_margin",
    "XSJLL": "net_margin",
    "ZZCJLL": "asset_return",
    "ZCFZL": "debt_asset_ratio",
    "TOTALOPERATEREVETZ": "revenue_growth_yoy",
    "PARENTNETPROFITTZ": "profit_growth_yoy",
    "KCFJCXSYJLRTZ": "deduct_profit_growth_yoy",
    "JYXJLYYSR": "operating_cashflow_to_revenue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Eastmoney A-share financial indicator history.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--symbols", help="Comma-separated symbols, e.g. 600519,000001")
    parser.add_argument("--symbols-file", type=Path, help="CSV/JSON/TXT file containing a symbol column or one symbol per line")
    parser.add_argument("--start-date", help="Earliest notice date to store")
    parser.add_argument("--end-date", help="Latest notice date to store")
    parser.add_argument("--indicator", default="按报告期", choices=["按报告期", "按单季度"])
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between symbols")
    parser.add_argument("--limit", type=int, help="Optional symbol limit for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def financial_secucode(symbol: str) -> str:
    normalized = normalize_valuation_symbol(symbol)
    if normalized.startswith("6"):
        return f"{normalized}.SH"
    return f"{normalized}.SZ"


def load_symbols(conn: sqlite3.Connection, symbols: str | None, symbols_file: Path | None, limit: int | None) -> list[str]:
    if symbols:
        selected = [normalize_valuation_symbol(item) for item in symbols.split(",") if item.strip()]
    elif symbols_file:
        selected = load_symbols_file(symbols_file)
    else:
        rows = conn.execute("select symbol from symbols order by symbol").fetchall()
        selected = [normalize_valuation_symbol(row[0]) for row in rows]
    unique = list(dict.fromkeys(selected))
    if limit is not None:
        unique = unique[:limit]
    if not unique:
        raise RuntimeError("no symbols selected")
    return unique


def normalize_financial_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "report_date",
        "notice_date",
        "update_date",
        "report_type",
        "report_year",
        *FIELD_MAP.values(),
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"REPORT_DATE", "NOTICE_DATE"}
    if not required <= set(frame.columns):
        raise RuntimeError(f"unexpected financial columns for {symbol}: {list(frame.columns)}")
    out = pd.DataFrame(index=frame.index)
    out["symbol"] = normalize_valuation_symbol(symbol)
    out["report_date"] = pd.to_datetime(frame["REPORT_DATE"], errors="coerce")
    out["notice_date"] = pd.to_datetime(frame["NOTICE_DATE"], errors="coerce")
    out["update_date"] = pd.to_datetime(frame.get("UPDATE_DATE"), errors="coerce") if "UPDATE_DATE" in frame else pd.NaT
    out["report_type"] = frame.get("REPORT_TYPE")
    out["report_year"] = pd.to_numeric(frame.get("REPORT_YEAR"), errors="coerce") if "REPORT_YEAR" in frame else pd.NA
    for source, target in FIELD_MAP.items():
        out[target] = pd.to_numeric(frame[source], errors="coerce") if source in frame else pd.NA
    out = out.dropna(subset=["report_date", "notice_date"])
    out["report_date"] = out["report_date"].dt.strftime("%Y-%m-%d")
    out["notice_date"] = out["notice_date"].dt.strftime("%Y-%m-%d")
    out["update_date"] = out["update_date"].dt.strftime("%Y-%m-%d")
    return out[columns].sort_values(["symbol", "notice_date", "report_date"])


def _date_filter(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    notices = pd.to_datetime(out["notice_date"], errors="coerce")
    if start_date:
        out = out[notices >= pd.Timestamp(start_date)]
        notices = notices.loc[out.index]
    if end_date:
        out = out[notices <= pd.Timestamp(end_date)]
    return out.sort_values(["symbol", "notice_date", "report_date"])


def fetch_symbol_financials(
    symbol: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    indicator: str = "按报告期",
) -> pd.DataFrame:
    secucode = financial_secucode(symbol)
    raw = ak.stock_financial_analysis_indicator_em(symbol=secucode, indicator=indicator)
    frame = _date_filter(normalize_financial_frame(symbol, raw), start_date, end_date)
    frame["source"] = SOURCE
    frame["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return frame


def write_symbol_financials(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    ensure_quant_schema(conn)
    if frame.empty:
        conn.commit()
        return 0
    rows = frame[
        [
            "symbol",
            "report_date",
            "notice_date",
            "update_date",
            "report_type",
            "report_year",
            *FIELD_MAP.values(),
            "source",
            "fetched_at",
        ]
    ].where(pd.notna(frame), None)
    conn.executemany(
        """
        INSERT OR REPLACE INTO symbol_financial_indicator (
            symbol, report_date, notice_date, update_date, report_type, report_year,
            roe, roic, gross_margin, net_margin, asset_return, debt_asset_ratio,
            revenue_growth_yoy, profit_growth_yoy, deduct_profit_growth_yoy,
            operating_cashflow_to_revenue, source, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows.itertuples(index=False, name=None),
    )
    conn.commit()
    return int(len(rows))


def main() -> int:
    args = parse_args()
    output = {
        "db": str(args.db),
        "source": SOURCE,
        "indicator": args.indicator,
        "dry_run": bool(args.dry_run),
        "symbols": 0,
        "rows": 0,
        "failures": [],
    }
    with connect(args.db) as conn:
        ensure_quant_schema(conn)
        selected = load_symbols(conn, args.symbols, args.symbols_file, args.limit)
        output["symbols"] = len(selected)
        for index, symbol in enumerate(selected, start=1):
            try:
                frame = fetch_symbol_financials(
                    symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    indicator=args.indicator,
                )
                rows = int(len(frame)) if args.dry_run else write_symbol_financials(conn, frame)
                output["rows"] += rows
                output.setdefault("samples", []).append(
                    {
                        "symbol": symbol,
                        "rows": rows,
                        "start_notice": frame["notice_date"].min() if not frame.empty else None,
                        "end_notice": frame["notice_date"].max() if not frame.empty else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - external provider failures are recorded per symbol.
                output["failures"].append({"symbol": symbol, "error": str(exc)})
            if index < len(selected) and args.sleep > 0:
                time.sleep(args.sleep)
    result_factory = provider_success if not output["failures"] else provider_failure
    provider_kwargs = {
        "provider": "eastmoney",
        "dataset": "symbol_financial_indicator",
        "rows": int(output["rows"]),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "metadata": {
            "source": SOURCE,
            "indicator": args.indicator,
            "symbols": output["symbols"],
            "failures": output["failures"],
            "dry_run": bool(args.dry_run),
        },
    }
    if output["failures"]:
        provider_kwargs["error"] = f"{len(output['failures'])} symbol financial fetch failures"
    output["provider_result"] = result_factory(**provider_kwargs).to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if output["failures"] and output["rows"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

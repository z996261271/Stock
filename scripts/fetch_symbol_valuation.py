#!/usr/bin/env python3
"""Fetch point-in-time A-share symbol valuation history into SQLite."""

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
from professional_quant.data.providers.base import provider_failure, provider_success  # noqa: E402
from quant_schema import ensure_quant_schema  # noqa: E402

SOURCE = "akshare.stock_zh_valuation_baidu"
INDICATORS = {
    "pe_ttm": "市盈率(TTM)",
    "pe_static": "市盈率(静)",
    "pb": "市净率",
    "pcf": "市现率",
    "total_market_cap": "总市值",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Baidu A-share symbol valuation history.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--symbols", help="Comma-separated symbols, e.g. 600519,000001")
    parser.add_argument("--symbols-file", type=Path, help="CSV/JSON/TXT file containing a symbol column or one symbol per line")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--period", default="全部", choices=["近一年", "近三年", "近五年", "近十年", "全部"])
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between symbols")
    parser.add_argument("--limit", type=int, help="Optional symbol limit for smoke tests")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_valuation_symbol(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        left, right = text.split(".", 1)
        text = left if right.upper() in {"SH", "SZ", "BJ"} else right
    upper = text.upper()
    for prefix in ("SH", "SZ", "BJ"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip().zfill(6)


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


def load_symbols_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            values = [item.get("symbol", item) if isinstance(item, dict) else item for item in data]
        else:
            values = data.get("symbols", [])
        return [normalize_valuation_symbol(value) for value in values if str(value).strip()]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = "symbol" if "symbol" in frame.columns else frame.columns[0]
        return [normalize_valuation_symbol(value) for value in frame[column].dropna().tolist()]
    return [normalize_valuation_symbol(line) for line in path.read_text().splitlines() if line.strip()]


def _date_filter(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"])
    if start_date:
        out = out[out["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["trade_date"] <= pd.Timestamp(end_date)]
    out["trade_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    return out.sort_values("trade_date")


def normalize_indicator_frame(symbol: str, indicator_key: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", indicator_key])
    required = {"date", "value"}
    if not required <= set(frame.columns):
        raise RuntimeError(f"unexpected {indicator_key} columns for {symbol}: {list(frame.columns)}")
    out = frame[["date", "value"]].rename(columns={"date": "trade_date", "value": indicator_key}).copy()
    out["symbol"] = normalize_valuation_symbol(symbol)
    out[indicator_key] = pd.to_numeric(out[indicator_key], errors="coerce")
    return out[["symbol", "trade_date", indicator_key]]


def fetch_symbol_valuation(
    symbol: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "全部",
) -> pd.DataFrame:
    normalized = normalize_valuation_symbol(symbol)
    merged: pd.DataFrame | None = None
    for key, indicator in INDICATORS.items():
        raw = ak.stock_zh_valuation_baidu(symbol=normalized, indicator=indicator, period=period)
        current = normalize_indicator_frame(normalized, key, raw)
        merged = current if merged is None else pd.merge(merged, current, on=["symbol", "trade_date"], how="outer")
    if merged is None:
        return pd.DataFrame()
    frame = _date_filter(merged, start_date, end_date)
    frame["source"] = SOURCE
    frame["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return frame


def write_symbol_valuation(conn: sqlite3.Connection, frame: pd.DataFrame) -> int:
    ensure_quant_schema(conn)
    if frame.empty:
        conn.commit()
        return 0
    rows = frame[
        [
            "symbol",
            "trade_date",
            "pe_ttm",
            "pe_static",
            "pb",
            "pcf",
            "total_market_cap",
            "source",
            "fetched_at",
        ]
    ].where(pd.notna(frame), None)
    conn.executemany(
        """
        INSERT OR REPLACE INTO symbol_valuation_daily (
            symbol, trade_date, pe_ttm, pe_static, pb, pcf, total_market_cap, source, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        "period": args.period,
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
                frame = fetch_symbol_valuation(
                    symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    period=args.period,
                )
                rows = int(len(frame)) if args.dry_run else write_symbol_valuation(conn, frame)
                output["rows"] += rows
                output.setdefault("samples", []).append(
                    {
                        "symbol": symbol,
                        "rows": rows,
                        "start_date": frame["trade_date"].min() if not frame.empty else None,
                        "end_date": frame["trade_date"].max() if not frame.empty else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - external data provider failures are recorded per symbol.
                output["failures"].append({"symbol": symbol, "error": str(exc)})
            if index < len(selected) and args.sleep > 0:
                time.sleep(args.sleep)
    result_factory = provider_success if not output["failures"] else provider_failure
    provider_kwargs = {
        "provider": "baidu",
        "dataset": "symbol_valuation_daily",
        "rows": int(output["rows"]),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "metadata": {
            "source": SOURCE,
            "period": args.period,
            "symbols": output["symbols"],
            "failures": output["failures"],
            "dry_run": bool(args.dry_run),
        },
    }
    if output["failures"]:
        provider_kwargs["error"] = f"{len(output['failures'])} symbol valuation fetch failures"
    output["provider_result"] = result_factory(**provider_kwargs).to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if output["failures"] and output["rows"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

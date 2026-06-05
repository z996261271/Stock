from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fetch_symbol_valuation as valuation  # noqa: E402


def test_normalize_indicator_frame_maps_baidu_columns():
    frame = pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "value": ["12.3", None]})

    out = valuation.normalize_indicator_frame("600519", "pe_ttm", frame)

    assert list(out.columns) == ["symbol", "trade_date", "pe_ttm"]
    assert out["symbol"].tolist() == ["600519", "600519"]
    assert out["pe_ttm"].iloc[0] == 12.3
    assert pd.isna(out["pe_ttm"].iloc[1])


def test_write_symbol_valuation_upserts_rows():
    frame = pd.DataFrame(
        {
            "symbol": ["600519", "600519"],
            "trade_date": ["2020-01-02", "2020-01-03"],
            "pe_ttm": [12.0, 13.0],
            "pe_static": [11.0, 12.0],
            "pb": [3.0, 3.1],
            "pcf": [20.0, 21.0],
            "total_market_cap": [100.0, 101.0],
            "source": [valuation.SOURCE, valuation.SOURCE],
            "fetched_at": ["2026-06-06T00:00:00", "2026-06-06T00:00:00"],
        }
    )
    with sqlite3.connect(":memory:") as conn:
        rows = valuation.write_symbol_valuation(conn, frame)
        rows_again = valuation.write_symbol_valuation(conn, frame)
        count = conn.execute("select count(*) from symbol_valuation_daily").fetchone()[0]
        pb = conn.execute(
            "select pb from symbol_valuation_daily where symbol = '600519' and trade_date = '2020-01-03'"
        ).fetchone()[0]

    assert rows == 2
    assert rows_again == 2
    assert count == 2
    assert pb == 3.1


def test_load_symbols_file_supports_text(tmp_path: Path):
    path = tmp_path / "symbols.txt"
    path.write_text("600519\n000001.SZ\n")

    assert valuation.load_symbols_file(path) == ["600519", "000001"]


if __name__ == "__main__":
    test_normalize_indicator_frame_maps_baidu_columns()
    test_write_symbol_valuation_upserts_rows()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_load_symbols_file_supports_text(Path(tmp))

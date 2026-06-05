from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fetch_symbol_financials as financials  # noqa: E402


def test_normalize_financial_frame_maps_eastmoney_columns_and_notice_date():
    frame = pd.DataFrame(
        {
            "REPORT_DATE": ["2020-12-31 00:00:00", "2021-03-31 00:00:00"],
            "NOTICE_DATE": ["2021-04-20 00:00:00", None],
            "UPDATE_DATE": ["2021-04-20 00:00:00", "2021-04-30 00:00:00"],
            "REPORT_TYPE": ["年报", "一季报"],
            "REPORT_YEAR": ["2020", "2021"],
            "ROEJQ": ["12.5", "13.0"],
            "ROIC": ["8.1", "8.2"],
            "XSMLL": ["30.0", "31.0"],
            "XSJLL": ["10.0", "11.0"],
            "ZZCJLL": ["5.0", "6.0"],
            "ZCFZL": ["40.0", "41.0"],
            "TOTALOPERATEREVETZ": ["15.0", "16.0"],
            "PARENTNETPROFITTZ": ["20.0", "21.0"],
            "KCFJCXSYJLRTZ": ["19.0", "20.0"],
            "JYXJLYYSR": ["0.8", "0.9"],
        }
    )

    out = financials.normalize_financial_frame("600519.SH", frame)

    assert len(out) == 1
    assert out.loc[0, "symbol"] == "600519"
    assert out.loc[0, "report_date"] == "2020-12-31"
    assert out.loc[0, "notice_date"] == "2021-04-20"
    assert out.loc[0, "roe"] == 12.5
    assert out.loc[0, "operating_cashflow_to_revenue"] == 0.8


def test_write_symbol_financials_upserts_rows():
    frame = pd.DataFrame(
        {
            "symbol": ["600519", "600519"],
            "report_date": ["2020-12-31", "2021-03-31"],
            "notice_date": ["2021-04-20", "2021-04-30"],
            "update_date": ["2021-04-20", "2021-04-30"],
            "report_type": ["年报", "一季报"],
            "report_year": [2020, 2021],
            "roe": [12.5, 13.0],
            "roic": [8.1, 8.2],
            "gross_margin": [30.0, 31.0],
            "net_margin": [10.0, 11.0],
            "asset_return": [5.0, 6.0],
            "debt_asset_ratio": [40.0, 41.0],
            "revenue_growth_yoy": [15.0, 16.0],
            "profit_growth_yoy": [20.0, 21.0],
            "deduct_profit_growth_yoy": [19.0, 20.0],
            "operating_cashflow_to_revenue": [0.8, 0.9],
            "source": [financials.SOURCE, financials.SOURCE],
            "fetched_at": ["2026-06-06T00:00:00", "2026-06-06T00:00:00"],
        }
    )
    with sqlite3.connect(":memory:") as conn:
        rows = financials.write_symbol_financials(conn, frame)
        rows_again = financials.write_symbol_financials(conn, frame)
        count = conn.execute("select count(*) from symbol_financial_indicator").fetchone()[0]
        roe = conn.execute(
            "select roe from symbol_financial_indicator where symbol = '600519' and notice_date = '2021-04-30'"
        ).fetchone()[0]

    assert rows == 2
    assert rows_again == 2
    assert count == 2
    assert roe == 13.0


if __name__ == "__main__":
    test_normalize_financial_frame_maps_eastmoney_columns_and_notice_date()
    test_write_symbol_financials_upserts_rows()

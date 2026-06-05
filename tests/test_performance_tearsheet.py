import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.reporting.performance import build_performance_report, render_markdown  # noqa: E402


def test_performance_report_summarizes_risk_drawdowns_trades_and_benchmark():
    equity = pd.DataFrame(
        [
            {"entry_date": "2021-01-04", "equity": 110.0, "period_return": 0.10, "trade": True},
            {"entry_date": "2021-01-05", "equity": 88.0, "period_return": -0.20, "trade": False},
            {"entry_date": "2021-01-06", "equity": 96.8, "period_return": 0.10, "trade": True},
            {"entry_date": "2021-02-01", "equity": 87.12, "period_return": -0.10, "trade": False},
        ]
    )
    trades = pd.DataFrame(
        [
            {"symbol": "000001", "side": "buy", "status": "filled", "desired_notional": 100, "filled_notional": 100},
            {"symbol": "000002", "side": "buy", "status": "partial", "desired_notional": 100, "filled_notional": 40},
        ]
    )
    metrics = {
        "initial_cash": 100.0,
        "benchmarks": {
            "unit": {
                "name": "unit",
                "daily_returns": [
                    {"date": "2021-01-04", "return": 0.05},
                    {"date": "2021-01-05", "return": -0.10},
                    {"date": "2021-01-06", "return": 0.02},
                    {"date": "2021-02-01", "return": -0.02},
                ],
            }
        },
    }

    report = build_performance_report(metrics=metrics, equity=equity, trades=trades, benchmark_name="unit")
    markdown = render_markdown(report)

    assert round(report["summary"]["total_return"], 6) == -0.1288
    assert report["return_breakdowns"]["consecutive_losing_months"]["max_streak"] == 2
    assert round(report["drawdowns"][0]["max_drawdown"], 6) == -0.208
    assert report["trade_statistics"]["fill_ratio"] == 0.7
    assert report["benchmark"]["relative"]["matched_periods"] == 4
    assert "Performance Tear Sheet" in markdown


if __name__ == "__main__":
    test_performance_report_summarizes_risk_drawdowns_trades_and_benchmark()

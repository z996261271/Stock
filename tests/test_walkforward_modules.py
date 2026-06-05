import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.backtest.walkforward import summarize_walkforward_result  # noqa: E402


def test_summarize_walkforward_result_scales_equity_and_counts_execution_fields():
    equity, picks, trades, metrics = summarize_walkforward_result(
        returns=np.asarray([0.10, -0.05], dtype=np.float32),
        active=np.asarray([True, True]),
        trades=np.asarray([True, False]),
        equity_rows=[
            {
                "signal_date": "2021-01-04",
                "entry_date": "2021-01-05",
                "equity": 1.10,
                "drawdown": 0.0,
                "trade_count": 1,
                "blocked_buy_count": 0,
                "blocked_sell_count": 0,
                "partial_buy_count": 0,
                "partial_sell_count": 0,
                "industry_blocked_count": 0,
                "turnover_blocked_count": 0,
                "turnover_value": 100.0,
                "unfilled_buy_value": 0.0,
                "unfilled_sell_value": 0.0,
                "turnover_blocked_value": 0.0,
                "turnover_pct": 0.5,
                "invested_weight": 0.8,
                "cash_weight": 0.2,
                "max_position_weight": 0.3,
                "max_industry_weight": 0.4,
                "portfolio_risk_off": False,
            },
            {
                "signal_date": "2021-01-05",
                "entry_date": "2021-01-06",
                "equity": 1.045,
                "drawdown": -0.05,
                "trade_count": 0,
                "blocked_buy_count": 1,
                "blocked_sell_count": 0,
                "partial_buy_count": 0,
                "partial_sell_count": 0,
                "industry_blocked_count": 0,
                "turnover_blocked_count": 0,
                "turnover_value": 0.0,
                "unfilled_buy_value": 10.0,
                "unfilled_sell_value": 0.0,
                "turnover_blocked_value": 0.0,
                "turnover_pct": 0.0,
                "invested_weight": 0.8,
                "cash_weight": 0.2,
                "max_position_weight": 0.3,
                "max_industry_weight": 0.4,
                "portfolio_risk_off": False,
            },
        ],
        pick_rows=[{"symbol": "000001"}],
        trade_event_rows=[{"status": "partial", "reason": "capacity_partial"}],
        initial_cash=1_000_000.0,
    )

    assert round(equity["equity"].iloc[-1], 2) == 1_045_000.0
    assert len(picks) == 1
    assert len(trades) == 1
    assert metrics["blocked_buy_count"] == 1
    assert metrics["trade_log_status_counts"] == {"partial": 1}
    assert metrics["date_validation"]["signal_before_entry"] is True


if __name__ == "__main__":
    test_summarize_walkforward_result_scales_equity_and_counts_execution_fields()

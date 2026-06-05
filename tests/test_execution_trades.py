import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.execution.trades import (  # noqa: E402
    side_cost_fraction_from_amounts,
    trade_block_reason_from_fields,
    trade_event_row,
    trade_status,
)


def test_trade_status_classifies_blocked_partial_and_filled():
    assert trade_status(100.0, 0.0) == "blocked"
    assert trade_status(100.0, 60.0) == "partial"
    assert trade_status(100.0, 100.0) == "filled"


def test_trade_block_reason_preserves_execution_priority():
    base = {
        "side": "buy",
        "desired_notional": 1_000.0,
        "filled_notional": 1_000.0,
        "price_valid": True,
        "symbol_present": True,
        "suspended": False,
        "executable": True,
        "amount_valid": True,
        "price": 10.0,
        "amount": 100_000.0,
        "lot_size": 100,
        "capacity_pct_of_amount": 0.02,
    }

    assert trade_block_reason_from_fields(**{**base, "price_valid": False}) == "missing_or_invalid_open"
    assert trade_block_reason_from_fields(**{**base, "symbol_present": False}) == "symbol_not_in_entry_universe"
    assert trade_block_reason_from_fields(**{**base, "suspended": True}) == "suspended_block"
    assert trade_block_reason_from_fields(**{**base, "executable": False}) == "limit_up_block"
    assert trade_block_reason_from_fields(**{**base, "side": "sell", "executable": False}) == "limit_down_block"
    assert trade_block_reason_from_fields(**{**base, "amount_valid": False}) == "missing_or_invalid_amount"
    assert trade_block_reason_from_fields(**{**base, "desired_notional": 500.0}) == "lot_size_block"
    assert trade_block_reason_from_fields(**{**base, "filled_notional": 700.0}) == "capacity_partial"
    assert trade_block_reason_from_fields(**base) == "filled"


def test_trade_event_row_is_report_stable():
    row = trade_event_row(
        signal_date=pd.Timestamp("2021-01-04"),
        entry_date=pd.Timestamp("2021-01-05"),
        symbol="000001",
        side="buy",
        desired_notional=1_000.0,
        filled_notional=600.0,
        weight_before=0.0,
        weight_after=0.06,
        reason="capacity_partial",
        entry_open=10.0,
        entry_amount=100_000.0,
    )

    assert row["signal_date"] == "2021-01-04"
    assert row["entry_date"] == "2021-01-05"
    assert row["status"] == "partial"
    assert row["unfilled_notional"] == 400.0


def test_side_cost_fraction_includes_fee_slippage_and_impact():
    cost = side_cost_fraction_from_amounts(
        amounts=np.asarray([100_000.0, 200_000.0]),
        valid=np.asarray([True, True]),
        notionals=np.asarray([1_000.0, 2_000.0]),
        equity_cash=100_000.0,
        side="buy",
        buy_cost=0.0003,
        sell_cost=0.0008,
        slippage_bps=5.0,
        impact_bps_per_pct_amount=2.0,
    )

    expected_rate = 0.0003 + (5.0 + 2.0) / 10_000.0
    assert abs(cost - (3_000.0 * expected_rate / 100_000.0)) < 1e-12


if __name__ == "__main__":
    test_trade_status_classifies_blocked_partial_and_filled()
    test_trade_block_reason_preserves_execution_priority()
    test_trade_event_row_is_report_stable()
    test_side_cost_fraction_includes_fee_slippage_and_impact()

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import backtest_dynamic_rebalance as dynamic  # noqa: E402
from backtest_dynamic_rebalance import CompactDayData  # noqa: E402
from professional_quant.execution.config import default_execution_config  # noqa: E402
from professional_quant.execution.constraints import limit_rate_for_symbol, locked_limit_masks, status_bool  # noqa: E402


def test_default_execution_config_matches_formal_runner_baseline():
    config = default_execution_config()

    assert config.buy_cost == 0.0003
    assert config.sell_cost == 0.0008
    assert config.slippage_bps == 5.0
    assert config.capacity_pct_of_amount == 0.02
    assert config.capacity_equity_mode == "compound"
    assert config.lot_size == 100
    assert config.block_limit_trades is True


def test_limit_rates_distinguish_main_growth_and_st():
    assert limit_rate_for_symbol("000001", "主板", False) == 0.10
    assert limit_rate_for_symbol("000001", "主板", True) == 0.05
    assert limit_rate_for_symbol("300001", "创业板", False) == 0.20
    assert limit_rate_for_symbol("688001", "科创板", False) == 0.20


def test_locked_limit_and_suspension_masks_block_expected_side():
    prev_close = np.asarray([10.0, 10.0, 10.0, 10.0])
    entry_open = np.asarray([10.99, 9.01, 10.2, 10.0])
    entry_high = np.asarray([10.99, 9.01, 10.3, 10.0])
    entry_low = np.asarray([10.99, 9.01, 10.0, 10.0])
    rates = np.asarray([0.10, 0.10, 0.10, 0.10])
    suspended = np.asarray([False, False, True, False])

    buyable, sellable = locked_limit_masks(
        prev_close=prev_close,
        entry_open=entry_open,
        entry_high=entry_high,
        entry_low=entry_low,
        limit_rate=rates,
        entry_suspended=suspended,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )

    assert buyable.tolist() == [False, True, False, True]
    assert sellable.tolist() == [True, False, False, True]


def test_status_bool_normalizes_nullable_status_flags():
    series = pd.Series([1, 0, None, "1", "bad"])

    assert status_bool(series, default=False).tolist() == [True, False, False, True, False]
    assert status_bool(series, default=True).tolist() == [True, False, True, True, True]


def test_execute_rebalance_filters_nonfinite_current_weights():
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=1.0,
        capacity_equity_mode="initial",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    dynamic.G_INITIAL_CASH = 1_000_000.0
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_INDUSTRY_WEIGHT = 0.0
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002"], dtype=str),
        entry_open=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_low=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_amount=np.asarray([100_000_000.0, 100_000_000.0], dtype=np.float32),
        entry_volume=np.asarray([1_000_000.0, 1_000_000.0], dtype=np.float32),
        entry_buyable=np.asarray([True, True]),
        entry_sellable=np.asarray([True, True]),
        signal_allowed=np.asarray([True, True]),
        features=np.zeros((2, 1), dtype=np.float32),
        amount21=np.asarray([100_000_000.0, 100_000_000.0], dtype=np.float32),
        close=np.asarray([10.0, 10.0], dtype=np.float32),
        trend_masks={},
    )

    result, _ = dynamic.execute_rebalance(
        day,
        current_symbols=np.asarray(["000001", "000002"], dtype=str),
        current_entry_open=np.asarray([10.0, 10.0], dtype=np.float32),
        target_symbols=np.asarray([], dtype=str),
        equity=1.0,
        current_weights=np.asarray([np.inf, np.nan]),
    )

    assert result.current_symbols.tolist() == []
    assert np.isfinite(result.current_weights).all()


def test_sell_residual_below_lot_size_can_clear_position_but_buy_cannot():
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_INDUSTRY_WEIGHT = 0.0
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    dynamic.G_INITIAL_CASH = 1_000_000
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=1.0,
        capacity_equity_mode="compound",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002"]),
        entry_open=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([10.5, 10.5], dtype=np.float32),
        entry_low=np.asarray([9.5, 9.5], dtype=np.float32),
        entry_amount=np.asarray([100_000_000.0, 100_000_000.0], dtype=np.float32),
        entry_volume=np.asarray([1_000_000.0, 1_000_000.0], dtype=np.float32),
        entry_buyable=np.asarray([True, True]),
        entry_sellable=np.asarray([True, True]),
        signal_allowed=np.asarray([True, True]),
        features=np.zeros((2, 2), dtype=np.float32),
        amount21=np.asarray([100_000_000.0, 100_000_000.0], dtype=np.float32),
        close=np.asarray([10.0, 10.0], dtype=np.float32),
        trend_masks={},
    )

    sell_result, _ = dynamic.execute_rebalance(
        day,
        current_symbols=np.asarray(["000001"], dtype=str),
        current_entry_open=np.asarray([10.0], dtype=np.float32),
        target_symbols=np.asarray([], dtype=str),
        equity=1.0,
        current_weights=np.asarray([0.0005], dtype=np.float32),
    )
    assert sell_result.current_symbols.tolist() == []
    assert sell_result.trade_events[0]["status"] == "filled"

    buy_result, _ = dynamic.execute_rebalance(
        day,
        current_symbols=np.asarray([], dtype=str),
        current_entry_open=np.asarray([], dtype=np.float32),
        target_symbols=np.asarray(["000002"], dtype=str),
        equity=1.0,
        target_weight_override=0.0005,
    )
    assert buy_result.current_symbols.tolist() == []
    assert buy_result.trade_events[0]["status"] == "blocked"
    assert buy_result.trade_events[0]["reason"] == "lot_size_block"


if __name__ == "__main__":
    test_default_execution_config_matches_formal_runner_baseline()
    test_limit_rates_distinguish_main_growth_and_st()
    test_locked_limit_and_suspension_masks_block_expected_side()
    test_status_bool_normalizes_nullable_status_flags()
    test_execute_rebalance_filters_nonfinite_current_weights()
    test_sell_residual_below_lot_size_can_clear_position_but_buy_cannot()

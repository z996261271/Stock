import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.backtest.day_data import (  # noqa: E402
    bool_for_symbols,
    close_for_symbols,
    open_for_symbols,
    open_to_open_return,
    return_since_entry,
    values_for_symbols,
    weighted_symbol_return,
)


@dataclass
class DummyDay:
    symbols: np.ndarray
    entry_open: np.ndarray
    close: np.ndarray
    entry_amount: np.ndarray
    entry_buyable: np.ndarray


def _day() -> DummyDay:
    return DummyDay(
        symbols=np.asarray(["000001", "000002", "000003"], dtype=str),
        entry_open=np.asarray([10.0, 20.0, np.nan], dtype=np.float32),
        close=np.asarray([11.0, 18.0, 30.0], dtype=np.float32),
        entry_amount=np.asarray([100_000.0, 200_000.0, 0.0], dtype=np.float32),
        entry_buyable=np.asarray([True, False, True], dtype=bool),
    )


def test_day_data_symbol_lookup_preserves_order_and_valid_mask():
    symbols = np.asarray(["000002", "000004", "000001"], dtype=str)

    opens, open_valid = open_for_symbols(_day(), symbols)
    amounts, amount_valid = values_for_symbols(_day(), symbols, "entry_amount")
    buyable = bool_for_symbols(_day(), symbols, "entry_buyable")

    assert opens[0] == 20.0
    assert np.isnan(opens[1])
    assert opens[2] == 10.0
    assert open_valid.tolist() == [True, False, True]
    assert amounts[0] == 200_000.0
    assert np.isnan(amounts[1])
    assert amounts[2] == 100_000.0
    assert amount_valid.tolist() == [True, False, True]
    assert buyable.tolist() == [False, False, True]


def test_day_data_returns_handle_weights_and_invalid_prices():
    symbols = np.asarray(["000001", "000002", "000003"], dtype=str)
    previous_open = np.asarray([9.0, 20.0, 30.0], dtype=np.float32)
    weights = np.asarray([0.5, 0.25, 0.25], dtype=np.float32)

    period_return, updated_open = open_to_open_return(_day(), symbols, previous_open, weights)
    entry_return = return_since_entry(_day(), symbols, np.asarray([10.0, 20.0, 30.0], dtype=np.float32), weights)

    assert abs(period_return - ((10.0 / 9.0 - 1.0) * 0.5)) < 1e-7
    assert updated_open.tolist() == [10.0, 20.0, 30.0]
    assert abs(entry_return - ((11.0 / 10.0 - 1.0) * 0.5 + (18.0 / 20.0 - 1.0) * 0.25)) < 1e-7
    assert weighted_symbol_return(np.asarray([], dtype=np.float32), None) == 0.0
    assert close_for_symbols(_day(), np.asarray([], dtype=str))[0].tolist() == []


if __name__ == "__main__":
    test_day_data_symbol_lookup_preserves_order_and_valid_mask()
    test_day_data_returns_handle_weights_and_invalid_prices()

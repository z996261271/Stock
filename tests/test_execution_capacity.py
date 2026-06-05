import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.execution.capacity import capacity_fill_notional_from_arrays  # noqa: E402


def test_capacity_fill_notional_clips_buys_to_lots_and_capacity():
    filled, filled_mask, partial_mask = capacity_fill_notional_from_arrays(
        prices=np.asarray([10.0, 20.0, 30.0]),
        price_valid=np.asarray([True, True, True]),
        amounts=np.asarray([100_000.0, 50_000.0, 10_000.0]),
        amount_valid=np.asarray([True, True, True]),
        desired_notional=np.asarray([3_000.0, 2_000.0, 500.0]),
        executable=np.asarray([True, True, True]),
        side="buy",
        lot_size=100,
        capacity_pct_of_amount=0.02,
    )

    assert filled.tolist() == [2_000.0, 0.0, 0.0]
    assert filled_mask.tolist() == [True, False, False]
    assert partial_mask.tolist() == [True, False, False]


def test_capacity_fill_notional_preserves_sell_capacity_without_lot_rounding():
    filled, filled_mask, partial_mask = capacity_fill_notional_from_arrays(
        prices=np.asarray([10.0, 20.0]),
        price_valid=np.asarray([True, True]),
        amounts=np.asarray([100_000.0, 1_000.0]),
        amount_valid=np.asarray([True, True]),
        desired_notional=np.asarray([1_500.0, 500.0]),
        executable=np.asarray([True, False]),
        side="sell",
        lot_size=100,
        capacity_pct_of_amount=0.02,
    )

    assert filled.tolist() == [1_500.0, 0.0]
    assert filled_mask.tolist() == [True, False]
    assert partial_mask.tolist() == [False, False]


def test_capacity_fill_notional_blocks_invalid_inputs():
    filled, filled_mask, partial_mask = capacity_fill_notional_from_arrays(
        prices=np.asarray([0.0, 10.0, 10.0]),
        price_valid=np.asarray([False, True, True]),
        amounts=np.asarray([100_000.0, 0.0, 100_000.0]),
        amount_valid=np.asarray([True, True, True]),
        desired_notional=np.asarray([1_000.0, 1_000.0, -1.0]),
        executable=np.asarray([True, True, True]),
        side="buy",
        lot_size=100,
        capacity_pct_of_amount=0.02,
    )

    assert filled.tolist() == [0.0, 0.0, 0.0]
    assert filled_mask.tolist() == [False, False, False]
    assert partial_mask.tolist() == [False, False, False]


if __name__ == "__main__":
    test_capacity_fill_notional_clips_buys_to_lots_and_capacity()
    test_capacity_fill_notional_preserves_sell_capacity_without_lot_rounding()
    test_capacity_fill_notional_blocks_invalid_inputs()

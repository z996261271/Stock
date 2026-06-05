import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.execution.precision import Money, Price, Quantity, price_quantity_for_target_notional  # noqa: E402


def test_price_quantity_money_round_to_a_share_lot_rules():
    price, quantity, money = price_quantity_for_target_notional(12_345.67, 10.037, lot_size=100)

    assert price is not None
    assert price.value == Decimal("10.04")
    assert quantity.shares == 1200
    assert money.value == Decimal("12048.00")
    assert Price.from_float(0) is None
    assert Quantity.from_notional(1_000.0, None).shares == 0
    assert Money.from_price_quantity(None, Quantity(100)).as_float() == 0.0


if __name__ == "__main__":
    test_price_quantity_money_round_to_a_share_lot_rules()


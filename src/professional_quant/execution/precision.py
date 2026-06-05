"""A-share price, quantity, and money precision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP


PRICE_TICK = Decimal("0.01")
MONEY_CENT = Decimal("0.01")
DEFAULT_LOT_SIZE = 100


@dataclass(frozen=True)
class Price:
    value: Decimal

    @classmethod
    def from_float(cls, value: float | int | str | Decimal | None, *, tick: Decimal = PRICE_TICK) -> "Price | None":
        if value is None:
            return None
        decimal = Decimal(str(value))
        if decimal <= 0:
            return None
        return cls(decimal.quantize(tick, rounding=ROUND_HALF_UP))

    def as_float(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Quantity:
    shares: int

    @classmethod
    def from_notional(cls, notional: float, price: Price | None, *, lot_size: int = DEFAULT_LOT_SIZE) -> "Quantity":
        if price is None or notional <= 0:
            return cls(0)
        lot = max(int(lot_size), 1)
        raw = Decimal(str(notional)) / price.value
        lots = (raw / Decimal(lot)).to_integral_value(rounding=ROUND_DOWN)
        return cls(int(lots) * lot)

    def as_float(self) -> float:
        return float(self.shares)


@dataclass(frozen=True)
class Money:
    value: Decimal

    @classmethod
    def from_price_quantity(cls, price: Price | None, quantity: Quantity) -> "Money":
        if price is None or quantity.shares <= 0:
            return cls(Decimal("0.00"))
        return cls((price.value * Decimal(quantity.shares)).quantize(MONEY_CENT, rounding=ROUND_HALF_UP))

    def as_float(self) -> float:
        return float(self.value)


def price_quantity_for_target_notional(
    target_notional: float,
    raw_price: float | int | str | Decimal | None,
    *,
    lot_size: int = DEFAULT_LOT_SIZE,
) -> tuple[Price | None, Quantity, Money]:
    """Return rounded price, lot-aligned quantity, and executable notional."""
    price = Price.from_float(raw_price)
    quantity = Quantity.from_notional(target_notional, price, lot_size=lot_size)
    money = Money.from_price_quantity(price, quantity)
    return price, quantity, money


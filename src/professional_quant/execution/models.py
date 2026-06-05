"""Shared signal, order, fill, and position objects for backtest/paper parity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SignalAction = Literal["buy", "sell", "hold"]
OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class Signal:
    strategy: str
    symbol: str
    signal_date: str
    action: SignalAction
    signal_id: str
    score: float | None = None
    weight: float | None = None
    price: float | None = None
    reason: str = "daily_signal"
    strategy_version: str | None = None
    signal_tag: str | None = None
    entry_tag: str | None = None
    exit_tag: str | None = None
    risk_tag: str | None = None
    source_factor_set: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def payload_with_tags(self) -> dict[str, Any]:
        payload = dict(self.payload)
        for key in (
            "strategy_version",
            "signal_tag",
            "entry_tag",
            "exit_tag",
            "risk_tag",
            "source_factor_set",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    def to_record(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "action": self.action,
            "score": self.score,
            "weight": self.weight,
            "reason": self.reason,
            "payload": self.payload_with_tags(),
        }


@dataclass(frozen=True)
class OrderIntent:
    strategy: str
    symbol: str
    signal_id: str
    signal_date: str
    entry_date: str
    side: OrderSide
    quantity: float
    price: float | None
    amount: float | None
    reason: str
    target_weight: float | None = None
    strategy_version: str | None = None
    entry_tag: str | None = None
    exit_tag: str | None = None
    risk_tag: str | None = None

    def to_trade_record(self, status: str = "planned") -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "amount": self.amount,
            "status": status,
            "signal_id": self.signal_id,
            "reason": self.reason,
            "payload": {
                "strategy_version": self.strategy_version,
                "entry_tag": self.entry_tag,
                "exit_tag": self.exit_tag,
                "risk_tag": self.risk_tag,
                "target_weight": self.target_weight,
            },
        }


@dataclass(frozen=True)
class OrderDecision:
    strategy: str
    symbol: str
    signal_id: str
    signal_date: str
    entry_date: str
    side: OrderSide
    accepted: bool
    quantity: float
    price: float | None
    amount: float | None
    reason: str
    blocked_reason: str | None = None
    target_weight: float | None = None
    strategy_version: str | None = None
    entry_tag: str | None = None
    exit_tag: str | None = None
    risk_tag: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_intent(
        cls,
        intent: OrderIntent,
        *,
        accepted: bool = True,
        blocked_reason: str | None = None,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> "OrderDecision":
        return cls(
            strategy=intent.strategy,
            symbol=intent.symbol,
            signal_id=intent.signal_id,
            signal_date=intent.signal_date,
            entry_date=intent.entry_date,
            side=intent.side,
            accepted=accepted,
            quantity=intent.quantity,
            price=intent.price,
            amount=intent.amount,
            reason=reason or intent.reason,
            blocked_reason=blocked_reason,
            target_weight=intent.target_weight,
            strategy_version=intent.strategy_version,
            entry_tag=intent.entry_tag,
            exit_tag=intent.exit_tag,
            risk_tag=intent.risk_tag,
            payload=payload or {},
        )

    def to_trade_record(self) -> dict[str, Any]:
        status = "planned" if self.accepted else "blocked"
        payload = {
            "strategy_version": self.strategy_version,
            "entry_tag": self.entry_tag,
            "exit_tag": self.exit_tag,
            "risk_tag": self.risk_tag,
            "target_weight": self.target_weight,
            "blocked_reason": self.blocked_reason,
            **self.payload,
        }
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "amount": self.amount,
            "status": status,
            "signal_id": self.signal_id,
            "reason": self.reason,
            "payload": {key: value for key, value in payload.items() if value is not None},
        }


@dataclass(frozen=True)
class ExecutionFill:
    strategy: str
    symbol: str
    signal_id: str | None
    trade_date: str
    side: OrderSide
    quantity: float
    price: float | None
    amount: float | None
    status: str
    reason: str | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class PositionSnapshot:
    strategy: str
    symbol: str
    as_of_date: str
    quantity: float
    avg_cost: float | None
    market_value: float | None
    cash: float | None
    source_signal_id: str | None = None
    target_weight: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "market_value": self.market_value,
            "cash": self.cash,
            "payload": {
                "source_signal_id": self.source_signal_id,
                "target_weight": self.target_weight,
            },
        }


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    strategy: str
    signal_date: str
    entry_date: str | None
    counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

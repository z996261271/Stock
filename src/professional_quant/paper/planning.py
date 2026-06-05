"""Paper-trading signal normalization and order-plan construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from professional_quant.execution.models import OrderIntent, PositionSnapshot, Signal
from professional_quant.execution.precision import DEFAULT_LOT_SIZE, price_quantity_for_target_notional


def normalize_signal_row(row: dict[str, Any], *, strategy: str, signal_date: str, signal_id: str) -> Signal:
    symbol = str(row.get("symbol", "")).strip()
    if not symbol:
        raise ValueError(f"signal row missing symbol: {row}")
    action = str(row.get("action", row.get("side", "buy"))).strip().lower()
    if action not in {"buy", "sell", "hold"}:
        raise ValueError(f"unsupported action for {symbol}: {action}")
    weight = row.get("weight")
    score = row.get("score")
    price = row.get("price", row.get("entry_open"))
    reason = str(row.get("reason", row.get("formula", row.get("signal_tag", "daily_signal"))))
    payload = {key: _json_safe(value) for key, value in row.items()}
    return Signal(
        strategy=strategy,
        symbol=symbol,
        signal_date=signal_date,
        action=action,  # type: ignore[arg-type]
        signal_id=signal_id,
        score=float(score) if pd.notna(score) else None,
        weight=float(weight) if pd.notna(weight) else None,
        price=float(price) if pd.notna(price) else None,
        reason=reason,
        strategy_version=_optional_text(row.get("strategy_version")),
        signal_tag=_optional_text(row.get("signal_tag", row.get("formula"))),
        entry_tag=_optional_text(row.get("entry_tag")),
        exit_tag=_optional_text(row.get("exit_tag")),
        risk_tag=_optional_text(row.get("risk_tag")),
        source_factor_set=_optional_text(row.get("source_factor_set", row.get("formula"))),
        payload=payload,
    )


def build_paper_plan(
    *,
    raw_rows: list[dict[str, Any]],
    strategy: str,
    signal_date: str,
    entry_date: str,
    cash: float,
    signal_id_fn,
    lot_size: int = DEFAULT_LOT_SIZE,
) -> dict[str, Any]:
    signals = [
        normalize_signal_row(
            row,
            strategy=strategy,
            signal_date=signal_date,
            signal_id=signal_id_fn(strategy, str(row.get("symbol", "")).strip(), signal_date, str(row.get("action", row.get("side", "buy"))).strip().lower()),
        )
        for row in raw_rows
    ]
    buy_weight = sum(float(signal.weight or 0.0) for signal in signals if signal.action == "buy")
    signal_records: list[dict[str, Any]] = []
    position_records: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    for signal in signals:
        signal_records.append(signal.to_record())
        intent = order_intent_from_signal(signal, entry_date=entry_date, cash=cash, lot_size=lot_size)
        if signal.action == "buy" and intent is not None:
            position_records.append(position_from_buy_intent(intent, cash=cash, buy_weight=buy_weight).to_record())
            trade_records.append(intent.to_trade_record())
        elif signal.action == "sell" and intent is not None:
            trade_records.append(intent.to_trade_record())
    return {
        "signal_date": signal_date,
        "entry_date": entry_date,
        "cash": cash,
        "signals": signal_records,
        "positions": position_records,
        "paper_trades": trade_records,
        "object_counts": {
            "signals": len(signals),
            "order_intents": len(trade_records),
            "position_snapshots": len(position_records),
        },
        "execution_precision": {
            "lot_size": int(lot_size),
            "price_tick": "0.01",
            "money_cent": "0.01",
        },
    }


def order_intent_from_signal(
    signal: Signal,
    *,
    entry_date: str,
    cash: float,
    lot_size: int = DEFAULT_LOT_SIZE,
) -> OrderIntent | None:
    if signal.action == "hold":
        return None
    if signal.action == "buy":
        weight = float(signal.weight or 0.0)
        target_notional = cash * weight
        price, quantity, executable_notional = price_quantity_for_target_notional(
            target_notional,
            signal.price,
            lot_size=lot_size,
        )
        return OrderIntent(
            strategy=signal.strategy,
            symbol=signal.symbol,
            signal_id=signal.signal_id,
            signal_date=signal.signal_date,
            entry_date=entry_date,
            side="buy",
            quantity=quantity.as_float(),
            price=price.as_float() if price is not None else None,
            amount=executable_notional.as_float(),
            reason=signal.reason,
            target_weight=weight,
            strategy_version=signal.strategy_version,
            entry_tag=signal.entry_tag or signal.signal_tag,
            risk_tag=signal.risk_tag,
        )
    quantity = float(signal.payload.get("quantity") or 0.0)
    return OrderIntent(
        strategy=signal.strategy,
        symbol=signal.symbol,
        signal_id=signal.signal_id,
        signal_date=signal.signal_date,
        entry_date=entry_date,
        side="sell",
        quantity=quantity,
        price=signal.price,
        amount=None,
        reason=signal.reason,
        strategy_version=signal.strategy_version,
        exit_tag=signal.exit_tag or signal.signal_tag,
        risk_tag=signal.risk_tag,
    )


def position_from_buy_intent(intent: OrderIntent, *, cash: float, buy_weight: float) -> PositionSnapshot:
    return PositionSnapshot(
        strategy=intent.strategy,
        symbol=intent.symbol,
        as_of_date=intent.entry_date,
        quantity=intent.quantity,
        avg_cost=intent.price,
        market_value=intent.amount,
        cash=max(cash * (1.0 - buy_weight), 0.0),
        source_signal_id=intent.signal_id,
        target_weight=intent.target_weight,
    )


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None

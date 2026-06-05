"""Strategy callback protocol for daily A-share workflows."""

from __future__ import annotations

from typing import Any, Protocol

from professional_quant.core.context import RunContext
from professional_quant.execution.models import OrderIntent, Signal


class StrategyProtocol(Protocol):
    """Minimal callback surface shared by research, backtest, and paper workflows."""

    def version(self) -> str:
        """Return a stable strategy version identifier."""

    def prepare_universe(self, context: RunContext) -> Any:
        """Prepare the tradable universe for the current run."""

    def compute_factors(self, context: RunContext) -> Any:
        """Compute or load factor data for the current run."""

    def select_targets(self, context: RunContext) -> list[Signal]:
        """Select target signals for the current run."""

    def confirm_entry(self, order_intent: OrderIntent, context: RunContext) -> bool:
        """Return whether an entry intent is allowed."""

    def confirm_exit(self, order_intent: OrderIntent, context: RunContext) -> bool:
        """Return whether an exit intent is allowed."""

    def size_order(self, signal: Signal, context: RunContext) -> OrderIntent | None:
        """Convert one signal into an order intent."""

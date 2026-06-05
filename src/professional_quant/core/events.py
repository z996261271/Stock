"""Lightweight event bus for quant workflow lifecycle hooks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


DATA_READY = "DATA_READY"
QUALITY_GATE_PASSED = "QUALITY_GATE_PASSED"
SIGNAL_GENERATED = "SIGNAL_GENERATED"
ORDER_PLAN_BUILT = "ORDER_PLAN_BUILT"
PAPER_RUN_RECORDED = "PAPER_RUN_RECORDED"
REPORT_PUBLISHED = "REPORT_PUBLISHED"
ALERT_DISPATCHED = "ALERT_DISPATCHED"

LifecycleHandler = Callable[["LifecycleEvent"], None]


@dataclass(frozen=True)
class LifecycleEvent:
    """One immutable workflow lifecycle event."""

    name: str
    run_id: str
    strategy: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    emitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class EventBus:
    """In-process event bus with deterministic handler order and audit history."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[LifecycleHandler]] = defaultdict(list)
        self._history: list[LifecycleEvent] = []

    def subscribe(self, event_name: str, handler: LifecycleHandler) -> None:
        self._handlers[event_name].append(handler)

    def emit(
        self,
        event_name: str,
        *,
        run_id: str,
        strategy: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> LifecycleEvent:
        event = LifecycleEvent(event_name, run_id=run_id, strategy=strategy, payload=payload or {})
        self._history.append(event)
        for handler in [*self._handlers.get(event_name, []), *self._handlers.get("*", [])]:
            handler(event)
        return event

    @property
    def history(self) -> tuple[LifecycleEvent, ...]:
        return tuple(self._history)


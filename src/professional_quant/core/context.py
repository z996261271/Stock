"""Run context shared by backtest, report, and paper workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from professional_quant.core.events import EventBus, LifecycleEvent


@dataclass
class RunContext:
    """Small explicit runtime context passed across workflow layers."""

    run_id: str
    strategy: str
    mode: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    event_bus: EventBus = field(default_factory=EventBus)

    def emit(self, event_name: str, payload: dict[str, Any] | None = None) -> LifecycleEvent:
        return self.event_bus.emit(event_name, run_id=self.run_id, strategy=self.strategy, payload=payload)

    def add_artifact(self, name: str, path: str | Path) -> None:
        self.artifacts[name] = str(path)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "run_context.v1",
            "run_id": self.run_id,
            "strategy": self.strategy,
            "mode": self.mode,
            "started_at": self.started_at,
            "metadata": self.metadata,
            "artifacts": self.artifacts,
            "events": [event.__dict__ for event in self.event_bus.history],
        }


"""Workflow mod protocol and manager."""

from __future__ import annotations

from typing import Protocol

from professional_quant.core.context import RunContext


class WorkflowMod(Protocol):
    """Extension point for workflow lifecycle hooks."""

    name: str

    def install(self, context: RunContext) -> None:
        """Register event handlers or add metadata to a run context."""


class ModManager:
    """Install a list of workflow mods into one run context."""

    def __init__(self, mods: list[WorkflowMod] | None = None) -> None:
        self.mods = mods or []

    def install_all(self, context: RunContext) -> list[str]:
        installed: list[str] = []
        for mod in self.mods:
            mod.install(context)
            installed.append(mod.name)
        context.metadata["installed_mods"] = installed
        return installed

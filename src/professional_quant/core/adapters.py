"""Adapter maturity and health metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ADAPTER_MATURITIES = ("planned", "beta", "stable", "deprecated")
ADAPTER_STATUSES = ("active", "degraded", "unavailable", "disabled")


@dataclass(frozen=True)
class AdapterMetadata:
    name: str
    kind: str
    maturity: str = "beta"
    status: str = "active"
    version: str | None = None
    message: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_non_empty("kind", self.kind)
        validate_adapter_maturity(self.maturity)
        validate_adapter_status(self.status)

    @property
    def is_usable(self) -> bool:
        return self.status in {"active", "degraded"} and self.maturity != "planned"

    @property
    def is_deprecated(self) -> bool:
        return self.maturity == "deprecated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adapter_metadata(
    *,
    name: str,
    kind: str,
    maturity: str = "beta",
    status: str = "active",
    version: str | None = None,
    message: str | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
) -> AdapterMetadata:
    return AdapterMetadata(
        name=name,
        kind=kind,
        maturity=normalize_adapter_maturity(maturity),
        status=normalize_adapter_status(status),
        version=version,
        message=message,
        tags=tuple(tags or ()),
    )


def normalize_adapter_maturity(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in ADAPTER_MATURITIES:
        raise ValueError(f"unknown adapter maturity: {value!r}")
    return normalized


def normalize_adapter_status(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in ADAPTER_STATUSES:
        raise ValueError(f"unknown adapter status: {value!r}")
    return normalized


def validate_adapter_maturity(value: str) -> None:
    normalize_adapter_maturity(value)


def validate_adapter_status(value: str) -> None:
    normalize_adapter_status(value)


def _require_non_empty(field_name: str, value: str) -> None:
    if not str(value).strip():
        raise ValueError(f"{field_name} must be non-empty")

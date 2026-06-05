"""Shared provider result contract for data ingestion scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol

from professional_quant.core.adapters import AdapterMetadata, adapter_metadata


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    dataset: str
    symbol: str | None
    start_date: str | None
    end_date: str | None
    rows: int
    status: str
    error: str | None = None
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    metadata: dict[str, Any] = field(default_factory=dict)
    adapter: AdapterMetadata | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataProvider(Protocol):
    provider_name: str

    def fetch(self, *, symbol: str | None, start_date: str | None, end_date: str | None) -> ProviderResult:
        """Fetch one logical dataset and return a standardized status row."""


def provider_success(
    *,
    provider: str,
    dataset: str,
    rows: int,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metadata: dict[str, Any] | None = None,
    adapter: AdapterMetadata | None = None,
    maturity: str = "beta",
    adapter_status: str = "active",
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        dataset=dataset,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        rows=int(rows),
        status="ok",
        metadata=metadata or {},
        adapter=adapter
        or adapter_metadata(
            name=provider,
            kind="data_provider",
            maturity=maturity,
            status=adapter_status,
        ),
    )


def provider_failure(
    *,
    provider: str,
    dataset: str,
    error: str,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    rows: int = 0,
    metadata: dict[str, Any] | None = None,
    adapter: AdapterMetadata | None = None,
    maturity: str = "beta",
    adapter_status: str = "degraded",
) -> ProviderResult:
    return ProviderResult(
        provider=provider,
        dataset=dataset,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        rows=int(rows),
        status="error",
        error=error,
        metadata=metadata or {},
        adapter=adapter
        or adapter_metadata(
            name=provider,
            kind="data_provider",
            maturity=maturity,
            status=adapter_status,
        ),
    )

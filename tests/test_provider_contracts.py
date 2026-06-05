import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.core.adapters import adapter_metadata  # noqa: E402
from professional_quant.data.providers.base import provider_failure, provider_success  # noqa: E402


def test_provider_result_success_and_failure_contracts_are_serializable():
    success = provider_success(
        provider="eastmoney",
        dataset="daily_bars",
        symbol="000001",
        start_date="2020-01-01",
        end_date="2020-01-31",
        rows=10,
        metadata={"adjust": "raw"},
    )
    failure = provider_failure(
        provider="baostock",
        dataset="daily_bars",
        symbol="000001",
        start_date="2020-01-01",
        end_date="2020-01-31",
        error="network timeout",
    )

    assert success.ok is True
    assert success.to_dict()["metadata"]["adjust"] == "raw"
    assert success.to_dict()["adapter"]["maturity"] == "beta"
    assert success.to_dict()["adapter"]["status"] == "active"
    assert failure.ok is False
    assert failure.to_dict()["status"] == "error"
    assert failure.to_dict()["rows"] == 0
    assert failure.to_dict()["adapter"]["status"] == "degraded"


def test_adapter_metadata_validates_maturity_and_status():
    adapter = adapter_metadata(
        name="eastmoney",
        kind="data_provider",
        maturity="stable",
        status="active",
        version="akshare",
        tags=["daily_bars"],
    )

    assert adapter.is_usable is True
    assert adapter.to_dict()["tags"] == ("daily_bars",)

    deprecated = adapter_metadata(name="legacy_csv", kind="data_provider", maturity="deprecated", status="disabled")
    assert deprecated.is_usable is False
    assert deprecated.is_deprecated is True


if __name__ == "__main__":
    test_provider_result_success_and_failure_contracts_are_serializable()
    test_adapter_metadata_validates_maturity_and_status()

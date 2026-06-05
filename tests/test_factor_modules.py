import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.factor.analysis import build_factor_report  # noqa: E402


def _picks_with_forward_returns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2021-01-04",
                "symbol": "000001",
                "industry_label": "bank",
                "weight": 0.25,
                "score": 0.90,
                "forward_return": 0.08,
            },
            {
                "signal_date": "2021-01-04",
                "symbol": "000002",
                "industry_label": "energy",
                "weight": 0.25,
                "score": 0.10,
                "forward_return": -0.04,
            },
            {
                "signal_date": "2021-01-05",
                "symbol": "000001",
                "industry_label": "bank",
                "weight": 0.20,
                "score": 0.80,
                "forward_return": 0.04,
            },
            {
                "signal_date": "2021-01-05",
                "symbol": "000003",
                "industry_label": "tech",
                "weight": 0.30,
                "score": 0.20,
                "forward_return": -0.02,
            },
        ]
    )


def test_factor_report_computes_ic_quantile_returns_and_stability():
    report = build_factor_report(_picks_with_forward_returns(), quantiles=2)

    assert report["status"] == "complete"
    assert round(report["information_coefficient"]["summary"]["mean_rank_ic"], 8) == 1.0
    assert report["top_bottom_spread"]["summary"]["mean_spread"] == 0.09
    assert report["quantile_returns"]["rows"][0]["quantile"] == 1
    assert report["quantile_returns"]["rows"][1]["mean_return"] == 0.06
    assert report["quantile_turnover"]["summary"]["mean_turnover"] == 0.5
    assert report["rank_autocorrelation"]["rows"][1]["overlap"] == 1
    assert report["industry_summary"]["rows"][0]["industry_label"] == "bank"


def test_factor_report_degrades_when_formal_picks_lack_forward_returns():
    report = build_factor_report(
        pd.DataFrame(
            [
                {"signal_date": "2021-01-04", "symbol": "000001", "industry_label": "bank", "score": 0.9},
                {"signal_date": "2021-01-04", "symbol": "000002", "industry_label": "energy", "score": 0.2},
                {"signal_date": "2021-01-05", "symbol": "000002", "industry_label": "energy", "score": 0.7},
            ]
        ),
        quantiles=2,
    )

    assert report["status"] == "missing_forward_returns"
    assert report["information_coefficient"]["status"] == "missing_forward_returns"
    assert report["quantile_returns"]["status"] == "missing_forward_returns"
    assert report["summary"]["signal_dates"] == 2
    assert report["quantile_turnover"]["summary"]["mean_turnover"] is not None
    assert any(warning["type"] == "missing_forward_returns" for warning in report["warnings"])


if __name__ == "__main__":
    test_factor_report_computes_ic_quantile_returns_and_stability()
    test_factor_report_degrades_when_formal_picks_lack_forward_returns()

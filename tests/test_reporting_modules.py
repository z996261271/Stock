import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.backtest.reporting import (  # noqa: E402
    compute_equal_weight_benchmark,
    period_return_breakdown,
    professional_performance_metrics,
    relative_performance_metrics,
)
from professional_quant.backtest.attribution import build_formal_attribution_report  # noqa: E402
from professional_quant.reporting.metadata import (  # noqa: E402
    default_split_policy,
    parse_json_metadata,
    parse_required_adjusts,
)
from professional_quant.reporting.registry import build_formal_release_registry  # noqa: E402
from professional_quant.risk.budget import risk_budget_report  # noqa: E402
from professional_quant.risk.defaults import (  # noqa: E402
    apply_formal_risk_defaults_to_mapping,
    apply_formal_risk_defaults_to_namespace,
)


def _equity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2020-12-31",
                "entry_date": "2021-01-04",
                "equity": 1_100_000.0,
                "period_return": 0.10,
                "trade": True,
                "active": True,
                "trade_count": 2,
                "blocked_buy_count": 0,
                "blocked_sell_count": 0,
                "turnover_value": 200_000.0,
            },
            {
                "signal_date": "2021-01-04",
                "entry_date": "2021-01-05",
                "equity": 1_045_000.0,
                "period_return": -0.05,
                "trade": False,
                "active": True,
                "trade_count": 0,
                "blocked_buy_count": 1,
                "blocked_sell_count": 0,
                "turnover_value": 0.0,
            },
            {
                "signal_date": "2021-01-05",
                "entry_date": "2021-02-01",
                "equity": 1_149_500.0,
                "period_return": 0.10,
                "trade": True,
                "active": True,
                "trade_count": 2,
                "blocked_buy_count": 0,
                "blocked_sell_count": 1,
                "turnover_value": 200_000.0,
            },
        ]
    )


def test_performance_breakdown_and_benchmark_metrics_are_report_ready():
    equity = _equity_frame()
    bars = pd.DataFrame(
        [
            {"symbol": "000001", "trade_date": "2021-01-04", "raw_close": 10.0},
            {"symbol": "000001", "trade_date": "2021-01-05", "raw_close": 11.0},
            {"symbol": "000001", "trade_date": "2021-02-01", "raw_close": 12.0},
            {"symbol": "000002", "trade_date": "2021-01-04", "raw_close": 20.0},
            {"symbol": "000002", "trade_date": "2021-01-05", "raw_close": 18.0},
            {"symbol": "000002", "trade_date": "2021-02-01", "raw_close": 19.8},
        ]
    )

    metrics = professional_performance_metrics(equity, 1_000_000.0)
    annual_rows = period_return_breakdown(equity, "Y")
    monthly_rows = period_return_breakdown(equity, "M")
    benchmark = compute_equal_weight_benchmark(
        bars,
        pd.Timestamp("2021-01-04"),
        pd.Timestamp("2021-02-01"),
        "unit_equal_weight",
    )
    relative = relative_performance_metrics(equity, benchmark)

    assert metrics["annual_volatility"] > 0
    assert metrics["sharpe"] is not None
    assert len(annual_rows) == 1
    assert annual_rows[0]["executed_trade_count"] == 4
    assert [row["period"] for row in monthly_rows] == ["2021-01", "2021-02"]
    assert benchmark["symbols"] == 2
    assert benchmark["periods"] == 2
    assert relative["matched_periods"] == 2


def test_risk_budget_summarizes_report_controls_and_industry_exposure():
    budget = risk_budget_report(
        {
            "max_drawdown": -0.1,
            "max_position_weight_observed": 0.5,
            "max_industry_weight_observed": 0.5,
            "unfilled_buy_value": 0.0,
            "unfilled_sell_value": 0.0,
            "max_period_turnover_pct": 1.0,
            "blocked_buy_count": 0,
            "blocked_sell_count": 0,
            "partial_buy_count": 0,
            "partial_sell_count": 0,
            "portfolio_risk_off_rate": 0.0,
            "avg_cash_weight": 0.0,
            "avg_invested_weight": 1.0,
        },
        pd.DataFrame(),
        pd.DataFrame(
            [
                {"industry_label": "bank", "weight": 0.5},
                {"industry_label": "energy", "weight": 0.5},
            ]
        ),
        {
            "portfolio_stop_loss": 0.0,
            "max_position_weight": 0.0,
            "max_industry_weight": 0.5,
            "capacity_pct_of_amount": 0.02,
            "max_turnover_pct": 0.0,
        },
    )

    assert any(row["name"] == "industry_concentration" for row in budget["risk_sources"])
    assert budget["industry_exposure_top"][0]["max_pick_weight"] == 0.5


def test_formal_risk_defaults_are_shared_for_config_and_cli_args():
    config = {"portfolio_stop_loss": 0.0, "max_position_weight": 0.0}
    args = argparse.Namespace(
        portfolio_stop_loss=0.0,
        max_position_weight=0.0,
        max_industry_weight=0.0,
        max_turnover_pct=0.0,
    )

    apply_formal_risk_defaults_to_mapping(config)
    apply_formal_risk_defaults_to_namespace(args)

    assert config["portfolio_stop_loss"] == 0.25
    assert config["max_position_weight"] == 0.2
    assert config["max_industry_weight"] == 0.35
    assert config["max_turnover_pct"] == 0.8
    assert args.portfolio_stop_loss == 0.25
    assert args.max_position_weight == 0.2
    assert args.max_industry_weight == 0.35
    assert args.max_turnover_pct == 0.8


def test_formal_metadata_helpers_parse_required_fields():
    split = default_split_policy(
        pd.Timestamp("2021-01-04"),
        pd.Timestamp("2021-12-31"),
        pd.Timestamp("2020-12-31"),
    )

    assert parse_required_adjusts("raw,qfq,hfq") == ("raw", "qfq", "hfq")
    assert parse_json_metadata('{"frozen": true}', "frozen_config") == {"frozen": True}
    assert split["current_result_segment"] == "walkforward"
    assert split["freeze_selection_date"] == "2020-12-31"


def test_formal_attribution_report_explains_cash_friction_and_capacity():
    metrics = {
        "config": {"strategy": "demo"},
        "is_formal_valid": True,
        "total_return": 0.10,
        "annual_return": 0.08,
        "max_drawdown": -0.12,
        "annual_breakdown": [
            {"period": "2021", "total_return": 0.10, "annual_return": 0.10, "max_drawdown": -0.08, "periods": 2}
        ],
        "benchmarks": {
            "main_board_equal_weight_raw_close": {
                "total_return": 0.30,
                "annual_return": 0.25,
                "max_drawdown": -0.10,
                "daily_returns": [
                    {"date": "2021-01-04", "return": 0.10},
                    {"date": "2021-01-05", "return": 0.05},
                ],
            }
        },
        "professional_metrics": {
            "relative_to_benchmarks": {
                "main_board_equal_weight_raw_close": {
                    "matched_periods": 2,
                    "total_excess_return": -0.15,
                    "information_ratio": -1.2,
                }
            }
        },
        "risk_budget": {"risk_sources": []},
    }
    equity = pd.DataFrame(
        [
            {
                "entry_date": "2021-01-04",
                "period_return": 0.02,
                "cash_weight": 0.30,
                "invested_weight": 0.70,
                "max_position_weight": 0.20,
                "max_industry_weight": 0.40,
                "top_industry": "bank",
                "blocked_buy_count": 1,
                "partial_buy_count": 1,
                "unfilled_buy_value": 100.0,
            },
            {
                "entry_date": "2021-01-05",
                "period_return": -0.01,
                "cash_weight": 0.25,
                "invested_weight": 0.75,
                "max_position_weight": 0.18,
                "max_industry_weight": 0.35,
                "top_industry": "bank",
                "blocked_buy_count": 0,
                "partial_buy_count": 0,
                "unfilled_buy_value": 0.0,
            },
        ]
    )
    picks = pd.DataFrame(
        [
            {"signal_date": "2021-01-04", "symbol": "000001", "industry_label": "bank", "weight": 0.2, "score": 0.9},
            {"signal_date": "2021-01-05", "symbol": "000002", "industry_label": "bank", "weight": 0.3, "score": 0.8},
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "side": "buy",
                "status": "partial",
                "reason": "capacity_partial",
                "desired_notional": 1000,
                "filled_notional": 800,
                "unfilled_notional": 200,
            },
            {
                "symbol": "000002",
                "side": "sell",
                "status": "blocked",
                "reason": "limit_down",
                "desired_notional": 500,
                "filled_notional": 0,
                "unfilled_notional": 500,
            },
        ]
    )
    capacity = pd.DataFrame(
        [
            {"initial_cash": 1_000_000, "annual_return": 0.05, "max_drawdown": -0.10, "is_current_setting": True},
            {"initial_cash": 5_000_000, "annual_return": -0.01, "max_drawdown": -0.20, "is_current_setting": False},
        ]
    )

    report = build_formal_attribution_report(metrics=metrics, equity=equity, picks=picks, trades=trades, capacity=capacity)

    assert report["benchmark_gap"]["total_excess_return"] == -0.15
    assert report["cash_exposure"]["avg_cash_weight"] == 0.275
    assert report["trade_friction"]["untradable_signal_ratio"] == 1.0
    assert report["capacity_and_untradable"]["capital_limit_non_negative_worst_case"] == 1_000_000.0
    assert any(item["type"] == "cash_drag" for item in report["findings"])


def test_formal_release_registry_binds_required_artifacts(tmp_path):
    reports = tmp_path / "formal"
    reports.mkdir()
    prefix = reports / "dynamic_rebalance_20210104_20211231_unit"
    (prefix.with_suffix(".metrics.json")).write_text(
        '{"is_formal_valid": true, "data_quality_red_flags": [], "config": {"strategy": "demo"}}\n',
        encoding="utf-8",
    )
    for suffix in (".manifest.json", ".data_quality.json"):
        prefix.with_suffix(suffix).write_text("{}\n", encoding="utf-8")
    for suffix in (".picks.csv", ".trades.csv", ".equity.csv"):
        prefix.with_suffix(suffix).write_text("symbol\n000001\n", encoding="utf-8")
    for suffix in (".factor_report.json", ".performance_report.json", ".performance_report.md"):
        prefix.with_suffix(suffix).write_text("{}\n", encoding="utf-8")

    registry = build_formal_release_registry(reports)
    release = registry["releases"][0]

    assert registry["release_count"] == 1
    assert release["release_id"].startswith("rel_20210104_20211231_")
    assert release["is_publishable"] is True
    assert sorted(release["artifacts"]) == [
        "data_quality",
        "equity",
        "factor_report",
        "manifest",
        "metrics",
        "performance_markdown",
        "performance_report",
        "picks",
        "trades",
    ]


if __name__ == "__main__":
    test_performance_breakdown_and_benchmark_metrics_are_report_ready()
    test_risk_budget_summarizes_report_controls_and_industry_exposure()
    test_formal_risk_defaults_are_shared_for_config_and_cli_args()
    test_formal_metadata_helpers_parse_required_fields()
    test_formal_attribution_report_explains_cash_friction_and_capacity()
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_formal_release_registry_binds_required_artifacts(Path(directory))

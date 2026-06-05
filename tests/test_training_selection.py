from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import backtest_dynamic_rebalance as dynamic  # noqa: E402


def _row(**overrides: float) -> dict[str, float]:
    row = {
        "annual_return": 0.25,
        "max_drawdown": -0.30,
        "active_period_rate": 0.45,
        "positive_period_rate": 0.48,
        "trade_period_rate": 0.35,
        "period_return_std": 0.035,
    }
    row.update(overrides)
    return row


def test_return40_profile_prefers_candidates_above_target_return():
    below_target = dynamic.training_score(_row(annual_return=0.35), "return40")
    above_target = dynamic.training_score(_row(annual_return=0.45), "return40")

    assert above_target > below_target


def test_return40_profile_rejects_untradable_or_extreme_candidates():
    assert dynamic.training_score(_row(active_period_rate=0.07), "return40") == -math.inf
    assert dynamic.training_score(_row(max_drawdown=-0.91), "return40") == -math.inf
    assert dynamic.training_score(_row(trade_period_rate=0.96), "return40") == -math.inf


def test_stable40_profile_requires_internal_train_stability():
    stable = _row(
        annual_return=0.30,
        max_drawdown=-0.35,
        subperiod_count=2,
        subperiod_min_annual_return=0.08,
        subperiod_worst_drawdown=-0.40,
        subperiod_min_positive_period_rate=0.44,
    )
    unstable = dict(stable)
    unstable["subperiod_worst_drawdown"] = -0.82

    assert math.isfinite(dynamic.training_score(stable, "stable40"))
    assert dynamic.training_score(unstable, "stable40") == -math.inf


def test_stable40q_profile_requires_quartered_train_stability():
    stable = _row(
        annual_return=0.34,
        max_drawdown=-0.32,
        positive_period_rate=0.49,
        trade_period_rate=0.30,
        subperiod_count=4,
        subperiod_min_annual_return=0.04,
        subperiod_worst_drawdown=-0.42,
        subperiod_min_positive_period_rate=0.43,
        subperiod_max_trade_period_rate=0.45,
    )
    unstable = dict(stable)
    unstable["subperiod_worst_drawdown"] = -0.70

    assert math.isfinite(dynamic.training_score(stable, "stable40q"))
    assert dynamic.training_score(unstable, "stable40q") == -math.inf


def test_stable40y_profile_requires_calendar_year_stability():
    stable = _row(
        annual_return=0.32,
        max_drawdown=-0.34,
        positive_period_rate=0.49,
        trade_period_rate=0.35,
        year_count=10,
        year_positive_rate=0.70,
        year_min_annual_return=-0.10,
        year_median_annual_return=0.12,
        year_max_annual_return=0.55,
        year_worst_drawdown=-0.38,
        year_max_trade_period_rate=0.45,
    )
    unstable = dict(stable)
    unstable["year_min_annual_return"] = -0.40
    low_return = dict(stable)
    low_return["annual_return"] = 0.20

    assert math.isfinite(dynamic.training_score(stable, "stable40y"))
    assert dynamic.training_score(unstable, "stable40y") == -math.inf
    assert dynamic.training_score(low_return, "stable40y") == -math.inf


def test_durable40_profile_requires_train_subperiod_durability():
    stable = _row(
        annual_return=0.30,
        max_drawdown=-0.34,
        active_period_rate=0.98,
        positive_period_rate=0.49,
        trade_period_rate=0.18,
        period_return_std=0.018,
        subperiod_count=2,
        subperiod_min_annual_return=0.16,
        subperiod_max_annual_return=0.42,
        subperiod_worst_drawdown=-0.38,
        subperiod_min_positive_period_rate=0.49,
        subperiod_max_trade_period_rate=0.18,
    )
    weak_half = dict(stable)
    weak_half["subperiod_min_annual_return"] = 0.03
    high_turnover = dict(stable)
    high_turnover["subperiod_max_trade_period_rate"] = 0.60

    assert math.isfinite(dynamic.training_score(stable, "durable40"))
    assert dynamic.training_score(weak_half, "durable40") == -math.inf
    assert dynamic.training_score(high_turnover, "durable40") == -math.inf


def test_recent40_profile_requires_recent_train_strength():
    stable = _row(
        annual_return=0.30,
        max_drawdown=-0.34,
        active_period_rate=0.98,
        positive_period_rate=0.49,
        trade_period_rate=0.18,
        period_return_std=0.018,
        subperiod_count=2,
        subperiod_first_annual_return=0.18,
        subperiod_last_annual_return=0.36,
        subperiod_min_annual_return=0.18,
        subperiod_last_drawdown=-0.30,
        subperiod_min_positive_period_rate=0.49,
        subperiod_max_trade_period_rate=0.18,
    )
    weak_recent = dict(stable)
    weak_recent["subperiod_last_annual_return"] = 0.08
    weak_early = dict(stable)
    weak_early["subperiod_first_annual_return"] = -0.02

    assert math.isfinite(dynamic.training_score(stable, "recent40"))
    assert dynamic.training_score(weak_recent, "recent40") == -math.inf
    assert dynamic.training_score(weak_early, "recent40") == -math.inf


def test_holdout40_profile_requires_four_part_train_holdout_strength():
    stable = _row(
        annual_return=0.30,
        max_drawdown=-0.34,
        active_period_rate=0.90,
        positive_period_rate=0.48,
        trade_period_rate=0.18,
        period_return_std=0.018,
        subperiod_count=4,
        subperiod_last_annual_return=0.28,
        subperiod_min_annual_return=0.02,
        subperiod_median_annual_return=0.14,
        subperiod_worst_drawdown=-0.38,
        subperiod_last_drawdown=-0.30,
        subperiod_min_positive_period_rate=0.44,
        subperiod_max_trade_period_rate=0.18,
    )
    weak_holdout = dict(stable)
    weak_holdout["subperiod_last_annual_return"] = 0.10
    weak_median = dict(stable)
    weak_median["subperiod_median_annual_return"] = 0.04

    assert math.isfinite(dynamic.training_score(stable, "holdout40"))
    assert dynamic.training_score(weak_holdout, "holdout40") == -math.inf
    assert dynamic.training_score(weak_median, "holdout40") == -math.inf


def test_regime40_profile_allows_lower_active_train_regimes():
    stable = _row(
        annual_return=0.18,
        max_drawdown=-0.24,
        active_period_rate=0.35,
        positive_period_rate=0.46,
        trade_period_rate=0.14,
        period_return_std=0.018,
        subperiod_count=4,
        subperiod_last_annual_return=0.18,
        subperiod_min_annual_return=-0.02,
        subperiod_median_annual_return=0.09,
        subperiod_worst_drawdown=-0.25,
        subperiod_last_drawdown=-0.20,
        subperiod_min_positive_period_rate=0.43,
        subperiod_max_trade_period_rate=0.18,
    )
    weak_recent = dict(stable)
    weak_recent["subperiod_last_annual_return"] = 0.02
    high_drawdown = dict(stable)
    high_drawdown["subperiod_worst_drawdown"] = -0.50

    assert math.isfinite(dynamic.training_score(stable, "regime40"))
    assert dynamic.training_score(weak_recent, "regime40") == -math.inf
    assert dynamic.training_score(high_drawdown, "regime40") == -math.inf


def test_new_price_volume_scope_includes_scan_ready_float_cap_formulas():
    formulas = dynamic.selected_formulas("expanded", "new_price_volume")
    names = {formula.name for formula in formulas}
    assert "small_cap_pullback_quality" in names
    assert "small_cap_dryup_reacceleration" in names
    assert "float_cap_repair" in names
    assert "steady_trend_low_noise" in names
    assert "small_float_steady_trend" in names
    assert "dryup_trend_quality" in names
    assert "industry_relative_steady_reversal" in names
    assert "quality_dryup_trend" in names
    assert "industry_low_noise_momentum" in names
    assert "capital_light_reacceleration" in names

    rank_names = {
        rank_name
        for formula in formulas
        for rank_name in formula.weights
        if rank_name not in dynamic.INDUSTRY_RELATIVE_FEATURES
    }
    assert rank_names <= set(dynamic.RANK_MAP)


def test_market_valuation_states_use_rolling_history_only():
    dates = pd.date_range("2020-01-01", periods=260, freq="D")
    pe_values = [10.0 + (index % 20) for index in range(260)]
    pb_values = [1.0 + (index % 20) / 10 for index in range(260)]
    pe_values[253] = 80.0
    pb_values[253] = 8.0
    pe_values[259] = 8.0
    pb_values[259] = 0.8
    valuation = pd.DataFrame(
        {
            "trade_date": dates,
            "middle_pe_ttm": pe_values,
            "middle_pb": pb_values,
        }
    )
    pe_pct = dynamic._rolling_last_percentile(pd.Series([10.0, 11.0, 12.0, 50.0, 9.0, 8.0]), window=3, min_periods=3)
    states = dynamic.market_valuation_states(valuation)

    assert pe_pct.iloc[2] == 1.0
    assert pe_pct.iloc[4] == 1 / 3
    assert dates[253] not in states["valuation_not_high"]
    assert dates[259] in states["valuation_low"]


def test_credible_grid_includes_market_valuation_filters():
    values = dynamic.grid_values("credible")

    assert "valuation_not_high" in values["market_filters"]
    assert "pe_ttm_low" in values["market_filters"]


def test_combined_market_states_are_intersections():
    market = {
        "risk_on": {pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")},
        "valuation_not_high": {pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")},
        "sw_breadth_ret21_50": {pd.Timestamp("2020-01-01")},
    }

    states = dynamic.add_combined_market_states(market)

    assert states["risk_on_valuation_not_high"] == {pd.Timestamp("2020-01-02")}
    assert states["risk_on_sw_breadth_ret21_50"] == {pd.Timestamp("2020-01-01")}


def test_regime_grid_requires_explicit_risk_state():
    values = dynamic.grid_values("regime")

    assert "none" not in values["market_filters"]
    assert "risk_on_valuation_not_high" in values["market_filters"]
    assert values["min_amounts"] == [50_000_000]
    assert values["min_prices"] == [10.0]


def test_symbol_valuation_features_use_backward_asof_only():
    bars = pd.DataFrame(
        {
            "symbol": ["600519", "600519", "600519", "000001"],
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-10", "2020-01-20", "2020-01-10"]),
        }
    )
    valuation = pd.DataFrame(
        {
            "symbol": ["600519", "600519"],
            "trade_date": pd.to_datetime(["2020-01-05", "2020-01-15"]),
            "pe_ttm": [10.0, 20.0],
            "pb": [1.0, 2.0],
            "total_market_cap": [100.0, 200.0],
        }
    )

    out = dynamic.add_symbol_valuation_features(bars, valuation, ["low_pe_ttm_r", "low_pb_r"])

    assert pd.isna(out.loc[0, "pe_ttm_asof"])
    assert out.loc[1, "pe_ttm_asof"] == 10.0
    assert out.loc[2, "pe_ttm_asof"] == 20.0
    assert pd.isna(out.loc[3, "pe_ttm_asof"])


def test_valuation_formula_scope_exposes_value_factors():
    formulas = dynamic.selected_formulas("expanded", "valuation")
    names = {formula.name for formula in formulas}
    rank_names = {rank for formula in formulas for rank in formula.weights}

    assert names == {"valuation_low_pe_pb", "valuation_quality_pullback", "small_value_reacceleration"}
    assert {"low_pe_ttm_r", "low_pb_r", "small_mcap_value_r"} <= rank_names


def test_symbol_financial_features_use_notice_date_backward_asof_only():
    bars = pd.DataFrame(
        {
            "symbol": ["600519", "600519", "600519", "000001"],
            "trade_date": pd.to_datetime(["2021-04-19", "2021-04-20", "2021-05-01", "2021-04-20"]),
            "close": [1.0, 1.0, 1.0, 1.0],
        }
    )
    financials = pd.DataFrame(
        {
            "symbol": ["600519", "600519", "000001"],
            "report_date": pd.to_datetime(["2020-12-31", "2021-03-31", "2020-12-31"]),
            "notice_date": pd.to_datetime(["2021-04-20", "2021-04-30", "2021-04-21"]),
            "roe": [10.0, 20.0, 5.0],
            "roic": [8.0, 9.0, 4.0],
            "gross_margin": [30.0, 31.0, 20.0],
            "net_margin": [12.0, 13.0, 8.0],
            "asset_return": [5.0, 6.0, 3.0],
            "debt_asset_ratio": [40.0, 41.0, 50.0],
            "revenue_growth_yoy": [15.0, 16.0, 7.0],
            "profit_growth_yoy": [18.0, 19.0, 6.0],
            "deduct_profit_growth_yoy": [17.0, 18.0, 5.0],
            "operating_cashflow_to_revenue": [0.8, 0.9, 0.4],
        }
    )

    out = dynamic.add_symbol_financial_features(bars, financials, ["quality_roe_r", "quality_roic_r"])

    assert pd.isna(out.loc[0, "roe_asof"])
    assert out.loc[1, "roe_asof"] == 10.0
    assert out.loc[2, "roe_asof"] == 20.0
    assert pd.isna(out.loc[3, "roe_asof"])


def test_quality_formula_scope_exposes_financial_quality_factors():
    formulas = dynamic.selected_formulas("expanded", "quality")
    names = {formula.name for formula in formulas}
    rank_names = {rank for formula in formulas for rank in formula.weights}

    assert names == {
        "quality_value_compounder",
        "cashflow_quality_pullback",
        "quality_growth_reacceleration",
        "cashflow_defensive_quality",
        "asset_light_quality_trend",
        "profit_growth_low_noise",
    }
    assert {"quality_roe_r", "quality_roic_r", "cashflow_to_revenue_r", "low_debt_r"} <= rank_names


def test_fixed_spec_parser_resolves_quality_scope_formula_without_inline_weights():
    spec = dynamic.spec_from_config(
        {
            "formula": "cashflow_quality_pullback",
            "market_filter": "sw_top_mom_63_pos",
            "top_n": 1,
            "min_amount": 50_000_000,
            "min_price": 3.0,
            "trend_filter": "none",
            "min_hold_days": 2,
            "max_hold_days": 10,
            "replace_count": 1,
            "stop_loss": None,
        }
    )

    assert spec.formula.name == "cashflow_quality_pullback"
    assert "cashflow_to_revenue_r" in spec.formula.weights


def test_strict_window_metrics_replays_from_window_start():
    calls = []
    spec = dynamic.DynamicSpec(
        formula=dynamic.Formula("demo", {"mom_21_r": 1.0}),
        market_filter="none",
        top_n=1,
        min_amount=0.0,
        min_price=0.0,
        trend_filter="none",
        min_hold_days=1,
        max_hold_days=1,
        replace_count=1,
        stop_loss=None,
    )
    days = [
        SimpleNamespace(signal_date=pd.Timestamp("2017-12-29"), entry_date=pd.Timestamp("2018-01-02")),
        SimpleNamespace(signal_date=pd.Timestamp("2018-01-02"), entry_date=pd.Timestamp("2018-01-03")),
        SimpleNamespace(signal_date=pd.Timestamp("2018-01-03"), entry_date=pd.Timestamp("2018-01-04")),
    ]

    def fake_simulate_dynamic(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return (
            np.asarray([0.10, -0.05], dtype=np.float32),
            np.asarray([True, True], dtype=bool),
            np.asarray([True, False], dtype=bool),
            [],
            [],
            [],
        )

    original = dynamic.simulate_dynamic
    try:
        dynamic.simulate_dynamic = fake_simulate_dynamic
        metrics = dynamic.strict_spec_window_metrics(
            spec,
            days,
            {},
            pd.Timestamp("2018-01-02"),
            pd.Timestamp("2018-01-03"),
        )
    finally:
        dynamic.simulate_dynamic = original

    assert calls[0]["kwargs"]["start_date"] == pd.Timestamp("2018-01-02")
    assert calls[0]["kwargs"]["end_date"] == pd.Timestamp("2018-01-03")
    assert metrics["periods"] == 2
    assert math.isclose(metrics["total_return"], 1.10 * 0.95 - 1.0, rel_tol=1e-6)


if __name__ == "__main__":
    test_return40_profile_prefers_candidates_above_target_return()
    test_return40_profile_rejects_untradable_or_extreme_candidates()
    test_stable40_profile_requires_internal_train_stability()
    test_stable40q_profile_requires_quartered_train_stability()
    test_stable40y_profile_requires_calendar_year_stability()
    test_durable40_profile_requires_train_subperiod_durability()
    test_recent40_profile_requires_recent_train_strength()
    test_holdout40_profile_requires_four_part_train_holdout_strength()
    test_regime40_profile_allows_lower_active_train_regimes()
    test_new_price_volume_scope_includes_scan_ready_float_cap_formulas()
    test_market_valuation_states_use_rolling_history_only()
    test_credible_grid_includes_market_valuation_filters()
    test_combined_market_states_are_intersections()
    test_regime_grid_requires_explicit_risk_state()
    test_symbol_valuation_features_use_backward_asof_only()
    test_valuation_formula_scope_exposes_value_factors()
    test_symbol_financial_features_use_notice_date_backward_asof_only()
    test_quality_formula_scope_exposes_financial_quality_factors()
    test_fixed_spec_parser_resolves_quality_scope_formula_without_inline_weights()
    test_strict_window_metrics_replays_from_window_start()

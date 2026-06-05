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


def test_new_price_volume_scope_includes_scan_ready_float_cap_formulas():
    formulas = dynamic.selected_formulas("expanded", "new_price_volume")
    names = {formula.name for formula in formulas}
    assert "small_cap_pullback_quality" in names
    assert "small_cap_dryup_reacceleration" in names
    assert "float_cap_repair" in names

    rank_names = {
        rank_name
        for formula in formulas
        for rank_name in formula.weights
        if rank_name not in dynamic.INDUSTRY_RELATIVE_FEATURES
    }
    assert rank_names <= set(dynamic.RANK_MAP)


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
    test_new_price_volume_scope_includes_scan_ready_float_cap_formulas()
    test_strict_window_metrics_replays_from_window_start()

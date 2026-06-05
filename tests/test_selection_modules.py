import math
import sys
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.backtest.selection import (  # noqa: E402
    choose_yearly_specs_from_series,
    metrics_from_returns,
    spec_to_row,
    training_calendar_year_metrics,
    training_subperiod_metrics,
    training_score,
)


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


def test_selection_helpers_compute_metrics_score_and_spec_rows():
    metrics = metrics_from_returns(
        np.asarray([0.10, -0.05, 0.02], dtype=np.float32),
        np.asarray([True, True, False]),
        np.asarray([True, False, False]),
        np.asarray(["2021-01-04", "2021-01-05", "2021-01-06"], dtype="datetime64[ns]"),
        np.asarray([True, True, False]),
    )
    spec = SimpleNamespace(
        formula=SimpleNamespace(name="demo", weights={"mom": 1.0}),
        market_filter="none",
        top_n=3,
        min_amount=50_000_000,
        min_price=10.0,
        trend_filter="ma126",
        min_hold_days=5,
        max_hold_days=10,
        replace_count=1,
        stop_loss=0.1,
    )
    row = spec_to_row(spec)

    assert metrics["periods"] == 2
    assert math.isclose(metrics["total_return"], 1.10 * 0.95 - 1.0, rel_tol=1e-6)
    assert training_score(_row(annual_return=0.45), "return40") > training_score(_row(annual_return=0.35), "return40")
    assert training_score(_row(active_period_rate=0.07), "return40") == -math.inf
    assert row["formula"] == "demo"
    assert row["weights"] == '{"mom": 1.0}'


def test_stable40_profile_penalizes_bad_training_subperiods():
    stable = _row(
        annual_return=0.30,
        max_drawdown=-0.35,
        positive_period_rate=0.48,
        subperiod_count=2,
        subperiod_min_annual_return=0.08,
        subperiod_worst_drawdown=-0.40,
        subperiod_min_positive_period_rate=0.44,
    )
    unstable = dict(stable)
    unstable["subperiod_min_annual_return"] = -0.10

    assert math.isfinite(training_score(stable, "stable40"))
    assert training_score(unstable, "stable40") == -math.inf


def test_stable40q_profile_requires_four_part_training_stability():
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
    unstable["subperiod_min_annual_return"] = -0.05

    assert math.isfinite(training_score(stable, "stable40q"))
    assert training_score(unstable, "stable40q") == -math.inf


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
    unstable["year_positive_rate"] = 0.50
    low_return = dict(stable)
    low_return["annual_return"] = 0.20

    assert math.isfinite(training_score(stable, "stable40y"))
    assert training_score(unstable, "stable40y") == -math.inf
    assert training_score(low_return, "stable40y") == -math.inf


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

    assert math.isfinite(training_score(stable, "durable40"))
    assert training_score(weak_half, "durable40") == -math.inf
    assert training_score(high_turnover, "durable40") == -math.inf


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

    assert math.isfinite(training_score(stable, "recent40"))
    assert training_score(weak_recent, "recent40") == -math.inf
    assert training_score(weak_early, "recent40") == -math.inf


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

    assert math.isfinite(training_score(stable, "holdout40"))
    assert training_score(weak_holdout, "holdout40") == -math.inf
    assert training_score(weak_median, "holdout40") == -math.inf


def test_training_subperiod_metrics_summarize_halves():
    metrics = training_subperiod_metrics(
        np.asarray([0.02, 0.02, -0.02, -0.02], dtype=np.float32),
        np.asarray([True, True, True, True]),
        np.asarray([True, False, True, False]),
        np.asarray(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[ns]"),
        np.asarray([True, True, True, True]),
    )

    assert metrics["subperiod_count"] == 2
    assert metrics["subperiod_first_annual_return"] > 0
    assert metrics["subperiod_last_annual_return"] < 0
    assert metrics["subperiod_min_annual_return"] < 0
    assert math.isclose(
        metrics["subperiod_median_annual_return"],
        (metrics["subperiod_first_annual_return"] + metrics["subperiod_last_annual_return"]) / 2,
    )
    assert metrics["subperiod_worst_drawdown"] < 0


def test_training_calendar_year_metrics_summarize_years():
    metrics = training_calendar_year_metrics(
        np.asarray([0.02, 0.02, -0.02, -0.02], dtype=np.float32),
        np.asarray([True, True, True, True]),
        np.asarray([True, False, True, False]),
        np.asarray(["2020-01-02", "2020-01-03", "2021-01-06", "2021-01-07"], dtype="datetime64[ns]"),
        np.asarray([True, True, True, True]),
    )

    assert metrics["year_count"] == 2
    assert metrics["year_positive_count"] == 1
    assert metrics["year_min_annual_return"] < 0


@dataclass
class UnitSeries:
    spec: object
    returns: np.ndarray
    active: np.ndarray
    trades: np.ndarray


def test_choose_yearly_specs_from_series_freezes_after_selection_date():
    spec_a = SimpleNamespace(
        formula=SimpleNamespace(name="a", weights={"mom": 1.0}),
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
    spec_b = SimpleNamespace(
        formula=SimpleNamespace(name="b", weights={"mom": 1.0}),
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
    signal_dates = np.asarray(["2019-01-02", "2019-01-03", "2020-01-02", "2021-01-04"], dtype="datetime64[ns]")
    entry_dates = np.asarray(["2019-01-03", "2019-01-04", "2020-01-03", "2021-01-05"], dtype="datetime64[ns]")
    yearly, diagnostics = choose_yearly_specs_from_series(
        series_list=[
            UnitSeries(
                spec=spec_a,
                returns=np.asarray([0.02, 0.02, 0.01, 0.01], dtype=np.float32),
                active=np.asarray([True, True, True, True]),
                trades=np.asarray([True, False, False, False]),
            ),
            UnitSeries(
                spec=spec_b,
                returns=np.asarray([0.01, 0.01, 0.03, 0.03], dtype=np.float32),
                active=np.asarray([True, True, True, True]),
                trades=np.asarray([True, False, False, False]),
            ),
        ],
        signal_dates=signal_dates,
        entry_dates=entry_dates,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2021-12-31"),
        train_years=1,
        min_train_periods=2,
        keep_top=2,
        score_profile="aggressive",
        freeze_selection_date=pd.Timestamp("2019-12-31"),
        validation_metrics_func=lambda spec: {
            "annual_return": 0.2,
            "max_drawdown": -0.1,
            "active_period_rate": 1.0,
            "positive_period_rate": 1.0,
            "trade_period_rate": 0.1,
            "period_return_std": 0.01,
        },
    )

    assert yearly[2020] is spec_a
    assert yearly[2021] is spec_a
    assert "frozen_selected" in set(diagnostics["status"])
    assert "selected_frozen" in set(diagnostics["status"])
    assert "validation_annual_return" in diagnostics.columns


def test_choose_yearly_specs_does_not_fallback_after_failed_freeze():
    spec = SimpleNamespace(
        formula=SimpleNamespace(name="demo", weights={"mom": 1.0}),
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
    signal_dates = np.asarray(["2019-01-02", "2019-01-03", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    entry_dates = np.asarray(["2019-01-03", "2019-01-04", "2020-01-03", "2020-01-06"], dtype="datetime64[ns]")

    yearly, diagnostics = choose_yearly_specs_from_series(
        series_list=[
            UnitSeries(
                spec=spec,
                returns=np.asarray([0.02, 0.02, 0.03, 0.03], dtype=np.float32),
                active=np.asarray([True, True, True, True]),
                trades=np.asarray([True, False, True, False]),
            ),
        ],
        signal_dates=signal_dates,
        entry_dates=entry_dates,
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
        train_years=1,
        min_train_periods=3,
        keep_top=1,
        score_profile="aggressive",
        freeze_selection_date=pd.Timestamp("2019-12-31"),
    )

    assert yearly == {}
    assert "frozen_selection_skipped_insufficient_training_periods" in set(diagnostics["status"])
    assert "skipped_frozen_selection_unavailable" in set(diagnostics["status"])
    assert "selected" not in set(diagnostics["status"])


if __name__ == "__main__":
    test_selection_helpers_compute_metrics_score_and_spec_rows()
    test_stable40_profile_penalizes_bad_training_subperiods()
    test_stable40q_profile_requires_four_part_training_stability()
    test_stable40y_profile_requires_calendar_year_stability()
    test_durable40_profile_requires_train_subperiod_durability()
    test_recent40_profile_requires_recent_train_strength()
    test_holdout40_profile_requires_four_part_train_holdout_strength()
    test_training_subperiod_metrics_summarize_halves()
    test_training_calendar_year_metrics_summarize_years()
    test_choose_yearly_specs_from_series_freezes_after_selection_date()
    test_choose_yearly_specs_does_not_fallback_after_failed_freeze()

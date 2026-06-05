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


def test_training_subperiod_metrics_summarize_halves():
    metrics = training_subperiod_metrics(
        np.asarray([0.02, 0.02, -0.02, -0.02], dtype=np.float32),
        np.asarray([True, True, True, True]),
        np.asarray([True, False, True, False]),
        np.asarray(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"], dtype="datetime64[ns]"),
        np.asarray([True, True, True, True]),
    )

    assert metrics["subperiod_count"] == 2
    assert metrics["subperiod_min_annual_return"] < 0
    assert metrics["subperiod_worst_drawdown"] < 0


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


if __name__ == "__main__":
    test_selection_helpers_compute_metrics_score_and_spec_rows()
    test_stable40_profile_penalizes_bad_training_subperiods()
    test_stable40q_profile_requires_four_part_training_stability()
    test_training_subperiod_metrics_summarize_halves()
    test_choose_yearly_specs_from_series_freezes_after_selection_date()

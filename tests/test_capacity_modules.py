import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.backtest.capacity import (  # noqa: E402
    build_capacity_stress_row,
    capacity_stress_plan,
    mark_capacity_stress_replayed,
)


def test_capacity_stress_helpers_build_plan_row_and_replay_metadata():
    plan = capacity_stress_plan(
        SimpleNamespace(
            initial_cash=1_000_000.0,
            capacity_pct_of_amount=0.02,
            slippage_bps=5.0,
            impact_bps_per_pct_amount=2.0,
            capacity_equity_mode="initial",
        )
    )
    row = build_capacity_stress_row(
        initial_cash=1_000_000.0,
        capacity_pct=0.02,
        slippage_bps=5.0,
        impact_bps=2.0,
        capacity_equity_mode="initial",
        current=plan["current"],
        stress_metrics={"periods": 2, "annual_return": 0.12, "max_drawdown": -0.08},
        picks=pd.DataFrame([{"symbol": "000001"}]),
        trade_log=pd.DataFrame([{"symbol": "000001"}, {"symbol": "000002"}]),
    )
    meta = mark_capacity_stress_replayed(plan, pd.DataFrame([row]))

    assert plan["recommended_grid"]["initial_cash"] == [1_000_000.0, 5_000_000.0, 10_000_000.0]
    assert row["is_current_setting"] is True
    assert row["pick_count"] == 1
    assert row["trade_log_rows"] == 2
    assert meta["status"] == "grid_replayed"
    assert meta["rows"] == 1


if __name__ == "__main__":
    test_capacity_stress_helpers_build_plan_row_and_replay_metadata()

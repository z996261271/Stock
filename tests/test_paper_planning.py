import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.paper.planning import build_paper_plan  # noqa: E402


def test_paper_plan_preserves_strategy_version_and_tags_in_payloads():
    plan = build_paper_plan(
        raw_rows=[
            {
                "symbol": "000001",
                "action": "buy",
                "weight": 0.4,
                "score": 0.9,
                "entry_open": 10.03,
                "formula": "low_turnover",
                "strategy_version": "v1",
                "entry_tag": "rebalance_buy",
                "risk_tag": "capacity_ok",
            },
            {
                "symbol": "000002",
                "action": "sell",
                "price": 8.0,
                "quantity": 100.0,
                "exit_tag": "target_removed",
            },
        ],
        strategy="demo",
        signal_date="2026-06-01",
        entry_date="2026-06-02",
        cash=1_000_000.0,
        signal_id_fn=lambda strategy, symbol, signal_date, action: f"sig_{symbol}_{action}",
    )

    buy_signal = plan["signals"][0]
    buy_trade = plan["paper_trades"][0]
    sell_trade = plan["paper_trades"][1]

    assert plan["object_counts"] == {"signals": 2, "order_intents": 2, "position_snapshots": 1}
    assert buy_signal["payload"]["strategy_version"] == "v1"
    assert buy_signal["payload"]["entry_tag"] == "rebalance_buy"
    assert buy_signal["payload"]["source_factor_set"] == "low_turnover"
    assert buy_trade["quantity"] == 39_800.0
    assert buy_trade["amount"] == 399_194.0
    assert buy_trade["payload"]["entry_tag"] == "rebalance_buy"
    assert buy_trade["payload"]["risk_tag"] == "capacity_ok"
    assert sell_trade["quantity"] == 100.0
    assert sell_trade["payload"]["exit_tag"] == "target_removed"
    assert plan["execution_precision"]["lot_size"] == 100


if __name__ == "__main__":
    test_paper_plan_preserves_strategy_version_and_tags_in_payloads()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.execution.models import OrderDecision, OrderIntent  # noqa: E402


def test_order_decision_converts_intent_to_planned_or_blocked_trade_record():
    intent = OrderIntent(
        strategy="demo",
        symbol="000001",
        signal_id="sig-1",
        signal_date="2026-06-01",
        entry_date="2026-06-02",
        side="buy",
        quantity=100.0,
        price=10.0,
        amount=1_000.0,
        reason="rebalance",
        target_weight=0.1,
        strategy_version="v1",
        entry_tag="top_score",
        risk_tag="limit_check",
    )

    planned = OrderDecision.from_intent(intent).to_trade_record()
    blocked = OrderDecision.from_intent(
        intent,
        accepted=False,
        blocked_reason="limit_up",
        payload={"limit_price": 10.5},
    ).to_trade_record()

    assert planned["status"] == "planned"
    assert planned["payload"]["entry_tag"] == "top_score"
    assert planned["payload"]["target_weight"] == 0.1
    assert "blocked_reason" not in planned["payload"]
    assert blocked["status"] == "blocked"
    assert blocked["payload"]["blocked_reason"] == "limit_up"
    assert blocked["payload"]["limit_price"] == 10.5


if __name__ == "__main__":
    test_order_decision_converts_intent_to_planned_or_blocked_trade_record()

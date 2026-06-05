"""Execution simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    buy_cost: float
    sell_cost: float
    slippage_bps: float
    impact_bps_per_pct_amount: float
    capacity_pct_of_amount: float
    capacity_equity_mode: str
    lot_size: int
    limit_epsilon: float
    block_limit_trades: bool


def default_execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        buy_cost=0.0003,
        sell_cost=0.0008,
        slippage_bps=5.0,
        impact_bps_per_pct_amount=2.0,
        capacity_pct_of_amount=0.02,
        capacity_equity_mode="compound",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )

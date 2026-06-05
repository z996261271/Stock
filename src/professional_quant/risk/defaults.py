"""Shared formal risk defaults and config helpers."""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
from typing import Any


FORMAL_RISK_DEFAULTS = {
    "portfolio_stop_loss": 0.25,
    "max_position_weight": 0.20,
    "max_industry_weight": 0.35,
    "max_turnover_pct": 0.80,
}


def apply_formal_risk_defaults_to_mapping(config: MutableMapping[str, Any]) -> None:
    """Fill missing or zero-valued formal risk controls in a config mapping."""
    for key, value in FORMAL_RISK_DEFAULTS.items():
        if float(config.get(key, 0.0) or 0.0) == 0.0:
            config[key] = value


def apply_formal_risk_defaults_to_namespace(args: argparse.Namespace) -> None:
    """Fill missing or zero-valued formal risk controls on parsed CLI args."""
    for key, value in FORMAL_RISK_DEFAULTS.items():
        if float(getattr(args, key, 0.0) or 0.0) == 0.0:
            setattr(args, key, value)

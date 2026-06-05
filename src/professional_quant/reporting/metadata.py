"""Formal report metadata parsing helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


FORMAL_ADJUSTS = {"raw", "qfq", "hfq"}


def parse_required_adjusts(value: str) -> tuple[str, ...]:
    """Parse and validate the comma-separated adjust streams required for formal reports."""
    adjusts = tuple(item.strip() for item in value.split(",") if item.strip())
    if not adjusts:
        raise ValueError("formal required adjusts must not be empty")
    invalid = sorted(set(adjusts).difference(FORMAL_ADJUSTS))
    if invalid:
        raise ValueError(f"unknown required adjusts: {invalid}")
    return adjusts


def parse_json_metadata(value: str | None, name: str) -> dict[str, Any]:
    """Parse a JSON object supplied as CLI metadata."""
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a JSON object")
    return data


def default_split_policy(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freeze_selection_date: pd.Timestamp | None,
) -> dict[str, Any]:
    """Build the default walk-forward split policy when a run is not formal."""
    return {
        "name": "rolling_walkforward",
        "current_result_segment": "walkforward",
        "report_start": start_date.strftime("%Y-%m-%d"),
        "report_end": end_date.strftime("%Y-%m-%d"),
        "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d") if freeze_selection_date is not None else None,
        "rule": "calendar-year test rows use prior completed training years",
    }

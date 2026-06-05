"""Backtest governance and multiple-testing diagnostics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def multiple_testing_summary(
    specs: list[Any],
    diagnostics: pd.DataFrame,
    formula_set: str,
    formula_scope: str,
    grid_profile: str,
    score_profile: str,
) -> dict[str, Any]:
    """Summarize model-selection search breadth for formal report governance."""
    selected_rows = 0
    candidate_rows = 0
    frozen_rows = 0
    if not diagnostics.empty and "status" in diagnostics:
        statuses = diagnostics["status"].astype(str)
        selected_rows = int(statuses.isin(["selected", "selected_frozen", "frozen_selected"]).sum())
        candidate_rows = int(statuses.isin(["train_candidate", "frozen_train_candidate"]).sum())
        frozen_rows = int(statuses.str.startswith("frozen_").sum())
    unique_formulas = sorted({_formula_name(spec) for spec in specs})
    return {
        "specs_evaluated": int(len(specs)),
        "unique_formulas": int(len(unique_formulas)),
        "formula_names": unique_formulas,
        "formula_set": formula_set,
        "formula_scope": formula_scope,
        "grid_profile": grid_profile,
        "score_profile": score_profile,
        "selected_rows": selected_rows,
        "train_candidate_rows_written": candidate_rows,
        "frozen_selection_rows": frozen_rows,
        "risk_note": (
            "More scanned specs increase overfit risk; treat validation/test results as decisive and do not tune "
            "using formal out-of-sample outcomes."
        ),
    }


def _formula_name(spec: Any) -> str:
    formula = getattr(spec, "formula", None)
    return str(getattr(formula, "name", formula))

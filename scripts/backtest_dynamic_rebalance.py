#!/usr/bin/env python3
"""Daily checked, trigger-based dynamic rebalance walk-forward backtest.

No-lookahead rules:
- Factors and rebalance decisions use signal_date close and earlier data only.
- Trades execute at entry_date, the next trading day's open.
- Yearly parameter selection uses only completed prior training years.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_data_quality import adjustment_coverage, build_quality_report, require_factor_adjust_coverage  # noqa: E402
from backtest_walkforward_no_lookahead import (  # noqa: E402
    FACTOR_CACHE_VERSION,
    Formula,
    add_factors,
    formulas,
    load_data,
    market_states,
    parse_date,
    signal_dates,
)
from quant_universe import board_scope_sql  # noqa: E402
from run_manifest import collect_manifest, write_manifest  # noqa: E402
from professional_quant.data.industry import (  # noqa: E402
    apply_industry_labels,
    load_symbol_industry_map,
)
from professional_quant.execution.capacity import capacity_fill_notional_from_arrays  # noqa: E402
from professional_quant.execution.config import ExecutionConfig, default_execution_config  # noqa: E402
from professional_quant.execution.constraints import (  # noqa: E402
    infer_board_label,
    limit_rate_for_symbol,
    locked_limit_masks,
    status_bool,
)
from professional_quant.execution.rebalance import normalize_current_weights, target_rebalance_weight  # noqa: E402
from professional_quant.execution.trades import (  # noqa: E402
    side_cost_fraction_from_amounts,
    trade_block_reason_from_fields,
    trade_event_row,
)
from professional_quant.backtest.reporting import (  # noqa: E402
    compute_equal_weight_benchmark,
    period_return_breakdown,
    professional_performance_metrics,
    relative_performance_metrics,
)
from professional_quant.backtest.walkforward import summarize_walkforward_result  # noqa: E402
from professional_quant.backtest.selection import (  # noqa: E402
    choose_yearly_specs_from_series,
    metrics_from_returns,
    spec_to_row,
    training_score as selection_training_score,
)
from professional_quant.backtest.capacity import (  # noqa: E402
    build_capacity_stress_row,
    capacity_stress_plan,
    mark_capacity_stress_replayed,
)
from professional_quant.backtest.day_data import (  # noqa: E402
    bool_for_symbols,
    open_for_symbols,
    open_to_open_return,
    return_since_entry,
    values_for_symbols,
)
from professional_quant.backtest.governance import multiple_testing_summary  # noqa: E402
from professional_quant.reporting.metadata import (  # noqa: E402
    default_split_policy,
    parse_json_metadata,
    parse_required_adjusts,
)
from professional_quant.risk.budget import risk_budget_report  # noqa: E402
from professional_quant.risk.defaults import apply_formal_risk_defaults_to_namespace  # noqa: E402
from professional_quant.risk.exposure import UNKNOWN_INDUSTRY, industry_exposure_map, max_industry_exposure  # noqa: E402


G_DAYS: list["CompactDayData"] = []
G_MARKET: dict[str, set[pd.Timestamp]] = {}
G_FEATURE_NAMES: list[str] = []
G_TARGETS: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
G_ROUND_TRIP_COST = 0.0011
G_EXECUTION: "ExecutionConfig"
G_INITIAL_CASH = 1_000_000.0
G_PORTFOLIO_STOP_LOSS = 0.0
G_PORTFOLIO_REENTRY_FILTER = "risk_on"
G_MAX_POSITION_WEIGHT = 0.0
G_MAX_INDUSTRY_WEIGHT = 0.0
G_MAX_TURNOVER_PCT = 0.0
G_BLACKLIST: dict[str, list[dict[str, Any]]] = {}
EMPTY_SYMBOLS = np.asarray([], dtype=str)
EMPTY_FLOATS = np.asarray([], dtype=np.float32)
EMPTY_TARGET = (EMPTY_SYMBOLS, EMPTY_FLOATS)

RANK_MAP = {
    "mom_252_21_r": ("mom_252_21", True),
    "mom_126_21_r": ("mom_126_21", True),
    "mom_63_r": ("mom_63", True),
    "mom_21_r": ("mom_21", True),
    "mom_10_r": ("mom_10", True),
    "mom_5_r": ("mom_5", True),
    "rev_20_r": ("rev_20", True),
    "rev_10_r": ("rev_10", True),
    "rev_5_r": ("rev_5", True),
    "near_high_252_r": ("near_high_252", True),
    "near_high_126_r": ("near_high_126", True),
    "near_high_63_r": ("near_high_63", True),
    "lowvol_21_r": ("vol_21", False),
    "lowvol_63_r": ("vol_63", False),
    "highvol_21_r": ("vol_21", True),
    "highvol_63_r": ("vol_63", True),
    "maxret_5_r": ("max_ret_5", True),
    "lowmax_5_r": ("max_ret_5", False),
    "lowmax_21_r": ("max_ret_21", False),
    "panic_21_r": ("panic_21", True),
    "liq_21_r": ("amount_ma_21", True),
    "lowliq_21_r": ("amount_ma_21", False),
    "surge_5_21_r": ("surge_5_21", True),
    "surge_21_63_r": ("surge_21_63", True),
    "trend_63_126_r": ("trend_63_126", True),
    "ma_stack_r": ("ma_stack", True),
    "time_series_r": ("time_series", True),
    "rsrs_18_r": ("rsrs_18", True),
    "range_contract_r": ("range_contract", True),
    "efficiency_21_r": ("efficiency_21", True),
    "money_strength_21_r": ("money_strength_21", True),
    "gap_5_r": ("gap_5", True),
    "amplitude_21_r": ("amplitude_21", True),
    "drawdown_63_r": ("drawdown_63", True),
    "turnover_21_r": ("turnover_ma_21", True),
    "low_turnover_21_r": ("turnover_ma_21", False),
    "turnover_contract_21_63_r": ("turnover_contract_21_63", True),
    "amount_contract_21_63_r": ("amount_contract_21_63", True),
    "vol_contract_21_63_r": ("vol_contract_21_63", True),
    "small_float_cap_21_r": ("float_cap_ma_21", False),
    "cap_contract_21_63_r": ("cap_contract_21_63", True),
    "low_downvol_63_r": ("down_vol_63", False),
    "low_beta_63_r": ("beta_63", False),
    "low_idio_vol_63_r": ("idio_vol_63", False),
    "deep_drawdown_63_r": ("drawdown_63", False),
    "ind_mom_126_21_r": ("ind_mom_126_21", True),
    "ind_rev_5_r": ("ind_rev_5", True),
    "ind_lowvol_63_r": ("ind_lowvol_63", True),
    "ind_lowmax_21_r": ("ind_lowmax_21", True),
    "ind_low_turnover_21_r": ("ind_low_turnover_21", True),
}

INDUSTRY_RELATIVE_FEATURES = {
    "ind_mom_126_21_r": ("ind_mom_126_21", {"mom_126_21"}),
    "ind_rev_5_r": ("ind_rev_5", {"rev_5"}),
    "ind_lowvol_63_r": ("ind_lowvol_63", {"vol_63"}),
    "ind_lowmax_21_r": ("ind_lowmax_21", {"max_ret_21"}),
    "ind_low_turnover_21_r": ("ind_low_turnover_21", {"turnover_ma_21"}),
}


@dataclass(frozen=True)
class DynamicSpec:
    formula: Formula
    market_filter: str
    top_n: int
    min_amount: float
    min_price: float
    trend_filter: str
    min_hold_days: int
    max_hold_days: int
    replace_count: int
    stop_loss: float | None


@dataclass
class CompactDayData:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    symbols: np.ndarray
    entry_open: np.ndarray
    entry_high: np.ndarray
    entry_low: np.ndarray
    entry_amount: np.ndarray
    entry_volume: np.ndarray
    entry_buyable: np.ndarray
    entry_sellable: np.ndarray
    signal_allowed: np.ndarray
    features: np.ndarray
    amount21: np.ndarray
    close: np.ndarray
    trend_masks: dict[str, np.ndarray]
    entry_suspended: np.ndarray | None = None
    entry_limit_rate: np.ndarray | None = None
    industry_labels: np.ndarray | None = None


@dataclass
class DynamicSeries:
    spec: DynamicSpec
    returns: np.ndarray
    active: np.ndarray
    trades: np.ndarray


@dataclass
class ExecutionResult:
    equity: float
    trade_count: int
    blocked_buy_count: int
    blocked_sell_count: int
    partial_buy_count: int
    partial_sell_count: int
    turnover_blocked_count: int
    turnover_value: float
    unfilled_buy_value: float
    unfilled_sell_value: float
    turnover_blocked_value: float
    current_symbols: np.ndarray
    current_prev_open: np.ndarray
    current_entry_open: np.ndarray
    current_weights: np.ndarray
    trade_events: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic daily checked walk-forward strategy scan.")
    parser.add_argument("--db", type=Path, default=Path("data/stock_daily.sqlite3"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--min-train-periods", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--keep-top", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-hold-days", type=int, help="override candidate grid minimum holding days")
    parser.add_argument("--max-hold-days", type=int, help="override candidate grid maximum holding days")
    parser.add_argument(
        "--stop-loss",
        help="override candidate grid stop-loss value; use none/null to disable stop-loss",
    )
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument("--buy-cost", type=float, default=0.0003)
    parser.add_argument("--sell-cost", type=float, default=0.0008)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-bps-per-pct-amount", type=float, default=2.0)
    parser.add_argument("--capacity-pct-of-amount", type=float, default=0.02)
    parser.add_argument("--capacity-equity-mode", choices=["compound", "initial"], default="compound")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--limit-epsilon", type=float, default=0.002)
    parser.add_argument("--no-limit-block", action="store_true")
    parser.add_argument("--portfolio-stop-loss", type=float, default=0.0)
    parser.add_argument("--portfolio-reentry-filter", default="risk_on")
    parser.add_argument(
        "--max-position-weight",
        type=float,
        default=0.0,
        help="optional max single-name portfolio weight; 0 disables the explicit cap",
    )
    parser.add_argument(
        "--max-industry-weight",
        type=float,
        default=0.0,
        help="optional max aggregate weight per industry label; 0 disables the explicit cap",
    )
    parser.add_argument(
        "--max-turnover-pct",
        type=float,
        default=0.0,
        help="optional max one-period turnover as a fraction of current equity; 0 disables the constraint",
    )
    parser.add_argument(
        "--blacklist-file",
        type=Path,
        help="optional CSV/JSON buy blacklist with symbol,start_date,end_date,reason columns",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument(
        "--score-profile",
        choices=["robust", "balanced", "aggressive", "return40", "stable40", "stable40q", "stable40y", "durable40"],
        default="robust",
    )
    parser.add_argument("--formula-set", choices=["base", "expanded"], default="base")
    parser.add_argument(
        "--formula-scope",
        choices=["selected", "all", "new_price_volume"],
        default="selected",
        help="selected keeps the search small around historically stable formulas",
    )
    parser.add_argument(
        "--grid-profile",
        choices=["credible", "wide", "smoke"],
        default="credible",
        help="candidate grid size: credible is the default bounded whitelist; wide keeps the old broad search",
    )
    parser.add_argument(
        "--fixed-spec-json",
        help="JSON object for replaying one mined DynamicSpec instead of scanning the grid",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--board-scope",
        choices=["main", "all"],
        default="main",
        help="stock universe: main keeps Shanghai/Shenzhen main-board prefixes only",
    )
    parser.add_argument(
        "--factor-adjust",
        choices=["raw", "qfq", "hfq"],
        default="hfq",
        help="price adjustment used for factors; execution always uses raw prices",
    )
    parser.add_argument(
        "--strict-factor-adjust",
        action="store_true",
        help="require factor-adjust rows instead of falling back to raw where adjusted rows are missing",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="formal report gate: require non-raw factor-adjust plus --strict-factor-adjust",
    )
    parser.add_argument(
        "--formal-required-adjusts",
        default="raw,qfq,hfq",
        help="comma-separated adjust streams required by the formal data-quality gate",
    )
    parser.add_argument(
        "--split-policy-json",
        help="JSON metadata for fixed train/validation/test split policy",
    )
    parser.add_argument(
        "--frozen-config-json",
        help="JSON metadata proving the formal run uses frozen parameter policy",
    )
    parser.add_argument(
        "--freeze-selection-date",
        help="last signal date allowed for parameter selection before frozen out-of-sample testing",
    )
    parser.add_argument(
        "--industry-source",
        default="sw.industry.index.一级行业",
        help="industry_daily_bars source used for sector market filters; empty disables sector filters",
    )
    parser.add_argument(
        "--skip-capacity-stress",
        action="store_true",
        help="skip the 27-row capacity/slippage replay grid for faster research iterations; not allowed with --formal",
    )
    args = parser.parse_args()
    if args.formal and args.factor_adjust == "raw":
        parser.error("--formal requires --factor-adjust qfq or hfq; raw is only for research/control runs")
    if args.formal and not args.strict_factor_adjust:
        parser.error("--formal requires --strict-factor-adjust")
    if args.formal and args.skip_capacity_stress:
        parser.error("--formal requires capacity stress; do not use --skip-capacity-stress")
    if args.formal:
        apply_formal_risk_defaults_to_namespace(args)
    if args.max_position_weight < 0 or args.max_position_weight > 1:
        parser.error("--max-position-weight must be between 0 and 1")
    if args.max_industry_weight < 0 or args.max_industry_weight > 1:
        parser.error("--max-industry-weight must be between 0 and 1")
    if args.max_turnover_pct < 0:
        parser.error("--max-turnover-pct must be non-negative")
    if args.min_hold_days is not None and args.min_hold_days < 1:
        parser.error("--min-hold-days must be >= 1")
    if args.max_hold_days is not None and args.max_hold_days < 1:
        parser.error("--max-hold-days must be >= 1")
    if args.min_hold_days is not None and args.max_hold_days is not None and args.min_hold_days > args.max_hold_days:
        parser.error("--min-hold-days must be <= --max-hold-days")
    if args.stop_loss is not None:
        stop_loss_text = str(args.stop_loss).strip().lower()
        if stop_loss_text in {"", "none", "null"}:
            args.stop_loss = None
        else:
            try:
                args.stop_loss = float(args.stop_loss)
            except ValueError:
                parser.error("--stop-loss must be a float or none/null")
            if args.stop_loss <= 0:
                parser.error("--stop-loss must be positive when provided")
    return args


G_EXECUTION = default_execution_config()


def selected_formulas(formula_set: str, scope: str) -> list[Formula]:
    pool = formulas(formula_set)
    if scope == "all":
        return pool
    if scope == "new_price_volume":
        selected = {
            "low_beta_pullback_trend",
            "dryup_reacceleration",
            "vol_compression_breakout",
            "small_cap_pullback_quality",
            "small_cap_dryup_reacceleration",
            "float_cap_repair",
            "drawdown_repair_quality",
            "steady_trend_low_noise",
            "small_float_steady_trend",
            "dryup_trend_quality",
            "industry_relative_steady_reversal",
        }
        return [formula for formula in pool if formula.name in selected]
    selected = {
        "low_volume_winner",
        "pullback_in_uptrend",
        "low_turnover_trend",
        "anti_lottery_momentum",
        "volume_breakout",
        "squeeze_breakout",
        "quiet_accumulation",
        "low_turnover_pullback",
        "low_turnover_pullback_calm",
        "low_turnover_pullback_fast",
        "low_turnover_pullback_trend",
        "squeeze_trend_quality",
        "pullback_defensive_trend",
        "industry_neutral_reversal",
        "industry_neutral_lowvol_pullback",
        "industry_neutral_defensive_trend",
        "low_beta_pullback_trend",
        "dryup_reacceleration",
        "vol_compression_breakout",
        "small_cap_pullback_quality",
        "small_cap_dryup_reacceleration",
        "float_cap_repair",
        "drawdown_repair_quality",
        "steady_trend_low_noise",
        "small_float_steady_trend",
        "dryup_trend_quality",
        "industry_relative_steady_reversal",
    }
    return [formula for formula in pool if formula.name in selected]


def grid_values(profile: str) -> dict[str, list]:
    if profile == "wide":
        return {
            "market_filters": [
                "none",
                "ma21",
                "ma63",
                "ma126",
                "ma21_63",
                "ret21_pos",
                "ret63_pos",
                "ret21_63_pos",
                "ma21_ret21",
                "ma63_ret63",
                "risk_on",
                "sw_eq_ma21",
                "sw_eq_ret21_pos",
                "sw_eq_ret63_pos",
                "sw_breadth_ma21_50",
                "sw_breadth_ret21_50",
                "sw_top_mom_63_pos",
                "pe_ttm_low",
                "pb_low",
                "valuation_low",
                "valuation_not_high",
            ],
            "min_amounts": [20_000_000, 50_000_000],
            "min_prices": [3.0, 10.0],
            "trend_filters": ["none", "ma126", "stack3", "ts3"],
            "min_hold_days": [2, 5],
            "max_hold_days": [5, 10, 20],
            "replace_counts": [1, 2],
            "stop_losses": [0.10, 0.20, None],
        }
    if profile == "smoke":
        return {
            "market_filters": ["none", "sw_top_mom_63_pos", "valuation_not_high"],
            "min_amounts": [50_000_000],
            "min_prices": [3.0],
            "trend_filters": ["none"],
            "min_hold_days": [2],
            "max_hold_days": [10],
            "replace_counts": [1],
            "stop_losses": [0.10, None],
        }
    if profile == "credible":
        return {
            "market_filters": [
                "none",
                "ma126",
                "ret63_pos",
                "ret21_63_pos",
                "ma63_ret63",
                "risk_on",
                "sw_eq_ret63_pos",
                "sw_breadth_ret21_50",
                "sw_top_mom_63_pos",
                "pe_ttm_low",
                "pb_low",
                "valuation_low",
                "valuation_not_high",
            ],
            "min_amounts": [50_000_000],
            "min_prices": [3.0, 10.0],
            "trend_filters": ["none", "ma126", "ts3"],
            "min_hold_days": [2, 5],
            "max_hold_days": [10, 20],
            "replace_counts": [1],
            "stop_losses": [0.10, None],
        }
    raise ValueError(f"unknown grid profile: {profile}")


def spec_grid(
    formula_set: str,
    formula_scope: str,
    top_n: int,
    grid_profile: str,
    min_hold_days: int | None = None,
    max_hold_days: int | None = None,
    stop_loss: float | None | str = "grid",
) -> list[DynamicSpec]:
    specs: list[DynamicSpec] = []
    values = grid_values(grid_profile)
    if min_hold_days is not None:
        values["min_hold_days"] = [min_hold_days]
    if max_hold_days is not None:
        values["max_hold_days"] = [max_hold_days]
    if stop_loss != "grid":
        values["stop_losses"] = [stop_loss]
    for formula in selected_formulas(formula_set, formula_scope):
        for market_filter in values["market_filters"]:
            for min_amount in values["min_amounts"]:
                for min_price in values["min_prices"]:
                    for trend_filter in values["trend_filters"]:
                        for min_hold_days in values["min_hold_days"]:
                            for max_hold_days in values["max_hold_days"]:
                                if min_hold_days > max_hold_days:
                                    continue
                                for replace_count in values["replace_counts"]:
                                    for stop_loss in values["stop_losses"]:
                                        specs.append(
                                            DynamicSpec(
                                                formula=formula,
                                                market_filter=market_filter,
                                                top_n=top_n,
                                                min_amount=min_amount,
                                                min_price=min_price,
                                                trend_filter=trend_filter,
                                                min_hold_days=min_hold_days,
                                                max_hold_days=max_hold_days,
                                                replace_count=replace_count,
                                                stop_loss=stop_loss,
                                            )
                                        )
    return specs


def spec_from_config(data: dict[str, Any]) -> DynamicSpec:
    formula_name = str(data.get("formula", "")).strip()
    if not formula_name:
        raise ValueError("fixed spec requires formula")
    formula_by_name = {formula.name: formula for formula in formulas("expanded")}
    formula = formula_by_name.get(formula_name)
    if formula is None:
        weights = data.get("weights")
        if isinstance(weights, str):
            weights = json.loads(weights)
        if not isinstance(weights, dict):
            raise ValueError(f"unknown formula without weights: {formula_name}")
        formula = Formula(formula_name, {str(key): float(value) for key, value in weights.items()})
    stop_loss = data.get("stop_loss")
    if pd.isna(stop_loss) or str(stop_loss).strip().lower() in {"", "nan", "none", "null"}:
        stop_loss = None
    else:
        stop_loss = float(stop_loss)
    return DynamicSpec(
        formula=formula,
        market_filter=str(data.get("market_filter", "none")),
        top_n=int(data.get("top_n", 1)),
        min_amount=float(data.get("min_amount", 50_000_000)),
        min_price=float(data.get("min_price", 3.0)),
        trend_filter=str(data.get("trend_filter", "none")),
        min_hold_days=int(data.get("min_hold_days", 5)),
        max_hold_days=int(data.get("max_hold_days", 20)),
        replace_count=int(data.get("replace_count", 1)),
        stop_loss=stop_loss,
    )


def feature_names_for_specs(specs: list[DynamicSpec]) -> list[str]:
    feature_names = sorted({name for spec in specs for name in spec.formula.weights})
    missing = [name for name in feature_names if name not in RANK_MAP]
    if missing:
        raise ValueError(f"missing rank map entries: {missing}")
    return feature_names


def source_columns_for_features(feature_names: list[str]) -> list[str]:
    columns: set[str] = set()
    for name in feature_names:
        if name in INDUSTRY_RELATIVE_FEATURES:
            columns.update(INDUSTRY_RELATIVE_FEATURES[name][1])
        else:
            columns.add(RANK_MAP[name][0])
    return sorted(columns)


def add_industry_relative_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    requested = [name for name in feature_names if name in INDUSTRY_RELATIVE_FEATURES]
    if not requested:
        return df
    if "industry_label" not in df.columns:
        out = df.copy()
        out["industry_label"] = UNKNOWN_INDUSTRY
    else:
        out = df.copy()
        out["industry_label"] = out["industry_label"].fillna(UNKNOWN_INDUSTRY).astype(str)

    grouped = out.groupby(["trade_date", "industry_label"], group_keys=False)
    for rank_name in requested:
        column, _dependencies = INDUSTRY_RELATIVE_FEATURES[rank_name]
        source_column = {
            "ind_mom_126_21_r": "mom_126_21",
            "ind_rev_5_r": "rev_5",
            "ind_lowvol_63_r": "vol_63",
            "ind_lowmax_21_r": "max_ret_21",
            "ind_low_turnover_21_r": "turnover_ma_21",
        }[rank_name]
        ascending = rank_name == "ind_mom_126_21_r" or rank_name == "ind_rev_5_r"
        out[column] = grouped[source_column].rank(pct=True, ascending=ascending)
    return out


def load_or_build_dynamic_factors(
    db: Path,
    cache_dir: Path,
    research_start: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_names: list[str],
    use_cache: bool,
    board_scope: str = "main",
    factor_adjust: str = "hfq",
    allow_factor_fallback: bool = True,
) -> pd.DataFrame:
    coverage_start = research_start - pd.Timedelta(days=460)
    with sqlite3.connect(db) as conn:
        if allow_factor_fallback:
            adjustment_coverage(conn, coverage_start, end_date, board_scope, factor_adjust)
        else:
            require_factor_adjust_coverage(conn, coverage_start, end_date, board_scope, factor_adjust)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_tag = "fallback" if allow_factor_fallback else "strict"
    cache_stem = (
        f"walkforward_factors_{FACTOR_CACHE_VERSION}_{board_scope}_{factor_adjust}_{fallback_tag}"
        f"_{research_start:%Y%m%d}_{end_date:%Y%m%d}"
    )
    feather_file = cache_dir / f"{cache_stem}.feather"
    pickle_file = cache_dir / f"{cache_stem}.pkl"
    needed_columns = {
        "symbol",
        "name",
        "trade_date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "amount",
        "open",
        "close",
        "factor_adjust_used",
        "amount_ma_21",
        "ma_126",
        "ma_stack",
        "time_series",
        *source_columns_for_features(feature_names),
    }
    cached_columns = sorted(needed_columns | {"industry_label"})
    needed_columns = sorted(needed_columns)
    if use_cache and feather_file.exists():
        print(f"loading selected factor columns from {feather_file}", flush=True)
        cached = pd.read_feather(feather_file)
        return cached[[column for column in cached_columns if column in cached.columns]].copy()
    if use_cache and pickle_file.exists():
        print(f"loading factor cache {pickle_file}", flush=True)
        cached = pd.read_pickle(pickle_file)
        return cached[[column for column in cached_columns if column in cached.columns]].copy()

    df = add_factors(load_data(db, research_start, end_date, board_scope, factor_adjust, allow_factor_fallback))
    if use_cache:
        try:
            temp_file = feather_file.with_suffix(".tmp.feather")
            df.reset_index(drop=True).to_feather(temp_file)
            temp_file.replace(feather_file)
            print(f"wrote factor cache {feather_file}", flush=True)
        except Exception as exc:
            temp_file = pickle_file.with_suffix(".tmp.pkl")
            df.to_pickle(temp_file)
            temp_file.replace(pickle_file)
            print(f"wrote pickle factor cache {pickle_file} because feather failed: {exc}", flush=True)
    return df[[column for column in cached_columns if column in df.columns]].copy()


def load_blacklist(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("blacklist", data) if isinstance(data, dict) else data
    else:
        rows = pd.read_csv(path).to_dict(orient="records")
    if not isinstance(rows, list):
        raise ValueError("blacklist file must contain rows")
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue
        output.setdefault(symbol, []).append(
            {
                "start_date": str(row.get("start_date", "") or ""),
                "end_date": str(row.get("end_date", "") or ""),
                "reason": str(row.get("reason", "blacklist")),
            }
        )
    return output


def blacklist_reason(symbol: str, signal_date: pd.Timestamp) -> str | None:
    for row in G_BLACKLIST.get(str(symbol), []):
        start = pd.Timestamp(row["start_date"]) if row.get("start_date") else None
        end = pd.Timestamp(row["end_date"]) if row.get("end_date") else None
        if start is not None and signal_date < start:
            continue
        if end is not None and signal_date > end:
            continue
        return str(row.get("reason") or "blacklist")
    return None


def industry_labels_for_symbols(day: CompactDayData, symbols: np.ndarray) -> np.ndarray:
    if len(symbols) == 0:
        return np.asarray([], dtype=str)
    if day.industry_labels is None:
        return np.full(len(symbols), UNKNOWN_INDUSTRY, dtype=object)
    positions = np.searchsorted(day.symbols, symbols)
    valid = positions < len(day.symbols)
    output = np.full(len(symbols), UNKNOWN_INDUSTRY, dtype=object)
    matched = np.zeros(len(symbols), dtype=bool)
    matched[valid] = day.symbols[positions[valid]] == symbols[valid]
    output[matched] = day.industry_labels[positions[matched]]
    return output.astype(str)


def filter_target_by_industry_weight(
    day: CompactDayData,
    target_symbols: np.ndarray,
    target_scores: np.ndarray,
    current_symbols: np.ndarray,
    current_weights: np.ndarray,
    target_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if G_MAX_INDUSTRY_WEIGHT <= 0 or len(target_symbols) == 0 or target_weight <= 0:
        return target_symbols, target_scores, 0
    current_labels = industry_labels_for_symbols(day, current_symbols)
    current_exposure = industry_exposure_map(current_symbols, current_weights, current_labels)
    current_set = set(str(symbol) for symbol in current_symbols)
    selected_symbols: list[str] = []
    selected_scores: list[float] = []
    exposure = dict(current_exposure)
    blocked = 0
    for symbol, score, label in zip(
        target_symbols,
        target_scores,
        industry_labels_for_symbols(day, target_symbols),
        strict=False,
    ):
        symbol_text = str(symbol)
        label_text = str(label or UNKNOWN_INDUSTRY)
        incremental = 0.0 if symbol_text in current_set else target_weight
        if exposure.get(label_text, 0.0) + incremental > G_MAX_INDUSTRY_WEIGHT + 1e-12:
            blocked += 1
            continue
        selected_symbols.append(symbol_text)
        selected_scores.append(float(score))
        exposure[label_text] = exposure.get(label_text, 0.0) + incremental
    return (
        np.asarray(selected_symbols, dtype=str),
        np.asarray(selected_scores, dtype=np.float32),
        blocked,
    )


def load_industry_close(
    db: Path,
    research_start: pd.Timestamp,
    end_date: pd.Timestamp,
    source: str,
) -> pd.DataFrame:
    if not source:
        return pd.DataFrame()
    with sqlite3.connect(db) as conn:
        boards = pd.read_sql_query(
            """
            select board_code, trade_date, close
            from industry_daily_bars
            where source = ?
              and trade_date >= ?
              and trade_date <= ?
              and close is not null
            """,
            conn,
            params=(source, research_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )
    if boards.empty:
        print(f"warning: no industry bars for source={source}", flush=True)
        return pd.DataFrame()
    boards["trade_date"] = pd.to_datetime(boards["trade_date"])
    boards["close"] = pd.to_numeric(boards["close"], errors="coerce")
    boards = boards.dropna(subset=["close"])
    return boards.pivot(index="trade_date", columns="board_code", values="close").sort_index()


def industry_market_states(industry_close: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    if industry_close.empty:
        return {}
    ret_1 = industry_close.pct_change(fill_method=None)
    eq_daily = ret_1.mean(axis=1).fillna(0.0)
    eq_curve = (1.0 + eq_daily).cumprod()
    eq_ma21 = eq_curve.rolling(21).mean()
    eq_ret21 = eq_curve.pct_change(21)
    eq_ret63 = eq_curve.pct_change(63)

    ma21 = industry_close.rolling(21).mean()
    board_ret21 = industry_close.pct_change(21)
    board_ret63 = industry_close.pct_change(63)
    breadth_ma21 = (industry_close > ma21).sum(axis=1) / industry_close.notna().sum(axis=1).replace(0, np.nan)
    breadth_ret21 = (board_ret21 > 0).sum(axis=1) / board_ret21.notna().sum(axis=1).replace(0, np.nan)
    top_mom63 = board_ret63.rank(axis=1, pct=True).where(board_ret63 > 0).max(axis=1)

    dates = industry_close.index
    return {
        "sw_eq_ma21": set(dates[eq_curve > eq_ma21]),
        "sw_eq_ret21_pos": set(dates[eq_ret21 > 0]),
        "sw_eq_ret63_pos": set(dates[eq_ret63 > 0]),
        "sw_breadth_ma21_50": set(dates[breadth_ma21 >= 0.50]),
        "sw_breadth_ret21_50": set(dates[breadth_ret21 >= 0.50]),
        "sw_top_mom_63_pos": set(dates[top_mom63 > 0]),
    }


def load_market_valuation(
    db: Path,
    research_start: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        has_table = conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = 'market_valuation_daily'"
        ).fetchone()
        if has_table is None:
            return pd.DataFrame()
        frame = pd.read_sql_query(
            """
            select trade_date, middle_pe_ttm, middle_pb
            from market_valuation_daily
            where trade_date >= ?
              and trade_date <= ?
            order by trade_date
            """,
            conn,
            params=(research_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["middle_pe_ttm"] = pd.to_numeric(frame["middle_pe_ttm"], errors="coerce")
    frame["middle_pb"] = pd.to_numeric(frame["middle_pb"], errors="coerce")
    return frame.dropna(subset=["trade_date"]).sort_values("trade_date")


def _rolling_last_percentile(series: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    def percentile(values: np.ndarray) -> float:
        current = values[-1]
        if not np.isfinite(current):
            return np.nan
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid <= current))

    return series.rolling(window=window, min_periods=min_periods).apply(percentile, raw=True)


def market_valuation_states(valuation: pd.DataFrame) -> dict[str, set[pd.Timestamp]]:
    if valuation.empty:
        return {}
    frame = valuation.copy().sort_values("trade_date")
    frame = frame.set_index("trade_date")
    pe_pct = _rolling_last_percentile(frame["middle_pe_ttm"])
    pb_pct = _rolling_last_percentile(frame["middle_pb"])
    low_pe = pe_pct <= 0.40
    low_pb = pb_pct <= 0.40
    not_high_pe = pe_pct <= 0.70
    not_high_pb = pb_pct <= 0.70
    dates = frame.index
    return {
        "pe_ttm_low": set(dates[low_pe.fillna(False)]),
        "pb_low": set(dates[low_pb.fillna(False)]),
        "valuation_low": set(dates[(low_pe & low_pb).fillna(False)]),
        "valuation_not_high": set(dates[(not_high_pe & not_high_pb).fillna(False)]),
    }


def _status_bool(series: pd.Series, default: bool = False) -> pd.Series:
    return status_bool(series, default)


def enrich_backtest_constraints(db: Path, df: pd.DataFrame, board_scope: str) -> tuple[pd.DataFrame, dict]:
    """Merge lifecycle and historical status constraints after factor cache loading."""
    if df.empty:
        return df, {}
    start = pd.Timestamp(df["trade_date"].min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(df["trade_date"].max()).strftime("%Y-%m-%d")
    symbols = sorted(str(symbol) for symbol in df["symbol"].dropna().astype(str).unique())
    enriched = df.copy()
    with sqlite3.connect(db) as conn:
        status = pd.DataFrame()
        lifecycle = pd.DataFrame()
        has_status_table = conn.execute(
            "select 1 from sqlite_master where type='table' and name='symbol_status_daily'"
        ).fetchone() is not None
        has_lifecycle_table = conn.execute(
            "select 1 from sqlite_master where type='table' and name='symbol_lifecycle'"
        ).fetchone() is not None
        if has_status_table and symbols:
            placeholders = ", ".join("?" for _ in symbols)
            status = pd.read_sql_query(
                f"""
                select symbol, trade_date,
                       coalesce(is_st, 0) as is_st,
                       coalesce(is_suspended, 0) as is_suspended,
                       board as status_board
                from symbol_status_daily
                where trade_date >= ?
                  and trade_date <= ?
                  and symbol in ({placeholders})
                """,
                conn,
                params=(start, end, *symbols),
            )
        if has_lifecycle_table and symbols:
            placeholders = ", ".join("?" for _ in symbols)
            lifecycle = pd.read_sql_query(
                f"""
                select symbol, list_date, delist_date, board as lifecycle_board
                from symbol_lifecycle
                where symbol in ({placeholders})
                """,
                conn,
                params=tuple(symbols),
            )

    if not status.empty:
        status["trade_date"] = pd.to_datetime(status["trade_date"])
        status["symbol"] = status["symbol"].astype(str)
        enriched = enriched.merge(status, on=["symbol", "trade_date"], how="left")
    else:
        enriched["is_st"] = 0
        enriched["is_suspended"] = 0
        enriched["status_board"] = None

    if not lifecycle.empty:
        lifecycle["symbol"] = lifecycle["symbol"].astype(str)
        enriched = enriched.merge(lifecycle, on="symbol", how="left")
    else:
        enriched["list_date"] = pd.NaT
        enriched["delist_date"] = pd.NaT
        enriched["lifecycle_board"] = None

    enriched["has_status"] = enriched["is_st"].notna() | enriched["is_suspended"].notna()
    enriched["is_st"] = _status_bool(enriched["is_st"])
    enriched["is_suspended"] = _status_bool(enriched["is_suspended"])
    list_date = pd.to_datetime(enriched["list_date"], errors="coerce")
    delist_date = pd.to_datetime(enriched["delist_date"], errors="coerce")
    trade_date = pd.to_datetime(enriched["trade_date"])
    lifecycle_allowed = (list_date.isna() | (trade_date >= list_date)) & (delist_date.isna() | (trade_date <= delist_date))
    enriched["lifecycle_allowed"] = lifecycle_allowed.astype(bool)
    inferred_board = enriched["symbol"].astype(str).map(infer_board_label)
    enriched["board"] = enriched["status_board"].combine_first(enriched["lifecycle_board"]).combine_first(inferred_board)
    enriched["limit_rate"] = [
        limit_rate_for_symbol(symbol, board, is_st)
        for symbol, board, is_st in zip(
            enriched["symbol"].astype(str),
            enriched["board"],
            enriched["is_st"],
            strict=False,
        )
    ]
    enriched["signal_allowed"] = (~enriched["is_st"]) & enriched["lifecycle_allowed"]
    summary = {
        "rows": int(len(enriched)),
        "symbols": int(enriched["symbol"].nunique()),
        "status_rows": int(enriched["has_status"].sum()),
        "status_row_coverage": float(enriched["has_status"].mean()) if len(enriched) else 0.0,
        "signal_st_rows": int(enriched["is_st"].sum()),
        "suspended_rows": int(enriched["is_suspended"].sum()),
        "lifecycle_blocked_rows": int((~enriched["lifecycle_allowed"]).sum()),
        "board_scope": board_scope,
    }
    return enriched, summary


def _group_limit_rates(group: pd.DataFrame) -> np.ndarray:
    if "limit_rate" in group:
        rates = pd.to_numeric(group["limit_rate"], errors="coerce").to_numpy(dtype=float)
        missing = ~np.isfinite(rates)
        if not bool(missing.any()):
            return rates
    else:
        rates = np.full(len(group), np.nan, dtype=float)
        missing = np.ones(len(group), dtype=bool)
    boards = group["board"] if "board" in group else pd.Series([None] * len(group), index=group.index)
    st_values = _status_bool(group["is_st"]) if "is_st" in group else pd.Series([False] * len(group), index=group.index)
    fallback = np.asarray(
        [
            limit_rate_for_symbol(symbol, board, bool(is_st))
            for symbol, board, is_st in zip(group["symbol"].astype(str), boards, st_values, strict=False)
        ],
        dtype=float,
    )
    rates[missing] = fallback[missing]
    return rates


def limit_trade_masks(
    group: pd.DataFrame,
    entry_open: np.ndarray,
    entry_high: np.ndarray,
    entry_low: np.ndarray,
    entry_limit_rate: np.ndarray | None = None,
    entry_suspended: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prev_close = group["raw_close"].to_numpy(dtype=float)
    rates = _group_limit_rates(group) if entry_limit_rate is None else np.asarray(entry_limit_rate, dtype=float)
    rates = np.where(np.isfinite(rates), rates, _group_limit_rates(group))
    return locked_limit_masks(
        prev_close=prev_close,
        entry_open=entry_open,
        entry_high=entry_high,
        entry_low=entry_low,
        limit_rate=rates,
        entry_suspended=entry_suspended,
        limit_epsilon=G_EXECUTION.limit_epsilon,
        block_limit_trades=G_EXECUTION.block_limit_trades,
    )


def add_rank_columns_for(signal_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    signal_df = signal_df.copy()
    grouped = signal_df.groupby("trade_date", group_keys=False)
    for rank_name in feature_names:
        column, ascending = RANK_MAP[rank_name]
        signal_df[rank_name] = grouped[column].rank(pct=True, ascending=ascending)
    return signal_df


def build_compact_day_data(
    df: pd.DataFrame,
    open_price: pd.DataFrame,
    high_price: pd.DataFrame,
    low_price: pd.DataFrame,
    amount: pd.DataFrame,
    volume: pd.DataFrame,
    is_st: pd.DataFrame,
    is_suspended: pd.DataFrame,
    limit_rate: pd.DataFrame,
    signal_allowed: pd.DataFrame,
    dates: list[pd.Timestamp],
    all_dates: pd.DatetimeIndex,
    feature_names: list[str],
) -> list[CompactDayData]:
    signal_df = add_rank_columns_for(df[df["trade_date"].isin(set(dates))].copy(), feature_names)
    output: list[CompactDayData] = []
    for signal_date in dates:
        future = all_dates[all_dates > signal_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        group = signal_df[signal_df["trade_date"] == signal_date].copy().sort_values("symbol")
        if group.empty:
            continue
        symbols = group["symbol"].astype(str).to_numpy()
        entry = open_price.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        entry_high = high_price.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        entry_low = low_price.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        entry_amount = amount.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        entry_volume = volume.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        entry_suspended = is_suspended.loc[entry_date].reindex(symbols).fillna(False).to_numpy(dtype=bool)
        entry_limit_rate = limit_rate.loc[entry_date].reindex(symbols).to_numpy(dtype=float)
        signal_allowed_values = signal_allowed.loc[signal_date].reindex(symbols).fillna(False).to_numpy(dtype=bool)
        industry_labels = (
            group["industry_label"].fillna(UNKNOWN_INDUSTRY).astype(str).to_numpy()
            if "industry_label" in group
            else np.full(len(group), UNKNOWN_INDUSTRY, dtype=object)
        )
        entry_buyable, entry_sellable = limit_trade_masks(
            group,
            entry,
            entry_high,
            entry_low,
            entry_limit_rate,
            entry_suspended,
        )
        output.append(
            CompactDayData(
                signal_date=signal_date,
                entry_date=entry_date,
                symbols=symbols,
                entry_open=entry.astype(np.float32),
                entry_high=entry_high.astype(np.float32),
                entry_low=entry_low.astype(np.float32),
                entry_amount=entry_amount.astype(np.float32),
                entry_volume=entry_volume.astype(np.float32),
                entry_buyable=entry_buyable,
                entry_sellable=entry_sellable,
                signal_allowed=signal_allowed_values,
                features=group[feature_names].to_numpy(dtype=np.float32).T,
                amount21=group["amount_ma_21"].to_numpy(dtype=np.float32),
                close=group["raw_close"].to_numpy(dtype=np.float32),
                trend_masks={
                    "none": np.ones(len(group), dtype=bool),
                    "ma126": (group["close"] > group["ma_126"]).to_numpy(dtype=bool),
                    "stack3": (group["ma_stack"] >= 3).to_numpy(dtype=bool),
                    "ts3": (group["time_series"] >= 3).to_numpy(dtype=bool),
                },
                entry_suspended=entry_suspended,
                entry_limit_rate=entry_limit_rate.astype(np.float32),
                industry_labels=industry_labels.astype(str),
            )
        )
    return output


def weights_for_formula(formula: Formula) -> np.ndarray:
    feature_index = {name: index for index, name in enumerate(G_FEATURE_NAMES)}
    weights = np.zeros(len(G_FEATURE_NAMES), dtype=np.float32)
    for name, weight in formula.weights.items():
        weights[feature_index[name]] = weight
    return weights


def target_key(spec: DynamicSpec) -> tuple:
    return (spec.formula.name, spec.top_n, spec.min_amount, spec.min_price, spec.trend_filter)


def select_target_from_score(
    day: CompactDayData,
    spec: DynamicSpec,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(score)
        & (day.amount21 >= spec.min_amount)
        & (day.close >= spec.min_price)
        & day.signal_allowed
        & day.trend_masks[spec.trend_filter]
    )
    if int(mask.sum()) < spec.top_n:
        return EMPTY_TARGET
    positions = np.flatnonzero(mask)
    masked_scores = score[positions]
    candidate_positions = positions[np.argsort(masked_scores)[::-1]]
    selected_positions: list[int] = []
    for position in candidate_positions:
        if blacklist_reason(str(day.symbols[position]), day.signal_date):
            continue
        selected_positions.append(int(position))
        if len(selected_positions) >= spec.top_n:
            break
    if len(selected_positions) < spec.top_n:
        return EMPTY_TARGET
    top_positions = np.asarray(selected_positions, dtype=int)
    target_open = day.entry_open[top_positions].astype(np.float32, copy=False)
    executable = np.isfinite(target_open) & (target_open > 0)
    return (
        day.symbols[top_positions][executable],
        score[top_positions].astype(np.float32, copy=False)[executable],
    )


def build_target_cache(
    days: list[CompactDayData],
    specs: list[DynamicSpec],
) -> dict[tuple, list[tuple[np.ndarray, np.ndarray]]]:
    formulas_by_name = {spec.formula.name: spec.formula for spec in specs}
    specs_by_formula: dict[str, list[DynamicSpec]] = {}
    for spec in specs:
        specs_by_formula.setdefault(spec.formula.name, [])
        key = target_key(spec)
        if not any(target_key(existing) == key for existing in specs_by_formula[spec.formula.name]):
            specs_by_formula[spec.formula.name].append(spec)

    targets = {target_key(spec): [] for values in specs_by_formula.values() for spec in values}
    for formula_name, formula in formulas_by_name.items():
        weights = weights_for_formula(formula)
        formula_specs = specs_by_formula[formula_name]
        for day in days:
            score = weights @ day.features
            for spec in formula_specs:
                targets[target_key(spec)].append(select_target_from_score(day, spec, score))
    return targets


def capacity_fill_notional(
    day: CompactDayData,
    symbols: np.ndarray,
    desired_notional: np.ndarray,
    executable: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip desired notional by capacity while preserving hard execution blocks."""
    prices, price_valid = open_for_symbols(day, symbols)
    amounts, amount_valid = values_for_symbols(day, symbols, "entry_amount")
    return capacity_fill_notional_from_arrays(
        prices=prices,
        price_valid=price_valid,
        amounts=amounts,
        amount_valid=amount_valid,
        desired_notional=desired_notional,
        executable=executable,
        side=side,
        lot_size=G_EXECUTION.lot_size,
        capacity_pct_of_amount=G_EXECUTION.capacity_pct_of_amount,
    )


def trade_block_reason(
    day: CompactDayData,
    symbol: str,
    side: str,
    desired_notional: float,
    filled_notional: float,
) -> str:
    prices, price_valid = open_for_symbols(day, np.asarray([symbol], dtype=str))
    amounts, amount_valid = values_for_symbols(day, np.asarray([symbol], dtype=str), "entry_amount")
    position = int(np.searchsorted(day.symbols, symbol))
    symbol_present = position < len(day.symbols) and day.symbols[position] == symbol
    suspended = bool(day.entry_suspended[position]) if symbol_present and day.entry_suspended is not None else False
    executable_field = "entry_buyable" if side == "buy" else "entry_sellable"
    executable = bool_for_symbols(day, np.asarray([symbol], dtype=str), executable_field)
    return trade_block_reason_from_fields(
        side=side,
        desired_notional=desired_notional,
        filled_notional=filled_notional,
        price_valid=bool(price_valid[0]),
        symbol_present=bool(symbol_present),
        suspended=suspended,
        executable=bool(executable[0]),
        amount_valid=bool(amount_valid[0]),
        price=float(prices[0]),
        amount=float(amounts[0]) if bool(amount_valid[0]) else np.nan,
        lot_size=G_EXECUTION.lot_size,
        capacity_pct_of_amount=G_EXECUTION.capacity_pct_of_amount,
    )


def build_trade_event(
    day: CompactDayData,
    symbol: str,
    side: str,
    desired_notional: float,
    filled_notional: float,
    weight_before: float,
    weight_after: float,
    reason_override: str | None = None,
) -> dict:
    reason = reason_override or trade_block_reason(day, symbol, side, desired_notional, filled_notional)
    prices, price_valid = open_for_symbols(day, np.asarray([symbol], dtype=str))
    amounts, amount_valid = values_for_symbols(day, np.asarray([symbol], dtype=str), "entry_amount")
    return trade_event_row(
        signal_date=day.signal_date,
        entry_date=day.entry_date,
        symbol=symbol,
        side=side,
        desired_notional=desired_notional,
        filled_notional=filled_notional,
        weight_before=weight_before,
        weight_after=weight_after,
        reason=reason,
        entry_open=float(prices[0]) if bool(price_valid[0]) else None,
        entry_amount=float(amounts[0]) if bool(amount_valid[0]) else None,
    )


def side_cost_fraction(day: CompactDayData, symbols: np.ndarray, notionals: np.ndarray, equity_cash: float, side: str) -> float:
    amounts, valid = values_for_symbols(day, symbols, "entry_amount")
    return side_cost_fraction_from_amounts(
        amounts=amounts,
        valid=valid,
        notionals=notionals,
        equity_cash=equity_cash,
        side=side,
        buy_cost=G_EXECUTION.buy_cost,
        sell_cost=G_EXECUTION.sell_cost,
        slippage_bps=G_EXECUTION.slippage_bps,
        impact_bps_per_pct_amount=G_EXECUTION.impact_bps_per_pct_amount,
    )


def execute_rebalance(
    day: CompactDayData,
    current_symbols: np.ndarray,
    current_entry_open: np.ndarray,
    target_symbols: np.ndarray,
    equity: float,
    current_weights: np.ndarray | None = None,
    target_weight_override: float | None = None,
) -> tuple[ExecutionResult, float]:
    equity_cash = max(equity * G_INITIAL_CASH, 1.0)
    if not np.isfinite(equity_cash) or equity_cash <= 0:
        equity_cash = 1.0
    capacity_cash = G_INITIAL_CASH if G_EXECUTION.capacity_equity_mode == "initial" else equity_cash
    current_weights = normalize_current_weights(current_symbols, current_weights)
    current_set = set(str(symbol) for symbol in current_symbols)
    target_set = set(str(symbol) for symbol in target_symbols)
    target_weight = target_rebalance_weight(target_symbols, target_weight_override, G_MAX_POSITION_WEIGHT)

    current_weight_map = {
        str(symbol): float(weight) for symbol, weight in zip(current_symbols, current_weights, strict=False)
    }
    current_industry_labels = industry_labels_for_symbols(day, current_symbols)
    industry_exposure = industry_exposure_map(current_symbols, current_weights, current_industry_labels)
    sell_candidates_list: list[str] = []
    sell_desired_list: list[float] = []
    for symbol, industry_label in zip(current_symbols, current_industry_labels, strict=False):
        symbol_text = str(symbol)
        industry_label_text = str(industry_label or UNKNOWN_INDUSTRY)
        current_weight = current_weight_map.get(symbol_text, 0.0)
        if symbol_text not in target_set:
            desired = current_weight * equity_cash
        elif G_MAX_POSITION_WEIGHT > 0 and current_weight > G_MAX_POSITION_WEIGHT:
            desired = (current_weight - G_MAX_POSITION_WEIGHT) * equity_cash
        else:
            desired = 0.0
        if G_MAX_INDUSTRY_WEIGHT > 0:
            label_exposure = industry_exposure.get(industry_label_text, 0.0)
            if label_exposure > G_MAX_INDUSTRY_WEIGHT and current_weight > 0:
                excess_weight = label_exposure - G_MAX_INDUSTRY_WEIGHT
                desired = max(desired, min(current_weight, excess_weight * current_weight / label_exposure) * equity_cash)
        if desired > 0:
            sell_candidates_list.append(symbol_text)
            sell_desired_list.append(desired)
    sell_candidates = np.asarray(sell_candidates_list, dtype=str)
    sell_desired = np.asarray(
        sell_desired_list,
        dtype=np.float64,
    )
    sell_fill, sell_filled_mask, partial_sell_mask = capacity_fill_notional(
        day,
        sell_candidates,
        sell_desired,
        bool_for_symbols(day, sell_candidates, "entry_sellable"),
        "sell",
    )
    sold = sell_candidates[sell_filled_mask]
    blocked_sells = sell_candidates[~sell_filled_mask]
    sell_unfilled_by_symbol = {
        str(symbol): max(float(desired - filled), 0.0)
        for symbol, desired, filled in zip(sell_candidates, sell_desired, sell_fill, strict=False)
    }
    sell_fill_by_symbol = {
        str(symbol): float(fill) for symbol, fill in zip(sell_candidates, sell_fill, strict=False)
    }

    buy_candidates = np.asarray([symbol for symbol in target_symbols if str(symbol) not in current_set], dtype=str)
    new_weight_notional = capacity_cash * target_weight if target_weight > 0 else 0.0
    buy_notionals = np.full(len(buy_candidates), new_weight_notional, dtype=np.float64)
    buy_fill, buy_filled_mask, partial_buy_mask = capacity_fill_notional(
        day,
        buy_candidates,
        buy_notionals,
        bool_for_symbols(day, buy_candidates, "entry_buyable"),
        "buy",
    )
    sell_slots = len(sold) + max(len(target_symbols) - len(current_symbols), 0)
    if sell_slots < len(buy_filled_mask):
        limited = np.zeros(len(buy_filled_mask), dtype=bool)
        keep_positions = np.flatnonzero(buy_filled_mask)[:sell_slots]
        limited[keep_positions] = True
        buy_fill[~limited] = 0.0
        partial_buy_mask[~limited] = False
        buy_filled_mask = limited
    bought = buy_candidates[buy_filled_mask]
    blocked_buys = buy_candidates[~buy_filled_mask]
    buy_fill_by_symbol = {
        str(symbol): float(fill) for symbol, fill in zip(buy_candidates, buy_fill, strict=False) if fill > 0
    }
    turnover_blocked_symbols: set[str] = set()
    if G_MAX_TURNOVER_PCT > 0:
        remaining_turnover_budget = equity_cash * G_MAX_TURNOVER_PCT
        for fill_array, symbols_array, partial_mask_array in (
            (sell_fill, sell_candidates, partial_sell_mask),
            (buy_fill, buy_candidates, partial_buy_mask),
        ):
            for index, symbol in enumerate(symbols_array):
                if fill_array[index] <= 0:
                    continue
                if remaining_turnover_budget <= 0:
                    fill_array[index] = 0.0
                    turnover_blocked_symbols.add(str(symbol))
                    continue
                if fill_array[index] > remaining_turnover_budget:
                    fill_array[index] = remaining_turnover_budget
                    turnover_blocked_symbols.add(str(symbol))
                    partial_mask_array[index] = True
                remaining_turnover_budget -= fill_array[index]
        sell_filled_mask = sell_fill > 0
        buy_filled_mask = buy_fill > 0
        partial_sell_mask = sell_filled_mask & (sell_fill < sell_desired)
        partial_buy_mask = buy_filled_mask & (buy_fill < buy_notionals)
        sold = sell_candidates[sell_filled_mask]
        bought = buy_candidates[buy_filled_mask]
        blocked_sells = sell_candidates[~sell_filled_mask]
        blocked_buys = buy_candidates[~buy_filled_mask]
        sell_unfilled_by_symbol = {
            str(symbol): max(float(desired - filled), 0.0)
            for symbol, desired, filled in zip(sell_candidates, sell_desired, sell_fill, strict=False)
        }
        sell_fill_by_symbol = {
            str(symbol): float(fill) for symbol, fill in zip(sell_candidates, sell_fill, strict=False)
        }
        buy_fill_by_symbol = {
            str(symbol): float(fill) for symbol, fill in zip(buy_candidates, buy_fill, strict=False) if fill > 0
        }

    remaining_symbols: list[str] = []
    remaining_weights: list[float] = []
    for symbol, weight in zip(current_symbols, current_weights, strict=False):
        symbol_text = str(symbol)
        if symbol_text in sell_unfilled_by_symbol:
            residual_notional = sell_unfilled_by_symbol.get(symbol_text, 0.0)
        else:
            residual_notional = weight * equity_cash
        if not np.isfinite(residual_notional) or residual_notional <= 0:
            continue
        residual_weight = residual_notional / equity_cash if equity_cash > 0 else 0.0
        if np.isfinite(residual_weight) and residual_weight > 0:
            remaining_symbols.append(symbol_text)
            remaining_weights.append(residual_weight)
    bought_weights = [
        buy_fill_by_symbol.get(str(symbol), 0.0) / equity_cash
        if np.isfinite(buy_fill_by_symbol.get(str(symbol), 0.0)) and equity_cash > 0
        else 0.0
        for symbol in bought
    ]
    new_symbols = np.asarray(remaining_symbols + [str(symbol) for symbol in bought], dtype=str)
    new_weights = np.asarray(remaining_weights + bought_weights, dtype=np.float64)
    new_prev_open, price_valid = open_for_symbols(day, new_symbols)
    old_entry_map = {str(symbol): float(price) for symbol, price in zip(current_symbols, current_entry_open, strict=False)}
    bought_set = set(str(symbol) for symbol in bought)
    new_entry_open = np.asarray(
        [
            float(new_prev_open[index]) if str(symbol) in bought_set else old_entry_map.get(str(symbol), float(new_prev_open[index]))
            for index, symbol in enumerate(new_symbols)
        ],
        dtype=np.float32,
    )
    if len(new_symbols):
        keep = price_valid & np.isfinite(new_prev_open) & (new_prev_open > 0)
        keep &= np.isfinite(new_weights) & (new_weights > 0)
        new_symbols = new_symbols[keep]
        new_prev_open = new_prev_open[keep]
        new_entry_open = new_entry_open[keep]
        new_weights = new_weights[keep]

    new_weight_map = {
        str(symbol): float(weight) for symbol, weight in zip(new_symbols, new_weights, strict=False)
    }
    trade_events: list[dict] = []
    for symbol, desired in zip(sell_candidates, sell_desired, strict=False):
        symbol_text = str(symbol)
        trade_events.append(
            build_trade_event(
                day,
                symbol_text,
                "sell",
                float(desired),
                sell_fill_by_symbol.get(symbol_text, 0.0),
                current_weight_map.get(symbol_text, 0.0),
                new_weight_map.get(symbol_text, 0.0),
                "turnover_block" if symbol_text in turnover_blocked_symbols else None,
            )
        )
    for symbol, desired in zip(buy_candidates, buy_notionals, strict=False):
        symbol_text = str(symbol)
        trade_events.append(
            build_trade_event(
                day,
                symbol_text,
                "buy",
                float(desired),
                buy_fill_by_symbol.get(symbol_text, 0.0),
                0.0,
                new_weight_map.get(symbol_text, 0.0),
                "turnover_block" if symbol_text in turnover_blocked_symbols else None,
            )
        )

    sell_cost = side_cost_fraction(day, sell_candidates[sell_filled_mask], sell_fill[sell_filled_mask], equity_cash, "sell")
    buy_cost = side_cost_fraction(day, buy_candidates[buy_filled_mask], buy_fill[buy_filled_mask], equity_cash, "buy")
    return ExecutionResult(
        equity=equity,
        trade_count=int(len(sold) + len(bought)),
        blocked_buy_count=int(len(blocked_buys)),
        blocked_sell_count=int(len(blocked_sells)),
        partial_buy_count=int(np.sum(partial_buy_mask)),
        partial_sell_count=int(np.sum(partial_sell_mask)),
        turnover_blocked_count=int(len(turnover_blocked_symbols)),
        turnover_value=float(np.sum(sell_fill) + np.sum(buy_fill)),
        unfilled_buy_value=float(np.sum(np.maximum(buy_notionals - buy_fill, 0.0))),
        unfilled_sell_value=float(np.sum(np.maximum(sell_desired - sell_fill, 0.0))),
        turnover_blocked_value=float(
            sum(
                max(float(desired - filled), 0.0)
                for symbol, desired, filled in zip(sell_candidates, sell_desired, sell_fill, strict=False)
                if str(symbol) in turnover_blocked_symbols
            )
            + sum(
                max(float(desired - filled), 0.0)
                for symbol, desired, filled in zip(buy_candidates, buy_notionals, buy_fill, strict=False)
                if str(symbol) in turnover_blocked_symbols
            )
        ),
        current_symbols=new_symbols,
        current_prev_open=new_prev_open.astype(np.float32, copy=False),
        current_entry_open=new_entry_open.astype(np.float32, copy=False),
        current_weights=new_weights.astype(np.float32, copy=False),
        trade_events=trade_events,
    ), sell_cost + buy_cost


def empty_execution_result(
    equity: float,
    current_symbols: np.ndarray,
    current_prev_open: np.ndarray,
    current_entry_open: np.ndarray,
    current_weights: np.ndarray,
) -> ExecutionResult:
    return ExecutionResult(
        equity=equity,
        trade_count=0,
        blocked_buy_count=0,
        blocked_sell_count=0,
        partial_buy_count=0,
        partial_sell_count=0,
        turnover_blocked_count=0,
        turnover_value=0.0,
        unfilled_buy_value=0.0,
        unfilled_sell_value=0.0,
        turnover_blocked_value=0.0,
        current_symbols=current_symbols,
        current_prev_open=current_prev_open,
        current_entry_open=current_entry_open,
        current_weights=current_weights,
        trade_events=[],
    )


def should_trade(
    current_symbols: np.ndarray,
    target_symbols: np.ndarray,
    hold_days: int,
    spec_changed: bool,
    stop_hit: bool,
    spec: DynamicSpec,
) -> bool:
    if spec_changed or stop_hit:
        return True
    if len(current_symbols) == 0:
        return len(target_symbols) > 0
    if len(target_symbols) == 0:
        return True
    if hold_days < spec.min_hold_days:
        return False
    if hold_days >= spec.max_hold_days:
        return True
    replacements = len(set(target_symbols).difference(set(current_symbols)))
    return replacements >= spec.replace_count


def simulate_dynamic(
    days: list[CompactDayData],
    spec_by_year: dict[int, DynamicSpec] | None,
    single_spec: DynamicSpec | None,
    market: dict[str, set[pd.Timestamp]],
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    collect_rows: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict], list[dict]]:
    current_symbols = np.asarray([], dtype=str)
    current_prev_open = np.asarray([], dtype=np.float32)
    current_entry_open = np.asarray([], dtype=np.float32)
    current_weights = np.asarray([], dtype=np.float32)
    current_spec: DynamicSpec | None = None
    hold_days = 0
    returns: list[float] = []
    active: list[bool] = []
    trades: list[bool] = []
    trade_counts: list[int] = []
    blocked_buy_counts: list[int] = []
    blocked_sell_counts: list[int] = []
    partial_buy_counts: list[int] = []
    partial_sell_counts: list[int] = []
    turnover_blocked_counts: list[int] = []
    turnover_values: list[float] = []
    unfilled_buy_values: list[float] = []
    unfilled_sell_values: list[float] = []
    turnover_blocked_values: list[float] = []
    industry_blocked_counts: list[int] = []
    equity_rows: list[dict] = []
    pick_rows: list[dict] = []
    trade_event_rows: list[dict] = []
    equity = 1.0
    peak = 1.0
    risk_peak = 1.0
    portfolio_risk_off = False

    for day_index, day in enumerate(days):
        if start_date is not None and day.signal_date < start_date:
            continue
        if end_date is not None and day.signal_date > end_date:
            continue
        spec = single_spec if single_spec is not None else spec_by_year.get(day.signal_date.year)  # type: ignore[union-attr]
        if spec is None:
            if len(current_symbols) > 0:
                daily_ret, current_prev_open = open_to_open_return(
                    day,
                    current_symbols,
                    current_prev_open,
                    current_weights,
                )
                equity *= 1.0 + daily_ret
                execution, cost = execute_rebalance(
                    day,
                    current_symbols,
                    current_entry_open,
                    EMPTY_SYMBOLS,
                    equity,
                    current_weights,
                )
                equity *= 1.0 - cost
                net_ret = (1.0 + daily_ret) * (1.0 - cost) - 1.0
                current_symbols = execution.current_symbols
                current_prev_open = execution.current_prev_open
                current_entry_open = execution.current_entry_open
                current_weights = execution.current_weights
                hold_days = 0
            else:
                net_ret = 0.0
                execution = empty_execution_result(
                    equity,
                    current_symbols,
                    current_prev_open,
                    current_entry_open,
                    current_weights,
                )
            returns.append(net_ret)
            active.append(len(current_symbols) > 0)
            trades.append(execution.trade_count > 0)
            trade_counts.append(execution.trade_count)
            blocked_buy_counts.append(execution.blocked_buy_count)
            blocked_sell_counts.append(execution.blocked_sell_count)
            partial_buy_counts.append(execution.partial_buy_count)
            partial_sell_counts.append(execution.partial_sell_count)
            turnover_blocked_counts.append(execution.turnover_blocked_count)
            turnover_values.append(execution.turnover_value)
            unfilled_buy_values.append(execution.unfilled_buy_value)
            unfilled_sell_values.append(execution.unfilled_sell_value)
            turnover_blocked_values.append(execution.turnover_blocked_value)
            industry_blocked_counts.append(0)
            if collect_rows and execution.trade_events:
                trade_event_rows.extend(execution.trade_events)
            continue

        spec_changed = current_spec is not None and spec != current_spec
        daily_ret = 0.0
        stop_hit = False
        if len(current_symbols) > 0:
            daily_ret, current_prev_open = open_to_open_return(
                day,
                current_symbols,
                current_prev_open,
                current_weights,
            )
            hold_days += 1
            if spec.stop_loss is not None:
                stop_hit = return_since_entry(day, current_symbols, current_entry_open, current_weights) <= -spec.stop_loss

        equity_before = equity
        marked_equity = equity * (1.0 + daily_ret)
        reentry_dates = market.get(G_PORTFOLIO_REENTRY_FILTER, set())
        if portfolio_risk_off and len(current_symbols) == 0 and day.signal_date in reentry_dates:
            portfolio_risk_off = False
            risk_peak = max(marked_equity, 1e-12)
        if not portfolio_risk_off:
            risk_peak = max(risk_peak, marked_equity)
            if G_PORTFOLIO_STOP_LOSS > 0 and marked_equity / max(risk_peak, 1e-12) - 1.0 <= -G_PORTFOLIO_STOP_LOSS:
                portfolio_risk_off = True

        if (not portfolio_risk_off) and day.signal_date in market.get(spec.market_filter, set()) and not stop_hit:
            target_symbols, target_scores = G_TARGETS[target_key(spec)][day_index]
            target_open, executable = open_for_symbols(day, target_symbols)
            if not bool(executable.all()):
                target_symbols = target_symbols[executable]
                target_scores = target_scores[executable]
                target_open = target_open[executable]
        else:
            target_symbols, target_scores = EMPTY_TARGET
            target_open = EMPTY_FLOATS

        target_weight = 1.0 / len(target_symbols) if len(target_symbols) else 0.0
        if G_MAX_POSITION_WEIGHT > 0:
            target_weight = min(target_weight, G_MAX_POSITION_WEIGHT)
        target_weight_before_industry_filter = target_weight
        target_symbols, target_scores, industry_blocked_count = filter_target_by_industry_weight(
            day,
            target_symbols,
            target_scores,
            current_symbols,
            current_weights,
            target_weight,
        )
        trade_now = should_trade(current_symbols, target_symbols, hold_days, spec_changed, stop_hit, spec)
        if not trade_now and industry_blocked_count > 0 and len(target_symbols) != len(current_symbols):
            trade_now = True
        if (
            not trade_now
            and G_MAX_POSITION_WEIGHT > 0
            and len(current_weights)
            and float(np.max(current_weights)) > G_MAX_POSITION_WEIGHT
        ):
            trade_now = True
        if (
            not trade_now
            and G_MAX_INDUSTRY_WEIGHT > 0
            and len(current_weights)
            and max_industry_exposure(
                current_symbols,
                current_weights,
                industry_labels_for_symbols(day, current_symbols),
            )
            > G_MAX_INDUSTRY_WEIGHT
        ):
            trade_now = True
        equity = marked_equity
        execution_equity = equity
        execution = empty_execution_result(
            equity,
            current_symbols,
            current_prev_open,
            current_entry_open,
            current_weights,
        )
        cost = 0.0
        if trade_now:
            execution, cost = execute_rebalance(
                day,
                current_symbols,
                current_entry_open,
                target_symbols,
                equity,
                current_weights,
                target_weight_override=target_weight_before_industry_filter,
            )
            equity *= 1.0 - cost
        net_ret = equity / equity_before - 1.0 if equity_before > 0 else 0.0
        peak = max(peak, equity)
        returns.append(net_ret)
        trades.append(execution.trade_count > 0)
        trade_counts.append(execution.trade_count)
        blocked_buy_counts.append(execution.blocked_buy_count)
        blocked_sell_counts.append(execution.blocked_sell_count)
        partial_buy_counts.append(execution.partial_buy_count)
        partial_sell_counts.append(execution.partial_sell_count)
        turnover_blocked_counts.append(execution.turnover_blocked_count)
        turnover_values.append(execution.turnover_value)
        unfilled_buy_values.append(execution.unfilled_buy_value)
        unfilled_sell_values.append(execution.unfilled_sell_value)
        turnover_blocked_values.append(execution.turnover_blocked_value)
        industry_blocked_counts.append(industry_blocked_count)

        if trade_now:
            current_symbols = execution.current_symbols
            current_prev_open = execution.current_prev_open
            current_entry_open = execution.current_entry_open
            current_weights = execution.current_weights
            current_spec = spec
            hold_days = 0
            if collect_rows:
                trade_event_rows.extend(execution.trade_events)
                score_map = {str(symbol): float(score) for symbol, score in zip(target_symbols, target_scores, strict=False)}
                current_industry_labels = industry_labels_for_symbols(day, current_symbols)
                for symbol, open_price, weight, industry_label in zip(
                    current_symbols,
                    current_prev_open,
                    current_weights,
                    current_industry_labels,
                    strict=False,
                ):
                    pick_rows.append(
                        {
                            "signal_date": day.signal_date.strftime("%Y-%m-%d"),
                            "entry_date": day.entry_date.strftime("%Y-%m-%d"),
                            "symbol": symbol,
                            "industry_label": str(industry_label),
                            "entry_open": float(open_price),
                            "weight": float(weight),
                            "score": score_map.get(str(symbol), np.nan),
                            "formula": spec.formula.name,
                        }
                    )

        is_active = len(current_symbols) > 0
        current_industry_labels = industry_labels_for_symbols(day, current_symbols)
        current_industry_exposure = industry_exposure_map(current_symbols, current_weights, current_industry_labels)
        active.append(is_active)
        if collect_rows:
            equity_rows.append(
                {
                    "signal_date": day.signal_date.strftime("%Y-%m-%d"),
                    "entry_date": day.entry_date.strftime("%Y-%m-%d"),
                    "year": day.signal_date.year,
                    "equity": float(equity),
                    "period_return": float(net_ret),
                    "drawdown": float(equity / peak - 1.0),
                    "trade": bool(trade_now),
                    "active": bool(is_active),
                    "positions": int(len(current_symbols)),
                    "max_position_weight": float(np.max(current_weights)) if len(current_weights) else 0.0,
                    "trade_count": int(execution.trade_count),
                    "blocked_buy_count": int(execution.blocked_buy_count),
                    "blocked_sell_count": int(execution.blocked_sell_count),
                    "partial_buy_count": int(execution.partial_buy_count),
                    "partial_sell_count": int(execution.partial_sell_count),
                    "industry_blocked_count": int(industry_blocked_count),
                    "turnover_blocked_count": int(execution.turnover_blocked_count),
                    "turnover_value": float(execution.turnover_value),
                    "turnover_pct": float(execution.turnover_value / max(execution_equity * G_INITIAL_CASH, 1.0)),
                    "unfilled_buy_value": float(execution.unfilled_buy_value),
                    "unfilled_sell_value": float(execution.unfilled_sell_value),
                    "turnover_blocked_value": float(execution.turnover_blocked_value),
                    "invested_weight": float(np.sum(current_weights)) if len(current_weights) else 0.0,
                    "cash_weight": float(max(1.0 - float(np.sum(current_weights)), 0.0)) if len(current_weights) else 1.0,
                    "max_industry_weight": float(max(current_industry_exposure.values()))
                    if current_industry_exposure
                    else 0.0,
                    "top_industry": max(current_industry_exposure, key=current_industry_exposure.get)
                    if current_industry_exposure
                    else None,
                    "portfolio_risk_off": bool(portfolio_risk_off),
                    "formula": spec.formula.name,
                    "market_filter": spec.market_filter,
                    "trend_filter": spec.trend_filter,
                    "min_hold_days": spec.min_hold_days,
                    "max_hold_days": spec.max_hold_days,
                    "replace_count": spec.replace_count,
                    "stop_loss": spec.stop_loss,
                }
            )

    return (
        np.asarray(returns, dtype=np.float32),
        np.asarray(active, dtype=bool),
        np.asarray(trades, dtype=bool),
        equity_rows,
        pick_rows,
        trade_event_rows,
    )


def build_series_for_spec(spec: DynamicSpec) -> DynamicSeries:
    returns, active, trades, _, _, _ = simulate_dynamic(G_DAYS, None, spec, G_MARKET)
    return DynamicSeries(spec=spec, returns=returns, active=active, trades=trades)



def training_score(row: dict, profile: str) -> float:
    return selection_training_score(row, profile)


def strict_spec_window_metrics(
    spec: DynamicSpec,
    days: list[CompactDayData],
    market: dict[str, set[pd.Timestamp]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict:
    returns, active, trades, _, _, _ = simulate_dynamic(
        days,
        None,
        spec,
        market,
        start_date=start_date,
        end_date=end_date,
        collect_rows=False,
    )
    entry_dates = np.asarray(
        [
            day.entry_date.to_datetime64()
            for day in days
            if start_date <= day.signal_date <= end_date
        ],
        dtype="datetime64[ns]",
    )
    if len(entry_dates) != len(returns):
        raise RuntimeError(
            f"window metric length mismatch for {spec.formula.name}: "
            f"entry_dates={len(entry_dates)} returns={len(returns)}"
        )
    mask = np.ones(len(returns), dtype=bool)
    return metrics_from_returns(returns, active, trades, entry_dates, mask)


def choose_specs_by_year(
    specs: list[DynamicSpec],
    days: list[CompactDayData],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_years: int,
    min_train_periods: int,
    workers: int,
    chunksize: int,
    keep_top: int,
    score_profile: str,
    freeze_selection_date: pd.Timestamp | None = None,
) -> tuple[dict[int, DynamicSpec], pd.DataFrame]:
    global G_DAYS, G_TARGETS
    G_DAYS = days
    G_TARGETS = build_target_cache(days, specs)
    context = mp.get_context("fork")
    series_list: list[DynamicSeries] = []
    with ProcessPoolExecutor(max_workers=max(workers, 1), mp_context=context) as executor:
        for series in executor.map(build_series_for_spec, specs, chunksize=max(chunksize, 1)):
            series_list.append(series)

    signal_dates_ = np.asarray([day.signal_date.to_datetime64() for day in days], dtype="datetime64[ns]")
    entry_dates = np.asarray([day.entry_date.to_datetime64() for day in days], dtype="datetime64[ns]")

    def validation_metrics_for_spec(spec: DynamicSpec) -> dict:
        return strict_spec_window_metrics(spec, days, G_MARKET, start_date, end_date)

    return choose_yearly_specs_from_series(
        series_list=series_list,
        signal_dates=signal_dates_,
        entry_dates=entry_dates,
        start_date=start_date,
        end_date=end_date,
        train_years=train_years,
        min_train_periods=min_train_periods,
        keep_top=keep_top,
        score_profile=score_profile,
        freeze_selection_date=freeze_selection_date,
        validation_metrics_func=validation_metrics_for_spec,
    )

def run_walkforward(
    days: list[CompactDayData],
    yearly_specs: dict[int, DynamicSpec],
    market: dict[str, set[pd.Timestamp]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    returns, active, trades, equity_rows, pick_rows, trade_event_rows = simulate_dynamic(
        days,
        yearly_specs,
        None,
        market,
        start_date=start_date,
        end_date=end_date,
        collect_rows=True,
    )
    return summarize_walkforward_result(
        returns=returns,
        active=active,
        trades=trades,
        equity_rows=equity_rows,
        pick_rows=pick_rows,
        trade_event_rows=trade_event_rows,
        initial_cash=initial_cash,
    )


def output_prefix(
    args: argparse.Namespace,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fixed_spec_data: dict[str, Any] | None,
) -> Path:
    run_suffix = ""
    if fixed_spec_data:
        fixed_payload = json.dumps(fixed_spec_data, sort_keys=True, ensure_ascii=False)
        run_suffix = f"_fixed{hashlib.sha1(fixed_payload.encode('utf-8')).hexdigest()[:10]}"
    if args.skip_capacity_stress:
        run_suffix += "_nostress"
    return args.output_dir / (
        f"dynamic_rebalance_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        f"_train{args.train_years}y_top{args.top_n}_{args.score_profile}_{args.formula_set}_{args.formula_scope}"
        f"_{args.grid_profile}_{args.board_scope}_{args.factor_adjust}"
        f"_cap{args.capacity_equity_mode}"
        f"_pstop{args.portfolio_stop_loss:g}"
        f"_{'sector' if args.industry_source else 'nosector'}"
        f"{run_suffix}"
    )


def compute_equal_weight_benchmark_from_db(
    db: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    board_scope: str,
    name: str,
) -> dict:
    board_clause, board_params = board_scope_sql(board_scope, "d")
    lookback_start = start_date - pd.Timedelta(days=10)
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(
            f"""
            select d.symbol, d.trade_date, d.close as raw_close
            from daily_bars d
            where d.adjust = 'raw'
              and d.trade_date >= ?
              and d.trade_date <= ?
              and d.close is not null
              {board_clause}
            order by d.symbol, d.trade_date
            """,
            conn,
            params=(lookback_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), *board_params),
        )
    return compute_equal_weight_benchmark(df, start_date, end_date, name)


def compute_benchmark_suite(db: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
    return {
        "all_equal_weight_raw_close": compute_equal_weight_benchmark_from_db(
            db,
            start_date,
            end_date,
            "all",
            "all_equal_weight_raw_close",
        ),
        "main_board_equal_weight_raw_close": compute_equal_weight_benchmark_from_db(
            db,
            start_date,
            end_date,
            "main",
            "main_board_equal_weight_raw_close",
        ),
    }




def run_capacity_stress(
    args: argparse.Namespace,
    days: list[CompactDayData],
    yearly_specs: dict[int, DynamicSpec],
    market: dict[str, set[pd.Timestamp]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Replay the already-selected strategy over the declared capacity/slippage grid."""
    global G_EXECUTION, G_INITIAL_CASH
    plan = capacity_stress_plan(args)
    grid = plan["recommended_grid"]
    original_execution = G_EXECUTION
    original_initial_cash = G_INITIAL_CASH
    rows: list[dict] = []
    try:
        for initial_cash in grid["initial_cash"]:
            for capacity_pct in grid["capacity_pct_of_amount"]:
                for slippage_bps in grid["slippage_bps"]:
                    for impact_bps in grid["impact_bps_per_pct_amount"]:
                        G_INITIAL_CASH = float(initial_cash)
                        G_EXECUTION = ExecutionConfig(
                            buy_cost=original_execution.buy_cost,
                            sell_cost=original_execution.sell_cost,
                            slippage_bps=float(slippage_bps),
                            impact_bps_per_pct_amount=float(impact_bps),
                            capacity_pct_of_amount=float(capacity_pct),
                            capacity_equity_mode=original_execution.capacity_equity_mode,
                            lot_size=original_execution.lot_size,
                            limit_epsilon=original_execution.limit_epsilon,
                            block_limit_trades=original_execution.block_limit_trades,
                        )
                        _, picks_df, trade_log, stress_metrics = run_walkforward(
                            days,
                            yearly_specs,
                            market,
                            start_date,
                            end_date,
                            float(initial_cash),
                        )
                        rows.append(
                            build_capacity_stress_row(
                                initial_cash=float(initial_cash),
                                capacity_pct=float(capacity_pct),
                                slippage_bps=float(slippage_bps),
                                impact_bps=float(impact_bps),
                                capacity_equity_mode=original_execution.capacity_equity_mode,
                                current=plan["current"],
                                stress_metrics=stress_metrics,
                                picks=picks_df,
                                trade_log=trade_log,
                            )
                        )
    finally:
        G_EXECUTION = original_execution
        G_INITIAL_CASH = original_initial_cash

    stress_df = pd.DataFrame(rows)
    return stress_df, mark_capacity_stress_replayed(plan, stress_df)


def main() -> int:
    global G_DAYS, G_FEATURE_NAMES, G_MARKET, G_ROUND_TRIP_COST, G_EXECUTION, G_INITIAL_CASH
    global G_TARGETS
    global G_MAX_POSITION_WEIGHT, G_MAX_INDUSTRY_WEIGHT, G_MAX_TURNOVER_PCT
    global G_BLACKLIST
    global G_PORTFOLIO_STOP_LOSS, G_PORTFOLIO_REENTRY_FILTER
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        db_end = pd.Timestamp(
            conn.execute("select max(trade_date) from daily_bars where adjust = 'raw'").fetchone()[0]
        )
    start_date = parse_date(args.start_date)
    end_date = min(parse_date(args.end_date), db_end) if args.end_date else db_end
    research_start = start_date - pd.DateOffset(years=args.train_years) + pd.Timedelta(days=1)
    coverage_start = research_start - pd.Timedelta(days=460)
    freeze_selection_date = parse_date(args.freeze_selection_date) if args.freeze_selection_date else None
    split_policy = parse_json_metadata(args.split_policy_json, "split_policy") or default_split_policy(
        start_date,
        end_date,
        freeze_selection_date,
    )
    frozen_config = parse_json_metadata(args.frozen_config_json, "frozen_config")
    fixed_spec_data = parse_json_metadata(args.fixed_spec_json, "fixed_spec_json")
    formal_required_adjusts = parse_required_adjusts(args.formal_required_adjusts)
    quality_report: dict | None = None
    if args.formal:
        if not args.split_policy_json:
            raise RuntimeError("--formal requires --split-policy-json so reports carry fixed train/validation/test periods")
        if not args.frozen_config_json:
            raise RuntimeError("--formal requires --frozen-config-json so out-of-sample runs declare frozen parameters")
        if freeze_selection_date is None:
            raise RuntimeError("--formal requires --freeze-selection-date")
        quality_report = build_quality_report(
            args.db,
            coverage_start,
            end_date,
            args.board_scope,
            formal_required_adjusts,
        )
        if quality_report.get("red_flags"):
            raise RuntimeError(
                "formal data quality gate failed: "
                + "; ".join(str(flag) for flag in quality_report["red_flags"][:20])
            )
    G_ROUND_TRIP_COST = args.round_trip_cost
    G_INITIAL_CASH = args.initial_cash
    G_MAX_POSITION_WEIGHT = args.max_position_weight
    G_MAX_INDUSTRY_WEIGHT = args.max_industry_weight
    G_MAX_TURNOVER_PCT = args.max_turnover_pct
    G_BLACKLIST = load_blacklist(args.blacklist_file)
    G_PORTFOLIO_STOP_LOSS = args.portfolio_stop_loss
    G_PORTFOLIO_REENTRY_FILTER = args.portfolio_reentry_filter
    G_EXECUTION = ExecutionConfig(
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        slippage_bps=args.slippage_bps,
        impact_bps_per_pct_amount=args.impact_bps_per_pct_amount,
        capacity_pct_of_amount=args.capacity_pct_of_amount,
        capacity_equity_mode=args.capacity_equity_mode,
        lot_size=args.lot_size,
        limit_epsilon=args.limit_epsilon,
        block_limit_trades=not args.no_limit_block,
    )
    with sqlite3.connect(args.db) as conn:
        factor_adjust_coverage = adjustment_coverage(
            conn,
            coverage_start,
            end_date,
            args.board_scope,
            args.factor_adjust,
        )
    fixed_spec = spec_from_config(fixed_spec_data) if fixed_spec_data else None
    specs = (
        [fixed_spec]
        if fixed_spec is not None
        else spec_grid(
            args.formula_set,
            args.formula_scope,
            args.top_n,
            args.grid_profile,
            args.min_hold_days,
            args.max_hold_days,
            args.stop_loss if args.stop_loss is not None else "grid",
        )
    )
    feature_names = feature_names_for_specs(specs)
    G_FEATURE_NAMES = feature_names

    print(
        f"loading {research_start.date()} -> {end_date.date()} "
        f"(dynamic test starts {start_date.date()}, features={len(feature_names)})",
        flush=True,
    )
    df = load_or_build_dynamic_factors(
        args.db,
        args.cache_dir,
        research_start,
        end_date,
        feature_names,
        use_cache=not args.no_cache,
        board_scope=args.board_scope,
        factor_adjust=args.factor_adjust,
        allow_factor_fallback=not args.strict_factor_adjust,
    )
    df, industry_coverage = apply_industry_labels(df, load_symbol_industry_map(args.db))
    df = add_industry_relative_features(df, feature_names)
    df, constraint_summary = enrich_backtest_constraints(args.db, df, args.board_scope)
    benchmarks = compute_benchmark_suite(args.db, start_date, end_date)
    all_dates = pd.DatetimeIndex(sorted(df["trade_date"].unique()))
    raw_open_price = df.pivot(index="trade_date", columns="symbol", values="raw_open").sort_index()
    raw_high_price = df.pivot(index="trade_date", columns="symbol", values="raw_high").sort_index()
    raw_low_price = df.pivot(index="trade_date", columns="symbol", values="raw_low").sort_index()
    raw_amount = df.pivot(index="trade_date", columns="symbol", values="amount").sort_index()
    raw_volume = df.pivot(index="trade_date", columns="symbol", values="raw_volume").sort_index()
    status_is_st = df.pivot(index="trade_date", columns="symbol", values="is_st").sort_index()
    status_suspended = df.pivot(index="trade_date", columns="symbol", values="is_suspended").sort_index()
    limit_rates = df.pivot(index="trade_date", columns="symbol", values="limit_rate").sort_index()
    signal_allowed = df.pivot(index="trade_date", columns="symbol", values="signal_allowed").sort_index()
    factor_close_price = df.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    market = market_states(factor_close_price)
    industry_close = load_industry_close(args.db, research_start, end_date, args.industry_source)
    industry_states = industry_market_states(industry_close)
    market.update(industry_states)
    valuation = load_market_valuation(args.db, research_start, end_date)
    valuation_states = market_valuation_states(valuation)
    market.update(valuation_states)
    G_MARKET = market
    days = build_compact_day_data(
        df,
        raw_open_price,
        raw_high_price,
        raw_low_price,
        raw_amount,
        raw_volume,
        status_is_st,
        status_suspended,
        limit_rates,
        signal_allowed,
        signal_dates(all_dates, research_start, end_date, "D"),
        all_dates,
        feature_names,
    )
    factor_adjust_used = sorted(str(value) for value in df["factor_adjust_used"].dropna().unique())
    universe_symbols = int(df["symbol"].nunique())
    valuation_coverage = {
        "rows": int(len(valuation)),
        "min_trade_date": valuation["trade_date"].min().strftime("%Y-%m-%d") if not valuation.empty else None,
        "max_trade_date": valuation["trade_date"].max().strftime("%Y-%m-%d") if not valuation.empty else None,
        "states": {name: int(len(dates)) for name, dates in valuation_states.items()},
        "percentile_rule": "rolling 756 trading days, minimum 252; current signal date and earlier only",
    }
    del (
        df,
        raw_open_price,
        raw_high_price,
        raw_low_price,
        raw_amount,
        raw_volume,
        status_is_st,
        status_suspended,
        limit_rates,
        signal_allowed,
        factor_close_price,
        industry_close,
    )
    gc.collect()
    print({"D": len(days)}, flush=True)
    print(
        f"dynamic selecting among {len(specs)} specs, top_n={args.top_n}, "
        f"score={args.score_profile}, workers={args.workers}",
        flush=True,
    )
    if fixed_spec is not None:
        yearly_specs = {year: fixed_spec for year in range(start_date.year, end_date.year + 1)}
        diagnostics = pd.DataFrame(
            [
                {
                    "year": year,
                    "status": "fixed_spec_replay",
                    **spec_to_row(fixed_spec),
                }
                for year in yearly_specs
            ]
        )
        G_DAYS = days
        G_TARGETS = build_target_cache(days, specs)
    else:
        yearly_specs, diagnostics = choose_specs_by_year(
            specs,
            days,
            start_date,
            end_date,
            args.train_years,
            args.min_train_periods,
            args.workers,
            args.chunksize,
            args.keep_top,
            args.score_profile,
            freeze_selection_date,
        )
    if not yearly_specs:
        prefix = output_prefix(args, start_date, end_date, fixed_spec_data)
        empty_equity = pd.DataFrame(columns=["date", "equity", "period_return", "active", "trade"])
        empty_picks = pd.DataFrame()
        empty_trades = pd.DataFrame()
        empty_capacity = pd.DataFrame()
        metrics = {
            "status": "failed_no_selected_specs",
            "failure_reason": "selection produced no valid yearly specs under the configured training-only score profile",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "annual_return": None,
            "total_return": None,
            "max_drawdown": None,
            "multiple_testing": multiple_testing_summary(
                specs,
                diagnostics,
                args.formula_set,
                args.formula_scope,
                args.grid_profile,
                args.score_profile,
            ),
            "data_quality_red_flags": list(quality_report.get("red_flags", [])) if quality_report else [],
            "split_policy": split_policy,
            "frozen_config": frozen_config,
            "config": {
                "strategy": "dynamic_daily_checked_rebalance",
                "train_years": args.train_years,
                "min_train_periods": args.min_train_periods,
                "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d")
                if freeze_selection_date is not None
                else None,
                "top_n": args.top_n,
                "score_profile": args.score_profile,
                "formula_set": args.formula_set,
                "formula_scope": args.formula_scope,
                "grid_profile": args.grid_profile,
                "board_scope": args.board_scope,
                "factor_adjust": args.factor_adjust,
                "factor_adjust_used": factor_adjust_used,
                "factor_adjust_coverage": factor_adjust_coverage,
                "factor_adjust_fallback": not args.strict_factor_adjust,
                "universe_symbols": universe_symbols,
                "constraint_summary": constraint_summary,
                "industry_coverage": industry_coverage,
                "market_valuation_coverage": valuation_coverage,
                "industry_source": args.industry_source,
                "industry_filters": sorted(industry_states.keys()),
                "valuation_filters": sorted(valuation_states.keys()),
                "cache_enabled": not args.no_cache,
                "selection_rule": "each calendar year uses only prior completed training years",
            },
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        empty_equity.to_csv(prefix.with_suffix(".equity.csv"), index=False)
        empty_picks.to_csv(prefix.with_suffix(".picks.csv"), index=False)
        empty_trades.to_csv(prefix.with_suffix(".trades.csv"), index=False)
        empty_capacity.to_csv(prefix.with_suffix(".capacity_stress.csv"), index=False)
        diagnostics.to_csv(prefix.with_suffix(".diagnostics.csv"), index=False)
        metrics_path = prefix.with_suffix(".metrics.json")
        manifest_path = prefix.with_suffix(".manifest.json")
        quality_report_path = prefix.with_suffix(".data_quality.json") if quality_report else None
        if quality_report_path is not None:
            with quality_report_path.open("w", encoding="utf-8") as fh:
                json.dump(quality_report, fh, ensure_ascii=False, indent=2)
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        write_manifest(
            manifest_path,
            collect_manifest(
                args.db,
                sys.argv,
                [
                    Path(__file__),
                    Path(__file__).with_name("backtest_walkforward_no_lookahead.py"),
                    Path(__file__).with_name("quant_data_quality.py"),
                    Path(__file__).with_name("quant_universe.py"),
                    Path(__file__).with_name("run_manifest.py"),
                ],
                {
                    "equity": prefix.with_suffix(".equity.csv"),
                    "picks": prefix.with_suffix(".picks.csv"),
                    "trades": prefix.with_suffix(".trades.csv"),
                    "capacity_stress": prefix.with_suffix(".capacity_stress.csv"),
                    "diagnostics": prefix.with_suffix(".diagnostics.csv"),
                    "metrics": metrics_path,
                    "manifest": manifest_path,
                    **({"data_quality": quality_report_path} if quality_report_path is not None else {}),
                },
                {
                    "strategy": metrics["config"]["strategy"],
                    "start_date": metrics["start_date"],
                    "end_date": metrics["end_date"],
                    "factor_adjust": args.factor_adjust,
                    "strict_factor_adjust": args.strict_factor_adjust,
                    "is_formal_valid": False,
                    "split_policy": split_policy,
                    "frozen_config": frozen_config,
                    "industry_source": args.industry_source,
                    "max_industry_weight": args.max_industry_weight,
                    "capacity_equity_mode": args.capacity_equity_mode,
                },
            ),
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return 0
    equity, picks, trade_log, metrics = run_walkforward(days, yearly_specs, market, start_date, end_date, args.initial_cash)
    metrics["professional_metrics"] = professional_performance_metrics(equity, args.initial_cash)
    metrics["professional_metrics"]["relative_to_benchmarks"] = {
        name: relative_performance_metrics(equity, benchmark)
        for name, benchmark in benchmarks.items()
        if benchmark
    }
    metrics["annual_breakdown"] = period_return_breakdown(equity, "Y")
    metrics["monthly_breakdown"] = period_return_breakdown(equity, "M")
    metrics["benchmarks"] = benchmarks
    metrics["benchmark"] = benchmarks.get(
        "main_board_equal_weight_raw_close" if args.board_scope == "main" else "all_equal_weight_raw_close",
        {},
    )
    metrics["multiple_testing"] = multiple_testing_summary(
        specs,
        diagnostics,
        args.formula_set,
        args.formula_scope,
        args.grid_profile,
        args.score_profile,
    )
    if args.skip_capacity_stress:
        capacity_stress = pd.DataFrame()
        capacity_stress_meta = {
            "status": "skipped_research_run",
            "next_step": "rerun without --skip-capacity-stress before promoting this result",
        }
    else:
        capacity_stress, capacity_stress_meta = run_capacity_stress(
            args,
            days,
            yearly_specs,
            market,
            start_date,
            end_date,
        )
    metrics["capacity_stress"] = capacity_stress_meta
    metrics["risk_budget"] = risk_budget_report(
        metrics,
        equity,
        picks,
        {
            "portfolio_stop_loss": args.portfolio_stop_loss,
            "max_position_weight": args.max_position_weight,
            "max_industry_weight": args.max_industry_weight,
            "capacity_pct_of_amount": args.capacity_pct_of_amount,
            "max_turnover_pct": args.max_turnover_pct,
        },
    )
    quality_red_flags = list(quality_report.get("red_flags", [])) if quality_report else []
    status_coverage_payload = {}
    if quality_report:
        status_coverage_payload = quality_report.get("symbol_status_daily", {}).get("row_coverage", {})
    metrics["is_formal_valid"] = bool(
        args.formal
        and args.factor_adjust != "raw"
        and args.strict_factor_adjust
        and not quality_red_flags
        and metrics.get("date_validation", {}).get("signal_before_entry", False)
    )
    metrics["data_quality_red_flags"] = quality_red_flags
    metrics["status_coverage"] = {
        "quality_report": status_coverage_payload,
        "backtest_constraints": constraint_summary,
    }
    metrics["split_policy"] = split_policy
    metrics["frozen_config"] = frozen_config
    metrics["config"] = {
        "strategy": "dynamic_daily_checked_rebalance",
        "formal": args.formal,
        "formal_required_adjusts": list(formal_required_adjusts),
        "train_years": args.train_years,
        "min_train_periods": args.min_train_periods,
        "freeze_selection_date": freeze_selection_date.strftime("%Y-%m-%d") if freeze_selection_date is not None else None,
        "top_n": args.top_n,
        "score_profile": args.score_profile,
        "formula_set": args.formula_set,
        "formula_scope": args.formula_scope,
        "grid_profile": args.grid_profile,
        "min_hold_days_override": args.min_hold_days,
        "max_hold_days_override": args.max_hold_days,
        "stop_loss_override": args.stop_loss,
        "fixed_spec": spec_to_row(fixed_spec) if fixed_spec is not None else None,
        "board_scope": args.board_scope,
        "factor_adjust": args.factor_adjust,
        "factor_adjust_used": factor_adjust_used,
        "factor_adjust_coverage": factor_adjust_coverage,
        "factor_adjust_fallback": not args.strict_factor_adjust,
        "universe_symbols": universe_symbols,
        "constraint_summary": constraint_summary,
        "industry_coverage": industry_coverage,
        "market_valuation_coverage": valuation_coverage,
        "industry_source": args.industry_source,
        "industry_filters": sorted(industry_states.keys()),
        "valuation_filters": sorted(valuation_states.keys()),
        "blacklist_file": str(args.blacklist_file) if args.blacklist_file else None,
        "blacklist_symbols": int(len(G_BLACKLIST)),
        "round_trip_cost": args.round_trip_cost,
        "execution": {
            "buy_cost": args.buy_cost,
            "sell_cost": args.sell_cost,
            "slippage_bps": args.slippage_bps,
            "impact_bps_per_pct_amount": args.impact_bps_per_pct_amount,
            "capacity_pct_of_amount": args.capacity_pct_of_amount,
            "capacity_equity_mode": args.capacity_equity_mode,
            "lot_size": args.lot_size,
            "limit_epsilon": args.limit_epsilon,
            "block_limit_trades": not args.no_limit_block,
            "portfolio_stop_loss": args.portfolio_stop_loss,
            "portfolio_reentry_filter": args.portfolio_reentry_filter,
            "max_position_weight": args.max_position_weight,
            "max_industry_weight": args.max_industry_weight,
            "max_turnover_pct": args.max_turnover_pct,
        },
        "cache_enabled": not args.no_cache,
        "selection_rule": "each calendar year uses only prior completed training years",
        "execution_rule": "daily close decision; trade next trading day open only when dynamic trigger fires",
        "bias_controls": [
            "main-board stock universe by symbol prefix" if args.board_scope == "main" else "all loaded A-share symbols",
            "signal candidates exclude historical ST rows and symbols outside lifecycle date bounds",
            "all rolling features use signal-date and earlier bars",
            (
                "out-of-sample years reuse the frozen pre-test selection"
                if freeze_selection_date is not None
                else "yearly parameter selection does not use the test year"
            ),
            "raw prices for execution; configured adjusted prices for factors with optional raw fallback",
            "next-open execution blocks suspended entries/exits and locked limit-up buys/locked limit-down sells",
            "price-limit bands use ST 5%, ChiNext/STAR 20%, and main-board 10%",
            "execution applies slippage, capacity checks, and 100-share lot-size minimum",
            "optional max single-name weight cap trims oversized positions" if args.max_position_weight > 0 else "no explicit max single-name weight cap configured",
            "optional max industry weight cap skips excess same-industry candidates"
            if args.max_industry_weight > 0
            else "no explicit max industry weight cap configured",
            "optional buy blacklist skips configured forbidden symbols" if G_BLACKLIST else "no explicit buy blacklist configured",
            "optional period turnover cap blocks or partially fills excess turnover"
            if args.max_turnover_pct > 0
            else "no explicit period turnover cap configured",
        ],
    }
    metrics["generated_at"] = datetime.now().isoformat(timespec="seconds")

    prefix = output_prefix(args, start_date, end_date, fixed_spec_data)
    equity.to_csv(prefix.with_suffix(".equity.csv"), index=False)
    picks.to_csv(prefix.with_suffix(".picks.csv"), index=False)
    trade_log.to_csv(prefix.with_suffix(".trades.csv"), index=False)
    capacity_stress.to_csv(prefix.with_suffix(".capacity_stress.csv"), index=False)
    diagnostics.to_csv(prefix.with_suffix(".diagnostics.csv"), index=False)
    metrics_path = prefix.with_suffix(".metrics.json")
    manifest_path = prefix.with_suffix(".manifest.json")
    quality_report_path = prefix.with_suffix(".data_quality.json") if quality_report else None
    if quality_report_path is not None:
        with quality_report_path.open("w", encoding="utf-8") as fh:
            json.dump(quality_report, fh, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    write_manifest(
        manifest_path,
        collect_manifest(
            args.db,
            sys.argv,
            [
                Path(__file__),
                Path(__file__).with_name("backtest_walkforward_no_lookahead.py"),
                Path(__file__).with_name("quant_data_quality.py"),
                Path(__file__).with_name("quant_universe.py"),
                Path(__file__).with_name("run_manifest.py"),
            ],
            {
                "equity": prefix.with_suffix(".equity.csv"),
                "picks": prefix.with_suffix(".picks.csv"),
                "trades": prefix.with_suffix(".trades.csv"),
                "capacity_stress": prefix.with_suffix(".capacity_stress.csv"),
                "diagnostics": prefix.with_suffix(".diagnostics.csv"),
                "metrics": metrics_path,
                "manifest": manifest_path,
                **({"data_quality": quality_report_path} if quality_report_path is not None else {}),
            },
            {
                "strategy": metrics["config"]["strategy"],
                "start_date": metrics["start_date"],
                "end_date": metrics["end_date"],
                "factor_adjust": args.factor_adjust,
                "strict_factor_adjust": args.strict_factor_adjust,
                "is_formal_valid": metrics["is_formal_valid"],
                "split_policy": split_policy,
                "frozen_config": frozen_config,
                "industry_source": args.industry_source,
                "max_industry_weight": args.max_industry_weight,
                "capacity_equity_mode": args.capacity_equity_mode,
            },
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

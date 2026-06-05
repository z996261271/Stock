import sqlite3
import subprocess
import argparse
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_dynamic_rebalance import (  # noqa: E402
    CompactDayData,
    DynamicSpec,
    capacity_stress_plan,
    compute_equal_weight_benchmark,
    default_execution_config,
    enrich_backtest_constraints,
    execute_rebalance,
    filter_target_by_industry_weight,
    industry_exposure_map,
    limit_rate_for_symbol,
    limit_trade_masks,
    multiple_testing_summary,
    period_return_breakdown,
    professional_performance_metrics,
    risk_budget_report,
    relative_performance_metrics,
    run_capacity_stress,
    select_target_from_score,
    simulate_dynamic,
    target_key,
)
import backtest_dynamic_rebalance as dynamic  # noqa: E402
import state_report as state_report_module  # noqa: E402
from fetch_akshare_daily import (  # noqa: E402
    build_jobs as build_daily_jobs,
    init_db as init_fetch_db,
    refresh_adj_factors_from_daily,
    update_status as update_fetch_status,
)
from fetch_symbol_lifecycle import save_lifecycle  # noqa: E402
from fetch_symbol_status_daily import save_status as save_symbol_status  # noqa: E402
from backtest_walkforward_no_lookahead import load_data  # noqa: E402
from quant_data_quality import build_quality_report  # noqa: E402
from quant_schema import REFERENCE_TABLES, STATE_TABLES, ensure_quant_schema  # noqa: E402
from quant_state import (  # noqa: E402
    record_alert,
    record_alert_attempt,
    record_paper_trade,
    record_position,
    record_signal,
    record_signal_run,
    state_counts,
)
from generate_daily_signals import build_plan, write_plan  # noqa: E402
from generate_daily_from_picks import build_rows_from_picks  # noqa: E402
from import_symbol_industries import main as import_symbol_industries_main  # noqa: E402
from professional_validation_report import build_report as build_professional_validation_report  # noqa: E402
from run_sensitivity_matrix import build_matrix, matrix_plan_rows  # noqa: E402
from validate_formal_reports import build_report as build_formal_report_validation  # noqa: E402
from quant_universe import is_main_board_symbol  # noqa: E402
from run_adjust_backfill_batches import remaining_symbols as remaining_adjust_symbols  # noqa: E402
from run_manifest import collect_manifest, write_manifest  # noqa: E402
from backup_sqlite import backup_sqlite  # noqa: E402
from professional_quant.data.industry import load_symbol_industry_map  # noqa: E402


def _insert_bar(conn, symbol, trade_date, adjust, open_, high, low, close, amount=100_000_000):
    conn.execute(
        """
        insert into daily_bars (
            symbol, trade_date, adjust, open, high, low, close, volume, amount,
            amplitude, pct_chg, chg, turnover, source, fetched_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, trade_date, adjust, open_, high, low, close, 1_000_000, amount, 0, 0, 0, 1, "test", "now"),
    )


def test_main_board_symbol_filter():
    assert is_main_board_symbol("000001")
    assert is_main_board_symbol("600519")
    assert is_main_board_symbol("002415")
    assert not is_main_board_symbol("300750")
    assert not is_main_board_symbol("688001")
    assert not is_main_board_symbol("430001")


def test_load_data_uses_raw_execution_and_hfq_factors(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        create table daily_bars (
            symbol text not null,
            trade_date text not null,
            adjust text not null,
            open real, high real, low real, close real, volume real, amount real,
            amplitude real, pct_chg real, chg real, turnover real,
            source text not null, fetched_at text not null,
            primary key(symbol, trade_date, adjust)
        )
        """
    )
    conn.execute("create table symbols (symbol text primary key, name text)")
    conn.execute("insert into symbols values ('000001', 'main')")
    conn.execute("insert into symbols values ('300001', 'growth')")
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-02", "hfq", 100, 110, 90, 105)
    _insert_bar(conn, "300001", "2020-01-02", "raw", 20, 21, 19, 20)
    _insert_bar(conn, "300001", "2020-01-02", "hfq", 200, 210, 190, 205)
    conn.commit()
    conn.close()

    df = load_data(
        db,
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
        board_scope="main",
        factor_adjust="hfq",
        allow_factor_fallback=False,
    )

    assert df["symbol"].tolist() == ["000001"]
    row = df.iloc[0]
    assert row["raw_open"] == 10
    assert row["raw_close"] == 10
    assert row["open"] == 100
    assert row["close"] == 105
    assert row["factor_adjust_used"] == "hfq"


def test_strict_factor_adjust_rejects_partial_missing_rows(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        create table daily_bars (
            symbol text not null,
            trade_date text not null,
            adjust text not null,
            open real, high real, low real, close real, volume real, amount real,
            amplitude real, pct_chg real, chg real, turnover real,
            source text not null, fetched_at text not null,
            primary key(symbol, trade_date, adjust)
        )
        """
    )
    conn.execute("create table symbols (symbol text primary key, name text)")
    conn.execute("insert into symbols values ('000001', 'has_hfq')")
    conn.execute("insert into symbols values ('000002', 'missing_hfq')")
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-02", "hfq", 100, 110, 90, 105)
    _insert_bar(conn, "000002", "2020-01-02", "raw", 20, 21, 19, 20)
    conn.commit()
    conn.close()

    fallback = load_data(
        db,
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
        board_scope="main",
        factor_adjust="hfq",
        allow_factor_fallback=True,
    )
    assert sorted(fallback["factor_adjust_used"].unique()) == ["hfq", "raw"]

    try:
        load_data(
            db,
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-02"),
            board_scope="main",
            factor_adjust="hfq",
            allow_factor_fallback=False,
        )
    except RuntimeError as exc:
        assert "strict factor-adjust validation failed" in str(exc)
        assert "missing_raw_rows=1" in str(exc)
    else:
        raise AssertionError("strict hfq load should fail when any raw row lacks hfq")


def test_daily_fetch_jobs_backfill_before_and_after_existing_range(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    _insert_bar(conn, "000001", "2020-01-01", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-10", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-05", "hfq", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-06", "hfq", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-07", "hfq", 10, 11, 9, 10)
    conn.commit()

    jobs, skipped = build_daily_jobs(
        conn,
        ["000001"],
        ["hfq"],
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-10").date(),
        full_refresh=False,
        progress_every=0,
    )

    assert skipped == 0
    assert [(job[4].isoformat(), job[5].isoformat()) for job in jobs] == [
        ("2020-01-01", "2020-01-04"),
        ("2020-01-08", "2020-01-10"),
    ]
    conn.close()


def test_adjusted_fetch_jobs_use_raw_listing_window(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    for trade_date in ["2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08"]:
        _insert_bar(conn, "000001", trade_date, "raw", 10, 11, 9, 10)
    for trade_date in ["2020-01-05", "2020-01-06", "2020-01-07"]:
        _insert_bar(conn, "000001", trade_date, "qfq", 10, 11, 9, 10)
    conn.commit()

    jobs, skipped = build_daily_jobs(
        conn,
        ["000001"],
        ["qfq"],
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-10").date(),
        full_refresh=False,
        progress_every=0,
    )

    assert skipped == 0
    assert [(job[4].isoformat(), job[5].isoformat()) for job in jobs] == [("2020-01-08", "2020-01-08")]
    conn.close()


def test_adjust_backfill_remaining_symbols_uses_daily_bar_coverage_not_status_rows(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    for trade_date in ["2020-01-05", "2020-01-06"]:
        _insert_bar(conn, "000001", trade_date, "raw", 10, 11, 9, 10)
        _insert_bar(conn, "000001", trade_date, "qfq", 10, 11, 9, 10)
    _insert_bar(conn, "000002", "2020-01-05", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000002", "2020-01-06", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000002", "2020-01-05", "qfq", 10, 11, 9, 10)
    update_fetch_status(
        conn,
        "000001",
        "qfq",
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-04").date(),
        "ok",
        0,
        "unit.source",
        "empty pre-listing range",
    )
    conn.commit()
    conn.close()

    assert remaining_adjust_symbols(db, "qfq", "20200101", "20200110") == ["000002"]


def test_adj_factors_are_derived_from_raw_qfq_hfq(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-02", "qfq", 8, 8.8, 7.2, 8)
    _insert_bar(conn, "000001", "2020-01-02", "hfq", 20, 22, 18, 25)
    conn.commit()

    refresh_adj_factors_from_daily(
        conn,
        "000001",
        pd.Timestamp("2020-01-02").date(),
        pd.Timestamp("2020-01-02").date(),
    )

    row = conn.execute(
        """
        select round(forward_factor, 4), round(backward_factor, 4), round(adj_factor, 4), source
        from adj_factors
        where symbol = '000001' and trade_date = '2020-01-02'
        """
    ).fetchone()
    assert row[:3] == (0.8, 2.5, 2.5)
    assert "test" in row[3]
    conn.close()


def test_fetch_status_source_used_column_is_migrated(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)

    update_fetch_status(
        conn,
        "000001",
        "hfq",
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-10").date(),
        "ok",
        7,
        "unit.source",
        "source=unit.source",
    )

    row = conn.execute(
        "select source_used, message from fetch_status where symbol='000001' and adjust='hfq'"
    ).fetchone()
    assert row == ("unit.source", "source=unit.source")
    conn.close()


def test_lifecycle_and_status_savers_populate_reference_tables(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)

    saved_lifecycle = save_lifecycle(
        conn,
        [
            {
                "symbol": "000001",
                "name": "平安银行",
                "list_date": "1991-04-03",
                "delist_date": None,
                "board": "主板",
                "market": "sz",
                "source": "unit",
            },
            {
                "symbol": "000003",
                "name": "PT金田Ａ",
                "list_date": "1991-01-14",
                "delist_date": "2002-06-14",
                "board": "主板",
                "market": "sz",
                "source": "unit",
            },
        ],
    )
    saved_status = save_symbol_status(
        conn,
        pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "trade_date": "2020-01-02",
                    "is_st": 1,
                    "is_suspended": 0,
                    "board": "主板",
                    "source": "unit",
                    "fetched_at": "now",
                }
            ]
        ),
    )

    assert saved_lifecycle == 2
    assert saved_status == 1
    assert conn.execute("select count(*) from symbol_lifecycle").fetchone()[0] == 2
    assert conn.execute("select is_st from symbol_status_daily where symbol='000001'").fetchone()[0] == 1
    conn.close()

    report = build_quality_report(
        db,
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
        board_scope="main",
        required_adjusts=("raw",),
    )
    assert report["symbol_status_daily"]["expected_symbols_with_raw"] == 1
    assert report["symbol_status_daily"]["covered_symbols_with_status"] == 1
    assert report["symbol_status_daily"]["missing_symbols_with_status"] == 0
    assert not any(flag.startswith("status_symbols_incomplete") for flag in report["red_flags"])


def test_symbol_industry_import_and_quality_coverage(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    ensure_quant_schema(conn)
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    save_lifecycle(
        conn,
        [
            {
                "symbol": "000001",
                "name": "平安银行",
                "list_date": "1991-04-03",
                "delist_date": None,
                "board": "主板",
                "market": "sz",
                "source": "unit",
            }
        ],
    )
    save_symbol_status(
        conn,
        pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "trade_date": "2020-01-02",
                    "is_st": 0,
                    "is_suspended": 0,
                    "board": "主板",
                    "source": "unit",
                    "fetched_at": "now",
                }
            ]
        ),
    )
    conn.commit()
    conn.close()
    industry_csv = tmp_path / "industries.csv"
    industry_csv.write_text("symbol,industry_name,industry_code\n000001,银行,801780\n", encoding="utf-8")
    old_argv = sys.argv
    try:
        sys.argv = [
            "import_symbol_industries.py",
            "--db",
            str(db),
            "--input",
            str(industry_csv),
            "--provider",
            "unit",
        ]
        stdout = StringIO()
        with redirect_stdout(stdout):
            assert import_symbol_industries_main() == 0
    finally:
        sys.argv = old_argv

    import_output = json.loads(stdout.getvalue())
    assert import_output["provider_result"]["dataset"] == "symbol_industries"
    assert import_output["provider_result"]["rows"] == 1
    assert import_output["provider_result"]["adapter"]["maturity"] == "beta"
    assert load_symbol_industry_map(db) == {"000001": "银行"}
    report = build_quality_report(
        db,
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
        board_scope="main",
        required_adjusts=("raw",),
    )
    assert report["symbol_industries"]["coverage"]["coverage_ratio"] == 1.0
    assert not any(flag.startswith("industry_symbols_incomplete") for flag in report["red_flags"])


def test_quality_report_flags_missing_status_rows(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    ensure_quant_schema(conn)
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    _insert_bar(conn, "000001", "2020-01-03", "raw", 10, 11, 9, 10)
    save_lifecycle(
        conn,
        [
            {
                "symbol": "000001",
                "name": "平安银行",
                "list_date": "1991-04-03",
                "delist_date": None,
                "board": "主板",
                "market": "sz",
                "source": "unit",
            }
        ],
    )
    save_symbol_status(
        conn,
        pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "trade_date": "2020-01-02",
                    "is_st": 0,
                    "is_suspended": 0,
                    "board": "主板",
                    "source": "unit",
                    "fetched_at": "now",
                }
            ]
        ),
    )
    conn.commit()
    conn.close()

    report = build_quality_report(
        db,
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        board_scope="main",
        required_adjusts=("raw",),
    )

    coverage = report["symbol_status_daily"]["row_coverage"]
    assert coverage["expected_raw_rows"] == 2
    assert coverage["covered_raw_rows"] == 1
    assert any(flag.startswith("status_rows_incomplete") for flag in report["red_flags"])
    assert report["is_formal_ready"] is False


def test_backtest_constraint_enrichment_uses_status_and_lifecycle(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    init_fetch_db(conn)
    ensure_quant_schema(conn)
    save_lifecycle(
        conn,
        [
            {
                "symbol": "000001",
                "name": "unit",
                "list_date": "1991-04-03",
                "delist_date": None,
                "board": "主板",
                "market": "sz",
                "source": "unit",
            }
        ],
    )
    save_symbol_status(
        conn,
        pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "trade_date": "2020-01-02",
                    "is_st": 1,
                    "is_suspended": 0,
                    "board": "主板",
                    "source": "unit",
                    "fetched_at": "now",
                }
            ]
        ),
    )
    conn.commit()
    conn.close()
    frame = pd.DataFrame({"symbol": ["000001"], "trade_date": [pd.Timestamp("2020-01-02")]})

    enriched, summary = enrich_backtest_constraints(db, frame, "main")

    assert bool(enriched["has_status"].iloc[0])
    assert bool(enriched["is_st"].iloc[0])
    assert not bool(enriched["signal_allowed"].iloc[0])
    assert enriched["limit_rate"].iloc[0] == 0.05
    assert summary["status_row_coverage"] == 1.0


def test_limit_locked_open_blocks_buy_and_sell():
    dynamic.G_EXECUTION = default_execution_config()
    group = pd.DataFrame({"symbol": ["000001", "000002"], "raw_close": [10.0, 10.0]})
    entry_open = np.asarray([11.0, 9.0])
    entry_high = np.asarray([11.0, 9.0])
    entry_low = np.asarray([11.0, 9.0])

    buyable, sellable = limit_trade_masks(group, entry_open, entry_high, entry_low)

    assert not bool(buyable[0])
    assert bool(sellable[0])
    assert bool(buyable[1])
    assert not bool(sellable[1])


def test_limit_rates_distinguish_st_and_growth_boards():
    assert limit_rate_for_symbol("000001", "主板", False) == 0.10
    assert limit_rate_for_symbol("000001", "主板", True) == 0.05
    assert limit_rate_for_symbol("300001", "创业板", False) == 0.20
    assert limit_rate_for_symbol("688001", "科创板", False) == 0.20


def test_status_constraints_block_signal_candidates_and_suspended_trades():
    dynamic.G_EXECUTION = default_execution_config()
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002"]),
        entry_open=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([10.5, 10.5], dtype=np.float32),
        entry_low=np.asarray([9.5, 9.5], dtype=np.float32),
        entry_amount=np.asarray([100_000_000, 100_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000, 1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True, True]),
        entry_sellable=np.asarray([True, True]),
        signal_allowed=np.asarray([False, True]),
        features=np.empty((0, 2), dtype=np.float32),
        amount21=np.asarray([100_000_000, 100_000_000], dtype=np.float32),
        close=np.asarray([10.0, 10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True, True])},
    )
    spec = DynamicSpec(
        formula=dynamic.Formula("unit", {}),
        market_filter="none",
        top_n=1,
        min_amount=0,
        min_price=0,
        trend_filter="none",
        min_hold_days=0,
        max_hold_days=1,
        replace_count=1,
        stop_loss=None,
    )

    symbols, scores = select_target_from_score(day, spec, np.asarray([0.99, 0.10], dtype=np.float32))

    assert symbols.tolist() == ["000002"]
    assert np.isclose(float(scores[0]), 0.10)

    group = pd.DataFrame({"symbol": ["000001"], "raw_close": [10.0], "limit_rate": [0.10]})
    buyable, sellable = limit_trade_masks(
        group,
        np.asarray([10.0]),
        np.asarray([10.5]),
        np.asarray([9.5]),
        entry_suspended=np.asarray([True]),
    )
    assert not bool(buyable[0])
    assert not bool(sellable[0])


def test_missing_market_filter_is_blocking_not_crashing():
    dynamic.G_EXECUTION = default_execution_config()
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001"]),
        entry_open=np.asarray([10.0], dtype=np.float32),
        entry_high=np.asarray([10.5], dtype=np.float32),
        entry_low=np.asarray([9.5], dtype=np.float32),
        entry_amount=np.asarray([100_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True]),
        entry_sellable=np.asarray([True]),
        signal_allowed=np.asarray([True]),
        features=np.empty((0, 1), dtype=np.float32),
        amount21=np.asarray([100_000_000], dtype=np.float32),
        close=np.asarray([10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True])},
    )
    spec = DynamicSpec(
        formula=dynamic.Formula("unit", {}),
        market_filter="sw_top_mom_63_pos",
        top_n=1,
        min_amount=0,
        min_price=0,
        trend_filter="none",
        min_hold_days=0,
        max_hold_days=1,
        replace_count=1,
        stop_loss=None,
    )
    dynamic.G_TARGETS = {target_key(spec): [(np.asarray(["000001"]), np.asarray([1.0], dtype=np.float32))]}

    returns, active, trades, _, _, _ = simulate_dynamic([day], None, spec, {"none": {day.signal_date}})

    assert returns.tolist() == [0.0]
    assert active.tolist() == [False]
    assert trades.tolist() == [False]


def test_execute_rebalance_keeps_position_when_sell_blocked():
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    dynamic.G_EXECUTION = default_execution_config()
    dynamic.G_INITIAL_CASH = 1_000_000
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002"]),
        entry_open=np.asarray([9.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([9.0, 10.5], dtype=np.float32),
        entry_low=np.asarray([9.0, 9.5], dtype=np.float32),
        entry_amount=np.asarray([100_000_000, 100_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000, 1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True, True]),
        entry_sellable=np.asarray([False, True]),
        signal_allowed=np.asarray([True, True]),
        features=np.empty((0, 2), dtype=np.float32),
        amount21=np.asarray([100_000_000, 100_000_000], dtype=np.float32),
        close=np.asarray([10.0, 10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True, True])},
    )

    result, cost = execute_rebalance(
        day,
        current_symbols=np.asarray(["000001"]),
        current_entry_open=np.asarray([10.0], dtype=np.float32),
        target_symbols=np.asarray([], dtype=str),
        equity=1.0,
    )

    assert cost == 0.0
    assert result.blocked_sell_count == 1
    assert result.current_symbols.tolist() == ["000001"]
    assert result.current_weights.tolist() == [1.0]
    assert result.unfilled_sell_value == 1_000_000.0
    assert result.trade_events[0]["side"] == "sell"
    assert result.trade_events[0]["status"] == "blocked"
    assert result.trade_events[0]["reason"] == "limit_down_block"


def test_execute_rebalance_partially_fills_when_capacity_is_low():
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=0.02,
        capacity_equity_mode="initial",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    dynamic.G_INITIAL_CASH = 1_000_000
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001"]),
        entry_open=np.asarray([10.0], dtype=np.float32),
        entry_high=np.asarray([10.5], dtype=np.float32),
        entry_low=np.asarray([9.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True]),
        entry_sellable=np.asarray([True]),
        signal_allowed=np.asarray([True]),
        features=np.empty((0, 1), dtype=np.float32),
        amount21=np.asarray([10_000_000], dtype=np.float32),
        close=np.asarray([10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True])},
    )

    result, cost = execute_rebalance(
        day,
        current_symbols=np.asarray([], dtype=str),
        current_entry_open=np.asarray([], dtype=np.float32),
        target_symbols=np.asarray(["000001"]),
        equity=1.0,
    )

    assert cost == 0.0
    assert result.trade_count == 1
    assert result.blocked_buy_count == 0
    assert result.partial_buy_count == 1
    assert result.turnover_value == 200_000.0
    assert result.unfilled_buy_value == 800_000.0
    assert result.current_symbols.tolist() == ["000001"]
    np.testing.assert_allclose(result.current_weights, np.asarray([0.2], dtype=np.float32))
    assert result.trade_events[0]["side"] == "buy"
    assert result.trade_events[0]["status"] == "partial"
    assert result.trade_events[0]["reason"] == "capacity_partial"
    next_day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-03"),
        entry_date=pd.Timestamp("2020-01-06"),
        symbols=np.asarray(["000001"]),
        entry_open=np.asarray([11.0], dtype=np.float32),
        entry_high=np.asarray([11.5], dtype=np.float32),
        entry_low=np.asarray([10.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True]),
        entry_sellable=np.asarray([True]),
        signal_allowed=np.asarray([True]),
        features=np.empty((0, 1), dtype=np.float32),
        amount21=np.asarray([10_000_000], dtype=np.float32),
        close=np.asarray([11.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True])},
    )
    period_return, _ = dynamic.open_to_open_return(
        next_day,
        result.current_symbols,
        result.current_prev_open,
        result.current_weights,
    )
    assert np.isclose(period_return, 0.02)


def test_execute_rebalance_respects_max_position_weight_cap():
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=1.0,
        capacity_equity_mode="initial",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    dynamic.G_INITIAL_CASH = 1_000_000
    dynamic.G_MAX_POSITION_WEIGHT = 0.1
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001"]),
        entry_open=np.asarray([10.0], dtype=np.float32),
        entry_high=np.asarray([10.5], dtype=np.float32),
        entry_low=np.asarray([9.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True]),
        entry_sellable=np.asarray([True]),
        signal_allowed=np.asarray([True]),
        features=np.empty((0, 1), dtype=np.float32),
        amount21=np.asarray([10_000_000], dtype=np.float32),
        close=np.asarray([10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True])},
    )

    result, cost = execute_rebalance(
        day,
        current_symbols=np.asarray([], dtype=str),
        current_entry_open=np.asarray([], dtype=np.float32),
        target_symbols=np.asarray(["000001"]),
        equity=1.0,
    )

    assert cost == 0.0
    assert result.trade_count == 1
    assert result.trade_events[0]["status"] == "filled"
    assert result.trade_events[0]["desired_notional"] == 100_000.0
    np.testing.assert_allclose(result.current_weights, np.asarray([0.1], dtype=np.float32))
    next_day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-03"),
        entry_date=pd.Timestamp("2020-01-06"),
        symbols=np.asarray(["000001"]),
        entry_open=np.asarray([11.0], dtype=np.float32),
        entry_high=np.asarray([11.5], dtype=np.float32),
        entry_low=np.asarray([10.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True]),
        entry_sellable=np.asarray([True]),
        signal_allowed=np.asarray([True]),
        features=np.empty((0, 1), dtype=np.float32),
        amount21=np.asarray([10_000_000], dtype=np.float32),
        close=np.asarray([11.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True])},
    )
    period_return, _ = dynamic.open_to_open_return(
        next_day,
        result.current_symbols,
        result.current_prev_open,
        result.current_weights,
    )
    assert np.isclose(period_return, 0.01)
    dynamic.G_MAX_POSITION_WEIGHT = 0.0


def test_execute_rebalance_respects_turnover_cap():
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_TURNOVER_PCT = 0.15
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=1.0,
        capacity_equity_mode="initial",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    dynamic.G_INITIAL_CASH = 1_000_000
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002"]),
        entry_open=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([10.5, 10.5], dtype=np.float32),
        entry_low=np.asarray([9.5, 9.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000, 10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000, 1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True, True]),
        entry_sellable=np.asarray([True, True]),
        signal_allowed=np.asarray([True, True]),
        features=np.empty((0, 2), dtype=np.float32),
        amount21=np.asarray([10_000_000, 10_000_000], dtype=np.float32),
        close=np.asarray([10.0, 10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True, True])},
    )

    result, cost = execute_rebalance(
        day,
        current_symbols=np.asarray([], dtype=str),
        current_entry_open=np.asarray([], dtype=np.float32),
        target_symbols=np.asarray(["000001", "000002"]),
        equity=1.0,
    )

    assert cost == 0.0
    assert result.turnover_value == 150_000.0
    assert result.turnover_blocked_count == 2
    assert result.turnover_blocked_value == 850_000.0
    assert result.trade_events[0]["status"] == "partial"
    assert result.trade_events[0]["reason"] == "turnover_block"
    assert result.trade_events[1]["status"] == "blocked"
    assert result.trade_events[1]["reason"] == "turnover_block"
    np.testing.assert_allclose(result.current_weights, np.asarray([0.15], dtype=np.float32))
    dynamic.G_MAX_TURNOVER_PCT = 0.0


def test_industry_weight_cap_filters_targets_and_reports_budget():
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_INDUSTRY_WEIGHT = 0.5
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002", "000003"]),
        entry_open=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([10.5, 10.5, 10.5], dtype=np.float32),
        entry_low=np.asarray([9.5, 9.5, 9.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000, 10_000_000, 10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000, 1_000_000, 1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True, True, True]),
        entry_sellable=np.asarray([True, True, True]),
        signal_allowed=np.asarray([True, True, True]),
        features=np.empty((0, 3), dtype=np.float32),
        amount21=np.asarray([10_000_000, 10_000_000, 10_000_000], dtype=np.float32),
        close=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True, True, True])},
        industry_labels=np.asarray(["bank", "bank", "energy"]),
    )

    symbols, scores, blocked = filter_target_by_industry_weight(
        day,
        np.asarray(["000001", "000002", "000003"]),
        np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        np.asarray([], dtype=str),
        np.asarray([], dtype=np.float32),
        target_weight=0.5,
    )

    assert symbols.tolist() == ["000001", "000003"]
    assert scores.tolist() == [0.8999999761581421, 0.699999988079071]
    assert blocked == 1
    exposure = industry_exposure_map(symbols, np.asarray([0.5, 0.5], dtype=np.float32), np.asarray(["bank", "energy"]))
    assert exposure == {"bank": 0.5, "energy": 0.5}
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=1.0,
        capacity_equity_mode="initial",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    dynamic.G_INITIAL_CASH = 1_000_000
    result, _ = dynamic.execute_rebalance(
        day,
        current_symbols=np.asarray([], dtype=str),
        current_entry_open=np.asarray([], dtype=np.float32),
        target_symbols=symbols,
        equity=1.0,
        target_weight_override=0.5,
    )
    np.testing.assert_allclose(result.current_weights, np.asarray([0.5, 0.5], dtype=np.float32))

    budget = risk_budget_report(
        {
            "max_drawdown": -0.1,
            "max_position_weight_observed": 0.5,
            "max_industry_weight_observed": 0.5,
            "unfilled_buy_value": 0.0,
            "unfilled_sell_value": 0.0,
            "max_period_turnover_pct": 1.0,
            "blocked_buy_count": 0,
            "blocked_sell_count": 0,
            "partial_buy_count": 0,
            "partial_sell_count": 0,
            "portfolio_risk_off_rate": 0.0,
            "avg_cash_weight": 0.0,
            "avg_invested_weight": 1.0,
        },
        pd.DataFrame(),
        pd.DataFrame(
            [
                {"industry_label": "bank", "weight": 0.5},
                {"industry_label": "energy", "weight": 0.5},
            ]
        ),
        {
            "portfolio_stop_loss": 0.0,
            "max_position_weight": 0.0,
            "max_industry_weight": 0.5,
            "capacity_pct_of_amount": 0.02,
            "max_turnover_pct": 0.0,
        },
    )
    assert any(row["name"] == "industry_concentration" for row in budget["risk_sources"])
    assert budget["industry_exposure_top"][0]["max_pick_weight"] == 0.5
    dynamic.G_MAX_INDUSTRY_WEIGHT = 0.0


def test_dynamic_report_p1_metrics_breakdowns_and_benchmark():
    equity = pd.DataFrame(
        [
            {
                "signal_date": "2020-12-31",
                "entry_date": "2021-01-04",
                "equity": 1_100_000.0,
                "period_return": 0.10,
                "drawdown": 0.0,
                "trade": True,
                "active": True,
                "trade_count": 2,
                "blocked_buy_count": 0,
                "blocked_sell_count": 0,
                "turnover_value": 200_000.0,
            },
            {
                "signal_date": "2021-01-04",
                "entry_date": "2021-01-05",
                "equity": 1_045_000.0,
                "period_return": -0.05,
                "drawdown": -0.05,
                "trade": False,
                "active": True,
                "trade_count": 0,
                "blocked_buy_count": 1,
                "blocked_sell_count": 0,
                "turnover_value": 0.0,
            },
            {
                "signal_date": "2021-01-05",
                "entry_date": "2021-02-01",
                "equity": 1_149_500.0,
                "period_return": 0.10,
                "drawdown": 0.0,
                "trade": True,
                "active": True,
                "trade_count": 2,
                "blocked_buy_count": 0,
                "blocked_sell_count": 1,
                "turnover_value": 200_000.0,
            },
        ]
    )
    metrics = professional_performance_metrics(equity, 1_000_000.0)
    annual_rows = period_return_breakdown(equity, "Y")
    monthly_rows = period_return_breakdown(equity, "M")
    bars = pd.DataFrame(
        [
            {"symbol": "000001", "trade_date": "2021-01-04", "raw_close": 10.0},
            {"symbol": "000001", "trade_date": "2021-01-05", "raw_close": 11.0},
            {"symbol": "000001", "trade_date": "2021-02-01", "raw_close": 12.0},
            {"symbol": "000002", "trade_date": "2021-01-04", "raw_close": 20.0},
            {"symbol": "000002", "trade_date": "2021-01-05", "raw_close": 18.0},
            {"symbol": "000002", "trade_date": "2021-02-01", "raw_close": 19.8},
        ]
    )
    benchmark = compute_equal_weight_benchmark(
        bars,
        pd.Timestamp("2021-01-04"),
        pd.Timestamp("2021-02-01"),
        "unit_equal_weight",
    )
    relative = relative_performance_metrics(equity, benchmark)
    stress = capacity_stress_plan(
        argparse.Namespace(
            initial_cash=1_000_000.0,
            capacity_pct_of_amount=0.02,
            slippage_bps=5.0,
            impact_bps_per_pct_amount=2.0,
            capacity_equity_mode="initial",
        )
    )

    assert metrics["annual_volatility"] > 0
    assert metrics["sharpe"] is not None
    assert len(annual_rows) == 1
    assert annual_rows[0]["executed_trade_count"] == 4
    assert annual_rows[0]["blocked_buy_count"] == 1
    assert [row["period"] for row in monthly_rows] == ["2021-01", "2021-02"]
    assert benchmark["name"] == "unit_equal_weight"
    assert benchmark["symbols"] == 2
    assert benchmark["periods"] == 2
    assert benchmark["total_return"] > 0
    assert len(benchmark["daily_returns"]) == 2
    assert relative["matched_periods"] == 2
    assert relative["beta"] is not None
    assert relative["alpha_annualized"] is not None
    assert stress["current"]["capacity_equity_mode"] == "initial"
    assert 10_000_000.0 in stress["recommended_grid"]["initial_cash"]


def test_capacity_stress_replays_grid_without_reselecting_specs():
    dynamic.G_MAX_POSITION_WEIGHT = 0.0
    dynamic.G_MAX_TURNOVER_PCT = 0.0
    dynamic.G_INITIAL_CASH = 1_000_000
    dynamic.G_EXECUTION = dynamic.ExecutionConfig(
        buy_cost=0.0,
        sell_cost=0.0,
        slippage_bps=0.0,
        impact_bps_per_pct_amount=0.0,
        capacity_pct_of_amount=0.02,
        capacity_equity_mode="initial",
        lot_size=100,
        limit_epsilon=0.002,
        block_limit_trades=True,
    )
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001"]),
        entry_open=np.asarray([10.0], dtype=np.float32),
        entry_high=np.asarray([10.5], dtype=np.float32),
        entry_low=np.asarray([9.5], dtype=np.float32),
        entry_amount=np.asarray([100_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True]),
        entry_sellable=np.asarray([True]),
        signal_allowed=np.asarray([True]),
        features=np.empty((0, 1), dtype=np.float32),
        amount21=np.asarray([100_000_000], dtype=np.float32),
        close=np.asarray([10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True])},
    )
    spec = DynamicSpec(
        formula=dynamic.Formula("unit", {}),
        market_filter="none",
        top_n=1,
        min_amount=0,
        min_price=0,
        trend_filter="none",
        min_hold_days=0,
        max_hold_days=1,
        replace_count=1,
        stop_loss=None,
    )
    dynamic.G_TARGETS = {target_key(spec): [(np.asarray(["000001"]), np.asarray([1.0], dtype=np.float32))]}
    original_execution = dynamic.G_EXECUTION

    stress_df, meta = run_capacity_stress(
        argparse.Namespace(
            initial_cash=1_000_000.0,
            capacity_pct_of_amount=0.02,
            slippage_bps=0.0,
            impact_bps_per_pct_amount=0.0,
            capacity_equity_mode="initial",
        ),
        [day],
        {2020: spec},
        {"none": {day.signal_date}},
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
    )

    assert len(stress_df) == 9
    assert meta["status"] == "grid_replayed"
    assert meta["rows"] == 9
    assert int(stress_df["is_current_setting"].sum()) == 1
    assert set(stress_df["periods"].dropna().astype(int)) == {1}
    assert dynamic.G_EXECUTION is original_execution
    assert dynamic.G_INITIAL_CASH == 1_000_000


def test_multiple_testing_summary_records_scan_size():
    specs = [
        DynamicSpec(
            formula=dynamic.Formula("f1", {}),
            market_filter="none",
            top_n=1,
            min_amount=0,
            min_price=0,
            trend_filter="none",
            min_hold_days=0,
            max_hold_days=1,
            replace_count=1,
            stop_loss=None,
        ),
        DynamicSpec(
            formula=dynamic.Formula("f2", {}),
            market_filter="none",
            top_n=1,
            min_amount=0,
            min_price=0,
            trend_filter="none",
            min_hold_days=0,
            max_hold_days=1,
            replace_count=1,
            stop_loss=None,
        ),
    ]
    diagnostics = pd.DataFrame(
        [
            {"status": "selected"},
            {"status": "train_candidate"},
            {"status": "frozen_selected"},
        ]
    )

    summary = multiple_testing_summary(specs, diagnostics, "base", "selected", "smoke", "aggressive")

    assert summary["specs_evaluated"] == 2
    assert summary["unique_formulas"] == 2
    assert summary["selected_rows"] == 2
    assert summary["train_candidate_rows_written"] == 1
    assert summary["frozen_selection_rows"] == 1
    assert "overfit" in summary["risk_note"]


def test_run_manifest_records_database_snapshot(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        create table daily_bars (
            symbol text not null,
            trade_date text not null,
            adjust text not null,
            open real, high real, low real, close real, volume real, amount real,
            amplitude real, pct_chg real, chg real, turnover real,
            source text not null, fetched_at text not null,
            primary key(symbol, trade_date, adjust)
        )
        """
    )
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    conn.commit()
    conn.close()

    path = tmp_path / "run.manifest.json"
    cache_registry = tmp_path / "cache_registry.json"
    cache_registry.write_text(
        json.dumps(
            {
                "schema_version": "cache_registry.v1",
                "generated_at": "2026-06-05T12:00:00",
                "entry_count": 1,
                "entries": [
                    {
                        "factor_adjust": "hfq",
                        "end_date": "2026-05-29",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = collect_manifest(
        db,
        ["backtest.py", "--start-date", "2020-01-02"],
        [Path(__file__)],
        {"metrics": tmp_path / "run.metrics.json"},
        cache_registry_path=cache_registry,
    )
    write_manifest(path, manifest)

    assert path.exists()
    assert manifest["database"]["daily_bars"][0]["adjust"] == "raw"
    assert manifest["database"]["daily_bars"][0]["rows"] == 1
    assert manifest["cache_registry"]["entry_count"] == 1
    assert manifest["cache_registry"]["latest_cache_end_date"] == "2026-05-29"
    assert manifest["cache_registry"]["factor_adjusts"] == ["hfq"]
    assert manifest["cache_registry"]["sha256"]


def test_quant_schema_and_quality_report_flag_missing_p0_data(tmp_path):
    db = tmp_path / "bars.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        create table daily_bars (
            symbol text not null,
            trade_date text not null,
            adjust text not null,
            open real, high real, low real, close real, volume real, amount real,
            amplitude real, pct_chg real, chg real, turnover real,
            source text not null, fetched_at text not null,
            primary key(symbol, trade_date, adjust)
        )
        """
    )
    _insert_bar(conn, "000001", "2020-01-02", "raw", 10, 11, 9, 10)
    ensure_quant_schema(conn)
    for table_name in (*REFERENCE_TABLES, *STATE_TABLES):
        assert conn.execute(
            "select 1 from sqlite_master where type = 'table' and name = ?",
            (table_name,),
        ).fetchone()
    conn.close()

    report = build_quality_report(
        db,
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
        board_scope="main",
        required_adjusts=("raw", "hfq"),
    )
    assert any(flag.startswith("adjust_missing:hfq") for flag in report["red_flags"])
    assert report["schema"]["missing_reference_tables"] == []
    assert report["schema"]["missing_state_tables"] == []


def test_quant_state_records_idempotent_paper_state(tmp_path):
    db = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(db)
    ensure_quant_schema(conn)

    run_id = record_signal_run(
        conn,
        strategy="demo",
        signal_date="2020-01-02",
        status="started",
        config={"top_n": 1},
    )
    record_signal_run(
        conn,
        strategy="demo",
        signal_date="2020-01-02",
        status="finished",
        config={"top_n": 1},
        run_id=run_id,
        finished_at="2020-01-02T16:00:00",
    )
    signal_id = record_signal(
        conn,
        strategy="demo",
        symbol="000001",
        signal_date="2020-01-02",
        action="buy",
        score=0.9,
        weight=0.2,
        reason="unit",
        payload={"rank": 1},
        run_id=run_id,
    )
    record_signal(
        conn,
        strategy="demo",
        symbol="000001",
        signal_date="2020-01-02",
        action="buy",
        score=0.95,
        weight=0.25,
        reason="unit-update",
        signal_id=signal_id,
        run_id=run_id,
    )
    record_position(conn, strategy="demo", symbol="000001", as_of_date="2020-01-03", quantity=100, avg_cost=10.0)
    alert_id = record_alert(conn, strategy="demo", alert_date="2020-01-03", severity="info", title="test")
    attempt_id = record_alert_attempt(conn, run_id=run_id, alert_id=alert_id, channel="stdout", status="failed")
    record_alert_attempt(conn, run_id=run_id, alert_id=alert_id, channel="stdout", status="sent")
    trade_id = record_paper_trade(
        conn,
        strategy="demo",
        symbol="000001",
        trade_date="2020-01-03",
        side="buy",
        quantity=100,
        price=10.0,
        amount=1000.0,
        status="filled",
        signal_id=signal_id,
    )

    assert signal_id.startswith("sig_")
    assert run_id.startswith("run_")
    assert alert_id.startswith("alert_")
    assert attempt_id.startswith("attempt_")
    assert trade_id.startswith("ptrade_")
    assert state_counts(conn, "demo") == {
        "signal_runs": 1,
        "paper_run_registry": 0,
        "signals": 1,
        "positions": 1,
        "alerts": 1,
        "alert_attempts": 1,
        "paper_trades": 1,
    }
    row = conn.execute("select run_id, score, weight, reason from signals where signal_id = ?", (signal_id,)).fetchone()
    assert row == (run_id, 0.95, 0.25, "unit-update")
    attempt = conn.execute(
        "select status, attempt_count from alert_attempts where attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    assert attempt == ("sent", 2)
    conn.close()


def test_generate_daily_signals_records_paper_state(tmp_path):
    db = tmp_path / "state.sqlite3"
    plan = build_plan(
        [
            {"symbol": "000001", "action": "buy", "score": 0.9, "weight": 0.25, "entry_open": 10.0, "reason": "unit"},
            {"symbol": "000002", "action": "hold", "score": 0.5, "weight": 0.0, "entry_open": 20.0, "reason": "unit"},
        ],
        "demo",
        "2020-01-03",
        "2020-01-06",
        1_000_000.0,
    )

    counts = write_plan(db, "demo", "run_unit", plan)

    assert counts["signal_runs"] == 1
    assert counts["signals"] == 2
    assert counts["positions"] == 1
    assert counts["paper_trades"] == 1
    assert counts["alert_attempts"] == 1
    conn = sqlite3.connect(db)
    row = conn.execute(
        "select symbol, action, weight, run_id from signals where symbol = '000001'"
    ).fetchone()
    trade = conn.execute(
        "select symbol, side, quantity, price, amount, status from paper_trades"
    ).fetchone()
    conn.close()
    assert row == ("000001", "buy", 0.25, "run_unit")
    assert trade == ("000001", "buy", 25_000.0, 10.0, 250_000.0, "planned")


def test_generate_daily_from_picks_creates_buy_hold_sell_rows(tmp_path):
    db = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(db)
    ensure_quant_schema(conn)
    record_position(
        conn,
        strategy="demo",
        symbol="000001",
        as_of_date="2020-01-02",
        quantity=100,
        avg_cost=10.0,
        market_value=1000.0,
    )
    record_position(
        conn,
        strategy="demo",
        symbol="000003",
        as_of_date="2020-01-02",
        quantity=200,
        avg_cost=5.0,
        market_value=1000.0,
    )
    conn.close()
    picks = tmp_path / "latest.picks.csv"
    picks.write_text(
        "signal_date,entry_date,symbol,weight,score,entry_open,formula\n"
        "2020-01-03,2020-01-06,000001,0.4,0.9,11,unit\n"
        "2020-01-03,2020-01-06,000002,0.6,0.8,20,unit\n",
        encoding="utf-8",
    )

    rows, signal_date, entry_date = build_rows_from_picks(picks, db, "demo")

    by_symbol = {row["symbol"]: row for row in rows}
    assert signal_date == "2020-01-03"
    assert entry_date == "2020-01-06"
    assert by_symbol["000001"]["action"] == "hold"
    assert by_symbol["000002"]["action"] == "buy"
    assert by_symbol["000003"]["action"] == "sell"
    assert by_symbol["000003"]["quantity"] == 200.0


def test_state_report_observation_audit_detects_ready_window(tmp_path):
    db = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(db)
    ensure_quant_schema(conn)
    for index in range(60):
        signal_date = f"2020-03-{index + 1:02d}"
        run_id = f"run_{index}"
        record_signal_run(
            conn,
            strategy="demo",
            signal_date=signal_date,
            status="finished",
            run_id=run_id,
            finished_at=f"{signal_date}T16:00:00",
        )
        record_signal(
            conn,
            strategy="demo",
            symbol="000001",
            signal_date=signal_date,
            action="hold",
            signal_id=f"sig_{index}",
            run_id=run_id,
        )
    audit = state_report_module.build_report(db, "demo", 3, 90)["paper_observation"]
    conn.close()

    assert audit["observed_distinct_signal_dates"] == 60
    assert audit["is_60_day_ready"] is True
    assert audit["is_90_day_ready"] is False


def test_sensitivity_matrix_plans_core_dimensions(tmp_path):
    config = {
        "start_date": "2021-01-04",
        "train_years": 4,
        "top_n": 3,
        "factor_adjust": "hfq",
        "strict_factor_adjust": True,
        "formal": True,
        "freeze_selection_date": "2020-12-31",
        "split_policy": {"name": "unit"},
        "frozen_config": {"frozen": True},
        "industry_source": "sw.industry.index.一级行业",
        "capacity_pct_of_amount": 0.02,
        "slippage_bps": 5.0,
        "output_dir": "reports/formal",
    }

    matrix = build_matrix(config, tmp_path / "sensitivity", worker_override=1)
    plan = matrix_plan_rows(matrix)
    dimensions = {row["dimension"] for row in plan}

    assert "base" in dimensions
    assert {"top_n", "min_hold_days", "max_hold_days", "stop_loss", "industry_source"}.issubset(dimensions)
    assert all(row["output_dir"] for row in plan)


def test_validate_formal_reports_enforces_required_fields_and_manifest(tmp_path):
    reports = tmp_path / "formal"
    reports.mkdir()
    metrics = reports / "run.metrics.json"
    payload = {
        "is_formal_valid": True,
        "data_quality_red_flags": [],
        "status_coverage": {},
        "split_policy": {},
        "frozen_config": {},
        "professional_metrics": {},
        "annual_breakdown": [],
        "monthly_breakdown": [],
        "benchmarks": {},
        "multiple_testing": {},
        "capacity_stress": {},
        "risk_budget": {},
        "config": {},
    }
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    (reports / "run.manifest.json").write_text("{}", encoding="utf-8")

    report = build_formal_report_validation(reports, require_formal_valid=True)

    assert report["is_valid"] is True
    assert report["metrics_files"] == 1


def test_blacklist_blocks_buy_candidates():
    dynamic.G_BLACKLIST = {
        "000001": [{"start_date": "2020-01-01", "end_date": "2020-12-31", "reason": "unit_blacklist"}]
    }
    day = CompactDayData(
        signal_date=pd.Timestamp("2020-01-02"),
        entry_date=pd.Timestamp("2020-01-03"),
        symbols=np.asarray(["000001", "000002"]),
        entry_open=np.asarray([10.0, 10.0], dtype=np.float32),
        entry_high=np.asarray([10.5, 10.5], dtype=np.float32),
        entry_low=np.asarray([9.5, 9.5], dtype=np.float32),
        entry_amount=np.asarray([10_000_000, 10_000_000], dtype=np.float32),
        entry_volume=np.asarray([1_000_000, 1_000_000], dtype=np.float32),
        entry_buyable=np.asarray([True, True]),
        entry_sellable=np.asarray([True, True]),
        signal_allowed=np.asarray([True, True]),
        features=np.empty((0, 2), dtype=np.float32),
        amount21=np.asarray([10_000_000, 10_000_000], dtype=np.float32),
        close=np.asarray([10.0, 10.0], dtype=np.float32),
        trend_masks={"none": np.asarray([True, True])},
    )
    spec = DynamicSpec(
        formula=dynamic.Formula("unit", {}),
        market_filter="none",
        top_n=1,
        min_amount=0,
        min_price=0,
        trend_filter="none",
        min_hold_days=0,
        max_hold_days=1,
        replace_count=1,
        stop_loss=None,
    )

    symbols, _ = select_target_from_score(day, spec, np.asarray([0.9, 0.8], dtype=np.float32))

    assert symbols.tolist() == ["000002"]
    dynamic.G_BLACKLIST = {}


def test_professional_validation_report_summarizes_artifacts(tmp_path):
    metrics_path = tmp_path / "run.metrics.json"
    diagnostics_path = tmp_path / "run.diagnostics.csv"
    capacity_path = tmp_path / "run.capacity_stress.csv"
    metrics_path.write_text(
        json.dumps(
            {
                "config": {"strategy": "demo"},
                "is_formal_valid": False,
                "split_policy": {"current_result_segment": "test"},
                "multiple_testing": {"specs_evaluated": 2, "selected_rows": 1, "train_candidate_rows_written": 1},
                "risk_budget": {"risk_sources": [{"name": "market_drawdown"}]},
            }
        ),
        encoding="utf-8",
    )
    diagnostics_path.write_text(
        "year,status,train_start,train_end,formula,annual_return,max_drawdown\n"
        "2021,selected,2018-01-01,2020-12-31,f1,0.2,-0.1\n",
        encoding="utf-8",
    )
    capacity_path.write_text(
        "initial_cash,capacity_pct_of_amount,slippage_bps,annual_return,max_drawdown,is_current_setting\n"
        "1000000,0.02,5,0.2,-0.1,True\n"
        "10000000,0.01,10,0.1,-0.2,False\n",
        encoding="utf-8",
    )

    report = build_professional_validation_report(metrics_path, diagnostics_path, capacity_path)

    assert report["nested_validation"]["status"] == "available"
    assert report["nested_validation"]["outer_rows"][0]["year"] == 2021
    assert report["sensitivity"]["status"] == "available"
    assert report["sensitivity"]["annual_return_min"] == 0.1


def test_backup_sqlite_writes_consistent_manifest(tmp_path):
    db = tmp_path / "sample.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("create table sample (id integer primary key, value text)")
    conn.execute("insert into sample(value) values ('ok')")
    conn.commit()
    conn.close()

    manifest = backup_sqlite(db, tmp_path / "backups", "unit")

    backup = Path(manifest["backup"])
    manifest_path = backup.with_suffix(".manifest.json")
    assert backup.exists()
    assert manifest_path.exists()
    assert manifest["counts_match"] is True
    assert manifest["backup_table_counts"]["sample"] == 1


def test_formal_dynamic_wrapper_dry_run(tmp_path):
    config = tmp_path / "formal.json"
    config.write_text(
        """
        {
          "start_date": "2020-01-02",
          "train_years": 1,
          "factor_adjust": "hfq",
          "strict_factor_adjust": true,
          "formal": true,
          "formal_required_adjusts": ["raw", "qfq", "hfq"],
          "freeze_selection_date": "2019-12-31",
          "split_policy": {
            "name": "unit_split",
            "train": {"start": "2010-01-01", "end": "2017-12-31"},
            "validation": {"start": "2018-01-01", "end": "2019-12-31"},
            "test": {"start": "2020-01-02", "end": "latest_available"},
            "current_result_segment": "test",
            "test_reselection_allowed": false
          },
          "frozen_config": {
            "frozen": true,
            "freeze_selection_date": "2019-12-31",
            "test_result_reselection_allowed": false
          },
          "board_scope": "main",
          "output_dir": "reports/formal-test"
        }
        """,
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "run_formal_dynamic.py"),
            "--config",
            str(config),
            "--db",
            str(tmp_path / "bars.sqlite3"),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--formal" in result.stdout
    assert "--strict-factor-adjust" in result.stdout
    assert "--factor-adjust" in result.stdout
    assert "--split-policy-json" in result.stdout
    assert "--frozen-config-json" in result.stdout
    assert "--freeze-selection-date" in result.stdout
    assert "--portfolio-stop-loss" in result.stdout
    assert "--max-position-weight" in result.stdout
    assert "--max-industry-weight" in result.stdout
    assert "--max-turnover-pct" in result.stdout
    assert "0.25" in result.stdout
    assert "0.2" in result.stdout
    assert "0.35" in result.stdout
    assert "0.8" in result.stdout
    assert "hfq" in result.stdout


if __name__ == "__main__":
    test_main_board_symbol_filter()
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_load_data_uses_raw_execution_and_hfq_factors(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_strict_factor_adjust_rejects_partial_missing_rows(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_daily_fetch_jobs_backfill_before_and_after_existing_range(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_adj_factors_are_derived_from_raw_qfq_hfq(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_fetch_status_source_used_column_is_migrated(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_lifecycle_and_status_savers_populate_reference_tables(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_symbol_industry_import_and_quality_coverage(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_quality_report_flags_missing_status_rows(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_backtest_constraint_enrichment_uses_status_and_lifecycle(Path(directory))
    test_limit_locked_open_blocks_buy_and_sell()
    test_limit_rates_distinguish_st_and_growth_boards()
    test_status_constraints_block_signal_candidates_and_suspended_trades()
    test_missing_market_filter_is_blocking_not_crashing()
    test_execute_rebalance_keeps_position_when_sell_blocked()
    test_execute_rebalance_partially_fills_when_capacity_is_low()
    test_execute_rebalance_respects_max_position_weight_cap()
    test_execute_rebalance_respects_turnover_cap()
    test_industry_weight_cap_filters_targets_and_reports_budget()
    test_dynamic_report_p1_metrics_breakdowns_and_benchmark()
    test_capacity_stress_replays_grid_without_reselecting_specs()
    test_multiple_testing_summary_records_scan_size()
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_run_manifest_records_database_snapshot(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_quant_schema_and_quality_report_flag_missing_p0_data(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_quant_state_records_idempotent_paper_state(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_generate_daily_signals_records_paper_state(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_generate_daily_from_picks_creates_buy_hold_sell_rows(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_state_report_observation_audit_detects_ready_window(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_sensitivity_matrix_plans_core_dimensions(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_validate_formal_reports_enforces_required_fields_and_manifest(Path(directory))
    test_blacklist_blocks_buy_candidates()
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_professional_validation_report_summarizes_artifacts(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_backup_sqlite_writes_consistent_manifest(Path(directory))
    with __import__("tempfile").TemporaryDirectory() as directory:
        test_formal_dynamic_wrapper_dry_run(Path(directory))

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import state_report as state_report_module  # noqa: E402
from quant_schema import ensure_quant_schema  # noqa: E402
from quant_state import record_signal, record_signal_run  # noqa: E402
from run_daily_paper_pipeline import build_pipeline_output  # noqa: E402


def _insert_raw_day(conn: sqlite3.Connection, trade_date: str) -> None:
    conn.execute(
        """
        insert into daily_bars (
            symbol, trade_date, adjust, open, high, low, close, volume, amount,
            amplitude, pct_chg, chg, turnover, source, fetched_at
        )
        values ('000001', ?, 'raw', 10, 11, 9, 10.5, 1000000, 100000000, 0, 0, 0, 1, 'test', 'now')
        """,
        (trade_date,),
    )


def _build_daily_bars_table(conn: sqlite3.Connection) -> None:
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


def test_state_report_requires_consecutive_market_dates_when_available(tmp_path):
    db = tmp_path / "state.sqlite3"
    with sqlite3.connect(db) as conn:
        _build_daily_bars_table(conn)
        ensure_quant_schema(conn)
        for index in range(1, 62):
            _insert_raw_day(conn, f"2020-03-{index:02d}")
        for index in range(1, 62):
            if index == 30:
                continue
            signal_date = f"2020-03-{index:02d}"
            run_id = f"run_{index}"
            record_signal_run(
                conn,
                strategy="demo",
                signal_date=signal_date,
                status="finished",
                config={"manual_confirmation_required": False},
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

    assert audit["expected_trading_dates_available"] is True
    assert "2020-03-30" in audit["missing_expected_signal_dates_60d"]
    assert audit["is_60_day_ready"] is False


def test_state_report_defaults_to_latest_60_trading_days(tmp_path):
    db = tmp_path / "state.sqlite3"
    with sqlite3.connect(db) as conn:
        _build_daily_bars_table(conn)
        ensure_quant_schema(conn)
        for index in range(1, 62):
            _insert_raw_day(conn, f"2020-03-{index:02d}")
            signal_date = f"2020-03-{index:02d}"
            run_id = f"run_{index}"
            record_signal_run(
                conn,
                strategy="demo",
                signal_date=signal_date,
                status="finished",
                config={"manual_confirmation_required": False},
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

    audit = state_report_module.build_report(db, "demo", 3)["paper_observation"]

    assert audit["requested_days"] == 60
    assert len(audit["expected_trading_dates"]) == 60
    assert audit["is_60_day_ready"] is True


def test_state_report_blocks_readiness_without_manual_confirmation(tmp_path):
    db = tmp_path / "state.sqlite3"
    with sqlite3.connect(db) as conn:
        ensure_quant_schema(conn)
        for index in range(60):
            signal_date = f"2020-03-{index + 1:02d}"
            run_id = f"run_{index}"
            record_signal_run(
                conn,
                strategy="demo",
                signal_date=signal_date,
                status="finished",
                config={"manual_confirmation_required": True, "manual_confirmed_at": None},
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

    audit = state_report_module.build_report(db, "demo", 3)["paper_observation"]

    assert audit["is_60_day_ready"] is False
    assert len(audit["manual_confirmation_missing"]) == 60


def test_daily_paper_pipeline_records_stale_data_failure_idempotently(tmp_path):
    db = tmp_path / "state.sqlite3"
    picks = tmp_path / "latest.picks.csv"
    picks.write_text(
        "signal_date,entry_date,symbol,weight,score,entry_open,formula\n"
        "2020-01-03,2020-01-06,000001,0.4,0.9,11,unit\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db) as conn:
        _build_daily_bars_table(conn)
        ensure_quant_schema(conn)
        _insert_raw_day(conn, "2020-01-02")
        conn.commit()

    output, status = build_pipeline_output(
        db=db,
        picks_path=picks,
        strategy="demo",
        cash=1_000_000.0,
        signal_date=None,
        entry_date=None,
        run_id="run_stale",
        dry_run=False,
        allow_stale_data=False,
    )

    assert status == 2
    assert output["status"] == "failed"
    with sqlite3.connect(db) as conn:
        row = conn.execute("select status, message from signal_runs where run_id = 'run_stale'").fetchone()
        attempts = conn.execute("select status, attempt_count from alert_attempts where run_id = 'run_stale'").fetchone()
        registry = conn.execute(
            "select status, data_fresh, observation_days from paper_run_registry where run_id = 'run_stale'"
        ).fetchone()
    assert row[0] == "failed"
    assert "stale" in row[1]
    assert attempts == ("failed", 1)
    assert registry == ("failed", 0, 60)


def test_daily_paper_pipeline_state_report_uses_60_day_default(tmp_path):
    db = tmp_path / "state.sqlite3"
    picks = tmp_path / "latest.picks.csv"
    picks.write_text(
        "signal_date,entry_date,symbol,weight,score,entry_open,formula\n"
        "2020-01-03,2020-01-06,000001,0.4,0.9,11,unit\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db) as conn:
        _build_daily_bars_table(conn)
        ensure_quant_schema(conn)
        _insert_raw_day(conn, "2020-01-03")
        conn.commit()

    output, status = build_pipeline_output(
        db=db,
        picks_path=picks,
        strategy="demo",
        cash=1_000_000.0,
        signal_date=None,
        entry_date=None,
        run_id="run_fresh",
        dry_run=False,
        allow_stale_data=False,
    )

    assert status == 0
    assert output["observation_days"] == 60
    assert output["state_report"]["paper_observation"]["requested_days"] == 60
    event_names = [event["name"] for event in output["run_context"]["events"]]
    latest_run = output["state_report"]["latest_signal_runs"][0]
    latest_paper_run = output["state_report"]["latest_paper_runs"][0]
    assert event_names == ["DATA_READY", "SIGNAL_GENERATED", "ORDER_PLAN_BUILT", "PAPER_RUN_RECORDED"]
    assert output["pipeline_manifest"]["run_context"]["events"][-1]["name"] == "PAPER_RUN_RECORDED"
    assert output["plan"]["object_counts"] == {"signals": 1, "order_intents": 1, "position_snapshots": 1}
    assert latest_run["manual_confirmation_required"] is True
    assert latest_run["manual_confirmed_at"] is None
    assert latest_run["data_delay_days"] == 0
    assert latest_paper_run["status"] == "finished"
    assert latest_paper_run["data_fresh"] is True
    assert latest_paper_run["manifest"]["plan_counts"]["signals"] == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_state_report_requires_consecutive_market_dates_when_available(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_state_report_defaults_to_latest_60_trading_days(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_state_report_blocks_readiness_without_manual_confirmation(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_daily_paper_pipeline_records_stale_data_failure_idempotently(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_daily_paper_pipeline_state_report_uses_60_day_default(Path(directory))

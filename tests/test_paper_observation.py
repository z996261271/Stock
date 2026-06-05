import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.paper.observation import (  # noqa: E402
    expected_trading_dates,
    observation_audit,
    parse_config_json,
)
from quant_schema import ensure_quant_schema  # noqa: E402
from quant_state import record_signal, record_signal_run  # noqa: E402


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


def test_observation_module_audits_consecutive_market_dates(tmp_path):
    db = tmp_path / "state.sqlite3"
    with sqlite3.connect(db) as conn:
        _build_daily_bars_table(conn)
        ensure_quant_schema(conn)
        for index in range(1, 62):
            trade_date = f"2020-03-{index:02d}"
            _insert_raw_day(conn, trade_date)
            run_id = f"run_{index}"
            record_signal_run(
                conn,
                strategy="demo",
                signal_date=trade_date,
                status="finished",
                config={"manual_confirmation_required": False},
                run_id=run_id,
                finished_at=f"{trade_date}T16:00:00",
            )
            record_signal(
                conn,
                strategy="demo",
                symbol="000001",
                signal_date=trade_date,
                action="hold",
                signal_id=f"sig_{index}",
                run_id=run_id,
            )

        assert expected_trading_dates(conn, "2020-03-61", 60)[0] == "2020-03-61"
        audit = observation_audit(conn, "demo", 60)

    assert audit["requested_days"] == 60
    assert audit["expected_trading_dates_available"] is True
    assert audit["is_60_day_ready"] is True
    assert audit["is_90_day_ready"] is False


def test_parse_config_json_marks_invalid_payloads():
    assert parse_config_json('{"manual_confirmation_required": true}')["manual_confirmation_required"] is True
    assert parse_config_json("[1, 2, 3]") == {}
    assert parse_config_json("{broken") == {"_invalid_config_json": True}


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_observation_module_audits_consecutive_market_dates(Path(directory))
    test_parse_config_json_marks_invalid_payloads()

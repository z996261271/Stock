import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from formal_readiness_gate import build_readiness_report  # noqa: E402
from fetch_em_symbol_industries import build_symbol_industry_rows  # noqa: E402
from quant_schema import ensure_quant_schema  # noqa: E402
from validate_formal_reports import build_report as build_formal_report_validation  # noqa: E402


def _insert_bar(conn, symbol: str, trade_date: str, adjust: str) -> None:
    conn.execute(
        """
        insert into daily_bars (
            symbol, trade_date, adjust, open, high, low, close, volume, amount,
            amplitude, pct_chg, chg, turnover, source, fetched_at
        )
        values (?, ?, ?, 10, 11, 9, 10.5, 1000000, 100000000, 0, 0, 0, 1, 'test', 'now')
        """,
        (symbol, trade_date, adjust),
    )


def _build_ready_db(db: Path) -> None:
    with sqlite3.connect(db) as conn:
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
        ensure_quant_schema(conn)
        for symbol in ("000001", "600000"):
            for trade_date in ("2020-01-02", "2020-01-03"):
                for adjust in ("raw", "qfq", "hfq"):
                    _insert_bar(conn, symbol, trade_date, adjust)
                conn.execute(
                    """
                    insert or replace into adj_factors
                    (symbol, trade_date, adj_factor, forward_factor, backward_factor, source, fetched_at)
                    values (?, ?, 1.0, 1.0, 1.0, 'test', 'now')
                    """,
                    (symbol, trade_date),
                )
                conn.execute(
                    """
                    insert or replace into symbol_status_daily
                    (symbol, trade_date, is_st, is_suspended, board, source, fetched_at)
                    values (?, ?, 0, 0, 'main', 'test', 'now')
                    """,
                    (symbol, trade_date),
                )
            conn.execute(
                """
                insert or replace into symbol_lifecycle
                (symbol, name, list_date, board, market, source, fetched_at)
                values (?, 'unit', '1991-01-01', 'main', 'CN', 'test', 'now')
                """,
                (symbol,),
            )
            conn.execute(
                """
                insert or replace into symbol_industries
                (symbol, industry_name, industry_code, industry_level, provider, source, fetched_at)
                values (?, 'bank', '801780', '一级行业', 'test', 'test', 'now')
                """,
                (symbol,),
            )
        conn.commit()


def _formal_payload() -> dict:
    return {
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


def test_invalid_formal_report_family_is_quarantined(tmp_path):
    reports = tmp_path / "formal"
    reports.mkdir()
    (reports / "old.metrics.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (reports / "old.manifest.json").write_text("{}", encoding="utf-8")
    (reports / "old.picks.csv").write_text("symbol\n000001\n", encoding="utf-8")

    report = build_formal_report_validation(
        reports,
        require_formal_valid=True,
        allow_empty=True,
        quarantine_invalid=True,
    )

    assert report["is_valid"] is True
    assert report["metrics_files"] == 0
    assert report["quarantined_issue_files"] == 1
    assert not (reports / "old.metrics.json").exists()
    assert list((reports / "_invalid").rglob("old.metrics.json"))


def test_formal_readiness_gate_requires_data_and_publishable_metrics(tmp_path):
    db = tmp_path / "quality.sqlite3"
    reports = tmp_path / "formal"
    reports.mkdir()
    _build_ready_db(db)
    (reports / "run.metrics.json").write_text(json.dumps(_formal_payload()), encoding="utf-8")
    (reports / "run.manifest.json").write_text("{}", encoding="utf-8")

    report = build_readiness_report(
        db=db,
        start_date="2020-01-02",
        end_date="2020-01-03",
        board_scope="main",
        required_adjusts=("raw", "qfq", "hfq"),
        reports_dir=reports,
    )

    assert report["is_ready"] is True
    assert report["data_quality"]["is_formal_ready"] is True
    assert report["formal_reports"]["is_valid"] is True


def test_quant_schema_migrates_legacy_signals_without_run_id(tmp_path):
    db = tmp_path / "legacy_state.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            create table signals (
                signal_id text primary key,
                strategy text not null,
                symbol text not null,
                signal_date text not null,
                action text not null,
                score real,
                weight real,
                reason text,
                payload_json text,
                created_at text not null
            )
            """
        )
        ensure_quant_schema(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(signals)")}

    assert "run_id" in columns
    assert "idx_signals_run_id" in indexes


def test_em_industry_constituents_convert_to_canonical_rows():
    boards = __import__("pandas").DataFrame(
        [
            {"板块名称": "银行", "板块代码": "BK0475"},
            {"板块名称": "证券", "板块代码": "BK0473"},
        ]
    )
    constituents = {
        "BK0475": __import__("pandas").DataFrame([{"代码": "000001"}, {"代码": "600000"}]),
        "BK0473": __import__("pandas").DataFrame([{"代码": "000001"}, {"代码": "600030"}]),
    }

    rows, duplicates = build_symbol_industry_rows(boards, constituents)

    by_symbol = {row["symbol"]: row for row in rows}
    assert sorted(by_symbol) == ["000001", "600000", "600030"]
    assert by_symbol["000001"]["industry_name"] == "银行"
    assert duplicates == 1


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_invalid_formal_report_family_is_quarantined(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_formal_readiness_gate_requires_data_and_publishable_metrics(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_quant_schema_migrates_legacy_signals_without_run_id(Path(directory))
    test_em_industry_constituents_convert_to_canonical_rows()

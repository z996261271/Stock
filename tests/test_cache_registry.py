import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.data.cache_registry import cache_stale_reasons, parse_cache_filename, scan_cache_dir, write_registry  # noqa: E402


def test_cache_registry_parses_factor_cache_names_and_source_dates(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "walkforward_factors_v4_main_hfq_strict_20170105_20260529.feather"
    cache_file.write_bytes(b"cache-bytes")
    (cache_dir / "cache_registry.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "script.py"
    script.write_text("print('x')\n", encoding="utf-8")
    db = tmp_path / "bars.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("create table daily_bars (trade_date text)")
        conn.execute("insert into daily_bars values ('2026-05-29')")
        conn.execute("create table symbol_status_daily (trade_date text)")
        conn.execute("insert into symbol_status_daily values ('2026-05-28')")
        conn.commit()

    parsed = parse_cache_filename(cache_file)
    entries = scan_cache_dir(cache_dir, db=db, script_path=script)
    output = tmp_path / "cache_registry.json"
    registry = write_registry(entries, output)

    assert parsed["cache_key"] == "walkforward_factors_v4"
    assert parsed["board_scope"] == "main"
    assert parsed["factor_adjust"] == "hfq"
    assert parsed["mode"] == "strict"
    assert entries[0].start_date == "2017-01-05"
    assert entries[0].end_date == "2026-05-29"
    assert entries[0].source_table_max_dates["daily_bars"] == "2026-05-29"
    assert entries[0].source_table_max_dates["symbol_status_daily"] == "2026-05-28"
    assert entries[0].freshness_status == "current"
    assert entries[0].stale_reasons == []
    assert entries[0].script_hash
    assert registry["entry_count"] == 1
    assert output.exists()


def test_cache_registry_marks_trade_date_sources_stale_without_fetched_at_false_positive():
    reasons = cache_stale_reasons(
        "2026-05-29",
        {
            "daily_bars": "2026-06-01",
            "symbol_status_daily": "2026-05-29",
            "symbol_industries": "2026-06-05T12:00:00",
        },
    )

    assert reasons == ["daily_bars:source_max_date=2026-06-01>cache_end_date=2026-05-29"]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_cache_registry_parses_factor_cache_names_and_source_dates(Path(directory))
    test_cache_registry_marks_trade_date_sources_stale_without_fetched_at_false_positive()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.cli import repo_root, resolve_command  # noqa: E402


def test_cli_router_resolves_supported_script_commands():
    command, forwarded = resolve_command(["data", "quality", "--db", "x.sqlite3"])

    assert command.script == "scripts/data_quality_report.py"
    assert forwarded == ["--db", "x.sqlite3"]


def test_cli_router_uses_repository_root_for_script_paths():
    assert (repo_root() / "scripts" / "quality_gate.py").exists()


if __name__ == "__main__":
    test_cli_router_resolves_supported_script_commands()
    test_cli_router_uses_repository_root_for_script_paths()

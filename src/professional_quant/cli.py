"""Unified command router for common professional quant scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScriptCommand:
    topic: str
    action: str
    script: str
    description: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.topic, self.action)


COMMANDS = (
    ScriptCommand("data", "quality", "scripts/data_quality_report.py", "Run data quality checks."),
    ScriptCommand("data", "cache-registry", "scripts/cache_registry.py", "Build the local cache registry."),
    ScriptCommand("backtest", "formal", "scripts/run_formal_dynamic.py", "Run the formal dynamic backtest wrapper."),
    ScriptCommand("report", "factor", "scripts/generate_factor_report.py", "Generate factor diagnostics."),
    ScriptCommand("report", "performance", "scripts/generate_performance_report.py", "Generate performance tear sheet."),
    ScriptCommand("paper", "run", "scripts/run_daily_paper_pipeline.py", "Run the daily paper pipeline."),
    ScriptCommand("paper", "status", "scripts/state_report.py", "Inspect paper trading state."),
)
COMMAND_MAP = {command.key: command for command in COMMANDS}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help()
        return 0
    if args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if args[0] == "commands":
        print_commands()
        return 0
    if len(args) < 2:
        parser.error("expected <topic> <action>; use 'commands' to list supported commands")
    dry_run = False
    if "--dry-run-router" in args:
        dry_run = True
        args.remove("--dry-run-router")
    command, forwarded = resolve_command(args)
    script_path = repo_root() / command.script
    run_argv = [sys.executable, str(script_path), *forwarded]
    if dry_run:
        print(" ".join(run_argv))
        return 0
    return subprocess.run(run_argv, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m professional_quant",
        description="Route common project workflows through one package entrypoint.",
    )
    parser.add_argument("topic", nargs="?", help="Command topic, for example data, backtest, report, paper.")
    parser.add_argument("action", nargs="?", help="Command action, for example quality, formal, run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying script.")
    parser.add_argument("--dry-run-router", action="store_true", help="Print routed script command without executing it.")
    return parser


def resolve_command(argv: list[str]) -> tuple[ScriptCommand, list[str]]:
    key = (argv[0], argv[1])
    command = COMMAND_MAP.get(key)
    if command is None:
        known = ", ".join(f"{item.topic} {item.action}" for item in COMMANDS)
        raise SystemExit(f"unknown command: {' '.join(key)}; known commands: {known}")
    return command, argv[2:]


def print_commands() -> None:
    for command in COMMANDS:
        print(f"{command.topic} {command.action}\t{command.script}\t{command.description}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())

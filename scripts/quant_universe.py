#!/usr/bin/env python3
"""Stock-universe helpers shared by daily A-share research scripts."""

from __future__ import annotations

MAIN_BOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")


def is_main_board_symbol(symbol: str) -> bool:
    """Return whether a six-digit A-share code is in the Shanghai/Shenzhen main board."""
    normalized = str(symbol).strip()
    return normalized.startswith(MAIN_BOARD_PREFIXES)


def board_scope_sql(scope: str, alias: str = "d") -> tuple[str, tuple[str, ...]]:
    """Return a SQLite predicate fragment and parameters for the requested board scope."""
    if scope == "all":
        return "", ()
    if scope != "main":
        raise ValueError(f"unknown board scope: {scope}")
    if not alias.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQL alias: {alias}")
    placeholders = ", ".join("?" for _ in MAIN_BOARD_PREFIXES)
    return f" and substr({alias}.symbol, 1, 3) in ({placeholders})", MAIN_BOARD_PREFIXES

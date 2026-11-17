"""TRUTH FIREWALL: nothing outside backend/sim may import _truth."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# My own area may import it; nothing else may.
ALLOWED_PREFIXES = ("backend/sim/", "tests/sim/", "docs/")


def test_no_truth_imports_outside_backend_sim() -> None:
    offenders: list[str] = []
    skip_dirs = {".venv", ".git", "node_modules", ".pytest_cache", ".ruff_cache"}
    for path in ROOT.rglob("*.py"):
        if skip_dirs & set(path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(ALLOWED_PREFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "_truth" in text:
            offenders.append(rel)
    assert not offenders, f"TRUTH FIREWALL violated by: {offenders}"

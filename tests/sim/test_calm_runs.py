"""Calm-run behaviour tests: calendar, curves, performance (ERRATA 1-6)."""

from __future__ import annotations

import gzip
import json
import time
import zoneinfo
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from backend.sim.contracts_stub import ScenarioSpec
from backend.sim.serialize import run_scenario
from scripts.make_calm_run import DEFAULT_START, SYNTHETIC_HOLIDAY, make_calm_run

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# (seed, expected day_type by day index Thu..Sun)
DAY_TYPES = [
    (1001, ["weekday", "weekday", "weekend", "weekend"]),
    (1002, ["weekday", "weekday", "weekend", "weekend"]),
    (1003, ["weekday", "weekday", "weekend", "weekend"]),
    (1004, ["weekday", "weekday", "weekend", "weekend"]),
    (1005, ["weekday", "holiday", "weekend", "weekend"]),
]

EXPECTED_START = datetime(2026, 11, 19, 0, 0, tzinfo=IST)


@pytest.fixture(scope="module")
def run1001() -> list[dict]:
    return run_scenario(ScenarioSpec(run_id="c1001", seed=1001, days=4))


@pytest.mark.parametrize("seed,expected", DAY_TYPES)
def test_day_type_on_every_tick(seed: int, expected: list[str]) -> None:
    envs = run_scenario(ScenarioSpec(run_id=f"calm-{seed}", seed=seed, days=4))
    assert len(envs) == 4 * 1440
    for env in envs:
        got = env["payload"]["context"]["day_type"]
        want = expected[env["t"] // 1440]
        assert got == want, f"t={env['t']} day_type={got} want={want}"
        assert env["payload"]["context"]["is_holiday"] == (got == "holiday")


def test_run_starts_thu_2026_11_19(run1001: list[dict]) -> None:
    assert datetime.fromisoformat(run1001[0]["ts"]) == EXPECTED_START
    assert date(2026, 11, 19).weekday() == 3  # Thursday
    assert run1001[-1]["t"] == 4 * 1440 - 1


def test_weekend_morning_traffic_peak_20pct_below_weekday(run1001: list[dict]) -> None:
    """Morning-window traffic peak: weekend must be >=20% below weekday.

    Morning window is 05:30-13:00 IST to cover the +45 min weekend shift
    (ERRATA item 3). Values averaged over zones within a tick.
    """
    peak: dict[str, float] = {}
    for label, days in (("weekday", (0, 1)), ("weekend", (2, 3))):
        hourly: dict[int, list[float]] = defaultdict(list)
        for d in days:
            for env in run1001[d * 1440 : (d + 1) * 1440]:
                minute = env["t"] % 1440
                if not (330 <= minute < 780):
                    continue
                vals = [r["value"] for r in env["payload"]["readings"] if r["system"] == "traffic"]
                hourly[minute // 60].append(float(np.mean(vals)))
        peak[label] = max(float(np.mean(v)) for v in hourly.values())
    assert peak["weekend"] <= 0.8 * peak["weekday"], peak


def test_holiday_matches_weekend_curves() -> None:
    """Seed 1005: Fri (holiday) traffic/water/power hourly profiles must
    match the Sat/Sun profiles (same curve family per ERRATA item 3)."""
    envs = run_scenario(ScenarioSpec(run_id="calm-1005", seed=1005, days=4))
    profile: dict[str, dict[int, float]] = {}
    for sysname in ("traffic", "power", "water"):
        for label, days in (("holiday", (1,)), ("weekend", (2, 3))):
            sums: dict[int, list[float]] = defaultdict(list)
            for d in days:
                for env in envs[d * 1440 : (d + 1) * 1440]:
                    h = (env["t"] % 1440) // 60
                    vals = [
                        r["value"] for r in env["payload"]["readings"] if r["system"] == sysname
                    ]
                    sums[h].append(float(np.mean(vals)))
            profile[f"{sysname}/{label}"] = {h: float(np.mean(v)) for h, v in sums.items() if v}
    for sysname in ("traffic", "power", "water"):
        hol = np.array([profile[f"{sysname}/holiday"][h] for h in range(24)])
        wkd = np.array([profile[f"{sysname}/weekend"][h] for h in range(24)])
        corr = float(np.corrcoef(hol, wkd)[0, 1])
        assert corr > 0.99, f"{sysname}: holiday/weekend correlation {corr:.4f}"
        assert float(np.abs(hol - wkd).max()) < 0.08, sysname


def test_calm_run_file_written(tmp_path: Path) -> None:
    """Script writes a gzipped 5760-line JSONL of valid envelopes."""
    out = make_calm_run(1001, tmp_path)
    assert out.name == "calm_run_seed1001.jsonl.gz"
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 5760
    first = json.loads(lines[0])
    assert first["channel"] == "sim.tick"
    assert first["run_id"] == "calm-1001"
    assert first["t"] == 0
    assert len(first["payload"]["readings"]) == 24
    last = json.loads(lines[-1])
    assert last["t"] == 5759


def test_generation_speed() -> None:
    """One 4-day run must generate in under 30 s (ERRATA 'otherwise vectorize')."""
    t0 = time.perf_counter()
    envs = run_scenario(ScenarioSpec(run_id="perf", seed=1003, days=4))
    elapsed = time.perf_counter() - t0
    assert len(envs) == 5760
    assert elapsed < 30.0, f"generation took {elapsed:.1f}s"
    print(f"\n4-day run generated in {elapsed:.2f}s")


def test_calendar_and_spec_dates_consistent() -> None:
    assert DEFAULT_START == "2026-11-19"
    assert SYNTHETIC_HOLIDAY == "2026-11-20"
    cal = json.loads((Path("data/calendar.json")).read_text(encoding="utf-8"))
    assert cal["synthetic"] is True
    dates = [h["date"] for h in cal["holidays"]]
    assert SYNTHETIC_HOLIDAY in dates
    for h in cal["holidays"]:
        assert h["source"] == "synthetic"

"""Envelope structure tests for the sim.tick channel."""

from __future__ import annotations

import json
import zoneinfo
from datetime import date, datetime, timedelta

import pytest

from backend.sim.contracts_stub import ScenarioSpec
from backend.sim.serialize import run_scenario

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

ZONE_ORDER = [
    "silk_board",
    "whitefield",
    "hebbal",
    "electronic_city",
    "koramangala",
    "indiranagar",
    "jayanagar",
    "yelahanka",
]
SYSTEMS = ("traffic", "power", "water")
METRICS = {"traffic": "flow", "power": "load", "water": "demand"}
UNITS = {"traffic": "veh_per_min", "power": "MW", "water": "kL_per_min"}


@pytest.fixture(scope="module")
def day1() -> list[dict]:
    spec = ScenarioSpec(run_id="test-1001", seed=1001, days=1)
    return run_scenario(spec)


def test_envelope_shape(day1: list[dict]) -> None:
    env = day1[0]
    assert set(env) == {"v", "channel", "run_id", "t", "ts", "zone", "payload"}
    assert env["v"] == 1
    assert env["channel"] == "sim.tick"
    assert env["t"] == 0
    assert env["zone"] is None
    assert set(env["payload"]) == {"readings", "context", "reports"}


def test_timestamps_are_ist_iso(day1: list[dict]) -> None:
    start = datetime(2026, 11, 19, 0, 0, tzinfo=IST)
    for env in day1[:60]:
        ts = datetime.fromisoformat(env["ts"])
        assert ts == start + timedelta(minutes=env["t"])
        assert ts.utcoffset() == timedelta(hours=5, minutes=30)


def test_exactly_24_readings_ordered(day1: list[dict]) -> None:
    for env in day1[::120]:
        readings = env["payload"]["readings"]
        assert len(readings) == 24
        expected = [(z, s) for z in ZONE_ORDER for s in SYSTEMS]
        assert [(r["zone"], r["system"]) for r in readings] == expected


def test_reading_fields_no_nulls(day1: list[dict]) -> None:
    for env in day1[::120]:
        for r in env["payload"]["readings"]:
            assert set(r) == {"zone", "system", "metric", "value", "raw", "unit", "sensor_id"}
            assert all(v is not None for v in r.values())
            assert r["metric"] == METRICS[r["system"]]
            assert r["unit"] == UNITS[r["system"]]
            assert r["sensor_id"] == f"{r['zone']}.{r['system']}.01"
            assert 0.0 <= r["value"] <= 1.3


def test_value_rounded_4_raw_rounded_3(day1: list[dict]) -> None:
    for env in day1[::60]:
        for r in env["payload"]["readings"]:
            assert round(r["value"], 4) == r["value"]
            assert round(r["raw"], 3) == r["raw"]


def test_context_fields(day1: list[dict]) -> None:
    for env in day1[::97]:
        ctx = env["payload"]["context"]
        assert set(ctx) == {"temp_c", "events", "is_holiday", "day_type"}
        assert isinstance(ctx["temp_c"], float)
        assert isinstance(ctx["events"], list)
        assert isinstance(ctx["is_holiday"], bool)
        assert ctx["day_type"] in {"weekday", "weekend", "holiday"}


def test_reports_are_synthetic(day1: list[dict]) -> None:
    for env in day1:
        for rep in env["payload"]["reports"]:
            assert set(rep) == {"id", "zone", "text", "lang", "source"}
            assert rep["source"] == "synthetic"
            assert rep["lang"] in {"en", "kn"}
            assert rep["zone"] in ZONE_ORDER


def test_envelopes_validate_against_stub_model(day1: list[dict]) -> None:
    from backend.sim.contracts_stub import Envelope

    for env in day1[::180]:
        Envelope.model_validate(env)


def test_jsonl_roundtrip(day1: list[dict]) -> None:
    """Envelope must survive a JSON round-trip without losing fields."""
    env = json.loads(json.dumps(day1[720]))
    assert env == day1[720]
    assert date.fromisoformat(env["ts"][:10]) == date(2026, 11, 19)

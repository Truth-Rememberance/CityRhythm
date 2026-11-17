import time

import numpy as np
import pytest

import backend.sync.module as sync_module
from backend.sync.csd import build_csd_baseline
from backend.sync.fake_ticks import generate
from backend.sync.module import SyncModule
from backend.sync.sync_math import build_profile, detrend


@pytest.fixture(scope="module")
def calibration():
    runs = []
    for seed in (1, 3, 5):
        rows = []
        ticks = []
        for envelope in generate(seed, 3):
            rows.append([reading["value"] for reading in envelope["payload"]["readings"]])
            ticks.append(envelope["t"])
        runs.append((np.asarray(rows), np.asarray(ticks)))

    profiles = {
        "weekday": build_profile(
            np.vstack([values for values, _ticks in runs]),
            np.concatenate([ticks for _values, ticks in runs]),
        )
    }
    residual_runs = [
        detrend(values, ticks, profiles["weekday"])
        for values, ticks in runs
    ]
    baseline = build_csd_baseline(residual_runs)
    return baseline, profiles


def _module(calibration):
    baseline, profiles = calibration
    return SyncModule(baseline=baseline, profiles=profiles)


def test_calm_run_stays_quiet(calibration):
    module = _module(calibration)
    critical_warnings = []
    index_values = []
    for tick in generate(4, 3):
        for event in module.on_tick(tick, None):
            if event["channel"] == "sync.warning" and event["payload"]["level"] == "critical":
                critical_warnings.append(event)
            if event["channel"] == "sync.index":
                index_values.append(event["payload"]["sync_index"])

    assert critical_warnings == []
    assert sum(value < 70 for value in index_values) / len(index_values) < 0.02


def test_stress_warning_precedes_silk_board_baseline_alarm(calibration):
    module = _module(calibration)
    first_warning = None
    first_alarm = None
    for tick in generate(2, 3, stress_from=2000):
        for event in module.on_tick(tick, None):
            if event["zone"] != "silk_board":
                continue
            if event["channel"] == "sync.warning" and first_warning is None:
                first_warning = tick["t"]
            if event["channel"] == "baseline.alarm" and first_alarm is None:
                first_alarm = tick["t"]
        if first_warning is not None and first_alarm is not None:
            break

    assert first_warning is not None
    assert first_alarm is not None
    assert first_warning >= 2000
    assert first_warning < first_alarm


def test_sync_index_events_have_contract_fields_and_valid_range(calibration):
    module = _module(calibration)
    expected = {
        "zone", "sync_index", "status", "phase_coherence",
        "csd_score", "coupling_break", "phases",
    }
    index_events = []
    for tick in generate(6, 1):
        index_events.extend(
            event for event in module.on_tick(tick, None)
            if event["channel"] == "sync.index"
        )
        if len(index_events) >= 8 * 200:
            break

    assert index_events
    for event in index_events:
        assert set(event["payload"]) == expected
        assert 0.0 <= event["payload"]["sync_index"] <= 100.0


def test_average_on_tick_time_is_under_200_ms(calibration):
    module = _module(calibration)
    ticks = generate(7, 1)
    elapsed = 0.0
    count = 0
    for tick in ticks:
        started = time.perf_counter()
        module.on_tick(tick, None)
        elapsed += time.perf_counter() - started
        count += 1
        if count == 500:
            break

    assert count == 500
    assert elapsed / count < 0.2


def test_empty_readings_return_no_events(calibration):
    module = _module(calibration)
    tick = {
        "run_id": "empty-readings",
        "t": 0,
        "payload": {"readings": [], "context": {}},
    }
    assert module.on_tick(tick, None) == []


def test_warning_waits_for_persist_duration(calibration, monkeypatch):
    module = _module(calibration)
    monkeypatch.setattr(sync_module, "status", lambda _index, _bands: "drifting")
    warning_events = []
    for tick in generate(8, 1):
        if tick["t"] >= module.warn_persist_min - 1:
            break
        warning_events.extend(
            event for event in module.on_tick(tick, None)
            if event["channel"] == "sync.warning"
            and event["payload"].get("baseline_fired") is False
        )

    assert warning_events == []


def test_warning_fires_when_persist_duration_is_reached(calibration, monkeypatch):
    module = _module(calibration)
    monkeypatch.setattr(sync_module, "status", lambda _index, _bands: "drifting")
    warning_events = []
    for tick in generate(8, 1):
        warning_events.extend(
            event for event in module.on_tick(tick, None)
            if event["channel"] == "sync.warning"
            and event["payload"].get("baseline_fired") is False
        )
        if tick["t"] >= module.warn_persist_min - 1:
            break

    silk_board_warnings = [
        event for event in warning_events if event["zone"] == "silk_board"
    ]
    assert len(silk_board_warnings) == 1

from __future__ import annotations

import numpy as np
import pytest

from backend.fix.metrics import (
    cut_percent,
    did_recover,
    recovery_time,
    resilience_curve,
    resilience_score,
    service_level,
    time_to_collapse,
)


def test_service_level_is_full_below_threshold_and_penalizes_joint_stress() -> None:
    assert service_level({"traffic": 0.7, "power": 0.7, "water": 0.7}) == pytest.approx(1.0)
    single_high = service_level({"traffic": 0.9, "power": 0.7, "water": 0.7})
    joint_high = service_level({"traffic": 0.9, "power": 0.9, "water": 0.7})
    assert joint_high < single_high
    assert service_level({"traffic": 1.0, "power": 1.0, "water": 1.0}) == pytest.approx(0.17)


def test_service_level_applies_capacity_and_configurable_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("backend.fix.metrics", fromlist=["SERVICE_WEIGHTS"]).SERVICE_WEIGHTS,
                        "traffic", 1.0)
    monkeypatch.setitem(__import__("backend.fix.metrics", fromlist=["SERVICE_WEIGHTS"]).SERVICE_WEIGHTS,
                        "power", 0.0)
    monkeypatch.setitem(__import__("backend.fix.metrics", fromlist=["SERVICE_WEIGHTS"]).SERVICE_WEIGHTS,
                        "water", 0.0)
    result = service_level({"traffic": 0.8}, capacity={"traffic": 0.8})
    assert result == pytest.approx(1.0 - (0.8 / 0.3) * (0.3 - 0.0))


def test_resilience_score_uses_only_incident_window_and_clips() -> None:
    assert resilience_score(np.array([1.0, 0.5, 0.5]), incident_start=1) == pytest.approx(50.0)
    assert resilience_score(np.array([2.0, 2.0])) == pytest.approx(100.0)
    assert resilience_score(np.array([0.0, 0.0]), baseline=0.0) == 0.0


def test_resilience_curve_is_trailing_rolling_mean() -> None:
    curve = resilience_curve(np.array([1.0, 0.5, 0.0]), window=2)
    np.testing.assert_allclose(curve, np.array([100.0, 75.0, 25.0]))


def test_recovery_requires_sustained_threshold_and_reports_cap() -> None:
    series = np.array([1.0, 0.8, 0.96, 0.94, 0.97, 0.98, 0.99])
    assert recovery_time(series, incident_start=0, threshold=0.95, sustain_min=3) == 3
    assert did_recover(series, incident_start=0, threshold=0.95, sustain_min=3)
    assert recovery_time(np.array([1.0, 0.8, 0.9]), 0, sustain_min=2) == 3
    assert not did_recover(np.array([1.0, 0.8, 0.9]), 0, sustain_min=2)


def test_recovery_handles_no_drop_start_below_and_short_series() -> None:
    assert recovery_time(np.array([0.99, 1.0]), 0, sustain_min=4) == 0
    assert recovery_time(np.array([0.8, 0.99, 0.99]), 0, sustain_min=2) == 1
    assert recovery_time(np.array([0.8, 0.99]), 0, sustain_min=3, cap=7) == 7


def test_time_to_collapse_and_safe_cut_percent() -> None:
    assert time_to_collapse(np.array([0.9, 0.6, 0.4])) == 2
    assert time_to_collapse(np.array([1.0, 0.7])) is None
    assert cut_percent(20.0, 5.0) == pytest.approx(75.0)
    assert cut_percent(0.0, 0.0) == 0.0
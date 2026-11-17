from __future__ import annotations

import numpy as np
import pytest

from backend.fix._local_contracts import Action
from backend.fix.rephase import (
    DEFAULT_FLEXIBILITY,
    _shift_movable,
    forecast_from_history,
    make_demo_profiles,
    peak_overlap,
    rephase_optimizer,
)


def test_shift_is_non_circular_and_preserves_energy_at_edge() -> None:
    profile = np.array([0.0, 0.2, 0.4, 0.6])
    shifted = _shift_movable(profile, 1, 1.0)
    np.testing.assert_allclose(shifted, np.array([0.0, 0.0, 0.2, 1.0]))
    assert float(np.sum(shifted)) == pytest.approx(float(np.sum(profile)))


def test_peak_overlap_counts_coincidence_and_overload() -> None:
    profiles = {
        "silk_board.traffic": np.array([0.9, 0.2]),
        "silk_board.power": np.array([0.9, 0.9]),
        "silk_board.water": np.array([0.2, 0.9]),
    }
    stats = peak_overlap(profiles)
    assert stats.coincidence_minutes == 2
    assert stats.overload_area == pytest.approx(0.2)
    assert stats.peak == pytest.approx(0.9)
    assert stats.score == pytest.approx(4.0)


def test_peak_overlap_optional_edges_add_lagged_contagion() -> None:
    profiles = {
        "silk_board.traffic": np.array([0.9, 0.2, 0.2]),
        "whitefield.power": np.array([0.2, 0.7, 0.2]),
        "whitefield.water": np.array([0.2, 0.81, 0.2]),
    }
    without_edges = peak_overlap(profiles, edges=None)
    with_edges = peak_overlap(
        profiles,
        edges=[
            {"src": "silk_board.traffic", "dst": "whitefield.power", "lag_min": 1, "strength": 1.0}
        ],
    )
    assert without_edges.coincidence_minutes == 0
    assert with_edges.coincidence_minutes == 1
    assert with_edges.score > without_edges.score


def test_demo_optimizer_reduces_score_and_returns_valid_bounded_actions() -> None:
    profiles = make_demo_profiles(seed=7)
    result = rephase_optimizer(profiles, budget_ms=1500, seed=7, max_actions=6, now=50)
    assert result.before_stats.score > 0.0
    assert result.after_stats.score < result.before_stats.score
    assert result.peak_overlap_cut_pct >= 20.0
    assert len(result.actions) <= 6
    assert all(isinstance(action, Action) for action in result.actions)
    assert all(action.start_t == 50 and action.duration_min == 360 for action in result.actions)
    for node, shift in result.shifts.items():
        system = node.rsplit(".", 1)[-1]
        assert abs(shift) <= DEFAULT_FLEXIBILITY[system][1]
    assert result.runtime_ms < 1500 * 1.5


def test_optimizer_is_deterministic_for_same_seed() -> None:
    profiles = make_demo_profiles(seed=3)
    first = rephase_optimizer(profiles, budget_ms=700, seed=19)
    second = rephase_optimizer(profiles, budget_ms=700, seed=19)
    assert first.shifts == second.shifts
    assert first.before_stats == second.before_stats
    assert first.after_stats == second.after_stats


def test_empty_and_flat_profiles_produce_no_actions() -> None:
    assert rephase_optimizer({}).actions == []
    flat = {
        "silk_board.traffic": np.full(120, 0.4),
        "silk_board.power": np.full(120, 0.4),
    }
    result = rephase_optimizer(flat)
    assert result.actions == []
    assert result.shifts == {}
    assert result.before_stats == result.after_stats


def test_custom_flexibility_bounds_shift_and_configures_action() -> None:
    profiles = {
        "silk_board.traffic": np.r_[np.full(20, 0.2), np.full(20, 0.95), np.full(20, 0.2)],
        "silk_board.power": np.r_[np.full(20, 0.2), np.full(20, 0.95), np.full(20, 0.2)],
    }
    flexibility = {"silk_board.traffic": (0.5, 10, "offset_signal", "traffic")}
    result = rephase_optimizer(profiles, flexibility=flexibility, budget_ms=300)
    assert all(abs(result.shifts.get("silk_board.traffic", 0)) <= 10 for _ in [0])
    if result.actions:
        assert result.actions[0].type in ("offset_signal", "shift_schedule")


def test_forecast_from_history_handles_empty_and_blends_seasonal_pattern() -> None:
    np.testing.assert_array_equal(forecast_from_history([], 4), np.zeros(4))
    forecast = forecast_from_history([0.1, 0.2, 0.3, 0.4], horizon=4, period=4)
    assert forecast.shape == (4,)
    assert forecast[0] != 0.4
    assert forecast[-1] == pytest.approx(0.4)
    unaligned = forecast_from_history([0.1, 0.2, 0.3, 0.4, 0.5], horizon=11, period=4)
    assert unaligned[-1] == pytest.approx(0.4)
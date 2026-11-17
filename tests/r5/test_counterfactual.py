from __future__ import annotations

import time

from backend.fix._local_contracts import ZONES, Action, CounterfactualPayload
from backend.fix.agents import Proposal
from backend.fix.conflicts import Plan, Resolution, build_plan
from backend.fix.counterfactual import (
    CounterfactualCoordinator,
    InterventionWindow,
)
from backend.fix.testing.toy_sim import ToySim


def make_plan(start_t: int, zone: str = "silk_board") -> Plan:
    actions = [
        Action(
            id=f"{zone}-traffic",
            type="offset_signal",
            dept="traffic",
            zone=zone,
            params={"offset_s": 60},
            start_t=start_t,
            duration_min=90,
        ),
        Action(
            id=f"{zone}-power",
            type="shed_load",
            dept="power",
            zone=zone,
            params={"pct": 80, "feeder": "main"},
            start_t=start_t,
            duration_min=90,
        ),
        Action(
            id=f"{zone}-water",
            type="shift_pump",
            dept="water",
            zone=zone,
            params={"shift_min": 1, "movable_fraction": 0.8},
            start_t=start_t,
            duration_min=90,
        ),
    ]
    proposals = [
        Proposal(
            action=action,
            utility=1.0,
            rationale="Test intervention.",
            touches=[f"{zone}.{system}"],
            expected_load_delta={f"{zone}.{system}": -0.1},
        )
        for action, system in zip(actions, ("traffic", "power", "water"), strict=True)
    ]
    return build_plan(Resolution(proposals, [], [], [], total_cost=9.0), start_t)


def test_toy_sim_forks_are_identical_and_independent() -> None:
    sim = ToySim(seed=17, start_t=1199)
    sim.inject("heatwave", start_t=1200, duration_min=20, zone="silk_board")
    baseline_snapshot = sim.snapshot()
    first, second = sim.fork(), sim.fork()
    assert first.step() == second.step()
    first.inject("signal_hijack", zone="silk_board", start_t=1201)
    assert sim.snapshot() == baseline_snapshot
    assert first.snapshot() != second.snapshot()


def test_counterfactual_plan_strictly_improves_heatwave_recovery() -> None:
    sim = ToySim(seed=5, start_t=1199)
    sim.inject("heatwave", start_t=1200, duration_min=75, zone="silk_board")
    result = CounterfactualCoordinator(sim, horizon_min=180).run(
        make_plan(1200), sim_now=1199, incident_start=1200
    )
    assert result["recovery_min"]["with_plan"] < result["recovery_min"]["no_action"]
    assert result["extras"]["no_action_recovered"]
    assert result["extras"]["with_plan_recovered"]
    assert result["recovery_min"] == {"no_action": 94.0, "with_plan": 0.0}


def test_payload_validates_and_all_series_are_time_aligned() -> None:
    sim = ToySim(seed=5, start_t=1199)
    sim.inject("heatwave", start_t=1200, duration_min=75, zone="silk_board")
    result = CounterfactualCoordinator(sim, horizon_min=60).run(
        make_plan(1200), sim_now=1199, incident_start=1200
    )
    payload = CounterfactualPayload.model_validate(
        {key: value for key, value in result.items() if key != "extras"}
    )
    assert len(payload.t) == 60
    assert len(payload.no_action.service_level) == len(payload.t)
    assert len(payload.with_plan.service_level) == len(payload.t)
    assert len(payload.no_action.resilience) == len(payload.t)
    assert len(payload.with_plan.resilience) == len(payload.t)
    assert set(result["extras"]["per_zone"]["no_action"]) == set(ZONES)
    assert all(
        len(curve) == 60
        for curve in result["extras"]["per_zone_resilience"]["with_plan"].values()
    )


def test_observe_is_cheap_and_checkpoint_ring_is_bounded() -> None:
    sim = ToySim(seed=2, start_t=1200)
    coordinator = CounterfactualCoordinator(
        sim, horizon_min=20, checkpoint_every=1, max_checkpoints=3
    )
    started = time.perf_counter()
    observation_times: list[float] = []
    for t in range(1200, 1205):
        tick = sim.step()
        observe_started = time.perf_counter()
        coordinator.observe(sim, tick.t)
        observation_times.append((time.perf_counter() - observe_started) * 1000)
    assert max(observation_times) < 5.0
    assert (time.perf_counter() - started) * 1000 > 0.0
    assert [point.t for point in coordinator._checkpoints] == [1203, 1204, 1205]


def test_intervention_window_reports_effective_checkpoints_and_root_cause() -> None:
    sim = ToySim(seed=8, start_t=1199)
    sim.inject("heatwave", start_t=1200, duration_min=90, boost=0.55)
    coordinator = CounterfactualCoordinator(
        sim, horizon_min=90, checkpoint_every=1, max_checkpoints=12
    )
    for _ in range(12):
        tick = sim.step()
        coordinator.observe(sim, tick.t)

    window = coordinator.intervention_window(lambda t: make_citywide_plan(t), 1200)
    assert isinstance(window, InterventionWindow)
    assert window.first_checkpoint_t is not None
    assert window.last_effective_t is not None
    baseline = coordinator.run(Plan("none", [], 0.0, {}, []), 1200, 1200)
    collapse_index = next(
        index
        for index, value in enumerate(baseline["no_action"]["service_level"])
        if value < 0.5
    )
    collapse_time = baseline["t"][collapse_index]
    assert window.first_checkpoint_t <= window.last_effective_t <= collapse_time
    assert window.root_cause_t is not None
    assert window.root_cause_node is not None
    assert window.saved_by_t


def test_async_run_matches_sync_run_and_drops_overlapping_job() -> None:
    sim = ToySim(seed=5, start_t=1199)
    sim.inject("heatwave", start_t=1200, duration_min=30, zone="silk_board")
    coordinator = CounterfactualCoordinator(sim, horizon_min=20)
    plan = make_plan(1200)
    expected = coordinator.run(plan, 1199, 1200)
    assert coordinator.run_async(plan, 1199, 1200)
    assert not coordinator.run_async(plan, 1199, 1200)
    deadline = time.perf_counter() + 3.0
    result = None
    while time.perf_counter() < deadline:
        result = coordinator.poll()
        if result is not None:
            break
        time.sleep(0.001)
    assert result == expected
    assert not coordinator.is_running()


def test_to_envelopes_uses_frozen_metric_channels() -> None:
    sim = ToySim(seed=4, start_t=1199)
    coordinator = CounterfactualCoordinator(sim, horizon_min=4)
    result = coordinator.run(Plan("empty", [], 0.0, {}, []), 1199, 1200)
    envelopes = coordinator.to_envelopes(result, "r_01", 1200)
    assert [envelope.channel for envelope in envelopes] == ["metrics.counterfactual"]
    assert envelopes[0].payload["recovery_min"] == result["recovery_min"]

    window = InterventionWindow(1200, 1200, [], 1202, "silk_board.traffic")
    window_envelope = coordinator.to_envelopes(window, "r_01", 1200)[0]
    assert window_envelope.channel == "metrics.intervention_window"


def make_citywide_plan(start_t: int) -> Plan:
    actions: list[Action] = []
    for zone in ZONES:
        actions.extend(
            [
                Action(
                    id=f"{zone}-traffic",
                    type="offset_signal",
                    dept="traffic",
                    zone=zone,
                    params={"offset_s": 60},
                    start_t=start_t,
                    duration_min=120,
                ),
                Action(
                    id=f"{zone}-power",
                    type="shed_load",
                    dept="power",
                    zone=zone,
                    params={"pct": 80},
                    start_t=start_t,
                    duration_min=120,
                ),
                Action(
                    id=f"{zone}-water",
                    type="shift_pump",
                    dept="water",
                    zone=zone,
                    params={"shift_min": 180, "movable_fraction": 0.8},
                    start_t=start_t,
                    duration_min=120,
                ),
            ]
        )
    return Plan("citywide", actions, 72.0, {}, [])
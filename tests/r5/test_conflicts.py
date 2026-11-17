from __future__ import annotations

from typing import Any

from backend.fix._local_contracts import Action
from backend.fix.agents import Proposal, TrafficAgent, ZoneSnapshot
from backend.fix.conflicts import (
    build_plan,
    detect_conflicts,
    resolve,
)
from backend.fix.fragility import load_costs


def proposal(
    proposal_id: str,
    action_type: str,
    zone: str,
    node: str,
    delta: float,
    utility: float = 1.0,
    **params: Any,
) -> Proposal:
    action = Action(
        id=proposal_id,
        type=action_type,
        dept="traffic" if action_type in ("offset_signal", "reroute") else "power",
        zone=zone,
        params=params,
        start_t=0,
        duration_min=15,
    )
    return Proposal(
        action=action,
        utility=utility,
        rationale="Test proposal.",
        touches=[node],
        expected_load_delta={node: delta},
    )


def test_reroute_overload_is_detected_and_resolver_compromises() -> None:
    costs = load_costs()
    snapshot = ZoneSnapshot(
        zone="silk_board",
        t=20,
        loads={
            "traffic": 0.9,
            "power": 0.5,
            "water": 0.4,
        },
        sync_index=20.0,
        neighbor_loads={"koramangala": 0.3, "electronic_city": 0.2, "jayanagar": 0.4},
    )
    reroute = next(
        item for item in TrafficAgent().propose(snapshot, costs) if item.action.type == "reroute"
    )
    loads = {
        "silk_board.traffic": 0.9,
        "electronic_city.traffic": 0.5,
        "electronic_city.power": 0.84,
    }
    detected = detect_conflicts([reroute], loads)
    assert len(detected) == 1
    assert detected[0].kind == "overload_transfer"
    resolution = resolve([reroute], detected, loads, costs)
    assert len(resolution.accepted) == 1
    assert len(resolution.modified) == 1
    modified = resolution.accepted[0]
    assert modified.action.params["fraction"] < reroute.action.params["fraction"]
    assert loads["electronic_city.power"] + modified.expected_load_delta[
        "electronic_city.power"
    ] <= 0.85 + 1e-12


def test_detects_same_node_and_opposing_effects() -> None:
    first = proposal("p1", "offset_signal", "silk_board", "silk_board.traffic", -0.1)
    duplicate = proposal("p2", "offset_signal", "silk_board", "silk_board.traffic", -0.05)
    opposing = proposal("p3", "reroute", "silk_board", "silk_board.traffic", 0.05)
    found = detect_conflicts([first, duplicate], {"silk_board.traffic": 0.7})
    assert [item.kind for item in found] == ["same_node"]
    found = detect_conflicts([first, opposing], {"silk_board.traffic": 0.7})
    assert [item.kind for item in found] == ["opposing_effect"]


def test_resolve_is_deterministic_and_builds_stable_plan() -> None:
    costs = load_costs()
    first = proposal("b", "offset_signal", "silk_board", "silk_board.traffic", -0.05, 0.8)
    second = proposal("a", "shift_schedule", "whitefield", "whitefield.power", -0.04, 0.8)
    left = resolve([first, second], [], {}, costs)
    right = resolve([second, first], [], {}, costs)
    assert [item.action.id for item in left.accepted] == [item.action.id for item in right.accepted]
    assert build_plan(left, 42) == build_plan(right, 42)


def test_budget_conflict_rejects_lower_utility_proposal() -> None:
    costs = load_costs()
    expensive = proposal("expensive", "shed_load", "whitefield", "whitefield.power", -0.1, 2.0)
    cheap = proposal("cheap", "offset_signal", "silk_board", "silk_board.traffic", -0.1, 1.0)
    resolution = resolve([cheap, expensive], [], {}, costs, budget=3.0)
    assert [item.action.id for item in resolution.accepted] == ["cheap"]
    assert resolution.rejected[0][0].action.id == "expensive"


def test_feature_flag_defaults_enabled_and_disabled_uses_greedy_accept(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    from backend.fix import conflicts as conflicts_module

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(conflicts_module, "_REPO_ROOT", tmp_path)
    assert conflicts_module.conflicts_enabled()
    flag_path = config_dir / "features.json"
    flag_path.write_text('{"conflict_resolver": false}', encoding="utf-8")
    assert not conflicts_module.conflicts_enabled()

    costs = load_costs()
    monkeypatch.setattr(conflicts_module, "conflicts_enabled", lambda: False)
    high = proposal("high", "offset_signal", "silk_board", "silk_board.traffic", -0.2, 2.0)
    low = proposal("low", "shed_load", "whitefield", "whitefield.power", -0.2, 1.0)
    result = resolve([low, high], [], {"silk_board.traffic": 0.9}, costs, budget=0.0)
    assert [item.action.id for item in result.accepted] == ["high", "low"]
    assert result.rejected == []


def test_plan_has_expected_effect_and_costs() -> None:
    costs = load_costs()
    accepted = proposal("p1", "offset_signal", "silk_board", "silk_board.traffic", -0.1)
    resolution = resolve([accepted], [], {}, costs)
    plan = build_plan(resolution, 10)
    assert plan.actions == [accepted.action]
    assert plan.total_cost == 1.0
    assert plan.expected_effect == {"silk_board.traffic": -0.1}
    assert plan.plan_id.startswith("plan_10_")
from __future__ import annotations

from pathlib import Path

import pytest

from backend.fix._local_contracts import ACTION_TYPES
from backend.fix.fragility import (
    RankedFix,
    ZoneState,
    cheapest_fix,
    edges_to_coupling,
    fragility,
    load_costs,
)


def state(
    *,
    loads: dict[str, float] | None = None,
    sync_index: float = 80.0,
    csd_score: float = 0.2,
    coupling_strength_in: float = 0.2,
    coupling_strength_out: float = 0.1,
    headroom: dict[str, float] | None = None,
) -> ZoneState:
    return ZoneState(
        loads=loads or {"traffic": 0.7, "power": 0.6, "water": 0.5},
        sync_index=sync_index,
        csd_score=csd_score,
        coupling_strength_in=coupling_strength_in,
        coupling_strength_out=coupling_strength_out,
        headroom=headroom or {},
    )


def test_fragility_is_monotonic_for_each_risk_component() -> None:
    baseline = state()
    baseline_score = fragility("silk_board", baseline)
    higher_loads = dict(baseline.loads, traffic=0.8)
    assert fragility("silk_board", state(loads=higher_loads)) >= baseline_score
    assert fragility("silk_board", state(csd_score=0.4)) >= baseline_score
    assert fragility("silk_board", state(coupling_strength_in=0.5)) >= baseline_score
    assert fragility("silk_board", state(sync_index=60.0)) >= baseline_score


def test_edges_to_coupling_aggregates_edges_by_direction() -> None:
    edges = [
        {"src": "silk_board.traffic", "dst": "whitefield.power", "strength": 0.5},
        {"src": "hebbal.water", "dst": "silk_board.power", "strength": 0.4},
        {"src": "koramangala.power", "dst": "silk_board.water", "strength": 0.5},
    ]
    incoming, outgoing = edges_to_coupling(edges, "silk_board")
    assert incoming == pytest.approx(0.7)
    assert outgoing == pytest.approx(0.5)


def test_load_costs_reads_repo_config_and_custom_path(tmp_path: Path) -> None:
    loaded = load_costs()
    assert loaded["action_costs"]["offset_signal"]["cost"] == 1
    custom_file = tmp_path / "costs.json"
    custom_file.write_text('{"action_costs": {}, "zone_overrides": {}}', encoding="utf-8")
    assert load_costs(custom_file) == {"action_costs": {}, "zone_overrides": {}}


def test_cheapest_fix_is_sorted_by_benefit_per_cost_and_uses_valid_actions() -> None:
    ranked = cheapest_fix("silk_board", state(loads={"traffic": 1.0, "power": 0.4, "water": 0.4}),
                          load_costs(), k=5)
    assert ranked
    assert all(isinstance(item, RankedFix) for item in ranked)
    ratios = [item.benefit_per_cost for item in ranked]
    assert ratios == sorted(ratios, reverse=True)
    assert all(item.action_type in ACTION_TYPES for item in ranked)


def test_cheapest_fix_handles_all_zero_state() -> None:
    zero_state = state(
        loads={"traffic": 0.0, "power": 0.0, "water": 0.0},
        sync_index=100.0,
        csd_score=0.0,
        coupling_strength_in=0.0,
        coupling_strength_out=0.0,
    )
    ranked = cheapest_fix("silk_board", zero_state, load_costs(), k=5)
    assert ranked
    assert all(item.expected_fragility_reduction == 0.0 for item in ranked)
    assert all(item.action_type in ACTION_TYPES for item in ranked)


def test_attack_candidates_use_only_frozen_actions_and_reference_entry_node() -> None:
    ranked = cheapest_fix(
        "silk_board",
        state(loads={"traffic": 1.0, "power": 0.4, "water": 0.4}),
        load_costs(),
        threat_label="attack",
        entry_node="silk_board.traffic",
        k=20,
    )
    assert all(item.action_type in ACTION_TYPES for item in ranked)
    assert any(item.params.get("mitigation") == "isolate_sensor" for item in ranked)
    assert any(item.params.get("avoid_node") == "silk_board.traffic" for item in ranked)
    assert all(item.rationale.endswith(".") for item in ranked)
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.fix._local_contracts import Reading, Tick
from backend.fix.agents import (
    PowerAgent,
    TrafficAgent,
    WaterAgent,
    ZoneSnapshot,
    propose_all,
    snapshot_from_events,
)
from backend.fix.fragility import load_costs


def test_healthy_zone_produces_no_agent_proposals() -> None:
    snapshot = ZoneSnapshot(
        zone="silk_board",
        t=10,
        loads={"traffic": 0.5, "power": 0.6, "water": 0.4},
        sync_index=80.0,
    )
    assert TrafficAgent().propose(snapshot, load_costs()) == []
    assert PowerAgent().propose(snapshot, load_costs()) == []
    assert WaterAgent().propose(snapshot, load_costs()) == []
    assert propose_all([snapshot], load_costs()) == []


def test_traffic_reroute_uses_least_loaded_neighbor_and_couples_power() -> None:
    snapshot = ZoneSnapshot(
        zone="silk_board",
        t=33,
         loads={"traffic": 0.9, "power": 0.5, "water": 0.4},
         sync_index=20.0,
         neighbor_loads={"koramangala": 0.4, "electronic_city": 0.1, "jayanagar": 0.2},
    )
    proposals = TrafficAgent().propose(snapshot, load_costs())
    reroute = next(item for item in proposals if item.action.type == "reroute")
    assert reroute.action.params["to_zone"] == "electronic_city"
    assert reroute.expected_load_delta["electronic_city.traffic"] > 0.0
    assert reroute.expected_load_delta["electronic_city.power"] > 0.0
    assert reroute.expected_load_delta["silk_board.traffic"] < 0.0
    assert reroute.action.id == "traffic_silk_board_reroute_33"


def test_power_prefers_shift_schedule_unless_critical() -> None:
    drifting = ZoneSnapshot(
        zone="whitefield", t=5, loads={"power": 0.9}, sync_index=50.0
    )
    critical = ZoneSnapshot(
        zone="whitefield", t=5, loads={"power": 0.9}, sync_index=20.0
    )
    assert PowerAgent().propose(drifting, load_costs())[0].action.type == "shift_schedule"
    assert PowerAgent().propose(critical, load_costs())[0].action.type == "shed_load"


def test_water_agent_proposes_pump_shift_above_threshold() -> None:
    snapshot = ZoneSnapshot(zone="hebbal", t=9, loads={"water": 0.8})
    proposal = WaterAgent().propose(snapshot, load_costs())[0]
    assert proposal.action.type == "shift_pump"
    assert proposal.action.params["pump_group"]


def test_attack_avoids_compromised_node_and_explains_avoidance() -> None:
    snapshot = ZoneSnapshot(
        zone="silk_board",
        t=12,
        loads={"traffic": 0.95},
        threat_label="attack",
        entry_node="silk_board.traffic",
    )
    proposals = TrafficAgent().propose(snapshot, load_costs())
    assert all(snapshot.entry_node not in proposal.touches for proposal in proposals)
    assert proposals == []

    separate_entry = ZoneSnapshot(
        zone="silk_board",
        t=12,
        loads={"traffic": 0.95},
        threat_label="attack",
        entry_node="whitefield.power",
    )
    proposals = TrafficAgent().propose(separate_entry, load_costs())
    assert proposals
    assert all("Avoiding compromised node" in proposal.rationale for proposal in proposals)


class FakeContext:
    def __init__(self) -> None:
        self.payloads: dict[tuple[str, str | None], Any] = {
            ("sync.index", "koramangala"): {"sync_index": 50.0},
            ("sync.graph", None): {
                "edges": [
                    {"src": "koramangala.traffic", "dst": "indiranagar.power",
                     "lag_min": 2.0, "strength": 0.8}
                ]
            },
            ("threat.assessment", "koramangala"): {
                "label": "attack", "entry_node": "koramangala.power"
            },
        }

    def latest(self, channel: str, zone: str | None) -> Any:
        return self.payloads.get((channel, zone))


def test_snapshot_from_events_uses_latest_context_and_readings() -> None:
    tick = Tick(
        t=17,
        ts="2026-11-17T00:17:00+05:30",
        readings=[
            Reading(zone="koramangala", system="traffic", sensor_id="kor.t.01", value=0.82),
            Reading(zone="indiranagar", system="traffic", sensor_id="ind.t.01", value=0.42),
            Reading(zone="indiranagar", system="power", sensor_id="ind.p.01", value=0.9),
        ],
    )
    snapshot = snapshot_from_events(FakeContext(), "koramangala", tick)
    assert snapshot.status == "drifting"
    assert snapshot.loads["traffic"] == 0.82
    assert snapshot.loads["water"] == 0.0
    assert snapshot.neighbor_loads["indiranagar"] == 0.42
    assert snapshot.threat_label == "attack"
    assert snapshot.upstream_edges[0][3] == 0.8


def test_snapshot_status_uses_configured_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from backend.fix import agents as agents_module

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "thresholds.json").write_text(
        json.dumps({"status_bands": {"healthy_min": 40, "drifting_min": 20}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(agents_module, "_REPO_ROOT", tmp_path)
    tick = Tick(t=3, ts="2026-11-17T00:03:00+05:30")
    snapshot = snapshot_from_events(FakeContext(), "koramangala", tick)
    assert snapshot.status == "healthy"


def test_snapshot_from_events_degrades_gracefully() -> None:
    class BrokenContext:
        def latest(self, channel: str, zone: str | None) -> Any:
            raise RuntimeError("not ready")

    tick = Tick(t=4, ts="2026-11-17T00:04:00+05:30")
    snapshot = snapshot_from_events(BrokenContext(), "yelahanka", tick)
    assert snapshot.status == "healthy"
    assert snapshot.loads == {"traffic": 0.0, "power": 0.0, "water": 0.0}
    assert snapshot.threat_label == "unknown"
    assert snapshot.upstream_edges == []
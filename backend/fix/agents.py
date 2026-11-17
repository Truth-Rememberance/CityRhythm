"""Deterministic department agents that propose actions from observed zone state."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.fix._local_contracts import ACTION_TYPES, ZONES, Action, Tick

Status = Literal["healthy", "drifting", "critical"]
ThreatLabel = Literal["natural", "attack", "unknown"]
UpstreamEdge = tuple[str, str, float, float]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FALLBACK_THRESHOLDS = (70.0, 40.0)
REROUTE_POWER_COUPLING = 0.25
STRONG_EDGE_THRESHOLD = 0.7

# Approximate geographic/commuter adjacency for deterministic reroute destinations.
ZONE_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "silk_board": ("koramangala", "electronic_city", "jayanagar"),
    "whitefield": ("indiranagar", "yelahanka", "koramangala"),
    "hebbal": ("yelahanka", "indiranagar", "koramangala"),
    "electronic_city": ("silk_board", "jayanagar", "koramangala"),
    "koramangala": ("silk_board", "indiranagar", "jayanagar", "whitefield"),
    "indiranagar": ("koramangala", "whitefield", "hebbal", "jayanagar"),
    "jayanagar": ("silk_board", "electronic_city", "koramangala", "indiranagar"),
    "yelahanka": ("hebbal", "whitefield", "indiranagar"),
}


@dataclass(frozen=True)
class ZoneSnapshot:
    """Observed zone state; loads are normalized 0-1 and t is simulation minutes."""

    zone: str
    t: int
    loads: dict[str, float]
    sync_index: float = 100.0
    threat_label: ThreatLabel = "unknown"
    entry_node: str | None = None
    upstream_edges: list[UpstreamEdge] = field(default_factory=list)
    neighbor_loads: dict[str, float] = field(default_factory=dict)
    status: Status = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "loads", _normalized_loads(self.loads))
        object.__setattr__(self, "neighbor_loads", _normalized_load_map(self.neighbor_loads))
        object.__setattr__(self, "status", status_for_sync(self.sync_index))


@dataclass(frozen=True)
class Proposal:
    """Action candidate, with normalized expected load deltas keyed by node ID."""

    action: Action
    utility: float
    rationale: str
    touches: list[str]
    expected_load_delta: dict[str, float]


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalized_loads(loads: Any) -> dict[str, float]:
    if not isinstance(loads, Mapping):
        loads = {}
    result: dict[str, float] = {}
    for system in ("traffic", "power", "water"):
        try:
            result[system] = min(1.0, max(0.0, float(loads.get(system, 0.0))))
        except (TypeError, ValueError):
            result[system] = 0.0
    return result


def _normalized_load_map(loads: Any) -> dict[str, float]:
    if not isinstance(loads, Mapping):
        return {}
    result: dict[str, float] = {}
    for zone, value in loads.items():
        if not isinstance(zone, str):
            continue
        try:
            result[zone] = min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            result[zone] = 0.0
    return result


def _numeric_threshold(value: Any, fallback: float) -> float:
    if isinstance(value, Mapping):
        value = value.get("min", value.get("threshold", value.get("value")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _status_thresholds() -> tuple[float, float]:
    path = _REPO_ROOT / "config" / "thresholds.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        bands = data.get("status_bands", data.get("thresholds", data))
        if not isinstance(bands, Mapping):
            return _FALLBACK_THRESHOLDS
        healthy = _numeric_threshold(
            bands.get("healthy_min", bands.get("healthy_threshold", bands.get("healthy"))),
            _FALLBACK_THRESHOLDS[0],
        )
        drifting = _numeric_threshold(
            bands.get(
                "drifting_min",
                bands.get("critical_below", bands.get("drifting", bands.get("critical"))),
            ),
            _FALLBACK_THRESHOLDS[1],
        )
        return healthy, drifting
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return _FALLBACK_THRESHOLDS


def status_for_sync(sync_index: float) -> Status:
    """Map Sync Index 0-100 into threshold-configured status bands."""
    healthy_min, drifting_min = _status_thresholds()
    if sync_index >= healthy_min:
        return "healthy"
    if sync_index >= drifting_min:
        return "drifting"
    return "critical"


def _cost_entry(costs: dict[str, Any], zone: str, action_type: str) -> dict[str, Any]:
    base = costs.get("action_costs", {}).get(action_type, {})
    override = costs.get("zone_overrides", {}).get(zone, {}).get(action_type, {})
    return {**base, **override} if isinstance(base, Mapping) and isinstance(override, Mapping) else {}


def _utility(costs: dict[str, Any], zone: str, action_type: str, magnitude: float) -> float:
    entry = _cost_entry(costs, zone, action_type)
    effect = max(0.0, float(entry.get("effect_scale", 0.0)))
    cost = max(0.01, float(entry.get("cost", 1.0)))
    disruption = max(0.0, float(entry.get("disruption", 0.0)))
    return max(0.0, effect * magnitude * (1.0 - min(disruption, 1.0)) / cost)


def _action(
    dept: str,
    snapshot: ZoneSnapshot,
    action_type: str,
    params: dict[str, Any],
) -> Action:
    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unsupported action type: {action_type}")
    return Action(
        id=f"{dept}_{snapshot.zone}_{action_type}_{snapshot.t}",
        type=action_type,
        dept=dept,
        zone=snapshot.zone,
        params=params,
        start_t=max(0, snapshot.t),
        duration_min=15,
    )


def _can_touch(snapshot: ZoneSnapshot, touches: Sequence[str]) -> bool:
    return not (
        snapshot.threat_label == "attack"
        and snapshot.entry_node is not None
        and snapshot.entry_node in touches
    )


def _rationale(snapshot: ZoneSnapshot, rationale: str) -> str:
    if snapshot.threat_label == "attack" and snapshot.entry_node:
        return f"{rationale} Avoiding compromised node {snapshot.entry_node}."
    return rationale


class TrafficAgent:
    """Propose signal offsets and routes for stressed traffic zones."""

    def propose(self, snapshot: ZoneSnapshot, costs: dict[str, Any]) -> list[Proposal]:
        load = snapshot.loads["traffic"]
        strong_upstream = any(
            src == f"{snapshot.zone}.traffic" and strength >= STRONG_EDGE_THRESHOLD
            for src, _dst, _lag, strength in snapshot.upstream_edges
        )
        if load <= 0.75 and not strong_upstream:
            return []

        proposals: list[Proposal] = []
        traffic_node = f"{snapshot.zone}.traffic"
        offset_entry = _cost_entry(costs, snapshot.zone, "offset_signal")
        offset_effect = max(0.0, float(offset_entry.get("effect_scale", 0.25)))
        offset_delta = min(load, offset_effect * 0.1)
        if _can_touch(snapshot, [traffic_node]):
            proposals.append(
                Proposal(
                    action=_action(
                        "traffic",
                        snapshot,
                        "offset_signal",
                        {"offset_s": max(1, round(offset_effect * 60)), "corridor": "main"},
                    ),
                    utility=_utility(costs, snapshot.zone, "offset_signal", max(load, 0.1)),
                    rationale=_rationale(snapshot, "Offset signals to reduce traffic pressure."),
                    touches=[traffic_node],
                    expected_load_delta={traffic_node: -offset_delta},
                )
            )

        neighbors = ZONE_NEIGHBORS.get(snapshot.zone, ())
        safe_neighbors = [
            neighbor
            for neighbor in neighbors
            if neighbor in ZONES
            and not (
                snapshot.threat_label == "attack"
                and snapshot.entry_node in (f"{neighbor}.traffic", f"{neighbor}.power")
            )
        ]
        if safe_neighbors:
            destination = min(
                safe_neighbors,
                key=lambda zone: (snapshot.neighbor_loads.get(zone, 1.0), zone),
            )
            fraction = min(0.15, max(0.02, float(_cost_entry(costs, snapshot.zone, "reroute").get(
                "effect_scale", 0.3
            )) * 0.5))
            transferred_load = min(load, load * fraction)
            destination_traffic = f"{destination}.traffic"
            destination_power = f"{destination}.power"
            touches = [traffic_node, destination_traffic, destination_power]
            if _can_touch(snapshot, touches):
                proposals.append(
                    Proposal(
                        action=_action(
                            "traffic",
                            snapshot,
                            "reroute",
                            {
                                "from_zone": snapshot.zone,
                                "to_zone": destination,
                                "fraction": fraction,
                            },
                        ),
                        utility=_utility(costs, snapshot.zone, "reroute", transferred_load),
                        rationale=_rationale(
                            snapshot,
                            f"Reroute a small traffic share from {snapshot.zone} to {destination}.",
                        ),
                        touches=touches,
                        expected_load_delta={
                            traffic_node: -transferred_load,
                            destination_traffic: transferred_load,
                            destination_power: transferred_load * REROUTE_POWER_COUPLING,
                        },
                    )
                )
        return proposals


class PowerAgent:
    """Propose schedule shifting or load shedding for stressed power zones."""

    def propose(self, snapshot: ZoneSnapshot, costs: dict[str, Any]) -> list[Proposal]:
        load = snapshot.loads["power"]
        if load <= 0.8:
            return []
        action_type = "shed_load" if snapshot.status == "critical" else "shift_schedule"
        node = f"{snapshot.zone}.power"
        if not _can_touch(snapshot, [node]):
            return []
        entry = _cost_entry(costs, snapshot.zone, action_type)
        effect = max(0.0, float(entry.get("effect_scale", 0.25)))
        delta = min(load, effect * 0.1)
        params = (
            {"pct": effect * 100.0, "feeder": "main"}
            if action_type == "shed_load"
            else {"shift_min": 15, "schedule": "flexible_loads"}
        )
        rationale = (
            "Shed a limited amount of load because the zone is critical."
            if action_type == "shed_load"
            else "Shift flexible demand to ease the stressed power period."
        )
        return [
            Proposal(
                action=_action("power", snapshot, action_type, params),
                utility=_utility(costs, snapshot.zone, action_type, load),
                rationale=_rationale(snapshot, rationale),
                touches=[node],
                expected_load_delta={node: -delta},
            )
        ]


class WaterAgent:
    """Propose pump shifting when water load is above its operating threshold."""

    def propose(self, snapshot: ZoneSnapshot, costs: dict[str, Any]) -> list[Proposal]:
        load = snapshot.loads["water"]
        if load <= 0.75:
            return []
        node = f"{snapshot.zone}.water"
        if not _can_touch(snapshot, [node]):
            return []
        entry = _cost_entry(costs, snapshot.zone, "shift_pump")
        effect = max(0.0, float(entry.get("effect_scale", 0.35)))
        delta = min(load, effect * 0.1)
        return [
            Proposal(
                action=_action(
                    "water", snapshot, "shift_pump", {"shift_min": 15, "pump_group": "main"}
                ),
                utility=_utility(costs, snapshot.zone, "shift_pump", load),
                rationale=_rationale(snapshot, "Shift pump timing to reduce water-system stress."),
                touches=[node],
                expected_load_delta={node: -delta},
            )
        ]


def propose_all(
    snapshots: Sequence[ZoneSnapshot], costs: dict[str, Any], max_per_zone: int = 3
) -> list[Proposal]:
    """Collect deterministic proposals for non-healthy zones, ranked by utility."""
    if max_per_zone <= 0:
        return []
    agents = (TrafficAgent(), PowerAgent(), WaterAgent())
    proposals: list[Proposal] = []
    for snapshot in snapshots:
        if snapshot.status == "healthy" and not (
            snapshot.loads["traffic"] > 0.75
            or snapshot.loads["power"] > 0.8
            or snapshot.loads["water"] > 0.75
        ):
            continue
        zone_proposals = [
            proposal
            for agent in agents
            for proposal in agent.propose(snapshot, costs)
        ]
        zone_proposals.sort(key=lambda proposal: (-proposal.utility, proposal.action.id))
        proposals.extend(zone_proposals[:max_per_zone])
    return sorted(proposals, key=lambda proposal: (-proposal.utility, proposal.action.id))


def _safe_latest(ctx: Any, channel: str, zone: str | None) -> Any:
    try:
        return ctx.latest(channel, zone)
    except Exception:  # noqa: BLE001 - malformed/missing upstream modules must degrade gracefully.
        return None


def _edge_rows(graph: Any, zone: str) -> list[UpstreamEdge]:
    edges = _value(graph, "edges", [])
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        return []
    rows: list[UpstreamEdge] = []
    prefix = f"{zone}."
    for edge in edges:
        src, dst = _value(edge, "src"), _value(edge, "dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if not (src.startswith(prefix) or dst.startswith(prefix)):
            continue
        try:
            lag = float(_value(edge, "lag_min", 0.0))
            strength = float(_value(edge, "strength", 0.0))
        except (TypeError, ValueError):
            continue
        rows.append((src, dst, lag, strength))
    return rows


def snapshot_from_events(ctx: Any, zone: str, tick: Tick) -> ZoneSnapshot:
    """Build a snapshot from tick readings and latest read-only context; missing data defaults."""
    try:
        raw_loads: dict[str, float] = {}
        neighbor_traffic: dict[str, float] = {}
        readings = _value(tick, "readings", [])
        if isinstance(readings, Sequence):
            for reading in readings:
                reading_zone = _value(reading, "zone")
                system = _value(reading, "system")
                if reading_zone != zone:
                    if (
                        system == "traffic"
                        and reading_zone in ZONE_NEIGHBORS.get(zone, ())
                    ):
                        try:
                            neighbor_traffic[reading_zone] = float(_value(reading, "value", 0.0))
                        except (TypeError, ValueError):
                            continue
                    continue
                if system in ("traffic", "power", "water"):
                    raw_loads[system] = float(_value(reading, "value", 0.0))
        loads = _normalized_loads(raw_loads)
        sync = _safe_latest(ctx, "sync.index", zone)
        try:
            sync_index = float(_value(sync, "sync_index", 100.0))
        except (TypeError, ValueError):
            sync_index = 100.0
        threat = _safe_latest(ctx, "threat.assessment", zone)
        threat_label = _value(threat, "label", "unknown")
        if threat_label not in ("natural", "attack", "unknown"):
            threat_label = "unknown"
        entry_node = _value(threat, "entry_node")
        if not isinstance(entry_node, str):
            entry_node = None
        try:
            tick_time = max(0, int(_value(tick, "t", 0)))
        except (TypeError, ValueError):
            tick_time = 0
        graph = _safe_latest(ctx, "sync.graph", None)
        return ZoneSnapshot(
            zone=zone,
            t=tick_time,
            loads=loads,
            sync_index=sync_index,
            threat_label=threat_label,
            entry_node=entry_node,
            upstream_edges=_edge_rows(graph, zone),
            neighbor_loads=neighbor_traffic,
        )
    except Exception:  # noqa: BLE001 - snapshot construction must tolerate incomplete inputs.
        return ZoneSnapshot(zone=zone, t=0, loads={})
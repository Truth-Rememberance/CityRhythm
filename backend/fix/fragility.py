"""Deterministic zone fragility and intervention ranking utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from backend.fix._local_contracts import ACTION_TYPES, SYSTEMS

# Components sum to one: load proximity, sync disruption, CSD, then incoming coupling.
FRAGILITY_WEIGHTS: dict[str, float] = {
    "load_proximity": 0.4,
    "sync_disruption": 0.3,
    "csd": 0.15,
    "incoming_coupling": 0.15,
}
_TIPPING_LOAD = 0.85
_SYSTEM_DEPT = {"traffic": "traffic", "power": "power", "water": "water"}
_SYSTEM_ACTIONS = {
    "traffic": ("offset_signal", "reroute", "shift_schedule"),
    "power": ("shed_load", "shift_schedule"),
    "water": ("shift_pump", "shed_load", "shift_schedule"),
}
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ZoneState:
    """Observed zone inputs; loads and headroom are normalized 0-1 values."""

    loads: dict[str, float]
    sync_index: float
    csd_score: float
    coupling_strength_in: float = 0.0
    coupling_strength_out: float = 0.0
    headroom: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedFix:
    """Cost-ranked department action with estimated 0-100 fragility benefit."""

    action_type: str
    dept: str
    params: dict[str, Any]
    cost: float
    expected_fragility_reduction: float
    benefit_per_cost: float
    rationale: str


def _bounded(value: float, low: float, high: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        return low
    return float(np.clip(number, low, high))


def _system_stress(state: ZoneState, system: str) -> float:
    load = _bounded(state.loads.get(system, 0.0), 0.0, 1.0)
    headroom = max(0.0, _bounded(state.headroom.get(system, 0.0), 0.0, 1.0))
    adjusted_load = load / (1.0 + headroom)
    return _bounded(adjusted_load / _TIPPING_LOAD, 0.0, 1.0)


def fragility(zone: str, state: ZoneState) -> float:
    """Return deterministic 0-100 fragility from load, sync, CSD, and coupling.

    Load proximity averages each system's utilization relative to the 0.85 tipping
    point; normalized headroom expands the tolerated load by its fractional reserve.
    Lower sync index, higher CSD, and stronger incoming coupling increase fragility.
    The weighted components are configured in FRAGILITY_WEIGHTS.
    """
    del zone  # Zone identity is carried for the public API; scoring uses its state only.
    load_proximity = float(np.mean([_system_stress(state, system) for system in SYSTEMS]))
    sync_disruption = 1.0 - _bounded(state.sync_index, 0.0, 100.0) / 100.0
    csd = _bounded(state.csd_score, 0.0, 1.0)
    incoming = _bounded(state.coupling_strength_in, 0.0, 1.0)
    score = 100.0 * (
        FRAGILITY_WEIGHTS["load_proximity"] * load_proximity
        + FRAGILITY_WEIGHTS["sync_disruption"] * sync_disruption
        + FRAGILITY_WEIGHTS["csd"] * csd
        + FRAGILITY_WEIGHTS["incoming_coupling"] * incoming
    )
    return _bounded(score, 0.0, 100.0)


def _edge_value(edge: Any, name: str) -> Any:
    if isinstance(edge, Mapping):
        return edge.get(name)
    return getattr(edge, name, None)


def _combined_strength(strengths: list[float]) -> float:
    remaining = 1.0
    for strength in strengths:
        remaining *= 1.0 - _bounded(strength, 0.0, 1.0)
    return 1.0 - remaining


def edges_to_coupling(edges: Iterable[Any], zone: str) -> tuple[float, float]:
    """Return normalized (incoming, outgoing) coupling for edges touching a zone."""
    incoming: list[float] = []
    outgoing: list[float] = []
    prefix = f"{zone}."
    for edge in edges:
        source = _edge_value(edge, "src")
        destination = _edge_value(edge, "dst")
        strength = _edge_value(edge, "strength")
        if not isinstance(source, str) or not isinstance(destination, str):
            continue
        if strength is None:
            continue
        try:
            value = float(strength)
        except (TypeError, ValueError):
            continue
        if destination.startswith(prefix):
            incoming.append(value)
        if source.startswith(prefix):
            outgoing.append(value)
    return _combined_strength(incoming), _combined_strength(outgoing)


def load_costs(path: str | Path | None = None) -> dict[str, Any]:
    """Load action costs from config/costs.json relative to the repository root."""
    costs_path = Path(path) if path is not None else Path("config/costs.json")
    if not costs_path.is_absolute():
        costs_path = _REPO_ROOT / costs_path
    with costs_path.open(encoding="utf-8") as costs_file:
        return json.load(costs_file)


def _action_costs(costs: dict[str, Any], zone: str, action_type: str) -> dict[str, Any]:
    base = costs.get("action_costs", {}).get(action_type, {})
    override = costs.get("zone_overrides", {}).get(zone, {}).get(action_type, {})
    return {**base, **override}


def _action_improvement(state: ZoneState, system: str, effect_scale: float) -> float:
    before = fragility("", state)
    current_load = _bounded(state.loads.get(system, 0.0), 0.0, 1.0)
    reduction = _bounded(effect_scale, 0.0, 1.0)
    revised_loads = dict(state.loads)
    revised_loads[system] = current_load * (1.0 - reduction)
    after = fragility("", replace(state, loads=revised_loads))
    return max(0.0, before - after)


def cheapest_fix(
    zone: str,
    state: ZoneState,
    costs: dict[str, Any],
    threat_label: str = "natural",
    k: int = 5,
    entry_node: str | None = None,
) -> list[RankedFix]:
    """Rank applicable interventions by estimated fragility reduction per cost point.

    Candidates target all systems tied for the highest tipping-point stress. For attack
    assessments, sensor isolation is expressed only as shed_load/offset_signal variants;
    reroute and shift_schedule candidates avoid a supplied compromised entry_node.
    """
    if k <= 0:
        return []
    stresses = {system: _system_stress(state, system) for system in SYSTEMS}
    highest_stress = max(stresses.values(), default=0.0)
    target_systems = [
        system for system, score in stresses.items() if np.isclose(score, highest_stress)
    ]
    candidates: list[RankedFix] = []

    for system in target_systems:
        for action_type in _SYSTEM_ACTIONS[system]:
            if action_type not in ACTION_TYPES:
                continue
            action_cost = _action_costs(costs, zone, action_type)
            if not action_cost:
                continue
            cost = max(0.0, float(action_cost.get("cost", 0.0)))
            effect_scale = max(0.0, float(action_cost.get("effect_scale", 0.0)))
            improvement = _action_improvement(state, system, effect_scale)
            params: dict[str, Any] = {
                "target_system": system,
                "effect_scale": effect_scale,
            }
            is_avoidance = threat_label == "attack" and entry_node is not None and (
                action_type in ("reroute", "shift_schedule")
            )
            if is_avoidance:
                params["avoid_node"] = entry_node
                improvement = min(fragility(zone, state), improvement * 1.25)
                rationale = f"{action_type} around {entry_node} to reduce {system} stress."
            else:
                rationale = f"Apply {action_type} to reduce tipping-point stress in {system}."
            benefit_per_cost = improvement / cost if cost > 0.0 else 0.0
            candidates.append(
                RankedFix(
                    action_type=action_type,
                    dept=_SYSTEM_DEPT[system],
                    params=params,
                    cost=cost,
                    expected_fragility_reduction=improvement,
                    benefit_per_cost=benefit_per_cost,
                    rationale=rationale,
                )
            )

            if threat_label == "attack" and action_type in ("shed_load", "offset_signal"):
                isolation_params = {**params, "mitigation": "isolate_sensor"}
                if entry_node is not None:
                    isolation_params["entry_node"] = entry_node
                candidates.append(
                    RankedFix(
                        action_type=action_type,
                        dept=_SYSTEM_DEPT[system],
                        params=isolation_params,
                        cost=cost,
                        expected_fragility_reduction=improvement,
                        benefit_per_cost=benefit_per_cost,
                        rationale=(
                            f"Isolate the suspected sensor while applying {action_type} "
                            f"to reduce {system} stress."
                        ),
                    )
                )

    candidates.sort(key=lambda candidate: candidate.benefit_per_cost, reverse=True)
    return candidates[:k]
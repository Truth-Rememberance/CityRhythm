"""Deterministic proposal conflict detection and greedy plan resolution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from backend.fix._local_contracts import Action
from backend.fix.agents import Proposal

ConflictKind = Literal["overload_transfer", "same_node", "opposing_effect", "budget"]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OVERLOAD_LIMIT = 0.85
_EPSILON = 1e-12


@dataclass(frozen=True)
class Conflict:
    """One detected issue among selected intervention proposals."""

    id: str
    kind: ConflictKind
    proposals: list[str]
    zone: str
    severity: float
    description: str


@dataclass(frozen=True)
class Resolution:
    """Accepted, rejected, and modified proposals from deterministic resolution."""

    accepted: list[Proposal]
    rejected: list[tuple[Proposal, str]]
    modified: list[tuple[Proposal, str]]
    compromise_notes: list[str]
    total_cost: float = 0.0


@dataclass(frozen=True)
class Plan:
    """Resolved action plan; expected_effect contains aggregate normalized load deltas."""

    plan_id: str
    actions: list[Action]
    total_cost: float
    expected_effect: dict[str, float]
    notes: list[str]


def _proposal_id(proposal: Proposal) -> str:
    return proposal.action.id


def _zone_for_node(node: str, fallback: str = "") -> str:
    return node.split(".", 1)[0] if "." in node else fallback


def _cost_for(proposal: Proposal, costs: Mapping[str, Any]) -> float:
    action_type = proposal.action.type
    zone = proposal.action.zone
    action_costs = costs.get("action_costs", {})
    overrides = costs.get("zone_overrides", {})
    base = action_costs.get(action_type, {}) if isinstance(action_costs, Mapping) else {}
    zone_costs = overrides.get(zone, {}) if isinstance(overrides, Mapping) else {}
    override = zone_costs.get(action_type, {}) if isinstance(zone_costs, Mapping) else {}
    entry = {**base, **override} if isinstance(base, Mapping) and isinstance(override, Mapping) else {}
    try:
        return max(0.0, float(entry.get("cost", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _severity(value: float) -> float:
    return min(1.0, max(0.0, value))


def detect_conflicts(
    proposals: Sequence[Proposal], loads: Mapping[str, float], budget: float | None = None,
    costs: Mapping[str, Any] | None = None,
) -> list[Conflict]:
    """Detect overload, duplicate-node, opposing-delta, and optional budget conflicts.

    Loads and delta keys are node IDs such as ``whitefield.power``. Budget detection
    requires both budget and costs; costs follow config/costs.json structure.
    """
    conflicts: list[Conflict] = []
    ordered = list(proposals)

    all_deltas: dict[str, float] = {}
    contributors: dict[str, list[str]] = {}
    for proposal in ordered:
        for node, delta in proposal.expected_load_delta.items():
            try:
                numeric_delta = float(delta)
            except (TypeError, ValueError):
                continue
            all_deltas[node] = all_deltas.get(node, 0.0) + numeric_delta
            if numeric_delta > 0.0:
                contributors.setdefault(node, []).append(_proposal_id(proposal))

    for node in sorted(all_deltas):
        try:
            resulting_load = float(loads.get(node, 0.0)) + all_deltas[node]
        except (TypeError, ValueError):
            resulting_load = all_deltas[node]
        if all_deltas[node] > 0.0 and resulting_load > _OVERLOAD_LIMIT + _EPSILON:
            proposal_ids = contributors.get(node, [])
            if not proposal_ids:
                proposal_ids = [
                    _proposal_id(proposal)
                    for proposal in ordered
                    if node in proposal.expected_load_delta
                ]
            zone = _zone_for_node(node, ordered[0].action.zone if ordered else "")
            conflicts.append(
                Conflict(
                    id=f"overload_transfer:{node}:{','.join(proposal_ids)}",
                    kind="overload_transfer",
                    proposals=proposal_ids,
                    zone=zone,
                    severity=_severity((resulting_load - _OVERLOAD_LIMIT) / 0.15),
                    description=(
                        f"Expected load on {node} reaches {resulting_load:.3f}, "
                        f"above the {_OVERLOAD_LIMIT:.2f} limit."
                    ),
                )
            )

    for index, first in enumerate(ordered):
        first_nodes = set(first.touches) | set(first.expected_load_delta)
        for second in ordered[index + 1 :]:
            second_nodes = set(second.touches) | set(second.expected_load_delta)
            shared_nodes = sorted(first_nodes & second_nodes)
            pair_ids = [_proposal_id(first), _proposal_id(second)]
            same_target_zone = first.action.zone == second.action.zone
            if first.action.type == second.action.type and (shared_nodes or same_target_zone):
                node = shared_nodes[0] if shared_nodes else f"{first.action.zone}.*"
                conflicts.append(
                    Conflict(
                        id=f"same_node:{node}:{','.join(pair_ids)}",
                        kind="same_node",
                        proposals=pair_ids,
                        zone=_zone_for_node(node, first.action.zone),
                        severity=0.5,
                        description=(
                            f"Both proposals apply {first.action.type} to {node}."
                        ),
                    )
                )
            opposing_nodes = [
                node
                for node in shared_nodes
                if float(first.expected_load_delta.get(node, 0.0))
                * float(second.expected_load_delta.get(node, 0.0))
                < 0.0
            ]
            if opposing_nodes:
                node = opposing_nodes[0]
                conflicts.append(
                    Conflict(
                        id=f"opposing_effect:{node}:{','.join(pair_ids)}",
                        kind="opposing_effect",
                        proposals=pair_ids,
                        zone=_zone_for_node(node, first.action.zone),
                        severity=0.5,
                        description=f"Proposals push load on {node} in opposite directions.",
                    )
                )

    if budget is not None and costs is not None:
        total_cost = sum(_cost_for(proposal, costs) for proposal in ordered)
        if total_cost > budget + _EPSILON:
            conflicts.append(
                Conflict(
                    id=f"budget:{','.join(_proposal_id(item) for item in ordered)}",
                    kind="budget",
                    proposals=[_proposal_id(item) for item in ordered],
                    zone="",
                    severity=_severity((total_cost - budget) / max(float(budget), 1.0)),
                    description=(
                        f"Selected actions cost {total_cost:.2f}, exceeding budget {budget:.2f}."
                    ),
                )
            )
    return conflicts


def conflicts_enabled() -> bool:
    """Read config/features.json; missing or malformed flags default to enabled."""
    flag_path = _REPO_ROOT / "config" / "features.json"
    try:
        data = json.loads(flag_path.read_text(encoding="utf-8"))
        return bool(data.get("conflict_resolver", True)) if isinstance(data, Mapping) else True
    except (OSError, json.JSONDecodeError, TypeError):
        return True


def _current_loads(loads: Mapping[str, float], proposals: Sequence[Proposal]) -> dict[str, float]:
    current: dict[str, float] = {}
    for node, value in loads.items():
        try:
            current[node] = float(value)
        except (TypeError, ValueError):
            current[node] = 0.0
    for proposal in proposals:
        for node, delta in proposal.expected_load_delta.items():
            current[node] = current.get(node, 0.0) + float(delta)
    return current


def _compromise_reroute(
    proposal: Proposal, accepted: Sequence[Proposal], loads: Mapping[str, float], conflict: Conflict
) -> tuple[Proposal, str] | None:
    if proposal.action.type != "reroute":
        return None
    current = _current_loads(loads, accepted)
    overload_node = next(
        (
            node
            for node, delta in proposal.expected_load_delta.items()
            if delta > 0.0 and current.get(node, 0.0) + delta > _OVERLOAD_LIMIT + _EPSILON
        ),
        None,
    )
    if overload_node is None:
        return None
    prior_load = current.get(overload_node, 0.0)
    original_delta = proposal.expected_load_delta[overload_node]
    allowable = max(0.0, _OVERLOAD_LIMIT - prior_load)
    scale = min(1.0, allowable / original_delta)
    if scale <= _EPSILON or scale >= 1.0 - _EPSILON:
        return None

    scaled_deltas = {node: float(delta) * scale for node, delta in proposal.expected_load_delta.items()}
    params = dict(proposal.action.params)
    original_fraction = float(params.get("fraction", 1.0))
    params["fraction"] = original_fraction * scale
    revised_action = proposal.action.model_copy(update={"params": params})
    revised = replace(
        proposal,
        action=revised_action,
        utility=proposal.utility * scale,
        expected_load_delta=scaled_deltas,
    )
    description = (
        f"Reduced reroute fraction from {original_fraction:.3f} to "
        f"{params['fraction']:.3f} to keep {overload_node} at or below "
        f"{_OVERLOAD_LIMIT:.2f}."
    )
    return revised, description


def resolve(
    proposals: Sequence[Proposal],
    conflicts: Sequence[Conflict],
    loads: Mapping[str, float],
    costs: Mapping[str, Any],
    budget: float | None = None,
) -> Resolution:
    """Greedily select proposals by utility, compromising reroutes before rejection.

    When the feature flag is disabled, proposals are accepted by utility order without
    conflict or budget checks. The conflicts argument supports caller-provided reports;
    the resolver recomputes conflicts on each trial against current accepted actions.
    """
    del conflicts
    ordered = sorted(proposals, key=lambda proposal: (-proposal.utility, proposal.action.id))
    if not conflicts_enabled():
        accepted = list(ordered)
        total = sum(_cost_for(proposal, costs) for proposal in accepted)
        return Resolution(accepted, [], [], [], total_cost=total)

    accepted: list[Proposal] = []
    rejected: list[tuple[Proposal, str]] = []
    modified: list[tuple[Proposal, str]] = []
    notes: list[str] = []
    total_cost = 0.0

    for proposal in ordered:
        trial = [*accepted, proposal]
        issues = detect_conflicts(trial, loads, budget=budget, costs=costs)
        if not issues:
            accepted.append(proposal)
            total_cost += _cost_for(proposal, costs)
            continue

        overload = next((item for item in issues if item.kind == "overload_transfer"), None)
        compromise = (
            _compromise_reroute(proposal, accepted, loads, overload)
            if overload is not None
            else None
        )
        if compromise is not None:
            revised, description = compromise
            revised_issues = detect_conflicts([*accepted, revised], loads, budget=budget, costs=costs)
            if not revised_issues:
                accepted.append(revised)
                revised_cost = _cost_for(revised, costs)
                total_cost += revised_cost
                modified.append((revised, description))
                notes.append(description)
                continue
            issues = revised_issues

        details = "; ".join(item.description for item in issues)
        rejected.append((proposal, details or "Conflicts with the selected plan."))

    return Resolution(accepted, rejected, modified, notes, total_cost=total_cost)


def build_plan(resolution: Resolution, t: int) -> Plan:
    """Build a stable plan envelope from accepted proposals at simulation minute t."""
    accepted = sorted(resolution.accepted, key=lambda proposal: proposal.action.id)
    action_ids = [proposal.action.id for proposal in accepted]
    seed = f"{max(0, int(t))}:{'|'.join(action_ids)}"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:10]
    effects: dict[str, float] = {}
    for proposal in accepted:
        for node, delta in proposal.expected_load_delta.items():
            effects[node] = effects.get(node, 0.0) + float(delta)
    return Plan(
        plan_id=f"plan_{max(0, int(t))}_{digest}",
        actions=[proposal.action for proposal in accepted],
        total_cost=resolution.total_cost,
        expected_effect=dict(sorted(effects.items())),
        notes=list(resolution.compromise_notes),
    )
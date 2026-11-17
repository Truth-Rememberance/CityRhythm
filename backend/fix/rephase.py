"""Deterministic load re-phasing optimizer; shifts timing rather than adding capacity."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from backend.fix._local_contracts import ACTION_TYPES, ZONES, Action
from backend.fix.metrics import cut_percent

FlexSpec: TypeAlias = tuple[float, int, str, str]

# Per-system defaults: movable fraction, absolute max shift (minutes), action type, department.
DEFAULT_FLEXIBILITY: dict[str, FlexSpec] = {
    "water": (0.6, 120, "shift_pump", "water"),
    "traffic": (0.3, 30, "offset_signal", "traffic"),
    "power": (0.4, 60, "shift_schedule", "power"),
}
SHIFT_STEP_MIN = 5
EDGE_CONTAGION_SCALE = 0.2
_OVERLOAD_THRESHOLD = 0.85


@dataclass(frozen=True)
class OverlapStats:
    """Aggregate overlap and overload measures for one forecast horizon."""

    coincidence_minutes: int
    overload_area: float
    peak: float
    score: float


@dataclass(frozen=True)
class RephaseResult:
    """Optimized shifts and their before/after overlap statistics."""

    shifts: dict[str, int]
    before_stats: OverlapStats
    after_stats: OverlapStats
    peak_overlap_cut_pct: float
    actions: list[Action]
    runtime_ms: float


def _as_profile_map(profiles: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    expected_len: int | None = None
    for node in sorted(profiles):
        values = np.asarray(profiles[node], dtype=float)
        if values.ndim != 1:
            raise ValueError(f"Profile for {node} must be one-dimensional")
        if expected_len is None:
            expected_len = len(values)
        elif len(values) != expected_len:
            raise ValueError("All profiles must have the same horizon length")
        normalized[node] = np.clip(values, 0.0, 1.0)
    return normalized


def _node_parts(node: str) -> tuple[str, str]:
    zone, separator, system = node.rpartition(".")
    if not separator:
        return node, ""
    return zone, system


def _edge_fields(edge: Any) -> tuple[str | None, str | None, float, float]:
    if isinstance(edge, Mapping):
        source, destination = edge.get("src"), edge.get("dst")
        lag, strength = edge.get("lag_min", 0), edge.get("strength", 0.0)
    else:
        source, destination = getattr(edge, "src", None), getattr(edge, "dst", None)
        lag, strength = getattr(edge, "lag_min", 0), getattr(edge, "strength", 0.0)
    try:
        return (
            source if isinstance(source, str) else None,
            destination if isinstance(destination, str) else None,
            float(lag),
            float(strength),
        )
    except (TypeError, ValueError):
        return None, None, 0.0, 0.0


def _with_contagion(
    profiles: Mapping[str, np.ndarray], edges: Sequence[Any] | None, hi: float
) -> dict[str, np.ndarray]:
    result = {node: values.copy() for node, values in profiles.items()}
    if edges is None:
        return result
    for edge in edges:
        source, destination, lag, strength = _edge_fields(edge)
        if source not in profiles or destination not in result or strength <= 0.0:
            continue
        lag_bins = max(0, round(lag))
        source_load = profiles[source]
        source_minutes = np.flatnonzero(source_load > hi)
        for minute in source_minutes:
            target_minute = int(minute) + lag_bins
            if target_minute < len(result[destination]):
                result[destination][target_minute] += strength * EDGE_CONTAGION_SCALE
    return result


def peak_overlap(
    profiles: dict[str, np.ndarray],
    hi: float = 0.8,
    edges: Sequence[Any] | None = None,
) -> OverlapStats:
    """Measure same-zone system coincidence and overload, optionally including contagion.

    Contagion adds ``strength * 0.2`` to a destination node one rounded lag minute
    after every source minute above hi. Cross-zone effects use the original source
    profiles and are applied once, not recursively.
    """
    normalized = _as_profile_map(profiles)
    adjusted = _with_contagion(normalized, edges, hi)
    by_zone: dict[str, list[np.ndarray]] = defaultdict(list)
    values = list(adjusted.values())
    for node, load in adjusted.items():
        zone, _system = _node_parts(node)
        by_zone[zone].append(load)
    coincidence = 0
    for systems in by_zone.values():
        if len(systems) < 2:
            continue
        above_count = np.sum(np.stack([load > hi for load in systems]), axis=0)
        coincidence += int(np.count_nonzero(above_count >= 2))
    if not values:
        overload_area = 0.0
        peak = 0.0
    else:
        joined = np.concatenate(values) if any(len(value) for value in values) else np.asarray([])
        overload_area = float(np.sum(np.maximum(0.0, joined - _OVERLOAD_THRESHOLD)))
        peak = float(np.max(joined)) if joined.size else 0.0
    score = float(coincidence + 10.0 * overload_area)
    return OverlapStats(coincidence, overload_area, peak, score)


def _shift_movable(profile: np.ndarray, shift_min: int, movable_fraction: float) -> np.ndarray:
    """Shift movable load without wrap; displaced energy is clipped into edge bins.

    Overflow from outside the horizon is accumulated at the nearest edge sample, which
    preserves the total energy while avoiding circular wraparound. Since multiple moved
    values may meet at an edge, an edge load can exceed 1.0 and represents overload.
    """
    values = np.asarray(profile, dtype=float)
    if values.size == 0 or shift_min == 0:
        return values.copy()
    fraction = float(np.clip(movable_fraction, 0.0, 1.0))
    movable = values * fraction
    shifted = np.zeros_like(movable)
    shift = int(shift_min)
    if abs(shift) >= len(values):
        shifted[0 if shift < 0 else -1] = float(np.sum(movable))
    elif shift > 0:
        shifted[shift:] = movable[:-shift]
        shifted[-1] += float(np.sum(movable[-shift:]))
    else:
        width = -shift
        shifted[:-width] = movable[width:]
        shifted[0] += float(np.sum(movable[:width]))
    return values - movable + shifted


def _spec_for(node: str, flexibility: Mapping[str, FlexSpec] | None) -> FlexSpec:
    _zone, system = _node_parts(node)
    override = (flexibility or {}).get(node, (flexibility or {}).get(system))
    spec = override if override is not None else DEFAULT_FLEXIBILITY.get(system)
    if spec is None:
        spec = (0.0, 0, "shift_schedule", "power")
    movable, max_shift, action_type, dept = spec
    if action_type not in ACTION_TYPES:
        raise ValueError(f"Unsupported action type in flexibility spec: {action_type}")
    return float(np.clip(movable, 0.0, 1.0)), max(0, int(max_shift)), action_type, dept


def _apply_shifts(
    profiles: Mapping[str, np.ndarray],
    shifts: Mapping[str, int],
    flexibility: Mapping[str, FlexSpec] | None,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for node, profile in profiles.items():
        movable, _max_shift, _action_type, _dept = _spec_for(node, flexibility)
        result[node] = _shift_movable(profile, shifts.get(node, 0), movable)
    return result


def _contribution_order(
    profiles: Mapping[str, np.ndarray],
    flexibility: Mapping[str, FlexSpec] | None,
    edges: Sequence[Any] | None,
    hi: float,
    deadline: float,
) -> list[str]:
    baseline = peak_overlap(dict(profiles), hi=hi, edges=edges).score
    contributions: list[tuple[float, str]] = []
    for node in profiles:
        if time.perf_counter() >= deadline:
            contributions.append((0.0, node))
            continue
        without = dict(profiles)
        without[node] = np.zeros_like(profiles[node])
        contribution = baseline - peak_overlap(without, hi=hi, edges=edges).score
        contributions.append((contribution, node))
    return [node for _contribution, node in sorted(contributions, key=lambda item: (-item[0], item[1]))]


def rephase_optimizer(
    profiles: dict[str, np.ndarray],
    flexibility: dict[str, FlexSpec] | None = None,
    edges: Sequence[Any] | None = None,
    budget_ms: float = 1500,
    seed: int = 0,
    max_actions: int = 6,
    now: int = 0,
) -> RephaseResult:
    """Greedily coordinate-descent time shifts within budget_ms; times are minutes.

    Refinement by differential evolution is intentionally omitted: deterministic greedy
    refinement keeps execution predictable and leaves optimizer time within the deadline.
    The seed is reserved for compatible deterministic optimizer refinements.
    """
    del seed
    started = time.perf_counter()
    normalized = _as_profile_map(profiles)
    before = peak_overlap(normalized, edges=edges)
    horizon = len(next(iter(normalized.values()))) if normalized else 0
    deadline = started + max(0.0, float(budget_ms)) / 1000.0
    if not normalized or horizon == 0 or before.score <= 0.0 or max_actions <= 0:
        return RephaseResult({}, before, before, 0.0, [], (time.perf_counter() - started) * 1000)

    specs = {node: _spec_for(node, flexibility) for node in normalized}
    if time.perf_counter() >= deadline:
        return RephaseResult(
            {}, before, before, 0.0, [], (time.perf_counter() - started) * 1000
        )
    ordered_nodes = _contribution_order(normalized, flexibility, edges, hi=0.8, deadline=deadline)
    current_shifts: dict[str, int] = {node: 0 for node in normalized}
    current_profiles = {node: values.copy() for node, values in normalized.items()}
    current_score = before.score
    improved = True

    while improved and time.perf_counter() < deadline:
        improved = False
        for node in ordered_nodes:
            if time.perf_counter() >= deadline:
                break
            movable, max_shift, _action_type, _dept = specs[node]
            if max_shift == 0 or movable <= 0.0:
                continue
            grid = set(range(-max_shift, max_shift + 1, SHIFT_STEP_MIN))
            grid.update((-max_shift, max_shift))
            best_shift = current_shifts[node]
            best_score = current_score
            best_profile = current_profiles[node]
            for candidate_shift in sorted(grid):
                if time.perf_counter() >= deadline:
                    break
                if candidate_shift == current_shifts[node]:
                    continue
                candidate_profile = _shift_movable(normalized[node], candidate_shift, movable)
                trial_profiles = dict(current_profiles)
                trial_profiles[node] = candidate_profile
                candidate_score = peak_overlap(trial_profiles, edges=edges).score
                if candidate_score < best_score - 1e-9:
                    best_shift = candidate_shift
                    best_score = candidate_score
                    best_profile = candidate_profile
            if best_shift != current_shifts[node]:
                current_shifts[node] = best_shift
                current_profiles[node] = best_profile
                current_score = best_score
                improved = True

    individual_improvements: list[tuple[float, str]] = []
    for node, shift in current_shifts.items():
        if time.perf_counter() >= deadline:
            break
        if shift == 0:
            continue
        single_profiles = dict(normalized)
        single_profiles[node] = current_profiles[node]
        improvement = before.score - peak_overlap(single_profiles, edges=edges).score
        individual_improvements.append((improvement, node))
    selected_nodes = {
        node
        for improvement, node in sorted(individual_improvements, key=lambda item: (-item[0], item[1]))
        if improvement > 1e-9
    }
    selected_nodes = {
        node
        for improvement, node in sorted(
            (item for item in individual_improvements if item[0] > 1e-9),
            key=lambda item: (-item[0], item[1]),
        )[:max_actions]
    }
    selected_shifts = {node: shift for node, shift in current_shifts.items() if node in selected_nodes}
    after_profiles = _apply_shifts(normalized, selected_shifts, flexibility)
    after = peak_overlap(after_profiles, edges=edges)
    if after.score >= before.score - 1e-9:
        selected_shifts = {}
        after = before

    actions: list[Action] = []
    for node in sorted(selected_shifts):
        zone, _system = _node_parts(node)
        if zone not in ZONES:
            continue
        movable, _max_shift, action_type, dept = specs[node]
        actions.append(
            Action(
                id=f"rephase_{zone}_{action_type}_{max(0, int(now))}_{node.rsplit('.', 1)[-1]}",
                type=action_type,
                dept=dept,
                zone=zone,
                params={
                    "shift_min": selected_shifts[node],
                    "movable_fraction": movable,
                    "node": node,
                },
                start_t=max(0, int(now)),
                duration_min=horizon,
            )
        )
    if len(actions) != len(selected_shifts):
        valid_nodes = {node for node in selected_shifts if _node_parts(node)[0] in ZONES}
        selected_shifts = {node: shift for node, shift in selected_shifts.items() if node in valid_nodes}
        after = peak_overlap(_apply_shifts(normalized, selected_shifts, flexibility), edges=edges)

    return RephaseResult(
        shifts=dict(sorted(selected_shifts.items())),
        before_stats=before,
        after_stats=after,
        peak_overlap_cut_pct=cut_percent(before.score, after.score),
        actions=actions,
        runtime_ms=(time.perf_counter() - started) * 1000,
    )


def forecast_from_history(
    history: list[float], horizon: int, period: int = 1440
) -> np.ndarray:
    """Return a clock-aligned seasonal-naive forecast blended from the last value."""
    forecast_horizon = max(0, int(horizon))
    if forecast_horizon == 0:
        return np.asarray([], dtype=float)
    if not history:
        return np.zeros(forecast_horizon, dtype=float)
    values = np.clip(np.asarray(history, dtype=float), 0.0, 1.0)
    if values.ndim != 1:
        raise ValueError("history must be one-dimensional")
    cycle = max(1, int(period))
    pattern = values[-min(cycle, len(values)) :]
    indices = np.arange(forecast_horizon) % len(pattern)
    result = pattern[indices].copy()
    blend_minutes = min(10, forecast_horizon)
    alpha = np.arange(1, blend_minutes + 1, dtype=float) / blend_minutes
    result[:blend_minutes] = values[-1] * (1.0 - alpha) + result[:blend_minutes] * alpha
    return np.clip(result, 0.0, 1.0)


def make_demo_profiles(seed: int = 0) -> dict[str, np.ndarray]:
    """Build 8-zone traffic/power/water forecasts with a shared 21:00 evening peak."""
    rng = np.random.default_rng(seed)
    horizon = 360
    minutes = np.arange(horizon, dtype=float)
    profiles: dict[str, np.ndarray] = {}
    for zone in ZONES:
        zone_offset = int(rng.integers(-3, 4))
        for system in ("traffic", "power", "water"):
            peak = 180 + zone_offset + int(rng.integers(-2, 3))
            width = float(rng.uniform(21.0, 25.0))
            amplitude = float(rng.uniform(0.55, 0.62))
            base = float(rng.uniform(0.27, 0.32))
            profile = base + amplitude * np.exp(-0.5 * ((minutes - peak) / width) ** 2)
            profiles[f"{zone}.{system}"] = np.clip(profile, 0.0, 1.0)
    return profiles
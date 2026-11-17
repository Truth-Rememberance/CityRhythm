"""Deterministic service, resilience, recovery, and collapse metrics."""

from __future__ import annotations

import numpy as np

SERVICE_WEIGHTS: dict[str, float] = {"traffic": 0.3, "power": 0.4, "water": 0.3}
_SYSTEMS = ("traffic", "power", "water")


def _series(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError("service_series must be one-dimensional")
    return result


def service_level(
    loads: dict[str, float], capacity: dict[str, float] | None = None
) -> float:
    """Return zone service in [0, 1]; loads and capacities use normalized units.

    Each system supplies full service through 0.7 utilization, then declines linearly
    to 0.2 service at full utilization. When at least two systems exceed 0.85, their
    weighted zone service receives an additional 15% coupling penalty. Configure the
    traffic/power/water weights by updating SERVICE_WEIGHTS; they are normalized here.
    """
    capacities = capacity or {}
    weights = np.asarray([max(0.0, SERVICE_WEIGHTS.get(system, 0.0)) for system in _SYSTEMS])
    weight_total = float(np.sum(weights))
    if weight_total == 0.0:
        weights = np.full(len(_SYSTEMS), 1.0 / len(_SYSTEMS))
    else:
        weights /= weight_total

    utilization: list[float] = []
    system_services: list[float] = []
    for system in _SYSTEMS:
        load = float(loads.get(system, 0.0))
        system_capacity = float(capacities.get(system, 1.0))
        if system_capacity <= 0.0:
            load = 1.0 if load > 0.0 else 0.0
        else:
            load /= system_capacity
        load = float(np.clip(load, 0.0, 1.0))
        utilization.append(load)
        system_services.append(1.0 if load <= 0.7 else 1.0 - (0.8 / 0.3) * (load - 0.7))

    zone_service = float(np.dot(weights, system_services))
    if sum(load > 0.85 for load in utilization) >= 2:
        zone_service *= 0.85
    return float(np.clip(zone_service, 0.0, 1.0))


def resilience_score(
    service_series: np.ndarray, baseline: float = 1.0, incident_start: int = 0
) -> float:
    """Return clipped 0-100 area-under-service score after incident_start minute."""
    values = np.clip(_series(service_series), 0.0, 1.0)
    start = min(max(int(incident_start), 0), len(values))
    remaining = values[start:]
    if remaining.size == 0 or baseline <= 0.0:
        return 0.0
    score = 100.0 * float(np.sum(remaining)) / (float(baseline) * remaining.size)
    return float(np.clip(score, 0.0, 100.0))


def resilience_curve(service_series: np.ndarray, window: int = 30) -> np.ndarray:
    """Return trailing rolling service resilience, 0-100 per minute.

    Early samples use all available history; subsequent samples use at most `window`
    minutes. Values are clipped to normalized service units before averaging.
    """
    values = np.clip(_series(service_series), 0.0, 1.0)
    if window <= 0:
        raise ValueError("window must be positive")
    if values.size == 0:
        return np.asarray([], dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(1, values.size + 1)
    starts = np.maximum(0, ends - window)
    sums = cumulative[ends] - cumulative[starts]
    counts = ends - starts
    return 100.0 * sums / counts


def _recovery_result(
    service_series: np.ndarray,
    incident_start: int,
    threshold: float,
    sustain_min: int,
    cap: int | None,
) -> tuple[int, bool]:
    values = _series(service_series)
    start = min(max(int(incident_start), 0), len(values))
    default_cap = len(values) - start
    recovery_cap = default_cap if cap is None else max(0, int(cap))
    if sustain_min <= 0:
        raise ValueError("sustain_min must be positive")
    dropped = np.flatnonzero(values[start:] < threshold)
    if dropped.size == 0:
        return 0, True

    drop_index = start + int(dropped[0])
    final_start = len(values) - sustain_min
    for recovery_start in range(drop_index + 1, final_start + 1):
        if np.all(values[recovery_start : recovery_start + sustain_min] >= threshold):
            return recovery_start - drop_index, True
    return recovery_cap, False


def recovery_time(
    service_series: np.ndarray,
    incident_start: int,
    threshold: float = 0.95,
    sustain_min: int = 10,
    cap: int | None = None,
) -> int:
    """Return recovery duration in minutes, or the cap if no sustained recovery occurs.

    A series that never drops below threshold has recovery time 0. Recovery requires
    `sustain_min` consecutive samples at or above threshold. The default failure cap is
    the number of samples from incident_start through the end of the series.
    """
    duration, _ = _recovery_result(service_series, incident_start, threshold, sustain_min, cap)
    return duration


def did_recover(
    service_series: np.ndarray,
    incident_start: int,
    threshold: float = 0.95,
    sustain_min: int = 10,
) -> bool:
    """Return whether the series avoids a drop or achieves sustained recovery."""
    _, recovered = _recovery_result(
        service_series, incident_start, threshold, sustain_min, cap=None
    )
    return recovered


def time_to_collapse(service_series: np.ndarray, collapse: float = 0.5) -> int | None:
    """Return the first minute below collapse, or None when collapse is not reached."""
    values = _series(service_series)
    collapsed = np.flatnonzero(values < collapse)
    return int(collapsed[0]) if collapsed.size else None


def cut_percent(before: float, after: float) -> float:
    """Return percentage reduction from before to after; zero baseline yields 0."""
    baseline = float(before)
    if baseline <= 0.0:
        return 0.0
    return 100.0 * (baseline - float(after)) / baseline
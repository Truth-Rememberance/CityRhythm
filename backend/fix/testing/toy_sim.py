"""Deterministic ROLE 5 simulator stand-in with tipping and delayed coupling."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from backend.fix._local_contracts import ZONES, Action, Reading, Tick
from backend.fix.metrics import service_level

_IST_START = datetime.fromisoformat("2026-11-17T00:00:00+05:30")
_SYSTEMS = ("traffic", "power", "water")
_SYSTEM_BASE = {"traffic": 0.28, "power": 0.30, "water": 0.27}
_SYSTEM_AMPLITUDE = {"traffic": 0.24, "power": 0.22, "water": 0.20}
_SYSTEM_PEAK_MIN = {"traffic": 1260, "power": 1140, "water": 1080}
_REGIONAL_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "silk_board": ("koramangala", "electronic_city", "jayanagar"),
    "whitefield": ("indiranagar", "yelahanka", "koramangala"),
    "hebbal": ("yelahanka", "indiranagar", "whitefield"),
    "electronic_city": ("silk_board", "jayanagar", "koramangala"),
    "koramangala": ("silk_board", "indiranagar", "jayanagar"),
    "indiranagar": ("koramangala", "whitefield", "hebbal"),
    "jayanagar": ("silk_board", "electronic_city", "koramangala"),
    "yelahanka": ("hebbal", "whitefield", "indiranagar"),
}
_RECOVERY_HYSTERESIS_MIN = 10
_OVERLOADED_SERVICE_FACTOR = 0.28
_COUPLING_SPREAD_SCALE = 0.65


class ToySim:
    """Small deterministic Sim implementation; t and all windows use minutes."""

    couplings: tuple[dict[str, Any], ...] = (
        {"src": "silk_board.traffic", "dst": "silk_board.power", "lag_min": 12, "strength": 0.8},
        {"src": "silk_board.power", "dst": "koramangala.traffic", "lag_min": 8, "strength": 0.8},
        {"src": "whitefield.power", "dst": "whitefield.water", "lag_min": 8, "strength": 0.55},
        {"src": "koramangala.traffic", "dst": "koramangala.power", "lag_min": 10, "strength": 0.6},
        {"src": "koramangala.power", "dst": "indiranagar.traffic", "lag_min": 8, "strength": 0.8},
        {"src": "indiranagar.traffic", "dst": "indiranagar.power", "lag_min": 10, "strength": 0.6},
        {"src": "indiranagar.power", "dst": "whitefield.water", "lag_min": 8, "strength": 0.75},
        {"src": "whitefield.water", "dst": "hebbal.power", "lag_min": 10, "strength": 0.7},
        {"src": "hebbal.water", "dst": "hebbal.power", "lag_min": 15, "strength": 0.5},
    )

    def __init__(self, seed: int = 0, start_t: int = 1200) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._now = max(0, int(start_t))
        self._zone_bias = {zone: float(self._rng.uniform(-0.035, 0.035)) for zone in ZONES}
        self._injections: list[dict[str, Any]] = []
        self._actions: list[Action] = []
        self._shift_queue: dict[int, dict[str, float]] = {}
        self._loads = {f"{zone}.{system}": 0.0 for zone in ZONES for system in _SYSTEMS}
        self._over_threshold_minutes = dict.fromkeys(self._loads, 0)
        self._overloaded: set[str] = set()
        self._recovery_minutes = dict.fromkeys(self._loads, 0)
        self._overload_started: dict[str, int] = {}
        self._spread_triggers: set[str] = set()
        self._last_service: dict[str, float] = {zone: 1.0 for zone in ZONES}

    def now(self) -> int:
        """Return the current simulation minute."""
        return self._now

    def snapshot(self) -> dict[str, Any]:
        """Return a detached, serializable snapshot of simulation state."""
        return {
            "now": self._now,
            "loads": dict(self._loads),
            "overloaded_nodes": sorted(self._overloaded),
            "overload_duration": {
                node: max(0, self._now - started + 1)
                for node, started in self._overload_started.items()
                if node in self._overloaded
            },
            "injections": copy.deepcopy(self._injections),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def fork(self) -> ToySim:
        """Return an independent deep copy at the same state and RNG position."""
        return copy.deepcopy(self)

    def inject(self, name: str, **kw: Any) -> None:
        """Schedule a supported disturbance; duration and start_t are simulation minutes."""
        if name not in {"heatwave", "match_end_9pm", "signal_hijack"}:
            raise ValueError(f"Unsupported toy simulator injection: {name}")
        start_t = kw.pop("start_t", None)
        if start_t is None:
            start_t = self._next_match_minute() if name == "match_end_9pm" else self._now
        default_duration = {"heatwave": 90, "match_end_9pm": 60, "signal_hijack": 60}[name]
        duration = max(1, int(kw.pop("duration_min", default_duration)))
        zone = kw.pop("zone", None)
        if zone is not None and zone not in ZONES:
            raise ValueError(f"Unknown zone: {zone}")
        injection = {
            "name": name,
            "start_t": max(0, int(start_t)),
            "end_t": max(0, int(start_t)) + duration,
            "zone": zone,
            "params": dict(kw),
        }
        self._injections.append(injection)

    def apply_action(self, action: Action) -> None:
        """Schedule or apply an intervention; effects last for duration_min minutes."""
        if action.id not in {item.id for item in self._actions}:
            self._actions.append(action.model_copy(deep=True))

    def step(self) -> Tick:
        """Advance one simulation minute and return readings plus zone service levels."""
        self._now += 1
        raw_loads = self._base_loads()
        self._add_injection_loads(raw_loads)
        self._apply_action_effects(raw_loads)
        queued = self._shift_queue.pop(self._now, {})
        for node, amount in queued.items():
            raw_loads[node] = raw_loads.get(node, 0.0) + amount
        self._update_tipping(raw_loads)

        readings: list[Reading] = []
        for node in sorted(raw_loads):
            zone, system = node.split(".", 1)
            value = float(np.clip(raw_loads[node], 0.0, 1.0))
            self._loads[node] = value
            readings.append(
                Reading(
                    zone=zone,
                    system=system,
                    sensor_id=f"{zone}.{system}.01",
                    value=value,
                    t=self._now,
                    ts=(_IST_START + timedelta(minutes=self._now)).isoformat(timespec="seconds"),
                )
            )

        service_by_zone: dict[str, float] = {}
        for zone in ZONES:
            zone_loads = {system: self._loads[f"{zone}.{system}"] for system in _SYSTEMS}
            zone_service = service_level(zone_loads)
            if any(node.startswith(f"{zone}.") for node in self._overloaded):
                zone_service *= _OVERLOADED_SERVICE_FACTOR
            service_by_zone[zone] = float(np.clip(zone_service, 0.0, 1.0))
        self._last_service = service_by_zone
        durations = {
            node: self._now - self._overload_started[node] + 1
            for node in self._overloaded
            if node in self._overload_started
        }
        reports = {
            "service_level": dict(service_by_zone),
            "overload_duration": durations,
            "overloaded_nodes": sorted(self._overloaded),
            "loads": dict(self._loads),
            "couplings": list(self.couplings),
        }
        return Tick(
            t=self._now,
            ts=(_IST_START + timedelta(minutes=self._now)).isoformat(timespec="seconds"),
            readings=readings,
            reports=reports,
        )

    def _next_match_minute(self) -> int:
        day_start = self._now - self._now % 1440
        match = day_start + 1260
        return match if match >= self._now else match + 1440

    def _base_loads(self) -> dict[str, float]:
        minute_of_day = self._now % 1440
        loads: dict[str, float] = {}
        for zone in ZONES:
            for system in _SYSTEMS:
                angle = 2.0 * np.pi * (minute_of_day - _SYSTEM_PEAK_MIN[system]) / 1440.0
                daily = _SYSTEM_BASE[system] + _SYSTEM_AMPLITUDE[system] * (1.0 + np.cos(angle)) / 2.0
                noise = float(self._rng.normal(0.0, 0.006))
                node = f"{zone}.{system}"
                loads[node] = max(0.0, daily + self._zone_bias[zone] + noise)
        return loads

    def _add_injection_loads(self, loads: dict[str, float]) -> None:
        for injection in self._injections:
            if not injection["start_t"] <= self._now < injection["end_t"]:
                continue
            name = injection["name"]
            zone = injection["zone"]
            params = injection["params"]
            default_boost = {"heatwave": 0.46, "match_end_9pm": 0.42, "signal_hijack": 0.42}[name]
            boost = float(params.get("boost", default_boost))
            ramp_min = max(1, int(params.get("ramp_min", 7)))
            ramp_progress = min(1.0, (self._now - injection["start_t"] + 1) / ramp_min)
            boost *= ramp_progress
            target_zones = (
                tuple(dict.fromkeys((zone, *_REGIONAL_NEIGHBORS.get(zone, ()))))
                if zone
                else ZONES
            )
            for target_zone in target_zones:
                systems = _SYSTEMS if name == "heatwave" else ("traffic",)
                for system in systems:
                    node = f"{target_zone}.{system}"
                    loads[node] = loads.get(node, 0.0) + boost

    def _apply_action_effects(self, loads: dict[str, float]) -> None:
        for action in self._actions:
            start = max(0, action.start_t)
            end = start + action.duration_min
            if not start <= self._now < end:
                continue
            params = action.params
            if action.type in ("shift_pump", "shift_schedule"):
                system = "water" if action.type == "shift_pump" else "power"
                node = f"{action.zone}.{system}"
                fraction = float(np.clip(params.get("movable_fraction", 0.35), 0.0, 1.0))
                delay = max(1, int(params.get("shift_min", 30)))
                moved = loads.get(node, 0.0) * fraction
                loads[node] = max(0.0, loads.get(node, 0.0) - moved)
                due = self._now + delay
                queue = self._shift_queue.setdefault(due, {})
                queue[node] = queue.get(node, 0.0) + moved
            elif action.type == "offset_signal":
                node = f"{action.zone}.traffic"
                reduction = min(0.6, max(0.0, float(params.get("offset_s", 20))) / 100.0)
                loads[node] = max(0.0, loads.get(node, 0.0) - reduction)
            elif action.type == "shed_load":
                node = f"{action.zone}.power"
                pct = float(np.clip(params.get("pct", 25.0), 0.0, 100.0)) / 100.0
                loads[node] = max(0.0, loads.get(node, 0.0) * (1.0 - pct))
            elif action.type == "reroute":
                source = f"{action.zone}.traffic"
                destination_zone = params.get("to_zone")
                if destination_zone not in ZONES:
                    continue
                fraction = float(np.clip(params.get("fraction", 0.1), 0.0, 0.5))
                moved = loads.get(source, 0.0) * fraction
                loads[source] = max(0.0, loads.get(source, 0.0) - moved)
                traffic_destination = f"{destination_zone}.traffic"
                power_destination = f"{destination_zone}.power"
                loads[traffic_destination] = loads.get(traffic_destination, 0.0) + moved
                loads[power_destination] = loads.get(power_destination, 0.0) + moved * 0.25

    def _update_tipping(self, loads: dict[str, float]) -> None:
        for node, load in loads.items():
            if node in self._overloaded:
                if load < 0.6:
                    self._recovery_minutes[node] += 1
                else:
                    self._recovery_minutes[node] = 0
                if self._recovery_minutes[node] >= _RECOVERY_HYSTERESIS_MIN:
                    self._overloaded.remove(node)
                    self._overload_started.pop(node, None)
                    self._recovery_minutes[node] = 0
                    self._over_threshold_minutes[node] = 0
                else:
                    loads[node] = 1.0
                continue
            if load > 0.85:
                self._over_threshold_minutes[node] += 1
                if self._over_threshold_minutes[node] >= 5:
                    self._overloaded.add(node)
                    self._overload_started[node] = self._now - 4
                    loads[node] = 1.0
                    self._schedule_spread(node)
            else:
                self._over_threshold_minutes[node] = 0

    def _schedule_spread(self, source: str) -> None:
        if source in self._spread_triggers:
            return
        self._spread_triggers.add(source)
        for edge in self.couplings:
            if edge["src"] != source:
                continue
            start = self._now + int(edge["lag_min"])
            amount = float(edge["strength"]) * _COUPLING_SPREAD_SCALE
            target = edge["dst"]
            for minute in range(start, start + 5):
                queue = self._shift_queue.setdefault(minute, {})
                queue[target] = queue.get(target, 0.0) + amount
"""Checkpointed, deterministic no-action versus plan counterfactual coordinator."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import numpy as np

from backend.fix._local_contracts import (
    CounterfactualPayload,
    Envelope,
    Tick,
    make_envelope,
)
from backend.fix.conflicts import Plan
from backend.fix.metrics import did_recover, recovery_time, resilience_curve, service_level
from backend.fix.sim_protocol import Sim

logger = logging.getLogger(__name__)


class CounterfactualBranch(TypedDict):
    resilience: list[float]
    service_level: list[float]


class CounterfactualResult(TypedDict):
    """Frozen counterfactual payload plus non-contract metadata under extras."""

    t: list[int]
    no_action: CounterfactualBranch
    with_plan: CounterfactualBranch
    recovery_min: dict[str, int]
    extras: dict[str, Any]


@dataclass(frozen=True)
class SavedCheckpoint:
    t: int
    sim: Sim


@dataclass(frozen=True)
class InterventionCheckpoint:
    t: int
    saved: bool
    min_service: float


@dataclass(frozen=True)
class InterventionWindow:
    last_effective_t: int | None
    first_checkpoint_t: int | None
    saved_by_t: list[InterventionCheckpoint]
    root_cause_t: int | None
    root_cause_node: str | None

    def payload(self) -> dict[str, Any]:
        """Return the JSON-ready intervention-window event payload."""
        return {
            "last_effective_t": self.last_effective_t,
            "first_checkpoint_t": self.first_checkpoint_t,
            "saved_by_t": [
                {"t": point.t, "saved": point.saved, "min_service": point.min_service}
                for point in self.saved_by_t
            ],
            "root_cause_t": self.root_cause_t,
            "root_cause_node": self.root_cause_node,
        }


class PlanFactory(Protocol):
    def __call__(self, checkpoint_t: int) -> Plan: ...


@dataclass
class _BranchRun:
    service: np.ndarray
    per_zone_service: dict[str, list[float]]
    t: list[int]
    max_overload_duration: dict[str, int]
    root_cause_t: int | None
    root_cause_node: str | None


class CounterfactualCoordinator:
    """Compare independent simulator forks and retain bounded lightweight checkpoints.

    Only one async job may run at a time; overlapping requests are dropped and
    ``run_async``/``run_intervention_window_async`` return False when busy.
    """

    def __init__(
        self,
        sim_factory_or_sim: Callable[[], Sim] | Sim,
        horizon_min: int = 180,
        checkpoint_every: int = 10,
        max_checkpoints: int = 36,
    ) -> None:
        self.horizon_min = max(0, int(horizon_min))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.max_checkpoints = max(1, int(max_checkpoints))
        self._factory: Callable[[], Sim] | None = (
            sim_factory_or_sim if callable(sim_factory_or_sim) else None
        )
        self._live_sim: Sim | None = None if self._factory else sim_factory_or_sim  # type: ignore[assignment]
        self._checkpoints: deque[SavedCheckpoint] = deque(maxlen=self.max_checkpoints)
        self._checkpoint_lock = threading.RLock()
        self._async_lock = threading.RLock()
        self._async_thread: threading.Thread | None = None
        self._async_result: CounterfactualResult | InterventionWindow | None = None
        self._async_error: BaseException | None = None

    def observe(self, sim: Sim, t: int) -> None:
        """Store a fork every checkpoint interval in a bounded ring; otherwise return quickly."""
        tick = max(0, int(t))
        if tick % self.checkpoint_every != 0:
            return
        checkpoint = SavedCheckpoint(tick, sim.fork())
        with self._checkpoint_lock:
            if self._checkpoints and self._checkpoints[-1].t == tick:
                self._checkpoints[-1] = checkpoint
            else:
                self._checkpoints.append(checkpoint)

    def run(self, plan: Plan, sim_now: int, incident_start: int) -> CounterfactualResult:
        """Run independent no-action/with-plan branches and return the frozen payload plus extras."""
        start = max(0, int(sim_now))
        base = self._starting_sim(start)
        no_action_sim, plan_sim = base.fork(), base.fork()
        for action in plan.actions:
            plan_sim.apply_action(action)

        no_action = self._run_branch(no_action_sim, start)
        with_plan = self._run_branch(plan_sim, start)
        incident_index = self._incident_index(incident_start, start, self.horizon_min)
        no_action_recovered = did_recover(no_action.service, incident_index)
        with_plan_recovered = did_recover(with_plan.service, incident_index)
        no_action_recovery = recovery_time(
            no_action.service, incident_index, cap=self.horizon_min - incident_index
        )
        with_plan_recovery = recovery_time(
            with_plan.service, incident_index, cap=self.horizon_min - incident_index
        )
        no_action_resilience = resilience_curve(no_action.service).tolist()
        with_plan_resilience = resilience_curve(with_plan.service).tolist()

        frozen_payload = {
            "t": no_action.t,
            "no_action": {
                "resilience": no_action_resilience,
                "service_level": no_action.service.tolist(),
            },
            "with_plan": {
                "resilience": with_plan_resilience,
                "service_level": with_plan.service.tolist(),
            },
            "recovery_min": {
                "no_action": no_action_recovery,
                "with_plan": with_plan_recovery,
            },
        }
        validated = CounterfactualPayload.model_validate(frozen_payload)
        extras = {
            "per_zone": {
                "no_action": no_action.per_zone_service,
                "with_plan": with_plan.per_zone_service,
            },
            "per_zone_resilience": {
                "no_action": {
                    zone: resilience_curve(np.asarray(values, dtype=float)).tolist()
                    for zone, values in no_action.per_zone_service.items()
                },
                "with_plan": {
                    zone: resilience_curve(np.asarray(values, dtype=float)).tolist()
                    for zone, values in with_plan.per_zone_service.items()
                },
            },
            "no_action_recovered": no_action_recovered,
            "with_plan_recovered": with_plan_recovered,
            "slowest_link": self._slowest_link(no_action),
            "root_cause_t": no_action.root_cause_t,
            "root_cause_node": no_action.root_cause_node,
        }
        return {
            **validated.model_dump(),
            "extras": extras,
        }

    def run_async(self, plan: Plan, sim_now: int, incident_start: int) -> bool:
        """Start one daemon counterfactual job; return False without queuing if busy."""
        return self._start_async(self.run, plan, sim_now, incident_start)

    def intervention_window(
        self, plan_factory: PlanFactory, incident_start: int
    ) -> InterventionWindow:
        """Evaluate stored checkpoints linearly to find the last effective intervention time."""
        with self._checkpoint_lock:
            checkpoints = [
                SavedCheckpoint(item.t, item.sim.fork())
                for item in self._checkpoints
                if item.t >= incident_start
            ]
        if not checkpoints:
            return InterventionWindow(None, None, [], None, None)

        first_t = checkpoints[0].t
        baseline = self._run_branch(checkpoints[0].sim.fork(), first_t)
        root_t, root_node = baseline.root_cause_t, baseline.root_cause_node
        incident_index = 0
        baseline_recovery = recovery_time(
            baseline.service, incident_index, cap=self.horizon_min
        )
        collapse_offset = next(
            (i for i, value in enumerate(baseline.service) if value < 0.5), self.horizon_min
        )
        collapse_t = first_t + collapse_offset
        saved_points: list[InterventionCheckpoint] = []

        for checkpoint in checkpoints:
            if checkpoint.t > collapse_t:
                break
            plan = plan_factory(checkpoint.t)
            no_action = self._run_branch(checkpoint.sim.fork(), checkpoint.t)
            plan_sim = checkpoint.sim.fork()
            for action in plan.actions:
                plan_sim.apply_action(action)
            with_plan = self._run_branch(plan_sim, checkpoint.t)
            minimum = float(np.min(with_plan.service)) if with_plan.service.size else 1.0
            with_plan_recovered = did_recover(with_plan.service, 0)
            with_plan_recovery = recovery_time(
                with_plan.service, 0, cap=self.horizon_min
            )
            no_action_recovered = did_recover(no_action.service, 0)
            no_action_recovery = recovery_time(no_action.service, 0, cap=self.horizon_min)
            bound = (
                no_action_recovery
                if no_action_recovered
                else max(no_action_recovery, baseline_recovery)
            )
            saved = (
                minimum > 0.5
                and with_plan_recovered
                and (not no_action_recovered or with_plan_recovery <= bound)
            )
            saved_points.append(InterventionCheckpoint(checkpoint.t, saved, minimum))

        effective = [point.t for point in saved_points if point.saved]
        last_effective = max(effective) if effective else None
        return InterventionWindow(last_effective, first_t, saved_points, root_t, root_node)

    def run_intervention_window_async(
        self, plan_factory: PlanFactory, incident_start: int
    ) -> bool:
        """Run intervention-window evaluation in the same single-job daemon slot."""
        return self._start_async(self.intervention_window, plan_factory, incident_start)

    def poll(self) -> CounterfactualResult | InterventionWindow | None:
        """Return and clear the completed async result, or None while pending/failed."""
        with self._async_lock:
            result = self._async_result
            self._async_result = None
            return result

    def is_running(self) -> bool:
        """Return whether the one async worker is still running."""
        with self._async_lock:
            return self._async_thread is not None and self._async_thread.is_alive()

    def to_envelopes(
        self,
        result: CounterfactualResult | InterventionWindow,
        run_id: str,
        t: int,
    ) -> list[Envelope]:
        """Wrap counterfactual and optional intervention-window payloads as events."""
        if isinstance(result, InterventionWindow):
            return [make_envelope("metrics.intervention_window", result.payload(), run_id, t)]
        payload = {key: value for key, value in result.items() if key != "extras"}
        envelopes = [make_envelope("metrics.counterfactual", payload, run_id, t)]
        intervention = result.get("extras", {}).get("intervention_window")
        if isinstance(intervention, InterventionWindow):
            envelopes.append(
                make_envelope("metrics.intervention_window", intervention.payload(), run_id, t)
            )
        elif isinstance(intervention, Mapping):
            envelopes.append(
                make_envelope("metrics.intervention_window", dict(intervention), run_id, t)
            )
        return envelopes

    def _start_async(self, target: Callable[..., Any], *args: Any) -> bool:
        with self._async_lock:
            if self._async_thread is not None and self._async_thread.is_alive():
                return False
            self._async_result = None
            self._async_error = None

            def worker() -> None:
                try:
                    result = target(*args)
                except Exception as error:
                    with self._async_lock:
                        self._async_error = error
                    logger.exception("Counterfactual background job failed")
                else:
                    with self._async_lock:
                        self._async_result = result

            self._async_thread = threading.Thread(target=worker, daemon=True)
            self._async_thread.start()
            return True

    def _starting_sim(self, sim_now: int) -> Sim:
        with self._checkpoint_lock:
            earlier = [point for point in self._checkpoints if point.t <= sim_now]
            if earlier:
                sim = earlier[-1].sim.fork()
                while sim.now() < sim_now:
                    sim.step()
                return sim
        if self._factory is not None:
            sim = self._factory()
        else:
            if self._live_sim is None:
                raise RuntimeError("No simulator is configured")
            sim = self._live_sim.fork()
        while sim.now() < sim_now:
            sim.step()
        return sim

    def _run_branch(self, sim: Sim, sim_now: int) -> _BranchRun:
        service_rows: list[float] = []
        per_zone: dict[str, list[float]] = {}
        ticks: list[int] = []
        max_overload: dict[str, int] = {}
        root_time: int | None = None
        root_node: str | None = None
        for _ in range(self.horizon_min):
            tick = sim.step()
            ticks.append(int(tick.t))
            zones = self._tick_zone_services(tick)
            if not zones:
                zones = {"silk_board": 1.0}
            for zone, value in zones.items():
                per_zone.setdefault(zone, []).append(value)
            service_rows.append(float(np.mean(list(zones.values()))))
            reports = tick.reports if isinstance(tick.reports, Mapping) else {}
            durations = reports.get("overload_duration", {})
            if isinstance(durations, Mapping):
                for node, duration in durations.items():
                    try:
                        max_overload[str(node)] = max(max_overload.get(str(node), 0), int(duration))
                    except (TypeError, ValueError):
                        continue
            if root_time is None:
                loads = reports.get("loads", {})
                if isinstance(loads, Mapping):
                    newly_overloaded = [
                        (str(node), float(value))
                        for node, value in loads.items()
                        if float(value) > 0.85
                    ]
                    if newly_overloaded:
                        root_node = min(newly_overloaded)[0]
                        root_time = tick.t
        return _BranchRun(
            service=np.asarray(service_rows, dtype=float),
            per_zone_service=per_zone,
            t=ticks,
            max_overload_duration=max_overload,
            root_cause_t=root_time,
            root_cause_node=root_node,
        )

    @staticmethod
    def _tick_zone_services(tick: Tick) -> dict[str, float]:
        reports = tick.reports if isinstance(tick.reports, Mapping) else {}
        reported = reports.get("service_level")
        if isinstance(reported, Mapping):
            result: dict[str, float] = {}
            for zone, value in reported.items():
                try:
                    result[str(zone)] = float(np.clip(float(value), 0.0, 1.0))
                except (TypeError, ValueError):
                    continue
            if result:
                return result
        loads: dict[str, dict[str, float]] = {}
        for reading in tick.readings:
            try:
                loads.setdefault(reading.zone, {})[reading.system] = float(reading.value)
            except (TypeError, ValueError):
                continue
        return {zone: service_level(values) for zone, values in loads.items()}

    @staticmethod
    def _incident_index(incident_start: int, sim_now: int, horizon: int) -> int:
        # The first collected sample is sim_now + 1; past/current incidents begin at sample 0.
        return min(horizon, max(0, int(incident_start) - sim_now - 1))

    @staticmethod
    def _slowest_link(branch: _BranchRun) -> dict[str, Any] | None:
        if not branch.max_overload_duration:
            return None
        node, duration = max(
            branch.max_overload_duration.items(), key=lambda item: (item[1], item[0])
        )
        return {"node": node, "duration_min": duration}
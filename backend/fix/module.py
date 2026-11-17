"""Minimal ROLE 5 module wiring for proposals, plans, metrics, and async proof."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from backend.fix._local_contracts import Envelope, Reading, Tick, make_envelope
from backend.fix.agents import Proposal, ZoneSnapshot, propose_all, snapshot_from_events
from backend.fix.conflicts import Plan, build_plan, conflicts_enabled, detect_conflicts, resolve
from backend.fix.copilot import brief
from backend.fix.counterfactual import (
    CounterfactualCoordinator,
    InterventionWindow,
)
from backend.fix.fragility import ZoneState, edges_to_coupling, fragility, load_costs
from backend.fix.metrics import resilience_curve, service_level
from backend.fix.rephase import forecast_from_history, rephase_optimizer

logger = logging.getLogger(__name__)


class _ContextView:
    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def latest(self, channel: str, zone: str | None) -> Any:
        try:
            latest = self._ctx.get("latest") if isinstance(self._ctx, Mapping) else getattr(
                self._ctx, "latest", None
            )
            if not callable(latest):
                return None
            value = latest(channel, zone)
            return _payload(value)
        except Exception:  # noqa: BLE001 - malformed upstream events are treated as missing context.
            return None

    def history(self, zone: str, system: str, minutes: int) -> list[float]:
        try:
            history = self._ctx.get("history") if isinstance(self._ctx, Mapping) else getattr(
                self._ctx, "history", None
            )
            values = history(zone, system, minutes) if callable(history) else []
            return [float(value) for value in values]
        except Exception:  # noqa: BLE001 - context storage is an optional dependency.
            return []


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _payload(value: Any) -> Any:
    mapping = _mapping(value)
    return mapping.get("payload", value) if mapping else getattr(value, "payload", value)


def _context_value(ctx: Any, name: str) -> Any:
    return ctx.get(name) if isinstance(ctx, Mapping) else getattr(ctx, name, None)


def _tick_parts(value: Any) -> tuple[Tick, str]:
    outer = _mapping(value)
    payload = outer.get("payload", outer)
    payload = _mapping(payload)
    t = max(0, int(payload.get("t", outer.get("t", 0)) or 0))
    run_id = str(outer.get("run_id", payload.get("run_id", "r_01")))
    readings: list[Reading] = []
    for raw in payload.get("readings", []) or []:
        item = _mapping(raw)
        zone, system = item.get("zone"), item.get("system")
        if not zone or system not in ("traffic", "power", "water"):
            continue
        try:
            readings.append(
                Reading(
                    zone=str(zone),
                    system=system,
                    sensor_id=str(item.get("sensor_id", f"{zone}.{system}.01")),
                    value=float(item.get("value", 0.0)),
                    unit=str(item.get("unit", "normalized")),
                    t=item.get("t", t),
                    ts=item.get("ts"),
                )
            )
        except (TypeError, ValueError):
            continue
    context = _mapping(payload.get("context", {}))
    events = context.get("events", [])
    if not isinstance(events, list) or not all(isinstance(item, str) for item in events):
        events = []
    reports = payload.get("reports", {})
    if not isinstance(reports, Mapping):
        reports = {"items": reports if isinstance(reports, list) else []}
    ts = str(payload.get("ts", outer.get("ts", "")))
    if not ts:
        ts = make_envelope("fix.plan", {}, run_id, t).ts
    tick = Tick(
        t=t,
        ts=ts,
        readings=readings,
        context={"temp_c": context.get("temp_c"), "events": events, "is_holiday": bool(context.get("is_holiday", False))},
        reports=dict(reports),
    )
    return tick, run_id


def _proposal_payload(proposal: Proposal) -> dict[str, Any]:
    return {
        "action": proposal.action.model_dump(),
        "utility": proposal.utility,
        "rationale": proposal.rationale,
        "touches": proposal.touches,
        "expected_load_delta": proposal.expected_load_delta,
    }


class FixModule:
    """Coordinating ROLE 5 interface; slow LLM and counterfactual work stays in threads."""

    def __init__(self, name: str = "fix") -> None:
        self.name = name
        self.publish_every = 5
        self.auto_mode = False
        self._tick_count = 0
        self._costs = load_costs()
        self._service_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._current_plan: Plan | None = None
        self._counterfactual: CounterfactualCoordinator | None = None
        self._counterfactual_phase: str | None = None
        self._counterfactual_plan_id: str | None = None
        self._latest_run_id = "r_01"
        self._latest_tick = 0
        self._brief_lock = threading.Lock()
        self._brief_thread: threading.Thread | None = None
        self._brief_result: dict[str, Any] | None = None
        self._brief_error: str | None = None

    def on_tick(self, tick: Any, ctx: Any) -> list[Envelope]:
        try:
            normalized_tick, run_id = _tick_parts(tick)
            now = normalized_tick.t
            self._latest_run_id = run_id
            self._latest_tick = now
            context = _ContextView(ctx)
            sim = _context_value(ctx, "sim") or _context_value(ctx, "simulator")
            events = self._poll_background(run_id, now)
            if sim is not None:
                if self._counterfactual is None:
                    self._counterfactual = CounterfactualCoordinator(
                        sim, horizon_min=180, checkpoint_every=1, max_checkpoints=36
                    )
                self._counterfactual.observe(sim, now)
            self._poll_briefing(events, run_id, now)

            self._tick_count += 1
            if self._tick_count % self.publish_every:
                return events

            zones = sorted({reading.zone for reading in normalized_tick.readings})
            snapshots = [snapshot_from_events(context, zone, normalized_tick) for zone in zones]
            loads = {
                f"{reading.zone}.{reading.system}": reading.value
                for reading in normalized_tick.readings
            }
            service_by_zone = self._service_by_zone(normalized_tick)
            fragility_by_zone: dict[str, float] = {}
            latest_graph = context.latest("sync.graph", None)
            edges = latest_graph.get("edges", []) if isinstance(latest_graph, Mapping) else []
            for snapshot in snapshots:
                sync = context.latest("sync.index", snapshot.zone)
                try:
                    csd_score = float(_mapping(sync).get("csd_score", 0.0))
                except (TypeError, ValueError):
                    csd_score = 0.0
                incoming, outgoing = edges_to_coupling(edges, snapshot.zone)
                state = ZoneState(
                    loads=snapshot.loads,
                    sync_index=snapshot.sync_index,
                    csd_score=csd_score,
                    coupling_strength_in=incoming,
                    coupling_strength_out=outgoing,
                )
                fragility_by_zone[snapshot.zone] = fragility(snapshot.zone, state)

            proposals = propose_all(snapshots, self._costs)
            conflicts = (
                detect_conflicts(proposals, loads, costs=self._costs)
                if conflicts_enabled()
                else []
            )
            resolution = resolve(proposals, conflicts, loads, self._costs)
            plan = build_plan(resolution, now)
            rephase = self._rephase(context, snapshots, now)
            if rephase.actions:
                plan = Plan(
                    plan_id=plan.plan_id,
                    actions=[*plan.actions, *rephase.actions],
                    total_cost=plan.total_cost,
                    expected_effect=plan.expected_effect,
                    notes=[*plan.notes, f"Re-phased {len(rephase.actions)} node(s)."],
                )
            self._current_plan = plan

            events.append(
                self._event(
                    "fix.proposals",
                    {"proposals": [_proposal_payload(item) for item in proposals]},
                    run_id,
                    now,
                )
            )
            if conflicts_enabled():
                events.append(
                    self._event(
                        "fix.conflicts",
                        {"conflicts": [asdict(item) for item in conflicts]},
                        run_id,
                        now,
                    )
                )
            events.append(
                self._event(
                    "fix.plan",
                    {
                        "plan_id": plan.plan_id,
                        "actions": [action.model_dump() for action in plan.actions],
                        "total_cost": plan.total_cost,
                        "expected_effect": plan.expected_effect,
                        "notes": plan.notes,
                        "fragility": fragility_by_zone,
                        "rephase_shifts": rephase.shifts,
                    },
                    run_id,
                    now,
                )
            )
            mean_service = (
                sum(service_by_zone.values()) / len(service_by_zone) if service_by_zone else 1.0
            )
            events.append(
                self._event(
                    "metrics.resilience",
                    {
                        "service_level": mean_service,
                        "per_zone": service_by_zone,
                        "resilience": {
                            zone: resilience_curve(np.asarray(history, dtype=float)).tolist()
                            for zone, history in self._service_history.items()
                        },
                        "fragility": fragility_by_zone,
                    },
                    run_id,
                    now,
                )
            )
            self._start_briefing(
                {
                    "zones": [
                        {
                            "zone": snapshot.zone,
                            "status": snapshot.status,
                            "fragility": fragility_by_zone.get(snapshot.zone, 0.0),
                            "service_level": service_by_zone.get(snapshot.zone, 1.0),
                        }
                        for snapshot in snapshots
                    ],
                    "proposals": proposals,
                    "plan_actions": plan.actions,
                },
                run_id,
                now,
            )
            if self.auto_mode and sim is not None:
                for action in plan.actions:
                    sim.apply_action(action)
            if proposals and plan.actions and sim is not None:
                self._start_counterfactual(sim, plan, now)
            self._poll_briefing(events, run_id, now)
            return events
        except Exception as error:
            logger.exception("FixModule on_tick failed")
            try:
                tick_map = _mapping(tick)
                return [
                    self._error_event(
                        str(error),
                        str(tick_map.get("run_id", self._latest_run_id)),
                        max(0, int(tick_map.get("t", self._latest_tick) or 0)),
                    )
                ]
            except Exception:  # noqa: BLE001 - uphold the module's never-raise callback contract.
                return [self._error_event("Fix module tick failed", self._latest_run_id, self._latest_tick)]

    def on_command(self, cmd: Any, ctx: Any) -> list[Envelope]:
        try:
            outer = _mapping(cmd)
            payload = _mapping(outer.get("payload", outer))
            channel = (
                cmd
                if isinstance(cmd, str)
                else str(outer.get("channel", outer.get("command", outer.get("type", ""))))
            )
            run_id = str(outer.get("run_id", self._latest_run_id))
            now = max(0, int(outer.get("t", self._latest_tick) or 0))
            if channel == "fix.auto_mode":
                enabled = payload.get("enabled", payload.get("value", False))
                self.auto_mode = enabled is True or str(enabled).lower() in ("1", "true", "on")
                return [
                    self._event(
                        "fix.plan",
                        {"command": "fix.auto_mode", "enabled": self.auto_mode},
                        run_id,
                        now,
                    )
                ]
            if channel == "fix.apply":
                if self._current_plan is None:
                    return [self._error_event("No plan is available to apply", run_id, now)]
                requested = payload.get("action_ids")
                if requested is None and payload.get("action_id") is not None:
                    requested = [payload["action_id"]]
                if not isinstance(requested, list):
                    requested = [action.id for action in self._current_plan.actions]
                actions = [
                    action for action in self._current_plan.actions if action.id in requested
                ]
                sim = _context_value(ctx, "sim") or _context_value(ctx, "simulator")
                if sim is None or not callable(getattr(sim, "apply_action", None)):
                    return [self._error_event("Simulator action interface is unavailable", run_id, now)]
                for action in actions:
                    sim.apply_action(action)
                return [
                    self._event(
                        "fix.plan",
                        {
                            "command": "fix.apply",
                            "applied": [action.id for action in actions],
                            "plan_id": self._current_plan.plan_id,
                        },
                        run_id,
                        now,
                    )
                ]
            return []
        except Exception as error:
            logger.exception("FixModule on_command failed")
            return [self._error_event(str(error), self._latest_run_id, self._latest_tick)]

    def _service_by_zone(self, tick: Tick) -> dict[str, float]:
        reported = tick.reports.get("service_level")
        if isinstance(reported, Mapping):
            values = {str(zone): float(value) for zone, value in reported.items()}
        else:
            grouped: dict[str, dict[str, float]] = defaultdict(dict)
            for reading in tick.readings:
                grouped[reading.zone][reading.system] = reading.value
            values = {zone: service_level(zone_loads) for zone, zone_loads in grouped.items()}
        for zone, value in values.items():
            self._service_history[zone].append(float(np.clip(value, 0.0, 1.0)))
        return values

    def _rephase(self, ctx: _ContextView, snapshots: Sequence[ZoneSnapshot], now: int) -> Any:
        profiles: dict[str, np.ndarray] = {}
        for snapshot in snapshots:
            for system in ("traffic", "power", "water"):
                history = ctx.history(snapshot.zone, system, 30)
                if not history:
                    history = [snapshot.loads[system]]
                profiles[f"{snapshot.zone}.{system}"] = forecast_from_history(history, 30)
        return rephase_optimizer(profiles, budget_ms=20, max_actions=3, now=now)

    def _start_briefing(self, state: dict[str, Any], run_id: str, now: int) -> None:
        if self._brief_thread is not None and self._brief_thread.is_alive():
            return
        with self._brief_lock:
            self._brief_result = None
            self._brief_error = None

        def worker() -> None:
            try:
                result = brief(state)
            except Exception as error:  # noqa: BLE001 - background failures are surfaced as events.
                with self._brief_lock:
                    self._brief_error = type(error).__name__
            else:
                with self._brief_lock:
                    self._brief_result = {**result, "run_id": run_id, "t": now}

        self._brief_thread = threading.Thread(target=worker, daemon=True)
        self._brief_thread.start()

    def _poll_briefing(self, events: list[Envelope], run_id: str, now: int) -> None:
        thread = self._brief_thread
        if thread is None or thread.is_alive():
            return
        with self._brief_lock:
            result, error = self._brief_result, self._brief_error
            self._brief_result = None
            self._brief_error = None
        self._brief_thread = None
        if error:
            events.append(self._error_event(f"Briefing failed: {error}", run_id, now))
        elif result:
            events.append(self._event("fix.briefing", result, run_id, now))

    def _start_counterfactual(self, sim: Any, plan: Plan, now: int) -> None:
        if self._counterfactual is None:
            self._counterfactual = CounterfactualCoordinator(
                sim, horizon_min=180, checkpoint_every=1, max_checkpoints=36
            )
        if plan.plan_id == self._counterfactual_plan_id:
            return
        if self._counterfactual_phase is not None:
            return
        self._counterfactual_plan_id = plan.plan_id
        if self._counterfactual.run_async(plan, now, now):
            self._counterfactual_phase = "counterfactual"

    def _poll_background(self, run_id: str, now: int) -> list[Envelope]:
        events: list[Envelope] = []
        coordinator = self._counterfactual
        if coordinator is None or self._counterfactual_phase is None:
            return events
        result = coordinator.poll()
        if result is None:
            if not coordinator.is_running():
                events.append(self._error_event("Counterfactual background job failed", run_id, now))
                self._counterfactual_phase = None
            return events
        if self._counterfactual_phase == "counterfactual" and isinstance(result, dict):
            events.extend(coordinator.to_envelopes(result, run_id, now))
            plan = self._current_plan
            if plan is None:
                self._counterfactual_phase = None
                return events

            def at_checkpoint(checkpoint_t: int) -> Plan:
                shifted = [action.model_copy(update={"start_t": checkpoint_t}) for action in plan.actions]
                return Plan(plan.plan_id, shifted, plan.total_cost, plan.expected_effect, plan.notes)

            if coordinator.run_intervention_window_async(at_checkpoint, max(0, now - 36)):
                self._counterfactual_phase = "intervention_window"
            else:
                self._counterfactual_phase = None
        elif self._counterfactual_phase == "intervention_window" and isinstance(
            result, InterventionWindow
        ):
            events.extend(coordinator.to_envelopes(result, run_id, now))
            self._counterfactual_phase = None
        else:
            events.append(self._error_event("Unexpected counterfactual result type", run_id, now))
            self._counterfactual_phase = None
        return events

    @staticmethod
    def _event(channel: str, payload: dict[str, Any], run_id: str, t: int) -> Envelope:
        return make_envelope(channel, payload, run_id, max(0, int(t)))

    def _error_event(self, message: str, run_id: str, t: int) -> Envelope:
        return self._event("system.error", {"module": self.name, "message": message}, run_id, t)
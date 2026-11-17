"""R3 CSD-only Sync module for sim.tick envelopes."""

import json
from collections import deque
from pathlib import Path

import numpy as np

from backend.sync.csd import csd_score, window_features, zone_scores
from backend.sync.sync_index import (
    load_bands,
    load_csd_floor,
    load_warn_persist,
    load_weights,
    soften,
    status,
    sync_index,
)
from backend.sync.threshold_alarm import ThresholdAlarm

BASELINE_PATH = Path(__file__).with_name("baseline_v1.json")
ZONES_PER_TICK = 8
READINGS_PER_TICK = 24
WINDOW = 120


def _event(channel, zone, payload):
    return {"channel": channel, "zone": zone, "payload": payload}


class SyncModule:
    name = "sync"

    def __init__(self, baseline=None, profiles=None):
        if baseline is None or profiles is None:
            with open(BASELINE_PATH, encoding="utf-8") as baseline_file:
                saved = json.load(baseline_file)
            if baseline is None:
                baseline = saved["csd"]
            if profiles is None:
                profiles = saved["profiles"]

        self.csd_baseline = baseline.get("csd", baseline)
        self.profiles = {
            day_type: np.asarray(profile, dtype=float)
            for day_type, profile in profiles.items()
        }
        self.weights = load_weights()
        self.csd_floor = load_csd_floor()
        self.warn_persist_min = load_warn_persist()
        self.bands = load_bands()
        self.threshold_alarm = ThresholdAlarm()
        self.last_run_id = None
        self._reset_state(None)

    def _reset_state(self, run_id):
        self.last_run_id = run_id
        self.residual_rows = deque(maxlen=WINDOW)
        self.cached_zone_scores = {}
        self.first_warning_tick = {}
        self.warning_persist_counts = {}
        self.last_status = {}
        self.threshold_alarm = ThresholdAlarm()

    def on_tick(self, tick, ctx):
        try:
            payload = tick.get("payload", tick)
            readings = payload.get("readings", [])
            run_id = tick.get("run_id")
            if run_id != self.last_run_id:
                self._reset_state(run_id)
            if not readings:
                return []
            if len(readings) != READINGS_PER_TICK:
                return []

            t = int(tick["t"])
            context = payload.get("context", {}) or {}
            day_type = context.get("day_type", "weekday")
            profile = self.profiles.get(day_type)
            if profile is None:
                profile = self.profiles.get("weekend", self.profiles.get("weekday"))
            residual = np.asarray(
                [reading["value"] for reading in readings], dtype=float
            ) - profile[t % 1440]
            self.residual_rows.append(residual)

            zones = list(dict.fromkeys(reading["zone"] for reading in readings))
            if len(zones) != ZONES_PER_TICK:
                return []

            if len(self.residual_rows) == WINDOW and t % 5 == 0:
                variance, ac1 = window_features(np.asarray(self.residual_rows))
                node_scores = csd_score(variance, ac1, self.csd_baseline)
                self.cached_zone_scores = dict(zip(zones, zone_scores(node_scores)))

            events = []
            current_status = {}
            current_indices = {}
            for zone in zones:
                score = float(self.cached_zone_scores.get(zone, 0.0))
                index = float(sync_index(
                    csd=soften(score, self.csd_floor), weights=self.weights
                ))
                zone_status = status(index, self.bands)
                current_status[zone] = zone_status
                current_indices[zone] = index
                events.append(_event(
                    "sync.index",
                    zone,
                    {
                        "zone": zone,
                        "sync_index": index,
                        "status": zone_status,
                        "phase_coherence": None,
                        "csd_score": score,
                        "coupling_break": None,
                        "phases": None,
                    },
                ))

                previous = self.last_status.get(zone)
                if zone_status == "healthy":
                    self.warning_persist_counts[zone] = 0
                    self.first_warning_tick.pop(zone, None)
                else:
                    count = self.warning_persist_counts.get(zone, 0) + 1
                    self.warning_persist_counts[zone] = count
                    initial_warning_due = (
                        zone not in self.first_warning_tick
                        and count >= self.warn_persist_min
                    )
                    critical_escalation = (
                        zone in self.first_warning_tick
                        and previous == "drifting"
                        and zone_status == "critical"
                    )
                    if initial_warning_due:
                        self.first_warning_tick[zone] = t
                    if initial_warning_due or critical_escalation:
                        events.append(_event(
                            "sync.warning",
                            zone,
                            {
                                "zone": zone,
                                "level": zone_status,
                                "sync_index": index,
                                "trigger": "csd",
                                "baseline_fired": False,
                                "lead_min": None,
                            },
                        ))
                self.last_status[zone] = zone_status

            for alarm in self.threshold_alarm.update(readings):
                zone = alarm["zone"]
                events.append(_event("baseline.alarm", zone, alarm))
                first_warning = self.first_warning_tick.get(zone)
                events.append(_event(
                    "sync.warning",
                    zone,
                    {
                        "zone": zone,
                        "level": current_status.get(zone, "critical"),
                        "sync_index": current_indices.get(zone, 0.0),
                        "trigger": "csd",
                        "baseline_fired": True,
                        "lead_min": None if first_warning is None else t - first_warning,
                    },
                ))

            return events
        except Exception:  # noqa: BLE001 -- the hub must never crash
            return []

    def on_command(self, cmd, ctx):
        return []

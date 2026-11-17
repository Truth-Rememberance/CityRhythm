"""Deterministic vectorized simulator core for CityRhythm R2.

Model (one tick = one sim minute):
  target(z, s, h)  = daily curve + profile modulation (+ hidden couplings)
  x_{t+1}          = x_t + alpha * (target - x_t) + sigma_s * e_t,
  where e_t is AR(1): e_t = phi * e_{t-1} + w_t, w ~ N(0, 1) per node.
  reported raw     = x + sensor noise; value = clip(raw, 0, 1.3).

Hidden coupling/curve parameters live only in _truth.py (firewall).
All randomness comes from three numpy Generator streams seeded per run
(process / sensor / events). State is deep-copyable via fork().
The hot loop is fully vectorized over the 24 nodes (numpy, no pandas).
"""

from __future__ import annotations

import datetime as dt
import json
import zoneinfo
from dataclasses import dataclass, field

import numpy as np

from backend.sim import _truth
from backend.sim import world as W
from backend.sim.contracts_stub import ScenarioSpec

_IST = zoneinfo.ZoneInfo("Asia/Kolkata")

SYSTEM_IDX: dict[str, int] = {s: i for i, s in enumerate(W.SYSTEMS)}


def _node_index(zi: int, system: str) -> int:
    """Flat index into the 24-vector: zone-major, then traffic/power/water."""
    return zi * 3 + SYSTEM_IDX[system]


@dataclass
class SimState:
    """Deep-copyable simulator state (arrays are copied on fork)."""

    t: int = 0  # ticks since run start (negative during warm-up)
    x: np.ndarray = field(default_factory=lambda: np.zeros(24))  # utilization
    ar: np.ndarray = field(default_factory=lambda: np.zeros(24))  # AR(1) state
    temp_drift: float = 0.0
    gen_process_state: dict | None = None
    gen_sensor_state: dict | None = None
    gen_events_state: dict | None = None

    def snapshot(self) -> SimState:
        """Deep copy: fresh arrays + captured Generator states."""
        return SimState(
            t=self.t,
            x=self.x.copy(),
            ar=self.ar.copy(),
            temp_drift=self.temp_drift,
            gen_process_state=dict(self.gen_process_state or {}),
            gen_sensor_state=dict(self.gen_sensor_state or {}),
            gen_events_state=dict(self.gen_events_state or {}),
        )


class Engine:
    """Simulates the 24-node city; deterministic for a given ScenarioSpec."""

    def __init__(self, spec: ScenarioSpec, config_path: str | None = None) -> None:
        self.spec = spec
        self.world = W.load_world(config_path)
        self.zone_order: list[str] = self.world["zone_order"]
        self.zones_meta = self.world["zones"]
        self.start_dt = dt.datetime.combine(
            dt.date.fromisoformat(spec.start_date),
            dt.time.fromisoformat(spec.start_time),
            tzinfo=_IST,
        )
        self._g_proc = np.random.default_rng(
            np.random.SeedSequence(spec.seed + _truth.STREAM_OFFSETS["process"])
        )
        self._g_sens = np.random.default_rng(
            np.random.SeedSequence(spec.seed + _truth.STREAM_OFFSETS["sensor"])
        )
        self._g_evt = np.random.default_rng(
            np.random.SeedSequence(spec.seed + _truth.STREAM_OFFSETS["events"])
        )
        self.state = SimState()
        # Precompute per-node static vectors (curve role, profile, day curve).
        self._prepare_static()

    # --- static preparation -------------------------------------------------
    def _prepare_static(self) -> None:
        n = 24
        self._prof = np.zeros(n)
        self._pbase = np.zeros(n)
        self._amp = np.ones((n, 3))  # morning/evening/midday scale per node
        for zi, zid in enumerate(self.zone_order):
            prof = self.zones_meta[zid]["profile"]
            m = _truth.PROFILE_MOD[prof]
            for s_i, _s in enumerate(W.SYSTEMS):
                i = zi * 3 + s_i
                self._prof[i] = m["base"] if s_i == 0 else 0.0
                self._amp[i] = [m["morning"], m["evening"], m["midday"]]
                if s_i == 1:  # power base shift per profile
                    self._pbase[i] = m.get("power_base", 0.0)

    # --- calendar -----------------------------------------------------------
    def day_type(self, date: dt.date) -> str:
        """holiday > weekend > weekday, from data/calendar.json + spec.

        A calendar holiday applies to a run when the entry has no "runs"
        list, or when this run's id is listed in it (ERRATA items 1-2).
        """
        iso = date.isoformat()
        if iso in self.spec.holidays:
            return "holiday"
        for hol in self._calendar_holidays():
            if hol["date"] == iso and ("runs" not in hol or self.spec.run_id in hol["runs"]):
                return "holiday"
        if date.weekday() >= 5:
            return "weekend"
        return "weekday"

    @staticmethod
    def _calendar_holidays() -> list[dict]:
        path = W._ROOT / "data" / "calendar.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8")).get("holidays", [])

    def ts_for(self, t: int) -> str:
        """ISO 8601 IST timestamp for tick t (minutes since run start)."""
        return (self.start_dt + dt.timedelta(minutes=t)).isoformat()

    # --- curve ---------------------------------------------------------------
    def _curve_target(self, t: int) -> np.ndarray:
        """Curve + profile target for all 24 nodes at tick t (vectorized)."""
        h = ((t % 1440) + 0.5) / 60.0  # tick centre as fractional hour
        day = self.start_dt.date() + dt.timedelta(days=t // 1440)
        weekend = self.day_type(day) != "weekday"
        target = np.zeros(24)
        for s_i, s in enumerate(W.SYSTEMS):
            curve = _truth.CURVES[s]
            bumps = []
            for role, b in zip(("morning", "evening", "midday"), curve["bumps"]):
                amp, mu, sig = b["amp"], b["mu"], b["sigma"]
                if weekend:
                    ov = _truth.WEEKEND_OVERRIDE[s]
                    amp *= ov.get("amp_scale", 1.0)
                    if role == "morning" and ov.get("morning_amp_scale"):
                        amp *= ov["morning_amp_scale"]
                    if role == "morning":
                        mu += ov.get("mu_shift", {}).get("morning", 0.0)
                    if role == "evening":
                        mu += ov.get("evening_mu_shift", 0.0)
                bumps.append((role, amp, mu, sig))
            for zi in range(8):
                i = zi * 3 + s_i
                val = curve["base"] + self._prof[i] + self._pbase[i]
                for role, amp, mu, sig in bumps:
                    val += (
                        self._amp[i][("morning", "evening", "midday").index(role)]
                        * amp
                        * float(np.exp(-0.5 * ((h - mu) / sig) ** 2))
                    )
                target[i] = val
        return target

    def _temperature(self, t: int) -> float:
        """Synthetic diurnal temperature with slow drift (degC)."""
        tm = _truth.TEMP_MODEL
        h = ((t % 1440) + 0.5) / 60.0
        diurnal = tm["diurnal_amp"] * float(np.sin(2 * np.pi * (h - 9.0) / 24.0))
        return tm["mean"] + diurnal + self.state.temp_drift

    # --- stepping -------------------------------------------------------------
    def step(self) -> None:
        """Advance one tick: dynamics are O(24) vectorized numpy."""
        st = self.state
        target = self._curve_target(st.t)
        # Hidden couplings (firewall): traffic congestion -> power; heat -> water.
        t_idx = np.arange(8) * 3 + 0
        p_idx = np.arange(8) * 3 + 1
        w_idx = np.arange(8) * 3 + 2
        congested = np.maximum(st.x[t_idx] - _truth.COUPLING["traffic_to_power_threshold"], 0.0)
        target[p_idx] += _truth.COUPLING["traffic_to_power"] * congested
        temp = self._temperature(st.t)
        target[w_idx] += _truth.COUPLING["temp_to_water"] * (temp - 24.0)
        # AR(1) process noise: innovation scaled to sigma, then integrated
        # by the persistent AR state (lag buffers filled during warm-up).
        w = self._g_proc.standard_normal(24)
        sigma_proc = np.array([_truth.SIGMA_PROCESS[s] for s in W.SYSTEMS])
        sigma_vec = np.tile(sigma_proc, 8)  # zone-major: traffic,power,water x8
        self.state.ar = _truth.DYNAMICS["phi"] * self.state.ar + sigma_vec * w
        common = self._g_proc.standard_normal() * _truth.COUPLING["common_factor_sigma"]
        temp_drift_step = self._g_proc.standard_normal() * _truth.TEMP_MODEL["sigma"]
        st.temp_drift = 0.95 * st.temp_drift + temp_drift_step
        st.x = st.x + _truth.DYNAMICS["alpha"] * (target - st.x)
        for s_i, s in enumerate(W.SYSTEMS):
            idx = np.arange(8) * 3 + s_i
            st.x[idx] += self.state.ar[idx] + common
        st.t += 1

    def readings(self) -> list[dict]:
        """24 sensor readings for the current tick, in emission order."""
        st = self.state
        raw = st.x + self._g_sens.standard_normal(24) * _truth.SIGMA_SENSOR
        val = np.clip(raw, 0.0, _truth.VALUE_MAX)
        out = []
        for zi, zid in enumerate(self.zone_order):
            for s_i, s in enumerate(W.SYSTEMS):
                i = zi * 3 + s_i
                out.append(
                    {
                        "zone": zid,
                        "system": s,
                        "metric": W.METRIC[s],
                        "value": round(float(val[i]), _truth.ROUND_VALUE),
                        "raw": round(float(raw[i]), _truth.ROUND_RAW),
                        "unit": W.UNIT[s],
                        "sensor_id": f"{zid}.{s}.01",
                    }
                )
        return out

    def reports(self) -> list[dict]:
        """0-2 synthetic citizen reports for the current tick (event stream)."""
        if self.state.t < 0 or self._g_evt.random() >= _truth.REPORT_P:
            return []
        n = 2 if self._g_evt.random() < _truth.REPORT_P_EXTRA else 1
        out = []
        for k in range(n):
            zi = int(self._g_evt.integers(0, 8))
            s_i = int(self._g_evt.integers(0, 3))
            lang = "kn" if self._g_evt.random() < 0.3 else "en"
            zid = self.zone_order[zi]
            s = W.SYSTEMS[s_i]
            text = _truth.REPORT_TEMPLATES[s][lang].format(zone=zid.replace("_", " "))
            out.append(
                {
                    "id": f"{self.spec.run_id}-r{self.state.t:06d}-{k}",
                    "zone": zid,
                    "text": text,
                    "lang": lang,
                    "source": "synthetic",
                }
            )
        return out

    def context(self) -> dict:
        """Per-tick context: temperature, events, holiday flag, day type."""
        day = self.start_dt.date() + dt.timedelta(days=self.state.t // 1440)
        dtype = self.day_type(day)
        return {
            "temp_c": round(self._temperature(self.state.t), 2),
            "events": [],
            "is_holiday": dtype == "holiday",
            "day_type": dtype,
        }

    # --- warm-up & fork -------------------------------------------------------
    def warm_up(self) -> None:
        """Simulate warmup_days*1440 ticks (starting at t=-warmup) and discard.

        Fills the AR buffers and lag state so emitted t=0 has no startup
        transient (ERRATA item 5). Never emits anything.
        """
        warm = self.spec.warmup_days * 1440
        self.state.t = -warm
        # Start from the curve at the first warm-up tick (no artificial zero).
        self.state.x = self._curve_target(-warm).copy()
        for _ in range(warm):
            self.step()
        assert self.state.t == 0, "warm-up must land exactly on t=0"

    def fork(self) -> Engine:
        """Deep-copyable fork: identical state, identical future trajectory."""
        clone = Engine.__new__(Engine)
        clone.spec = self.spec.model_copy(deep=True)
        clone.world = self.world
        clone.zone_order = self.zone_order
        clone.zones_meta = self.zones_meta
        clone.start_dt = self.start_dt
        clone._g_proc = self._revive(self._g_proc)
        clone._g_sens = self._revive(self._g_sens)
        clone._g_evt = self._revive(self._g_evt)
        clone._prepare_static()
        clone.state = self.state.snapshot()
        return clone

    @staticmethod
    def _revive(gen: np.random.Generator) -> np.random.Generator:
        new = np.random.default_rng()
        new.bit_generator.state = gen.bit_generator.state
        return new

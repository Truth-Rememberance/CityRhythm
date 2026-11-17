"""Local pydantic stubs mirroring the R1 contract schemas.

TODO(replace with contracts): contracts/ currently has no models (only a
README). If R1 lands models in contracts/, import them here instead and
delete this file. Shape must stay compatible with docs/R2_SPEC.md:
envelope {v, channel, run_id, t, ts, zone, payload{readings, context, reports}}.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictFloat, StrictInt, StrictStr

Channel = Literal["sim.tick"]
DayType = Literal["weekday", "weekend", "holiday"]


class Reading(BaseModel):
    """One sensor reading: reported utilization = load / capacity (0..1.3)."""

    zone: StrictStr
    system: Literal["traffic", "power", "water"]
    metric: Literal["flow", "load", "demand"]
    value: StrictFloat = Field(ge=0.0, le=1.3)  # rounded to 4 decimals
    raw: StrictFloat  # noisy pre-clip value, rounded to 3 decimals
    unit: Literal["veh_per_min", "MW", "kL_per_min"]
    sensor_id: StrictStr  # "<zone>.<system>.01"


class Context(BaseModel):
    """Per-tick synthetic context. `day_type` added by ERRATA item 4."""

    temp_c: StrictFloat
    events: list[StrictStr] = Field(default_factory=list)
    is_holiday: bool
    day_type: DayType


class Report(BaseModel):
    """A synthetic citizen report; always source='synthetic'."""

    id: StrictStr
    zone: StrictStr
    text: StrictStr
    lang: Literal["en", "kn"]
    source: Literal["synthetic"]


class Payload(BaseModel):
    readings: list[Reading]
    context: Context
    reports: list[Report]


class Envelope(BaseModel):
    """Wire envelope for channel sim.tick (one per simulation step)."""

    v: Literal[1]
    channel: Channel
    run_id: StrictStr
    t: StrictInt  # minutes since run start
    ts: StrictStr  # ISO 8601 with IST offset +05:30
    zone: None
    payload: Payload


class EventSpec(BaseModel):
    """A single scenario event (extended in later R2 milestones)."""

    kind: StrictStr
    at_t: StrictInt  # tick at which the event fires
    zone: StrictStr | None = None
    magnitude: StrictFloat = 1.0


class ScenarioSpec(BaseModel):
    """Scenario definition. Dates are IST; start_date defaults to Thu 2026-11-19."""

    run_id: StrictStr
    seed: StrictInt
    start_date: StrictStr = "2026-11-19"  # YYYY-MM-DD, a Thursday
    start_time: StrictStr = "00:00"  # HH:MM IST
    days: StrictInt = 4
    warmup_days: StrictInt = 1  # simulated then discarded (ERRATA 5)
    holidays: list[StrictStr] = Field(default_factory=list)  # synthetic dates
    events: list[EventSpec] = Field(default_factory=list)

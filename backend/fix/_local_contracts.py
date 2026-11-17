"""TEMP: replace with imports from contracts/ once R1 freezes them"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ZONES = (
    "silk_board",
    "whitefield",
    "hebbal",
    "electronic_city",
    "koramangala",
    "indiranagar",
    "jayanagar",
    "yelahanka",
)
SYSTEMS = ("traffic", "power", "water")
ACTION_TYPES = ("shift_pump", "offset_signal", "shed_load", "reroute", "shift_schedule")
_IST_START = datetime.fromisoformat("2026-11-17T00:00:00+05:30")


class Envelope(BaseModel):
    """Versioned module message envelope; t is simulation minutes."""

    model_config = ConfigDict(extra="forbid")

    v: Literal[1] = 1
    channel: str
    run_id: str
    t: int = Field(ge=0)
    ts: str
    zone: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """A department intervention scheduled in simulation minutes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal[
        "shift_pump", "offset_signal", "shed_load", "reroute", "shift_schedule"
    ]
    dept: Literal["traffic", "power", "water"]
    zone: str
    params: dict[str, Any] = Field(default_factory=dict)
    start_t: int = Field(ge=0)
    duration_min: int = Field(gt=0)

    @field_validator("zone")
    @classmethod
    def zone_must_be_frozen_zone(cls, value: str) -> str:
        if value not in ZONES:
            raise ValueError(f"zone must be one of {ZONES}")
        return value


class Reading(BaseModel):
    """A normalized or unit-bearing sensor reading at a simulation time."""

    zone: str
    system: Literal["traffic", "power", "water"]
    sensor_id: str
    value: float
    unit: str = "normalized"
    t: int | None = Field(default=None, ge=0)
    ts: str | None = None

    @field_validator("zone")
    @classmethod
    def reading_zone_must_be_frozen_zone(cls, value: str) -> str:
        if value not in ZONES:
            raise ValueError(f"zone must be one of {ZONES}")
        return value


class TickContext(BaseModel):
    """Environmental context supplied with a tick."""

    temp_c: float | None = None
    events: list[str] = Field(default_factory=list)
    is_holiday: bool = False


class Tick(BaseModel):
    """Simulator snapshot; t and all durations are in simulation minutes."""

    t: int = Field(ge=0)
    ts: str
    readings: list[Reading] = Field(default_factory=list)
    context: TickContext = Field(default_factory=TickContext)
    reports: dict[str, Any] = Field(default_factory=dict)


class SyncIndex(BaseModel):
    """Sync/early-warning index for one zone."""

    zone: str
    sync_index: float = Field(ge=0, le=100)
    status: Literal["healthy", "drifting", "critical"]
    phase_coherence: float = Field(ge=0, le=1)
    csd_score: float
    coupling_break: bool
    phases: dict[Literal["traffic", "power", "water"], float] = Field(default_factory=dict)

    @field_validator("zone")
    @classmethod
    def sync_zone_must_be_frozen_zone(cls, value: str) -> str:
        if value not in ZONES:
            raise ValueError(f"zone must be one of {ZONES}")
        return value


class Edge(BaseModel):
    """Directed dependency edge; lag_min is measured in simulation minutes."""

    src: str
    dst: str
    lag_min: float = Field(ge=0)
    strength: float = Field(ge=0, le=1)


class SyncGraph(BaseModel):
    """Cross-system dependency graph."""

    edges: list[Edge] = Field(default_factory=list)


class Evidence(BaseModel):
    kind: str
    detail: str
    score: float


class ThreatAssessment(BaseModel):
    """Threat signal for a zone, without access to simulator hidden truth."""

    zone: str
    label: Literal["natural", "attack", "unknown"]
    confidence: float = Field(ge=0, le=1)
    entry_node: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("zone")
    @classmethod
    def threat_zone_must_be_frozen_zone(cls, value: str) -> str:
        if value not in ZONES:
            raise ValueError(f"zone must be one of {ZONES}")
        return value


class CounterfactualMetrics(BaseModel):
    """Time-aligned metrics from one counterfactual simulation."""

    resilience: list[float] = Field(default_factory=list)
    service_level: list[float] = Field(default_factory=list)


class RecoveryMinutes(BaseModel):
    no_action: float = Field(ge=0)
    with_plan: float = Field(ge=0)


class CounterfactualPayload(BaseModel):
    """Time-aligned comparison; all t and recovery values are in minutes."""

    t: list[int] = Field(default_factory=list)
    no_action: CounterfactualMetrics
    with_plan: CounterfactualMetrics
    recovery_min: RecoveryMinutes


def make_envelope(
    channel: str,
    payload: dict[str, Any],
    run_id: str,
    t: int,
    zone: str | None = None,
) -> Envelope:
    """Build an envelope with IST timestamp at t minutes after the fixed demo start."""
    timestamp = _IST_START + timedelta(minutes=t)
    return Envelope(
        channel=channel,
        payload=payload,
        run_id=run_id,
        t=t,
        ts=timestamp.isoformat(timespec="seconds"),
        zone=zone,
    )
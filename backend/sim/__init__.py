"""CityRhythm R2: synthetic Bengaluru simulator (Simulation & Scenarios).

Public surface: Engine, ScenarioSpec, run_scenario, make_envelope, world.
Hidden ground truth stays in backend.sim._truth (never import it from
outside backend/sim).
"""

from backend.sim.contracts_stub import (
    Context,
    Envelope,
    EventSpec,
    Payload,
    Reading,
    Report,
    ScenarioSpec,
)
from backend.sim.engine import Engine, SimState
from backend.sim.serialize import make_envelope, run_scenario
from backend.sim.world import SYSTEMS, load_world, node_ids

__all__ = [
    "SYSTEMS",
    "Context",
    "Engine",
    "Envelope",
    "EventSpec",
    "Payload",
    "Reading",
    "Report",
    "ScenarioSpec",
    "SimState",
    "load_world",
    "make_envelope",
    "node_ids",
    "run_scenario",
]

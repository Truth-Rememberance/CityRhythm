"""Serialization of sim state to sim.tick envelopes (validated by the stub)."""

from __future__ import annotations

from backend.sim.contracts_stub import Envelope, Payload, ScenarioSpec
from backend.sim.engine import Engine


def make_envelope(engine: Engine) -> dict:
    """Build the sim.tick envelope dict for the engine's current tick.

    One envelope per step: {v, channel, run_id, t, ts, zone: null, payload}.
    """
    t = engine.state.t
    assert t >= 0, "warm-up ticks must be discarded, not emitted"
    env = Envelope(
        v=1,
        channel="sim.tick",
        run_id=engine.spec.run_id,
        t=t,
        ts=engine.ts_for(t),
        zone=None,
        payload=Payload(
            readings=engine.readings(),  # type: ignore[arg-type]
            context=engine.context(),  # type: ignore[arg-type]
            reports=engine.reports(),  # type: ignore[arg-type]
        ),
    )
    return env.model_dump(mode="json")


def run_scenario(spec: ScenarioSpec, config_path: str | None = None) -> list[dict]:
    """Run a full scenario (warm-up discarded) and return all envelopes.

    Deterministic: same spec -> byte-identical JSONL when re-serialized.
    """
    eng = Engine(spec, config_path=config_path)
    eng.warm_up()
    total = spec.days * 1440
    out: list[dict] = []
    for _ in range(total):
        out.append(make_envelope(eng))
        eng.step()
    return out

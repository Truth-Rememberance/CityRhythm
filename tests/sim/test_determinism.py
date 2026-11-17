"""Determinism tests: seeding, fork() deep-copy, stream separation."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from backend.sim.contracts_stub import ScenarioSpec
from backend.sim.engine import Engine
from backend.sim.serialize import make_envelope

SPEC = ScenarioSpec(run_id="det", seed=777, days=2)


def collect(eng: Engine, ticks: int) -> list[dict]:
    out = [make_envelope(eng) for _ in range(ticks)]
    return out


def test_same_seed_identical_bytes() -> None:
    a = collect(Engine(SPEC), 600)
    b = collect(Engine(SPEC), 600)
    assert a == b


def test_fork_continues_identically() -> None:
    eng = Engine(SPEC)
    eng.warm_up()
    for _ in range(500):
        eng.step()
    base = collect(eng.fork(), 400)
    eng2 = copy.deepcopy(eng)  # dataclass state must be deep-copyable too
    other = collect(eng2, 400)
    assert base == other


def test_fork_is_deep_not_shared() -> None:
    eng = Engine(SPEC)
    eng.warm_up()
    twin = eng.fork()
    x_at_fork = eng.state.x.copy()
    assert twin.state.x is not eng.state.x  # separate buffers, not views
    for _ in range(10):
        eng.step()
    assert np.array_equal(twin.state.x, x_at_fork)  # twin untouched by parent
    assert not np.array_equal(eng.state.x, x_at_fork)  # parent moved on
    assert twin.state.t == eng.state.t - 10


def test_streams_are_independent() -> None:
    """Advancing the sensor stream must not perturb the process trajectory."""
    eng_a = Engine(SPEC)
    eng_a.warm_up()
    eng_b = eng_a.fork()
    eng_a.readings()  # consumes ONLY the sensor stream
    xs_a, xs_b = [], []
    for _ in range(200):
        eng_a.step()
        eng_b.step()
        xs_a.append(eng_a.state.x.copy())
        xs_b.append(eng_b.state.x.copy())
    # Latent dynamics identical: process stream untouched by readings().
    assert np.allclose(np.array(xs_a), np.array(xs_b))
    # But emitted raws differ: sensor stream has advanced in eng_a only.
    raw_a = [r["raw"] for r in eng_a.readings()]
    raw_b = [r["raw"] for r in eng_b.readings()]
    assert raw_a != raw_b


def test_different_seeds_diverge() -> None:
    a = collect(Engine(ScenarioSpec(run_id="s1", seed=1001, days=2)), 300)
    b = collect(Engine(ScenarioSpec(run_id="s2", seed=1002, days=2)), 300)
    assert a != b


def test_values_in_range_after_long_run() -> None:
    eng = Engine(SPEC)
    eng.warm_up()
    for _ in range(3000):
        eng.step()
        for r in eng.readings():
            assert 0.0 <= r["value"] <= 1.3


@pytest.mark.parametrize("seed", [1001, 1005])
def test_warmup_lands_on_t0(seed: int) -> None:
    eng = Engine(ScenarioSpec(run_id="w", seed=seed))
    eng.warm_up()
    assert eng.state.t == 0
    assert eng.readings()[0]["value"] > 0.0  # no startup transient

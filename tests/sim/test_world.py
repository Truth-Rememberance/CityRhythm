"""World/config tests: zones, order, nodes, adjacency, capacities."""

from __future__ import annotations

import pytest

from backend.sim.world import SYSTEMS, capacities, load_world, node_ids

EXPECTED_ORDER = [
    "silk_board",
    "whitefield",
    "hebbal",
    "electronic_city",
    "koramangala",
    "indiranagar",
    "jayanagar",
    "yelahanka",
]

# Adjacency exactly as specified (only the five listed sources have entries).
EXPECTED_ADJACENCY = {
    "silk_board": {"koramangala", "electronic_city", "jayanagar"},
    "koramangala": {"silk_board", "indiranagar", "jayanagar"},
    "indiranagar": {"koramangala", "whitefield", "hebbal"},
    "hebbal": {"indiranagar", "yelahanka"},
    "jayanagar": {"silk_board", "koramangala"},
}

PROFILES = {
    "whitefield": "it_corridor",
    "electronic_city": "it_corridor",
    "silk_board": "junction",
    "koramangala": "commercial",
    "indiranagar": "commercial",
    "jayanagar": "residential",
    "yelahanka": "residential",
    "hebbal": "mixed",
}


def test_zone_order_is_exact() -> None:
    world = load_world()
    assert world["zone_order"] == EXPECTED_ORDER
    assert list(world["zones"]) == EXPECTED_ORDER


def test_24_nodes_in_order() -> None:
    nodes = node_ids(load_world())
    expected = [f"{z}.{s}" for z in EXPECTED_ORDER for s in ("traffic", "power", "water")]
    assert nodes == expected
    assert len(nodes) == 24


def test_adjacency_matches_spec() -> None:
    world = load_world()
    adj = {z: set(v) for z, v in world["adjacency"].items()}
    assert adj == EXPECTED_ADJACENCY
    # Zones not listed as sources have no adjacency entry in the spec.
    assert not ({"whitefield", "electronic_city", "yelahanka"} & set(adj))


def test_profiles_match_spec() -> None:
    world = load_world()
    assert {z: meta["profile"] for z, meta in world["zones"].items()} == PROFILES


def test_default_capacities() -> None:
    caps = capacities(load_world())
    assert caps["whitefield"]["power"] == pytest.approx(60 * 1.15)
    assert caps["silk_board"]["traffic"] == pytest.approx(200 * 1.2)
    assert caps["jayanagar"]["water"] == pytest.approx(30 * 1.05)
    assert caps["hebbal"]["traffic"] == 200.0
    assert set(next(iter(caps.values()))) == set(SYSTEMS)

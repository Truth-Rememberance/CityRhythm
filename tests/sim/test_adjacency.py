"""C.8: adjacency must be symmetric and every zone must have a neighbour.

The loader turns the config's edge list into an undirected graph: each edge
is present at both endpoints, so zones that only ever appear as neighbours
(whitefield, electronic_city, yelahanka) still list their own neighbours.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.sim.world import load_world

ZONE_ORDER = [
    "silk_board",
    "whitefield",
    "hebbal",
    "electronic_city",
    "koramangala",
    "indiranagar",
    "jayanagar",
    "yelahanka",
]


def test_adjacency_is_symmetric() -> None:
    adj = load_world()["adjacency"]
    for zone, neighbours in adj.items():
        for n in neighbours:
            assert zone in adj[n], f"edge {zone}->{n} has no reverse edge"


def test_all_zones_have_a_neighbour() -> None:
    adj = load_world()["adjacency"]
    assert set(adj) == set(ZONE_ORDER)
    for zone, neighbours in adj.items():
        assert neighbours, f"{zone} has no neighbours"


def test_reverse_edges_present() -> None:
    adj = {z: set(n) for z, n in load_world()["adjacency"].items()}
    assert adj["whitefield"] == {"indiranagar"}
    assert adj["electronic_city"] == {"silk_board"}
    assert adj["yelahanka"] == {"hebbal"}
    assert adj["indiranagar"] == {"koramangala", "whitefield", "hebbal"}


def test_no_self_loops_and_only_known_zones() -> None:
    adj = load_world()["adjacency"]
    for zone, neighbours in adj.items():
        assert zone not in neighbours
        assert set(neighbours) <= set(ZONE_ORDER)


def test_neighbour_lists_follow_zone_order() -> None:
    adj = load_world()["adjacency"]
    for neighbours in adj.values():
        assert neighbours == [z for z in ZONE_ORDER if z in neighbours]


def test_loader_symmetrises_one_directional_config(tmp_path: Path) -> None:
    """Even a config listing each edge once loads as an undirected graph."""
    raw = json.loads((Path("config/zones.json")).read_text(encoding="utf-8"))
    raw["adjacency"] = {
        "silk_board": ["koramangala", "electronic_city", "jayanagar"],
        "koramangala": ["silk_board", "indiranagar", "jayanagar"],
        "indiranagar": ["koramangala", "whitefield", "hebbal"],
        "hebbal": ["indiranagar", "yelahanka"],
        "jayanagar": ["silk_board", "koramangala"],
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    adj = load_world(str(path))["adjacency"]
    for zone, neighbours in adj.items():
        for n in neighbours:
            assert zone in adj[n]
    for zone in ZONE_ORDER:
        assert adj[zone]
    assert adj["whitefield"] == ["indiranagar"]


def test_loader_rejects_unknown_neighbour(tmp_path: Path) -> None:
    raw = json.loads((Path("config/zones.json")).read_text(encoding="utf-8"))
    raw["adjacency"]["whitefield"] = ["atlantis"]
    path = tmp_path / "bad_zones.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown neighbour"):
        load_world(str(path))

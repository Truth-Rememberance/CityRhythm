"""Synthetic Bengaluru world: zones, adjacency, nodes, capacities.

Loaded from config/zones.json (exact zone order per ERRATA item 7).
Node ids are "<zone>.<system>"; system order within a zone is
traffic, power, water. All data is synthetic.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SYSTEMS: tuple[str, ...] = ("traffic", "power", "water")
METRIC: dict[str, str] = {"traffic": "flow", "power": "load", "water": "demand"}
UNIT: dict[str, str] = {
    "traffic": "veh_per_min",
    "power": "MW",
    "water": "kL_per_min",
}

_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_world(config_path: str | None = None) -> dict:
    """Load and validate config/zones.json into a plain dict.

    Raises if the zone order or node set does not match the spec.
    """
    path = Path(config_path) if config_path else _ROOT / "config" / "zones.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    zones = raw["zones"]
    order = [z["id"] for z in zones]
    if order != raw["zone_order"]:
        raise ValueError(f"zone order mismatch in {path}: {order} != {raw['zone_order']}")
    if len(set(order)) != 8:
        raise ValueError("expected exactly 8 zones")
    nodes = [f"{z['id']}.{s}" for z in zones for s in SYSTEMS]
    if len(nodes) != 24:
        raise ValueError("expected 24 nodes (8 zones x 3 systems)")
    return {
        "zone_order": order,
        "zones": {z["id"]: z for z in zones},
        "adjacency": _symmetric_adjacency(raw["adjacency"], order),
        "default_capacity": raw["default_capacity"],
        "nodes": nodes,
    }


def _symmetric_adjacency(raw: dict[str, list[str]], order: list[str]) -> dict[str, list[str]]:
    """Return an undirected adjacency map: every edge is listed at both ends.

    Config may list an edge once (the spec's one-directional shorthand);
    the loader adds the reverse edge. Neighbour lists are emitted in zone
    order (ERRATA item 7) for deterministic diffing.
    """
    zones = set(order)
    undirected: dict[str, set[str]] = {z: set() for z in order}
    for z, neighbours in raw.items():
        if z not in zones:
            raise ValueError(f"adjacency references unknown zone {z!r}")
        for n in neighbours:
            if n not in zones:
                raise ValueError(f"zone {z!r} has unknown neighbour {n!r}")
            if n == z:
                raise ValueError(f"zone {z!r} cannot be adjacent to itself")
            undirected[z].add(n)
            undirected[n].add(z)
    empty = [z for z in order if not undirected[z]]
    if empty:
        raise ValueError(f"zones with no neighbours: {empty}")
    return {z: [n for n in order if n in undirected[z]] for z in order}


def node_ids(world: dict) -> list[str]:
    """24 node ids in emission order: zones in order, traffic/power/water."""
    return list(world["nodes"])


def capacities(world: dict) -> dict[str, dict[str, float]]:
    """Effective capacity per zone/system after per-zone multipliers."""
    out: dict[str, dict[str, float]] = {}
    for zid, z in world["zones"].items():
        out[zid] = {}
        for s in SYSTEMS:
            base = world["default_capacity"][s]["value"]
            out[zid][s] = base * z["capacity_multiplier"][s]
    return out

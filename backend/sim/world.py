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
        "adjacency": raw["adjacency"],
        "default_capacity": raw["default_capacity"],
        "nodes": nodes,
    }


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

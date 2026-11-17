"""R3: combine detector scores into one Sync Index per zone (0-100, higher = healthier)."""
import json
from pathlib import Path

import numpy as np

WEIGHTS_PATH = Path(__file__).with_name("weights.json")
SCORE_KEYS = ("csd", "coupling", "phase")


def load_weights(path=WEIGHTS_PATH):
    with open(path) as f:
        raw = json.load(f)
    return {k: float(raw[k]) for k in SCORE_KEYS}


def soften(score, floor):
    """Scores below the floor count as 0; the rest is stretched to 0-1."""
    s = np.asarray(score, dtype=float)
    return np.clip((s - floor) / (1.0 - floor), 0.0, 1.0)


def load_csd_floor(path=WEIGHTS_PATH):
    with open(path) as f:
        return float(json.load(f).get("csd_floor", 0.0))


def load_warn_persist(path=WEIGHTS_PATH):
    with open(path) as f:
        return int(json.load(f).get("warn_persist_min", 0))


def load_bands(path=None):
    """Read the status cut-offs from config/thresholds.json if it exists.

    Falls back to 70 / 40 (the frozen contract values) until R2 adds the file.
    """
    path = path or Path(__file__).resolve().parents[2] / "config" / "thresholds.json"
    try:
        with open(path) as f:
            raw = json.load(f)
        return float(raw["healthy_min"]), float(raw["drifting_min"])
    except (OSError, KeyError, ValueError, TypeError):
        return 70.0, 40.0


def sync_index(csd=None, coupling=None, phase=None, weights=None):
    """Each input is a 0-1 'badness' score per zone (0 = normal), or None if
    not available. Returns the index per zone, 0-100 (higher = healthier).

    Missing scores are left out and the remaining weights are rescaled.
    """
    weights = weights or load_weights()
    scores = {"csd": csd, "coupling": coupling, "phase": phase}
    used = {k: np.asarray(v, dtype=float) for k, v in scores.items() if v is not None}
    if not used:
        raise ValueError("sync_index needs at least one score")
    total = sum(weights[k] for k in used)
    badness = sum(weights[k] * s for k, s in used.items()) / total
    return 100.0 * (1.0 - np.clip(badness, 0.0, 1.0))


def status(index, bands=None):
    """'healthy' / 'drifting' / 'critical' for one index value."""
    healthy_min, drifting_min = bands or load_bands()
    if index >= healthy_min:
        return "healthy"
    if index >= drifting_min:
        return "drifting"
    return "critical"
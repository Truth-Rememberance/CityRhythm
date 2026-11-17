"""R3 sync math, step 1: load a run and remove the daily cycle (detrend).

Idea: every node has a normal "shape of the day". If we subtract it, what is
left (the residual) is the unusual part. All later checks (phase coherence,
critical slowing down, coupling breaks) work on the residual, not raw values.
"""
import json

import numpy as np

MINUTES_PER_DAY = 1440


def load_run(path):
    """Read a .jsonl of sim.tick envelopes -> (values[n, 24], node_ids[24], t[n])."""
    rows, ts, nodes = [], [], None
    with open(path) as f:
        for line in f:
            env = json.loads(line)
            readings = env["payload"]["readings"]
            if nodes is None:
                nodes = [f'{r["zone"]}.{r["system"]}' for r in readings]
            rows.append([r["value"] for r in readings])
            ts.append(env["t"])
    return np.array(rows, dtype=float), nodes, np.array(ts)


def build_profile(values, t, smooth_min=15):
    """Typical value for each minute of the day, per node -> shape (1440, n_nodes).

    TODO when R2's real calm runs arrive: build one profile per day type
    (weekday / weekend / holiday), because R2 says weekends differ.
    """
    minute = t % MINUTES_PER_DAY
    profile = np.zeros((MINUTES_PER_DAY, values.shape[1]))
    for m in range(MINUTES_PER_DAY):
        profile[m] = values[minute == m].mean(axis=0)
    # Smooth in a circle so 23:59 joins up with 00:00
    kernel = np.ones(smooth_min) / smooth_min
    padded = np.vstack([profile[-smooth_min:], profile, profile[:smooth_min]])
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), 0, padded)
    return smoothed[smooth_min:-smooth_min]


def detrend(values, t, profile):
    """Residual = reading minus the typical value for that minute of the day."""
    return values - profile[t % MINUTES_PER_DAY]

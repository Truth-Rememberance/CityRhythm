"""R3: load R2's real calm runs (gzipped) and build per-day-type profiles."""
import gzip
import json

import numpy as np

from backend.sync.sync_math import MINUTES_PER_DAY

DAY_TYPES = ["weekday", "weekend", "holiday"]


def load_run_gz(path):
    """Read a .jsonl.gz of sim.tick envelopes.

    Returns (values[n, 24], node_ids[24], t[n], day_types[n]).
    """
    rows, ts, days, nodes = [], [], [], None
    with gzip.open(path, "rt") as f:
        for line in f:
            env = json.loads(line)
            readings = env["payload"]["readings"]
            if nodes is None:
                nodes = [f'{r["zone"]}.{r["system"]}' for r in readings]
            rows.append([r["value"] for r in readings])
            ts.append(env["t"])
            days.append(env["payload"]["context"]["day_type"])
    return np.array(rows, dtype=float), nodes, np.array(ts), np.array(days)


def build_profiles(values, t, day_types, smooth_min=15):
    """One typical-day profile per day type -> {day_type: (1440, n_nodes)}.

    A day type with no data in this run is left out of the dict.
    """
    minute = t % MINUTES_PER_DAY
    kernel = np.ones(smooth_min) / smooth_min
    profiles = {}
    for dtype in DAY_TYPES:
        mask = day_types == dtype
        if not mask.any():
            continue
        prof = np.full((MINUTES_PER_DAY, values.shape[1]), np.nan)
        for m in range(MINUTES_PER_DAY):
            sel = mask & (minute == m)
            if sel.any():
                prof[m] = values[sel].mean(axis=0)
        # Fill any empty minutes by carrying neighbours, then smooth in a circle
        for col in range(prof.shape[1]):
            c = prof[:, col]
            ok = ~np.isnan(c)
            c[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), c[ok], period=MINUTES_PER_DAY)
        padded = np.vstack([prof[-smooth_min:], prof, prof[:smooth_min]])
        smoothed = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), 0, padded)
        profiles[dtype] = smoothed[smooth_min:-smooth_min]
    return profiles


def detrend_by_day(values, t, day_types, profiles):
    """Residual = reading minus the profile for that minute AND that day type.

    A day type with no profile falls back to the weekend profile if there is
    one, otherwise the weekday one.
    """
    minute = t % MINUTES_PER_DAY
    resid = np.zeros_like(values)
    for dtype in DAY_TYPES:
        mask = day_types == dtype
        if not mask.any():
            continue
        prof = profiles.get(dtype)
        if prof is None:
            prof = profiles["weekend"] if "weekend" in profiles else profiles["weekday"]
        resid[mask] = values[mask] - prof[minute[mask]]
    return resid
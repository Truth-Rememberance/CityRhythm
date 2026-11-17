"""R3 sync math, step 3: critical slowing down (CSD), the early-warning score.

Idea: a system close to its tipping point gets "sluggish". After a small
bump it takes longer to settle, so its leftover wobble (residual) gets
  - bigger  (variance goes up), and
  - stickier (lag-1 autocorrelation goes up: each minute looks like the last).
We compare both against what a calm city looks like (the baseline) and turn
the difference into a 0-1 score. 0 = normal, close to 1 = very sluggish.

Input is the RESIDUAL from detrend(), 24 columns (8 zones x traffic/power/water).
"""
import numpy as np

N_ZONES, N_SYSTEMS = 8, 3


def window_features(w):
    """w: (minutes, nodes) residuals. Returns (variance[nodes], ac1[nodes]).

    A straight-line trend inside the window is removed first, so a slow
    drift upward is not mistaken for sluggishness.
    """
    w = np.asarray(w, dtype=float)
    tt = np.arange(w.shape[0])
    slope, intercept = np.polyfit(tt, w, 1)
    x = w - (np.outer(tt, slope) + intercept)
    var = x.var(axis=0)
    ac1 = (x[:-1] * x[1:]).sum(axis=0) / np.maximum((x * x).sum(axis=0), 1e-12)
    return var, ac1


def rolling_features(resid, window=120, step=5):
    """Slide a window over a whole run (offline use: baselines and eval).
    Returns (end_ticks, variance[m, nodes], ac1[m, nodes])."""
    ends, vs, acs = [], [], []
    for end in range(window, resid.shape[0] + 1, step):
        v, a = window_features(resid[end - window:end])
        ends.append(end - 1)
        vs.append(v)
        acs.append(a)
    return np.array(ends), np.array(vs), np.array(acs)


def build_csd_baseline(resid_runs, window=120, step=5):
    """What 'calm' looks like. resid_runs = list of detrended calm runs.
    Returns a plain dict (lists), ready to go into data/baseline.json."""
    vs, acs = [], []
    for resid in resid_runs:
        _, v, a = rolling_features(resid, window, step)
        vs.append(v)
        acs.append(a)
    logv = np.log(np.maximum(np.vstack(vs), 1e-12))
    ac = np.vstack(acs)
    return {
        "window": window,
        "logvar_mean": logv.mean(axis=0).tolist(),
        "logvar_std": np.maximum(logv.std(axis=0), 0.05).tolist(),
        "ac1_mean": ac.mean(axis=0).tolist(),
        "ac1_std": np.maximum(ac.std(axis=0), 0.02).tolist(),
    }


def csd_score(var, ac1, base):
    """Per-node score 0-1 (0 = normal, near 1 = very sluggish)."""
    z_var = (np.log(np.maximum(var, 1e-12)) - np.array(base["logvar_mean"])) / np.array(base["logvar_std"])
    z_ac = (np.asarray(ac1) - np.array(base["ac1_mean"])) / np.array(base["ac1_std"])
    z = np.maximum(0.0, (z_var + z_ac) / 2)
    return 1.0 - np.exp(-z / 4.0)


def zone_scores(node_scores):
    """Collapse 24 node scores to 8 zone scores (worst system in each zone)."""
    s = np.asarray(node_scores)
    return s.reshape(*s.shape[:-1], N_ZONES, N_SYSTEMS).max(axis=-1)

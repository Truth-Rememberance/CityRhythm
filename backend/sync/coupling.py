"""R3 sync math, step 4: coupling break.

Idea: in a healthy zone some systems move together with a delay (for example
power load follows traffic about 12 minutes later). We measure how strongly
each pair is linked right now and compare it with the healthy baseline. If a
pair that is normally linked suddenly stops moving together, something is
wrong: a spoofed meter, a hijacked signal, a replayed feed.

Input is the RESIDUAL from detrend(), 24 columns (8 zones x traffic/power/water).
Pairs, in this order: traffic-power, traffic-water, power-water.
"""
import numpy as np

N_ZONES = 8
PAIRS = [(0, 1), (0, 2), (1, 2)]
ACTIVE_FLOOR = 0.35  # only judge pairs that are clearly linked when healthy


def pair_coupling(resid_window, max_lag=20):
    """resid_window: (minutes, 24). Returns (8, 3): for each zone and pair, the
    strongest absolute correlation found at any delay up to +/- max_lag minutes."""
    m = resid_window.shape[0]
    x = resid_window.reshape(m, N_ZONES, 3)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-9)
    out = np.zeros((N_ZONES, len(PAIRS)))
    for k, (a, b) in enumerate(PAIRS):
        best = np.zeros(N_ZONES)
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                c = (x[:m - lag, :, a] * x[lag:, :, b]).sum(axis=0) / (m - lag)
            else:
                c = (x[-lag:, :, a] * x[:m + lag, :, b]).sum(axis=0) / (m + lag)
            best = np.maximum(best, np.abs(c))
        out[:, k] = best
    return out


def rolling_pair_coupling(resid, window=240, step=10, max_lag=20):
    """Slide over a whole run (offline use). Returns (end_ticks, (m, 8, 3))."""
    ends, vals = [], []
    for end in range(window, resid.shape[0] + 1, step):
        ends.append(end - 1)
        vals.append(pair_coupling(resid[end - window:end], max_lag))
    return np.array(ends), np.array(vals)


def build_coupling_baseline(resid_runs, window=240, step=10, max_lag=20):
    """What 'healthy coupling' looks like. Returns a plain dict for baseline.json."""
    allv = np.vstack([rolling_pair_coupling(r, window, step, max_lag)[1] for r in resid_runs])
    mean = allv.mean(axis=0)
    return {
        "window": window,
        "max_lag": max_lag,
        "mean": mean.tolist(),
        "std": np.maximum(allv.std(axis=0), 0.03).tolist(),
        "active": (mean >= ACTIVE_FLOOR).tolist(),
    }


def coupling_break(pairs, base):
    """pairs: (8, 3) from pair_coupling (or (m, 8, 3)). Returns zone scores 0-1:
    0 = links as healthy, near 1 = a normally linked pair has gone quiet."""
    drop = (np.array(base["mean"]) - pairs) / np.array(base["std"])
    drop = np.where(np.array(base["active"]), drop, 0.0)
    worst = np.maximum(0.0, drop).max(axis=-1)
    return 1.0 - np.exp(-worst / 4.0)

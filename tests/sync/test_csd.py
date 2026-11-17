import numpy as np

from backend.sync.csd import (build_csd_baseline, csd_score, rolling_features,
                 window_features, zone_scores)
from backend.sync.fake_ticks import generate
from backend.sync.sync_math import build_profile, detrend


def _ar1(phi, n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 0.02)
    return x[:, None]


def _run(seed, stress_from=None):
    """Fake run held in memory -> (values[n,24], node_ids, t[n])."""
    rows, t, nodes = [], [], None
    for env in generate(seed, 3, stress_from):
        r = env["payload"]["readings"]
        if nodes is None:
            nodes = [f'{x["zone"]}.{x["system"]}' for x in r]
        rows.append([x["value"] for x in r])
        t.append(env["t"])
    return np.array(rows), nodes, np.array(t)


def test_sluggish_signal_has_higher_variance_and_autocorrelation():
    v_ok, a_ok = window_features(_ar1(0.5))
    v_slow, a_slow = window_features(_ar1(0.95))
    assert a_slow[0] > a_ok[0] + 0.2
    assert v_slow[0] > v_ok[0]


def test_early_warning_beats_threshold_alarm_and_calm_stays_quiet():
    calm = [_run(s) for s in (1, 3, 5)]
    profile = build_profile(np.vstack([c[0] for c in calm]),
                            np.concatenate([c[2] for c in calm]))
    base = build_csd_baseline([detrend(c[0], c[2], profile) for c in calm])

    # Calm city (seed 4, not used for the baseline): almost no alarms
    v, _, t = _run(4)
    ends, var, ac1 = rolling_features(detrend(v, t, profile))
    calm_zone = zone_scores(csd_score(var, ac1, base))
    assert (calm_zone > 0.5).mean() < 0.02

    # Slow-burn stress at silk_board: CSD fires before value passes 0.9
    v, nodes, t = _run(2, stress_from=2000)
    ends, var, ac1 = rolling_features(detrend(v, t, profile))
    zs = zone_scores(csd_score(var, ac1, base))
    assert (zs[:, 0] > 0.5).any()
    csd_tick = ends[np.argmax(zs[:, 0] > 0.5)]
    alarm_tick = np.argmax(v[:, nodes.index("silk_board.traffic")] > 0.9)
    assert alarm_tick - csd_tick > 60  # at least an hour of early warning

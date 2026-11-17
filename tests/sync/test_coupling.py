import numpy as np

from backend.sync.coupling import (
    build_coupling_baseline,
    coupling_break,
    pair_coupling,
    rolling_pair_coupling,
)
from backend.sync.fake_ticks import generate
from backend.sync.sync_math import build_profile, detrend


def _run(seed, stress_from=None):
    rows, t = [], []
    for env in generate(seed, 3, stress_from):
        rows.append([x["value"] for x in env["payload"]["readings"]])
        t.append(env["t"])
    return np.array(rows), np.array(t)


def test_lagged_link_is_found_and_broken_link_is_not():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, (240, 24))
    w[12:, 1] = w[:-12, 0] + rng.normal(0, 0.3, 228)  # zone 0 power follows traffic by 12 min
    linked = pair_coupling(w)[0, 0]
    w[:, 1] = rng.normal(0, 1, 240)  # link broken
    broken = pair_coupling(w)[0, 0]
    assert linked > 0.7
    assert broken < linked - 0.3


def test_calm_stays_quiet_and_hijack_is_caught():
    calm = [_run(s) for s in (1, 3, 5)]
    profile = build_profile(np.vstack([c[0] for c in calm]),
                            np.concatenate([c[1] for c in calm]))
    base = build_coupling_baseline([detrend(c[0], c[1], profile) for c in calm])
    assert base["active"][0][0]  # silk_board traffic-power is a healthy link

    v, t = _run(4)
    ends, pc = rolling_pair_coupling(detrend(v, t, profile))
    assert (coupling_break(pc, base) > 0.5).mean() < 0.02

    # Hijack: from tick 2000 silk_board traffic is replaced by unrelated noise
    rng = np.random.default_rng(99)
    ar = np.zeros(len(t))
    for i in range(1, len(t)):
        ar[i] = 0.6 * ar[i - 1] + rng.normal(0, 0.02)
    hijacked = v.copy()
    hijacked[2000:, 0] = profile[t % 1440][2000:, 0] + ar[2000:]
    ends, pc = rolling_pair_coupling(detrend(hijacked, t, profile))
    scores = coupling_break(pc, base)
    hit = scores[:, 0] > 0.5
    assert hit.any()
    assert 2000 < ends[np.argmax(hit)] < 2400

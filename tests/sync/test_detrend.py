import numpy as np

from backend.sync.fake_ticks import generate
from backend.sync.sync_math import build_profile, detrend


def _run(seed, stress_from=None):
    """Fake run built in memory -> (values[n, 24], node_ids, t[n])."""
    rows, t, nodes = [], [], None
    for env in generate(seed, 3, stress_from):
        r = env["payload"]["readings"]
        if nodes is None:
            nodes = [f'{x["zone"]}.{x["system"]}' for x in r]
        rows.append([x["value"] for x in r])
        t.append(env["t"])
    return np.array(rows), nodes, np.array(t)


def test_detrend_removes_daily_cycle():
    values, nodes, t = _run(1)
    profile = build_profile(values, t)
    resid = detrend(values, t, profile)
    assert resid.shape == values.shape
    # The daily cycle is gone, so the residual is much flatter than the raw data
    assert resid.std() < 0.5 * values.std()
    # And it is centred near zero for every node
    assert np.abs(resid.mean(axis=0)).max() < 0.02


def test_stress_shows_up_in_residual_variance():
    calm, nodes, t = _run(1)
    profile = build_profile(calm, t)
    stress, _, ts = _run(2, stress_from=2000)
    resid = detrend(stress, ts, profile)
    i = nodes.index("silk_board.traffic")
    early = resid[1000:1500, i].var()
    late = resid[3000:3500, i].var()
    assert late > 1.5 * early
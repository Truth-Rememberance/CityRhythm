import numpy as np

from backend.sync.phase import phase_coherence


def _waves(periods, phases, n=180, noise=0.1, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    x = np.stack([np.sin(2 * np.pi * t / p + ph) for p, ph in zip(periods, phases)], axis=1)
    return x + rng.normal(0, noise, x.shape)


def test_in_sync_waves_give_high_coherence():
    w = _waves([60, 60, 60], [0.0, 0.1, 0.2])
    r, phases = phase_coherence(w)
    assert r > 0.9
    assert len(phases) == 3


def test_drifting_waves_give_lower_coherence():
    in_sync, _ = phase_coherence(_waves([60, 60, 60], [0.0, 0.1, 0.2]))
    drifting, _ = phase_coherence(_waves([47, 61, 83], [0.3, 2.1, 4.0]))
    assert drifting < 0.75
    assert in_sync - drifting > 0.2


def test_coherence_stays_between_0_and_1():
    rng = np.random.default_rng(1)
    r, _ = phase_coherence(rng.normal(0, 1, (180, 3)))
    assert 0.0 <= r <= 1.0

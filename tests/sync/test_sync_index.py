import numpy as np

from backend.sync.sync_index import soften, status, sync_index

W = {"csd": 0.5, "coupling": 0.3, "phase": 0.2}
BANDS = (70.0, 40.0)


def test_calm_zone_scores_100():
    assert float(sync_index(csd=0.0, coupling=0.0, phase=0.0, weights=W)) == 100.0


def test_all_detectors_firing_gives_low_index():
    idx = float(sync_index(csd=0.9, coupling=0.8, phase=0.7, weights=W))
    assert abs(idx - 17.0) < 1e-6


def test_missing_scores_are_rescaled():
    # Only CSD available: its weight is rescaled to 1, so index = 100 * (1 - 0.6)
    assert abs(float(sync_index(csd=0.6, weights=W)) - 40.0) < 1e-6


def test_works_on_all_eight_zones_at_once():
    csd = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5])
    idx = sync_index(csd=csd, weights=W)
    assert idx.shape == (8,)
    assert idx.min() >= 0.0 and idx.max() <= 100.0


def test_status_bands():
    assert status(85, BANDS) == "healthy"
    assert status(70, BANDS) == "healthy"
    assert status(55, BANDS) == "drifting"
    assert status(40, BANDS) == "drifting"
    assert status(20, BANDS) == "critical"


def test_soften_floor_is_zero():
    assert float(soften(0.4, 0.4)) == 0.0


def test_soften_one_is_one():
    assert float(soften(1.0, 0.4)) == 1.0


def test_soften_stretches_scores_above_floor():
    assert abs(float(soften(0.7, 0.4)) - 0.5) < 1e-6


def test_soften_array_keeps_shape():
    scores = np.array([[0.4, 0.7], [1.0, 0.2]])
    assert soften(scores, 0.4).shape == scores.shape

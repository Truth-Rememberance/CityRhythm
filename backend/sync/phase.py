"""R3 sync math, step 2: phase coherence (Kuramoto-style).

Idea: every system (traffic, power, water) wobbles up and down. Each wobble
has a "phase" (where it is in its cycle, like the hand of a clock). If the
three hands point the same way, the zone is "in sync". Phase coherence R is
how aligned the hands are: 1.0 = perfectly together, ~0.6 = random.

Input is the RESIDUAL from detrend() for ONE zone, not raw values.
"""
import numpy as np
from scipy.signal import hilbert


def _smooth(x, k):
    kernel = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, x)


def phase_coherence(resid_window, smooth_min=10, trim=15):
    """resid_window: array (n_minutes, 3) = one zone's traffic, power, water
    residuals over the last ~120-180 minutes.

    Returns (R, phases):
      R      = average alignment of the phases over the window, 0-1
      phases = each system's phase (radians) at the latest minute, for the
               "phases" field of the sync.index payload

    The Hilbert transform is unreliable at the very edges of a window, so the
    first and last `trim` minutes are left out of R.
    """
    x = _smooth(np.asarray(resid_window, dtype=float), smooth_min)
    x = x - x.mean(axis=0)
    theta = np.angle(hilbert(x, axis=0))
    middle = theta[trim:-trim]
    r = np.abs(np.exp(1j * middle).mean(axis=1)).mean()
    return float(r), theta[-1]

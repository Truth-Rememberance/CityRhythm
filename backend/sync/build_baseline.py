"""R3: build the CSD baseline from R2's real calm runs (v1, provisional)."""
import json
import os

import numpy as np

from backend.sync.csd import (
    build_csd_baseline,
    csd_score,
    rolling_features,
    zone_scores,
)
from backend.sync.real_data import (
    build_profiles,
    detrend_by_day,
    load_run_gz,
)

DATA_DIR = os.path.expanduser("~/scratch/real")
OUT_PATH = os.path.expanduser("~/scratch/baseline_v1.json")
TRAIN_SEEDS = [1001, 1002, 1003]
TEST_SEEDS = [1004, 1005]


def _load(seed):
    return load_run_gz(os.path.join(DATA_DIR, f"calm_run_seed{seed}.jsonl.gz"))


def build(train_seeds=TRAIN_SEEDS):
    """Pool the training runs, build day-type profiles, then the CSD baseline."""
    runs = [_load(s) for s in train_seeds]
    values = np.vstack([r[0] for r in runs])
    t = np.concatenate([r[2] for r in runs])
    days = np.concatenate([r[3] for r in runs])
    profiles = build_profiles(values, t, days)
    resids = [detrend_by_day(r[0], r[2], r[3], profiles) for r in runs]
    return profiles, build_csd_baseline(resids)


def check(seed, profiles, base):
    """Score a calm run that was NOT used for training."""
    v, _, t, d = _load(seed)
    resid = detrend_by_day(v, t, d, profiles)
    _, var, ac1 = rolling_features(resid)
    zs = zone_scores(csd_score(var, ac1, base))
    print(f"seed {seed}: share of zone-scores above 0.5 = "
          f"{(zs > 0.5).mean():.3f}, max = {zs.max():.2f}")


if __name__ == "__main__":
    profiles, base = build()
    for seed in TEST_SEEDS:
        check(seed, profiles, base)
    out = {
        "version": "v1-provisional",
        "csd": base,
        "profiles": {k: np.round(p, 4).tolist() for k, p in profiles.items()},
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print("wrote", OUT_PATH)
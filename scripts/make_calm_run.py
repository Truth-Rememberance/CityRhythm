#!/usr/bin/env python
"""Generate calm-run datasets: 4-day gzipped sim.tick streams.

Usage:
    .venv/bin/python scripts/make_calm_run.py                # seeds 1001-1005
    .venv/bin/python scripts/make_calm_run.py --seeds 1001   # single seed
    .venv/bin/python scripts/make_calm_run.py --out /tmp/x   # custom dir

Per ERRATA items 1-6: each run is 4 sim days (5760 ticks) from
Thu 2026-11-19T00:00+05:30; one day of warm-up is simulated and discarded;
values are rounded; output is gzip-compressed JSONL, one envelope per line,
written to <out>/calm_run_seed<seed>.jsonl.gz. Seed 1005 applies the
synthetic holiday on Fri 2026-11-20 (declared in data/calendar.json).
All data is synthetic.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.sim.contracts_stub import ScenarioSpec
from backend.sim.serialize import run_scenario

DEFAULT_START = "2026-11-19"  # Thursday (ERRATA items 1-2)
SYNTHETIC_HOLIDAY = "2026-11-20"  # Friday, seed 1005 only
DEFAULT_SEEDS = (1001, 1002, 1003, 1004, 1005)


def make_calm_run(seed: int, out_dir: Path) -> Path:
    """Generate one 4-day calm run and write calm_run_seed<seed>.jsonl.gz."""
    spec = ScenarioSpec(
        run_id=f"calm-{seed}",
        seed=seed,
        start_date=DEFAULT_START,
        start_time="00:00",
        days=4,
        holidays=[SYNTHETIC_HOLIDAY] if seed == 1005 else [],
    )
    envelopes = run_scenario(spec)
    out = out_dir / f"calm_run_seed{seed}.jsonl.gz"
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
        for env in envelopes:
            fh.write(json.dumps(env, separators=(",", ":")) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        t0 = time.perf_counter()
        path = make_calm_run(seed, args.out)
        dt_s = time.perf_counter() - t0
        print(f"seed {seed}: {path} ({path.stat().st_size / 1e6:.1f} MB) in {dt_s:.1f}s")


if __name__ == "__main__":
    main()

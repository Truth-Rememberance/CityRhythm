"""Dev-only fake tick generator (R3). NOT part of the hub.

Writes sim.tick envelopes in R2's exact format (1 tick = 1 sim minute,
24 readings: 8 zones x traffic/power/water, "value" 0-1).

Usage:
    python fake_ticks.py 1            # calm run, seed 1, 3 days
    python fake_ticks.py 2 2000       # slow-burn stress starting at tick 2000
Output goes to stdout as JSONL, so redirect it:
    python fake_ticks.py 1 > calm_fake_1.jsonl
"""
import json
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
ZONES = ["silk_board", "whitefield", "hebbal", "electronic_city",
         "koramangala", "indiranagar", "jayanagar", "yelahanka"]
SYSTEMS = [("traffic", "flow", "veh_per_min", 200.0),
           ("power", "load", "MW", 60.0),
           ("water", "demand", "kL_per_min", 30.0)]
NODES = [(z, s) for z in ZONES for s in SYSTEMS]  # 24, zone-major order


def generate(seed=1, days=3, stress_from=None):
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = np.arange(n)

    # Stress ramp: after stress_from, silk_board drifts up and gets "sluggish"
    # (autocorrelation rises), which mimics critical slowing down.
    ramp = np.zeros(n)
    phi = np.full((n, len(NODES)), 0.6)
    if stress_from is not None:
        ramp = np.clip((t - stress_from) / 600, 0, 1)
        phi[:, :3] = (0.6 + 0.37 * ramp)[:, None]  # only silk_board's 3 nodes

    noise = np.zeros((n, len(NODES)))
    for i in range(1, n):
        noise[i] = phi[i] * noise[i - 1] + rng.normal(0, 0.02, len(NODES))

    # Hidden-style coupling: power follows traffic with a 12 minute lag
    for zi in range(len(ZONES)):
        tr, pw = zi * 3, zi * 3 + 1
        noise[12:, pw] += 0.5 * noise[:-12, tr]

    start = datetime(2026, 11, 16, 0, 0, tzinfo=IST)
    for i in range(n):
        readings = []
        for k, (zone, (system, metric, unit, scale)) in enumerate(NODES):
            daily = 0.15 * np.sin(2 * np.pi * (i - 360 - 60 * (k % 3)) / 1440)
            drift = 0.3 * ramp[i] if zone == "silk_board" else 0.0
            value = round(float(0.55 + daily + noise[i, k] + drift), 3)
            readings.append({
                "zone": zone, "system": system, "metric": metric,
                "value": value, "raw": round(value * scale, 1), "unit": unit,
                "sensor_id": f"{zone}.{system}.01",
            })
        yield {
            "v": 1, "channel": "sim.tick", "run_id": f"r_fake_{seed}",
            "t": i, "ts": (start + timedelta(minutes=i)).isoformat(),
            "zone": None,
            "payload": {"readings": readings,
                        "context": {"temp_c": 27.0, "events": [], "is_holiday": False},
                        "reports": []},
        }


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    stress = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for env in generate(seed, 3, stress):
        print(json.dumps(env))

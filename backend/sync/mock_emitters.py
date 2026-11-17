"""Deterministic mock Sync events for downstream R4/R6 integration.

No exact sync.graph payload shape was found in docs/ or contracts/. This
module therefore uses the proposed shape: nodes is the list of eight zone
names, and edges contain source, target, lag_min, and strength fields.
"""

import argparse
import json
import math
from pathlib import Path

from backend.sync.module import _event
from backend.sync.sync_index import load_bands, load_csd_floor, status

ZONES = (
    "silk_board",
    "whitefield",
    "hebbal",
    "electronic_city",
    "koramangala",
    "indiranagar",
    "jayanagar",
    "yelahanka",
)
SYSTEMS = ("traffic", "power", "water")
STRESS_START = 2000
STRESS_END = 2600
ALARM_TICK = 2700
HEALTHY_CENTER = 90.0
HEALTHY_WOBBLE = 3.0


def _index_for(zone, tick):
    zone_index = ZONES.index(zone)
    wobble = HEALTHY_WOBBLE * math.sin(2.0 * math.pi * tick / 720.0 + zone_index * 0.37)
    if zone != "silk_board" or tick < STRESS_START:
        return HEALTHY_CENTER + wobble

    start_wobble = HEALTHY_WOBBLE * math.sin(
        2.0 * math.pi * STRESS_START / 720.0
    )
    start_index = HEALTHY_CENTER + start_wobble
    progress = min((tick - STRESS_START) / (STRESS_END - STRESS_START), 1.0)
    return start_index + (30.0 - start_index) * progress


def _transition_tick(target_status, bands):
    for tick in range(STRESS_START, STRESS_END + 1):
        if status(_index_for("silk_board", tick), bands) == target_status:
            return tick
    return None


BANDS = load_bands()
CSD_FLOOR = load_csd_floor()
FIRST_DRIFTING_TICK = _transition_tick("drifting", BANDS)
FIRST_CRITICAL_TICK = _transition_tick("critical", BANDS)


def _raw_csd_score(index):
    softened_score = 1.0 - index / 100.0
    return CSD_FLOOR + softened_score * (1.0 - CSD_FLOOR)


def mock_events(tick, fill_nulls=False):
    """Return deterministic mock sync.index, sync.warning and sync.graph events."""
    tick = int(tick)
    events = []
    silk_index = _index_for("silk_board", tick)
    progress = min(max((tick - STRESS_START) / (STRESS_END - STRESS_START), 0.0), 1.0)

    for zone in ZONES:
        index = _index_for(zone, tick)
        coherence = max(0.0, min(1.0, index / 100.0)) if fill_nulls else None
        coupling_break = max(0.0, min(1.0, 1.0 - index / 100.0)) if fill_nulls else None
        phases = None
        if fill_nulls:
            zone_phase = ZONES.index(zone) * 0.19
            phases = {
                system: (tick * (0.008 + system_index * 0.001) + zone_phase) % (2.0 * math.pi)
                for system_index, system in enumerate(SYSTEMS)
            }
        events.append(_event(
            "sync.index",
            zone,
            {
                "zone": zone,
                "sync_index": index,
                "status": status(index, BANDS),
                "phase_coherence": coherence,
                "csd_score": _raw_csd_score(index),
                "coupling_break": coupling_break,
                "phases": phases,
            },
        ))

    if tick == FIRST_DRIFTING_TICK:
        events.append(_event(
            "sync.warning",
            "silk_board",
            {
                "zone": "silk_board",
                "level": "drifting",
                "sync_index": silk_index,
                "trigger": "csd",
                "baseline_fired": False,
                "lead_min": None,
            },
        ))
    if tick == FIRST_CRITICAL_TICK:
        events.append(_event(
            "sync.warning",
            "silk_board",
            {
                "zone": "silk_board",
                "level": "critical",
                "sync_index": silk_index,
                "trigger": "csd",
                "baseline_fired": False,
                "lead_min": None,
            },
        ))
    if tick == ALARM_TICK:
        events.append(_event(
            "baseline.alarm",
            "silk_board",
            {
                "zone": "silk_board",
                "system": "traffic",
                "metric": "flow",
                "value": 0.95,
                "threshold": 0.9,
                "sustained_min": 5,
            },
        ))
        first_warning_tick = FIRST_DRIFTING_TICK
        events.append(_event(
            "sync.warning",
            "silk_board",
            {
                "zone": "silk_board",
                "level": status(silk_index, BANDS),
                "sync_index": silk_index,
                "trigger": "csd",
                "baseline_fired": True,
                "lead_min": ALARM_TICK - first_warning_tick,
            },
        ))

    if tick % 60 == 0:
        events.append(_event(
            "sync.graph",
            None,
            {
                "nodes": list(ZONES),
                "edges": [
                    {
                        "source": "silk_board",
                        "target": zone,
                        "lag_min": 5 + zone_index,
                        "strength": 0.1 + 0.8 * progress,
                    }
                    for zone_index, zone in enumerate(ZONES)
                    if zone != "silk_board"
                ],
            },
        ))

    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write deterministic mock Sync events as JSONL.")
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--fill-nulls", action="store_true")
    args = parser.parse_args(argv)
    if args.ticks < 0:
        parser.error("--ticks must be non-negative")

    output_path = Path(__file__).with_name("mock_events.jsonl")
    event_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for tick in range(args.ticks):
            for event in mock_events(tick, fill_nulls=args.fill_nulls):
                output.write(json.dumps({"t": tick, **event}, sort_keys=True) + "\n")
                event_count += 1
    print(f"Wrote {event_count} events to {output_path.as_posix()}")


if __name__ == "__main__":
    main()

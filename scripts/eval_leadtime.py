"""Provisional lead-time evaluation for SyncModule versus a threshold alarm."""

import argparse
import gzip
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.sync.module import SyncModule
from backend.sync.threshold_alarm import ThresholdAlarm

MINUTES_PER_DAY = 1440


def evaluate_envelopes(
    envelopes: Iterable[dict],
    incident_tick: int | None = None,
    zone: str = "silk_board",
    baseline_threshold: float = 1.0,
    baseline_sustained: int = 15,
) -> dict:
    """Evaluate one stream of sim.tick envelopes and return lead-time metrics."""
    module = SyncModule()
    alarm = ThresholdAlarm(
        threshold=baseline_threshold,
        sustained_min=baseline_sustained,
    )
    first_warning_tick = None
    first_baseline_alarm_tick = None
    false_warnings = 0
    false_alarms = 0
    tick_count = 0
    pre_incident_tick_count = 0

    for envelope in envelopes:
        tick = int(envelope["t"])
        tick_count += 1
        before_incident = incident_tick is None or tick < incident_tick
        if incident_tick is not None and before_incident:
            pre_incident_tick_count += 1

        events = module.on_tick(envelope, None)
        readings = envelope.get("payload", envelope).get("readings", [])
        alarm_events = alarm.update(readings)

        for event in events:
            if (
                event["channel"] == "sync.warning"
                and event["zone"] == zone
                and event["payload"].get("baseline_fired") is False
            ):
                if before_incident:
                    false_warnings += 1
                elif first_warning_tick is None:
                    first_warning_tick = tick

        for event in alarm_events:
            if event["zone"] != zone:
                continue
            if before_incident:
                false_alarms += 1
            elif first_baseline_alarm_tick is None:
                first_baseline_alarm_tick = tick

    lead_min = None
    if first_warning_tick is not None and first_baseline_alarm_tick is not None:
        lead_min = first_baseline_alarm_tick - first_warning_tick

    exposure_ticks = (
        pre_incident_tick_count if incident_tick is not None else tick_count
    )
    exposure_days = exposure_ticks / MINUTES_PER_DAY
    false_warning_per_day = (
        false_warnings / exposure_days if exposure_days else 0.0
    )
    false_alarm_per_day = false_alarms / exposure_days if exposure_days else 0.0
    return {
        "zone": zone,
        "incident_tick": incident_tick,
        "first_warning_tick": first_warning_tick,
        "first_baseline_alarm_tick": first_baseline_alarm_tick,
        "lead_min": lead_min,
        "false_warnings": false_warnings,
        "false_alarms": false_alarms,
        "false_warning_per_day": false_warning_per_day,
        "false_alarm_per_day": false_alarm_per_day,
        "ticks": tick_count,
    }


def _read_envelopes(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _format_tick(value):
    return "-" if value is None else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Input sim.tick .jsonl.gz paths")
    parser.add_argument(
        "--incident-tick",
        nargs="+",
        type=int,
        help="Incident tick for all runs, or one tick per run",
    )
    parser.add_argument("--zone", default="silk_board")
    parser.add_argument("--baseline-threshold", type=float, default=1.0)
    parser.add_argument("--baseline-sustained", type=int, default=15)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if args.incident_tick and len(args.incident_tick) not in (1, len(args.runs)):
        parser.error("--incident-tick must provide one value or one value per run")
    threshold = 1.1 if args.strict else args.baseline_threshold
    sustained = 15 if args.strict else args.baseline_sustained
    incident_ticks = args.incident_tick or [None]
    if len(incident_ticks) == 1:
        incident_ticks *= len(args.runs)

    results = []
    for path, incident in zip(args.runs, incident_ticks):
        result = evaluate_envelopes(
            _read_envelopes(path),
            incident_tick=incident,
            zone=args.zone,
            baseline_threshold=threshold,
            baseline_sustained=sustained,
        )
        result["run"] = Path(path).name
        results.append(result)

    print(f"zone={args.zone} baseline_threshold={threshold:g} baseline_sustained={sustained}")
    print("run | incident_tick | sync_warning_tick | baseline_alarm_tick | lead_min | false_warnings/day | false_alarms/day")
    for result in results:
        print(
            f"{result['run']} | {_format_tick(result['incident_tick'])} | "
            f"{_format_tick(result['first_warning_tick'])} | "
            f"{_format_tick(result['first_baseline_alarm_tick'])} | "
            f"{_format_tick(result['lead_min'])} | "
            f"{result['false_warning_per_day']:.6f} | "
            f"{result['false_alarm_per_day']:.6f}"
        )

    total_ticks = sum(result["ticks"] for result in results)
    total_days = total_ticks / MINUTES_PER_DAY
    false_warnings = sum(result["false_warnings"] for result in results)
    false_alarms = sum(result["false_alarms"] for result in results)
    leads = [result["lead_min"] for result in results if result["lead_min"] is not None]
    mean_lead = sum(leads) / len(leads) if leads else None
    print(
        "summary: "
        f"runs={len(results)} false_warnings={false_warnings} "
        f"false_alarms={false_alarms} "
        f"false_warnings/day={false_warnings / total_days if total_days else 0.0:.6f} "
        f"false_alarms/day={false_alarms / total_days if total_days else 0.0:.6f} "
        f"mean_lead_min={_format_tick(mean_lead)}"
    )
    print("PROVISIONAL: not from docs/RESULTS.md")


if __name__ == "__main__":
    main()

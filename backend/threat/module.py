
from r4_detector import (
    physics_consistency,
    replay_detector,
    sensor_trust,
    classify
)

from threat.redteam import redteam
from threat.citizen import (
    extract_report,
    report_sensor_agreement
)


class ThreatModule:

    def __init__(self):
        self.history_by_zone = {}
        self.max_history = 10
        self.graph = {"edges": []}

    def on_tick(self, tick, ctx):

        events = []

        # Read simulation envelope
        payload = tick.get("payload", tick)

        readings = payload.get("readings", [])
        reports = payload.get("reports", [])

        # ------------------------------------------------
        # 1. Group readings zone-wise
        # ------------------------------------------------

        zones = {}

        for reading in readings:
            zone = reading.get("zone")

            if zone:
                zones.setdefault(zone, []).append(reading)

        # ------------------------------------------------
        # 2. Threat assessment and sensor trust per zone
        # ------------------------------------------------

        for zone, zone_readings in zones.items():

            history = self.history_by_zone.setdefault(zone, [])

            history.append(zone_readings)
            history[:] = history[-self.max_history:]

            physics = physics_consistency(zone_readings)

            replay = replay_detector(history)

            trust = sensor_trust(history)

            context = payload.get("context", {})
            assessment = classify(
    physics,
    replay,
    context
)

            events.append({
                "channel": "threat.assessment",
                "zone": zone,
                "payload": {
                    "zone": zone,
                    "label": assessment["label"],
                    "confidence": assessment["confidence"],
                    "entry_node": assessment["entry_node"],
                    "evidence": assessment["evidence"]
                }
            })

            events.append({
                "channel": "threat.trust",
                "zone": zone,
                "payload": {
                    "zone": zone,
                    "sensors": trust
                }
            })

        # ------------------------------------------------
        # 3. Citizen report extraction
        # ------------------------------------------------

        extracted_reports = []

        for report in reports:

            extracted = extract_report(
                report.get("text", "")
            )

            extracted_reports.append({
                "id": report.get("id"),
                "zone": report.get("zone"),
                "category": extracted["category"],
                "language": extracted["language"],
                "keywords": extracted["keywords"],
                "label": report.get("label", "unlabelled")
            })

        # ------------------------------------------------
        # 4. Citizen report and sensor agreement
        # ------------------------------------------------

        report_checks = report_sensor_agreement(
            reports,
            readings
        )

        events.append({
            "channel": "threat.report_check",
            "zone": None,
            "payload": {
                "reports": extracted_reports,
                "zone_checks": report_checks
            }
        })

        # ------------------------------------------------
        # 5. Read graph from R3
        # ------------------------------------------------

        latest = getattr(ctx, "latest", None)

        if callable(latest):

            graph_event = latest("sync.graph", None)

            if graph_event:

                if isinstance(graph_event, dict):
                    candidate = graph_event.get(
                        "payload", graph_event
                    )
                else:
                    candidate = getattr(
                        graph_event, "payload", None
                    )

                if (
                    isinstance(candidate, dict)
                    and isinstance(candidate.get("edges"), list)
                ):
                    self.graph = candidate

        # ------------------------------------------------
        # 6. Red-team analysis
        # ------------------------------------------------

        attack_paths = redteam(self.graph)

        events.append({
            "channel": "threat.redteam",
            "zone": None,
            "payload": {
                "attack_paths": attack_paths
            }
        })

        return events

    def on_command(self, cmd, ctx):
        return []

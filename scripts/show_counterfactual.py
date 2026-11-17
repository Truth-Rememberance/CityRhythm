from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.fix._local_contracts import Action
from backend.fix.agents import Proposal
from backend.fix.conflicts import Resolution, build_plan
from backend.fix.counterfactual import CounterfactualCoordinator
from backend.fix.testing.toy_sim import ToySim


def main() -> None:
    sim_now = 1199
    incident_start = 1200
    response_zones = (
        "silk_board",
        "koramangala",
        "indiranagar",
        "whitefield",
        "hebbal",
        "electronic_city",
        "jayanagar",
    )
    sim = ToySim(seed=5, start_t=sim_now)
    sim.inject("heatwave", start_t=incident_start, duration_min=90, zone="silk_board")
    actions: list[Action] = []
    for zone in response_zones:
        actions.extend(
            [
                Action(
                    id=f"{zone}-traffic",
                    type="offset_signal",
                    dept="traffic",
                    zone=zone,
                    params={"offset_s": 45},
                    start_t=incident_start + 9,
                    duration_min=150,
                ),
                Action(
                    id=f"{zone}-power",
                    type="shed_load",
                    dept="power",
                    zone=zone,
                    params={"pct": 55, "feeder": "main"},
                    start_t=incident_start + 9,
                    duration_min=150,
                ),
                Action(
                    id=f"{zone}-water",
                    type="shift_pump",
                    dept="water",
                    zone=zone,
                    params={"shift_min": 35, "movable_fraction": 0.5},
                    start_t=incident_start + 9,
                    duration_min=150,
                ),
            ]
        )
    proposals = [
        Proposal(
            action=action,
            utility=1.0,
            rationale="Test intervention.",
            touches=[f"{action.zone}.{action.dept}"],
            expected_load_delta={f"{action.zone}.{action.dept}": -0.1},
        )
        for action in actions
    ]
    plan = build_plan(Resolution(proposals, [], [], [], total_cost=9.0), incident_start)
    result = CounterfactualCoordinator(sim, horizon_min=180).run(
        plan, sim_now=sim_now, incident_start=incident_start
    )

    for branch in ("no_action", "with_plan"):
        service = result[branch]["service_level"]
        minimum = min(service) if service else 0.0
        minimum_minute = result["t"][service.index(minimum)] if service else None
        print(f"{branch}:")
        print(f"  recovery_min: {result['recovery_min'][branch]:.0f}")
        print(f"  did_recover: {result['extras'][f'{branch}_recovered']}")
        print(f"  min_service_level: {minimum:.3f}")
        print(f"  minute_of_minimum: {minimum_minute}")
        print(f"  mean_service_level: {sum(service) / len(service):.3f}" if service else "  mean_service_level: n/a")
        affected_zones = sum(
            min(zone_service, default=1.0) < 0.95
            for zone_service in result["extras"]["per_zone"][branch].values()
        )
        print(f"  zones_affected (service below 0.95): {affected_zones}")


if __name__ == "__main__":
    main()
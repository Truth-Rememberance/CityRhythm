from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.fix._local_contracts import ACTION_TYPES, ZONES, Action
from backend.fix.conflicts import Plan
from backend.fix.counterfactual import CounterfactualCoordinator
from backend.fix.metrics import cut_percent
from backend.fix.rephase import make_demo_profiles, rephase_optimizer
from backend.fix.testing.toy_sim import ToySim

SEEDS = tuple(range(10))
RESPONSE_ZONES = (
    "silk_board",
    "koramangala",
    "indiranagar",
    "whitefield",
    "hebbal",
    "electronic_city",
    "jayanagar",
)


def _heatwave_plan(start_t: int) -> Plan:
    actions: list[Action] = []
    for zone in RESPONSE_ZONES:
        actions.extend(
            [
                Action(
                    id=f"{zone}-traffic",
                    type="offset_signal",
                    dept="traffic",
                    zone=zone,
                    params={"offset_s": 45},
                    start_t=start_t,
                    duration_min=150,
                ),
                Action(
                    id=f"{zone}-power",
                    type="shed_load",
                    dept="power",
                    zone=zone,
                    params={"pct": 55, "feeder": "main"},
                    start_t=start_t,
                    duration_min=150,
                ),
                Action(
                    id=f"{zone}-water",
                    type="shift_pump",
                    dept="water",
                    zone=zone,
                    params={"shift_min": 35, "movable_fraction": 0.5},
                    start_t=start_t,
                    duration_min=150,
                ),
            ]
        )
    assert all(action.type in ACTION_TYPES for action in actions)
    assert all(action.zone in ZONES for action in actions)
    return Plan("eval_heatwave", actions, 0.0, {}, [])


def evaluate_seed(seed: int) -> tuple[float, float, int, int]:
    sim_now, incident_start = 1199, 1200
    sim = ToySim(seed=seed, start_t=sim_now)
    sim.inject("heatwave", start_t=incident_start, duration_min=90, zone="silk_board")
    result = CounterfactualCoordinator(sim, horizon_min=240).run(
        _heatwave_plan(incident_start + 9), sim_now, incident_start
    )
    recovery_cut = cut_percent(
        result["recovery_min"]["no_action"], result["recovery_min"]["with_plan"]
    )
    rephase = rephase_optimizer(make_demo_profiles(seed), seed=seed, budget_ms=1500)
    return (
        recovery_cut,
        rephase.peak_overlap_cut_pct,
        result["recovery_min"]["no_action"],
        result["recovery_min"]["with_plan"],
    )


def main() -> None:
    rows = [(seed, *evaluate_seed(seed)) for seed in SEEDS]
    markdown = [
        "# ROLE 5 Fix Evaluation",
        "",
        "Ten deterministic ToySim heatwave seeds; recovery cut compares the two counterfactual branches.",
        "Peak-overlap cut uses the corresponding seeded demo profiles and re-phase optimizer.",
        "",
        "| Seed | Recovery-time cut | Peak-overlap cut | No-action recovery (min) | With-plan recovery (min) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, recovery_cut, overlap_cut, before, after in rows:
        markdown.append(
            f"| {seed} | {recovery_cut:.1f}% | {overlap_cut:.1f}% | {before} | {after} |"
        )
    mean_recovery = sum(row[1] for row in rows) / len(rows)
    mean_overlap = sum(row[2] for row in rows) / len(rows)
    markdown.extend(
        [
            f"| **Mean** | **{mean_recovery:.1f}%** | **{mean_overlap:.1f}%** | | |",
            "",
        ]
    )
    results_path = Path(__file__).resolve().parents[1] / "docs" / "RESULTS_fix.md"
    results_path.write_text("\n".join(markdown), encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
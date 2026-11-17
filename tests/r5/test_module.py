from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.fix.copilot import brief
from backend.fix.module import FixModule
from backend.fix.testing.toy_sim import ToySim


class ModuleContext:
    def __init__(self, sim: ToySim) -> None:
        self.sim = sim

    def latest(self, channel: str, zone: str | None) -> Any:
        return None

    def history(self, zone: str, system: str, minutes: int) -> list[float]:
        return []


def test_brief_template_returns_three_lines_and_top_actions(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_FORCE_FALLBACK", "1")
    result = brief(
        {
            "zones": [
                {"zone": "silk_board", "status": "critical", "fragility": 82, "service_level": 0.4}
            ],
            "proposals": [
                {
                    "action": {"type": "shed_load", "zone": "silk_board"},
                    "utility": 0.9,
                },
                {
                    "action": {"type": "offset_signal", "zone": "silk_board"},
                    "utility": 0.8,
                },
                {
                    "action": {"type": "shift_pump", "zone": "silk_board"},
                    "utility": 0.7,
                },
                {
                    "action": {"type": "reroute", "zone": "silk_board"},
                    "utility": 0.1,
                },
            ],
            "plan_actions": [],
        }
    )
    assert len(result["lines"]) == 3
    assert len(result["top_actions"]) == 3
    assert result["top_actions"][0] == "shed_load in silk_board"


def test_fix_module_survives_100_ticks_and_publishes_required_channels(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("LLM_FORCE_FALLBACK", "1")
    sim = ToySim(seed=12, start_t=1199)
    sim.inject("heatwave", start_t=1200, duration_min=90, zone="silk_board")
    ctx = ModuleContext(sim)
    module = FixModule()
    channels: set[str] = set()

    for _ in range(100):
        tick = sim.step()
        try:
            events = module.on_tick(tick, ctx)
        except Exception as error:
            raise AssertionError(f"on_tick raised at t={tick.t}: {error}") from error
        assert isinstance(events, list)
        assert all(hasattr(event, "channel") for event in events)
        assert all(event.channel != "system.error" for event in events)
        channels.update(event.channel for event in events)

    assert {
        "fix.proposals",
        "fix.plan",
        "fix.briefing",
        "metrics.resilience",
        "metrics.counterfactual",
        "metrics.intervention_window",
    } <= channels


def test_fix_commands_auto_mode_and_apply_are_safe(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_FORCE_FALLBACK", "1")
    sim = ToySim(seed=3, start_t=1199)
    ctx = ModuleContext(sim)
    module = FixModule()
    enabled = module.on_command(
        {"channel": "fix.auto_mode", "payload": {"enabled": True}, "t": 1199}, ctx
    )
    assert module.auto_mode
    assert enabled[0].channel == "fix.plan"
    missing_plan = module.on_command(
        {"channel": "fix.apply", "payload": {}, "t": 1199}, ctx
    )
    assert missing_plan[0].channel == "system.error"
    assert isinstance(missing_plan[0].payload, Mapping)


def test_fix_module_bad_tick_returns_system_error() -> None:
    events = FixModule().on_tick({"t": "bad"}, object())
    assert len(events) == 1
    assert events[0].channel == "system.error"
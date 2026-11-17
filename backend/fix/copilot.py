"""Three-line mayor briefing backed by the shared LLM helper and a deterministic fallback."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from backend.common.llm import complete


class Briefing(BaseModel):
    lines: list[str] = Field(min_length=3, max_length=3)
    top_actions: list[str] = Field(max_length=3)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _action_label(item: Any) -> str:
    action = _value(item, "action", item)
    action_type = _value(action, "type", _value(action, "action_type", "intervention"))
    zone = _value(action, "zone", _value(item, "zone", "city"))
    return f"{action_type} in {zone}"[:100]


def _template_brief(state: Any) -> dict[str, Any]:
    zones = _value(state, "zones", [])
    if not isinstance(zones, Sequence) or isinstance(zones, (str, bytes)):
        zones = []
    scored: list[tuple[float, str, str]] = []
    services: list[float] = []
    for zone_state in zones:
        zone = str(_value(zone_state, "zone", "city"))
        status = str(_value(zone_state, "status", "healthy"))
        try:
            score = float(_value(zone_state, "fragility", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        try:
            service = float(_value(zone_state, "service_level", 1.0))
        except (TypeError, ValueError):
            service = 1.0
        services.append(service)
        scored.append((score, zone, status))
    mean_service = sum(services) / len(services) if services else 1.0
    critical_count = sum(status == "critical" for _score, _zone, status in scored)
    highest = max(scored, default=(0.0, "city", "healthy"))
    proposals = _value(state, "proposals", [])
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        proposals = []
    if not proposals:
        proposals = _value(state, "plan_actions", [])
    ordered = sorted(
        proposals,
        key=lambda item: (-float(_value(item, "utility", 0.0)), _action_label(item)),
    )
    actions = [_action_label(item) for item in ordered[:3]]
    return {
        "lines": [
            f"City service is {mean_service:.0%}; {critical_count} zones are critical.",
            f"Highest fragility is {highest[1]} at {highest[0]:.0f}/100 ({highest[2]}).",
            f"The current plan contains {len(_value(state, 'plan_actions', []))} actions.",
        ],
        "top_actions": actions,
    }


def brief(state: Any) -> dict[str, list[str]]:
    """Return exactly three concise briefing lines and up to three ranked actions."""
    template = lambda: _template_brief(state)
    prompt = (
        "Write exactly three short mayor briefing lines and preserve the three-or-fewer "
        "ranked actions from this state. Do not invent actions. State:\n"
        + json.dumps(state, default=str, sort_keys=True)
    )
    result = complete(prompt, Briefing, template=template)
    if not isinstance(result, Briefing):
        return _template_brief(state)
    return {"lines": result.lines, "top_actions": result.top_actions[:3]}
from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from backend.common import llm
from backend.fix._local_contracts import Action, make_envelope


class Briefing(BaseModel):
    message: str = Field(min_length=1)


@pytest.fixture(autouse=True)
def clear_llm_state() -> None:
    with llm._CACHE_LOCK:
        llm._CACHE.clear()
    with llm._STATE_LOCK:
        llm._PROVIDER_FAILURES.clear()
        llm._PROVIDER_OPEN_UNTIL.clear()


def test_force_fallback_uses_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FORCE_FALLBACK", "1")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama,gemini,template")
    result = llm.complete("brief", template=lambda: "from template")
    assert result == "from template"
    assert llm.last_provider() == "template"


def test_force_fallback_validates_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FORCE_FALLBACK", "1")
    result = llm.complete("brief", Briefing, template=lambda: {"message": "ready"})
    assert isinstance(result, Briefing)
    assert result.message == "ready"


def test_invalid_template_uses_valid_schema_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_FORCE_FALLBACK", "1")
    result = llm.complete("brief", Briefing, template=dict)
    assert isinstance(result, Briefing)
    assert result.message


def test_network_failure_still_returns_nonempty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_FORCE_FALLBACK", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama,template")

    def fail_post(*args: Any, **kwargs: Any) -> None:
        raise httpx_error()

    monkeypatch.setattr(llm.httpx, "post", fail_post)
    result = llm.complete("brief")
    assert isinstance(result, str)
    assert result


def test_circuit_breaker_skips_provider_after_two_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_FORCE_FALLBACK", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama,template")
    calls = 0

    def fail_post(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        raise httpx_error()

    monkeypatch.setattr(llm.httpx, "post", fail_post)
    llm.complete("first")
    llm.complete("second")
    llm.complete("third")
    assert calls == 2


def test_cache_reuses_provider_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_FORCE_FALLBACK", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama,template")
    calls = 0

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "cached answer"}

    def successful_post(*args: Any, **kwargs: Any) -> Response:
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr(llm.httpx, "post", successful_post)
    assert llm.complete("same prompt") == "cached answer"
    assert llm.complete("same prompt") == "cached answer"
    assert calls == 1


def test_invalid_action_zone_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Action(
            id="a1",
            type="shift_pump",
            dept="water",
            zone="unknown_zone",
            params={},
            start_t=1,
            duration_min=5,
        )


def test_action_rejects_invalid_timing() -> None:
    with pytest.raises(ValidationError):
        Action(
            id="a1",
            type="shift_pump",
            dept="water",
            zone="silk_board",
            params={},
            start_t=0,
            duration_min=0,
        )


def test_make_envelope_uses_ist_demo_time() -> None:
    envelope = make_envelope("fix.plan", {}, "r_01", 1440, "silk_board")
    assert envelope.ts == "2026-11-18T00:00:00+05:30"


def httpx_error() -> Exception:
    return llm.httpx.ConnectError("provider unavailable")
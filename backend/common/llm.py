from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin
from urllib.parse import quote

import httpx
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

logger = logging.getLogger(__name__)
_UTC = UTC
_CACHE_LIMIT = 128
_CACHE: OrderedDict[str, str | BaseModel] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_STATE_LOCK = threading.RLock()
_PROVIDER_FAILURES: dict[str, int] = {}
_PROVIDER_OPEN_UNTIL: dict[str, float] = {}
_LAST_PROVIDER = "template"


def _load_dotenv() -> None:
	"""Load simple KEY=VALUE pairs from the repository .env without overwriting env vars."""
	env_path = Path(__file__).resolve().parents[2] / ".env"
	try:
		lines = env_path.read_text(encoding="utf-8").splitlines()
	except OSError:
		return

	for line in lines:
		stripped = line.strip()
		if not stripped or stripped.startswith("#") or "=" not in stripped:
			continue
		key, value = stripped.split("=", 1)
		key = key.strip()
		value = value.strip()
		if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
			value = value[1:-1]
		if key:
			os.environ.setdefault(key, value)


_load_dotenv()


def last_provider() -> str:
	"""Return the provider that answered the most recent uncached request."""
	with _STATE_LOCK:
		return _LAST_PROVIDER


def _set_last_provider(provider: str) -> None:
	global _LAST_PROVIDER
	with _STATE_LOCK:
		_LAST_PROVIDER = provider


def _cache_key(prompt: str, schema: type[BaseModel] | None) -> str:
	schema_json = "" if schema is None else json.dumps(schema.model_json_schema(), sort_keys=True)
	return hashlib.sha256(f"{prompt}\0{schema_json}".encode()).hexdigest()


def _schema_prompt(prompt: str, schema: type[BaseModel] | None) -> str:
	if schema is None:
		return prompt
	schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
	return f"{prompt}\n\nReturn only JSON matching this JSON schema:\n{schema_json}"


def _parse_json_text(value: str) -> Any:
	cleaned = value.strip()
	if cleaned.startswith("```"):
		cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
	try:
		return json.loads(cleaned)
	except json.JSONDecodeError:
		start = cleaned.find("{")
		end = cleaned.rfind("}")
		if start >= 0 and end > start:
			return json.loads(cleaned[start : end + 1])
		raise


def _validated_result(value: Any, schema: type[BaseModel] | None) -> str | BaseModel:
	if schema is None:
		if isinstance(value, str):
			return value or "No language model response is available."
		return json.dumps(value, sort_keys=True, ensure_ascii=True)
	if isinstance(value, str):
		value = _parse_json_text(value)
	return schema.model_validate(value)


def _ollama(prompt: str, schema: type[BaseModel] | None, timeout_s: float) -> Any:
	body: dict[str, Any] = {
		"model": os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
		"prompt": _schema_prompt(prompt, schema),
		"stream": False,
	}
	if schema is not None:
		body["format"] = "json"
	response = httpx.post("http://localhost:11434/api/generate", json=body, timeout=timeout_s)
	response.raise_for_status()
	return response.json()["response"]


def _gemini(prompt: str, schema: type[BaseModel] | None, timeout_s: float) -> Any:
	api_key = os.environ.get("GEMINI_API_KEY", "")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY is not configured")
	model = quote(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"), safe="")
	endpoint = (
		"https://generativelanguage.googleapis.com/v1beta/models/"
		f"{model}:generateContent?key={quote(api_key, safe='')}"
	)
	body: dict[str, Any] = {"contents": [{"parts": [{"text": _schema_prompt(prompt, schema)}]}]}
	if schema is not None:
		body["generationConfig"] = {"responseMimeType": "application/json"}
	response = httpx.post(endpoint, json=body, timeout=timeout_s)
	response.raise_for_status()
	return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def _provider_is_open(provider: str) -> bool:
	with _STATE_LOCK:
		return _PROVIDER_OPEN_UNTIL.get(provider, 0.0) > time.monotonic()


def _record_success(provider: str) -> None:
	with _STATE_LOCK:
		_PROVIDER_FAILURES[provider] = 0
		_PROVIDER_OPEN_UNTIL.pop(provider, None)


def _record_failure(provider: str) -> None:
	with _STATE_LOCK:
		failures = _PROVIDER_FAILURES.get(provider, 0) + 1
		_PROVIDER_FAILURES[provider] = failures
		if failures >= 2:
			_PROVIDER_OPEN_UNTIL[provider] = time.monotonic() + 60.0


def _fallback_value(annotation: Any) -> Any:
	origin = get_origin(annotation)
	args = get_args(annotation)
	if origin is Literal:
		return args[0] if args else None
	if origin in (Union, UnionType):
		non_none = [arg for arg in args if arg is not type(None)]
		return _fallback_value(non_none[0]) if non_none else None
	if origin in (list, tuple, set, frozenset):
		return [] if origin is list else origin()
	if origin is dict:
		return {}
	if isinstance(annotation, type) and issubclass(annotation, BaseModel):
		return _construct_fallback_model(annotation)
	if isinstance(annotation, type) and issubclass(annotation, Enum):
		return next(iter(annotation))
	if annotation is str:
		return "fallback"
	if annotation is bool:
		return False
	if annotation is int:
		return 0
	if annotation is float:
		return 0.0
	if annotation is datetime:
		return datetime(2026, 11, 17, tzinfo=_UTC)
	if annotation is date:
		return date(2026, 11, 17)
	if annotation is Any:
		return "fallback"
	return None


def _construct_fallback_model(schema: type[BaseModel]) -> BaseModel:
	values: dict[str, Any] = {}
	for name, field in schema.model_fields.items():
		if field.default is not PydanticUndefined or field.default_factory is not None:
			continue
		values[name] = _fallback_value(field.annotation)
	try:
		return schema.model_validate(values)
	except Exception:  # noqa: BLE001 - malformed model schemas must still get a constructed fallback.
		return schema.model_construct(**values)


def _template_result(
	template: Callable[[], str | dict[str, Any]] | None,
	prompt: str,
	schema: type[BaseModel] | None,
) -> str | BaseModel:
	try:
		raw: Any = template() if template is not None else (
			f"No language model response is available; request: {prompt[:160]}"
		)
		result = _validated_result(raw, schema)
		if isinstance(result, str) and not result:
			raise ValueError("Template produced an empty response")
		return result
	except Exception:  # noqa: BLE001 - templates are untrusted fallback callbacks.
		if schema is not None:
			return _construct_fallback_model(schema)
		return f"No language model response is available; request: {prompt[:160]}"


def _complete(
	prompt: str,
	schema: type[BaseModel] | None = None,
	*,
	template: Callable[[], str | dict[str, Any]] | None = None,
	timeout_s: float = 4.0,
) -> str | BaseModel:
	"""Complete a prompt, falling back deterministically; timeout_s is the network timeout in seconds."""
	try:
		key = _cache_key(prompt, schema)
	except Exception:  # noqa: BLE001 - schema introspection must not make completion fail.
		key = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()
	with _CACHE_LOCK:
		cached = _CACHE.get(key)
		if cached is not None:
			_CACHE.move_to_end(key)
			return cached

	result: str | BaseModel | None = None
	selected_provider = "template"
	force_fallback = os.environ.get("LLM_FORCE_FALLBACK", "") == "1"
	order = [
		provider.strip().lower()
		for provider in os.environ.get("LLM_PROVIDER_ORDER", "ollama,gemini,template").split(",")
		if provider.strip()
	]
	if "template" not in order:
		order.append("template")
	if force_fallback:
		order = ["template"]

	providers = {"ollama": _ollama, "gemini": _gemini}
	for provider in order:
		if provider == "template":
			result = _template_result(template, prompt, schema)
			selected_provider = "template"
			break
		provider_call = providers.get(provider)
		if provider_call is None or _provider_is_open(provider):
			continue
		if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
			continue
		try:
			raw_result = provider_call(prompt, schema, max(0.1, timeout_s))
			result = _validated_result(raw_result, schema)
			if isinstance(result, str) and not result:
				raise ValueError("Provider returned an empty response")
			_record_success(provider)
			selected_provider = provider
			break
		except Exception as error:  # noqa: BLE001 - providers may fail in many ways.
			_record_failure(provider)
			logger.warning("LLM provider %s failed (%s)", provider, type(error).__name__)

	if result is None:
		result = _template_result(template, prompt, schema)
	if isinstance(result, str) and not result:
		result = "No language model response is available."
	_set_last_provider(selected_provider)
	logger.info("LLM response provider=%s", selected_provider)
	with _CACHE_LOCK:
		_CACHE[key] = result
		_CACHE.move_to_end(key)
		while len(_CACHE) > _CACHE_LIMIT:
			_CACHE.popitem(last=False)
	return result


def complete(
	prompt: str,
	schema: type[BaseModel] | None = None,
	*,
	template: Callable[[], str | dict[str, Any]] | None = None,
	timeout_s: float = 4.0,
) -> str | BaseModel:
	"""Complete a prompt without raising; timeout_s is the network timeout in seconds."""
	try:
		return _complete(prompt, schema, template=template, timeout_s=timeout_s)
	except Exception as error:  # noqa: BLE001 - the public contract explicitly never raises.
		logger.error("LLM completion failed (%s)", type(error).__name__)
		if schema is not None:
			try:
				result = _construct_fallback_model(schema)
			except Exception:  # noqa: BLE001 - last resort keeps valid BaseModel schemas constructible.
				result = schema.model_construct()
			_set_last_provider("template")
			return result
		_set_last_provider("template")
		return f"No language model response is available; request: {prompt[:160]}"

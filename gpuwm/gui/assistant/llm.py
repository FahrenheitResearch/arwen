"""A small client for a chat-completions endpoint, the /v1 API LM Studio, Ollama and llama.cpp serve (standard library only).

llama.cpp's ``llama-server``, LM Studio and Ollama all answer
``POST {base}/chat/completions``.  The assistant needs three things from
it: tool calls, output held to a JSON schema, and the log probabilities
of the tokens it chose (the typed decisions read their probabilities
from those).  A server that cannot give log probabilities still works;
its decisions are shown without a measured probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any
import urllib.error
import urllib.request

DEFAULT_TIMEOUT_S = 300.0


class LlmError(Exception):
    """The endpoint could not be reached or answered with an error; the message says which."""


@dataclass
class Completion:
    message: dict[str, Any]
    logprobs: list[dict[str, Any]] | None
    finish: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_per_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.message.get("content") or "")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(self.message.get("tool_calls") or [])


def _post(url: str, body: dict[str, Any], *, key: str | None, timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise LlmError(f"{url} answered {error.code}: {detail}") from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise LlmError(f"Could not reach {url}: {getattr(error, 'reason', error)}") from error
    except ValueError as error:
        raise LlmError(f"{url} did not answer with JSON.") from error


def get_json(url: str, *, key: str | None = None, timeout: float = 3.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LlmError(f"{url} answered {error.code}.") from error
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as error:
        raise LlmError(f"Could not reach {url}.") from error


class Chat:
    """One endpoint and one model name."""

    def __init__(self, base_url: str, model: str = "local", *, key: str | None = None,
                 extra: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.extra = dict(extra or {})
        self.timeout = timeout

    def models(self) -> list[str]:
        document = get_json(f"{self.base_url}/models", key=self.key)
        return [str(row.get("id")) for row in document.get("data") or [] if row.get("id")]

    def complete(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None,
                 schema: dict[str, Any] | None = None, top_logprobs: int = 0, max_tokens: int = 700,
                 temperature: float = 0.2) -> Completion:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature, "stream": False, **self.extra}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(top_logprobs)
        started = time.perf_counter()
        document = _post(f"{self.base_url}/chat/completions", body, key=self.key, timeout=self.timeout)
        seconds = time.perf_counter() - started
        choices = document.get("choices") or []
        if not choices:
            raise LlmError("The model answered with no choices.")
        first = choices[0]
        usage = document.get("usage") or {}
        timings = document.get("timings") or {}
        output_tokens = int(usage.get("completion_tokens") or 0)
        rate = timings.get("predicted_per_second")
        if rate is None and output_tokens and seconds > 0:
            rate = output_tokens / seconds
        logprobs = (first.get("logprobs") or {}).get("content") if isinstance(first.get("logprobs"), dict) else None
        return Completion(message=dict(first.get("message") or {}), logprobs=logprobs,
                          finish=str(first.get("finish_reason") or ""), seconds=seconds,
                          prompt_tokens=int(usage.get("prompt_tokens") or 0), output_tokens=output_tokens,
                          tokens_per_s=None if rate is None else round(float(rate), 1), raw=document)


def parse_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """A tool call's arguments as a dict; malformed arguments become {}."""

    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = ["Chat", "Completion", "LlmError", "get_json", "parse_arguments"]

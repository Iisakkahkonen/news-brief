"""Gemini API client (Google's Interactions endpoint), standard library only.

Written from Google's current API reference (checked 2026-11-05):

    POST https://generativelanguage.googleapis.com/v1beta/interactions
    header  x-goog-api-key: <key>
    body    {"model": ..., "input": <prompt>, "store": false, "generation_config": {...}}
    reply   {"status": "completed", "steps": [{"type": "model_output",
             "content": [{"type": "text", "text": ...}]}], ...}
    errors  HTTP status with {"error": {"code": ..., "message": ...}}

Rules this client keeps:

* The key goes in a header, never in the URL, and is scrubbed from any error text.
* ``store`` is false so Google does not keep the prompt.
* Calls are spaced out. When a model is overloaded the next model in the list is
  tried straight away (each model has its own capacity); if every model fails,
  the client waits once and goes through the list one more time, then gives up.
  A daily-quota error, or any 401/402/403, ends the run at once
  (ModelUnavailable) because waiting will not fix it. A hard cap on HTTP
  requests per client keeps a run inside the free tier whatever happens.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence

from news_brief.model import ModelError, ModelUnavailable

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
# All four are listed as free of charge on Google's pricing page (2026-11-05).
# Order: best quality first; the lighter Flash-Lite models are the last resort.
DEFAULT_MODELS = ("gemini-3.6-flash", "gemini-3.8-flash",
                  "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
DEFAULT_MAX_REQUESTS = 60
MAX_ERROR_TEXT = 300


class _Retryable(Exception):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class GeminiClient:
    def __init__(self, api_key: str, model: str | Sequence[str] = DEFAULT_MODELS, *,
                 timeout: float = 60.0, min_interval: float = 4.0, retry_wait: float = 30.0,
                 max_output_tokens: int = 4096, max_requests: int = DEFAULT_MAX_REQUESTS,
                 notify: Callable[[str], None] | None = None) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("a Gemini API key is required")
        self._api_key = api_key.strip()
        names = model.split(",") if isinstance(model, str) else list(model)
        self.models = [name.strip() for name in names if name.strip()]
        if not self.models:
            raise ValueError("at least one model name is required")
        self._max_requests = max_requests
        self.usage: Counter[str] = Counter()  # successful answers per model
        self._timeout = timeout
        self._min_interval = min_interval
        self._retry_wait = retry_wait
        self._max_output_tokens = max_output_tokens
        self._last_call: float | None = None
        self._notify = notify or (lambda message: None)
        self.requests_made = 0  # HTTP requests sent, retries included

    # -- public ---------------------------------------------------------------

    @property
    def model(self) -> str:
        return self.models[0]

    def complete(self, prompt: str) -> str:
        self._pace()
        last: _Retryable | None = None
        for round_number in (1, 2):
            for index, name in enumerate(self.models):
                try:
                    text = self._once(prompt, name)
                except _Retryable as exc:
                    last = exc
                    if index + 1 < len(self.models):
                        self._notify(f"    {name} replied: {exc.message}\n"
                                     f"    trying {self.models[index + 1]} instead")
                    continue
                self.usage[name] += 1
                return text
            if round_number == 1:
                self._notify(f"    no model could answer ({last.message})\n"
                             f"    waiting {self._retry_wait:.0f}s, then trying once more")
                time.sleep(self._retry_wait)
        if last.status == 429:
            raise ModelUnavailable(f"HTTP 429 twice: {last.message}")
        raise ModelError(f"{last.message} (after one retry)")

    # -- internals ------------------------------------------------------------

    def _pace(self) -> None:
        if self._last_call is not None:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)

    def _clean(self, text: str) -> str:
        return text.replace(self._api_key, "[key removed]")[:MAX_ERROR_TEXT]

    def _once(self, prompt: str, model: str) -> str:
        if self.requests_made >= self._max_requests:
            raise ModelUnavailable(
                f"request cap of {self._max_requests} reached for this run")
        body = json.dumps({
            "model": model,
            "input": prompt,
            "store": False,
            "generation_config": {
                "max_output_tokens": self._max_output_tokens,
                "thinking_level": "low",
            },
        }).encode("utf-8")
        request = urllib.request.Request(
            ENDPOINT, data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
        )
        self._last_call = time.monotonic()
        self.requests_made += 1
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            self._raise_for_http_error(exc)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _Retryable(None, self._clean(f"network error: {exc}")) from None
        return self._extract_text(raw)

    def _raise_for_http_error(self, exc: urllib.error.HTTPError) -> None:
        code, message = "", ""
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace"))
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            code, message = str(error.get("code", "")), str(error.get("message", ""))
        except (ValueError, AttributeError, OSError):
            pass
        detail = self._clean(f"HTTP {exc.code} {code}: {message}".strip())

        if exc.code == 429:
            if code == "quota_exceeded":
                raise ModelUnavailable(f"daily quota used up - {detail}") from None
            raise _Retryable(429, detail) from None
        if exc.code in (401, 402, 403):
            hint = ""
            if exc.code == 403:
                hint = (" (a 403 can mean this project is not eligible for the free tier "
                        "or the key is blocked)")
            raise ModelUnavailable(detail + hint) from None
        if exc.code == 408 or exc.code >= 500:
            raise _Retryable(exc.code, detail) from None
        raise ModelError(detail) from None

    def _extract_text(self, raw: bytes) -> str:
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ModelError("reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ModelError("reply was not a JSON object")

        status = data.get("status")
        if status != "completed":
            problems = data.get("errors")
            raise ModelError(self._clean(
                f"interaction status {status!r}"
                + (f", errors: {json.dumps(problems)}" if problems else "")))

        pieces = []
        for step in data.get("steps") or []:
            if isinstance(step, dict) and step.get("type") == "model_output":
                for part in step.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "text":
                        pieces.append(str(part.get("text", "")))
        text = "".join(pieces).strip()
        if not text:
            raise ModelError("reply contained no text; top-level keys: "
                             + ", ".join(sorted(data)))
        return text

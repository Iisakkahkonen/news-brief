"""The one small interface between the pipeline and any AI model.

The pipeline only ever calls ``complete(prompt) -> text``. A provider
(Gemini, Groq, OpenRouter, a fake in tests) is anything with that method, so
swapping providers never touches grouping, summarising or verification.
"""

from __future__ import annotations

from typing import Protocol


class ModelError(Exception):
    """The model could not be used for this run."""


class ModelUnavailable(ModelError):
    """The provider refused the call: quota used up, rate limited or payment
    required (HTTP 402/429). The run stops instead of retrying in a loop."""


class BudgetExceeded(ModelError):
    """The run reached its own cap on model calls."""


class ModelClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class BudgetedModel:
    """Wraps a model and refuses to make more than ``max_calls`` calls.

    This is the code-side guard that keeps a run inside a free tier even if
    the provider's own limits are unclear.
    """

    def __init__(self, inner: ModelClient, max_calls: int) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self._inner = inner
        self.max_calls = max_calls
        self.calls = 0

    @property
    def usage(self) -> dict[str, int]:
        """Answers per model name, when the wrapped client keeps that count."""
        return dict(getattr(self._inner, "usage", {}) or {})

    def complete(self, prompt: str) -> str:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget of {self.max_calls} reached")
        self.calls += 1
        return self._inner.complete(prompt)

"""Perplexity API client (paper-friendly slim version).

One method: ``ask(prompt, model="sonar-pro") -> str``. Returns the raw assistant
content. The orchestrator parses JSON itself so we don't lock callers into a
single response shape.

Auth: Bearer ``PERPLEXITY_API_KEY`` (or ``zeroday/perplexity`` secret with
``api_key``). 8s timeout, 2 retries on 5xx / transport errors. Any 4xx is
raised as :class:`PerplexityAuthError` (for 401/403) or
:class:`PerplexityError` (for everything else).

Reference: ``/Users/sri/Documents/GitHub/zeroday-trading/data/perplexity_client.py``.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zeroday_paper.secrets import perplexity_api_key

logger = structlog.get_logger(__name__)

PERPLEXITY_BASE = "https://api.perplexity.ai"
DEFAULT_MODEL = "sonar-pro"


class PerplexityError(RuntimeError):
    pass


class PerplexityAuthError(PerplexityError):
    pass


class PerplexityClient:
    """Minimal async chat-completions wrapper.

    Parameters
    ----------
    api_key:
        Override; otherwise read via ``perplexity_api_key()``.
    timeout:
        HTTP timeout (seconds). Default 8s — tight on purpose so the brief
        never blocks the EventBridge schedule.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 8.0):
        self._api_key = api_key or perplexity_api_key()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> PerplexityClient:
        self._client = httpx.AsyncClient(
            base_url=PERPLEXITY_BASE,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "zeroday-paper/0.1",
            },
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ask(self, prompt: str, *, model: str = DEFAULT_MODEL) -> str:
        """Submit ``prompt`` and return the assistant string content.

        The caller is responsible for parsing JSON when expected. We do not
        force ``response_format`` (Perplexity returns 400 for ``json_object``;
        the prompt itself should say "return JSON only").
        """
        assert self._client is not None, "use as async context manager"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException)
            ),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code in (401, 403):
                    raise PerplexityAuthError(
                        f"Perplexity rejected key: {resp.status_code}"
                    )
                if resp.status_code == 429:
                    # Rate-limited: surface immediately so the orchestrator can mark
                    # the section [UNAVAILABLE] rather than burning retries.
                    raise PerplexityError(f"Perplexity rate-limited (429): {resp.text[:200]}")
                resp.raise_for_status()
                envelope = resp.json()

        return _extract_content(envelope)


def _extract_content(envelope: dict[str, Any]) -> str:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PerplexityError(
            f"Perplexity envelope has no choices: {sorted(envelope.keys())}"
        )
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    if not isinstance(message, dict):
        raise PerplexityError("Perplexity envelope choice[0] missing 'message'")
    content = message.get("content")
    if not isinstance(content, str):
        raise PerplexityError(
            f"Perplexity envelope content is non-string: {type(content).__name__}"
        )
    return content

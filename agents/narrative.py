from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings, ContextStatus


# ---------------------------------------------------------------------------
# Pluggable LLM provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base for LLM providers used by the Narrative Agent.

    Subclass for OpenAI, Anthropic, Gemini, or any other provider.
    """

    @abstractmethod
    async def query(self, prompt: str) -> str:
        """Send a prompt and return the model's text response."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider (GPT-4o-mini, etc.).

    Reuses a single aiohttp.ClientSession for the lifetime of the provider
    to avoid repeated SSL/TLS handshake overhead.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        import aiohttp

        self._api_key = api_key
        self._model = model
        self._session: aiohttp.ClientSession | None = None
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _ensure_session(self) -> "aiohttp.ClientSession":
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session

    async def query(self, prompt: str) -> str:
        session = await self._ensure_session()
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0,
        }

        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
        ) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (claude-sonnet-4-20250514, claude-opus-4-20250514, etc.).

    Uses the Anthropic Messages REST API directly via aiohttp.
    """

    _URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import aiohttp

        self._api_key = api_key
        self._model = model
        self._session: aiohttp.ClientSession | None = None
        self._headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    async def _ensure_session(self) -> "aiohttp.ClientSession":
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session

    async def query(self, prompt: str) -> str:
        session = await self._ensure_session()
        payload = {
            "model": self._model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with session.post(self._URL, json=payload) as resp:
            data = await resp.json()
            return data["content"][0]["text"]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class GeminiProvider(LLMProvider):
    """Google Gemini provider via the REST generateContent API.

    Uses the v1beta endpoint with API key auth (no OAuth needed).
    """

    _URL_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}"
        ":generateContent?key={key}"
    )

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        import aiohttp

        self._api_key = api_key
        self._model = model
        self._session: aiohttp.ClientSession | None = None
        self._url = self._URL_TEMPLATE.format(model=self._model, key=self._api_key)

    async def _ensure_session(self) -> "aiohttp.ClientSession":
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def query(self, prompt: str) -> str:
        session = await self._ensure_session()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 2048,
                "temperature": 0.0,
            },
        }

        async with session.post(self._url, json=payload) as resp:
            data = await resp.json()
            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                reason = candidate.get("finishReason", "unknown")
                raise ValueError(f"Gemini returned no content (finishReason={reason})")
            return parts[0]["text"]

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Narrative Agent
# ---------------------------------------------------------------------------

class NarrativeAgent(BaseAgent):
    """Out-of-band context monitor. Runs independently, never in the trade path.

    Continuously monitors active games and queries the LLM for critical
    negative context (injuries, ejections, technical fouls).  Writes the
    result to the Redis context cache:

        game:context:{game_id} = "SAFE"  (TTL 5min)
        game:context:{game_id} = "VETO:{reason}"  (TTL 5min)

    The Quant Engine reads this cache synchronously (microsecond redis.get)
    so no LLM latency ever enters the hot path.
    """

    # Focused prompt template — kept tight so the LLM responds fast
    _PROMPT_TEMPLATE = (
        "You are an NBA game context monitor for an automated trading system. "
        "For the game {home} vs {away} (currently Q{quarter}, {clock}):\n\n"
        "Answer ONLY 'SAFE' unless you have SPECIFIC knowledge of a critical "
        "negative event that happened in THIS game — such as a star player "
        "injury, ejection, or technical foul on a key player.\n\n"
        "If you do not have specific information about a negative event, "
        "answer 'SAFE'. Do NOT speculate. Do NOT veto based on score or "
        "general uncertainty.\n\n"
        "Your answer (SAFE or VETO: <reason>):"
    )

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
        llm: LLMProvider | None = None,
    ) -> None:
        super().__init__("narrative", settings, bus, client)
        self._llm = llm or self._default_llm(settings)
        self._active_games: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _default_llm(settings: AppSettings) -> LLMProvider:
        provider = settings.LLM_PROVIDER.lower()
        if provider == "anthropic":
            return AnthropicProvider(
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.ANTHROPIC_MODEL,
            )
        if provider == "gemini":
            return GeminiProvider(
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
            )
        return OpenAIProvider(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        game_listener = asyncio.create_task(self._track_games())
        context_loop = asyncio.create_task(self._context_loop())
        await asyncio.gather(game_listener, context_loop)

    async def _track_games(self) -> None:
        """Subscribe to game:state to maintain the set of active games."""
        await self.bus.subscribe(["game:state"], self._on_game_state)

    async def _on_game_state(self, _channel: str, data: dict) -> None:
        game_id = data.get("game_id", "")
        if game_id:
            self._active_games[game_id] = data

    async def _context_loop(self) -> None:
        """Periodically query the LLM for each active game and update the cache."""
        while self._running:
            if self._active_games:
                self.log.info(
                    "Evaluating context for {} active game(s)",
                    len(self._active_games),
                )
            for game_id, game in list(self._active_games.items()):
                await self._evaluate_context(game_id, game)
            await asyncio.sleep(self.settings.CONTEXT_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    async def _evaluate_context(self, game_id: str, game: dict) -> None:
        prompt = self._PROMPT_TEMPLATE.format(
            home=game.get("home_team", "?"),
            away=game.get("away_team", "?"),
            quarter=game.get("quarter", "?"),
            clock=game.get("clock", "?"),
        )

        try:
            response = await self._llm.query(prompt)
            response = response.strip()
        except Exception:
            self.log.exception("LLM query failed for game {}", game_id)
            await self.bus.set_context(
                game_id,
                ContextStatus.VETO,
                "LLM query failed – fail-close",
                ttl=self.settings.CONTEXT_TTL,
            )
            return

        self.log.info("LLM response for game {}: '{}'", game_id, response[:120])

        if response.upper().startswith("SAFE"):
            await self.bus.set_context(
                game_id, ContextStatus.SAFE, ttl=self.settings.CONTEXT_TTL
            )
            self.log.info("Context SAFE for game {}", game_id)
        else:
            reason = response[5:].strip() if response.upper().startswith("VETO") else response
            await self.bus.set_context(
                game_id, ContextStatus.VETO, reason, ttl=self.settings.CONTEXT_TTL
            )
            self.log.warning("Context VETO for game {}: {}", game_id, reason)

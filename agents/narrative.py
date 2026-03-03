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
        "You are a real-time NBA game analyst. For the game {home} vs {away} "
        "(currently Q{quarter}, {clock}), answer ONLY with 'SAFE' if there is "
        "no critical negative context, or 'VETO: <reason>' if there is.\n\n"
        "Critical negative context includes: star player injuries, ejections, "
        "technical fouls on key players, or any event that would drastically "
        "change the expected outcome.\n\n"
        "Your answer (one word or one short sentence):"
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
        return OpenAIProvider(api_key=settings.LLM_API_KEY, model=settings.LLM_MODEL)

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

        if response.upper().startswith("SAFE"):
            await self.bus.set_context(
                game_id, ContextStatus.SAFE, ttl=self.settings.CONTEXT_TTL
            )
            self.log.debug("Context SAFE for game {}", game_id)
        else:
            reason = response[5:].strip() if response.upper().startswith("VETO") else response
            await self.bus.set_context(
                game_id, ContextStatus.VETO, reason, ttl=self.settings.CONTEXT_TTL
            )
            self.log.warning("Context VETO for game {}: {}", game_id, reason)

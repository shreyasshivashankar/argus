from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import aiohttp
from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings, ContextStatus

_BDL_PLAYS_URL = "https://api.balldontlie.io/nba/v1/plays"
_RECENT_PLAYS_LIMIT = 20


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
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"OpenAI API HTTP {resp.status}: {error_text}")
            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError(f"OpenAI returned no choices. Raw: {data}")
            return choices[0]["message"]["content"]

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
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Anthropic API HTTP {resp.status}: {error_text}")
            data = await resp.json()
            content = data.get("content", [])
            if not content:
                stop = data.get("stop_reason", "unknown")
                raise ValueError(f"Anthropic returned no content (stop_reason={stop})")
            return content[0]["text"]

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
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Gemini API HTTP {resp.status}: {error_text}")

            data = await resp.json()

            if "promptFeedback" in data and "blockReason" in data["promptFeedback"]:
                reason = data["promptFeedback"]["blockReason"]
                raise ValueError(f"Gemini safety block triggered: {reason}")

            candidate = data.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                reason = candidate.get("finishReason", "unknown")
                raise ValueError(
                    f"Gemini returned no text (finishReason={reason}). Raw: {data}"
                )
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

    # Focused prompt template — play-by-play gives LLM real data to parse
    _PROMPT_TEMPLATE = (
        "You are an NBA game context monitor for an automated trading system. "
        "For the game {home} vs {away} (currently Q{quarter}, {clock}):\n\n"
        "Here are the most recent play-by-play events:\n"
        "{recent_plays}\n\n"
        "Answer ONLY 'SAFE' unless the play-by-play indicates a critical "
        "negative event in THIS game — such as a star player injury, ejection, "
        "or technical foul on a key player.\n\n"
        "If the plays do not show a negative event, answer 'SAFE'. Do NOT "
        "speculate. Do NOT veto based on score or general uncertainty.\n\n"
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
        self._last_llm_ok: bool = True
        self._http_session: aiohttp.ClientSession | None = None

    def _heartbeat_payload(self) -> dict:
        return {**super()._heartbeat_payload(), "api_ok": self._last_llm_ok}

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
            tasks = [
                self._evaluate_context(game_id, game)
                for game_id, game in self._active_games.items()
            ]
            await asyncio.gather(*tasks)
            await asyncio.sleep(self.settings.CONTEXT_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    async def _fetch_plays(self, game_id: str) -> list[str]:
        """Fetch freshest play-by-play right before LLM call (avoids stale data)."""
        if not self.settings.BALLDONTLIE_API_KEY:
            return []
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                headers={"Authorization": self.settings.BALLDONTLIE_API_KEY},
                timeout=aiohttp.ClientTimeout(total=15),
            )
        try:
            async with self._http_session.get(
                _BDL_PLAYS_URL, params={"game_id": game_id}
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception:
            self.log.debug("Plays fetch failed for game {}", game_id)
            return []

        items = data.get("data", []) or []
        texts: list[str] = []
        for p in sorted(items, key=lambda x: x.get("order", 0), reverse=True)[
            :_RECENT_PLAYS_LIMIT
        ]:
            t = p.get("text", "").strip()
            if t:
                texts.append(t)
        return list(reversed(texts))

    async def _evaluate_context(self, game_id: str, game: dict) -> None:
        plays = await self._fetch_plays(game_id)
        recent_plays = "\n".join(plays) if plays else "No recent plays available."
        prompt = self._PROMPT_TEMPLATE.format(
            home=game.get("home_team", "?"),
            away=game.get("away_team", "?"),
            quarter=game.get("quarter", "?"),
            clock=game.get("clock", "?"),
            recent_plays=recent_plays,
        )

        try:
            response = await self._llm.query(prompt)
            response = response.strip()
            self._last_llm_ok = True
        except Exception:
            self._last_llm_ok = False
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

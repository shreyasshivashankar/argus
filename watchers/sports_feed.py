from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings, GameState


class SportsFeed(ABC):
    """Abstract base for live sports data providers.

    Subclass and implement connect() / listen() to integrate any provider.
    The watcher publishes GameState messages to the ``game:state`` Redis channel.
    """

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        self.settings = settings
        self.bus = bus
        self._running = True

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def listen(self) -> AsyncIterator[GameState]:
        """Yield GameState objects as they arrive from the provider."""
        yield  # type: ignore[misc]

    async def run(self) -> None:
        await self.connect()
        async for game_state in self.listen():
            await self.bus.publish("game:state", game_state)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# API-SPORTS WebSocket feed (production tier)
# ---------------------------------------------------------------------------

class APISportsFeed(SportsFeed):
    """WebSocket-based feed from API-SPORTS (api-sports.io).

    Connects to the provider's WSS endpoint and pushes score changes
    to the Redis bus as GameState objects. Replace the URL and message
    parsing with the actual API-SPORTS WebSocket contract.
    """

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        super().__init__(settings, bus)
        self._ws_url = settings.SPORTS_API_WS_URL
        self._api_key = settings.SPORTS_API_KEY
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        headers = {"x-apisports-key": self._api_key}
        self._ws = await self._session.ws_connect(self._ws_url, headers=headers)
        logger.info("Connected to API-SPORTS WebSocket: {}", self._ws_url)

    async def listen(self) -> AsyncIterator[GameState]:
        assert self._ws is not None
        async for msg in self._ws:
            if not self._running:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    game_state = self._parse(msg.json())
                    if game_state:
                        yield game_state
                except Exception:
                    logger.exception("Failed to parse API-SPORTS message")
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                logger.warning("API-SPORTS WS closed/errored, reconnecting…")
                await self._reconnect()

    async def _reconnect(self) -> None:
        await asyncio.sleep(2)
        await self.connect()

    @staticmethod
    def _parse(data: dict) -> GameState | None:
        """Parse a raw API-SPORTS WS message into a GameState.

        This is a template; adapt field names to the actual API-SPORTS
        WebSocket payload schema once the subscription is active.
        """
        try:
            return GameState(
                game_id=str(data["id"]),
                home_team=data["teams"]["home"]["name"],
                away_team=data["teams"]["away"]["name"],
                home_score=data["scores"]["home"]["total"],
                away_score=data["scores"]["away"]["total"],
                quarter=data.get("periods", {}).get("current", 0),
                clock=data.get("status", {}).get("clock", "0:00"),
                timestamp=datetime.utcnow(),
            )
        except (KeyError, TypeError):
            return None

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()


# ---------------------------------------------------------------------------
# Balldontlie REST feed (paper trading / research)
# ---------------------------------------------------------------------------

_LIVE_STATUSES = frozenset({
    "1st Qtr", "2nd Qtr", "3rd Qtr", "4th Qtr", "Halftime",
    "OT", "1OT", "2OT", "3OT",
})


class BalldontlieFeed(SportsFeed):
    """REST polling feed from Balldontlie (api.balldontlie.io).

    Suitable for paper trading and research. Polls today's games and only
    emits live in-progress games. Free tier allows 5 req/min so the default
    poll interval is 15 seconds.

    Do NOT use for live trading — REST polling adds unacceptable latency
    compared to a WebSocket feed like Sportradar.
    """

    _BASE_URL = "https://api.balldontlie.io/v1"

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
    ) -> None:
        super().__init__(settings, bus)
        self._poll_interval = settings.SPORTS_POLL_INTERVAL
        self._api_key = settings.BALLDONTLIE_API_KEY
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": self._api_key},
        )
        logger.info(
            "BalldontlieFeed connected (poll every {}s) — paper/research only",
            self._poll_interval,
        )

    async def listen(self) -> AsyncIterator[GameState]:
        assert self._session is not None
        while self._running:
            try:
                today = datetime.utcnow().strftime("%Y-%m-%d")
                url = f"{self._BASE_URL}/games?dates[]={today}&per_page=100"
                async with self._session.get(url) as resp:
                    if resp.status == 401:
                        logger.error("Balldontlie 401 — check BALLDONTLIE_API_KEY")
                        await asyncio.sleep(self._poll_interval)
                        continue
                    if resp.status == 429:
                        logger.warning("Balldontlie rate limited, backing off")
                        await asyncio.sleep(60)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    live_count = 0
                    for game in data.get("data", []):
                        if game.get("status") not in _LIVE_STATUSES:
                            continue
                        gs = self._parse(game)
                        if gs:
                            live_count += 1
                            yield gs
                    if live_count:
                        logger.debug("Balldontlie: {} live games", live_count)
            except aiohttp.ClientError:
                logger.exception("Balldontlie poll failed (network)")
            except Exception:
                logger.exception("Balldontlie poll failed")
            await asyncio.sleep(self._poll_interval)

    @staticmethod
    def _parse(game: dict) -> GameState | None:
        try:
            status = game.get("status", "")
            clock = game.get("time", "") or "0:00"
            quarter = game.get("period", 0)

            return GameState(
                game_id=str(game["id"]),
                home_team=game["home_team"]["full_name"],
                away_team=game["visitor_team"]["full_name"],
                home_score=game.get("home_team_score", 0),
                away_score=game.get("visitor_team_score", 0),
                quarter=quarter,
                clock=clock if clock.strip() else status,
                timestamp=datetime.utcnow(),
            )
        except (KeyError, TypeError):
            return None

    async def close(self) -> None:
        if self._session:
            await self._session.close()

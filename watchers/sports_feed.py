from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import AsyncIterator

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings, GameState, PlayerBoxScore


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
    emits live in-progress games.

    Tier support:
      - free: game scores only (5 req/min)
      - all-star: adds per-player stats via /v1/stats ($9.99/mo, 60 req/min)
      - goat: adds live box scores, odds, player props ($39.99/mo, 600 req/min)

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
        self._tier = settings.BALLDONTLIE_TIER.lower()
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": self._api_key},
        )
        logger.info(
            "BalldontlieFeed connected (poll every {}s, tier={}) — paper/research only",
            self._poll_interval, self._tier,
        )

    async def listen(self) -> AsyncIterator[GameState]:
        assert self._session is not None
        while self._running:
            try:
                if self._tier == "goat":
                    async for gs in self._poll_live_box_scores():
                        yield gs
                else:
                    async for gs in self._poll_games():
                        yield gs
            except aiohttp.ClientError:
                logger.exception("Balldontlie poll failed (network)")
            except Exception:
                logger.exception("Balldontlie poll failed")
            await asyncio.sleep(self._poll_interval)

    async def _poll_games(self) -> AsyncIterator[GameState]:
        """Free / ALL-STAR path: poll /games + optionally /stats."""
        assert self._session is not None
        now = datetime.utcnow()
        today = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        url = (
            f"{self._BASE_URL}/games"
            f"?dates[]={yesterday}&dates[]={today}&per_page=100"
        )
        async with self._session.get(url) as resp:
            if resp.status == 401:
                logger.error("Balldontlie 401 — check BALLDONTLIE_API_KEY")
                return
            if resp.status == 429:
                logger.warning("Balldontlie rate limited, backing off")
                await asyncio.sleep(60)
                return
            resp.raise_for_status()
            data = await resp.json()

        games = data.get("data", [])
        live_games = [g for g in games if g.get("status") in _LIVE_STATUSES]

        stats_by_game: dict[str, list[PlayerBoxScore]] = {}
        if live_games and self._tier == "all-star":
            stats_by_game = await self._fetch_player_stats(
                [str(g["id"]) for g in live_games]
            )

        for game in live_games:
            game_id = str(game["id"])
            gs = self._parse(game, stats_by_game.get(game_id, []))
            if gs:
                yield gs

        statuses = {g.get("status", "?") for g in games}
        logger.info(
            "Balldontlie poll: {} total, {} live | statuses: {}",
            len(games), len(live_games), statuses or "none",
        )

    async def _poll_live_box_scores(self) -> AsyncIterator[GameState]:
        """GOAT tier path: single call to /box_scores/live returns games
        with embedded per-player stats — no second request needed.
        """
        assert self._session is not None
        url = f"{self._BASE_URL}/box_scores/live"
        async with self._session.get(url) as resp:
            if resp.status == 401:
                logger.error("Balldontlie 401 — check BALLDONTLIE_API_KEY")
                return
            if resp.status == 429:
                logger.warning("Balldontlie rate limited on /box_scores/live")
                await asyncio.sleep(30)
                return
            resp.raise_for_status()
            data = await resp.json()

        box_scores = data.get("data", [])
        live_count = 0
        for bs in box_scores:
            if bs.get("status") not in _LIVE_STATUSES:
                continue
            gs = self._parse_box_score(bs)
            if gs:
                live_count += 1
                yield gs

        logger.info(
            "Balldontlie /box_scores/live: {} games, {} live",
            len(box_scores), live_count,
        )

    async def _fetch_player_stats(
        self,
        game_ids: list[str],
    ) -> dict[str, list[PlayerBoxScore]]:
        """Fetch live player stats for a batch of game IDs.

        Uses the /v1/stats endpoint (ALL-STAR tier). Paginates with cursor
        to capture all players across all games in a single pass.
        """
        assert self._session is not None
        result: dict[str, list[PlayerBoxScore]] = {}
        ids_param = "&".join(f"game_ids[]={gid}" for gid in game_ids)
        cursor: int | None = None

        while True:
            url = f"{self._BASE_URL}/stats?{ids_param}&per_page=100"
            if cursor is not None:
                url += f"&cursor={cursor}"
            try:
                async with self._session.get(url) as resp:
                    if resp.status in (401, 403):
                        logger.warning(
                            "Balldontlie /stats returned {} — upgrade to ALL-STAR tier "
                            "for player box scores",
                            resp.status,
                        )
                        return result
                    if resp.status == 429:
                        logger.warning("Rate limited on /stats, skipping this cycle")
                        return result
                    resp.raise_for_status()
                    data = await resp.json()
            except Exception:
                logger.exception("Failed to fetch player stats")
                return result

            for entry in data.get("data", []):
                pbs = self._parse_player_stat(entry)
                if pbs:
                    gid = str(entry.get("game", {}).get("id", ""))
                    result.setdefault(gid, []).append(pbs)

            next_cursor = data.get("meta", {}).get("next_cursor")
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

        total = sum(len(v) for v in result.values())
        logger.debug("Fetched {} player stat lines across {} games", total, len(result))
        return result

    @staticmethod
    def _parse_player_stat(entry: dict) -> PlayerBoxScore | None:
        """Parse a single stats entry from /v1/stats into a PlayerBoxScore."""
        try:
            player = entry.get("player", {})
            team = entry.get("team", {})
            min_str = entry.get("min", "0") or "0"
            try:
                minutes = float(min_str.split(":")[0]) if ":" in min_str else float(min_str)
            except (ValueError, TypeError):
                minutes = 0.0

            return PlayerBoxScore(
                player_id=str(player.get("id", "")),
                first_name=player.get("first_name", ""),
                last_name=player.get("last_name", ""),
                team_abbr=team.get("abbreviation", ""),
                minutes=minutes,
                pts=entry.get("pts", 0) or 0,
                fgm=entry.get("fgm", 0) or 0,
                fga=entry.get("fga", 0) or 0,
                fg3m=entry.get("fg3m", 0) or 0,
                fg3a=entry.get("fg3a", 0) or 0,
                ftm=entry.get("ftm", 0) or 0,
                fta=entry.get("fta", 0) or 0,
                reb=entry.get("reb", 0) or 0,
                ast=entry.get("ast", 0) or 0,
                stl=entry.get("stl", 0) or 0,
                blk=entry.get("blk", 0) or 0,
                turnover=entry.get("turnover", 0) or 0,
                pf=entry.get("pf", 0) or 0,
                plus_minus=entry.get("plus_minus", 0) or 0,
            )
        except (KeyError, TypeError):
            return None

    @classmethod
    def _parse_box_score(cls, bs: dict) -> GameState | None:
        """Parse a /box_scores/live entry (GOAT tier).

        The response embeds player stats inside home_team.players and
        visitor_team.players, so we extract them inline.
        """
        try:
            home_team = bs.get("home_team", {})
            visitor_team = bs.get("visitor_team", {})
            home_abbr = home_team.get("abbreviation", "")
            away_abbr = visitor_team.get("abbreviation", "")

            players: list[PlayerBoxScore] = []
            for p_entry in home_team.get("players", []):
                pbs = cls._parse_embedded_player(p_entry, home_abbr)
                if pbs:
                    players.append(pbs)
            for p_entry in visitor_team.get("players", []):
                pbs = cls._parse_embedded_player(p_entry, away_abbr)
                if pbs:
                    players.append(pbs)

            clock = bs.get("time", "") or "0:00"
            status = bs.get("status", "")

            return GameState(
                game_id=str(bs.get("id", bs.get("game_id", ""))),
                home_team=home_team.get("full_name", ""),
                away_team=visitor_team.get("full_name", ""),
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                home_score=bs.get("home_team_score", 0),
                away_score=bs.get("visitor_team_score", 0),
                quarter=bs.get("period", 0),
                clock=clock if clock.strip() else status,
                timestamp=datetime.utcnow(),
                player_stats=players,
            )
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _parse_embedded_player(entry: dict, team_abbr: str) -> PlayerBoxScore | None:
        """Parse a single player from the box_scores/live embedded format."""
        try:
            player = entry.get("player", {})
            min_str = entry.get("min", "0") or "0"
            try:
                minutes = float(min_str.split(":")[0]) if ":" in min_str else float(min_str)
            except (ValueError, TypeError):
                minutes = 0.0

            return PlayerBoxScore(
                player_id=str(player.get("id", "")),
                first_name=player.get("first_name", ""),
                last_name=player.get("last_name", ""),
                team_abbr=team_abbr,
                minutes=minutes,
                pts=entry.get("pts", 0) or 0,
                fgm=entry.get("fgm", 0) or 0,
                fga=entry.get("fga", 0) or 0,
                fg3m=entry.get("fg3m", 0) or 0,
                fg3a=entry.get("fg3a", 0) or 0,
                ftm=entry.get("ftm", 0) or 0,
                fta=entry.get("fta", 0) or 0,
                reb=entry.get("reb", 0) or 0,
                ast=entry.get("ast", 0) or 0,
                stl=entry.get("stl", 0) or 0,
                blk=entry.get("blk", 0) or 0,
                turnover=entry.get("turnover", 0) or 0,
                pf=entry.get("pf", 0) or 0,
                plus_minus=entry.get("plus_minus", 0) or 0,
            )
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _parse(game: dict, player_stats: list[PlayerBoxScore] | None = None) -> GameState | None:
        try:
            status = game.get("status", "")
            clock = game.get("time", "") or "0:00"
            quarter = game.get("period", 0)

            return GameState(
                game_id=str(game["id"]),
                home_team=game["home_team"]["full_name"],
                away_team=game["visitor_team"]["full_name"],
                home_abbr=game["home_team"].get("abbreviation", ""),
                away_abbr=game["visitor_team"].get("abbreviation", ""),
                home_score=game.get("home_team_score", 0),
                away_score=game.get("visitor_team_score", 0),
                quarter=quarter,
                clock=clock if clock.strip() else status,
                timestamp=datetime.utcnow(),
                player_stats=player_stats or [],
            )
        except (KeyError, TypeError):
            return None

    async def close(self) -> None:
        if self._session:
            await self._session.close()

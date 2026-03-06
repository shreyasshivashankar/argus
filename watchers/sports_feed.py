from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings, GameState, PlayerBoxScore


class SportsFeed(ABC):
    """Abstract base for live sports data providers."""

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        self.settings = settings
        self.bus = bus
        self._running = True

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def listen(self) -> AsyncIterator[GameState]:
        yield  # type: ignore[misc]

    async def run(self) -> None:
        await self.connect()
        async for game_state in self.listen():
            await self.bus.publish("game:state", game_state)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# BallDontLie REST feed — GOAT tier 600 req/min, poll ~500/min
# ---------------------------------------------------------------------------

_BDL_BASE = "https://api.balldontlie.io/v1"
_BDL_LIVE_STATUS_PREFIXES = ("1st Qtr", "2nd Qtr", "Halftime", "3rd Qtr", "4th Qtr", "OT")


class BallDontLieFeed(SportsFeed):
    """NBA feed via BallDontLie REST. GOAT: 600 req/min. Polls aggressively under limit."""

    STATS_POLL_INTERVAL_MIN = 0.2  # GOAT: 600 req/min; .env can go down to 0.2

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        super().__init__(settings, bus)
        self._api_key = settings.BALLDONTLIE_API_KEY
        self._games_poll_interval = max(0.25, settings.SPORTS_GAMES_POLL_INTERVAL)
        self._stats_poll_interval = max(
            settings.SPORTS_POLL_INTERVAL, self.STATS_POLL_INTERVAL_MIN
        )
        self._session: aiohttp.ClientSession | None = None
        self._games_cache: dict[str, dict] = {}
        self._player_stats_cache: dict[str, list[PlayerBoxScore]] = {}
        self._last_api_ok: bool = True

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": self._api_key},
            timeout=aiohttp.ClientTimeout(total=30, sock_read=15),
        )
        await self._fetch_games()
        logger.info(
            "BallDontLieFeed connected ({} games, games poll {}s, stats poll {}s)",
            len(self._games_cache),
            self._games_poll_interval,
            self._stats_poll_interval,
        )

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await self.bus.publish(
                "signal:heartbeat",
                {"agent": "sports_feed", "api_ok": self._last_api_ok},
            )
            await asyncio.sleep(30)

    async def listen(self) -> AsyncIterator[GameState]:
        assert self._session is not None
        self._queue: asyncio.Queue[GameState] = asyncio.Queue()

        games_task = asyncio.create_task(self._games_poll_loop(), name="bdl_games")
        stats_task = asyncio.create_task(self._stats_poll_loop(), name="bdl_stats")
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="bdl_heartbeat")

        try:
            while self._running:
                gs = await self._queue.get()
                yield gs
        finally:
            games_task.cancel()
            stats_task.cancel()
            heartbeat_task.cancel()
            await asyncio.gather(games_task, stats_task, heartbeat_task, return_exceptions=True)

    def _is_live(self, status: str) -> bool:
        if not status or status == "Final":
            return False
        return any(status.startswith(p) for p in _BDL_LIVE_STATUS_PREFIXES)

    async def _fetch_games(self) -> None:
        assert self._session is not None
        us_date = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%d")
        url = f"{_BDL_BASE}/games?dates[]={us_date}&per_page=100"
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    logger.error("BallDontLie games HTTP {}: {}", resp.status, await resp.text())
                    self._last_api_ok = False
                    return
                data = await resp.json()
                self._last_api_ok = True
        except Exception:
            logger.exception("Failed to fetch BallDontLie games")
            self._last_api_ok = False
            return

        for g in data.get("data", []) or []:
            gid = str(g.get("id", ""))
            if gid:
                self._games_cache[gid] = g

    async def _games_poll_loop(self) -> None:
        while self._running:
            await self._fetch_games()
            for gid, g in list(self._games_cache.items()):
                if self._is_live(g.get("status", "")):
                    gs = self._game_to_state(g, gid)
                    if gs:
                        await self._queue.put(gs)
            await asyncio.sleep(self._games_poll_interval)

    async def _stats_poll_loop(self) -> None:
        while self._running:
            live_ids = [
                gid for gid, g in self._games_cache.items()
                if self._is_live(g.get("status", ""))
            ]
            if live_ids:
                await self._fetch_stats(live_ids)
                for gid in live_ids:
                    g = self._games_cache.get(gid)
                    if g:
                        gs = self._game_to_state(g, gid)
                        if gs:
                            await self._queue.put(gs)
            await asyncio.sleep(self._stats_poll_interval)

    async def _fetch_stats(self, game_ids: list[str]) -> None:
        assert self._session is not None
        params = "&".join(f"game_ids[]={gid}" for gid in game_ids[:15])
        by_game: dict[str, list[PlayerBoxScore]] = {}
        cursor: int | None = None

        while True:
            url = f"{_BDL_BASE}/stats?{params}&per_page=100"
            if cursor is not None:
                url += f"&cursor={cursor}"
            try:
                async with self._session.get(url) as resp:
                    if resp.status == 401:
                        self._last_api_ok = False
                        return
                    if resp.status != 200:
                        self._last_api_ok = False
                        return
                    data = await resp.json()
                    self._last_api_ok = True
            except Exception:
                self._last_api_ok = False
                return

            items = data.get("data", []) or []
            for entry in items:
                game = entry.get("game", {})
                gid = str(game.get("id", "")) if isinstance(game, dict) else ""
                if not gid:
                    continue
                pbs = self._parse_stat_entry(entry)
                if pbs:
                    by_game.setdefault(gid, []).append(pbs)

            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor")
            if next_cursor is None or len(items) < 100:
                break
            cursor = next_cursor

        for gid, players in by_game.items():
            self._player_stats_cache[gid] = players

    def _game_to_state(self, g: dict, gid: str) -> GameState | None:
        try:
            home = g.get("home_team") or {}
            away = g.get("visitor_team") or {}
            home_full = home.get("full_name", "")
            away_full = away.get("full_name", "")
            home_abbr = home.get("abbreviation", "")
            away_abbr = away.get("abbreviation", "")

            if not home_full or not away_full:
                return None

            period = int(g.get("period", 0) or 0)
            time_str = g.get("time", "") or ""
            clock = time_str if time_str and time_str != "Final" else f"Q{period}"

            return GameState(
                game_id=gid,
                home_team=home_full,
                away_team=away_full,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                home_score=int(g.get("home_team_score", 0) or 0),
                away_score=int(g.get("visitor_team_score", 0) or 0),
                quarter=period,
                clock=clock,
                timestamp=datetime.now(timezone.utc),
                player_stats=self._player_stats_cache.get(gid, []),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _parse_stat_entry(self, entry: dict) -> PlayerBoxScore | None:
        try:
            player = entry.get("player", {}) or {}
            team = entry.get("team", {}) or {}
            team_abbr = team.get("abbreviation", "") if isinstance(team, dict) else ""

            def _min_to_float(s: str | int | float) -> float:
                if s is None:
                    return 0.0
                if isinstance(s, (int, float)):
                    return float(s)
                try:
                    return float(str(s).strip())
                except ValueError:
                    return 0.0

            return PlayerBoxScore(
                player_id=str(player.get("id", "")),
                first_name=player.get("first_name", ""),
                last_name=player.get("last_name", ""),
                team_abbr=team_abbr,
                minutes=_min_to_float(entry.get("min")),
                pts=int(entry.get("pts", 0) or 0),
                fgm=int(entry.get("fgm", 0) or 0),
                fga=int(entry.get("fga", 0) or 0),
                fg3m=int(entry.get("fg3m", 0) or 0),
                fg3a=int(entry.get("fg3a", 0) or 0),
                ftm=int(entry.get("ftm", 0) or 0),
                fta=int(entry.get("fta", 0) or 0),
                reb=int(entry.get("reb", 0) or 0),
                ast=int(entry.get("ast", 0) or 0),
                stl=int(entry.get("stl", 0) or 0),
                blk=int(entry.get("blk", 0) or 0),
                turnover=int(entry.get("turnover", 0) or 0),
                pf=int(entry.get("pf", 0) or 0),
                plus_minus=int(entry.get("plus_minus", 0) or 0),
            )
        except (KeyError, TypeError):
            return None

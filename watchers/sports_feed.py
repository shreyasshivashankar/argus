from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import AsyncIterator

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings, GameState, PlayerBoxScore

_NBA_SPORT_ID = 4

_LIVE_STATUSES = frozenset({
    "STATUS_IN_PROGRESS",
    "STATUS_HALFTIME",
    "STATUS_END_PERIOD",
    "STATUS_END_OF_REGULATION",
    "STATUS_OVERTIME",
    "STATUS_FIRST_HALF",
    "STATUS_SECOND_HALF",
})

_NBA_STAT_ABBR_MAP = {
    "PTS": "pts", "AST": "ast", "REB": "reb", "STL": "stl",
    "BLK": "blk", "TO": "turnover", "TOV": "turnover",
    "PF": "pf", "FGM": "fgm", "FGA": "fga",
    "3PM": "fg3m", "3PA": "fg3a", "FTM": "ftm", "FTA": "fta",
    "MIN": "minutes", "+/-": "plus_minus",
    "OREB": "oreb", "DREB": "dreb",
}


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
        yield  # type: ignore[misc]

    async def run(self) -> None:
        await self.connect()
        async for game_state in self.listen():
            await self.bus.publish("game:state", game_state)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# TheRundown WebSocket + REST stats feed (Ultra tier)
# ---------------------------------------------------------------------------

_TR_WS_URL = "wss://therundown.io/api/v1/ws"
_TR_REST_BASE = "https://therundown.io/api/v2"


class TheRundownFeed(SportsFeed):
    """Real-time NBA feed via TheRundown V1 WebSocket + V2 REST player stats.

    Architecture:
      - **Primary (scores)**: V1 WebSocket streams event updates with scores,
        game_period, display_clock, and event_status in real-time. WebSocket
        connections do NOT count against the REST rate limit.
      - **Supplement (player stats)**: REST polling of
        ``GET /api/v2/events/{eventID}/players/stats`` for live player box
        scores. Polls at ``SPORTS_POLL_INTERVAL`` (default 30s, min 10s)
        per game. Each call returns ~450 data points (30 players x 15 stats),
        so aggressive polling burns the monthly quota fast.
      - **Fallback**: If the WebSocket disconnects, falls back to REST event
        polling until reconnection.

    Rate limit budget (Ultra tier):
      The tier's REST rate limit is checked via response headers. The feed
      tracks ``X-RateLimit-Remaining`` and backs off before hitting the cap.
      WebSocket pushes are free.
    """

    WS_RECONNECT_DELAY = 3
    WS_MAX_RECONNECT_DELAY = 60
    STATS_POLL_INTERVAL_MIN = 10.0
    REST_BUDGET_FLOOR = 20

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        super().__init__(settings, bus)
        self._api_key = settings.THERUNDOWN_API_KEY
        self._stats_poll_interval = max(
            settings.SPORTS_POLL_INTERVAL, self.STATS_POLL_INTERVAL_MIN
        )
        self._session: aiohttp.ClientSession | None = None

        self._event_cache: dict[str, dict] = {}
        self._player_stats_cache: dict[str, list[PlayerBoxScore]] = {}

        self._rate_remaining: int | None = None
        self._rate_limit: int | None = None
        self._team_id_to_abbr: dict[int, str] = {}

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"X-TheRundown-Key": self._api_key},
            timeout=aiohttp.ClientTimeout(total=60, sock_read=30),
        )
        await self._resolve_team_abbrs()
        await self._bootstrap_todays_events()
        logger.info(
            "TheRundownFeed connected ({} events today, {} teams, stats poll {}s)",
            len(self._event_cache),
            len(self._team_id_to_abbr),
            self._stats_poll_interval,
        )

    async def _heartbeat_loop(self) -> None:
        """Publish periodic heartbeat so the monitor knows the feed is alive."""
        while self._running:
            await self.bus.publish("signal:heartbeat", {"agent": "sports_feed"})
            await asyncio.sleep(30)

    async def listen(self) -> AsyncIterator[GameState]:
        assert self._session is not None

        stats_task = asyncio.create_task(
            self._player_stats_poll_loop(), name="tr_stats_poll"
        )
        event_refresh_task = asyncio.create_task(
            self._event_refresh_loop(), name="tr_event_refresh"
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="tr_heartbeat"
        )

        reconnect_delay = self.WS_RECONNECT_DELAY
        while self._running:
            try:
                async for gs in self._ws_stream():
                    yield gs
                    reconnect_delay = self.WS_RECONNECT_DELAY
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("TheRundown WS stream failed")

            if not self._running:
                break

            logger.warning(
                "WS disconnected, reconnecting in {}s (falling back to REST)",
                reconnect_delay,
            )
            try:
                async for gs in self._rest_fallback(reconnect_delay):
                    yield gs
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("REST fallback failed")

            reconnect_delay = min(
                reconnect_delay * 2, self.WS_MAX_RECONNECT_DELAY
            )

        stats_task.cancel()
        event_refresh_task.cancel()
        heartbeat_task.cancel()
        await asyncio.gather(
            stats_task, event_refresh_task, heartbeat_task,
            return_exceptions=True,
        )

    # ------------------------------------------------------------------
    # WebSocket stream
    # ------------------------------------------------------------------

    async def _ws_stream(self) -> AsyncIterator[GameState]:
        """Connect to the V1 WebSocket and yield GameState on every event update."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ws_url = (
            f"{_TR_WS_URL}?key={self._api_key}"
            f"&sport_ids={_NBA_SPORT_ID}"
            f"&date={today}"
        )
        logger.info("Opening TheRundown V1 WebSocket: sport_ids={}", _NBA_SPORT_ID)

        async with self._session.ws_connect(  # type: ignore[union-attr]
            ws_url,
            heartbeat=20,
            receive_timeout=45,
        ) as ws:
            logger.info("TheRundown WebSocket connected")
            # Yield any live events from cache immediately (WS may not push until score changes)
            for eid, ev in list(self._event_cache.items()):
                if ev.get("score", {}).get("event_status") in _LIVE_STATUSES:
                    gs = self._event_to_game_state(ev, eid)
                    if gs:
                        yield gs
            async for msg in ws:
                if not self._running:
                    break
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    if "heartbeat" in payload or payload.get("meta", {}).get("type") == "heartbeat":
                        continue

                    gs = self._parse_ws_event(payload)
                    if gs:
                        yield gs
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning("WS msg type={}, breaking", msg.type)
                    break

    # ------------------------------------------------------------------
    # REST fallback (used when WS disconnects)
    # ------------------------------------------------------------------

    async def _rest_fallback(self, duration: float) -> AsyncIterator[GameState]:
        """Poll REST events for ``duration`` seconds while WS reconnects."""
        assert self._session is not None
        deadline = asyncio.get_event_loop().time() + duration
        while self._running and asyncio.get_event_loop().time() < deadline:
            await self._bootstrap_todays_events()
            for event_id, ev in self._event_cache.items():
                status = ev.get("score", {}).get("event_status", "")
                if status not in _LIVE_STATUSES:
                    continue
                gs = self._event_to_game_state(ev, event_id)
                if gs:
                    yield gs
            await asyncio.sleep(min(5.0, duration))

    # ------------------------------------------------------------------
    # Event bootstrap (REST)
    # ------------------------------------------------------------------

    async def _bootstrap_todays_events(self) -> None:
        """Fetch today's NBA events via REST to populate the event cache."""
        assert self._session is not None
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"{_TR_REST_BASE}/sports/{_NBA_SPORT_ID}/events/{today}"
        try:
            async with self._session.get(url) as resp:
                self._update_rate_headers(resp)
                if resp.status != 200:
                    logger.error("TheRundown events HTTP {}: {}", resp.status, await resp.text())
                    return
                data = await resp.json()
        except Exception:
            logger.exception("Failed to bootstrap today's events")
            return

        events = data.get("events", [])
        for ev in events:
            eid = ev.get("event_id", "")
            if eid:
                self._event_cache[eid] = ev

        statuses = {
            ev.get("score", {}).get("event_status", "?") for ev in events
        }
        logger.info(
            "Bootstrapped {} NBA events | statuses: {}",
            len(events), statuses,
        )

    async def _event_refresh_loop(self) -> None:
        """Periodically re-bootstrap events to discover new games and status changes."""
        while self._running:
            await asyncio.sleep(300)
            await self._bootstrap_todays_events()

    # ------------------------------------------------------------------
    # Player stats polling (REST)
    # ------------------------------------------------------------------

    async def _player_stats_poll_loop(self) -> None:
        """Poll player stats for all live games at a controlled interval."""
        while self._running:
            live_event_ids = [
                eid for eid, ev in self._event_cache.items()
                if ev.get("score", {}).get("event_status", "") in _LIVE_STATUSES
            ]

            if not live_event_ids:
                await asyncio.sleep(self._stats_poll_interval)
                continue

            if self._rate_remaining is not None and self._rate_remaining < self.REST_BUDGET_FLOOR:
                logger.warning(
                    "Rate budget low ({}/{}), skipping stats poll",
                    self._rate_remaining, self._rate_limit,
                )
                await asyncio.sleep(self._stats_poll_interval * 2)
                continue

            tasks = [
                self._fetch_player_stats(eid) for eid in live_event_ids
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self._stats_poll_interval)

    async def _fetch_player_stats(self, event_id: str) -> None:
        """Fetch per-player box scores for a single event via REST."""
        assert self._session is not None
        url = f"{_TR_REST_BASE}/events/{event_id}/players/stats"
        try:
            async with self._session.get(url) as resp:
                self._update_rate_headers(resp)
                if resp.status == 429:
                    logger.warning("TheRundown rate limited on player stats")
                    return
                if resp.status != 200:
                    logger.debug("Player stats HTTP {} for {}", resp.status, event_id)
                    return
                data = await resp.json()
        except Exception:
            logger.debug("Failed to fetch player stats for {}", event_id)
            return

        if not isinstance(data, list):
            return

        players: list[PlayerBoxScore] = []
        for entry in data:
            pbs = self._parse_player_stat_entry(entry, event_id)
            if pbs:
                players.append(pbs)

        if players:
            self._player_stats_cache[event_id] = players
            logger.debug(
                "Updated {} player stats for event {}",
                len(players), event_id[:12],
            )

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_ws_event(self, payload: dict) -> GameState | None:
        """Parse a V1 WebSocket event update into a GameState."""
        try:
            event = payload if "event_id" in payload else payload.get("event", payload)
            if "event_id" not in event:
                return None

            event_id = event["event_id"]
            score = event.get("score", {})
            status = score.get("event_status", "")

            self._event_cache[event_id] = event

            if status not in _LIVE_STATUSES:
                return None

            return self._event_to_game_state(event, event_id)
        except (KeyError, TypeError):
            logger.debug("Failed to parse WS event payload")
            return None

    def _event_to_game_state(self, event: dict, event_id: str) -> GameState | None:
        """Convert a TheRundown V1 event dict into a GameState."""
        try:
            score = event.get("score", {})
            teams = event.get("teams", [])
            teams_norm = event.get("teams_normalized", teams)

            away_team_info = next(
                (t for t in teams_norm if t.get("is_away")),
                teams_norm[0] if len(teams_norm) >= 2 else {},
            )
            home_team_info = next(
                (t for t in teams_norm if t.get("is_home")),
                teams_norm[1] if len(teams_norm) >= 2 else {},
            )

            away_name = away_team_info.get("name", "")
            away_mascot = away_team_info.get("mascot", "")
            home_name = home_team_info.get("name", "")
            home_mascot = home_team_info.get("mascot", "")
            away_abbr = away_team_info.get("abbreviation", "")
            home_abbr = home_team_info.get("abbreviation", "")

            away_full = f"{away_name} {away_mascot}".strip() or away_abbr
            home_full = f"{home_name} {home_mascot}".strip() or home_abbr

            display_clock = score.get("display_clock", "") or ""
            game_period = score.get("game_period", 0) or 0
            clock_str = display_clock if display_clock else f"Q{game_period}"

            player_stats = self._player_stats_cache.get(event_id, [])

            return GameState(
                game_id=event_id,
                home_team=home_full,
                away_team=away_full,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                home_score=score.get("score_home", 0) or 0,
                away_score=score.get("score_away", 0) or 0,
                quarter=game_period,
                clock=clock_str,
                timestamp=datetime.now(timezone.utc),
                player_stats=player_stats,
            )
        except (KeyError, TypeError, IndexError):
            logger.debug("Failed to convert event {} to GameState", event_id)
            return None

    def _parse_player_stat_entry(
        self, entry: dict, event_id: str
    ) -> PlayerBoxScore | None:
        """Parse a single player stat entry from the V2 stats endpoint."""
        try:
            player = entry.get("player", {})
            stats_list = entry.get("stats", [])

            stat_values: dict[str, float] = {}
            for s in stats_list:
                stat_def = s.get("stat", {})
                abbr = stat_def.get("abbreviation", "")
                field = _NBA_STAT_ABBR_MAP.get(abbr)
                if field:
                    try:
                        stat_values[field] = float(s.get("value", 0))
                    except (ValueError, TypeError):
                        pass

            team_id = player.get("team_id", 0)
            team_abbr = self._team_id_to_abbr.get(team_id, str(team_id))

            return PlayerBoxScore(
                player_id=str(player.get("id", "")),
                first_name=player.get("first_name", ""),
                last_name=player.get("last_name", ""),
                team_abbr=team_abbr,
                minutes=stat_values.get("minutes", 0.0),
                pts=int(stat_values.get("pts", 0)),
                fgm=int(stat_values.get("fgm", 0)),
                fga=int(stat_values.get("fga", 0)),
                fg3m=int(stat_values.get("fg3m", 0)),
                fg3a=int(stat_values.get("fg3a", 0)),
                ftm=int(stat_values.get("ftm", 0)),
                fta=int(stat_values.get("fta", 0)),
                reb=int(stat_values.get("reb", 0)),
                ast=int(stat_values.get("ast", 0)),
                stl=int(stat_values.get("stl", 0)),
                blk=int(stat_values.get("blk", 0)),
                turnover=int(stat_values.get("turnover", 0)),
                pf=int(stat_values.get("pf", 0)),
                plus_minus=int(stat_values.get("plus_minus", 0)),
            )
        except (KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Rate limit tracking
    # ------------------------------------------------------------------

    def _update_rate_headers(self, resp: aiohttp.ClientResponse) -> None:
        """Track REST rate limit budget from response headers."""
        try:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            limit = resp.headers.get("X-RateLimit-Limit")
            if remaining is not None:
                self._rate_remaining = int(remaining)
            if limit is not None:
                self._rate_limit = int(limit)
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Team abbreviation resolver
    # ------------------------------------------------------------------

    async def _resolve_team_abbrs(self) -> None:
        """Fetch team list to map team_id -> abbreviation for player stats."""
        assert self._session is not None
        url = f"{_TR_REST_BASE}/sports/{_NBA_SPORT_ID}/teams"
        try:
            async with self._session.get(url) as resp:
                self._update_rate_headers(resp)
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception:
            return

        teams = data if isinstance(data, list) else data.get("teams", [])
        self._team_id_to_abbr: dict[int, str] = {}
        for t in teams:
            tid = t.get("team_id") or t.get("id")
            abbr = t.get("abbreviation", "")
            if tid and abbr:
                self._team_id_to_abbr[int(tid)] = abbr

        logger.info("Resolved {} team abbreviations", len(self._team_id_to_abbr))

    async def close(self) -> None:
        if self._session:
            await self._session.close()

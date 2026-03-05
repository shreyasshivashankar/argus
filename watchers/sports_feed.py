from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import AsyncIterator

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings, GameState, PlayerBoxScore

# NBA team alias -> standard 3-letter abbreviation
_SR_ALIAS_MAP: dict[str, str] = {
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "CHA": "CHA",
    "CHI": "CHI", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GS": "GSW", "GSW": "GSW", "HOU": "HOU",
    "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NO": "NOP",
    "NOP": "NOP", "NY": "NYK", "NYK": "NYK", "OKC": "OKC",
    "ORL": "ORL", "PHI": "PHI", "PHO": "PHX", "PHX": "PHX",
    "POR": "POR", "SAC": "SAC", "SA": "SAS", "SAS": "SAS",
    "TOR": "TOR", "UTA": "UTA", "WAS": "WAS",
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
        """Yield GameState objects as they arrive from the provider."""
        yield  # type: ignore[misc]

    async def run(self) -> None:
        await self.connect()
        async for game_state in self.listen():
            await self.bus.publish("game:state", game_state)

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Sportradar Push Statistics feed (production tier)
# ---------------------------------------------------------------------------

_SR_BASE = "https://api.sportradar.com/nba"


class SportradarFeed(SportsFeed):
    """Sportradar NBA Push Statistics streaming feed.

    Uses two mechanisms:

    1. **Daily Schedule** (REST, called once at startup and every 6 hours)
       to discover today's game IDs, teams, and statuses.

    2. **Push Statistics** (HTTP chunked-transfer streaming) which holds
       a long-lived connection and pushes real-time JSON payloads with
       full player-level box scores as they update. Heartbeats arrive
       every ~5 seconds to keep the connection alive.

    If the Push Statistics stream disconnects, the feed falls back to
    polling the **Game Summary** REST endpoint every 10 seconds until
    the stream reconnects.

    This is the production-grade feed — sub-second latency on stat
    updates, full player box scores, no polling.
    """

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        super().__init__(settings, bus)
        self._api_key = settings.SPORTRADAR_API_KEY
        self._access = settings.SPORTRADAR_ACCESS_LEVEL
        self._session: aiohttp.ClientSession | None = None

        self._game_meta: dict[str, dict] = {}
        self._schedule_refresh_interval = 6 * 3600

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession(
            headers={"x-api-key": self._api_key},
            timeout=aiohttp.ClientTimeout(total=None, sock_read=30),
        )
        for attempt in range(6):
            await self._load_daily_schedule()
            if self._game_meta:
                break
            wait = 10 * (attempt + 1)
            logger.warning("Schedule empty (rate limited?), retrying in {}s", wait)
            await asyncio.sleep(wait)
        logger.info(
            "SportradarFeed connected ({} games today, access={})",
            len(self._game_meta), self._access,
        )

    async def listen(self) -> AsyncIterator[GameState]:
        assert self._session is not None

        schedule_task = asyncio.create_task(self._schedule_refresh_loop())

        while self._running:
            try:
                async for gs in self._stream_push_statistics():
                    yield gs
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Push Statistics stream failed, falling back to REST polling")

            if not self._running:
                break

            logger.info("Falling back to REST polling for Game Summary (10s interval)")
            try:
                async for gs in self._poll_game_summaries():
                    yield gs
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("REST polling failed, retrying stream in 5s")
                await asyncio.sleep(5)

        schedule_task.cancel()

    # ------------------------------------------------------------------
    # Daily schedule
    # ------------------------------------------------------------------

    async def _load_daily_schedule(self) -> None:
        assert self._session is not None
        now = datetime.utcnow()
        dates = [
            now.strftime("%Y/%m/%d"),
            (now - timedelta(days=1)).strftime("%Y/%m/%d"),
        ]
        for date_str in dates:
            url = (
                f"{_SR_BASE}/{self._access}/v8/en/"
                f"games/{date_str}/schedule.json"
            )
            try:
                async with self._session.get(url) as resp:
                    if resp.status != 200:
                        logger.error("Sportradar schedule HTTP {} for {}: {}", resp.status, date_str, await resp.text())
                        continue
                    data = await resp.json()

                for game in data.get("games", []):
                    gid = game.get("id", "")
                    home = game.get("home", {})
                    away = game.get("away", {})
                    self._game_meta[gid] = {
                        "home_team": home.get("name", ""),
                        "home_market": home.get("market", ""),
                        "home_alias": home.get("alias", ""),
                        "away_team": away.get("name", ""),
                        "away_market": away.get("market", ""),
                        "away_alias": away.get("alias", ""),
                        "status": game.get("status", ""),
                    }
            except Exception:
                logger.exception("Failed to load schedule for {}", date_str)

        statuses = {m["status"] for m in self._game_meta.values()}
        logger.info(
            "Loaded {} games from daily schedule | statuses: {}",
            len(self._game_meta), statuses,
        )

    async def _schedule_refresh_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._schedule_refresh_interval)
            await self._load_daily_schedule()

    # ------------------------------------------------------------------
    # Push Statistics stream
    # ------------------------------------------------------------------

    async def _stream_push_statistics(self) -> AsyncIterator[GameState]:
        """Connect to Push Statistics and yield GameState on every update."""
        assert self._session is not None
        url = (
            f"{_SR_BASE}/{self._access}/stream/en/"
            f"statistics/subscribe"
        )
        logger.info("Opening Push Statistics stream: {}", url)

        async with self._session.get(url) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Push Statistics HTTP {resp.status}: {body[:500]}")

            logger.info("Push Statistics stream connected")
            buffer = b""
            async for chunk in resp.content.iter_any():
                if not self._running:
                    break
                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if payload.get("heartbeat"):
                        continue

                    gs = self._parse_push_payload(payload)
                    if gs:
                        yield gs

    # ------------------------------------------------------------------
    # REST fallback: Game Summary polling
    # ------------------------------------------------------------------

    async def _poll_game_summaries(self) -> AsyncIterator[GameState]:
        """Poll Game Summary for each in-progress game every 10s."""
        assert self._session is not None
        _LIVE = {"inprogress", "halftime", "in_progress", "in progress"}
        _SCHEDULE_REFRESH_INTERVAL = 300
        polls_since_refresh = 0
        while self._running:
            live_count = 0
            for gid, meta in list(self._game_meta.items()):
                status = (meta.get("status") or "").lower().strip()
                if status not in _LIVE:
                    continue
                live_count += 1
                url = (
                    f"{_SR_BASE}/{self._access}/v8/en/"
                    f"games/{gid}/summary.json"
                )
                try:
                    async with self._session.get(url) as resp:
                        if resp.status == 429:
                            logger.warning("Sportradar rate limited, backing off")
                            await asyncio.sleep(30)
                            continue
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        gs = self._parse_game_summary(gid, data)
                        if gs:
                            yield gs
                except Exception:
                    logger.exception("Failed to poll game summary for {}", gid)

            if live_count == 0:
                per_game = {m.get("home_alias", "?")+"/"+m.get("away_alias", "?"): m.get("status", "?") for m in self._game_meta.values()}
                logger.warning("No live games to poll | per-game statuses: {}", per_game)
                await asyncio.sleep(30)
                await self._load_daily_schedule()
                polls_since_refresh = 0
            else:
                logger.info("Polled {} live game(s) for summary", live_count)
                await asyncio.sleep(10)
                polls_since_refresh += 1
                if polls_since_refresh * 10 >= _SCHEDULE_REFRESH_INTERVAL:
                    await self._load_daily_schedule()
                    polls_since_refresh = 0

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse_push_payload(self, payload: dict) -> GameState | None:
        """Parse a Push Statistics JSON payload into a GameState.

        The push payload wraps a game object with home/away teams,
        each containing a ``players`` array with full statistics.
        """
        try:
            game = payload.get("game", payload)
            gid = game.get("id", "")

            home = game.get("home", {})
            away = game.get("away", {})

            home_alias = home.get("alias", "")
            away_alias = away.get("alias", "")
            home_abbr = _SR_ALIAS_MAP.get(home_alias, home_alias)
            away_abbr = _SR_ALIAS_MAP.get(away_alias, away_alias)

            scoring = game.get("scoring", [])
            quarter = len(scoring) if scoring else 0

            clock = game.get("clock", "0:00") or "0:00"

            players: list[PlayerBoxScore] = []
            for team_key, abbr in [("home", home_abbr), ("away", away_abbr)]:
                team_data = game.get(team_key, {})
                for p in team_data.get("players", []):
                    pbs = self._parse_sr_player(p, abbr)
                    if pbs:
                        players.append(pbs)

            return GameState(
                game_id=gid,
                home_team=f"{home.get('market', '')} {home.get('name', '')}".strip(),
                away_team=f"{away.get('market', '')} {away.get('name', '')}".strip(),
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                home_score=home.get("points", 0) or 0,
                away_score=away.get("points", 0) or 0,
                quarter=quarter,
                clock=clock,
                timestamp=datetime.utcnow(),
                player_stats=players,
            )
        except (KeyError, TypeError):
            logger.debug("Failed to parse push payload")
            return None

    def _parse_game_summary(self, gid: str, data: dict) -> GameState | None:
        """Parse a Game Summary REST response into a GameState."""
        try:
            home = data.get("home", {})
            away = data.get("away", {})

            home_alias = home.get("alias", "")
            away_alias = away.get("alias", "")
            home_abbr = _SR_ALIAS_MAP.get(home_alias, home_alias)
            away_abbr = _SR_ALIAS_MAP.get(away_alias, away_alias)

            scoring = home.get("scoring", [])
            quarter = len(scoring) if scoring else 0
            clock = data.get("clock", "0:00") or "0:00"

            players: list[PlayerBoxScore] = []
            for team_data, abbr in [(home, home_abbr), (away, away_abbr)]:
                for p in team_data.get("players", []):
                    pbs = self._parse_sr_player(p, abbr)
                    if pbs:
                        players.append(pbs)

            if gid in self._game_meta:
                self._game_meta[gid]["status"] = data.get("status", "")

            return GameState(
                game_id=gid,
                home_team=f"{home.get('market', '')} {home.get('name', '')}".strip(),
                away_team=f"{away.get('market', '')} {away.get('name', '')}".strip(),
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                home_score=home.get("points", 0) or 0,
                away_score=away.get("points", 0) or 0,
                quarter=quarter,
                clock=clock,
                timestamp=datetime.utcnow(),
                player_stats=players,
            )
        except (KeyError, TypeError):
            logger.debug("Failed to parse game summary for {}", gid)
            return None

    @staticmethod
    def _parse_sr_player(p: dict, team_abbr: str) -> PlayerBoxScore | None:
        """Parse a Sportradar player object into a PlayerBoxScore."""
        try:
            stats = p.get("statistics", {})
            full_name = p.get("full_name", "")
            parts = full_name.rsplit(" ", 1)
            first = parts[0] if len(parts) > 1 else full_name
            last = parts[-1] if len(parts) > 1 else ""

            min_str = stats.get("minutes", "0") or "0"
            try:
                if ":" in min_str:
                    m, s = min_str.split(":", 1)
                    minutes = int(m) + int(s) / 60.0
                else:
                    minutes = float(min_str)
            except (ValueError, TypeError):
                minutes = 0.0

            return PlayerBoxScore(
                player_id=p.get("id", p.get("sr_id", "")),
                first_name=first,
                last_name=last,
                team_abbr=team_abbr,
                minutes=minutes,
                pts=stats.get("points", 0) or 0,
                fgm=stats.get("field_goals_made", 0) or 0,
                fga=stats.get("field_goals_att", 0) or 0,
                fg3m=stats.get("three_points_made", 0) or 0,
                fg3a=stats.get("three_points_att", 0) or 0,
                ftm=stats.get("free_throws_made", 0) or 0,
                fta=stats.get("free_throws_att", 0) or 0,
                reb=stats.get("rebounds", 0) or 0,
                ast=stats.get("assists", 0) or 0,
                stl=stats.get("steals", 0) or 0,
                blk=stats.get("blocks", 0) or 0,
                turnover=stats.get("turnovers", 0) or 0,
                pf=stats.get("personal_fouls", 0) or 0,
                plus_minus=stats.get("pls_min", 0) or 0,
            )
        except (KeyError, TypeError):
            return None

    async def close(self) -> None:
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

    Polls today's games and emits live in-progress games.

    Tier support:
      - free: game scores only (5 req/min)
      - all-star: adds per-player stats via /v1/stats ($9.99/mo, 60 req/min)
      - goat: adds live box scores via /box_scores/live ($39.99/mo, 600 req/min)

    At GOAT tier with SPORTS_POLL_INTERVAL=1.0, effective latency is ~1s.
    The single /box_scores/live call returns all games with embedded player
    stats, so even with 10 simultaneous games it stays at 60 req/min.
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

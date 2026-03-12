"""Fetch and cache player season averages from BallDontLie.

Provides per-minute prior rates used by the Bayesian projection model.
Fetched once at startup for all players in today's games, then cached
for the session.  BallDontLie GOAT tier: 600 req/min — this uses < 10.

Usage:
    cache = SeasonAverageCache(settings)
    await cache.load_for_games(game_ids)
    prior = cache.get("player_id")  # PlayerPrior or None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger

from core.schemas import AppSettings

_BDL_BASE = "https://api.balldontlie.io/v1"
_CURRENT_SEASON = 2025  # BDL uses start year of the season


@dataclass(frozen=True)
class PlayerPrior:
    """Season-average per-minute rates for Bayesian prior."""

    player_id: str
    first_name: str
    last_name: str
    games_played: int
    avg_minutes: float  # per game
    pts_per_min: float
    reb_per_min: float
    ast_per_min: float
    stl_per_min: float
    blk_per_min: float
    fg3m_per_min: float
    fga_per_min: float  # for usage rate prior


class SeasonAverageCache:
    """Fetches and caches season averages for active players."""

    def __init__(self, settings: AppSettings) -> None:
        self._api_key = settings.BALLDONTLIE_API_KEY
        self._cache: dict[str, PlayerPrior] = {}
        self._session: aiohttp.ClientSession | None = None
        self._loaded_games: set[str] = set()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": self._api_key},
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def load_for_games(self, game_ids: list[str]) -> None:
        """Fetch season averages for all players in the given games."""
        new_ids = [gid for gid in game_ids if gid not in self._loaded_games]
        if not new_ids:
            return

        session = await self._ensure_session()

        # Step 1: get player IDs from today's box scores
        player_ids: set[str] = set()
        for gid in new_ids:
            params = f"game_ids[]={gid}&per_page=100"
            url = f"{_BDL_BASE}/stats?{params}"
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                for entry in data.get("data", []) or []:
                    player = entry.get("player", {})
                    pid = str(player.get("id", ""))
                    if pid:
                        player_ids.add(pid)
            except Exception:
                logger.debug("Failed to fetch stats for game {}", gid)

        if not player_ids:
            for gid in new_ids:
                self._loaded_games.add(gid)
            return

        # Step 2: fetch season averages for these players
        # BDL season_averages endpoint: /season_averages?season=2025&player_ids[]=1&player_ids[]=2...
        # Process in batches of 25 (API limit)
        pid_list = list(player_ids - set(self._cache.keys()))
        for i in range(0, len(pid_list), 25):
            batch = pid_list[i : i + 25]
            params = "&".join(f"player_ids[]={pid}" for pid in batch)
            url = f"{_BDL_BASE}/season_averages?season={_CURRENT_SEASON}&{params}"
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning("BDL season_averages HTTP {}", resp.status)
                        continue
                    data = await resp.json()
            except Exception:
                logger.debug("Failed to fetch season averages batch")
                continue

            for entry in data.get("data", []) or []:
                prior = self._parse_entry(entry)
                if prior:
                    self._cache[prior.player_id] = prior

        for gid in new_ids:
            self._loaded_games.add(gid)

        logger.info(
            "Season averages loaded: {} players cached ({} new games)",
            len(self._cache),
            len(new_ids),
        )

    def get(self, player_id: str) -> PlayerPrior | None:
        return self._cache.get(player_id)

    def get_by_name(self, last_name: str, first_initial: str = "") -> PlayerPrior | None:
        """Fuzzy lookup by last name + optional first initial."""
        last_upper = last_name.upper()
        candidates = [
            p for p in self._cache.values()
            if p.last_name.upper() == last_upper
        ]
        if first_initial and len(candidates) > 1:
            candidates = [
                p for p in candidates
                if p.first_name and p.first_name[0].upper() == first_initial.upper()
            ]
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _parse_entry(entry: dict[str, Any]) -> PlayerPrior | None:
        try:
            pid = str(entry.get("player_id", ""))
            games = int(entry.get("games_played", 0) or 0)
            if games < 5:  # not enough data for a reliable prior
                return None

            avg_min = float(entry.get("min", "0") or 0)
            if isinstance(avg_min, str):
                # BDL returns "32:15" format sometimes
                if ":" in avg_min:
                    parts = avg_min.split(":")
                    avg_min = float(parts[0]) + float(parts[1]) / 60.0
                else:
                    avg_min = float(avg_min)

            if avg_min < 5.0:
                return None

            pts = float(entry.get("pts", 0) or 0)
            reb = float(entry.get("reb", 0) or 0)
            ast = float(entry.get("ast", 0) or 0)
            stl = float(entry.get("stl", 0) or 0)
            blk = float(entry.get("blk", 0) or 0)
            fg3m = float(entry.get("fg3m", 0) or 0)
            fga = float(entry.get("fga", 0) or 0)

            # First/last name not in season_averages response — store ID for now
            # Will be populated on first box score match
            return PlayerPrior(
                player_id=pid,
                first_name="",
                last_name="",
                games_played=games,
                avg_minutes=avg_min,
                pts_per_min=pts / avg_min if avg_min > 0 else 0,
                reb_per_min=reb / avg_min if avg_min > 0 else 0,
                ast_per_min=ast / avg_min if avg_min > 0 else 0,
                stl_per_min=stl / avg_min if avg_min > 0 else 0,
                blk_per_min=blk / avg_min if avg_min > 0 else 0,
                fg3m_per_min=fg3m / avg_min if avg_min > 0 else 0,
                fga_per_min=fga / avg_min if avg_min > 0 else 0,
            )
        except (ValueError, TypeError, KeyError):
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

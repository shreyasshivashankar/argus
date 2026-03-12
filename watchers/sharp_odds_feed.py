"""SharpAPI sharp book odds watcher.

Polls SharpAPI (sharpapi.io) for real-time Pinnacle odds on NBA games.
Provides sharp lines as Bayesian priors for game totals, spreads, and
moneylines, and detects Kalshi-vs-Pinnacle discrepancies.

SharpAPI free tier: 12 req/min, 2 sportsbooks, 60s delay.
Paid tiers: instant data, more books, EV/arb detection.

Does NOT replace the Kalshi WebSocket feed — that still handles order
book data, fills, and order updates.  This is an auxiliary data source
for the Bayesian projection model.

Usage in runner.py:
    sharp_odds = SharpOddsFeed(settings, bus)
    asyncio.create_task(sharp_odds.run(), name="sharp_odds")
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings

_BASE_URL = "https://api.sharpapi.io/api/v1"


class SharpLine:
    """A sharp book's line for a specific market."""

    __slots__ = ("game_id", "market_type", "line", "sharp_prob", "timestamp")

    def __init__(
        self,
        game_id: str,
        market_type: str,
        line: float,
        sharp_prob: float,
        timestamp: datetime,
    ) -> None:
        self.game_id = game_id
        self.market_type = market_type
        self.line = line
        self.sharp_prob = sharp_prob
        self.timestamp = timestamp


class SharpOddsFeed:
    """Polls SharpAPI for Pinnacle odds and caches them for Bayesian priors."""

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        self.settings = settings
        self.bus = bus
        self._api_key = getattr(settings, "SHARPAPI_KEY", "")
        self._poll_interval = getattr(settings, "SHARP_ODDS_POLL_INTERVAL", 30.0)
        self._session: aiohttp.ClientSession | None = None
        self._running = True

        # Cache: event_name (normalized) → {market_type → SharpLine}
        self._sharp_lines: dict[str, dict[str, SharpLine]] = {}

        # game_id → event key mapping (set by external caller or auto-matched)
        self._game_id_map: dict[str, str] = {}

    async def run(self) -> None:
        if not self._api_key:
            logger.warning("SharpAPI: no API key — sharp odds feed disabled")
            return

        self._session = aiohttp.ClientSession(
            headers={"X-API-Key": self._api_key},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        logger.info("SharpAPI odds feed started (poll {}s)", self._poll_interval)

        try:
            while self._running:
                await self._poll_odds()
                await asyncio.sleep(self._poll_interval)
        finally:
            if self._session and not self._session.closed:
                await self._session.close()

    async def _poll_odds(self) -> None:
        """Fetch NBA Pinnacle odds from SharpAPI."""
        assert self._session is not None

        now = datetime.now(timezone.utc)
        markets_fetched = 0

        # Fetch totals, spreads, moneyline in one call
        for market_type in ("total", "spread", "moneyline"):
            url = f"{_BASE_URL}/odds"
            params = {
                "league": "nba",
                "sportsbook": "pinnacle",
                "market": market_type,
                "limit": 200,
            }

            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning("SharpAPI rate limited — backing off")
                        await asyncio.sleep(10)
                        return
                    if resp.status != 200:
                        logger.debug("SharpAPI HTTP {} for {}", resp.status, market_type)
                        continue
                    data = await resp.json()
            except Exception:
                logger.debug("SharpAPI poll failed for {}", market_type)
                continue

            items = data.get("data", []) or []
            for item in items:
                self._process_odds_item(item, market_type, now)
                markets_fetched += 1

        if markets_fetched > 0:
            logger.debug(
                "SharpAPI: cached {} lines across {} events",
                markets_fetched,
                len(self._sharp_lines),
            )

    def _process_odds_item(
        self, item: dict[str, Any], market_type: str, now: datetime
    ) -> None:
        """Process a single odds item from SharpAPI response."""
        # Build a normalized event key from team names
        home = (item.get("home_team") or "").strip()
        away = (item.get("away_team") or "").strip()
        if not home or not away:
            return

        event_key = f"{away} @ {home}".upper()

        prob = item.get("probability", 0)
        if not prob:
            odds_am = item.get("odds_american", 0)
            if odds_am:
                prob = self._american_to_prob(float(odds_am))

        if not prob or prob <= 0:
            return

        line = 0.0
        mtype_key = market_type.upper()

        selection = (item.get("selection") or "").strip()

        if market_type == "total":
            # selection like "Over 224.5" or "Under 224.5"
            parts = selection.split()
            if len(parts) >= 2:
                try:
                    line = float(parts[-1])
                except ValueError:
                    return
            # We want the over probability; if this is under, invert
            if "under" in selection.lower():
                prob = 1.0 - prob

        elif market_type == "spread":
            # selection like "PHO Suns -3.5"
            parts = selection.rsplit(None, 1)
            if len(parts) >= 2:
                try:
                    line = abs(float(parts[-1]))
                except ValueError:
                    return

        # Store
        if event_key not in self._sharp_lines:
            self._sharp_lines[event_key] = {}

        self._sharp_lines[event_key][mtype_key] = SharpLine(
            game_id=event_key,
            market_type=mtype_key,
            line=line,
            sharp_prob=prob,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Public interface (same as TheRundownWatcher for drop-in replacement)
    # ------------------------------------------------------------------

    def get_sharp_line(self, game_id: str, market_type: str) -> SharpLine | None:
        """Retrieve the latest sharp line for a game/market type."""
        event_key = self._game_id_map.get(game_id)
        if event_key:
            lines = self._sharp_lines.get(event_key, {})
            return lines.get(market_type)
        return None

    def map_game(self, game_id: str, home_team: str, away_team: str) -> None:
        """Map a BDL game_id to a SharpAPI event key using team names.

        Called by the quant agent or runner when a new game is detected.
        Matches by checking if team names appear in the event key.
        """
        if game_id in self._game_id_map:
            return

        home_upper = home_team.upper()
        away_upper = away_team.upper()

        for ek in self._sharp_lines:
            # Event key format: "PHO SUNS @ PHI 76ERS"
            if home_upper in ek or away_upper in ek:
                self._game_id_map[game_id] = ek
                logger.debug("Mapped game {} → SharpAPI event '{}'", game_id, ek)
                return

    def get_total_line(self, game_id: str) -> float | None:
        """Get the Pinnacle total line for Bayesian prior."""
        sl = self.get_sharp_line(game_id, "TOTAL")
        return sl.line if sl and sl.line > 0 else None

    def get_spread_line(self, game_id: str) -> float | None:
        """Get the Pinnacle spread line."""
        sl = self.get_sharp_line(game_id, "SPREAD")
        return sl.line if sl and sl.line > 0 else None

    def get_discrepancy(self, game_id: str, market_type: str) -> float | None:
        """Return sharp_prob for this market (caller compares to Kalshi)."""
        sl = self.get_sharp_line(game_id, market_type)
        return sl.sharp_prob if sl else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _american_to_prob(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds == 0:
            return 0.5
        if odds > 0:
            return 100.0 / (odds + 100.0)
        return abs(odds) / (abs(odds) + 100.0)

    def stop(self) -> None:
        self._running = False

"""TheRundown sharp book odds watcher.

Polls TheRundown REST API for real-time odds from sharp books (Pinnacle)
and retail books (Kalshi).  Publishes sharp lines to Redis so the
Bayesian model can use them as priors, and detects Kalshi-vs-sharp
discrepancies for the sharp book reference strategy.

Does NOT replace the Kalshi WebSocket feed — that still handles order
book data, fills, and order updates.  This is an auxiliary data source.

Usage in runner.py:
    therundown = TheRundownWatcher(settings, bus)
    asyncio.create_task(therundown.run(), name="therundown")
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import aiohttp
from loguru import logger

from core.bus import SignalBus
from core.schemas import AppSettings

_BASE_URL = "https://api.therundown.io/v2"


class SharpLine:
    """A sharp book's line for a specific market."""

    __slots__ = ("game_id", "market_type", "line", "sharp_prob", "kalshi_prob", "timestamp")

    def __init__(
        self,
        game_id: str,
        market_type: str,
        line: float,
        sharp_prob: float,
        kalshi_prob: float,
        timestamp: datetime,
    ) -> None:
        self.game_id = game_id
        self.market_type = market_type
        self.line = line
        self.sharp_prob = sharp_prob
        self.kalshi_prob = kalshi_prob
        self.timestamp = timestamp


class TheRundownWatcher:
    """Polls TheRundown for sharp book odds and publishes discrepancies."""

    def __init__(self, settings: AppSettings, bus: SignalBus) -> None:
        self.settings = settings
        self.bus = bus
        self._api_key = getattr(settings, "THERUNDOWN_API_KEY", "")
        self._poll_interval = getattr(settings, "THERUNDOWN_POLL_INTERVAL", 15.0)
        self._session: aiohttp.ClientSession | None = None
        self._running = True

        # Cache: game_id → {market_type → SharpLine}
        self._sharp_lines: dict[str, dict[str, SharpLine]] = {}

        # Affiliate IDs for books we care about
        self._PINNACLE_ID = "3"   # Pinnacle — sharpest book
        self._KALSHI_ID = "245"   # Kalshi

    async def run(self) -> None:
        if not self._api_key:
            logger.warning("TheRundown: no API key — sharp book watcher disabled")
            return

        self._session = aiohttp.ClientSession(
            headers={"x-rapidapi-key": self._api_key},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        logger.info("TheRundown watcher started (poll {}s)", self._poll_interval)

        try:
            while self._running:
                await self._poll_odds()
                await asyncio.sleep(self._poll_interval)
        finally:
            if self._session and not self._session.closed:
                await self._session.close()

    async def _poll_odds(self) -> None:
        """Fetch today's NBA odds from TheRundown."""
        assert self._session is not None
        url = f"{_BASE_URL}/sports/4/events"  # sport_id=4 is NBA
        params = {"include": "scores,lines"}

        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.debug("TheRundown HTTP {}", resp.status)
                    return
                data = await resp.json()
        except Exception:
            logger.debug("TheRundown poll failed")
            return

        events = data.get("events", []) or []
        for event in events:
            self._process_event(event)

    def _process_event(self, event: dict[str, Any]) -> None:
        """Extract sharp lines from a single event."""
        event_id = str(event.get("event_id", ""))
        if not event_id:
            return

        lines = event.get("lines", {}) or {}
        now = datetime.now(timezone.utc)

        # Get Pinnacle and Kalshi lines
        pinnacle = lines.get(self._PINNACLE_ID, {})
        kalshi = lines.get(self._KALSHI_ID, {})

        if not pinnacle:
            return

        game_lines: dict[str, SharpLine] = {}

        # --- Game total ---
        pin_total = pinnacle.get("total", {})
        kal_total = kalshi.get("total", {})
        if pin_total:
            total_line = float(pin_total.get("total_over", 0) or 0)
            if total_line > 0:
                pin_over_odds = float(pin_total.get("over_price", -110) or -110)
                pin_prob = self._american_to_prob(pin_over_odds)

                kal_prob = None
                if kal_total:
                    kal_over_odds = float(kal_total.get("over_price", 0) or 0)
                    kal_prob = self._american_to_prob(kal_over_odds) if kal_over_odds else None

                game_lines["TOTAL"] = SharpLine(
                    game_id=event_id,
                    market_type="TOTAL",
                    line=total_line,
                    sharp_prob=pin_prob,
                    kalshi_prob=kal_prob or 0.0,
                    timestamp=now,
                )

        # --- Spread ---
        pin_spread = pinnacle.get("spread", {})
        kal_spread = kalshi.get("spread", {})
        if pin_spread:
            spread_line = float(pin_spread.get("point_spread_home", 0) or 0)
            if spread_line != 0:
                pin_home_odds = float(pin_spread.get("home_price", -110) or -110)
                pin_prob = self._american_to_prob(pin_home_odds)

                kal_prob = None
                if kal_spread:
                    kal_home_odds = float(kal_spread.get("home_price", 0) or 0)
                    kal_prob = self._american_to_prob(kal_home_odds) if kal_home_odds else None

                game_lines["SPREAD"] = SharpLine(
                    game_id=event_id,
                    market_type="SPREAD",
                    line=abs(spread_line),
                    sharp_prob=pin_prob,
                    kalshi_prob=kal_prob or 0.0,
                    timestamp=now,
                )

        # --- Moneyline (for reference, not traded) ---
        pin_ml = pinnacle.get("moneyline", {})
        if pin_ml:
            home_ml = float(pin_ml.get("moneyline_home", 0) or 0)
            if home_ml != 0:
                game_lines["MONEYLINE"] = SharpLine(
                    game_id=event_id,
                    market_type="MONEYLINE",
                    line=0,
                    sharp_prob=self._american_to_prob(home_ml),
                    kalshi_prob=0.0,
                    timestamp=now,
                )

        if game_lines:
            self._sharp_lines[event_id] = game_lines

    def get_sharp_line(self, game_id: str, market_type: str) -> SharpLine | None:
        """Retrieve the latest sharp line for a game/market type."""
        game = self._sharp_lines.get(game_id)
        if game is None:
            return None
        return game.get(market_type)

    def get_total_line(self, game_id: str) -> float | None:
        """Convenience: get the sharp total line for Bayesian prior."""
        sl = self.get_sharp_line(game_id, "TOTAL")
        return sl.line if sl else None

    def get_spread_line(self, game_id: str) -> float | None:
        """Convenience: get the sharp spread line."""
        sl = self.get_sharp_line(game_id, "SPREAD")
        return sl.line if sl else None

    def get_discrepancy(self, game_id: str, market_type: str) -> float | None:
        """Return sharp_prob - kalshi_prob. Positive = Kalshi is underpriced."""
        sl = self.get_sharp_line(game_id, market_type)
        if sl is None or sl.kalshi_prob == 0.0:
            return None
        return sl.sharp_prob - sl.kalshi_prob

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

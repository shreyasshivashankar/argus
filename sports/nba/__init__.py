"""NBA sport module — registers with the sport registry on import."""
from __future__ import annotations

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings
from sports import register
from sports.base import SportModule, SportsFeed


@register("nba")
class NBAModule(SportModule):
    """NBA sport module: BallDontLie feed + multi-strategy quant agent."""

    @property
    def name(self) -> str:
        return "nba"

    @property
    def ticker_prefix(self) -> str:
        return "KXNBA"

    def create_feed(self, settings: AppSettings, bus: SignalBus) -> SportsFeed:
        from sports.nba.feed import BallDontLieFeed
        return BallDontLieFeed(settings, bus)

    def create_quant_agent(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
    ) -> BaseAgent:
        from sports.nba.quant import NBAQuantAgent
        return NBAQuantAgent(settings, bus, client)

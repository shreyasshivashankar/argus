"""Abstract base classes for sport modules."""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings, GameState


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


class SportModule(ABC):
    """Interface every sport module must implement.

    A sport module provides:
    - A data feed (scores, stats) for that sport
    - A quant agent (strategies + evaluation engine)
    - A ticker prefix for matching Kalshi markets
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"nba"``, ``"tennis"``."""

    @property
    @abstractmethod
    def ticker_prefix(self) -> str:
        """Kalshi ticker prefix, e.g. ``"KXNBA"``."""

    @abstractmethod
    def create_feed(self, settings: AppSettings, bus: SignalBus) -> SportsFeed:
        """Build the live data feed for this sport."""

    @abstractmethod
    def create_quant_agent(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
    ) -> BaseAgent:
        """Build the quant/strategy agent for this sport."""

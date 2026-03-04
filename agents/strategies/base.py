from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import GameState, MarketState, Signal


class BaseStrategy(ABC):
    """Interface every market strategy must implement.

    ``can_evaluate`` acts as a fast filter so the OmniQuant agent only
    runs a strategy against tickers it understands.  ``evaluate``
    contains the actual math and returns a Signal when +EV, else None.
    """

    name: str

    @abstractmethod
    def can_evaluate(self, market: MarketState) -> bool:
        """Return True if this strategy knows how to price *market*."""

    @abstractmethod
    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        """Return a Signal if a +EV edge is found, else None."""

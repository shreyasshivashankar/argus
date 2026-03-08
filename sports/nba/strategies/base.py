from __future__ import annotations

from abc import ABC, abstractmethod

from core.schemas import GameState, MarketState, Signal
from core.utils import MINUTES_PER_QUARTER, team_minutes_played


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

    def model_probability(self, game: GameState, market: MarketState) -> float | None:
        """Return the strategy's raw model probability for the YES side, or None.

        Used by the bailout monitor to re-evaluate open positions without the
        EV threshold gate.  Strategies that can't price the market return None,
        which suppresses bailout for that position (safe default: no action).
        """
        return None

    @staticmethod
    def time_adjusted_ev_threshold(
        base_threshold: float,
        game: GameState,
        quarter_multipliers: tuple[float, float, float, float] = (2.5, 1.75, 1.25, 0.75),
    ) -> float:
        """Piecewise-linear EV threshold that interpolates within each quarter.

        ``quarter_multipliers`` maps Q1..Q4 start-of-quarter multipliers.
        Within each quarter the multiplier decays linearly toward the next
        quarter's value.  Overtime clamps to Q4's floor.
        """
        elapsed = team_minutes_played(game)
        qi = int(elapsed // MINUTES_PER_QUARTER)  # 0-based quarter index

        if qi >= 3:
            return base_threshold * quarter_multipliers[3]

        frac = (elapsed - qi * MINUTES_PER_QUARTER) / MINUTES_PER_QUARTER
        m_start = quarter_multipliers[qi]
        m_end = quarter_multipliers[qi + 1]
        multiplier = m_start + frac * (m_end - m_start)
        return base_threshold * multiplier

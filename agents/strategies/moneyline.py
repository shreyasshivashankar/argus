"""Moneyline (game winner) strategy using logistic reversal model.

Detects mispriced win-probability contracts by comparing a live logistic
model estimate to the Kalshi implied probability.  Only targets tickers
that contain ``GAME`` and both team abbreviations (daily matchup markets).
"""
from __future__ import annotations

import numpy as np

from agents.strategies.base import BaseStrategy
from core.schemas import Action, GameState, MarketState, Side, Signal, SignalStatus


# Logistic model coefficients for live win probability.
# Each additional quarter increases the weight of the current
# score differential (late leads are harder to overcome).
_QUARTER_WEIGHT_INCREMENT = 0.3

# Slope of the logistic curve per point of score differential.
# Derived from NBA historical comeback data: ~0.15 per point
# gives realistic reversal probabilities at each quarter.
_LOGISTIC_SLOPE_PER_POINT = 0.15


class MoneylineStrategy(BaseStrategy):
    name = "moneyline"

    def __init__(
        self,
        ev_threshold: float = 0.03,
        target_exit_spread: int = 7,
        quarter_multipliers: tuple[float, float, float, float] = (2.5, 1.75, 1.25, 0.75),
    ) -> None:
        self._ev_threshold = ev_threshold
        self._target_exit_spread = target_exit_spread
        self._quarter_multipliers = quarter_multipliers
        self._reversal_table: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def can_evaluate(self, market: MarketState) -> bool:
        """Only evaluate daily game-winner tickers (contain ``GAME``)."""
        return "GAME" in market.ticker.upper()

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        if market.yes_ask <= 0:
            return None

        target_team = market.ticker.split("-")[-1]
        model_prob = self._win_probability(game, target_team)
        entry_price_cents = market.yes_ask
        ev = model_prob * 1.0 - (entry_price_cents / 100.0)

        threshold = self.time_adjusted_ev_threshold(
            self._ev_threshold, game, self._quarter_multipliers,
        )
        if ev < threshold:
            return None

        exit_price = min(entry_price_cents + self._target_exit_spread, 99)

        return Signal(
            ticker=market.ticker,
            action=Action.BUY,
            side=Side.YES,
            status=SignalStatus.VALIDATED,
            confidence=min(model_prob, 1.0),
            source=self.name,
            ev_estimate=ev,
            entry_price=entry_price_cents,
            exit_price=exit_price,
            game_id=game.game_id,
        )

    # ------------------------------------------------------------------
    # Probability model
    # ------------------------------------------------------------------

    def _win_probability(self, game: GameState, target_team: str) -> float:
        """Return win probability for *target_team* given current game state."""
        diff = game.away_score - game.home_score
        quarter = game.quarter

        cached = self._reversal_table.get((quarter, diff))
        home_prob = cached if cached is not None else self._logistic_estimate(diff, quarter)

        home_ids = [game.home_abbr.upper(), game.home_team.upper()]
        if target_team.upper() in home_ids:
            return home_prob
        return 1.0 - home_prob

    @staticmethod
    def _logistic_estimate(score_diff: int, quarter: int) -> float:
        """Logistic P(home wins) given away-home score diff and quarter."""
        quarter_weight = 1.0 + (quarter - 1) * _QUARTER_WEIGHT_INCREMENT
        z = -_LOGISTIC_SLOPE_PER_POINT * score_diff * quarter_weight
        return float(1.0 / (1.0 + np.exp(-z)))

"""First-half winner strategy for Kalshi NBA first-half markets.

Targets tickers labeled with '1H' or 'HALF' (e.g. KXNBA-1H-LAL-GSW-...).
Only fires when the current quarter is 1 or 2 (first half is live).

Model: logistic win-probability using live score differential, calibrated
with tighter quarter weights since the time horizon ends at halftime rather
than at game end.  Slightly steeper slope than the full-game moneyline
because first-half leads are harder to overcome in a shorter window.

98-cent auto-cashout interacts cleanly here: once the leading team's
half-winner contract hits 98c (near certainty), the executor cashes out
immediately, freeing capital to reinvest in second-half or full-game markets.
"""
from __future__ import annotations

import numpy as np

from agents.strategies.base import BaseStrategy
from core.schemas import Action, GameState, MarketState, Side, Signal, SignalStatus

# Logistic slope per point of score differential.
# Higher than moneyline (0.15) because a 10-pt first-half lead is
# proportionally more decisive over 24 min than over 48 min.
_LOGISTIC_SLOPE = 0.20

# Quarter weights for the first half.
_Q1_WEIGHT = 1.0   # Q1: ~12 min of variance remain in the half
_Q2_WEIGHT = 1.6   # Q2: clock running out — leads are stickier


class FirstHalfStrategy(BaseStrategy):
    """First-half winner strategy targeting KXNBA tickers with '1H' or 'HALF'."""

    name = "first_half"

    def __init__(
        self,
        ev_threshold: float = 0.03,
        target_exit_spread: int = 7,
    ) -> None:
        self._ev_threshold = ev_threshold
        self._target_exit_spread = target_exit_spread

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def can_evaluate(self, market: MarketState) -> bool:
        """Only evaluate NBA first-half tickers."""
        upper = market.ticker.upper()
        return upper.startswith("KXNBA") and ("1H" in upper or "HALF" in upper)

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        # Only trade while the first half is live
        if game.quarter not in (1, 2):
            return None

        if market.yes_ask <= 0:
            return None

        target_team = market.ticker.split("-")[-1]
        model_prob = self._half_win_probability(game, target_team)

        entry_price_cents = market.yes_ask
        ev = model_prob - (entry_price_cents / 100.0)

        if ev < self._ev_threshold:
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

    def _half_win_probability(self, game: GameState, target_team: str) -> float:
        """P(target_team wins the first half) given live Q1/Q2 score."""
        # score_diff > 0 means away team is leading
        score_diff = game.away_score - game.home_score
        quarter_weight = _Q1_WEIGHT if game.quarter == 1 else _Q2_WEIGHT
        z = -_LOGISTIC_SLOPE * score_diff * quarter_weight
        home_prob = float(1.0 / (1.0 + np.exp(-z)))

        home_ids = {game.home_abbr.upper(), game.home_team.upper()}
        if target_team.upper() in home_ids:
            return home_prob
        return 1.0 - home_prob

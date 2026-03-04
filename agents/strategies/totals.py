"""Over/Under totals strategy using live pace projection.

Projects the final combined score from the current scoring pace and
compares it to the Kalshi line embedded in the ticker to find +EV
over/under opportunities.

Targets tickers that contain ``TOTAL`` or ``T`` followed by a numeric
line (e.g. ``KXNBATOTAL-26MAR03-WASORL-O225``).
"""
from __future__ import annotations

import re

from agents.strategies.base import BaseStrategy
from core.schemas import Action, GameState, MarketState, Side, Signal, SignalStatus

_LINE_RE = re.compile(r"[OU](\d+(?:\.\d+)?)", re.IGNORECASE)
_MINUTES_PER_GAME = 48.0


class TotalsStrategy(BaseStrategy):
    name = "totals"

    def __init__(
        self,
        ev_threshold: float = 0.03,
        target_exit_spread: int = 7,
        min_minutes: float = 6.0,
    ) -> None:
        self._ev_threshold = ev_threshold
        self._target_exit_spread = target_exit_spread
        self._min_minutes = min_minutes

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def can_evaluate(self, market: MarketState) -> bool:
        ticker_upper = market.ticker.upper()
        return "TOTAL" in ticker_upper or bool(_LINE_RE.search(ticker_upper))

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        if market.yes_ask <= 0:
            return None

        line = self._extract_line(market.ticker)
        if line is None:
            return None

        minutes_played = self._minutes_played(game)
        if minutes_played < self._min_minutes:
            return None

        current_total = game.home_score + game.away_score
        pace_per_minute = current_total / minutes_played
        projected_final = pace_per_minute * _MINUTES_PER_GAME

        is_over = self._is_over_ticker(market.ticker)
        if is_over:
            model_prob = self._over_probability(projected_final, line, minutes_played)
        else:
            model_prob = 1.0 - self._over_probability(projected_final, line, minutes_played)

        entry_price_cents = market.yes_ask
        ev = model_prob * 1.0 - (entry_price_cents / 100.0)

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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_line(ticker: str) -> float | None:
        """Pull the numeric line from a totals ticker (e.g. O225.5 -> 225.5)."""
        m = _LINE_RE.search(ticker)
        return float(m.group(1)) if m else None

    @staticmethod
    def _is_over_ticker(ticker: str) -> bool:
        """True if the ticker represents the OVER side."""
        upper = ticker.upper()
        if "-O" in upper or "OVER" in upper:
            return True
        if "-U" in upper or "UNDER" in upper:
            return False
        # Default to over if ambiguous
        return True

    @staticmethod
    def _minutes_played(game: GameState) -> float:
        """Estimate minutes elapsed from quarter and clock string.

        Clock formats seen from Balldontlie: ``"5:30"``, ``":08.7"``,
        ``"Half"``, ``"END Q3"``.
        """
        q = max(game.quarter, 1)
        completed_minutes = (q - 1) * 12.0

        clock = game.clock.strip()
        if not clock or clock.upper() in ("HALF", "END"):
            return completed_minutes

        # Strip leading 'Q' labels like "Q3 5:30"
        parts = clock.split()
        raw = parts[-1] if parts else clock

        try:
            if ":" in raw:
                mins_str, secs_str = raw.split(":", 1)
                mins = int(mins_str) if mins_str else 0
                secs = float(secs_str)
                remaining = mins + secs / 60.0
            else:
                remaining = float(raw) / 60.0
            return completed_minutes + (12.0 - remaining)
        except (ValueError, TypeError):
            return completed_minutes

    @staticmethod
    def _over_probability(
        projected: float,
        line: float,
        minutes_played: float,
    ) -> float:
        """Estimate P(total > line) using a simple normal approximation.

        Variance shrinks as more of the game is played (less time for
        pace to change).  The standard deviation is scaled by the
        fraction of game remaining.
        """
        fraction_remaining = max(1.0 - minutes_played / _MINUTES_PER_GAME, 0.01)
        std_dev = 15.0 * fraction_remaining ** 0.5

        if std_dev < 0.01:
            return 1.0 if projected > line else 0.0

        z = (projected - line) / std_dev

        # Fast sigmoid approximation of normal CDF
        return float(1.0 / (1.0 + 2.718281828 ** (-1.7 * z)))

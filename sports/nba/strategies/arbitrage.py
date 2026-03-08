"""Binary arbitrage strategy — buys both YES and NO when combined ask < 100c.

On Kalshi, YES + NO must settle to exactly $1.  When the market is
temporarily mispriced so that YES_ask + NO_ask < 100c, buying both
sides guarantees a profit regardless of outcome.

Edge accounting
---------------
Gross spread  = 100 - YES_ask - NO_ask
Kalshi fee    ≈ ceil(0.07 * p * (1-p) * 100) per contract per side
Typical fee   ≈ 1–2c per side at mid-range prices
Net spread    = gross - fee_yes - fee_no

We require net_spread >= 1c and only fire when gross spread covers fees
with a buffer.  The companion NO order is signalled via ``no_entry_price``
on the Signal; the executor places both legs simultaneously.
"""
from __future__ import annotations

import math

from sports.nba.strategies.base import BaseStrategy
from core.schemas import Action, GameState, MarketState, Side, Signal, SignalStatus


class ArbitrageStrategy(BaseStrategy):
    name = "arbitrage"

    def __init__(self, max_combined_cents: int = 95) -> None:
        self._max_combined = max_combined_cents

    # ------------------------------------------------------------------
    # Filter — only NBA markets with valid two-sided quotes
    # ------------------------------------------------------------------

    def can_evaluate(self, market: MarketState) -> bool:
        if not market.ticker.upper().startswith("KXNBA"):
            return False
        return market.yes_ask > 0 and market.no_ask > 0

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        yes_ask = market.yes_ask
        no_ask = market.no_ask

        if yes_ask <= 0 or no_ask <= 0:
            return None

        combined = yes_ask + no_ask
        if combined > self._max_combined:
            return None

        gross_spread = 100 - combined

        # Taker fee per contract: ceil(0.07 * p * (1-p) * 100) cents
        fee_yes = math.ceil(0.07 * yes_ask * (100 - yes_ask) / 100)
        fee_no = math.ceil(0.07 * no_ask * (100 - no_ask) / 100)
        net_spread = gross_spread - fee_yes - fee_no

        if net_spread < 1:
            return None

        # Use moderate confidence so Kelly sizes conservatively (not infinity)
        # Arb is ~certain, but we set 0.62 to keep position sizes sane
        confidence = 0.62
        ev = net_spread / 100.0

        return Signal(
            ticker=market.ticker,
            action=Action.BUY,
            side=Side.YES,
            status=SignalStatus.VALIDATED,
            confidence=confidence,
            source=self.name,
            ev_estimate=ev,
            entry_price=yes_ask,
            exit_price=99,      # Hold to settlement; 98c auto-cashout handles this
            game_id=game.game_id,
            no_entry_price=no_ask,  # Executor places companion NO order
        )

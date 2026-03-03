from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import (
    Action,
    AppSettings,
    ContextStatus,
    GameState,
    MarketState,
    Side,
    Signal,
    SignalStatus,
)


class NBAQuantAgent(BaseAgent):
    """Stage 1: Synchronous Quant Trigger for NBA markets.

    Subscribes to ``game:state`` and ``market:state``.  On every update the
    hot path runs without external async I/O:

    1. Derive implied probability from Kalshi bid/ask.
    2. Compute model probability from game state + historical reversal data.
    3. If +EV exceeds threshold → synchronous ``redis.get`` context check
       (fail-close: None → VETO).
    4. If SAFE → publish VALIDATED signal directly to ``signal:validated``.

    To add a new sport, subclass BaseAgent with the same pattern and plug
    in sport-specific probability models.
    """

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
    ) -> None:
        super().__init__("nba_quant", settings, bus, client)

        # Latest state caches (updated on every WS push)
        self._games: dict[str, GameState] = {}
        self._markets: dict[str, MarketState] = {}

        # game_id → market_ticker mapping (configured externally or discovered)
        self._game_to_ticker: dict[str, str] = {}

        # Pre-loaded reversal probability table: (quarter, score_diff) → p(win)
        # Populated from historical backtest data in data/
        self._reversal_table: dict[tuple[int, int], float] = (
            self._load_reversal_table()
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.bus.subscribe(
            ["game:state", "market:state"], self._on_message
        )

    async def _on_message(self, channel: str, data: dict[str, Any]) -> None:
        if channel == "game:state":
            self._update_game(data)
        elif channel == "market:state":
            self._update_market(data)
        await self._evaluate_all()

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _update_game(self, data: dict) -> None:
        try:
            gs = GameState(**data)
            self._games[gs.game_id] = gs
        except Exception:
            self.log.warning("Bad game:state payload: {}", data)

    def _update_market(self, data: dict) -> None:
        try:
            ms = MarketState(**data)
            self._markets[ms.ticker] = ms
        except Exception:
            self.log.warning("Bad market:state payload: {}", data)

    # ------------------------------------------------------------------
    # Core evaluation — no external I/O in the hot path
    # ------------------------------------------------------------------

    async def _evaluate_all(self) -> None:
        for game_id, game in self._games.items():
            ticker = self._game_to_ticker.get(game_id)
            if not ticker or ticker not in self._markets:
                continue
            market = self._markets[ticker]
            await self._evaluate(game, market)

    async def _evaluate(self, game: GameState, market: MarketState) -> None:
        if market.yes_ask <= 0:
            return

        implied_prob = market.yes_ask / 100.0
        model_prob = self._model_probability(game)
        payout = 1.0  # Kalshi binary: $1 payout
        entry_price_cents = market.yes_ask
        ev = model_prob * payout - (entry_price_cents / 100.0)

        if ev < self.settings.EV_THRESHOLD / 100.0:
            return

        # --- Synchronous context cache read (fail-close) ---
        status, reason = await self.bus.get_context(game.game_id)

        if status != ContextStatus.SAFE:
            self.log.info(
                "VETO for {} ({}): {}", game.game_id, market.ticker, reason
            )
            return

        exit_price = entry_price_cents + self.settings.TARGET_EXIT_SPREAD

        signal = Signal(
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

        await self.bus.publish("signal:validated", signal)
        self.log.info(
            "+EV signal: {} EV={:.4f} entry={} exit={} model_p={:.3f} implied_p={:.3f}",
            market.ticker, ev, entry_price_cents, exit_price,
            model_prob, implied_prob,
        )

    # ------------------------------------------------------------------
    # Probability model
    # ------------------------------------------------------------------

    def _model_probability(self, game: GameState) -> float:
        """Compute win probability for the underdog from game state.

        Uses a pre-loaded reversal probability table keyed by
        (quarter, score_differential).  Falls back to a simple logistic
        estimate when the lookup misses.
        """
        diff = game.away_score - game.home_score  # positive = away leading
        quarter = game.quarter

        cached = self._reversal_table.get((quarter, diff))
        if cached is not None:
            return cached

        return self._logistic_estimate(diff, quarter)

    @staticmethod
    def _logistic_estimate(score_diff: int, quarter: int) -> float:
        """Fallback logistic model: P(underdog wins) given score diff and quarter.

        Higher quarters amplify the impact of the deficit because there is
        less time to recover.
        """
        quarter_weight = 1.0 + (quarter - 1) * 0.3
        z = -0.15 * score_diff * quarter_weight
        return float(1.0 / (1.0 + np.exp(-z)))

    @staticmethod
    def _load_reversal_table() -> dict[tuple[int, int], float]:
        """Load historical reversal probabilities from data/.

        Returns an empty dict if no backtest data is available yet.
        Populate by running the Balldontlie backtest pipeline.
        """
        # TODO: load from data/nba_reversal_probs.csv once backtest is run
        return {}

    # ------------------------------------------------------------------
    # Market mapping
    # ------------------------------------------------------------------

    def register_game_market(self, game_id: str, ticker: str) -> None:
        """Map a live game to its Kalshi market ticker."""
        self._game_to_ticker[game_id] = ticker
        self.log.info("Mapped game {} → ticker {}", game_id, ticker)

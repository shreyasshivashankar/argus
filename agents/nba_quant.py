"""OmniQuant Agent — multi-strategy portfolio manager for NBA markets.

Subscribes to ``game:state``, ``market:state``, and ``portfolio:state``.
On every update it fans out evaluation to all registered strategies,
ranks the proposals by EV, applies a correlation filter (max 1 open
position per game), and fires the best signal.

Strategies are pluggable via the constructor; add new ones without
touching this file.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import Any

from loguru import logger

from agents.strategies import MoneylineStrategy, TotalsStrategy
from agents.strategies.base import BaseStrategy
from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import (
    Action,
    AppSettings,
    ContextStatus,
    GameState,
    MarketState,
    PortfolioState,
    Side,
    Signal,
    SignalStatus,
)


class NBAQuantAgent(BaseAgent):
    """Multi-strategy quant agent for NBA markets.

    Replaces the monolithic single-model approach with a Strategy Pattern:
    each strategy independently evaluates markets it understands, and this
    agent ranks, filters, and publishes the best opportunity.
    """

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
        strategies: list[BaseStrategy] | None = None,
    ) -> None:
        super().__init__("nba_quant", settings, bus, client)

        self._strategies: list[BaseStrategy] = strategies or [
            MoneylineStrategy(
                ev_threshold=settings.EV_THRESHOLD / 100.0,
                target_exit_spread=settings.TARGET_EXIT_SPREAD,
            ),
            TotalsStrategy(
                ev_threshold=settings.EV_THRESHOLD / 100.0,
                target_exit_spread=settings.TARGET_EXIT_SPREAD,
            ),
        ]

        self._games: dict[str, GameState] = {}
        self._markets: dict[str, MarketState] = {}

        # game_id -> list of matched market tickers
        self._game_to_tickers: dict[str, list[str]] = {}

        self._portfolio: PortfolioState | None = None

        # Throttle sets
        self._logged_games: set[str] = set()
        self._logged_unmapped: set[str] = set()
        self._logged_tickers: set[str] = set()

        # Per-ticker cooldown
        self._signal_cooldowns: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self.bus.subscribe(
            ["game:state", "market:state", "portfolio:state"], self._on_message
        )

    async def _on_message(self, channel: str, data: dict[str, Any]) -> None:
        if channel == "game:state":
            self._update_game(data)
        elif channel == "market:state":
            self._update_market(data)
        elif channel == "portfolio:state":
            self._update_portfolio(data)
            return
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
            if ms.ticker not in self._logged_tickers:
                self.log.info(
                    "New Kalshi market: {} bid={} ask={}",
                    ms.ticker, ms.yes_bid, ms.yes_ask,
                )
                self._logged_tickers.add(ms.ticker)
        except Exception:
            self.log.warning("Bad market:state payload: {}", data)

    def _update_portfolio(self, data: dict) -> None:
        try:
            self._portfolio = PortfolioState(**data)
        except Exception:
            self.log.warning("Bad portfolio:state payload: {}", data)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def _evaluate_all(self) -> None:
        self._auto_map_tickers()

        for game_id, game in self._games.items():
            if game_id not in self._logged_games:
                self.log.info(
                    "Tracking live game: {} @ {}",
                    game.away_team, game.home_team,
                )
                self._logged_games.add(game_id)

            tickers = self._game_to_tickers.get(game_id, [])
            if not tickers:
                if game_id not in self._logged_unmapped:
                    self.log.info("No Kalshi market mapping for game {}", game_id)
                    self._logged_unmapped.add(game_id)
                continue

            # Gather proposals from all strategies across all markets for this game
            proposals: list[Signal] = []
            for ticker in tickers:
                market = self._markets.get(ticker)
                if market is None:
                    continue
                for strategy in self._strategies:
                    if not strategy.can_evaluate(market):
                        continue
                    signal = strategy.evaluate(game, market)
                    if signal is not None:
                        proposals.append(signal)

            if not proposals:
                continue

            # Rank by EV, best first
            proposals.sort(key=lambda s: s.ev_estimate, reverse=True)
            best = proposals[0]

            await self._try_execute(best, game)

    async def _try_execute(self, signal: Signal, game: GameState) -> None:
        """Context check, cooldown, capital check, and publish."""
        # --- Fail-close context check ---
        status, reason = await self.bus.get_context(game.game_id)
        if status != ContextStatus.SAFE:
            self.log.info(
                "VETO for {} ({}): {}", game.game_id, signal.ticker, reason
            )
            return

        # --- Per-ticker cooldown ---
        now = datetime.utcnow()
        last = self._signal_cooldowns.get(signal.ticker)
        if last and (now - last).total_seconds() < 60:
            return

        entry_price_cents = signal.entry_price

        if self._can_fund_trade(entry_price_cents):
            self._signal_cooldowns[signal.ticker] = now
            await self.bus.publish("signal:validated", signal)
            self.log.info(
                "+EV signal [{}]: {} EV={:.4f} entry={} exit={} conf={:.3f}",
                signal.source, signal.ticker, signal.ev_estimate,
                signal.entry_price, signal.exit_price, signal.confidence,
            )
        else:
            await self._try_reallocate(
                signal.ev_estimate, entry_price_cents,
                signal.confidence, self._markets[signal.ticker], game,
            )

    # ------------------------------------------------------------------
    # Capital awareness
    # ------------------------------------------------------------------

    def _can_fund_trade(self, entry_price_cents: int) -> bool:
        if self._portfolio is None:
            return True
        bankroll_cents = self._portfolio.bankroll * 100
        return bankroll_cents >= entry_price_cents

    async def _try_reallocate(
        self,
        new_ev: float,
        new_entry_price: int,
        model_prob: float,
        market: MarketState,
        game: GameState,
    ) -> None:
        """Evaluate whether liquidating a resting exit frees enough capital
        to fund a strictly better trade (unit-correct hurdle rate)."""
        if self._portfolio is None or not self._portfolio.positions:
            return

        for pos in self._portfolio.positions:
            ms = self._markets.get(pos.ticker)
            if ms is None:
                continue

            live_bid = ms.yes_bid
            if live_bid < self.settings.MIN_REALLOCATE_BID:
                continue

            freed_capital_cents = live_bid * pos.remaining_count
            expected_new_count = math.floor(
                freed_capital_cents * self.settings.KELLY_FRACTION / new_entry_price
            ) if new_entry_price > 0 else 0
            if expected_new_count < 1:
                continue

            total_new_ev_cents = expected_new_count * (new_ev * 100)
            foregone_profit = (pos.target_exit_price - live_bid) * pos.remaining_count
            fees = pos.remaining_count * self.settings.TAKER_FEE_CENTS

            if total_new_ev_cents <= foregone_profit + fees:
                continue

            signal = Signal(
                ticker=pos.ticker,
                action=Action.SELL,
                side=pos.side,
                status=SignalStatus.REALLOCATE,
                confidence=min(model_prob, 1.0),
                source=self.name,
                ev_estimate=new_ev,
                entry_price=live_bid,
                exit_price=pos.target_exit_price,
                game_id=game.game_id,
                target_order_id=pos.client_order_id,
            )
            await self.bus.publish("signal:reallocate", signal)
            self.log.info(
                "REALLOCATE: liquidate {} x{} @{} (foregone={:.0f}c, fees={:.0f}c) "
                "for new EV={:.0f}c on {}",
                pos.ticker, pos.remaining_count, live_bid,
                foregone_profit, fees, total_new_ev_cents, market.ticker,
            )
            return

    # ------------------------------------------------------------------
    # Market mapping
    # ------------------------------------------------------------------

    def _auto_map_tickers(self) -> None:
        """Map live games to all matching Kalshi NBA market tickers.

        A game can map to multiple tickers (moneyline, totals, spreads).
        Each ticker must contain both team abbreviations to confirm
        it belongs to this specific matchup.
        """
        for game_id, game in self._games.items():
            if game_id in self._game_to_tickers:
                continue

            home = game.home_abbr.upper() if game.home_abbr else game.home_team.upper()
            away = game.away_abbr.upper() if game.away_abbr else game.away_team.upper()

            matched: list[str] = []
            for ticker in self._markets:
                ticker_upper = ticker.upper()
                if not ticker_upper.startswith("KXNBA"):
                    continue
                if home in ticker_upper and away in ticker_upper:
                    matched.append(ticker)

            if matched:
                self._game_to_tickers[game_id] = matched
                self._logged_unmapped.discard(game_id)
                for t in matched:
                    self.log.info("Mapped game {} → ticker {}", game_id, t)

    def register_game_market(self, game_id: str, ticker: str) -> None:
        """Manually map a game to a market ticker."""
        tickers = self._game_to_tickers.setdefault(game_id, [])
        if ticker not in tickers:
            tickers.append(ticker)
        self._logged_unmapped.discard(game_id)
        self.log.info("Mapped game {} → ticker {}", game_id, ticker)

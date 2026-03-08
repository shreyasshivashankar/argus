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
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from sports.nba.config import build_strategies
from sports.nba.strategies.base import BaseStrategy
from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import (
    Action,
    AppSettings,
    ContextStatus,
    GameState,
    MarketState,
    PortfolioPosition,
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

        self._strategies: list[BaseStrategy] = strategies or build_strategies(settings)

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

        # Brief global pause after reallocation to let exchange cash settle
        self._global_realloc_lock: datetime | None = None

        # Per-ticker bailout cooldown: prevent hammering the same position
        self._bailout_cooldowns: dict[str, datetime] = {}

        # Per-game pending lock: set immediately when a signal is published,
        # cleared when portfolio:state confirms a position for that game.
        # Prevents multiple signals firing for the same game before the first
        # fill/portfolio update arrives (race condition with MAX_GAME_EXPOSURE).
        self._pending_game_orders: set[str] = set()

        self.log.info(
            "NBA strategies enabled: {}",
            [s.name for s in self._strategies],
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        monitor_task = asyncio.create_task(self._manage_open_positions())
        await self.bus.subscribe(
            ["game:state", "market:state", "portfolio:state"], self._on_message
        )
        monitor_task.cancel()

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
            # Clear pending locks for games whose positions are now confirmed.
            if self._portfolio.positions and self._pending_game_orders:
                confirmed_tickers = {pos.ticker for pos in self._portfolio.positions}
                cleared = {
                    gid for gid in self._pending_game_orders
                    if set(self._game_to_tickers.get(gid, [])) & confirmed_tickers
                }
                self._pending_game_orders -= cleared
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

            # Rank by EV, best first; execute first proposal that clears all hurdles
            proposals.sort(key=lambda s: s.ev_estimate, reverse=True)
            for proposal in proposals:
                if await self._try_execute(proposal, game):
                    break

    async def _try_execute(self, signal: Signal, game: GameState) -> bool:
        """Context check, cooldown, capital check, and publish.

        Returns True if a signal or reallocation was published.
        """
        # --- Fail-close context check ---
        status, reason = await self.bus.get_context(game.game_id)
        if status != ContextStatus.SAFE:
            self.log.info(
                "VETO for {} ({}): {}", game.game_id, signal.ticker, reason
            )
            return False

        # --- Per-ticker cooldown ---
        now = datetime.utcnow()
        last = self._signal_cooldowns.get(signal.ticker)
        if last and (now - last).total_seconds() < 60:
            return False

        # --- Per-game exposure cap ---
        # Arbitrage bypasses both checks: it's risk-free and needs both YES + NO
        # to fire on the same game. All other strategies enforce the pending lock
        # (race-condition guard) and the portfolio exposure cap.
        if signal.source != "arbitrage":
            if game.game_id in self._pending_game_orders:
                return False
            if self._get_game_exposure(game.game_id) >= self.settings.MAX_GAME_EXPOSURE:
                return False

        entry_price_cents = signal.entry_price

        if self._can_fund_trade(entry_price_cents):
            self._signal_cooldowns[signal.ticker] = now
            self._pending_game_orders.add(game.game_id)
            await self.bus.publish("signal:validated", signal)
            self.log.info(
                "+EV signal [{}]: {} EV={:.4f} entry={} exit={} conf={:.3f}",
                signal.source, signal.ticker, signal.ev_estimate,
                signal.entry_price, signal.exit_price, signal.confidence,
            )
            return True

        if self._global_realloc_lock and now < self._global_realloc_lock:
            return False

        reallocated = await self._try_reallocate(
            signal.ev_estimate, entry_price_cents,
            signal.confidence, self._markets[signal.ticker], game,
        )
        if reallocated:
            self._global_realloc_lock = now + timedelta(seconds=3)
        return reallocated

    # ------------------------------------------------------------------
    # Capital awareness
    # ------------------------------------------------------------------

    def _can_fund_trade(self, entry_price_cents: int) -> bool:
        if self._portfolio is None:
            return True
        bankroll_cents = self._portfolio.bankroll * 100
        return bankroll_cents >= entry_price_cents

    @staticmethod
    def _taker_fee_cents(contracts: int, price_cents: int) -> float:
        """Kalshi taker fee: ceil(0.07 * contracts * price * (1 - price))."""
        if price_cents <= 0 or price_cents >= 100:
            return 0.0
        p = price_cents / 100.0
        raw_cents = 0.07 * contracts * p * (1.0 - p) * 100
        return float(math.ceil(raw_cents))

    async def _try_reallocate(
        self,
        new_ev: float,
        new_entry_price: int,
        model_prob: float,
        market: MarketState,
        game: GameState,
    ) -> bool:
        """Evaluate whether liquidating a resting exit frees enough capital
        to fund a strictly better trade (unit-correct hurdle rate).

        Returns True if a reallocation signal was published.
        """
        if self._portfolio is None or not self._portfolio.positions:
            return False

        for pos in self._portfolio.positions:
            ms = self._markets.get(pos.ticker)
            if ms is None:
                continue

            live_bid = ms.yes_bid
            if live_bid < self.settings.MIN_REALLOCATE_BID:
                continue

            freed_capital_cents = live_bid * pos.remaining_count
            expected_new_count = math.floor(
                freed_capital_cents / new_entry_price
            ) if new_entry_price > 0 else 0
            if expected_new_count < 1:
                continue

            total_new_ev_cents = expected_new_count * (new_ev * 100)
            foregone_profit = (pos.target_exit_price - live_bid) * pos.remaining_count
            fees = self._taker_fee_cents(pos.remaining_count, live_bid)

            # Probability-weight: resting exit not guaranteed to fill
            expected_foregone = foregone_profit * self.settings.FOREGONE_PROFIT_MULTIPLIER

            # Time-decay: longer-stuck positions get lower hurdle
            if pos.created_at:
                created = pos.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_minutes = (
                    datetime.now(timezone.utc) - created
                ).total_seconds() / 60.0
                decay = max(
                    self.settings.REALLOCATE_DECAY_FLOOR,
                    1.0
                    - (age_minutes / self.settings.REALLOCATE_DECAY_MINUTES)
                    * (1.0 - self.settings.REALLOCATE_DECAY_FLOOR),
                )
                expected_foregone *= decay

            if total_new_ev_cents <= expected_foregone + fees:
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
                "REALLOCATE: liquidate {} x{} @{} (foregone={:.0f}c→{:.0f}c hurdle, fees={:.0f}c) "
                "for new EV={:.0f}c on {}",
                pos.ticker, pos.remaining_count, live_bid,
                foregone_profit, expected_foregone, fees, total_new_ev_cents, market.ticker,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Per-game exposure tracking
    # ------------------------------------------------------------------

    def _get_game_exposure(self, game_id: str) -> int:
        """Count open (not yet cashed-out) positions for this game.

        Uses portfolio positions (resting exit orders) as the source of truth.
        98c auto-cashout fills remove positions from the portfolio immediately,
        so cashed-out slots are automatically available for reinvestment.
        """
        if not self._portfolio or not self._portfolio.positions:
            return 0
        valid_tickers = self._game_to_tickers.get(game_id, [])
        return sum(
            1 for pos in self._portfolio.positions
            if pos.ticker in valid_tickers and pos.remaining_count > 0
        )

    # ------------------------------------------------------------------
    # Market mapping
    # ------------------------------------------------------------------

    def _auto_map_tickers(self) -> None:
        """Map live games to all matching Kalshi NBA market tickers.

        A game can map to multiple tickers (moneyline, totals, spreads,
        player props). Team-level markets require both team abbreviations.
        Player prop tickers (containing ``PLAYERPTS`` / ``PTS``) only
        need a single team abbreviation since they reference an individual.
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

                is_prop = "PLAYERPTS" in ticker_upper or (
                    "PTS" in ticker_upper
                    and "GAME" not in ticker_upper
                    and "TOTAL" not in ticker_upper
                )
                is_first_half = "1H" in ticker_upper or "HALF" in ticker_upper
                if is_prop:
                    if home in ticker_upper or away in ticker_upper:
                        matched.append(ticker)
                elif is_first_half:
                    if home in ticker_upper and away in ticker_upper:
                        matched.append(ticker)
                elif "GAME" in ticker_upper or "TOTAL" in ticker_upper:
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

    # ------------------------------------------------------------------
    # EV-based bailout monitor
    # ------------------------------------------------------------------

    async def _manage_open_positions(self) -> None:
        """Background loop: re-evaluate every open position every BAILOUT_POLL_INTERVAL.

        If the model's fair value has fallen well below the current bid
        (meaning the market now agrees we are losing), fire an aggressive
        BAILOUT sell to cut losses before the position expires worthless.
        """
        while self._running:
            await asyncio.sleep(self.settings.BAILOUT_POLL_INTERVAL)
            if not self._portfolio or not self._portfolio.positions:
                continue
            for pos in list(self._portfolio.positions):
                await self._check_bailout(pos)

    async def _check_bailout(self, pos: PortfolioPosition) -> None:
        """Re-evaluate a single position; fire BAILOUT signal if warranted."""
        # Per-ticker bailout cooldown (60 s) to avoid signal storms.
        now = datetime.utcnow()
        last = self._bailout_cooldowns.get(pos.client_order_id)
        if last and (now - last).total_seconds() < 60:
            return

        market = self._markets.get(pos.ticker)
        if market is None:
            return

        game = self._find_game_for_ticker(pos.ticker)
        if game is None:
            return

        # Ask every strategy that can price this market for its raw model prob.
        # model_prob is always P(YES wins) regardless of our held side.
        model_prob: float | None = None
        for strategy in self._strategies:
            if strategy.can_evaluate(market):
                model_prob = strategy.model_probability(game, market)
                if model_prob is not None:
                    break

        if model_prob is None:
            return

        # Side-aware fair value and current bid.
        # If we hold YES: fair value = P(YES) cents, sell at yes_bid.
        # If we hold NO:  fair value = P(NO) = (1 - P(YES)) cents, sell at no_bid.
        if pos.side == Side.YES:
            current_bid = market.yes_bid
            fair_value_cents = model_prob * 100.0
        else:
            current_bid = market.no_bid
            fair_value_cents = (1.0 - model_prob) * 100.0

        if current_bid <= 0:
            return

        bailout_threshold = current_bid - self.settings.BAILOUT_MARGIN_CENTS

        if fair_value_cents > bailout_threshold:
            return

        self._bailout_cooldowns[pos.client_order_id] = now
        self.log.warning(
            "BAILOUT triggered: {} ({}) fair={:.1f}c bid={}c threshold={}c — cutting loss",
            pos.ticker, pos.side, fair_value_cents, current_bid, bailout_threshold,
        )

        signal = Signal(
            ticker=pos.ticker,
            action=Action.SELL,
            side=pos.side,
            status=SignalStatus.BAILOUT,
            confidence=fair_value_cents / 100.0,
            source=self.name,
            ev_estimate=fair_value_cents / 100.0 - current_bid / 100.0,
            entry_price=current_bid,
            exit_price=pos.target_exit_price,
            game_id=game.game_id,
            target_order_id=pos.client_order_id,
        )
        await self.bus.publish("signal:bailout", signal)

    def _find_game_for_ticker(self, ticker: str) -> GameState | None:
        """Return the GameState whose market mapping includes *ticker*."""
        for game_id, tickers in self._game_to_tickers.items():
            if ticker in tickers:
                return self._games.get(game_id)
        return None

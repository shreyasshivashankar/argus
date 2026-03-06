"""Flash crash liquidity snatcher — captures mean-reversion bounces.

Prediction markets frequently overreact to short-term variance.  This
strategy monitors a rolling window of ``yes_bid`` prices for each ticker
and fires when a rapid drop is detected *without* a corresponding score
run by the opposing team.  The edge is purely behavioral (panic selling),
so we enter aggressively and exit with a tight spread to capture the
immediate bounce.

All state lives in-memory via ``collections.deque`` for microsecond
latency — no Redis or API calls on the hot path.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from agents.strategies.base import BaseStrategy
from core.schemas import Action, GameState, MarketState, Side, Signal, SignalStatus


class FlashCrashStrategy(BaseStrategy):
    name = "flash_crash"

    def __init__(
        self,
        window_seconds: float = 10.0,
        drop_threshold_cents: int = 20,
        exit_spread: int = 6,
        min_price_cents: int = 15,
        score_delta_limit: int = 3,
    ) -> None:
        self._window_seconds = window_seconds
        self._drop_threshold = drop_threshold_cents
        self._exit_spread = exit_spread
        self._min_price = min_price_cents
        self._score_delta_limit = score_delta_limit

        self._price_history: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=500)
        )
        self._score_history: dict[str, tuple[float, int, int]] = {}

    def can_evaluate(self, market: MarketState) -> bool:
        return market.ticker.upper().startswith("KXNBA")

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        now = time.monotonic()
        ticker = market.ticker
        bid = market.yes_bid

        self._record_price(ticker, now, bid)

        result = self._evaluate_inner(game, market, now, ticker, bid)
        self._record_score(game, now)
        return result

    def _evaluate_inner(
        self,
        game: GameState,
        market: MarketState,
        now: float,
        ticker: str,
        bid: int,
    ) -> Signal | None:
        if bid < self._min_price:
            return None

        window_max = self._window_max(ticker, now)
        if window_max is None:
            return None

        drop = window_max - bid
        if drop < self._drop_threshold:
            return None

        if self._score_run_detected(game, now):
            return None

        entry_price = market.yes_ask
        if entry_price <= 0:
            return None

        exit_price = min(entry_price + self._exit_spread, 99)

        ev = (drop / 100.0) * 0.6 - (entry_price / 100.0) * 0.05
        confidence = min(drop / (self._drop_threshold * 2.0), 1.0)

        return Signal(
            ticker=ticker,
            action=Action.BUY,
            side=Side.YES,
            status=SignalStatus.VALIDATED,
            confidence=confidence,
            source=self.name,
            ev_estimate=ev,
            entry_price=entry_price,
            exit_price=exit_price,
            game_id=game.game_id,
        )

    def _record_price(self, ticker: str, now: float, bid: int) -> None:
        buf = self._price_history[ticker]
        buf.append((now, bid))
        while buf and buf[0][0] < now - self._window_seconds * 3:
            buf.popleft()

    def _window_max(self, ticker: str, now: float) -> int | None:
        buf = self._price_history[ticker]
        cutoff = now - self._window_seconds
        prices_in_window = [price for ts, price in buf if ts >= cutoff]
        if len(prices_in_window) < 2:
            return None
        return max(prices_in_window)

    def _record_score(self, game: GameState, now: float) -> None:
        self._score_history[game.game_id] = (
            now, game.home_score, game.away_score,
        )

    def _score_run_detected(self, game: GameState, now: float) -> bool:
        """True if either team went on a run exceeding SCORE_DELTA_LIMIT
        during the lookback window — meaning the drop is likely justified."""
        prev = self._score_history.get(game.game_id)
        if prev is None:
            return True
        prev_ts, prev_home, prev_away = prev
        if now - prev_ts > self._window_seconds:
            return True

        home_run = abs(game.home_score - prev_home)
        away_run = abs(game.away_score - prev_away)
        return max(home_run, away_run) > self._score_delta_limit

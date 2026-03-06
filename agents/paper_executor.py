"""PaperExecutor — simulated matching engine for paper trading.

Subscribes to the same ``signal:validated`` channel as the real OrderExecutor
but never touches the Kalshi API.  Instead it operates as a realistic local
exchange simulator:

    1. On validated signal → simulate entry fill at current ask.
    2. Place a simulated exit limit sell into ``_resting_exits``.
    3. Subscribe to ``market:state`` from Redis.  On each update, check if
       the real order book's ``yes_bid`` has touched or exceeded any resting
       exit's target price.
    4. Only when the real bid touches the limit does it register a fill and
       log hypothetical P&L.

No optimistic fills — if the bid never reaches target, the exit stays resting.
"""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import (
    Action,
    AppSettings,
    MarketState,
    Signal,
    SignalStatus,
)


@dataclass
class PaperPosition:
    """Tracks a single entry→exit paper trade."""
    signal_id: str
    ticker: str
    side: str
    count: int
    entry_price: int
    exit_target: int
    entry_time: datetime = field(default_factory=datetime.utcnow)
    filled: bool = False


class PaperExecutor(BaseAgent):
    """Paper trading executor with a simulated matching engine.

    Uses identical Kelly sizing logic and kill switch as the production
    OrderExecutor, but all fills are simulated against the live order book.
    """

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
    ) -> None:
        super().__init__("paper_executor", settings, bus, client)

        self.current_bankroll: float = 0.0
        self._resting_exits: dict[str, PaperPosition] = {}
        self._daily_realized_pnl: float = 0.0
        self._kill_switch_tripped: bool = False
        self._trade_count: int = 0
        self._win_count: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        balance_task = asyncio.create_task(self._balance_poll_loop())
        signal_task = asyncio.create_task(
            self.bus.subscribe(["signal:validated"], self._on_signal)
        )
        market_task = asyncio.create_task(
            self.bus.subscribe(["market:state"], self._on_market_update)
        )
        gc_task = asyncio.create_task(self._gc_loop())
        await asyncio.gather(balance_task, signal_task, market_task, gc_task)

    # ------------------------------------------------------------------
    # Background bankroll cache (same as production)
    # ------------------------------------------------------------------

    _DEFAULT_PAPER_BANKROLL: float = 1000.0

    async def _balance_poll_loop(self) -> None:
        while self._running:
            try:
                balance = await self.client.get_balance()
                self.current_bankroll = balance if balance > 0 else self._DEFAULT_PAPER_BANKROLL
            except Exception:
                if self.current_bankroll <= 0:
                    self.current_bankroll = self._DEFAULT_PAPER_BANKROLL
                self.log.debug("Balance poll failed, using paper bankroll ${:.2f}", self.current_bankroll)
            self.log.info(
                "[PAPER] Bankroll: ${:.2f} | {} resting exits",
                self.current_bankroll, len(self._resting_exits),
            )
            await asyncio.sleep(self.settings.BALANCE_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    async def _on_signal(self, _channel: str, data: dict[str, Any]) -> None:
        try:
            signal = Signal(**data)
        except Exception:
            self.log.warning("Bad signal payload: {}", data)
            return

        if signal.status != SignalStatus.VALIDATED:
            return

        if self._kill_switch_tripped:
            self.log.warning("[PAPER] Kill switch active — dropping signal")
            return

        await self._simulate_entry(signal)

    # ------------------------------------------------------------------
    # Simulated entry
    # ------------------------------------------------------------------

    async def _simulate_entry(self, signal: Signal) -> None:
        count = self._kelly_size(signal)
        if count < 1:
            self.log.info("[PAPER] Kelly size < 1, skipping {}", signal.ticker)
            return

        entry_price = signal.entry_price + self.settings.SLIPPAGE_TICKS
        exit_target = entry_price + self.settings.TARGET_EXIT_SPREAD - self.settings.SLIPPAGE_TICKS

        position = PaperPosition(
            signal_id=signal.signal_id,
            ticker=signal.ticker,
            side=signal.side,
            count=count,
            entry_price=entry_price,
            exit_target=exit_target,
        )

        pos_id = str(uuid.uuid4())
        self._resting_exits[pos_id] = position

        self.log.info(
            "[PAPER] Entry: {} {} x{} @{} → exit target @{}",
            signal.ticker, signal.side, count, entry_price, exit_target,
        )

    # ------------------------------------------------------------------
    # Market state monitor — simulated matching engine
    # ------------------------------------------------------------------

    async def _on_market_update(self, _channel: str, data: dict[str, Any]) -> None:
        try:
            market = MarketState(**data)
        except Exception:
            return

        filled_ids: list[str] = []
        for pos_id, pos in self._resting_exits.items():
            if pos.filled or pos.ticker != market.ticker:
                continue

            if market.yes_bid >= pos.exit_target:
                pos.filled = True
                filled_ids.append(pos_id)
                await self._simulate_exit_fill(pos, market.yes_bid)

        for pos_id in filled_ids:
            del self._resting_exits[pos_id]

    async def _simulate_exit_fill(self, pos: PaperPosition, fill_price: int) -> None:
        pnl_cents = (fill_price - pos.entry_price) * pos.count
        pnl_dollars = pnl_cents / 100.0
        self._daily_realized_pnl += pnl_dollars
        self._trade_count += 1
        if pnl_dollars > 0:
            self._win_count += 1

        win_rate = (self._win_count / self._trade_count * 100) if self._trade_count > 0 else 0

        self.log.info(
            "[PAPER] Exit fill: {} x{} entry@{} exit@{} P&L=${:.2f} "
            "(daily=${:.2f}, trades={}, win_rate={:.1f}%)",
            pos.ticker, pos.count, pos.entry_price, fill_price,
            pnl_dollars, self._daily_realized_pnl,
            self._trade_count, win_rate,
        )

        await self._check_kill_switch()

    # ------------------------------------------------------------------
    # Kill switch (identical logic to production)
    # ------------------------------------------------------------------

    async def _check_kill_switch(self) -> None:
        if self._daily_realized_pnl >= 0:
            return
        if abs(self._daily_realized_pnl) >= self.settings.DAILY_STOP_LOSS_USD:
            self._kill_switch_tripped = True
            self._resting_exits.clear()
            self.log.critical(
                "[PAPER] KILL SWITCH: daily loss ${:.2f} exceeds limit ${:.2f}",
                abs(self._daily_realized_pnl),
                self.settings.DAILY_STOP_LOSS_USD,
            )

    # ------------------------------------------------------------------
    # Garbage collection for orphaned resting exits
    # ------------------------------------------------------------------

    async def _gc_loop(self) -> None:
        """Evict resting exits older than ORDER_GC_TTL to prevent memory leaks."""
        while self._running:
            await asyncio.sleep(self.settings.ORDER_GC_INTERVAL)
            now = datetime.utcnow()
            stale = [
                pid for pid, pos in self._resting_exits.items()
                if not pos.filled
                and (now - pos.entry_time).total_seconds() > self.settings.ORDER_GC_TTL
            ]
            for pid in stale:
                pos = self._resting_exits.pop(pid)
                self.log.info(
                    "[PAPER] GC: evicted stale exit {} @{} (age={:.0f}s)",
                    pos.ticker, pos.exit_target,
                    (now - pos.entry_time).total_seconds(),
                )
            if stale:
                self.log.debug("[PAPER] GC: evicted {} stale positions", len(stale))

    # ------------------------------------------------------------------
    # Kelly criterion (identical to production OrderExecutor)
    # ------------------------------------------------------------------

    def _kelly_size(self, signal: Signal) -> int:
        if self.current_bankroll <= 0:
            return 0

        entry_cents = signal.entry_price
        if entry_cents <= 0 or entry_cents >= 100:
            return 0

        p = signal.confidence
        q = 1.0 - p
        b = (100.0 - entry_cents) / entry_cents

        kelly_full = (p * b - q) / b if b > 0 else 0
        kelly_fraction = max(kelly_full * self.settings.KELLY_FRACTION, 0)

        bankroll_cents = self.current_bankroll * 100
        bet_cents = kelly_fraction * bankroll_cents
        count = math.floor(bet_cents / entry_cents)

        return max(count, 0)

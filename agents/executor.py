from __future__ import annotations

import asyncio
import math
import subprocess
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import (
    Action,
    AppSettings,
    ManagedOrder,
    Order,
    OrderState,
    Signal,
    SignalStatus,
)


class OrderExecutor(BaseAgent):
    """Fill-aware execution state machine with Kelly sizing and a kill switch.

    Lifecycle per trade:
        VALIDATED signal → place limit buy (Kelly-sized)
        → wait for Kalshi fill WS event → place paired limit sell
        → wait for exit fill → done

    Invariants enforced:
        - Bankroll is background-cached (no REST in the hot path)
        - Exit orders only dispatch after fill confirms inventory
        - Daily stop-loss halts all trading and sends Telegram alert
    """

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
    ) -> None:
        super().__init__("executor", settings, bus, client)

        # Background-cached bankroll (updated every BALANCE_POLL_INTERVAL)
        self.current_bankroll: float = 0.0

        # Active managed orders keyed by client_order_id
        self._orders: dict[str, ManagedOrder] = {}

        # Reverse lookup: kalshi_order_id → client_order_id
        self._kalshi_to_client: dict[str, str] = {}

        # Daily P&L tracking
        self._daily_realized_pnl: float = 0.0
        self._kill_switch_tripped: bool = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        balance_task = asyncio.create_task(self._balance_poll_loop())
        signal_task = asyncio.create_task(
            self.bus.subscribe(["signal:validated"], self._on_signal)
        )
        await asyncio.gather(balance_task, signal_task)

    # ------------------------------------------------------------------
    # Background bankroll cache
    # ------------------------------------------------------------------

    async def _balance_poll_loop(self) -> None:
        while self._running:
            try:
                self.current_bankroll = await self.client.get_balance()
                self.log.debug("Bankroll cached: ${:.2f}", self.current_bankroll)
            except Exception:
                self.log.exception("Balance poll failed")
            await asyncio.sleep(self.settings.BALANCE_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Signal handling
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
            self.log.warning("Kill switch active — dropping signal {}", signal.signal_id)
            return

        await self._execute_entry(signal)

    # ------------------------------------------------------------------
    # Entry order
    # ------------------------------------------------------------------

    async def _execute_entry(self, signal: Signal) -> None:
        count = self._kelly_size(signal)
        if count < 1:
            self.log.info("Kelly size < 1 contract, skipping {}", signal.ticker)
            return

        entry_price = signal.entry_price + self.settings.SLIPPAGE_TICKS

        order = Order(
            ticker=signal.ticker,
            action=Action.BUY,
            side=signal.side,
            count=count,
            yes_price=entry_price if signal.side == "yes" else None,
            no_price=entry_price if signal.side == "no" else None,
        )

        managed = ManagedOrder(
            order=order,
            state=OrderState.PLACED,
            signal_id=signal.signal_id,
        )
        self._orders[order.client_order_id] = managed

        try:
            resp = await self.client.place_order(order)
            kalshi_id = resp.get("order", {}).get("order_id", "")
            managed.kalshi_order_id = kalshi_id
            self._kalshi_to_client[kalshi_id] = order.client_order_id
            self.log.info(
                "Entry placed: {} {} x{} @{} (kalshi_id={})",
                signal.ticker, signal.side, count, entry_price, kalshi_id,
            )
        except Exception:
            self.log.exception("Failed to place entry for {}", signal.ticker)
            managed.state = OrderState.CANCELED
            return

    # ------------------------------------------------------------------
    # Fill callback (registered by main.py on the KalshiFeedWatcher)
    # ------------------------------------------------------------------

    async def on_fill(self, msg: dict[str, Any]) -> None:
        """Handle a fill event from the Kalshi WS fill channel."""
        order_id = msg.get("order_id", "")
        client_id = (
            msg.get("client_order_id")
            or self._kalshi_to_client.get(order_id, "")
        )

        managed = self._orders.get(client_id)
        if managed is None:
            return

        fill_count = int(msg.get("count", 0))
        fill_price = int(msg.get("yes_price", 0))
        post_position = int(msg.get("post_position", 0))

        managed.fill_count += fill_count
        managed.remaining_count = max(
            managed.order.count - managed.fill_count, 0
        )

        if managed.remaining_count == 0:
            managed.state = OrderState.FILLED
        else:
            managed.state = OrderState.PARTIALLY_FILLED

        self.log.info(
            "Fill on {}: +{} @{} (total filled={}/{})",
            managed.order.ticker, fill_count, fill_price,
            managed.fill_count, managed.order.count,
        )

        # If this is an entry order that just filled → place the exit
        if managed.state == OrderState.FILLED and not managed.paired_exit_order_id:
            await self._place_exit(managed)

        # If this is an exit order that just filled → trade complete
        if managed.paired_exit_order_id and managed.state == OrderState.FILLED:
            await self._trade_complete(managed, fill_price)

    async def on_order_update(self, msg: dict[str, Any]) -> None:
        """Handle an order status update from the Kalshi WS user_orders channel.

        Used as reconciliation fallback alongside the fill channel.
        """
        order_id = msg.get("order_id", "")
        client_id = (
            msg.get("client_order_id")
            or self._kalshi_to_client.get(order_id, "")
        )
        managed = self._orders.get(client_id)
        if managed is None:
            return

        status = msg.get("status", "")
        if status == "resting" and managed.state == OrderState.PLACED:
            managed.state = OrderState.RESTING
        elif status == "canceled":
            managed.state = OrderState.CANCELED
            self.log.warning("Order canceled by exchange: {}", order_id)

    # ------------------------------------------------------------------
    # Exit order (only after fill confirms inventory)
    # ------------------------------------------------------------------

    async def _place_exit(self, entry: ManagedOrder) -> None:
        """Place the paired limit sell at TARGET_EXIT_SPREAD above entry."""
        exit_price = entry.order.yes_price or entry.order.no_price or 0
        exit_price += self.settings.TARGET_EXIT_SPREAD - self.settings.SLIPPAGE_TICKS

        exit_order = Order(
            ticker=entry.order.ticker,
            action=Action.SELL,
            side=entry.order.side,
            count=entry.fill_count,
            yes_price=exit_price if entry.order.side == "yes" else None,
            no_price=exit_price if entry.order.side == "no" else None,
        )

        exit_managed = ManagedOrder(
            order=exit_order,
            state=OrderState.PLACED,
            signal_id=entry.signal_id,
        )
        self._orders[exit_order.client_order_id] = exit_managed
        entry.paired_exit_order_id = exit_order.client_order_id

        try:
            resp = await self.client.place_order(exit_order)
            kalshi_id = resp.get("order", {}).get("order_id", "")
            exit_managed.kalshi_order_id = kalshi_id
            self._kalshi_to_client[kalshi_id] = exit_order.client_order_id
            self.log.info(
                "Exit placed: {} SELL x{} @{} (kalshi_id={})",
                exit_order.ticker, exit_order.count, exit_price, kalshi_id,
            )
        except Exception:
            self.log.exception("Failed to place exit for {}", exit_order.ticker)

    # ------------------------------------------------------------------
    # Trade completion + P&L
    # ------------------------------------------------------------------

    async def _trade_complete(self, exit_managed: ManagedOrder, exit_price: int) -> None:
        entry_id = None
        for cid, m in self._orders.items():
            if m.paired_exit_order_id == exit_managed.order.client_order_id:
                entry_id = cid
                break

        entry_price = 0
        if entry_id:
            entry_managed = self._orders[entry_id]
            entry_price = entry_managed.order.yes_price or entry_managed.order.no_price or 0

        pnl_cents = (exit_price - entry_price) * exit_managed.fill_count
        pnl_dollars = pnl_cents / 100.0
        self._daily_realized_pnl += pnl_dollars

        self.log.info(
            "Trade complete: {} P&L=${:.2f} (daily=${:.2f})",
            exit_managed.order.ticker, pnl_dollars, self._daily_realized_pnl,
        )

        signal = Signal(
            ticker=exit_managed.order.ticker,
            action=Action.SELL,
            side=exit_managed.order.side,
            status=SignalStatus.EXECUTED,
            confidence=1.0,
            source=self.name,
            ev_estimate=pnl_dollars,
            entry_price=entry_price,
            exit_price=exit_price,
            game_id="",
            signal_id=exit_managed.signal_id,
        )
        await self.bus.publish("signal:executed", signal)

        await self._check_kill_switch()

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    async def _check_kill_switch(self) -> None:
        if self._daily_realized_pnl >= 0:
            return
        if abs(self._daily_realized_pnl) >= self.settings.DAILY_STOP_LOSS_USD:
            self._kill_switch_tripped = True
            self.log.critical(
                "KILL SWITCH: daily loss ${:.2f} exceeds limit ${:.2f}",
                abs(self._daily_realized_pnl),
                self.settings.DAILY_STOP_LOSS_USD,
            )
            await self._cancel_all_resting()
            await self._send_telegram_alert()

    async def _cancel_all_resting(self) -> None:
        for managed in self._orders.values():
            if managed.state in (OrderState.PLACED, OrderState.RESTING):
                if managed.kalshi_order_id:
                    try:
                        await self.client.cancel_order(managed.kalshi_order_id)
                        managed.state = OrderState.CANCELED
                        self.log.warning("Canceled resting order: {}", managed.kalshi_order_id)
                    except Exception:
                        self.log.exception("Failed to cancel {}", managed.kalshi_order_id)

    async def _send_telegram_alert(self) -> None:
        msg = (
            f"🚨 ARGUS KILL SWITCH TRIPPED\n"
            f"Daily loss: ${abs(self._daily_realized_pnl):.2f}\n"
            f"Limit: ${self.settings.DAILY_STOP_LOSS_USD:.2f}\n"
            f"All trading halted. All resting orders canceled."
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "telegram-send", msg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self.log.info("Telegram alert sent")
        except Exception:
            self.log.exception("Failed to send Telegram alert")

    # ------------------------------------------------------------------
    # Kelly criterion sizing
    # ------------------------------------------------------------------

    def _kelly_size(self, signal: Signal) -> int:
        """Fractional Kelly Criterion: f* = (p*b - q) / b, scaled by fraction.

        Uses self.current_bankroll (background-cached, no REST call here).
        """
        if self.current_bankroll <= 0:
            return 0

        entry_cents = signal.entry_price
        if entry_cents <= 0 or entry_cents >= 100:
            return 0

        p = signal.confidence
        q = 1.0 - p
        b = (100.0 - entry_cents) / entry_cents  # payout odds

        kelly_full = (p * b - q) / b if b > 0 else 0
        kelly_fraction = max(kelly_full * self.settings.KELLY_FRACTION, 0)

        bankroll_cents = self.current_bankroll * 100
        bet_cents = kelly_fraction * bankroll_cents
        count = math.floor(bet_cents / entry_cents)

        return max(count, 0)

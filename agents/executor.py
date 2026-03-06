from __future__ import annotations

import asyncio
import math
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
    PortfolioPosition,
    PortfolioState,
    Signal,
    SignalStatus,
    Side,
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

        # Deferred fills: fills that arrived before place_order returned.
        # Keyed by kalshi_order_id so they can be replayed once the mapping exists.
        self._deferred_fills: list[dict[str, Any]] = []

        # Cancel confirmation events: on_order_update sets these when
        # Kalshi WS confirms a cancel, so _execute_reallocate can wait
        # deterministically instead of a blind sleep.
        self._cancel_events: dict[str, asyncio.Event] = {}

        # Daily P&L tracking
        self._daily_realized_pnl: float = 0.0
        self._kill_switch_tripped: bool = False

    # ------------------------------------------------------------------
    # Startup reconciliation — crash-only architecture
    # ------------------------------------------------------------------

    async def _reconcile_state_on_boot(self) -> None:
        """Synchronize in-memory state with Kalshi (the source of truth).

        On every startup we:
        1. Fetch the live cash balance.
        2. Fetch all resting orders from the exchange.
        3. For each resting order, reconstruct a ManagedOrder so that
           fill callbacks, reallocation, and the kill switch all work
           as if the bot had never restarted.
        4. Publish an accurate PortfolioState so the quant agent knows
           exactly how much capital is locked up.
        """
        self.log.info("Reconciling state with Kalshi...")

        try:
            self.current_bankroll = await self.client.get_balance()
        except Exception:
            self.log.exception("Failed to fetch balance during reconciliation")
            return

        try:
            resting = await self.client.get_orders(status="resting")
        except Exception:
            self.log.exception("Failed to fetch resting orders during reconciliation")
            return

        if not resting:
            self.log.info(
                "No resting orders on Kalshi — clean slate (bankroll=${:.2f})",
                self.current_bankroll,
            )
            return

        for raw in resting:
            kalshi_id = raw.get("order_id", "")
            client_id = raw.get("client_order_id") or kalshi_id
            ticker = raw.get("ticker", "")
            action_str = raw.get("action", "buy")
            side_str = raw.get("side", "yes")
            yes_price = raw.get("yes_price")
            no_price = raw.get("no_price")
            remaining = int(raw.get("remaining_count", 0))
            fill_count = int(raw.get("fill_count", 0))
            initial_count = int(raw.get("initial_count", remaining + fill_count))

            is_exit = action_str == "sell"

            order = Order(
                ticker=ticker,
                action=Action(action_str),
                side=Side(side_str),
                count=initial_count,
                yes_price=yes_price,
                no_price=no_price,
                client_order_id=client_id,
            )

            vwap = 0.0
            if fill_count > 0:
                try:
                    fills = await self.client.get_fills(kalshi_id)
                    total_cost = sum(
                        int(f.get("yes_price", 0)) * int(f.get("count", 0))
                        for f in fills
                    )
                    vwap = total_cost / fill_count if fill_count else 0.0
                except Exception:
                    self.log.warning(
                        "Could not fetch fills for {} — using limit price as VWAP",
                        kalshi_id,
                    )
                    vwap = float(yes_price or no_price or 0)

            managed = ManagedOrder(
                order=order,
                state=OrderState.RESTING,
                kalshi_order_id=kalshi_id,
                fill_count=fill_count,
                remaining_count=remaining,
                signal_id="",
                is_exit=is_exit,
                vwap_cents=vwap,
            )

            self._orders[client_id] = managed
            self._kalshi_to_client[kalshi_id] = client_id

            label = "EXIT" if is_exit else "ENTRY"
            price = yes_price or no_price or 0
            self.log.info(
                "Recovered {} order: {} {} x{} @{}c (filled={}, kalshi={})",
                label, ticker, side_str, remaining, price, fill_count, kalshi_id,
            )

        await self._publish_portfolio()
        self.log.info(
            "Reconciliation complete: {} orders recovered, bankroll=${:.2f}",
            len(self._orders), self.current_bankroll,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.log.info("Booting Executor — synchronizing state with Kalshi")
        await self._reconcile_state_on_boot()
        self.log.info(
            "State synced. {} resting orders recovered.", len(self._orders)
        )

        balance_task = asyncio.create_task(self._balance_poll_loop())
        signal_task = asyncio.create_task(
            self.bus.subscribe(
                ["signal:validated", "signal:reallocate"], self._on_signal
            )
        )
        gc_task = asyncio.create_task(self._gc_loop())
        await asyncio.gather(balance_task, signal_task, gc_task)

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

            await self._publish_portfolio()
            await asyncio.sleep(self.settings.BALANCE_POLL_INTERVAL)

    async def _publish_portfolio(self) -> None:
        """Build a PortfolioState snapshot and publish to portfolio:state."""
        resting_states = {OrderState.PLACED, OrderState.RESTING}
        positions: list[PortfolioPosition] = []
        for cid, managed in self._orders.items():
            if not managed.is_exit or managed.state not in resting_states:
                continue
            entry = self._orders.get(managed.parent_entry_id or "")
            entry_vwap = entry.vwap_cents if entry else 0.0
            exit_price = managed.order.yes_price or managed.order.no_price or 0
            positions.append(PortfolioPosition(
                client_order_id=cid,
                ticker=managed.order.ticker,
                side=managed.order.side,
                remaining_count=managed.order.count,
                entry_vwap=entry_vwap,
                target_exit_price=exit_price,
                kalshi_order_id=managed.kalshi_order_id or "",
                created_at=managed.created_at,
            ))
        state = PortfolioState(bankroll=self.current_bankroll, positions=positions)
        await self.bus.publish("portfolio:state", state)

    # ------------------------------------------------------------------
    # Order garbage collection
    # ------------------------------------------------------------------

    async def _gc_loop(self) -> None:
        """Periodically evict terminal orders older than ORDER_GC_TTL seconds."""
        terminal_states = {OrderState.FILLED, OrderState.CANCELED}
        while self._running:
            await asyncio.sleep(self.settings.ORDER_GC_INTERVAL)
            now = datetime.utcnow()
            stale = [
                cid for cid, m in self._orders.items()
                if m.state in terminal_states
                and (now - m.created_at).total_seconds() > self.settings.ORDER_GC_TTL
            ]
            for cid in stale:
                managed = self._orders.pop(cid, None)
                if managed and managed.kalshi_order_id:
                    self._kalshi_to_client.pop(managed.kalshi_order_id, None)
            if stale:
                self.log.debug("GC: evicted {} terminal orders", len(stale))

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    async def _on_signal(self, _channel: str, data: dict[str, Any]) -> None:
        try:
            signal = Signal.model_validate(data)
        except Exception:
            self.log.warning("Bad signal payload: {}", data)
            return

        if self._kill_switch_tripped:
            self.log.warning("Kill switch active — dropping signal {}", signal.signal_id)
            return

        if signal.status == SignalStatus.VALIDATED:
            await self._execute_entry(signal)
        elif signal.status == SignalStatus.REALLOCATE:
            await self._execute_reallocate(signal)

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

        bet_dollars = (count * entry_price) / 100.0
        self.current_bankroll -= bet_dollars

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
            self.current_bankroll += bet_dollars
            return

        await self._replay_deferred_fills()

    # ------------------------------------------------------------------
    # Reallocation (liquidate resting exit to free capital)
    # ------------------------------------------------------------------

    async def _execute_reallocate(self, signal: Signal) -> None:
        """Cancel a resting exit and immediately cross the spread to sell.

        The quant agent sets signal.target_order_id to the specific resting
        exit's client_order_id, and signal.entry_price to the current bid
        (the aggressive sell price).
        """
        target_id = signal.target_order_id
        if not target_id or target_id not in self._orders:
            self.log.warning("REALLOCATE: unknown target_order_id {}", target_id)
            return

        managed = self._orders[target_id]
        if not managed.is_exit or not managed.kalshi_order_id:
            self.log.warning("REALLOCATE: target {} is not a resting exit", target_id)
            return

        cancel_event = asyncio.Event()
        self._cancel_events[managed.kalshi_order_id] = cancel_event

        try:
            await self.client.cancel_order(managed.kalshi_order_id)
            self.log.info(
                "REALLOCATE: cancel sent for {} (kalshi={}), awaiting WS confirm",
                target_id, managed.kalshi_order_id,
            )
        except Exception:
            self._cancel_events.pop(managed.kalshi_order_id, None)
            self.log.exception("REALLOCATE: failed to cancel {}", managed.kalshi_order_id)
            return

        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=5.0)
            await asyncio.sleep(0.2)
        except asyncio.TimeoutError:
            self.log.warning(
                "REALLOCATE: cancel confirm timed out for {} — proceeding anyway",
                managed.kalshi_order_id,
            )
        finally:
            self._cancel_events.pop(managed.kalshi_order_id, None)

        managed.state = OrderState.CANCELED

        aggressive_price = signal.entry_price
        sell_order = Order(
            ticker=managed.order.ticker,
            action=Action.SELL,
            side=managed.order.side,
            count=managed.order.count,
            yes_price=aggressive_price if managed.order.side == Side.YES else None,
            no_price=aggressive_price if managed.order.side == Side.NO else None,
        )

        sell_managed = ManagedOrder(
            order=sell_order,
            state=OrderState.PLACED,
            signal_id=managed.signal_id,
            is_exit=True,
            parent_entry_id=managed.parent_entry_id,
        )
        self._orders[sell_order.client_order_id] = sell_managed

        try:
            resp = await self.client.place_order(sell_order)
            kalshi_id = resp.get("order", {}).get("order_id", "")
            sell_managed.kalshi_order_id = kalshi_id
            self._kalshi_to_client[kalshi_id] = sell_order.client_order_id
            self.log.info(
                "REALLOCATE: aggressive sell placed {} x{} @{} (kalshi={})",
                sell_order.ticker, sell_order.count, aggressive_price, kalshi_id,
            )
        except Exception:
            self.log.exception("REALLOCATE: failed to place aggressive sell")

        await self._replay_deferred_fills()

    # ------------------------------------------------------------------
    # Fill callback (registered by main.py on the KalshiFeedWatcher)
    # ------------------------------------------------------------------

    async def on_fill(self, msg: dict[str, Any]) -> None:
        """Handle a fill event from the Kalshi WS fill channel.

        Resolves by client_order_id first (always present if Kalshi echoes it),
        then falls back to kalshi_order_id lookup.  If neither resolves (fill
        arrived before place_order returned), the fill is deferred and replayed
        once the mapping is established.
        """
        client_id = msg.get("client_order_id", "")
        if not client_id or client_id not in self._orders:
            order_id = msg.get("order_id", "")
            client_id = self._kalshi_to_client.get(order_id, "")

        managed = self._orders.get(client_id)
        if managed is None:
            retries = msg.get("_retries", 0)
            if retries < 3:
                msg["_retries"] = retries + 1
                self._deferred_fills.append(msg)
                self.log.debug("Deferred fill (no mapping yet, attempt {}): {}", retries + 1, msg.get("order_id"))
            else:
                self.log.warning("Dropping orphaned fill after 3 retries: {}", msg.get("order_id"))
            return

        fill_count = int(msg.get("count", 0))
        fill_price = int(msg.get("yes_price", 0))

        prev_fill_count = managed.fill_count
        managed.fill_count += fill_count
        managed.remaining_count = max(
            managed.order.count - managed.fill_count, 0
        )

        # Update VWAP: weighted average of execution prices
        if managed.fill_count > 0:
            managed.vwap_cents = (
                (managed.vwap_cents * prev_fill_count) + (fill_price * fill_count)
            ) / managed.fill_count

        if managed.remaining_count == 0:
            managed.state = OrderState.FILLED
        else:
            managed.state = OrderState.PARTIALLY_FILLED

        self.log.info(
            "Fill on {}: +{} @{} (total filled={}/{}, vwap={:.1f})",
            managed.order.ticker, fill_count, fill_price,
            managed.fill_count, managed.order.count, managed.vwap_cents,
        )

        if managed.is_exit:
            await self._trade_complete(managed, fill_count, fill_price)
        elif fill_count > 0:
            await self._place_exit(managed, fill_count)

    async def _replay_deferred_fills(self) -> None:
        """Replay any fills that arrived before place_order returned."""
        if not self._deferred_fills:
            return
        pending = self._deferred_fills[:]
        self._deferred_fills.clear()
        for fill_msg in pending:
            await self.on_fill(fill_msg)

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
            cancel_ev = self._cancel_events.get(order_id)
            if cancel_ev:
                cancel_ev.set()
            else:
                self.log.warning("Order canceled by exchange: {}", order_id)

    # ------------------------------------------------------------------
    # Exit order (only after fill confirms inventory)
    # ------------------------------------------------------------------

    async def _place_exit(self, entry: ManagedOrder, batch_count: int) -> None:
        """Place a paired limit sell for a specific fill batch.

        Called on every fill event, not just full fill, so partial fills
        are hedged immediately instead of waiting for the full order.
        """
        exit_price = entry.order.yes_price or entry.order.no_price or 0
        exit_price += self.settings.TARGET_EXIT_SPREAD - self.settings.SLIPPAGE_TICKS

        exit_order = Order(
            ticker=entry.order.ticker,
            action=Action.SELL,
            side=entry.order.side,
            count=batch_count,
            yes_price=exit_price if entry.order.side == "yes" else None,
            no_price=exit_price if entry.order.side == "no" else None,
        )

        exit_managed = ManagedOrder(
            order=exit_order,
            state=OrderState.PLACED,
            signal_id=entry.signal_id,
            is_exit=True,
            parent_entry_id=entry.order.client_order_id,
        )
        self._orders[exit_order.client_order_id] = exit_managed
        entry.paired_exit_order_ids.append(exit_order.client_order_id)

        try:
            resp = await self.client.place_order(exit_order)
            kalshi_id = resp.get("order", {}).get("order_id", "")
            exit_managed.kalshi_order_id = kalshi_id
            self._kalshi_to_client[kalshi_id] = exit_order.client_order_id
            self.log.info(
                "Exit placed: {} SELL x{} @{} (kalshi_id={})",
                exit_order.ticker, batch_count, exit_price, kalshi_id,
            )
        except Exception:
            self.log.exception("Failed to place exit for {}", exit_order.ticker)

        await self._replay_deferred_fills()

    # ------------------------------------------------------------------
    # Trade completion + P&L
    # ------------------------------------------------------------------

    async def _trade_complete(
        self, exit_managed: ManagedOrder, batch_count: int, fill_price: int
    ) -> None:
        """Called on each exit fill. Computes P&L from execution VWAPs, not limit prices."""
        entry_vwap = 0.0
        if exit_managed.parent_entry_id:
            entry = self._orders.get(exit_managed.parent_entry_id)
            if entry:
                entry_vwap = entry.vwap_cents

        if exit_managed.order.side == Side.YES:
            pnl_cents = (fill_price - entry_vwap) * batch_count
        else:
            pnl_cents = (entry_vwap - fill_price) * batch_count
        pnl_dollars = pnl_cents / 100.0
        self._daily_realized_pnl += pnl_dollars

        revenue_dollars = (fill_price * batch_count) / 100.0
        self.current_bankroll += revenue_dollars

        self.log.info(
            "Exit fill: {} x{} @{} (entry vwap={:.1f}) P&L=${:.2f} (daily=${:.2f})",
            exit_managed.order.ticker, batch_count, fill_price,
            entry_vwap, pnl_dollars, self._daily_realized_pnl,
        )

        if exit_managed.state == OrderState.FILLED:
            signal = Signal(
                ticker=exit_managed.order.ticker,
                action=Action.SELL,
                side=exit_managed.order.side,
                status=SignalStatus.EXECUTED,
                confidence=1.0,
                source=self.name,
                ev_estimate=pnl_dollars,
                entry_price=int(entry_vwap),
                exit_price=int(exit_managed.vwap_cents),
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

    async def _cancel_resting_entries(self) -> int:
        """Cancel only resting entry orders; exit orders stay alive on the exchange.

        Returns the number of exit orders left resting.
        """
        exits_left = 0
        for managed in self._orders.values():
            if managed.state not in (OrderState.PLACED, OrderState.RESTING):
                continue
            if managed.is_exit:
                exits_left += 1
                continue
            if managed.kalshi_order_id:
                try:
                    await self.client.cancel_order(managed.kalshi_order_id)
                    managed.state = OrderState.CANCELED
                    self.log.warning("Canceled resting entry: {}", managed.kalshi_order_id)
                except Exception:
                    self.log.exception("Failed to cancel {}", managed.kalshi_order_id)
        return exits_left

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

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

import websockets
from loguru import logger

from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import MarketState


OnFillCallback = Callable[[dict[str, Any]], Awaitable[None]]
OnOrderCallback = Callable[[dict[str, Any]], Awaitable[None]]


class KalshiFeedWatcher:
    """Single authenticated WebSocket connection to Kalshi multiplexing:

    - orderbook_delta (private) — maintains local order book, publishes MarketState
    - ticker (public) — backup price feed
    - fill (private) — immediate fill notifications for executor
    - user_orders (private) — order status reconciliation

    Register callbacks via on_fill() and on_order_update() before calling run().
    """

    def __init__(
        self,
        client: KalshiAsyncClient,
        bus: SignalBus,
        market_tickers: list[str] | None = None,
    ) -> None:
        self._client = client
        self._bus = bus
        self._market_tickers = market_tickers or []
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._msg_id = 1
        self._running = True

        # Local order book: ticker -> {"yes": {price_cents: qty}, "no": {price_cents: qty}}
        # Dict avoids O(N log N) sort on every delta; best bid/ask via max/min
        self._orderbooks: dict[str, dict[str, dict[int, int]]] = {}

        # Executor callbacks
        self._fill_callbacks: list[OnFillCallback] = []
        self._order_callbacks: list[OnOrderCallback] = []

        # Strong refs to fire-and-forget callback tasks (prevents GC mid-flight)
        self._bg_tasks: set[asyncio.Task] = set()

        # Queue for snapshot-derived MarketState publishes (sync → async bridge)
        self._snapshot_publish_queue: list[MarketState] = []

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_fill(self, callback: OnFillCallback) -> None:
        self._fill_callbacks.append(callback)

    def on_order_update(self, callback: OnOrderCallback) -> None:
        self._order_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        while self._running:
            try:
                await self._connect_and_listen()
            except (
                websockets.ConnectionClosed,
                websockets.InvalidStatusCode,
                OSError,
            ) as exc:
                logger.warning("Kalshi WS disconnected: {}. Reconnecting…", exc)
                await asyncio.sleep(2)

    async def _connect_and_listen(self) -> None:
        headers = self._client.sign_ws_headers()
        ws_url = self._client.get_ws_url()

        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            ping_interval=10,
            ping_timeout=30,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._orderbooks.clear()
            logger.info("Connected to Kalshi WebSocket: {}", ws_url)
            await self._subscribe_all()

            async for raw in ws:
                if not self._running:
                    break
                await self._dispatch(json.loads(raw))

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _send(self, cmd: str, params: dict) -> None:
        assert self._ws is not None
        msg = {"id": self._msg_id, "cmd": cmd, "params": params}
        self._msg_id += 1
        await self._ws.send(json.dumps(msg))

    async def _subscribe_all(self) -> None:
        await self._send("subscribe", {"channels": ["ticker"]})

        if self._market_tickers:
            await self._send(
                "subscribe",
                {
                    "channels": ["orderbook_delta"],
                    "market_tickers": self._market_tickers,
                },
            )

        # Private channels (fill + user_orders) — no market filter = all fills
        await self._send("subscribe", {"channels": ["fill"]})
        await self._send("subscribe", {"channels": ["user_orders"]})

    async def update_markets(self, tickers: list[str]) -> None:
        """Dynamically add market tickers to the orderbook subscription."""
        new = [t for t in tickers if t not in self._market_tickers]
        if not new:
            return
        self._market_tickers.extend(new)
        if self._ws:
            await self._send(
                "subscribe",
                {"channels": ["orderbook_delta"], "market_tickers": new},
            )

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, data: dict) -> None:
        # Flush any queued snapshot publishes
        while self._snapshot_publish_queue:
            ms = self._snapshot_publish_queue.pop(0)
            await self._bus.publish("market:state", ms)

        msg_type = data.get("type", "")

        if msg_type == "ticker":
            await self._handle_ticker(data.get("msg", {}))
        elif msg_type == "orderbook_snapshot":
            self._handle_ob_snapshot(data.get("msg", {}))
        elif msg_type == "orderbook_delta":
            await self._handle_ob_delta(data.get("msg", {}))
        elif msg_type == "fill":
            await self._handle_fill(data.get("msg", {}))
        elif msg_type == "user_order":
            await self._handle_order_update(data.get("msg", {}))
        elif msg_type == "subscribed":
            logger.debug("Subscribed: {}", data)
        elif msg_type == "error":
            logger.error("Kalshi WS error: {}", data)

    # ------------------------------------------------------------------
    # Ticker
    # ------------------------------------------------------------------

    async def _handle_ticker(self, msg: dict) -> None:
        try:
            market_state = MarketState(
                ticker=msg["market_ticker"],
                yes_bid=msg.get("yes_bid", 0),
                yes_ask=msg.get("yes_ask", 0),
                no_bid=msg.get("no_bid", 0),
                no_ask=msg.get("no_ask", 0),
                volume=msg.get("volume", 0),
                timestamp=datetime.utcnow(),
            )
            await self._bus.publish("market:state", market_state)
        except (KeyError, TypeError):
            logger.warning("Malformed ticker msg: {}", msg)

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def _handle_ob_snapshot(self, msg: dict) -> None:
        ticker = msg.get("market_ticker", "")
        # Kalshi V2: yes_dollars_fp / no_dollars_fp with [["0.4200", "500.00"], ...]
        yes_raw = msg.get("yes_dollars_fp", msg.get("yes", msg.get("bids", [])))
        no_raw = msg.get("no_dollars_fp", msg.get("no", msg.get("asks", [])))
        yes_book: dict[int, int] = {}
        no_book: dict[int, int] = {}
        for lvl in yes_raw:
            if len(lvl) >= 2:
                price_cents = int(round(float(lvl[0]) * 100))
                qty = int(round(float(lvl[1])))
                if qty > 0 and price_cents > 0:
                    yes_book[price_cents] = qty
        for lvl in no_raw:
            if len(lvl) >= 2:
                price_cents = int(round(float(lvl[0]) * 100))
                qty = int(round(float(lvl[1])))
                if qty > 0 and price_cents > 0:
                    no_book[price_cents] = qty
        self._orderbooks[ticker] = {"yes": yes_book, "no": no_book}
        if yes_book or no_book:
            yes_bid = max(yes_book) if yes_book else 0
            no_bid = max(no_book) if no_book else 0
            yes_ask = (100 - no_bid) if no_bid > 0 else 0
            logger.info(
                "OB snapshot for {}: yes_bid={}c yes_ask={}c ({} yes/{} no levels)",
                ticker, yes_bid, yes_ask, len(yes_book), len(no_book),
            )
            # Publish initial market state from snapshot
            self._snapshot_publish_queue.append(MarketState(
                ticker=ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=(100 - yes_bid) if yes_bid > 0 else 0,
                volume=0,
                timestamp=datetime.utcnow(),
            ))

    async def _handle_ob_delta(self, msg: dict) -> None:
        ticker = msg.get("market_ticker", "")

        # Auto-initialize orderbook if we haven't seen a snapshot
        if ticker not in self._orderbooks:
            if ticker.startswith("KXNBA"):
                self._orderbooks[ticker] = {"yes": {}, "no": {}}
            else:
                return

        book = self._orderbooks[ticker]

        # Kalshi V2 WS format: single level per message
        # {price_dollars: "0.42", delta_fp: "100.00", side: "yes"}
        price_str = msg.get("price_dollars")
        delta_str = msg.get("delta_fp")
        side = msg.get("side", "")

        if price_str and delta_str and side in ("yes", "no"):
            price_cents = int(round(float(price_str) * 100))
            quantity = int(round(float(delta_str)))
            self._apply_delta(book[side], price_cents, quantity)
        else:
            # Legacy batched format fallback
            for price, quantity in msg.get("bids", []):
                self._apply_delta(book.get("yes", {}), price, quantity)
            for price, quantity in msg.get("asks", []):
                self._apply_delta(book.get("no", {}), price, quantity)

        yes_book = book.get("yes", {})
        no_book = book.get("no", {})
        try:
            yes_bid = max(yes_book) if yes_book else 0
            # yes_ask = 100 - best_no_bid (Kalshi binary market)
            no_bid = max(no_book) if no_book else 0
            yes_ask = (100 - no_bid) if no_bid > 0 else 0

            market_state = MarketState(
                ticker=ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=(100 - yes_bid) if yes_bid > 0 else 0,
                volume=0,
                timestamp=datetime.utcnow(),
            )
            await self._bus.publish("market:state", market_state)
        except (ValueError, KeyError):
            pass

    @staticmethod
    def _apply_delta(book: dict[int, int], price: int, quantity: int) -> None:
        """Apply an absolute-quantity delta. O(1) update, no sort needed."""
        if quantity == 0:
            book.pop(price, None)
        else:
            book[price] = quantity

    # ------------------------------------------------------------------
    # Fill + Order callbacks (for executor)
    # ------------------------------------------------------------------

    async def _handle_fill(self, msg: dict) -> None:
        logger.info("Fill event: order={} ticker={} count={}", msg.get("order_id"), msg.get("market_ticker"), msg.get("count"))
        for cb in self._fill_callbacks:
            task = asyncio.create_task(cb(msg))
            self._bg_tasks.add(task)
            task.add_done_callback(self._task_done)

    async def _handle_order_update(self, msg: dict) -> None:
        logger.debug("Order update: {} status={}", msg.get("order_id"), msg.get("status"))
        for cb in self._order_callbacks:
            task = asyncio.create_task(cb(msg))
            self._bg_tasks.add(task)
            task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._bg_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.exception("Callback error: {}", exc, exc_info=exc)

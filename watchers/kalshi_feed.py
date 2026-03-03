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

        # Local order book: ticker -> {"yes": [[price, qty], ...], "no": [...]}
        self._orderbooks: dict[str, dict[str, list[list[int]]]] = defaultdict(
            lambda: {"yes": [], "no": []}
        )

        # Executor callbacks
        self._fill_callbacks: list[OnFillCallback] = []
        self._order_callbacks: list[OnOrderCallback] = []

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

        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            self._ws = ws
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
        self._orderbooks[ticker] = {
            "yes": msg.get("yes", []),
            "no": msg.get("no", []),
        }
        logger.debug("OB snapshot for {}: {} levels", ticker, len(msg.get("yes", [])))

    async def _handle_ob_delta(self, msg: dict) -> None:
        ticker = msg.get("market_ticker", "")
        price = msg.get("price", 0)
        delta = msg.get("delta", 0)
        side = msg.get("side", "yes")

        book = self._orderbooks[ticker][side]
        self._apply_delta(book, price, delta)

        ob = self._orderbooks[ticker]
        yes_levels = ob["yes"]
        no_levels = ob["no"]
        try:
            market_state = MarketState(
                ticker=ticker,
                yes_bid=yes_levels[-1][0] if yes_levels else 0,
                yes_ask=yes_levels[0][0] if yes_levels else 0,
                no_bid=no_levels[-1][0] if no_levels else 0,
                no_ask=no_levels[0][0] if no_levels else 0,
                volume=0,
                timestamp=datetime.utcnow(),
            )
            await self._bus.publish("market:state", market_state)
        except (IndexError, KeyError):
            pass

    @staticmethod
    def _apply_delta(book: list[list[int]], price: int, delta: int) -> None:
        for level in book:
            if level[0] == price:
                level[1] += delta
                if level[1] <= 0:
                    book.remove(level)
                return
        if delta > 0:
            book.append([price, delta])
            book.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Fill + Order callbacks (for executor)
    # ------------------------------------------------------------------

    async def _handle_fill(self, msg: dict) -> None:
        logger.info("Fill event: order={} ticker={} count={}", msg.get("order_id"), msg.get("market_ticker"), msg.get("count"))
        for cb in self._fill_callbacks:
            try:
                await cb(msg)
            except Exception:
                logger.exception("Fill callback error")

    async def _handle_order_update(self, msg: dict) -> None:
        logger.debug("Order update: {} status={}", msg.get("order_id"), msg.get("status"))
        for cb in self._order_callbacks:
            try:
                await cb(msg)
            except Exception:
                logger.exception("Order callback error")

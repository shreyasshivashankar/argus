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

        # Local order book: ticker -> {"bids": [[price, qty], ...], "asks": [...]}
        # Bids sorted descending (highest first), asks sorted ascending (lowest first)
        self._orderbooks: dict[str, dict[str, list[list[int]]]] = defaultdict(
            lambda: {"bids": [], "asks": []}
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
            "bids": [list(lvl) for lvl in msg.get("yes", [])],
            "asks": [list(lvl) for lvl in msg.get("no", [])],
        }
        self._sort_book(ticker)
        logger.debug(
            "OB snapshot for {}: {} bid levels, {} ask levels",
            ticker, len(self._orderbooks[ticker]["bids"]),
            len(self._orderbooks[ticker]["asks"]),
        )

    async def _handle_ob_delta(self, msg: dict) -> None:
        ticker = msg.get("market_ticker", "")
        price = msg.get("price", 0)
        quantity = msg.get("delta", 0)
        side = msg.get("side", "")

        book_side = "bids" if side == "yes" else "asks"
        book = self._orderbooks[ticker][book_side]
        self._apply_delta(book, price, quantity, ascending=(book_side == "asks"))

        ob = self._orderbooks[ticker]
        bids = ob["bids"]
        asks = ob["asks"]
        try:
            market_state = MarketState(
                ticker=ticker,
                yes_bid=bids[0][0] if bids else 0,
                yes_ask=asks[0][0] if asks else 0,
                no_bid=0,
                no_ask=0,
                volume=0,
                timestamp=datetime.utcnow(),
            )
            await self._bus.publish("market:state", market_state)
        except (IndexError, KeyError):
            pass

    def _sort_book(self, ticker: str) -> None:
        ob = self._orderbooks[ticker]
        ob["bids"].sort(key=lambda x: x[0], reverse=True)
        ob["asks"].sort(key=lambda x: x[0])

    @staticmethod
    def _apply_delta(
        book: list[list[int]], price: int, quantity: int, *, ascending: bool
    ) -> None:
        """Apply an absolute-quantity delta to a price level.

        Kalshi deltas are absolute: quantity=0 means remove the level,
        quantity>0 means set (not add) the volume at that price.
        """
        for i, level in enumerate(book):
            if level[0] == price:
                if quantity == 0:
                    book.pop(i)
                else:
                    level[1] = quantity
                return
        if quantity > 0:
            book.append([price, quantity])
            book.sort(key=lambda x: x[0], reverse=(not ascending))

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

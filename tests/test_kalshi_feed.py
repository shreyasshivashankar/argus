"""Tests for KalshiFeedWatcher order book logic."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from watchers.kalshi_feed import KalshiFeedWatcher


# ---------------------------------------------------------------------------
# Fixture: watcher with mocked client/bus
# ---------------------------------------------------------------------------

@pytest.fixture
def watcher():
    client = AsyncMock()
    client.get_ws_url.return_value = "wss://test"
    client.sign_ws_headers.return_value = {}
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return KalshiFeedWatcher(client, bus, market_tickers=["NBA-YES-LAL"])


# ===========================================================================
# _apply_delta — pure function tests (parameterized)
# ===========================================================================

class TestApplyDelta:
    """Kalshi deltas are absolute: quantity=0 removes, quantity>0 sets. Dict-based O(1)."""

    def test_add_new_level(self):
        book: dict[int, int] = {}
        KalshiFeedWatcher._apply_delta(book, 50, 100)
        assert book == {50: 100}

    def test_set_existing_level_absolute(self):
        """Quantity replaces, not adds."""
        book: dict[int, int] = {50: 100}
        KalshiFeedWatcher._apply_delta(book, 50, 200)
        assert book == {50: 200}

    def test_remove_level_quantity_zero(self):
        book: dict[int, int] = {40: 50, 50: 100, 60: 75}
        KalshiFeedWatcher._apply_delta(book, 50, 0)
        assert book == {40: 50, 60: 75}

    def test_remove_nonexistent_level_noop(self):
        book: dict[int, int] = {50: 100}
        KalshiFeedWatcher._apply_delta(book, 99, 0)
        assert book == {50: 100}

    def test_add_zero_quantity_noop(self):
        """Adding a level with quantity 0 should not insert anything."""
        book: dict[int, int] = {}
        KalshiFeedWatcher._apply_delta(book, 50, 0)
        assert book == {}

    def test_multiple_levels_dict(self):
        """Dict stores all levels; best bid/ask via max/min."""
        book: dict[int, int] = {}
        for price in [50, 10, 20]:
            KalshiFeedWatcher._apply_delta(book, price, 100)
        assert book == {50: 100, 10: 100, 20: 100}
        assert max(book) == 50
        assert min(book) == 10

    def test_multiple_updates_sequence(self):
        """Simulate a realistic delta sequence."""
        book: dict[int, int] = {}
        KalshiFeedWatcher._apply_delta(book, 30, 150)
        KalshiFeedWatcher._apply_delta(book, 40, 200)
        KalshiFeedWatcher._apply_delta(book, 30, 50)  # update, not add
        KalshiFeedWatcher._apply_delta(book, 40, 0)   # remove
        assert book == {30: 50}


# ===========================================================================
# _handle_ob_snapshot
# ===========================================================================

class TestHandleOBSnapshot:

    def test_snapshot_populates_bids_asks(self, watcher):
        msg = {
            "market_ticker": "NBA-YES-LAL",
            "bids": [[50, 100], [45, 200]],
            "asks": [[55, 80], [60, 120]],
        }
        watcher._handle_ob_snapshot(msg)
        ob = watcher._orderbooks["NBA-YES-LAL"]
        assert ob["bids"] == {50: 100, 45: 200}
        assert ob["asks"] == {55: 80, 60: 120}

    def test_snapshot_best_bid_ask_via_max_min(self, watcher):
        msg = {
            "market_ticker": "T1",
            "bids": [[10, 50], [30, 50], [20, 50]],
            "asks": [[60, 50], [40, 50], [50, 50]],
        }
        watcher._handle_ob_snapshot(msg)
        ob = watcher._orderbooks["T1"]
        assert max(ob["bids"]) == 30
        assert min(ob["asks"]) == 40


# ===========================================================================
# _handle_ob_delta → publishes MarketState (Kalshi V2 batch format)
# ===========================================================================

class TestHandleOBDelta:

    @pytest.mark.asyncio
    async def test_delta_publishes_market_state(self, watcher):
        watcher._orderbooks["NBA-YES-LAL"] = {
            "bids": {50: 100},
            "asks": {55: 80},
        }
        msg = {
            "market_ticker": "NBA-YES-LAL",
            "bids": [[48, 200]],
            "asks": [],
        }
        await watcher._handle_ob_delta(msg)
        watcher._bus.publish.assert_called_once()
        call_args = watcher._bus.publish.call_args
        assert call_args[0][0] == "market:state"
        market = call_args[0][1]
        assert market.ticker == "NBA-YES-LAL"
        assert market.yes_bid == 50

    @pytest.mark.asyncio
    async def test_delta_removes_level(self, watcher):
        watcher._orderbooks["T1"] = {
            "bids": {50: 100, 45: 200},
            "asks": {55: 80},
        }
        msg = {
            "market_ticker": "T1",
            "bids": [[50, 0]],
            "asks": [],
        }
        await watcher._handle_ob_delta(msg)
        assert watcher._orderbooks["T1"]["bids"] == {45: 200}

    @pytest.mark.asyncio
    async def test_delta_batch_updates(self, watcher):
        """Multiple bid and ask updates in a single delta message."""
        watcher._orderbooks["T2"] = {
            "bids": {50: 100},
            "asks": {55: 80},
        }
        msg = {
            "market_ticker": "T2",
            "bids": [[48, 150], [50, 0]],
            "asks": [[55, 200], [60, 100]],
        }
        await watcher._handle_ob_delta(msg)
        assert watcher._orderbooks["T2"]["bids"] == {48: 150}
        assert watcher._orderbooks["T2"]["asks"] == {55: 200, 60: 100}

    @pytest.mark.asyncio
    async def test_delta_unknown_ticker_ignored(self, watcher):
        """Delta for a ticker with no snapshot should be silently skipped."""
        msg = {
            "market_ticker": "UNKNOWN",
            "bids": [[50, 100]],
            "asks": [],
        }
        await watcher._handle_ob_delta(msg)
        watcher._bus.publish.assert_not_called()

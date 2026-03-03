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
    """Kalshi deltas are absolute: quantity=0 removes, quantity>0 sets."""

    def test_add_new_level(self):
        book: list[list[int]] = []
        KalshiFeedWatcher._apply_delta(book, 50, 100, ascending=True)
        assert book == [[50, 100]]

    def test_set_existing_level_absolute(self):
        """Quantity replaces, not adds."""
        book: list[list[int]] = [[50, 100]]
        KalshiFeedWatcher._apply_delta(book, 50, 200, ascending=True)
        assert book == [[50, 200]]

    def test_remove_level_quantity_zero(self):
        book: list[list[int]] = [[40, 50], [50, 100], [60, 75]]
        KalshiFeedWatcher._apply_delta(book, 50, 0, ascending=True)
        assert book == [[40, 50], [60, 75]]

    def test_remove_nonexistent_level_noop(self):
        book: list[list[int]] = [[50, 100]]
        KalshiFeedWatcher._apply_delta(book, 99, 0, ascending=True)
        assert book == [[50, 100]]

    def test_add_zero_quantity_noop(self):
        """Adding a level with quantity 0 should not insert anything."""
        book: list[list[int]] = []
        KalshiFeedWatcher._apply_delta(book, 50, 0, ascending=True)
        assert book == []

    @pytest.mark.parametrize(
        "ascending, expected_prices",
        [
            (True, [10, 20, 50]),   # asks: ascending
            (False, [50, 20, 10]),  # bids: descending
        ],
    )
    def test_sort_invariant(self, ascending, expected_prices):
        book: list[list[int]] = []
        for price in [50, 10, 20]:
            KalshiFeedWatcher._apply_delta(book, price, 100, ascending=ascending)
        assert [lvl[0] for lvl in book] == expected_prices

    def test_multiple_updates_sequence(self):
        """Simulate a realistic delta sequence."""
        book: list[list[int]] = []
        KalshiFeedWatcher._apply_delta(book, 30, 150, ascending=True)
        KalshiFeedWatcher._apply_delta(book, 40, 200, ascending=True)
        KalshiFeedWatcher._apply_delta(book, 30, 50, ascending=True)  # update, not add
        KalshiFeedWatcher._apply_delta(book, 40, 0, ascending=True)   # remove
        assert book == [[30, 50]]


# ===========================================================================
# _handle_ob_snapshot
# ===========================================================================

class TestHandleOBSnapshot:

    def test_snapshot_populates_bids_asks(self, watcher):
        msg = {
            "market_ticker": "NBA-YES-LAL",
            "yes": [[50, 100], [45, 200]],
            "no": [[55, 80], [60, 120]],
        }
        watcher._handle_ob_snapshot(msg)
        ob = watcher._orderbooks["NBA-YES-LAL"]
        # Bids from "yes" sorted descending
        assert ob["bids"] == [[50, 100], [45, 200]]
        # Asks from "no" sorted ascending
        assert ob["asks"] == [[55, 80], [60, 120]]

    def test_snapshot_sorts_correctly(self, watcher):
        msg = {
            "market_ticker": "T1",
            "yes": [[10, 50], [30, 50], [20, 50]],  # unsorted
            "no": [[60, 50], [40, 50], [50, 50]],    # unsorted
        }
        watcher._handle_ob_snapshot(msg)
        ob = watcher._orderbooks["T1"]
        assert [lvl[0] for lvl in ob["bids"]] == [30, 20, 10]  # descending
        assert [lvl[0] for lvl in ob["asks"]] == [40, 50, 60]  # ascending


# ===========================================================================
# _handle_ob_delta → publishes MarketState
# ===========================================================================

class TestHandleOBDelta:

    @pytest.mark.asyncio
    async def test_delta_publishes_market_state(self, watcher):
        watcher._orderbooks["NBA-YES-LAL"] = {
            "bids": [[50, 100]],
            "asks": [[55, 80]],
        }
        msg = {
            "market_ticker": "NBA-YES-LAL",
            "price": 48,
            "delta": 200,
            "side": "yes",  # yes → bids
        }
        await watcher._handle_ob_delta(msg)
        watcher._bus.publish.assert_called_once()
        call_args = watcher._bus.publish.call_args
        assert call_args[0][0] == "market:state"
        market = call_args[0][1]
        assert market.ticker == "NBA-YES-LAL"
        # Best bid is the highest: 50 (existing, sorted descending)
        assert market.yes_bid == 50

    @pytest.mark.asyncio
    async def test_delta_removes_level(self, watcher):
        watcher._orderbooks["T1"] = {
            "bids": [[50, 100], [45, 200]],
            "asks": [[55, 80]],
        }
        msg = {
            "market_ticker": "T1",
            "price": 50,
            "delta": 0,  # remove this level
            "side": "yes",
        }
        await watcher._handle_ob_delta(msg)
        assert watcher._orderbooks["T1"]["bids"] == [[45, 200]]

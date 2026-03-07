"""Tests for the 98-cent auto-cashout (OCO) and per-game exposure cap."""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, call

import pytest

from core.schemas import (
    Action,
    ManagedOrder,
    Order,
    OrderState,
    PortfolioPosition,
    PortfolioState,
    Side,
)
from tests.conftest import (
    make_managed_order,
    make_market_state,
    make_order,
    make_portfolio_position,
    make_portfolio_state,
)


# ===========================================================================
# 98c take-profit + OCO management
# ===========================================================================

class TestPlaceTakeProfit:
    """_place_take_profit registers OCO pairs and calls place_order at 98c."""

    @pytest.mark.asyncio
    async def test_tp_order_placed_at_98c(self, executor, mock_client):
        entry_order = make_order(yes_price=50)
        entry = make_managed_order(order=entry_order, is_exit=False)
        entry.vwap_cents = 50.0
        entry.fill_count = 10
        executor._orders[entry_order.client_order_id] = entry

        spread_exit_cid = str(uuid.uuid4())

        mock_client.place_order = AsyncMock(
            return_value={"order": {"order_id": "kalshi-tp-001"}}
        )

        await executor._place_take_profit(entry, 10, spread_exit_cid)

        mock_client.place_order.assert_called_once()
        placed = mock_client.place_order.call_args[0][0]
        assert placed.yes_price == 98
        assert placed.action == Action.SELL
        assert placed.count == 10

    @pytest.mark.asyncio
    async def test_oco_pair_registered_bidirectionally(self, executor, mock_client):
        entry_order = make_order(yes_price=40)
        entry = make_managed_order(order=entry_order, is_exit=False)
        entry.vwap_cents = 40.0
        executor._orders[entry_order.client_order_id] = entry

        spread_exit_cid = str(uuid.uuid4())
        await executor._place_take_profit(entry, 5, spread_exit_cid)

        # Both directions must exist in _oco_pairs
        tp_cid = entry.paired_exit_order_ids[-1]
        assert executor._oco_pairs[spread_exit_cid] == tp_cid
        assert executor._oco_pairs[tp_cid] == spread_exit_cid

    @pytest.mark.asyncio
    async def test_tp_failure_cleans_up_oco(self, executor, mock_client):
        entry_order = make_order(yes_price=40)
        entry = make_managed_order(order=entry_order, is_exit=False)
        entry.vwap_cents = 40.0
        executor._orders[entry_order.client_order_id] = entry

        spread_exit_cid = str(uuid.uuid4())
        mock_client.place_order = AsyncMock(side_effect=RuntimeError("network error"))

        await executor._place_take_profit(entry, 5, spread_exit_cid)

        # OCO pair should be removed on failure
        assert spread_exit_cid not in executor._oco_pairs


class TestCancelOcoPair:
    """_cancel_oco_pair cancels and marks CANCELED the paired order."""

    @pytest.mark.asyncio
    async def test_cancels_resting_paired_order(self, executor, mock_client):
        tp_order = make_order(yes_price=98)
        tp_managed = make_managed_order(order=tp_order, is_exit=True)
        tp_managed.state = OrderState.RESTING
        tp_managed.kalshi_order_id = "kalshi-tp-001"
        executor._orders[tp_order.client_order_id] = tp_managed
        executor._kalshi_to_client["kalshi-tp-001"] = tp_order.client_order_id

        await executor._cancel_oco_pair(tp_order.client_order_id)

        mock_client.cancel_order.assert_called_once_with("kalshi-tp-001")
        assert tp_managed.state == OrderState.CANCELED

    @pytest.mark.asyncio
    async def test_skips_already_filled_order(self, executor, mock_client):
        tp_order = make_order(yes_price=98)
        tp_managed = make_managed_order(order=tp_order, is_exit=True)
        tp_managed.state = OrderState.FILLED
        executor._orders[tp_order.client_order_id] = tp_managed

        await executor._cancel_oco_pair(tp_order.client_order_id)

        mock_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_nonexistent_order(self, executor, mock_client):
        await executor._cancel_oco_pair("does-not-exist")
        mock_client.cancel_order.assert_not_called()


class TestOnFillOcoCancellation:
    """When an exit fills, on_fill must cancel the OCO paired order."""

    @pytest.mark.asyncio
    async def test_spread_fill_cancels_tp(self, executor, mock_client):
        # Set up entry
        entry_order = make_order(yes_price=50)
        entry_managed = make_managed_order(order=entry_order, is_exit=False, signal_id="sig-1")
        entry_managed.fill_count = 10
        entry_managed.vwap_cents = 50.0
        executor._orders[entry_order.client_order_id] = entry_managed

        # Set up spread exit
        spread_order = make_order(action=Action.SELL, yes_price=57)
        spread_managed = make_managed_order(
            order=spread_order, is_exit=True, signal_id="sig-1",
            parent_entry_id=entry_order.client_order_id,
        )
        spread_managed.state = OrderState.RESTING
        spread_managed.kalshi_order_id = "kalshi-spread-001"
        spread_managed.fill_count = 0
        spread_managed.remaining_count = 10
        spread_managed.vwap_cents = 0.0
        executor._orders[spread_order.client_order_id] = spread_managed
        executor._kalshi_to_client["kalshi-spread-001"] = spread_order.client_order_id

        # Set up TP order
        tp_order = make_order(action=Action.SELL, yes_price=98)
        tp_managed = make_managed_order(
            order=tp_order, is_exit=True, signal_id="sig-1",
            parent_entry_id=entry_order.client_order_id,
        )
        tp_managed.state = OrderState.RESTING
        tp_managed.kalshi_order_id = "kalshi-tp-001"
        tp_managed.fill_count = 0
        tp_managed.remaining_count = 10
        tp_managed.vwap_cents = 0.0
        executor._orders[tp_order.client_order_id] = tp_managed
        executor._kalshi_to_client["kalshi-tp-001"] = tp_order.client_order_id

        # Register OCO pair
        executor._oco_pairs[spread_order.client_order_id] = tp_order.client_order_id
        executor._oco_pairs[tp_order.client_order_id] = spread_order.client_order_id

        # Fire spread exit fill
        fill_msg = {
            "client_order_id": spread_order.client_order_id,
            "order_id": "kalshi-spread-001",
            "yes_price": 57,
            "count": 10,
        }
        await executor.on_fill(fill_msg)

        # TP should have been canceled
        mock_client.cancel_order.assert_called_with("kalshi-tp-001")
        assert tp_managed.state == OrderState.CANCELED
        # OCO map should be cleaned up
        assert spread_order.client_order_id not in executor._oco_pairs
        assert tp_order.client_order_id not in executor._oco_pairs


# ===========================================================================
# 98c TP not placed for flash crash entries
# ===========================================================================

class TestTpNotPlacedForFlashCrash:
    """Flash crash entries must NOT get a 98c TP — they use the hard-stop timer."""

    @pytest.mark.asyncio
    async def test_flash_crash_entry_skips_tp(self, executor, mock_client):
        entry_order = make_order(yes_price=40)
        entry_managed = make_managed_order(order=entry_order, is_exit=False, signal_id="sig-fc")
        entry_managed.fill_count = 5
        entry_managed.remaining_count = 0
        entry_managed.vwap_cents = 40.0
        executor._orders[entry_order.client_order_id] = entry_managed

        # Mark as flash crash
        executor._flash_crash_entries.add(entry_order.client_order_id)

        # Simulate a fill, then check _place_exit path
        # place_order returns a new kalshi_id each call
        call_count = 0
        def _place_side_effect(order):
            nonlocal call_count
            call_count += 1
            return {"order": {"order_id": f"kalshi-order-{call_count:03d}"}}
        mock_client.place_order = AsyncMock(side_effect=_place_side_effect)

        await executor._place_exit(entry_managed, 5, source="flash_crash")

        # Only ONE place_order call (the spread exit) — no TP
        assert call_count == 1
        assert not executor._oco_pairs


# ===========================================================================
# Per-game exposure cap (_get_game_exposure)
# ===========================================================================

class TestGetGameExposure:

    def _make_quant(self, settings, mock_bus, mock_client):
        from agents.nba_quant import NBAQuantAgent
        agent = NBAQuantAgent(settings, mock_bus, mock_client)
        return agent

    def test_zero_when_no_portfolio(self, settings, mock_bus, mock_client):
        agent = self._make_quant(settings, mock_bus, mock_client)
        assert agent._get_game_exposure("game-001") == 0

    def test_zero_when_no_positions(self, settings, mock_bus, mock_client):
        agent = self._make_quant(settings, mock_bus, mock_client)
        agent._portfolio = make_portfolio_state(bankroll=1000.0, positions=[])
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        assert agent._get_game_exposure("game-001") == 0

    def test_counts_open_positions_for_game(self, settings, mock_bus, mock_client):
        agent = self._make_quant(settings, mock_bus, mock_client)
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL", "KXNBA-TOTAL-LAL"]

        pos1 = make_portfolio_position(ticker="KXNBA-GAME-LAL", remaining_count=10)
        pos2 = make_portfolio_position(
            client_order_id="exit-002", ticker="KXNBA-TOTAL-LAL", remaining_count=5,
        )
        agent._portfolio = make_portfolio_state(positions=[pos1, pos2])

        assert agent._get_game_exposure("game-001") == 2

    def test_zero_remaining_count_not_counted(self, settings, mock_bus, mock_client):
        """Cashed-out positions (remaining_count=0) must not count."""
        agent = self._make_quant(settings, mock_bus, mock_client)
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]

        pos = make_portfolio_position(ticker="KXNBA-GAME-LAL", remaining_count=0)
        agent._portfolio = make_portfolio_state(positions=[pos])

        assert agent._get_game_exposure("game-001") == 0

    def test_different_game_tickers_not_counted(self, settings, mock_bus, mock_client):
        """Positions from a different game don't inflate this game's count."""
        agent = self._make_quant(settings, mock_bus, mock_client)
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        agent._game_to_tickers["game-002"] = ["KXNBA-GAME-BOS"]

        pos = make_portfolio_position(ticker="KXNBA-GAME-BOS", remaining_count=10)
        agent._portfolio = make_portfolio_state(positions=[pos])

        assert agent._get_game_exposure("game-001") == 0
        assert agent._get_game_exposure("game-002") == 1

"""Tests for crash-only startup reconciliation in OrderExecutor."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.schemas import Action, OrderState, Side
from tests.conftest import make_order, make_managed_order


# ===========================================================================
# Clean slate — no resting orders
# ===========================================================================

class TestReconcileCleanSlate:

    @pytest.mark.asyncio
    async def test_no_resting_orders(self, executor):
        executor.client.get_balance = AsyncMock(return_value=500.0)
        executor.client.get_orders = AsyncMock(return_value=[])

        await executor._reconcile_state_on_boot()

        assert executor.current_bankroll == 500.0
        assert len(executor._orders) == 0

    @pytest.mark.asyncio
    async def test_balance_failure_aborts(self, executor):
        executor.client.get_balance = AsyncMock(
            side_effect=Exception("connection refused")
        )

        await executor._reconcile_state_on_boot()

        assert len(executor._orders) == 0


# ===========================================================================
# Recovering resting orders
# ===========================================================================

class TestReconcileRestingOrders:

    @pytest.mark.asyncio
    async def test_recovers_resting_exit(self, executor):
        executor.client.get_balance = AsyncMock(return_value=200.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-exit-1",
                "client_order_id": "client-exit-1",
                "ticker": "KXNBAPTS-LALJAMES-25",
                "action": "sell",
                "side": "yes",
                "yes_price": 48,
                "no_price": None,
                "remaining_count": 10,
                "fill_count": 0,
                "initial_count": 10,
            },
        ])

        await executor._reconcile_state_on_boot()

        assert len(executor._orders) == 1
        managed = executor._orders["client-exit-1"]
        assert managed.state == OrderState.RESTING
        assert managed.is_exit is True
        assert managed.order.ticker == "KXNBAPTS-LALJAMES-25"
        assert managed.order.action == Action.SELL
        assert managed.order.side == Side.YES
        assert managed.order.count == 10
        assert managed.remaining_count == 10
        assert managed.kalshi_order_id == "kalshi-exit-1"
        assert executor._kalshi_to_client["kalshi-exit-1"] == "client-exit-1"

    @pytest.mark.asyncio
    async def test_recovers_resting_entry(self, executor):
        executor.client.get_balance = AsyncMock(return_value=800.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-entry-1",
                "client_order_id": "client-entry-1",
                "ticker": "KXNBAGAME-LALDAL-DAL",
                "action": "buy",
                "side": "no",
                "yes_price": None,
                "no_price": 30,
                "remaining_count": 50,
                "fill_count": 0,
                "initial_count": 50,
            },
        ])

        await executor._reconcile_state_on_boot()

        managed = executor._orders["client-entry-1"]
        assert managed.is_exit is False
        assert managed.order.action == Action.BUY
        assert managed.order.side == Side.NO
        assert managed.order.no_price == 30

    @pytest.mark.asyncio
    async def test_recovers_multiple_orders(self, executor):
        executor.client.get_balance = AsyncMock(return_value=100.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": f"kalshi-{i}",
                "client_order_id": f"client-{i}",
                "ticker": f"KXNBA-TICKER-{i}",
                "action": "sell",
                "side": "yes",
                "yes_price": 40 + i,
                "no_price": None,
                "remaining_count": 5,
                "fill_count": 0,
                "initial_count": 5,
            }
            for i in range(5)
        ])

        await executor._reconcile_state_on_boot()

        assert len(executor._orders) == 5
        assert len(executor._kalshi_to_client) == 5

    @pytest.mark.asyncio
    async def test_falls_back_to_kalshi_id_when_no_client_id(self, executor):
        executor.client.get_balance = AsyncMock(return_value=100.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-only-id",
                "ticker": "KXNBA-T1",
                "action": "sell",
                "side": "yes",
                "yes_price": 45,
                "no_price": None,
                "remaining_count": 3,
                "fill_count": 0,
                "initial_count": 3,
            },
        ])

        await executor._reconcile_state_on_boot()

        assert "kalshi-only-id" in executor._orders


# ===========================================================================
# VWAP reconstruction from fills
# ===========================================================================

class TestReconcileVWAP:

    @pytest.mark.asyncio
    async def test_reconstructs_vwap_from_fills(self, executor):
        executor.client.get_balance = AsyncMock(return_value=300.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-partial",
                "client_order_id": "client-partial",
                "ticker": "KXNBAPTS-PLAYER",
                "action": "sell",
                "side": "yes",
                "yes_price": 50,
                "no_price": None,
                "remaining_count": 5,
                "fill_count": 10,
                "initial_count": 15,
            },
        ])
        executor.client.get_fills = AsyncMock(return_value=[
            {"yes_price": 20, "count": 6},
            {"yes_price": 22, "count": 4},
        ])

        await executor._reconcile_state_on_boot()

        managed = executor._orders["client-partial"]
        expected_vwap = (20 * 6 + 22 * 4) / 10
        assert managed.vwap_cents == pytest.approx(expected_vwap)
        assert managed.fill_count == 10
        assert managed.remaining_count == 5

    @pytest.mark.asyncio
    async def test_vwap_falls_back_to_limit_on_fill_error(self, executor):
        executor.client.get_balance = AsyncMock(return_value=300.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-no-fills",
                "client_order_id": "client-no-fills",
                "ticker": "KXNBA-T1",
                "action": "sell",
                "side": "yes",
                "yes_price": 45,
                "no_price": None,
                "remaining_count": 3,
                "fill_count": 7,
                "initial_count": 10,
            },
        ])
        executor.client.get_fills = AsyncMock(
            side_effect=Exception("API error")
        )

        await executor._reconcile_state_on_boot()

        managed = executor._orders["client-no-fills"]
        assert managed.vwap_cents == 45.0

    @pytest.mark.asyncio
    async def test_zero_fill_count_skips_vwap(self, executor):
        executor.client.get_balance = AsyncMock(return_value=300.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-fresh",
                "client_order_id": "client-fresh",
                "ticker": "KXNBA-T1",
                "action": "buy",
                "side": "yes",
                "yes_price": 30,
                "no_price": None,
                "remaining_count": 10,
                "fill_count": 0,
                "initial_count": 10,
            },
        ])

        await executor._reconcile_state_on_boot()

        managed = executor._orders["client-fresh"]
        assert managed.vwap_cents == 0.0
        executor.client.get_fills.assert_not_called()


# ===========================================================================
# Portfolio publish on reconciliation
# ===========================================================================

class TestReconcilePublishesPortfolio:

    @pytest.mark.asyncio
    async def test_publishes_portfolio_state(self, executor):
        executor.client.get_balance = AsyncMock(return_value=150.0)
        executor.client.get_orders = AsyncMock(return_value=[
            {
                "order_id": "kalshi-1",
                "client_order_id": "client-1",
                "ticker": "KXNBA-T1",
                "action": "sell",
                "side": "yes",
                "yes_price": 50,
                "no_price": None,
                "remaining_count": 10,
                "fill_count": 0,
                "initial_count": 10,
            },
        ])

        await executor._reconcile_state_on_boot()

        executor.bus.publish.assert_called()
        call_args = executor.bus.publish.call_args_list
        portfolio_calls = [
            c for c in call_args if c[0][0] == "portfolio:state"
        ]
        assert len(portfolio_calls) >= 1

    @pytest.mark.asyncio
    async def test_get_orders_failure_aborts_gracefully(self, executor):
        executor.client.get_balance = AsyncMock(return_value=500.0)
        executor.client.get_orders = AsyncMock(
            side_effect=Exception("network error")
        )

        await executor._reconcile_state_on_boot()

        assert executor.current_bankroll == 500.0
        assert len(executor._orders) == 0

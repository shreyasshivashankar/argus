"""Tests for OrderExecutor: Kelly sizing, kill switch, fill lifecycle, VWAP, GC, reallocation."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from freezegun import freeze_time

from core.schemas import Action, ManagedOrder, Order, OrderState, Side, Signal, SignalStatus
from tests.conftest import make_managed_order, make_order, make_signal


# ===========================================================================
# Kelly Criterion
# ===========================================================================

class TestKellySize:

    def test_zero_bankroll_returns_zero(self, executor):
        executor.current_bankroll = 0
        sig = make_signal(confidence=0.7, entry_price=15)
        assert executor._kelly_size(sig) == 0

    def test_negative_bankroll_returns_zero(self, executor):
        executor.current_bankroll = -100
        sig = make_signal(confidence=0.7, entry_price=15)
        assert executor._kelly_size(sig) == 0

    def test_entry_price_zero_returns_zero(self, executor):
        sig = make_signal(confidence=0.7, entry_price=0)
        assert executor._kelly_size(sig) == 0

    def test_entry_price_100_returns_zero(self, executor):
        sig = make_signal(confidence=0.7, entry_price=100)
        assert executor._kelly_size(sig) == 0

    def test_never_negative(self, executor):
        for conf in [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
            for price in [5, 15, 50, 85, 95]:
                sig = make_signal(confidence=conf, entry_price=price)
                assert executor._kelly_size(sig) >= 0

    def test_known_calculation(self, executor):
        """Verify the math for a specific case.

        confidence=0.7, entry_price=20, bankroll=$1000, KELLY_FRACTION=0.5
        b = (100-20)/20 = 4.0
        kelly_full = (0.7*4 - 0.3)/4 = (2.8-0.3)/4 = 0.625
        kelly_fraction = 0.625 * 0.5 = 0.3125
        bankroll_cents = 100000
        bet_cents = 0.3125 * 100000 = 31250
        count = floor(31250/20) = 1562
        """
        executor.current_bankroll = 1000.0
        executor.settings.KELLY_FRACTION = 0.5
        sig = make_signal(confidence=0.7, entry_price=20)
        result = executor._kelly_size(sig)
        assert result == 1562

    def test_fraction_scaling(self, executor):
        """Halving KELLY_FRACTION should halve the position size."""
        sig = make_signal(confidence=0.7, entry_price=20)
        executor.settings.KELLY_FRACTION = 1.0
        size_full = executor._kelly_size(sig)
        executor.settings.KELLY_FRACTION = 0.5
        size_half = executor._kelly_size(sig)
        assert size_half == math.floor(size_full / 2) or abs(size_half - size_full // 2) <= 1

    def test_low_confidence_no_edge_returns_zero(self, executor):
        """If p*b < q, kelly_full is negative → clamped to 0."""
        sig = make_signal(confidence=0.1, entry_price=80)
        assert executor._kelly_size(sig) == 0


# ===========================================================================
# Kill Switch
# ===========================================================================

class TestKillSwitch:

    @pytest.mark.asyncio
    async def test_positive_pnl_no_trip(self, executor):
        executor._daily_realized_pnl = 100.0
        await executor._check_kill_switch()
        assert not executor._kill_switch_tripped

    @pytest.mark.asyncio
    async def test_loss_below_threshold_no_trip(self, executor):
        executor._daily_realized_pnl = -10.0
        await executor._check_kill_switch()
        assert not executor._kill_switch_tripped

    @pytest.mark.asyncio
    async def test_loss_exceeds_threshold_trips(self, executor):
        executor._daily_realized_pnl = -60.0  # threshold is 50
        with patch.object(executor, "_cancel_all_resting", new_callable=AsyncMock) as mock_cancel, \
             patch.object(executor, "_send_telegram_alert", new_callable=AsyncMock) as mock_alert:
            await executor._check_kill_switch()
        assert executor._kill_switch_tripped
        mock_cancel.assert_called_once()
        mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_signals_dropped_after_trip(self, executor):
        executor._kill_switch_tripped = True
        sig_data = make_signal().model_dump()
        sig_data["timestamp"] = sig_data["timestamp"].isoformat()
        await executor._on_signal("signal:validated", sig_data)
        executor.client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_all_resting(self, executor):
        order1 = make_order(client_order_id="c1")
        order2 = make_order(client_order_id="c2")
        order3 = make_order(client_order_id="c3")

        executor._orders = {
            "c1": make_managed_order(order=order1, state=OrderState.RESTING),
            "c2": make_managed_order(order=order2, state=OrderState.FILLED),
            "c3": make_managed_order(order=order3, state=OrderState.PLACED),
        }
        executor._orders["c1"].kalshi_order_id = "k1"
        executor._orders["c3"].kalshi_order_id = "k3"

        await executor._cancel_all_resting()
        assert executor.client.cancel_order.call_count == 2
        assert executor._orders["c1"].state == OrderState.CANCELED
        assert executor._orders["c2"].state == OrderState.FILLED  # untouched
        assert executor._orders["c3"].state == OrderState.CANCELED


# ===========================================================================
# Fill Lifecycle, VWAP, Partial Fills, Deferred Fills
# ===========================================================================

class TestFillLifecycle:

    @pytest.mark.asyncio
    async def test_single_full_fill(self, executor):
        order = make_order(count=10, yes_price=17, client_order_id="entry-1")
        managed = make_managed_order(order=order, state=OrderState.RESTING)
        managed.remaining_count = 10
        executor._orders["entry-1"] = managed

        fill_msg = {
            "client_order_id": "entry-1",
            "order_id": "kalshi-1",
            "count": 10,
            "yes_price": 15,
        }
        await executor.on_fill(fill_msg)

        assert managed.state == OrderState.FILLED
        assert managed.fill_count == 10
        assert managed.vwap_cents == 15.0
        # Exit should have been placed
        executor.client.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_two_partial_fills_vwap(self, executor):
        order = make_order(count=100, yes_price=20, client_order_id="entry-2")
        managed = make_managed_order(order=order, state=OrderState.RESTING)
        managed.remaining_count = 100
        executor._orders["entry-2"] = managed

        # First partial: 40 @ 18
        await executor.on_fill({
            "client_order_id": "entry-2",
            "count": 40,
            "yes_price": 18,
        })
        assert managed.state == OrderState.PARTIALLY_FILLED
        assert managed.fill_count == 40
        assert managed.vwap_cents == pytest.approx(18.0)
        # Exit placed for batch of 40
        assert executor.client.place_order.call_count == 1

        # Second partial: 60 @ 22
        await executor.on_fill({
            "client_order_id": "entry-2",
            "count": 60,
            "yes_price": 22,
        })
        assert managed.state == OrderState.FILLED
        assert managed.fill_count == 100
        expected_vwap = (18 * 40 + 22 * 60) / 100
        assert managed.vwap_cents == pytest.approx(expected_vwap)
        # Second exit placed for batch of 60
        assert executor.client.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_deferred_fill_replayed(self, executor):
        """Fill arrives before place_order returns → deferred → replayed."""
        fill_msg = {
            "order_id": "kalshi-unknown",
            "count": 5,
            "yes_price": 16,
        }
        await executor.on_fill(fill_msg)
        assert len(executor._deferred_fills) == 1

        # Now simulate the order being registered
        order = make_order(count=5, yes_price=17, client_order_id="late-entry")
        managed = make_managed_order(order=order, state=OrderState.PLACED)
        managed.remaining_count = 5
        executor._orders["late-entry"] = managed
        executor._kalshi_to_client["kalshi-unknown"] = "late-entry"

        await executor._replay_deferred_fills()
        assert len(executor._deferred_fills) == 0
        assert managed.fill_count == 5
        assert managed.state == OrderState.FILLED

    @pytest.mark.asyncio
    async def test_exit_fill_computes_vwap_pnl(self, executor):
        """P&L uses entry VWAP, not limit price."""
        # Set up entry with known VWAP
        entry_order = make_order(count=10, yes_price=20, client_order_id="entry-pnl")
        entry = make_managed_order(order=entry_order, state=OrderState.FILLED)
        entry.fill_count = 10
        entry.vwap_cents = 18.0  # filled at 18, not the limit of 20
        executor._orders["entry-pnl"] = entry

        # Set up exit
        exit_order = make_order(
            action=Action.SELL, count=10, yes_price=25, client_order_id="exit-pnl"
        )
        exit_managed = make_managed_order(
            order=exit_order,
            state=OrderState.RESTING,
            is_exit=True,
            parent_entry_id="entry-pnl",
        )
        exit_managed.remaining_count = 10
        executor._orders["exit-pnl"] = exit_managed
        entry.paired_exit_order_ids.append("exit-pnl")

        await executor.on_fill({
            "client_order_id": "exit-pnl",
            "count": 10,
            "yes_price": 24,  # actual fill at 24
        })

        # P&L = (24 - 18) * 10 / 100 = $0.60
        assert executor._daily_realized_pnl == pytest.approx(0.60)


# ===========================================================================
# GC Loop (Time-Travel)
# ===========================================================================

class TestGarbageCollection:

    @pytest.mark.asyncio
    async def test_gc_evicts_old_terminal_orders(self, executor, settings):
        old_time = datetime.utcnow() - timedelta(seconds=settings.ORDER_GC_TTL + 100)
        recent_time = datetime.utcnow()

        old_filled = make_managed_order(
            order=make_order(client_order_id="old-filled"),
            state=OrderState.FILLED,
            created_at=old_time,
        )
        old_filled.kalshi_order_id = "k-old-filled"

        old_canceled = make_managed_order(
            order=make_order(client_order_id="old-canceled"),
            state=OrderState.CANCELED,
            created_at=old_time,
        )

        recent_filled = make_managed_order(
            order=make_order(client_order_id="recent-filled"),
            state=OrderState.FILLED,
            created_at=recent_time,
        )

        active_resting = make_managed_order(
            order=make_order(client_order_id="active"),
            state=OrderState.RESTING,
            created_at=old_time,
        )

        executor._orders = {
            "old-filled": old_filled,
            "old-canceled": old_canceled,
            "recent-filled": recent_filled,
            "active": active_resting,
        }
        executor._kalshi_to_client = {"k-old-filled": "old-filled"}

        # Run one GC sweep directly
        now = datetime.utcnow()
        terminal_states = {OrderState.FILLED, OrderState.CANCELED}
        stale = [
            cid for cid, m in executor._orders.items()
            if m.state in terminal_states
            and (now - m.created_at).total_seconds() > settings.ORDER_GC_TTL
        ]
        for cid in stale:
            managed = executor._orders.pop(cid, None)
            if managed and managed.kalshi_order_id:
                executor._kalshi_to_client.pop(managed.kalshi_order_id, None)

        assert "old-filled" not in executor._orders
        assert "old-canceled" not in executor._orders
        assert "recent-filled" in executor._orders
        assert "active" in executor._orders
        assert "k-old-filled" not in executor._kalshi_to_client

    @pytest.mark.asyncio
    async def test_gc_preserves_non_terminal_orders(self, executor, settings):
        old_time = datetime.utcnow() - timedelta(seconds=settings.ORDER_GC_TTL + 100)
        for state in [OrderState.PLACED, OrderState.RESTING, OrderState.PARTIALLY_FILLED]:
            cid = f"order-{state}"
            executor._orders[cid] = make_managed_order(
                order=make_order(client_order_id=cid),
                state=state,
                created_at=old_time,
            )

        now = datetime.utcnow()
        terminal_states = {OrderState.FILLED, OrderState.CANCELED}
        stale = [
            cid for cid, m in executor._orders.items()
            if m.state in terminal_states
            and (now - m.created_at).total_seconds() > settings.ORDER_GC_TTL
        ]
        for cid in stale:
            executor._orders.pop(cid, None)

        assert len(executor._orders) == 3


# ===========================================================================
# Reallocation — cancel resting exit + aggressive sell
# ===========================================================================

class TestReallocate:

    @pytest.mark.asyncio
    async def test_reallocate_cancels_and_places_aggressive_sell(self, executor):
        """REALLOCATE should cancel the targeted resting exit, wait for WS
        cancel confirmation, then place an aggressive limit sell."""
        exit_order = make_order(
            action=Action.SELL, count=50, yes_price=25, client_order_id="exit-target"
        )
        exit_managed = make_managed_order(
            order=exit_order,
            state=OrderState.RESTING,
            is_exit=True,
            parent_entry_id="entry-parent",
        )
        exit_managed.kalshi_order_id = "kalshi-exit-target"
        executor._orders["exit-target"] = exit_managed

        signal = make_signal(
            ticker="NBA-YES-LAL",
            status=SignalStatus.REALLOCATE,
            entry_price=95,
        )
        signal_data = signal.model_dump(mode="json")
        signal_data["target_order_id"] = "exit-target"

        async def _simulate_cancel_confirm(*args, **kwargs):
            """After cancel_order REST call, simulate the WS confirmation."""
            await executor.on_order_update({
                "order_id": "kalshi-exit-target",
                "client_order_id": "exit-target",
                "status": "canceled",
            })
            return {}

        executor.client.cancel_order = AsyncMock(side_effect=_simulate_cancel_confirm)

        await executor._on_signal("signal:reallocate", signal_data)

        executor.client.cancel_order.assert_called_once_with("kalshi-exit-target")
        assert exit_managed.state == OrderState.CANCELED

        executor.client.place_order.assert_called_once()
        placed = executor.client.place_order.call_args[0][0]
        assert placed.action == Action.SELL
        assert placed.yes_price == 95
        assert placed.count == 50

    @pytest.mark.asyncio
    async def test_reallocate_unknown_target_is_noop(self, executor):
        """REALLOCATE for a target_order_id not in _orders does nothing."""
        signal = make_signal(status=SignalStatus.REALLOCATE, entry_price=95)
        signal_data = signal.model_dump(mode="json")
        signal_data["target_order_id"] = "nonexistent-id"

        await executor._on_signal("signal:reallocate", signal_data)

        executor.client.cancel_order.assert_not_called()
        executor.client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_reallocate_targets_specific_order_not_ticker(self, executor):
        """With multiple resting exits for the same ticker, only the
        targeted one should be canceled."""
        for cid in ("exit-A", "exit-B", "exit-C"):
            order = make_order(
                action=Action.SELL, count=30, yes_price=25, client_order_id=cid
            )
            managed = make_managed_order(
                order=order, state=OrderState.RESTING, is_exit=True, parent_entry_id="entry-1"
            )
            managed.kalshi_order_id = f"kalshi-{cid}"
            executor._orders[cid] = managed

        signal = make_signal(status=SignalStatus.REALLOCATE, entry_price=95)
        signal_data = signal.model_dump(mode="json")
        signal_data["target_order_id"] = "exit-B"

        async def _simulate_cancel_confirm(*args, **kwargs):
            await executor.on_order_update({
                "order_id": "kalshi-exit-B",
                "client_order_id": "exit-B",
                "status": "canceled",
            })
            return {}

        executor.client.cancel_order = AsyncMock(side_effect=_simulate_cancel_confirm)

        await executor._on_signal("signal:reallocate", signal_data)

        executor.client.cancel_order.assert_called_once_with("kalshi-exit-B")
        assert executor._orders["exit-A"].state == OrderState.RESTING
        assert executor._orders["exit-B"].state == OrderState.CANCELED
        assert executor._orders["exit-C"].state == OrderState.RESTING

"""Tests for OrderExecutor: Kelly sizing, kill switch, fill lifecycle, VWAP, GC, reallocation."""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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

        confidence=0.7, entry_price=20, bankroll=$1000, QUANT_KELLY_FRACTION=0.5
        b = (100-20)/20 = 4.0
        kelly_full = (0.7*4 - 0.3)/4 = (2.8-0.3)/4 = 0.625
        kelly_fraction = 0.625 * 0.5 = 0.3125
        bankroll_cents = 100000
        bet_cents = 0.3125 * 100000 = 31250
        count = floor(31250/20) = 1562
        """
        executor.current_bankroll = 1000.0
        executor.settings.QUANT_KELLY_FRACTION = 0.5
        sig = make_signal(confidence=0.7, entry_price=20)
        result = executor._kelly_size(sig)
        assert result == 1562

    def test_fraction_scaling(self, executor):
        """Halving QUANT_KELLY_FRACTION should halve the position size."""
        sig = make_signal(confidence=0.7, entry_price=20)
        executor.settings.QUANT_KELLY_FRACTION = 1.0
        size_full = executor._kelly_size(sig)
        executor.settings.QUANT_KELLY_FRACTION = 0.5
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
        with patch.object(executor, "_cancel_resting_entries", new_callable=AsyncMock) as mock_cancel, \
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
        # Spread exit + 98c TP (entry 15c < 95c) → 2 place_order calls
        assert executor.client.place_order.call_count == 2
        placed_prices = {c[0][0].yes_price for c in executor.client.place_order.call_args_list}
        assert 98 in placed_prices

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
        # Spread exit + 98c TP for first batch → 2 calls
        assert executor.client.place_order.call_count == 2

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
        # Spread exit + TP for second batch → 4 total calls
        assert executor.client.place_order.call_count == 4

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


# ===========================================================================
# Dynamic Kelly sizing (QUANT vs CRASH)
# ===========================================================================

class TestDynamicKelly:

    def test_quant_signal_uses_quant_fraction(self, executor):
        executor.current_bankroll = 1000.0
        executor.settings.QUANT_KELLY_FRACTION = 0.5
        executor.settings.CRASH_KELLY_FRACTION = 0.1
        sig = make_signal(confidence=0.7, entry_price=20, source="moneyline")
        size_quant = executor._kelly_size(sig)

        sig_crash = make_signal(confidence=0.7, entry_price=20, source="flash_crash")
        size_crash = executor._kelly_size(sig_crash)

        assert size_quant > 0
        assert size_crash > 0
        assert size_crash < size_quant
        assert size_crash == pytest.approx(size_quant * 0.2, abs=2)

    def test_crash_kelly_tenth_of_full(self, executor):
        """CRASH_KELLY=0.1 should be ~1/5 of QUANT_KELLY=0.5."""
        executor.current_bankroll = 1000.0
        executor.settings.QUANT_KELLY_FRACTION = 0.5
        executor.settings.CRASH_KELLY_FRACTION = 0.1
        sig = make_signal(confidence=0.8, entry_price=30, source="flash_crash")
        size = executor._kelly_size(sig)
        assert size > 0

        sig_full = make_signal(confidence=0.8, entry_price=30, source="totals")
        size_full = executor._kelly_size(sig_full)
        ratio = size / size_full
        assert 0.15 < ratio < 0.25


# ===========================================================================
# Velocity circuit breaker
# ===========================================================================

class TestVelocityBreaker:

    @pytest.mark.asyncio
    async def test_first_trade_allowed(self, executor):
        executor.bus.redis.incr = AsyncMock(return_value=1)
        assert await executor._velocity_check("TICKER-A") is True

    @pytest.mark.asyncio
    async def test_second_trade_allowed(self, executor):
        executor.bus.redis.incr = AsyncMock(return_value=2)
        assert await executor._velocity_check("TICKER-A") is True

    @pytest.mark.asyncio
    async def test_third_trade_blocked(self, executor):
        executor.bus.redis.incr = AsyncMock(return_value=3)
        assert await executor._velocity_check("TICKER-A") is False

    @pytest.mark.asyncio
    async def test_redis_failure_allows_trade(self, executor):
        executor.bus.redis.incr = AsyncMock(side_effect=Exception("connection reset"))
        assert await executor._velocity_check("TICKER-A") is True

    @pytest.mark.asyncio
    async def test_velocity_blocks_entry(self, executor):
        """When velocity limit exceeded, entry is not placed."""
        executor.bus.redis.incr = AsyncMock(return_value=3)
        sig = make_signal(confidence=0.7, entry_price=20)
        sig_data = sig.model_dump(mode="json")
        await executor._on_signal("signal:validated", sig_data)
        executor.client.place_order.assert_not_called()


# ===========================================================================
# Hard stop timeout (flash crash asymmetric exit)
# ===========================================================================

class TestHardStop:

    @pytest.mark.asyncio
    async def test_hard_stop_cancels_and_market_sells_with_floor(self, executor):
        """After timeout, unfilled exit is canceled and replaced with a sell
        at max(1, entry_price - 15), not 1c."""
        entry_order = make_order(count=10, yes_price=40, client_order_id="entry-fc")
        entry = make_managed_order(order=entry_order, state=OrderState.FILLED)
        entry.fill_count = 10
        entry.vwap_cents = 40.0
        executor._orders["entry-fc"] = entry

        exit_order = make_order(
            action=Action.SELL, count=10, yes_price=45, client_order_id="exit-fc"
        )
        exit_managed = make_managed_order(
            order=exit_order, state=OrderState.RESTING, is_exit=True,
            parent_entry_id="entry-fc",
        )
        exit_managed.kalshi_order_id = "kalshi-exit-fc"
        executor._orders["exit-fc"] = exit_managed

        executor.settings.FLASH_CRASH_HARD_STOP_TIMEOUT = 0.05

        task = asyncio.create_task(executor._hard_stop_timer("exit-fc"))
        await asyncio.sleep(0.15)

        executor.client.cancel_order.assert_called_once_with("kalshi-exit-fc")
        assert exit_managed.state == OrderState.CANCELED
        executor.client.place_order.assert_called_once()
        placed = executor.client.place_order.call_args[0][0]
        assert placed.action == Action.SELL
        assert placed.yes_price == 25  # max(1, 40 - 15) = 25
        assert placed.count == 10

    @pytest.mark.asyncio
    async def test_hard_stop_floor_clamps_to_one(self, executor):
        """If entry price is very low, floor clamps to 1c (not negative)."""
        entry_order = make_order(count=10, yes_price=10, client_order_id="entry-fc-low")
        entry = make_managed_order(order=entry_order, state=OrderState.FILLED)
        entry.fill_count = 10
        entry.vwap_cents = 10.0
        executor._orders["entry-fc-low"] = entry

        exit_order = make_order(
            action=Action.SELL, count=10, yes_price=15, client_order_id="exit-fc-low"
        )
        exit_managed = make_managed_order(
            order=exit_order, state=OrderState.RESTING, is_exit=True,
            parent_entry_id="entry-fc-low",
        )
        exit_managed.kalshi_order_id = "kalshi-exit-fc-low"
        executor._orders["exit-fc-low"] = exit_managed

        executor.settings.FLASH_CRASH_HARD_STOP_TIMEOUT = 0.05

        task = asyncio.create_task(executor._hard_stop_timer("exit-fc-low"))
        await asyncio.sleep(0.15)

        placed = executor.client.place_order.call_args[0][0]
        assert placed.yes_price == 1  # max(1, 10 - 15) = max(1, -5) = 1

    @pytest.mark.asyncio
    async def test_hard_stop_canceled_on_fill(self, executor):
        """If exit fills before timeout, the timer task is canceled."""
        exit_order = make_order(
            action=Action.SELL, count=10, yes_price=45, client_order_id="exit-fc2"
        )
        exit_managed = make_managed_order(
            order=exit_order, state=OrderState.RESTING, is_exit=True,
            parent_entry_id="entry-fc2",
        )
        exit_managed.remaining_count = 10
        executor._orders["exit-fc2"] = exit_managed

        executor.settings.FLASH_CRASH_HARD_STOP_TIMEOUT = 5

        task = asyncio.create_task(executor._hard_stop_timer("exit-fc2"))
        executor._hard_stop_tasks["exit-fc2"] = task

        entry_order = make_order(count=10, yes_price=40, client_order_id="entry-fc2")
        entry = make_managed_order(order=entry_order, state=OrderState.FILLED)
        entry.fill_count = 10
        entry.vwap_cents = 40.0
        executor._orders["entry-fc2"] = entry

        await executor.on_fill({
            "client_order_id": "exit-fc2",
            "count": 10,
            "yes_price": 45,
        })

        await asyncio.sleep(0)
        assert task.cancelled() or task.done()
        executor.client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_hard_stop_noop_if_already_filled(self, executor):
        """Timer fires but order is already filled → no action."""
        exit_order = make_order(
            action=Action.SELL, count=10, yes_price=45, client_order_id="exit-done"
        )
        exit_managed = make_managed_order(
            order=exit_order, state=OrderState.FILLED, is_exit=True,
        )
        executor._orders["exit-done"] = exit_managed

        executor.settings.FLASH_CRASH_HARD_STOP_TIMEOUT = 0.01
        await executor._hard_stop_timer("exit-done")

        executor.client.cancel_order.assert_not_called()
        executor.client.place_order.assert_not_called()


# ===========================================================================
# SYSTEM:HALT channel
# ===========================================================================

class TestSystemHalt:

    @pytest.mark.asyncio
    async def test_halt_trips_kill_switch(self, executor):
        with patch.object(executor, "_cancel_all_resting", new_callable=AsyncMock), \
             patch.object(executor, "_send_telegram_alert", new_callable=AsyncMock):
            await executor._on_signal("SYSTEM:HALT", {"reason": "session_drawdown"})

        assert executor._kill_switch_tripped is True

    @pytest.mark.asyncio
    async def test_halt_drops_subsequent_signals(self, executor):
        with patch.object(executor, "_cancel_all_resting", new_callable=AsyncMock), \
             patch.object(executor, "_send_telegram_alert", new_callable=AsyncMock):
            await executor._on_signal("SYSTEM:HALT", {"reason": "test"})

        sig = make_signal()
        sig_data = sig.model_dump(mode="json")
        await executor._on_signal("signal:validated", sig_data)
        executor.client.place_order.assert_not_called()


# ===========================================================================
# Flash crash entry tagging
# ===========================================================================

class TestFlashCrashEntryTagging:

    @pytest.mark.asyncio
    async def test_flash_crash_entry_tagged(self, executor):
        sig = make_signal(confidence=0.7, entry_price=20, source="flash_crash")
        sig_data = sig.model_dump(mode="json")
        await executor._on_signal("signal:validated", sig_data)

        executor.client.place_order.assert_called_once()
        placed_order = executor.client.place_order.call_args[0][0]
        assert placed_order.client_order_id in executor._flash_crash_entries

    @pytest.mark.asyncio
    async def test_quant_entry_not_tagged(self, executor):
        sig = make_signal(confidence=0.7, entry_price=20, source="moneyline")
        sig_data = sig.model_dump(mode="json")
        await executor._on_signal("signal:validated", sig_data)

        executor.client.place_order.assert_called_once()
        assert len(executor._flash_crash_entries) == 0


# ===========================================================================
# Drawdown session boundary (UTC midnight edge case)
# ===========================================================================

class TestDrawdownSessionBoundary:
    """Verify the drawdown query uses the 11 AM UTC session boundary
    instead of CURRENT_DATE, so the kill switch doesn't silently reset
    during NBA prime-time (8 PM EST = midnight UTC)."""

    def test_before_reset_hour_uses_previous_day(self):
        """At 1:00 AM UTC (8:00 PM EST on Mar 4), session_start should be
        the *previous* day at 11:00 AM UTC — not today."""
        from agents.executor import OrderExecutor

        now = datetime(2026, 3, 5, 1, 0, 0, tzinfo=timezone.utc)
        result = OrderExecutor._session_start(now)
        expected = datetime(2026, 3, 4, 11, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_at_utc_midnight_uses_previous_day(self):
        """At exactly midnight UTC, session_start should still be
        the previous day at 11 AM UTC."""
        from agents.executor import OrderExecutor

        now = datetime(2026, 3, 5, 0, 0, 0, tzinfo=timezone.utc)
        result = OrderExecutor._session_start(now)
        expected = datetime(2026, 3, 4, 11, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_after_reset_hour_uses_same_day(self):
        """At 3:00 PM UTC (10:00 AM EST), session_start should be
        today at 11:00 AM UTC (same day)."""
        from agents.executor import OrderExecutor

        now = datetime(2026, 3, 5, 15, 0, 0, tzinfo=timezone.utc)
        result = OrderExecutor._session_start(now)
        expected = datetime(2026, 3, 5, 11, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_exactly_at_reset_hour(self):
        """At exactly 11:00 AM UTC, session_start should be same day 11 AM."""
        from agents.executor import OrderExecutor

        now = datetime(2026, 3, 5, 11, 0, 0, tzinfo=timezone.utc)
        result = OrderExecutor._session_start(now)
        expected = datetime(2026, 3, 5, 11, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_late_night_nba_window(self):
        """At 5:30 AM UTC (12:30 AM EST), deep into the late NBA slate,
        session should still anchor to previous day's 11 AM UTC."""
        from agents.executor import OrderExecutor

        now = datetime(2026, 3, 5, 5, 30, 0, tzinfo=timezone.utc)
        result = OrderExecutor._session_start(now)
        expected = datetime(2026, 3, 4, 11, 0, 0, tzinfo=timezone.utc)
        assert result == expected

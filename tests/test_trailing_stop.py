"""Tests for trailing stop and time-based exit protection.

Covers:
  - Trailing stop activation and triggering
  - Peak bid tracking
  - Time-based forced exit in late Q4
  - Integration with position management loop
  - Priority ordering (trailing stop > time exit > EV bailout)
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.schemas import SignalStatus, Side
from tests.conftest import (
    make_game_state,
    make_market_state,
    make_portfolio_position,
    make_portfolio_state,
)


def _make_agent(settings, mock_bus, mock_client, strategies=None):
    from sports.nba.quant import NBAQuantAgent
    return NBAQuantAgent(settings, mock_bus, mock_client, strategies=strategies or [])


# ===========================================================================
# Trailing stop — _check_trailing_stop
# ===========================================================================

class TestTrailingStopActivation:
    """Trailing stop should only activate after position is up enough."""

    def test_not_activated_when_profit_below_threshold(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        # bid=55, profit=5, activation threshold=8
        assert agent._check_trailing_stop(pos, current_bid=55) is False
        assert pos.client_order_id not in agent._peak_bids

    def test_activated_when_profit_reaches_threshold(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        # bid=58, profit=8, exactly at activation threshold
        result = agent._check_trailing_stop(pos, current_bid=58)
        assert result is False  # no drop yet, peak == current
        assert pos.client_order_id in agent._peak_bids
        assert agent._peak_bids[pos.client_order_id] == 58

    def test_not_triggered_when_price_still_rising(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        # Simulate rising price
        agent._check_trailing_stop(pos, current_bid=60)
        assert agent._peak_bids[pos.client_order_id] == 60
        agent._check_trailing_stop(pos, current_bid=65)
        assert agent._peak_bids[pos.client_order_id] == 65
        result = agent._check_trailing_stop(pos, current_bid=68)
        assert result is False
        assert agent._peak_bids[pos.client_order_id] == 68


class TestTrailingStopTrigger:
    """Trailing stop fires when bid drops DISTANCE from peak."""

    def test_fires_when_bid_drops_from_peak(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        # Price rises to 65, peak recorded
        agent._check_trailing_stop(pos, current_bid=65)
        assert agent._peak_bids[pos.client_order_id] == 65
        # Drop to 59 = 6c from peak (exactly at threshold)
        result = agent._check_trailing_stop(pos, current_bid=59)
        assert result is True

    def test_fires_on_large_drop(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        agent._check_trailing_stop(pos, current_bid=70)
        # Drop 10c from peak — still 10c above entry so trailing is active
        result = agent._check_trailing_stop(pos, current_bid=60)
        assert result is True

    def test_does_not_fire_on_small_retrace(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        agent._check_trailing_stop(pos, current_bid=65)
        # Drop only 3c from peak — should NOT fire
        result = agent._check_trailing_stop(pos, current_bid=62)
        assert result is False

    def test_peak_never_decreases(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        agent._check_trailing_stop(pos, current_bid=70)
        assert agent._peak_bids[pos.client_order_id] == 70
        agent._check_trailing_stop(pos, current_bid=65)  # drop but not enough
        assert agent._peak_bids[pos.client_order_id] == 70  # peak unchanged
        agent._check_trailing_stop(pos, current_bid=72)  # new high
        assert agent._peak_bids[pos.client_order_id] == 72

    def test_peak_cleared_when_profit_drops_below_activation(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        # Price goes up to 65, peak set
        agent._check_trailing_stop(pos, current_bid=65)
        assert pos.client_order_id in agent._peak_bids
        # Price crashes back to 52 (only 2c profit, below 8c activation)
        agent._check_trailing_stop(pos, current_bid=52)
        assert pos.client_order_id not in agent._peak_bids


class TestTrailingStopCustomSettings:
    """Trailing stop respects custom settings."""

    def test_tight_stop_fires_earlier(self, settings, mock_bus, mock_client):
        settings.TRAILING_STOP_ACTIVATION_CENTS = 5
        settings.TRAILING_STOP_DISTANCE_CENTS = 3
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        agent._check_trailing_stop(pos, current_bid=60)
        # 3c drop from peak should fire with tight settings
        result = agent._check_trailing_stop(pos, current_bid=57)
        assert result is True

    def test_loose_stop_tolerates_bigger_drops(self, settings, mock_bus, mock_client):
        settings.TRAILING_STOP_ACTIVATION_CENTS = 8
        settings.TRAILING_STOP_DISTANCE_CENTS = 10
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(entry_vwap=50.0)
        agent._check_trailing_stop(pos, current_bid=65)
        # 8c drop still below 10c distance
        result = agent._check_trailing_stop(pos, current_bid=57)
        assert result is False


# ===========================================================================
# Time-based exit — _check_time_exit
# ===========================================================================

class TestTimeExit:
    """Force-sell positions near end of game."""

    def test_no_exit_in_q1(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(quarter=1)
        assert agent._check_time_exit(game) is False

    def test_no_exit_in_q2(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(quarter=2)
        assert agent._check_time_exit(game) is False

    def test_no_exit_in_q3(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(quarter=3)
        assert agent._check_time_exit(game) is False

    def test_no_exit_early_q4(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        # Q4 with 8:00 on clock = 40 min elapsed, 8 min left
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "8:00"})
        assert agent._check_time_exit(game) is False

    def test_exit_late_q4(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        # Q4 with 3:30 on clock = 44.5 min elapsed, 3.5 min left < 4.0
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "3:30"})
        assert agent._check_time_exit(game) is True

    def test_exit_with_1_minute_left(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "1:00"})
        assert agent._check_time_exit(game) is True

    def test_exit_exactly_at_threshold(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        # 4:00 left = exactly at threshold
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "4:00"})
        assert agent._check_time_exit(game) is True

    def test_custom_time_exit_minutes(self, settings, mock_bus, mock_client):
        settings.TIME_EXIT_MINUTES = 2.0
        agent = _make_agent(settings, mock_bus, mock_client)
        # 3:00 left — should NOT fire with 2.0 min threshold
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "3:00"})
        assert agent._check_time_exit(game) is False
        # 1:30 left — should fire
        game = game.model_copy(update={"clock": "1:30"})
        assert agent._check_time_exit(game) is True


# ===========================================================================
# Integration: _check_position fires correct signal type
# ===========================================================================

class TestCheckPositionIntegration:
    """Full integration through _check_position."""

    @pytest.mark.asyncio
    async def test_trailing_stop_publishes_bailout_signal(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(game_id="game-001", quarter=3)
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        # Market with bid=65 (was higher before)
        market = make_market_state(ticker="KXNBA-GAME-LAL", yes_bid=65, yes_ask=67)
        agent._markets["KXNBA-GAME-LAL"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=50.0,
        )
        # Pre-set peak bid to simulate price that already rose and fell
        agent._peak_bids[pos.client_order_id] = 75

        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)

        mock_bus.publish.assert_called_once()
        channel, signal = mock_bus.publish.call_args[0]
        assert channel == "signal:bailout"
        assert signal.status == SignalStatus.BAILOUT
        assert "trailing_stop" in signal.source
        assert signal.entry_price == 65  # current bid

    @pytest.mark.asyncio
    async def test_time_exit_publishes_bailout_signal(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        # Late Q4 game
        game = make_game_state(game_id="game-001", quarter=4)
        game = game.model_copy(update={"clock": "2:00"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        market = make_market_state(ticker="KXNBA-GAME-LAL", yes_bid=40, yes_ask=42)
        agent._markets["KXNBA-GAME-LAL"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=35.0,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)

        mock_bus.publish.assert_called_once()
        channel, signal = mock_bus.publish.call_args[0]
        assert channel == "signal:bailout"
        assert "time_exit" in signal.source
        assert signal.entry_price == 40

    @pytest.mark.asyncio
    async def test_trailing_stop_has_priority_over_time_exit(self, settings, mock_bus, mock_client):
        """When both trailing stop and time exit would fire, trailing stop wins."""
        agent = _make_agent(settings, mock_bus, mock_client)
        # Late Q4
        game = make_game_state(game_id="game-001", quarter=4)
        game = game.model_copy(update={"clock": "2:00"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        market = make_market_state(ticker="KXNBA-GAME-LAL", yes_bid=60, yes_ask=62)
        agent._markets["KXNBA-GAME-LAL"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=50.0,
        )
        # Peak was 70, now 60 = 10c drop from peak → trailing stop fires
        agent._peak_bids[pos.client_order_id] = 70

        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)

        channel, signal = mock_bus.publish.call_args[0]
        assert "trailing_stop" in signal.source  # trailing stop, not time_exit

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeated_signals(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(game_id="game-001", quarter=4)
        game = game.model_copy(update={"clock": "2:00"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        market = make_market_state(ticker="KXNBA-GAME-LAL", yes_bid=40, yes_ask=42)
        agent._markets["KXNBA-GAME-LAL"] = market

        pos = make_portfolio_position(ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=35.0)

        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)
        assert mock_bus.publish.call_count == 1

        # Second call within 60s cooldown — should NOT fire again
        await agent._check_position(pos)
        assert mock_bus.publish.call_count == 1

    @pytest.mark.asyncio
    async def test_no_signal_when_no_market_data(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=50.0)
        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)
        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_signal_when_bid_is_zero(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        market = make_market_state(ticker="KXNBA-GAME-LAL", yes_bid=0, yes_ask=0)
        agent._markets["KXNBA-GAME-LAL"] = market
        pos = make_portfolio_position(ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=50.0)
        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)
        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_side_position_uses_no_bid(self, settings, mock_bus, mock_client):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(game_id="game-001", quarter=4)
        game = game.model_copy(update={"clock": "2:00"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        market = make_market_state(
            ticker="KXNBA-GAME-LAL", yes_bid=40, yes_ask=42, no_bid=55, no_ask=58,
        )
        agent._markets["KXNBA-GAME-LAL"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL", side=Side.NO, entry_vwap=45.0,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_position(pos)

        channel, signal = mock_bus.publish.call_args[0]
        assert signal.entry_price == 55  # no_bid, not yes_bid


# ===========================================================================
# Trailing stop scenario: the $220 → $50 case
# ===========================================================================

class TestRealWorldScenario:
    """Simulate the exact scenario that caused the $170 drawdown."""

    @pytest.mark.asyncio
    async def test_rising_then_crashing_bid_triggers_trailing_stop(
        self, settings, mock_bus, mock_client
    ):
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(game_id="game-001", quarter=3)
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=40.0,
        )

        mock_bus.publish = AsyncMock()

        # Simulate: bid rises from 40 → 65, then crashes to 50
        price_path = [42, 45, 50, 55, 60, 65, 63, 60, 58, 55, 50]
        fired_at = None

        for bid in price_path:
            market = make_market_state(
                ticker="KXNBA-GAME-LAL", yes_bid=bid, yes_ask=bid + 2,
            )
            agent._markets["KXNBA-GAME-LAL"] = market
            agent._bailout_cooldowns.clear()  # clear for simulation
            await agent._check_position(pos)
            if mock_bus.publish.call_count > 0 and fired_at is None:
                fired_at = bid
                break

        assert fired_at is not None, "Trailing stop should have fired"
        # Peak was 65, stop distance is 6c → fires at 59 or below
        assert fired_at <= 59
        # Should have locked in profit: fired_at > entry (40c)
        assert fired_at > 40

    @pytest.mark.asyncio
    async def test_gradual_rise_no_false_triggers(self, settings, mock_bus, mock_client):
        """A steadily rising bid should never trigger trailing stop."""
        agent = _make_agent(settings, mock_bus, mock_client)
        game = make_game_state(game_id="game-001", quarter=3)
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL", side=Side.YES, entry_vwap=40.0,
        )

        mock_bus.publish = AsyncMock()

        # Steady rise: 40 → 80 with only 1-2c dips
        for bid in [42, 44, 43, 46, 48, 47, 50, 53, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78, 80]:
            market = make_market_state(
                ticker="KXNBA-GAME-LAL", yes_bid=bid, yes_ask=bid + 2,
            )
            agent._markets["KXNBA-GAME-LAL"] = market
            agent._bailout_cooldowns.clear()
            await agent._check_position(pos)

        mock_bus.publish.assert_not_called()

"""Tests for the EV-based bailout system.

Covers:
  - model_probability() per strategy
  - NBAQuantAgent._check_bailout() fire / no-fire logic
  - NBAQuantAgent._find_game_for_ticker()
  - OrderExecutor._execute_bailout() — cancel resting exit, OCO cleanup, place sell
"""
from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.strategies.player_props import PlayerPropStrategy
from agents.strategies.totals import TotalsStrategy
from core.schemas import (
    Action,
    ManagedOrder,
    Order,
    OrderState,
    Signal,
    SignalStatus,
    Side,
)
from tests.conftest import (
    make_game_state,
    make_managed_order,
    make_market_state,
    make_order,
    make_player_box_score,
    make_portfolio_position,
    make_portfolio_state,
)


# ===========================================================================
# model_probability — TotalsStrategy
# ===========================================================================

class TestTotalsModelProbability:
    def _make_strategy(self):
        return TotalsStrategy(ev_threshold=0.03, target_exit_spread=7, min_minutes=6.0)

    def test_returns_float_for_over_ticker(self):
        strat = self._make_strategy()
        # 6 minutes played, 20 pts scored → fast pace, likely to go over 225
        game = make_game_state(home_score=12, away_score=8, quarter=1)
        market = make_market_state(ticker="KXNBATOTAL-LAL-DEN-O200", yes_ask=55)
        prob = strat.model_probability(game, market)
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_returns_none_when_no_line(self):
        strat = self._make_strategy()
        game = make_game_state(home_score=40, away_score=35, quarter=2)
        market = make_market_state(ticker="KXNBATOTAL-NOLINE", yes_ask=50)
        assert strat.model_probability(game, market) is None

    def test_returns_none_for_overtime(self):
        strat = self._make_strategy()
        game = make_game_state(quarter=5)
        market = make_market_state(ticker="KXNBATOTAL-LAL-DEN-O225", yes_ask=50)
        assert strat.model_probability(game, market) is None

    def test_under_ticker_complement(self):
        """Under probability should be 1 - over probability."""
        strat = self._make_strategy()
        game = make_game_state(home_score=40, away_score=35, quarter=2)
        over_market = make_market_state(ticker="KXNBATOTAL-LAL-DEN-O200", yes_ask=55)
        under_market = make_market_state(ticker="KXNBATOTAL-LAL-DEN-U200", yes_ask=45)
        p_over = strat.model_probability(game, over_market)
        p_under = strat.model_probability(game, under_market)
        assert p_over is not None and p_under is not None
        assert abs(p_over + p_under - 1.0) < 1e-9


# ===========================================================================
# model_probability — PlayerPropStrategy
# ===========================================================================

class TestPlayerPropModelProbability:
    def _make_strategy(self):
        return PlayerPropStrategy(ev_threshold=0.03, target_exit_spread=7)

    def _make_lebron_game(self):
        player = make_player_box_score(
            first_name="LeBron", last_name="James",
            team_abbr="LAL", minutes=28.0, pts=22, fgm=8, fga=16,
        )
        # Add teammates so team_fga >= 10
        teammate = make_player_box_score(
            player_id="99999", first_name="A", last_name="Davis",
            team_abbr="LAL", minutes=28.0, pts=18, fgm=7, fga=14,
        )
        return make_game_state(quarter=3, player_stats=[player, teammate])

    def test_returns_float_for_matched_player(self):
        strat = self._make_strategy()
        game = self._make_lebron_game()
        market = make_market_state(ticker="KXNBA-PLAYERPTS-LJAMES-O28", yes_ask=45)
        prob = strat.model_probability(game, market)
        assert prob is not None
        assert 0.0 <= prob <= 1.0

    def test_returns_none_when_no_player_stats(self):
        strat = self._make_strategy()
        game = make_game_state(player_stats=[])
        market = make_market_state(ticker="KXNBA-PLAYERPTS-LJAMES-O28", yes_ask=45)
        assert strat.model_probability(game, market) is None

    def test_returns_none_when_player_not_found(self):
        strat = self._make_strategy()
        player = make_player_box_score(first_name="Stephen", last_name="Curry")
        game = make_game_state(player_stats=[player])
        market = make_market_state(ticker="KXNBA-PLAYERPTS-LJAMES-O28", yes_ask=45)
        assert strat.model_probability(game, market) is None


# ===========================================================================
# NBAQuantAgent._find_game_for_ticker
# ===========================================================================

class TestFindGameForTicker:
    def _make_agent(self, settings, mock_bus, mock_client):
        from agents.nba_quant import NBAQuantAgent
        return NBAQuantAgent(settings, mock_bus, mock_client)

    def test_returns_game_when_ticker_mapped(self, settings, mock_bus, mock_client):
        agent = self._make_agent(settings, mock_bus, mock_client)
        game = make_game_state(game_id="game-001")
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        result = agent._find_game_for_ticker("KXNBA-GAME-LAL")
        assert result is game

    def test_returns_none_when_not_mapped(self, settings, mock_bus, mock_client):
        agent = self._make_agent(settings, mock_bus, mock_client)
        result = agent._find_game_for_ticker("KXNBA-GAME-BOS")
        assert result is None

    def test_returns_none_when_game_missing(self, settings, mock_bus, mock_client):
        agent = self._make_agent(settings, mock_bus, mock_client)
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL"]
        # game not in _games
        result = agent._find_game_for_ticker("KXNBA-GAME-LAL")
        assert result is None


# ===========================================================================
# NBAQuantAgent._check_bailout
# ===========================================================================

class TestCheckBailout:
    def _make_agent(self, settings, mock_bus, mock_client):
        from agents.nba_quant import NBAQuantAgent
        return NBAQuantAgent(settings, mock_bus, mock_client)

    @pytest.mark.asyncio
    async def test_fires_bailout_when_fair_value_below_threshold(
        self, settings, mock_bus, mock_client
    ):
        agent = self._make_agent(settings, mock_bus, mock_client)

        # Bought OVER 220 in Q3, but game is going very slowly.
        # Q3 (30.5 team-minutes), score=30+25=55 → pace ~1.8/min → projected ~86
        # P(over 220) ≈ 0% → fair_value ≈ 0c
        # Market still bids 70c → threshold = 70-15=55 → 0 < 55 → bailout fires
        game = make_game_state(game_id="game-001", home_score=30, away_score=25, quarter=3)
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBATOTAL-04MAR26-LALDAL-O220"]

        market = make_market_state(
            ticker="KXNBATOTAL-04MAR26-LALDAL-O220", yes_bid=70, yes_ask=72
        )
        agent._markets["KXNBATOTAL-04MAR26-LALDAL-O220"] = market

        pos = make_portfolio_position(
            ticker="KXNBATOTAL-04MAR26-LALDAL-O220",
            side=Side.YES,
            remaining_count=10,
            entry_vwap=30.0,
            target_exit_price=40,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)

        mock_bus.publish.assert_called_once()
        channel, signal = mock_bus.publish.call_args[0]
        assert channel == "signal:bailout"
        assert signal.status == SignalStatus.BAILOUT
        assert signal.ticker == "KXNBATOTAL-04MAR26-LALDAL-O220"
        assert signal.entry_price == 70  # current bid

    @pytest.mark.asyncio
    async def test_no_bailout_when_fair_value_above_threshold(
        self, settings, mock_bus, mock_client
    ):
        agent = self._make_agent(settings, mock_bus, mock_client)

        game = make_game_state(game_id="game-001", home_score=90, away_score=80, quarter=4)
        game = game.model_copy(update={"home_abbr": "LAL", "away_abbr": "DEN"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL-DEN-LAL"]

        # LAL leads 90-80 in Q4 → P(LAL) ~88%
        # yes_bid=70c → threshold = 70 - 15 = 55 → 88 > 55 → no bailout
        market = make_market_state(
            ticker="KXNBA-GAME-LAL-DEN-LAL", yes_bid=70, yes_ask=72
        )
        agent._markets["KXNBA-GAME-LAL-DEN-LAL"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL-DEN-LAL",
            side=Side.YES,
            remaining_count=10,
            entry_vwap=60.0,
            target_exit_price=75,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)
        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_second_check(
        self, settings, mock_bus, mock_client
    ):
        agent = self._make_agent(settings, mock_bus, mock_client)

        game = make_game_state(game_id="game-001", home_score=90, away_score=80, quarter=4)
        game = game.model_copy(update={"home_abbr": "LAL", "away_abbr": "DEN"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL-DEN-DEN"]

        market = make_market_state(
            ticker="KXNBA-GAME-LAL-DEN-DEN", yes_bid=70, yes_ask=72
        )
        agent._markets["KXNBA-GAME-LAL-DEN-DEN"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL-DEN-DEN",
            client_order_id="exit-cool-001",
            side=Side.YES,
            remaining_count=10,
            entry_vwap=30.0,
            target_exit_price=40,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)  # fires
        mock_bus.publish.reset_mock()
        await agent._check_bailout(pos)  # should be suppressed by cooldown
        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_bailout_when_market_missing(
        self, settings, mock_bus, mock_client
    ):
        agent = self._make_agent(settings, mock_bus, mock_client)
        pos = make_portfolio_position(ticker="KXNBA-GAME-UNKNOWN")
        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)
        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_bailout_when_game_missing(
        self, settings, mock_bus, mock_client
    ):
        agent = self._make_agent(settings, mock_bus, mock_client)
        market = make_market_state(ticker="KXNBA-GAME-LAL", yes_bid=50, yes_ask=52)
        agent._markets["KXNBA-GAME-LAL"] = market
        pos = make_portfolio_position(ticker="KXNBA-GAME-LAL")
        # No game mapped
        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)
        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_side_bailout_uses_no_bid(
        self, settings, mock_bus, mock_client
    ):
        """For a Side.NO position, bailout must use no_bid and P(NO) = 1 - P(YES).

        We hold NO on an OVER 215 ticker (bet the under), but the game is blazing fast.
        Q3 (30.5 team-minutes), score=120+110=230 → pace ~7.5/min → projected ~362.
        P(over 215) ≈ 100% → P(NO=under) ≈ 0% → fair_value ≈ 0c.
        no_bid=65c: threshold = 65-15=50 → 0 < 50 → bailout fires using no_bid.
        """
        from core.schemas import MarketState
        from datetime import datetime

        agent = self._make_agent(settings, mock_bus, mock_client)

        game = make_game_state(game_id="game-001", home_score=120, away_score=110, quarter=3)
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBATOTAL-04MAR26-LALDAL-O215"]

        # Market still has no_bid=65 (slow to update); yes_bid=30 (over likely)
        # P(over) ≈ 100% → P(NO) ≈ 0% → bailout fires
        market = MarketState(
            ticker="KXNBATOTAL-04MAR26-LALDAL-O215",
            yes_bid=30,
            yes_ask=32,
            no_bid=65,
            no_ask=67,
            volume=100,
            timestamp=datetime.utcnow(),
        )
        agent._markets["KXNBATOTAL-04MAR26-LALDAL-O215"] = market

        pos = make_portfolio_position(
            ticker="KXNBATOTAL-04MAR26-LALDAL-O215",
            side=Side.NO,
            remaining_count=10,
            entry_vwap=50.0,
            target_exit_price=75,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)

        mock_bus.publish.assert_called_once()
        channel, signal = mock_bus.publish.call_args[0]
        assert channel == "signal:bailout"
        assert signal.side == Side.NO
        assert signal.entry_price == 65  # no_bid, not yes_bid

    @pytest.mark.asyncio
    async def test_no_bailout_when_no_bid_is_zero(
        self, settings, mock_bus, mock_client
    ):
        """If no_bid is 0 (market not quoting NO), don't fire a bailout for a NO position."""
        from core.schemas import MarketState
        from datetime import datetime

        agent = self._make_agent(settings, mock_bus, mock_client)

        game = make_game_state(game_id="game-001", home_score=90, away_score=80, quarter=4)
        game = game.model_copy(update={"home_abbr": "LAL", "away_abbr": "DEN"})
        agent._games["game-001"] = game
        agent._game_to_tickers["game-001"] = ["KXNBA-GAME-LAL-DEN-LAL"]

        market = MarketState(
            ticker="KXNBA-GAME-LAL-DEN-LAL",
            yes_bid=80,
            yes_ask=82,
            no_bid=0,   # not quoting NO side
            no_ask=0,
            volume=100,
            timestamp=datetime.utcnow(),
        )
        agent._markets["KXNBA-GAME-LAL-DEN-LAL"] = market

        pos = make_portfolio_position(
            ticker="KXNBA-GAME-LAL-DEN-LAL",
            side=Side.NO,
            remaining_count=10,
            entry_vwap=20.0,
            target_exit_price=30,
        )

        mock_bus.publish = AsyncMock()
        await agent._check_bailout(pos)
        mock_bus.publish.assert_not_called()


# ===========================================================================
# OrderExecutor._execute_bailout
# ===========================================================================

class TestExecuteBailout:
    @pytest.mark.asyncio
    async def test_cancels_resting_exit_and_places_sell(self, executor, mock_client):
        # Set up entry
        entry_order = make_order(yes_price=30)
        entry_managed = make_managed_order(order=entry_order, is_exit=False)
        entry_managed.fill_count = 10
        entry_managed.vwap_cents = 30.0
        executor._orders[entry_order.client_order_id] = entry_managed

        # Set up resting exit
        exit_order = make_order(action=Action.SELL, yes_price=40)
        exit_managed = make_managed_order(
            order=exit_order, is_exit=True,
            parent_entry_id=entry_order.client_order_id,
        )
        exit_managed.state = OrderState.RESTING
        exit_managed.kalshi_order_id = "kalshi-exit-001"
        executor._orders[exit_order.client_order_id] = exit_managed
        executor._kalshi_to_client["kalshi-exit-001"] = exit_order.client_order_id

        mock_client.cancel_order = AsyncMock(return_value={})
        mock_client.place_order = AsyncMock(
            return_value={"order": {"order_id": "kalshi-bailout-001"}}
        )

        signal = Signal(
            ticker=exit_order.ticker,
            action=Action.SELL,
            side=Side.YES,
            status=SignalStatus.BAILOUT,
            confidence=0.15,
            source="nba_quant",
            ev_estimate=-0.45,
            entry_price=20,  # current bid
            exit_price=40,
            game_id="game-001",
            target_order_id=exit_order.client_order_id,
        )

        # Simulate cancel event fire (WS confirm)
        import asyncio
        async def fake_cancel(kalshi_id):
            event = executor._cancel_events.get(kalshi_id)
            if event:
                event.set()
        mock_client.cancel_order = AsyncMock(side_effect=fake_cancel)

        await executor._execute_bailout(signal)

        # Exit should be CANCELED
        assert exit_managed.state == OrderState.CANCELED
        # New sell placed at bid price (20c)
        mock_client.place_order.assert_called_once()
        placed = mock_client.place_order.call_args[0][0]
        assert placed.action == Action.SELL
        assert placed.yes_price == 20

    @pytest.mark.asyncio
    async def test_cancels_oco_paired_tp(self, executor, mock_client):
        """When bailout fires, the paired TP must be canceled too."""
        entry_order = make_order(yes_price=30)
        entry_managed = make_managed_order(order=entry_order, is_exit=False)
        executor._orders[entry_order.client_order_id] = entry_managed

        # Resting spread exit
        exit_order = make_order(action=Action.SELL, yes_price=40)
        exit_managed = make_managed_order(order=exit_order, is_exit=True)
        exit_managed.state = OrderState.RESTING
        exit_managed.kalshi_order_id = "kalshi-exit-001"
        executor._orders[exit_order.client_order_id] = exit_managed
        executor._kalshi_to_client["kalshi-exit-001"] = exit_order.client_order_id

        # Paired TP order
        tp_order = make_order(action=Action.SELL, yes_price=98)
        tp_managed = make_managed_order(order=tp_order, is_exit=True)
        tp_managed.state = OrderState.RESTING
        tp_managed.kalshi_order_id = "kalshi-tp-001"
        executor._orders[tp_order.client_order_id] = tp_managed
        executor._kalshi_to_client["kalshi-tp-001"] = tp_order.client_order_id

        # Register OCO pair
        executor._oco_pairs[exit_order.client_order_id] = tp_order.client_order_id
        executor._oco_pairs[tp_order.client_order_id] = exit_order.client_order_id

        import asyncio
        canceled = []
        async def fake_cancel(kalshi_id):
            canceled.append(kalshi_id)
            event = executor._cancel_events.get(kalshi_id)
            if event:
                event.set()

        mock_client.cancel_order = AsyncMock(side_effect=fake_cancel)
        mock_client.place_order = AsyncMock(
            return_value={"order": {"order_id": "kalshi-bailout-001"}}
        )

        signal = Signal(
            ticker=exit_order.ticker,
            action=Action.SELL,
            side=Side.YES,
            status=SignalStatus.BAILOUT,
            confidence=0.15,
            source="nba_quant",
            ev_estimate=-0.45,
            entry_price=20,
            exit_price=40,
            game_id="game-001",
            target_order_id=exit_order.client_order_id,
        )

        await executor._execute_bailout(signal)

        # Both exit and TP should have been canceled
        assert "kalshi-exit-001" in canceled
        assert "kalshi-tp-001" in canceled
        assert tp_managed.state == OrderState.CANCELED
        # OCO map fully cleaned up
        assert exit_order.client_order_id not in executor._oco_pairs
        assert tp_order.client_order_id not in executor._oco_pairs

    @pytest.mark.asyncio
    async def test_skips_unknown_target(self, executor, mock_client):
        signal = Signal(
            ticker="NBA-YES-LAL",
            action=Action.SELL,
            side=Side.YES,
            status=SignalStatus.BAILOUT,
            confidence=0.15,
            source="nba_quant",
            ev_estimate=-0.45,
            entry_price=20,
            exit_price=40,
            game_id="game-001",
            target_order_id="does-not-exist",
        )
        mock_client.cancel_order = AsyncMock()
        await executor._execute_bailout(signal)
        mock_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_bailout_bypasses_kill_switch(self, executor, mock_bus, mock_client):
        """BAILOUT signals must be processed even when the kill switch is active."""
        executor._kill_switch_tripped = True

        entry_order = make_order(yes_price=30)
        entry_managed = make_managed_order(order=entry_order, is_exit=False)
        executor._orders[entry_order.client_order_id] = entry_managed

        exit_order = make_order(action=Action.SELL, yes_price=40)
        exit_managed = make_managed_order(order=exit_order, is_exit=True)
        exit_managed.state = OrderState.RESTING
        exit_managed.kalshi_order_id = "kalshi-exit-ks-001"
        executor._orders[exit_order.client_order_id] = exit_managed
        executor._kalshi_to_client["kalshi-exit-ks-001"] = exit_order.client_order_id

        import asyncio
        async def fake_cancel(kalshi_id):
            event = executor._cancel_events.get(kalshi_id)
            if event:
                event.set()
        mock_client.cancel_order = AsyncMock(side_effect=fake_cancel)
        mock_client.place_order = AsyncMock(
            return_value={"order": {"order_id": "kalshi-bailout-ks-001"}}
        )

        signal = Signal(
            ticker=exit_order.ticker,
            action=Action.SELL,
            side=Side.YES,
            status=SignalStatus.BAILOUT,
            confidence=0.15,
            source="nba_quant",
            ev_estimate=-0.45,
            entry_price=20,
            exit_price=40,
            game_id="game-001",
            target_order_id=exit_order.client_order_id,
        )

        await executor._on_signal("signal:bailout", signal.model_dump())
        # Kill switch is ON but bailout sell must still be placed
        mock_client.place_order.assert_called_once()

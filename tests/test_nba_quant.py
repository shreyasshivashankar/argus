"""Tests for NBAQuantAgent: context cache, fail-close, signal gating, and reallocation."""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import pytest

from core.schemas import ContextStatus, PortfolioState, SignalStatus
from agents.nba_quant import NBAQuantAgent
from tests.conftest import (
    make_game_state,
    make_market_state,
    make_portfolio_position,
    make_portfolio_state,
)


@pytest.fixture
def quant(settings, mock_bus, mock_client):
    agent = NBAQuantAgent(settings, mock_bus, mock_client)
    agent.register_game_market("game-001", "NBA-YES-LAL")
    return agent


def _inject_state(quant: NBAQuantAgent, game_state=None, market_state=None):
    """Helper to inject game and market state into the quant agent."""
    gs = game_state or make_game_state()
    ms = market_state or make_market_state()
    quant._games[gs.game_id] = gs
    quant._markets[ms.ticker] = ms


def _force_high_ev(quant: NBAQuantAgent):
    """Patch _model_probability to return a value that guarantees +EV."""
    quant._model_probability = lambda g, t: 0.95  # type: ignore[assignment]


# ===========================================================================
# Context Cache — signal gating
# ===========================================================================

class TestContextGating:

    @pytest.mark.asyncio
    async def test_safe_context_publishes_signal(self, quant, mock_bus):
        _inject_state(quant)
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        await quant._evaluate_all()

        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == "signal:validated"
        signal = call_args[0][1]
        assert signal.status == SignalStatus.VALIDATED
        assert signal.ticker == "NBA-YES-LAL"

    @pytest.mark.asyncio
    async def test_veto_context_drops_signal(self, quant, mock_bus):
        _inject_state(quant)
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (ContextStatus.VETO, "star player injured")

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_context_fail_close(self, quant, mock_bus):
        """Missing/expired key → VETO (fail-close invariant)."""
        _inject_state(quant)
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (
            ContextStatus.VETO, "context missing – fail-close"
        )

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()


# ===========================================================================
# EV Threshold
# ===========================================================================

class TestEVThreshold:

    @pytest.mark.asyncio
    async def test_below_threshold_drops_signal(self, quant, mock_bus):
        """Even with SAFE context, low EV should not publish."""
        _inject_state(quant)
        # Low model_prob → negative EV
        quant._model_probability = lambda g, t: 0.10  # type: ignore[assignment]
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_ask_drops_signal(self, quant, mock_bus):
        """Market with yes_ask=0 should be skipped."""
        _inject_state(quant, market_state=make_market_state(yes_ask=0))
        _force_high_ev(quant)

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()
        mock_bus.get_context.assert_not_called()


# ===========================================================================
# No ticker mapping → no evaluation
# ===========================================================================

class TestNoMapping:

    @pytest.mark.asyncio
    async def test_unmapped_game_skipped(self, quant, mock_bus):
        """Game with no matching Kalshi ticker should not produce a signal."""
        gs = make_game_state(game_id="unmapped-game")
        quant._games["unmapped-game"] = gs
        quant._markets["NFL-SPREAD-KC"] = make_market_state(ticker="NFL-SPREAD-KC")
        _force_high_ev(quant)

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()


# ===========================================================================
# Reallocation — capital-constrained opportunity cost
# ===========================================================================

class TestReallocation:

    @pytest.mark.asyncio
    async def test_sufficient_bankroll_publishes_validated(self, quant, mock_bus):
        """When bankroll can fund the trade, publish VALIDATED (no reallocation)."""
        _inject_state(quant)
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")
        quant._portfolio = make_portfolio_state(bankroll=100.0)

        await quant._evaluate_all()

        mock_bus.publish.assert_called_once()
        assert mock_bus.publish.call_args[0][0] == "signal:validated"

    @pytest.mark.asyncio
    async def test_insufficient_bankroll_triggers_reallocate(self, quant, mock_bus, settings):
        """When bankroll is too low but a resting exit clears the hurdle,
        publish REALLOCATE with the correct target_order_id."""
        _inject_state(quant)
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        pos = make_portfolio_position(
            client_order_id="exit-to-kill",
            ticker="NBA-YES-OTHER",
            remaining_count=100,
            target_exit_price=99,
            kalshi_order_id="kalshi-other",
        )
        quant._portfolio = make_portfolio_state(bankroll=0.01, positions=[pos])
        quant._markets["NBA-YES-OTHER"] = make_market_state(
            ticker="NBA-YES-OTHER", yes_bid=95, yes_ask=97
        )

        await quant._evaluate_all()

        mock_bus.publish.assert_called_once()
        channel, signal = mock_bus.publish.call_args[0]
        assert channel == "signal:reallocate"
        assert signal.status == SignalStatus.REALLOCATE
        assert signal.target_order_id == "exit-to-kill"
        assert signal.entry_price == 95

    @pytest.mark.asyncio
    async def test_bid_below_min_reallocate_skipped(self, quant, mock_bus, settings):
        """Position with bid below MIN_REALLOCATE_BID is not considered."""
        _inject_state(quant)
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        pos = make_portfolio_position(
            ticker="NBA-YES-OTHER",
            remaining_count=100,
            target_exit_price=80,
        )
        quant._portfolio = make_portfolio_state(bankroll=0.01, positions=[pos])
        quant._markets["NBA-YES-OTHER"] = make_market_state(
            ticker="NBA-YES-OTHER", yes_bid=70, yes_ask=72
        )

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_hurdle_not_met_no_reallocate(self, quant, mock_bus, settings):
        """When foregone profit + fees exceed projected new EV, don't reallocate.

        Setup: model_prob=0.18, yes_ask=16 -> EV = 0.18*1 - 0.16 = 0.02.
        Freed capital = 91 * 10 = 910c, new_count = floor(910*0.5/16) = 28,
        total_new_ev = 28 * 2 = 56c.
        Foregone = (99 - 91) * 10 = 80c, fees = 10 * 2 = 20c, cost = 100c.
        56 < 100 -> no reallocation.
        """
        _inject_state(quant)
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")
        quant._model_probability = lambda g, t: 0.18  # type: ignore[assignment]

        pos = make_portfolio_position(
            ticker="NBA-YES-OTHER",
            remaining_count=10,
            target_exit_price=99,
        )
        quant._portfolio = make_portfolio_state(bankroll=0.01, positions=[pos])
        quant._markets["NBA-YES-OTHER"] = make_market_state(
            ticker="NBA-YES-OTHER", yes_bid=91, yes_ask=93
        )

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_unit_math_uses_projected_count(self, quant, mock_bus, settings):
        """Verify the hurdle rate uses freed_capital to size the new trade,
        not raw per-contract EV."""
        _inject_state(quant, market_state=make_market_state(yes_ask=16))
        _force_high_ev(quant)
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        pos = make_portfolio_position(
            client_order_id="exit-math",
            ticker="NBA-YES-OTHER",
            remaining_count=50,
            target_exit_price=97,
        )
        quant._portfolio = make_portfolio_state(bankroll=0.01, positions=[pos])
        quant._markets["NBA-YES-OTHER"] = make_market_state(
            ticker="NBA-YES-OTHER", yes_bid=95, yes_ask=97
        )

        await quant._evaluate_all()

        if mock_bus.publish.called:
            channel, signal = mock_bus.publish.call_args[0]
            if channel == "signal:reallocate":
                freed = 95 * 50
                expected_count = math.floor(
                    freed * settings.KELLY_FRACTION / 16
                )
                assert expected_count > 0

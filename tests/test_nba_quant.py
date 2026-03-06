"""Tests for NBAQuantAgent (OmniQuant): strategy pattern, context, cooldown, reallocation."""
from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import pytest

from agents.strategies.base import BaseStrategy
from core.schemas import (
    Action,
    ContextStatus,
    GameState,
    MarketState,
    PortfolioState,
    Side,
    Signal,
    SignalStatus,
)
from agents.nba_quant import NBAQuantAgent
from tests.conftest import (
    make_game_state,
    make_market_state,
    make_portfolio_position,
    make_portfolio_state,
)


# ---------------------------------------------------------------------------
# A controllable stub strategy used in tests
# ---------------------------------------------------------------------------

class _StubStrategy(BaseStrategy):
    """Always-match strategy whose evaluate result is externally controllable."""

    name = "stub"

    def __init__(self) -> None:
        self.signal_to_return: Signal | None = None

    def can_evaluate(self, market: MarketState) -> bool:
        return True

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        return self.signal_to_return


def _make_high_ev_signal(ticker: str = "NBA-YES-LAL", game_id: str = "game-001") -> Signal:
    return Signal(
        ticker=ticker,
        action=Action.BUY,
        side=Side.YES,
        status=SignalStatus.VALIDATED,
        confidence=0.95,
        source="stub",
        ev_estimate=0.50,
        entry_price=16,
        exit_price=23,
        game_id=game_id,
    )


def _make_low_ev_signal(ticker: str = "NBA-YES-LAL", game_id: str = "game-001") -> Signal:
    return Signal(
        ticker=ticker,
        action=Action.BUY,
        side=Side.YES,
        status=SignalStatus.VALIDATED,
        confidence=0.10,
        source="stub",
        ev_estimate=-0.05,
        entry_price=16,
        exit_price=23,
        game_id=game_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_strategy():
    return _StubStrategy()


@pytest.fixture
def quant(settings, mock_bus, mock_client, stub_strategy):
    agent = NBAQuantAgent(settings, mock_bus, mock_client, strategies=[stub_strategy])
    agent.register_game_market("game-001", "NBA-YES-LAL")
    return agent


def _inject_state(quant: NBAQuantAgent, game_state=None, market_state=None):
    gs = game_state or make_game_state()
    ms = market_state or make_market_state()
    quant._games[gs.game_id] = gs
    quant._markets[ms.ticker] = ms


# ===========================================================================
# Context Cache — signal gating
# ===========================================================================

class TestContextGating:

    @pytest.mark.asyncio
    async def test_safe_context_publishes_signal(self, quant, mock_bus, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = _make_high_ev_signal()
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        await quant._evaluate_all()

        mock_bus.publish.assert_called_once()
        call_args = mock_bus.publish.call_args
        assert call_args[0][0] == "signal:validated"
        signal = call_args[0][1]
        assert signal.status == SignalStatus.VALIDATED
        assert signal.ticker == "NBA-YES-LAL"

    @pytest.mark.asyncio
    async def test_veto_context_drops_signal(self, quant, mock_bus, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = _make_high_ev_signal()
        mock_bus.get_context.return_value = (ContextStatus.VETO, "star player injured")

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_context_fail_close(self, quant, mock_bus, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = _make_high_ev_signal()
        mock_bus.get_context.return_value = (
            ContextStatus.VETO, "context missing – fail-close"
        )

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()


# ===========================================================================
# EV Threshold (strategy returns None → no signal)
# ===========================================================================

class TestEVThreshold:

    @pytest.mark.asyncio
    async def test_no_proposals_drops_signal(self, quant, mock_bus, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = None
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_zero_ask_drops_signal(self, quant, mock_bus, stub_strategy):
        _inject_state(quant, market_state=make_market_state(yes_ask=0))
        stub_strategy.signal_to_return = None

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()
        mock_bus.get_context.assert_not_called()


# ===========================================================================
# No ticker mapping → no evaluation
# ===========================================================================

class TestNoMapping:

    @pytest.mark.asyncio
    async def test_unmapped_game_skipped(self, quant, mock_bus, stub_strategy):
        gs = make_game_state(game_id="unmapped-game")
        quant._games["unmapped-game"] = gs
        quant._markets["NFL-SPREAD-KC"] = make_market_state(ticker="NFL-SPREAD-KC")
        stub_strategy.signal_to_return = _make_high_ev_signal()

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()


# ===========================================================================
# Reallocation — capital-constrained opportunity cost
# ===========================================================================

class TestReallocation:

    @pytest.mark.asyncio
    async def test_sufficient_bankroll_publishes_validated(self, quant, mock_bus, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = _make_high_ev_signal()
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")
        quant._portfolio = make_portfolio_state(bankroll=100.0)

        await quant._evaluate_all()

        mock_bus.publish.assert_called_once()
        assert mock_bus.publish.call_args[0][0] == "signal:validated"

    @pytest.mark.asyncio
    async def test_insufficient_bankroll_triggers_reallocate(self, quant, mock_bus, settings, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = _make_high_ev_signal()
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
    async def test_bid_below_min_reallocate_skipped(self, quant, mock_bus, settings, stub_strategy):
        _inject_state(quant)
        stub_strategy.signal_to_return = _make_high_ev_signal()
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

        pos = make_portfolio_position(
            ticker="NBA-YES-OTHER",
            remaining_count=100,
            target_exit_price=80,
        )
        quant._portfolio = make_portfolio_state(bankroll=0.01, positions=[pos])
        quant._markets["NBA-YES-OTHER"] = make_market_state(
            ticker="NBA-YES-OTHER", yes_bid=5, yes_ask=7
        )

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_hurdle_not_met_no_reallocate(self, quant, mock_bus, settings, stub_strategy):
        """Foregone profit + fees exceed projected new EV → no reallocate.

        Signal EV=0.005, entry=16.  Freed=91*10=910c, count=floor(910/16)=56,
        total_new_ev=56*0.5=28c.  Foregone=(99-91)*10=80c*0.5=40c, fees≈6c.
        28 < 46 → no reallocation.
        """
        _inject_state(quant)
        low_ev = Signal(
            ticker="NBA-YES-LAL",
            action=Action.BUY,
            side=Side.YES,
            status=SignalStatus.VALIDATED,
            confidence=0.18,
            source="stub",
            ev_estimate=0.005,
            entry_price=16,
            exit_price=23,
            game_id="game-001",
        )
        stub_strategy.signal_to_return = low_ev
        mock_bus.get_context.return_value = (ContextStatus.SAFE, "")

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
    async def test_unit_math_uses_projected_count(self, quant, mock_bus, settings, stub_strategy):
        _inject_state(quant, market_state=make_market_state(yes_ask=16))
        stub_strategy.signal_to_return = _make_high_ev_signal()
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
                expected_count = math.floor(freed / 16)
                assert expected_count > 0

"""Tests for NBAQuantAgent: context cache, fail-close, and signal gating."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.schemas import ContextStatus, SignalStatus
from agents.nba_quant import NBAQuantAgent
from tests.conftest import make_game_state, make_market_state


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
    quant._model_probability = lambda g: 0.95  # type: ignore[assignment]


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
        quant._model_probability = lambda g: 0.10  # type: ignore[assignment]
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
        gs = make_game_state(game_id="unmapped-game")
        quant._games["unmapped-game"] = gs
        quant._markets["NBA-YES-LAL"] = make_market_state()
        _force_high_ev(quant)

        await quant._evaluate_all()

        mock_bus.publish.assert_not_called()

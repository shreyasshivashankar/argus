"""Shared fixtures for the Argus test suite."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.schemas import (
    Action,
    AppSettings,
    ContextStatus,
    GameState,
    ManagedOrder,
    MarketState,
    Order,
    OrderState,
    PlayerBoxScore,
    PortfolioPosition,
    PortfolioState,
    Side,
    Signal,
    SignalStatus,
)


# ---------------------------------------------------------------------------
# Settings fixture (no .env required, all defaults overridden)
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        KALSHI_API_KEY_ID="test-key",
        KALSHI_PRIVATE_KEY_PATH="/dev/null",
        KALSHI_ENV="demo",
        REDIS_URL="redis://localhost:6379",
        OPENAI_API_KEY="test-llm-key",
        OPENAI_MODEL="gpt-4o-mini",
        TELEGRAM_CHAT_ID="",
        DAILY_STOP_LOSS_USD=50.0,
        SLIPPAGE_TICKS=2,
        KELLY_FRACTION=0.5,
        TARGET_EXIT_SPREAD=7,
        CONTEXT_POLL_INTERVAL=15.0,
        BALANCE_POLL_INTERVAL=10.0,
        HEARTBEAT_INTERVAL=30.0,
        CONTEXT_TTL=300,
        EV_THRESHOLD=3.0,
        ORDER_GC_INTERVAL=300.0,
        ORDER_GC_TTL=7200.0,
    )


# ---------------------------------------------------------------------------
# Mock bus
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.get_context = AsyncMock(return_value=(ContextStatus.SAFE, ""))
    bus.set_context = AsyncMock()
    bus.close = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# Mock Kalshi client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.place_order = AsyncMock(
        return_value={"order": {"order_id": "kalshi-order-001"}}
    )
    client.cancel_order = AsyncMock(return_value={})
    client.get_balance = AsyncMock(return_value=1000.0)
    client.get_positions = AsyncMock(return_value=[])
    client.get_orders = AsyncMock(return_value=[])
    client.get_fills = AsyncMock(return_value=[])
    client.close = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_signal(
    *,
    ticker: str = "NBA-YES-LAL",
    confidence: float = 0.7,
    entry_price: int = 15,
    exit_price: int = 22,
    game_id: str = "game-001",
    ev_estimate: float = 0.05,
    status: SignalStatus = SignalStatus.VALIDATED,
) -> Signal:
    return Signal(
        ticker=ticker,
        action=Action.BUY,
        side=Side.YES,
        status=status,
        confidence=confidence,
        source="test",
        ev_estimate=ev_estimate,
        entry_price=entry_price,
        exit_price=exit_price,
        game_id=game_id,
    )


def make_order(
    *,
    ticker: str = "NBA-YES-LAL",
    action: Action = Action.BUY,
    side: Side = Side.YES,
    count: int = 10,
    yes_price: int | None = 17,
    client_order_id: str | None = None,
) -> Order:
    return Order(
        ticker=ticker,
        action=action,
        side=side,
        count=count,
        yes_price=yes_price,
        client_order_id=client_order_id or str(uuid.uuid4()),
    )


def make_managed_order(
    *,
    order: Order | None = None,
    state: OrderState = OrderState.PLACED,
    signal_id: str = "sig-001",
    is_exit: bool = False,
    parent_entry_id: str | None = None,
    created_at: datetime | None = None,
) -> ManagedOrder:
    if order is None:
        order = make_order()
    kwargs: dict[str, Any] = {
        "order": order,
        "state": state,
        "signal_id": signal_id,
        "is_exit": is_exit,
    }
    if parent_entry_id is not None:
        kwargs["parent_entry_id"] = parent_entry_id
    if created_at is not None:
        kwargs["created_at"] = created_at
    return ManagedOrder(**kwargs)


def make_player_box_score(
    *,
    player_id: str = "12345",
    first_name: str = "LeBron",
    last_name: str = "James",
    team_abbr: str = "LAL",
    minutes: float = 28.0,
    pts: int = 22,
    fgm: int = 8,
    fga: int = 16,
    fg3m: int = 3,
    fg3a: int = 6,
    ftm: int = 3,
    fta: int = 4,
    reb: int = 7,
    ast: int = 5,
) -> PlayerBoxScore:
    return PlayerBoxScore(
        player_id=player_id,
        first_name=first_name,
        last_name=last_name,
        team_abbr=team_abbr,
        minutes=minutes,
        pts=pts,
        fgm=fgm,
        fga=fga,
        fg3m=fg3m,
        fg3a=fg3a,
        ftm=ftm,
        fta=fta,
        reb=reb,
        ast=ast,
    )


def make_game_state(
    *,
    game_id: str = "game-001",
    home_score: int = 80,
    away_score: int = 85,
    quarter: int = 3,
    player_stats: list[PlayerBoxScore] | None = None,
) -> GameState:
    return GameState(
        game_id=game_id,
        home_team="LAL",
        away_team="DEN",
        home_abbr="LAL",
        away_abbr="DEN",
        home_score=home_score,
        away_score=away_score,
        quarter=quarter,
        clock="5:30",
        timestamp=datetime.utcnow(),
        player_stats=player_stats or [],
    )


def make_market_state(
    *,
    ticker: str = "NBA-YES-LAL",
    yes_bid: int = 14,
    yes_ask: int = 16,
) -> MarketState:
    return MarketState(
        ticker=ticker,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=0,
        no_ask=0,
        volume=100,
        timestamp=datetime.utcnow(),
    )


def make_portfolio_position(
    *,
    client_order_id: str = "exit-001",
    ticker: str = "NBA-YES-LAL",
    side: Side = Side.YES,
    remaining_count: int = 100,
    entry_vwap: float = 15.0,
    target_exit_price: int = 22,
    kalshi_order_id: str = "kalshi-exit-001",
) -> PortfolioPosition:
    return PortfolioPosition(
        client_order_id=client_order_id,
        ticker=ticker,
        side=side,
        remaining_count=remaining_count,
        entry_vwap=entry_vwap,
        target_exit_price=target_exit_price,
        kalshi_order_id=kalshi_order_id,
    )


def make_portfolio_state(
    *,
    bankroll: float = 0.50,
    positions: list[PortfolioPosition] | None = None,
) -> PortfolioState:
    return PortfolioState(
        bankroll=bankroll,
        positions=positions or [],
    )


# ---------------------------------------------------------------------------
# Executor fixture (wired to mocks)
# ---------------------------------------------------------------------------

@pytest.fixture
def executor(settings, mock_bus, mock_client):
    from agents.executor import OrderExecutor

    exc = OrderExecutor(settings, mock_bus, mock_client)
    exc.current_bankroll = 1000.0
    return exc

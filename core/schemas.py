from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalStatus(StrEnum):
    PENDING_BUY = "PENDING_BUY"
    VALIDATED = "VALIDATED"
    VETOED = "VETOED"
    EXECUTED = "EXECUTED"


class ContextStatus(StrEnum):
    SAFE = "SAFE"
    VETO = "VETO"


class OrderState(StrEnum):
    PLACED = "PLACED"
    RESTING = "RESTING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"


class Side(StrEnum):
    YES = "yes"
    NO = "no"


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

class GameState(BaseModel):
    """Live game state pushed by the sports data watcher."""

    model_config = ConfigDict(strict=True, frozen=True)

    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    quarter: int
    clock: str
    timestamp: datetime


class MarketState(BaseModel):
    """Live Kalshi order book state pushed by the Kalshi WS watcher."""

    model_config = ConfigDict(strict=True, frozen=True)

    ticker: str
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    volume: int
    timestamp: datetime


class Signal(BaseModel):
    """Trade signal that flows through the Redis bus lifecycle."""

    model_config = ConfigDict(strict=True)

    ticker: str
    action: Action
    side: Side
    status: SignalStatus
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    ev_estimate: float
    entry_price: int
    exit_price: int
    game_id: str
    veto_reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Order(BaseModel):
    """Kalshi limit order payload. Market orders are forbidden by design."""

    model_config = ConfigDict(strict=True)

    ticker: str
    action: Action
    side: Side
    count: int = Field(ge=1)
    type: str = Field(default="limit", frozen=True)
    yes_price: Optional[int] = None
    no_price: Optional[int] = None
    client_order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ManagedOrder(BaseModel):
    """Executor-internal tracking wrapper around a placed order."""

    model_config = ConfigDict(strict=True)

    order: Order
    state: OrderState = OrderState.PLACED
    kalshi_order_id: Optional[str] = None
    fill_count: int = 0
    remaining_count: int = 0
    paired_exit_order_id: Optional[str] = None
    signal_id: str = ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    """Type-safe environment variable management via pydantic-settings."""

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Kalshi
    KALSHI_API_KEY_ID: str
    KALSHI_PRIVATE_KEY_PATH: str
    KALSHI_ENV: str = "demo"  # "demo" or "prod"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Sports data
    SPORTS_API_KEY: str = ""
    SPORTS_API_WS_URL: str = ""

    # LLM
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # Alerts
    TELEGRAM_CHAT_ID: str = ""

    # Risk management
    DAILY_STOP_LOSS_USD: float = 200.0
    SLIPPAGE_TICKS: int = 2
    KELLY_FRACTION: float = 0.5
    TARGET_EXIT_SPREAD: int = 7  # cents above entry

    # Polling intervals (seconds)
    CONTEXT_POLL_INTERVAL: float = 15.0
    BALANCE_POLL_INTERVAL: float = 10.0
    HEARTBEAT_INTERVAL: float = 30.0

    # Context cache TTL (seconds)
    CONTEXT_TTL: int = 300

    # EV threshold (cents) to trigger a trade signal
    EV_THRESHOLD: float = 3.0

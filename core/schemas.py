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
    REALLOCATE = "REALLOCATE"


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
    """Trade signal that flows through the Redis bus lifecycle.

    Strict mode is disabled because signals are deserialized from JSON
    strings via Redis Pub/Sub (enums arrive as str, timestamps as ISO str).
    """

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
    target_order_id: Optional[str] = None
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
    paired_exit_order_ids: list[str] = Field(default_factory=list)
    signal_id: str = ""
    is_exit: bool = False
    parent_entry_id: Optional[str] = None
    vwap_cents: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PortfolioPosition(BaseModel):
    """A resting exit order visible to the reallocation engine."""

    model_config = ConfigDict(strict=True, frozen=True)

    client_order_id: str
    ticker: str
    side: Side
    remaining_count: int
    entry_vwap: float
    target_exit_price: int
    kalshi_order_id: str


class PortfolioState(BaseModel):
    """Snapshot of the executor's active portfolio, published to Redis."""

    model_config = ConfigDict(strict=True)

    bankroll: float
    positions: list[PortfolioPosition] = Field(default_factory=list)


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

    # LLM — provider selection: "openai", "anthropic", or "gemini"
    LLM_PROVIDER: str = "gemini"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Alerts
    TELEGRAM_CHAT_ID: str = ""

    # Risk management
    DAILY_STOP_LOSS_USD: float = 100.0
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

    # Order GC: how often to sweep (seconds) and max age of terminal orders (seconds)
    ORDER_GC_INTERVAL: float = 300.0
    ORDER_GC_TTL: float = 7200.0

    # Capital rebalancing
    MIN_REALLOCATE_BID: int = 90
    TAKER_FEE_CENTS: float = 2.0

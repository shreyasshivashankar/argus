"""TrackAgent — persistent Postgres trade tracker.

Subscribes to ``signal:validated`` and ``signal:executed`` on the Redis bus
and writes every event to the ``trades`` table in PostgreSQL.  Runs as an
optional agent enabled via the ``--track`` CLI flag.

Key design decisions:
    - Uses asyncpg for high-performance async Postgres I/O.
    - Every row carries ``is_paper`` so paper and live data never pollute
      each other in the same database.
    - On ``signal:executed``, the tracker UPSERTs: it updates the matching
      ``signal_id`` row if one exists (from the earlier VALIDATED insert),
      or inserts a new row if the tracker started after validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg
from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings

_CREATE_SQL = """\
CREATE TABLE IF NOT EXISTS trades (
    id          SERIAL PRIMARY KEY,
    signal_id   TEXT    UNIQUE,
    ticker      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    confidence  DOUBLE PRECISION,
    ev_estimate DOUBLE PRECISION,
    entry_price INTEGER,
    exit_price  INTEGER,
    game_id     TEXT,
    source      TEXT,
    pnl_dollars DOUBLE PRECISION DEFAULT 0.0,
    is_paper    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL
);
"""

_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_trades_is_paper ON trades(is_paper);",
    "CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);",
]


class TrackAgent(BaseAgent):
    """Persists trade lifecycle events to PostgreSQL."""

    def __init__(
        self,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
        *,
        paper_mode: bool = False,
    ) -> None:
        super().__init__("track", settings, bus, client)
        self._paper_mode = paper_mode
        self._pool: asyncpg.Pool | None = None

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def _init_db(self) -> asyncpg.Pool:
        pool = await asyncpg.create_pool(
            self.settings.DATABASE_URL,
            min_size=1,
            max_size=3,
        )
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_SQL)
            for idx_sql in _INDEXES_SQL:
                await conn.execute(idx_sql)
        self.log.info(
            "Trade DB initialized (Postgres, is_paper={})", self._paper_mode
        )
        return pool

    async def _close_db(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._pool = await self._init_db()
        try:
            await self.bus.subscribe(
                ["signal:validated", "signal:executed"], self._on_message
            )
        finally:
            await self._close_db()

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, channel: str, data: dict[str, Any]) -> None:
        if channel == "signal:validated":
            await self._on_validated(data)
        elif channel == "signal:executed":
            await self._on_executed(data)

    async def _on_validated(self, data: dict[str, Any]) -> None:
        assert self._pool is not None
        now = datetime.utcnow()
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO trades
                       (signal_id, ticker, side, status, confidence, ev_estimate,
                        entry_price, exit_price, game_id, source, pnl_dollars,
                        is_paper, created_at)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,0.0,$11,$12)
                       ON CONFLICT (signal_id) DO NOTHING""",
                    data.get("signal_id", ""),
                    data.get("ticker", ""),
                    data.get("side", ""),
                    "VALIDATED",
                    data.get("confidence"),
                    data.get("ev_estimate"),
                    data.get("entry_price"),
                    data.get("exit_price"),
                    data.get("game_id", ""),
                    data.get("source", ""),
                    self._paper_mode,
                    now,
                )
        except Exception:
            self.log.exception("Failed to insert VALIDATED trade")

    async def _on_executed(self, data: dict[str, Any]) -> None:
        """UPSERT: update existing VALIDATED row or insert fresh if missing."""
        assert self._pool is not None
        signal_id = data.get("signal_id", "")
        entry_price = int(data.get("entry_price", 0))
        exit_price = int(data.get("exit_price", 0))
        pnl = (exit_price - entry_price) / 100.0
        now = datetime.utcnow()

        try:
            async with self._pool.acquire() as conn:
                updated = await conn.execute(
                    """UPDATE trades
                       SET status = 'EXECUTED',
                           pnl_dollars = pnl_dollars + $1,
                           entry_price = $2,
                           exit_price = $3
                       WHERE signal_id = $4""",
                    pnl,
                    entry_price,
                    exit_price,
                    signal_id,
                )
                if updated == "UPDATE 0":
                    await conn.execute(
                        """INSERT INTO trades
                           (signal_id, ticker, side, status, confidence, ev_estimate,
                            entry_price, exit_price, game_id, source, pnl_dollars,
                            is_paper, created_at)
                           VALUES ($1,$2,$3,'EXECUTED',$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
                        signal_id,
                        data.get("ticker", ""),
                        data.get("side", ""),
                        data.get("confidence"),
                        data.get("ev_estimate"),
                        entry_price,
                        exit_price,
                        data.get("game_id", ""),
                        data.get("source", ""),
                        pnl,
                        self._paper_mode,
                        now,
                    )
        except Exception:
            self.log.exception("Failed to upsert EXECUTED trade")

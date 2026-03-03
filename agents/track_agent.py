"""TrackAgent — persistent SQLite trade tracker.

Subscribes to ``signal:validated`` and ``signal:executed`` on the Redis bus
and writes every event to ``data/trades.db``.  Runs as an optional agent
enabled via the ``--track`` CLI flag.

Key design decisions:
    - WAL journal mode eliminates writer-reader locking during trade flurries.
    - Every row carries ``is_paper`` so paper and live data never pollute each
      other in the same database file.
    - On ``signal:executed``, the tracker UPSERTs: it updates the matching
      ``signal_id`` row if one exists (from the earlier VALIDATED insert),
      or inserts a new row if the tracker started after validation.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import aiosqlite
from loguru import logger

from core.base_agent import BaseAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings

_DB_DIR = "data"
_DB_PATH = os.path.join(_DB_DIR, "trades.db")

_CREATE_SQL = """\
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id   TEXT    UNIQUE,
    ticker      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    confidence  REAL,
    ev_estimate REAL,
    entry_price INTEGER,
    exit_price  INTEGER,
    game_id     TEXT,
    source      TEXT,
    pnl_dollars REAL    DEFAULT 0.0,
    is_paper    BOOLEAN NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL
);
"""

_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_trades_is_paper ON trades(is_paper);",
]


class TrackAgent(BaseAgent):
    """Persists trade lifecycle events to a local SQLite database."""

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
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def _init_db(self) -> aiosqlite.Connection:
        os.makedirs(_DB_DIR, exist_ok=True)
        db = await aiosqlite.connect(_DB_PATH)
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute(_CREATE_SQL)
        for idx_sql in _INDEXES_SQL:
            await db.execute(idx_sql)
        await db.commit()
        self.log.info(
            "Trade DB initialized at {} (WAL mode, is_paper={})",
            _DB_PATH,
            self._paper_mode,
        )
        return db

    async def _close_db(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._db = await self._init_db()
        try:
            await self.bus.subscribe(
                ["signal:validated", "signal:executed", "signal:reallocate"],
                self._on_message,
            )
        finally:
            await self._close_db()

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _on_message(self, channel: str, data: dict[str, Any]) -> None:
        if channel in ("signal:validated", "signal:reallocate"):
            await self._on_validated(data)
        elif channel == "signal:executed":
            await self._on_executed(data)

    async def _on_validated(self, data: dict[str, Any]) -> None:
        assert self._db is not None
        now = datetime.utcnow().isoformat()
        try:
            await self._db.execute(
                """INSERT OR IGNORE INTO trades
                   (signal_id, ticker, side, status, confidence, ev_estimate,
                    entry_price, exit_price, game_id, source, pnl_dollars,
                    is_paper, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?)""",
                (
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
                    int(self._paper_mode),
                    now,
                ),
            )
            await self._db.commit()
        except Exception:
            self.log.exception("Failed to insert VALIDATED trade")

    async def _on_executed(self, data: dict[str, Any]) -> None:
        """UPSERT: update existing VALIDATED row or insert fresh if missing."""
        assert self._db is not None
        signal_id = data.get("signal_id", "")
        pnl = float(data.get("ev_estimate", 0.0))
        entry_price = data.get("entry_price", 0)
        exit_price = data.get("exit_price", 0)
        now = datetime.utcnow().isoformat()

        try:
            cursor = await self._db.execute(
                "SELECT id FROM trades WHERE signal_id = ?", (signal_id,)
            )
            row = await cursor.fetchone()

            if row:
                await self._db.execute(
                    """UPDATE trades
                       SET status = 'EXECUTED',
                           pnl_dollars = pnl_dollars + ?,
                           entry_price = ?,
                           exit_price = ?
                       WHERE signal_id = ?""",
                    (pnl, entry_price, exit_price, signal_id),
                )
            else:
                await self._db.execute(
                    """INSERT INTO trades
                       (signal_id, ticker, side, status, confidence, ev_estimate,
                        entry_price, exit_price, game_id, source, pnl_dollars,
                        is_paper, created_at)
                       VALUES (?, ?, ?, 'EXECUTED', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
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
                        int(self._paper_mode),
                        now,
                    ),
                )
            await self._db.commit()
        except Exception:
            self.log.exception("Failed to upsert EXECUTED trade")

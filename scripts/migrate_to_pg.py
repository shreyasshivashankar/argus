"""One-shot migration: SQLite trades.db → Postgres.

Reads all rows from the local SQLite file, connects to Postgres,
creates the table if needed, and inserts every row (skipping duplicates
by signal_id).

Usage:
    docker-compose up -d postgres          # ensure PG is running
    docker-compose run --rm --entrypoint "" argus python -m scripts.migrate_to_pg
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime

import psycopg2

_SQLITE_PATH = os.path.join("data", "trades.db")

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
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_is_paper ON trades(is_paper);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
"""

_INSERT_SQL = """\
INSERT INTO trades
    (signal_id, ticker, side, status, confidence, ev_estimate,
     entry_price, exit_price, game_id, source, pnl_dollars,
     is_paper, created_at)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (signal_id) DO NOTHING
"""


def main() -> None:
    if not os.path.exists(_SQLITE_PATH):
        print(f"No SQLite database at {_SQLITE_PATH} — nothing to migrate.")
        sys.exit(0)

    dsn = os.environ.get(
        "DATABASE_URL", "postgresql://argus:argus@localhost:5432/argus"
    )

    lite = sqlite3.connect(_SQLITE_PATH)
    lite.row_factory = sqlite3.Row
    rows = lite.execute("SELECT * FROM trades ORDER BY id").fetchall()
    lite.close()
    print(f"Read {len(rows)} rows from SQLite")

    if not rows:
        print("Nothing to migrate.")
        return

    pg = psycopg2.connect(dsn)
    cur = pg.cursor()
    cur.execute(_CREATE_SQL)
    pg.commit()

    migrated = 0
    skipped = 0
    for row in rows:
        created_at = row["created_at"]
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.utcnow()

        is_paper = bool(row["is_paper"])

        cur.execute(
            _INSERT_SQL,
            (
                row["signal_id"],
                row["ticker"],
                row["side"],
                row["status"],
                row["confidence"],
                row["ev_estimate"],
                row["entry_price"],
                row["exit_price"],
                row["game_id"],
                row["source"],
                row["pnl_dollars"],
                is_paper,
                created_at,
            ),
        )
        if cur.rowcount > 0:
            migrated += 1
        else:
            skipped += 1

    pg.commit()
    cur.close()
    pg.close()

    print(f"Migration complete: {migrated} inserted, {skipped} skipped (duplicates)")


if __name__ == "__main__":
    main()

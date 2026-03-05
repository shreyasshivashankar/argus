"""Shared utilities for the Argus trading system."""
from __future__ import annotations

from core.schemas import GameState

MINUTES_PER_QUARTER = 12.0
MINUTES_PER_GAME = 48.0


def team_minutes_played(game: GameState) -> float:
    """Estimate total team-level minutes elapsed from quarter and clock.

    Clock formats observed from providers:
      - ``"5:30"``      — standard MM:SS
      - ``":08.7"``     — sub-minute with decimal seconds
      - ``"Q3 5:30"``   — quarter-prefixed
      - ``"Half"``      — halftime
      - ``"END"``       — end of quarter/game
    """
    q = max(game.quarter, 1)
    completed = (q - 1) * MINUTES_PER_QUARTER

    clock = game.clock.strip()
    if not clock or clock.upper() in ("HALF", "END"):
        return completed

    parts = clock.split()
    raw = parts[-1] if parts else clock

    try:
        if ":" in raw:
            mins_str, secs_str = raw.split(":", 1)
            mins = int(mins_str) if mins_str else 0
            secs = float(secs_str)
            remaining = mins + secs / 60.0
        else:
            remaining = float(raw) / 60.0
        return completed + (MINUTES_PER_QUARTER - remaining)
    except (ValueError, TypeError):
        return completed

"""Tests for BallDontLieFeed."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.schemas import AppSettings, GameState, PlayerBoxScore
from watchers.sports_feed import BallDontLieFeed


@pytest.fixture
def mock_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def bdl_settings() -> AppSettings:
    return AppSettings(
        KALSHI_API_KEY_ID="test",
        KALSHI_PRIVATE_KEY_PATH="/dev/null",
        KALSHI_ENV="demo",
        REDIS_URL="redis://localhost:6379",
        BALLDONTLIE_API_KEY="test-bdl-key",
        SPORTS_GAMES_POLL_INTERVAL=0.5,
        SPORTS_POLL_INTERVAL=0.5,
        OPENAI_API_KEY="",
        TELEGRAM_CHAT_ID="",
    )


@pytest.fixture
def bdl_feed(bdl_settings, mock_bus) -> BallDontLieFeed:
    return BallDontLieFeed(bdl_settings, mock_bus)


class TestBallDontLieFeed:
    def test_is_live(self, bdl_feed):
        assert bdl_feed._is_live("1st Qtr") is True
        assert bdl_feed._is_live("4th Qtr") is True
        assert bdl_feed._is_live("Final") is False
        assert bdl_feed._is_live("7:00 pm ET") is False

    def test_game_to_state(self, bdl_feed):
        game = {
            "id": 12345,
            "status": "3rd Qtr",
            "period": 3,
            "time": "5:30",
            "home_team_score": 85,
            "visitor_team_score": 78,
            "home_team": {"full_name": "Los Angeles Lakers", "abbreviation": "LAL"},
            "visitor_team": {"full_name": "Boston Celtics", "abbreviation": "BOS"},
        }
        gs = bdl_feed._game_to_state(game, "12345")
        assert gs is not None
        assert gs.game_id == "12345"
        assert gs.home_team == "Los Angeles Lakers"
        assert gs.away_team == "Boston Celtics"
        assert gs.home_score == 85
        assert gs.away_score == 78
        assert gs.quarter == 3
        assert gs.clock == "5:30"

    def test_game_to_state_missing_teams_returns_none(self, bdl_feed):
        game = {"id": 1, "home_team": {}, "visitor_team": {}}
        assert bdl_feed._game_to_state(game, "1") is None

    def test_parse_stat_entry(self, bdl_feed):
        entry = {
            "player": {"id": 70, "first_name": "Jaylen", "last_name": "Brown"},
            "team": {"abbreviation": "BOS"},
            "min": "28",
            "pts": 22,
            "ast": 4,
            "reb": 6,
        }
        pbs = bdl_feed._parse_stat_entry(entry)
        assert pbs is not None
        assert pbs.player_id == "70"
        assert pbs.first_name == "Jaylen"
        assert pbs.team_abbr == "BOS"
        assert pbs.pts == 22

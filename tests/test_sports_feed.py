"""Tests for TheRundownFeed — WebSocket event parsing, REST player stats, and rate limiting."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.schemas import AppSettings, GameState, PlayerBoxScore
from watchers.sports_feed import TheRundownFeed, _LIVE_STATUSES, _NBA_STAT_ABBR_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        KALSHI_API_KEY_ID="test",
        KALSHI_PRIVATE_KEY_PATH="/dev/null",
        KALSHI_ENV="demo",
        REDIS_URL="redis://localhost:6379",
        THERUNDOWN_API_KEY="test-tr-key",
        SPORTS_POLL_INTERVAL=5.0,
        OPENAI_API_KEY="",
        TELEGRAM_CHAT_ID="",
    )


@pytest.fixture
def mock_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def feed(settings, mock_bus) -> TheRundownFeed:
    f = TheRundownFeed(settings, mock_bus)
    f._team_id_to_abbr = {11: "ATL", 12: "CHA", 30: "GSW", 31: "LAL"}
    return f


# ---------------------------------------------------------------------------
# V1 WebSocket event parsing
# ---------------------------------------------------------------------------

def _make_ws_event(
    *,
    event_id: str = "abc123",
    status: str = "STATUS_IN_PROGRESS",
    score_home: int = 55,
    score_away: int = 50,
    game_period: int = 3,
    display_clock: str = "5:30",
) -> dict:
    return {
        "event_id": event_id,
        "score": {
            "event_status": status,
            "score_home": score_home,
            "score_away": score_away,
            "game_period": game_period,
            "display_clock": display_clock,
        },
        "teams_normalized": [
            {
                "name": "Atlanta",
                "mascot": "Hawks",
                "abbreviation": "ATL",
                "is_away": True,
                "is_home": False,
            },
            {
                "name": "Charlotte",
                "mascot": "Hornets",
                "abbreviation": "CHA",
                "is_away": False,
                "is_home": True,
            },
        ],
    }


class TestParseWSEvent:
    def test_live_event_produces_game_state(self, feed):
        payload = _make_ws_event()
        gs = feed._parse_ws_event(payload)

        assert gs is not None
        assert gs.game_id == "abc123"
        assert gs.home_score == 55
        assert gs.away_score == 50
        assert gs.quarter == 3
        assert gs.clock == "5:30"
        assert gs.home_abbr == "CHA"
        assert gs.away_abbr == "ATL"
        assert gs.home_team == "Charlotte Hornets"
        assert gs.away_team == "Atlanta Hawks"

    def test_scheduled_event_returns_none(self, feed):
        payload = _make_ws_event(status="STATUS_SCHEDULED")
        gs = feed._parse_ws_event(payload)
        assert gs is None

    def test_final_event_returns_none(self, feed):
        payload = _make_ws_event(status="STATUS_FINAL")
        gs = feed._parse_ws_event(payload)
        assert gs is None

    def test_halftime_event_produces_game_state(self, feed):
        payload = _make_ws_event(status="STATUS_HALFTIME")
        gs = feed._parse_ws_event(payload)
        assert gs is not None

    def test_overtime_event_produces_game_state(self, feed):
        payload = _make_ws_event(status="STATUS_OVERTIME")
        gs = feed._parse_ws_event(payload)
        assert gs is not None

    def test_event_cached_on_parse(self, feed):
        payload = _make_ws_event(event_id="ev-999")
        feed._parse_ws_event(payload)
        assert "ev-999" in feed._event_cache

    def test_no_event_id_returns_none(self, feed):
        gs = feed._parse_ws_event({"score": {}})
        assert gs is None

    def test_nested_event_key(self, feed):
        """V1 WS sometimes wraps the event under an 'event' key."""
        inner = _make_ws_event(event_id="nested-1")
        payload = {"event": inner}
        gs = feed._parse_ws_event(payload)
        assert gs is not None
        assert gs.game_id == "nested-1"

    def test_missing_display_clock_falls_back_to_quarter(self, feed):
        payload = _make_ws_event(display_clock="", game_period=2)
        gs = feed._parse_ws_event(payload)
        assert gs is not None
        assert gs.clock == "Q2"

    def test_heartbeat_detection(self, feed):
        assert feed._parse_ws_event({"heartbeat": True}) is None
        assert feed._parse_ws_event({"meta": {"type": "heartbeat"}}) is None

    def test_player_stats_merged_from_cache(self, feed):
        pbs = PlayerBoxScore(
            player_id="100", first_name="Trae", last_name="Young",
            team_abbr="ATL", pts=28, ast=11,
        )
        feed._player_stats_cache["ev-with-stats"] = [pbs]
        payload = _make_ws_event(event_id="ev-with-stats")
        gs = feed._parse_ws_event(payload)
        assert gs is not None
        assert len(gs.player_stats) == 1
        assert gs.player_stats[0].pts == 28


# ---------------------------------------------------------------------------
# Event -> GameState conversion
# ---------------------------------------------------------------------------

class TestEventToGameState:
    def test_basic_conversion(self, feed):
        event = _make_ws_event()
        gs = feed._event_to_game_state(event, "abc123")
        assert gs is not None
        assert isinstance(gs, GameState)
        assert gs.timestamp.tzinfo is not None

    def test_zero_scores(self, feed):
        event = _make_ws_event(score_home=0, score_away=0)
        gs = feed._event_to_game_state(event, "abc123")
        assert gs is not None
        assert gs.home_score == 0
        assert gs.away_score == 0

    def test_empty_teams_produces_empty_names(self, feed):
        event = {
            "event_id": "bad",
            "score": {"event_status": "STATUS_IN_PROGRESS"},
            "teams_normalized": [],
        }
        gs = feed._event_to_game_state(event, "bad")
        assert gs is not None
        assert gs.home_abbr == ""
        assert gs.away_abbr == ""


# ---------------------------------------------------------------------------
# Player stat entry parsing (V2 REST)
# ---------------------------------------------------------------------------

def _make_player_stat_entry(
    *,
    player_id: int = 16627,
    team_id: int = 11,
    first_name: str = "Trae",
    last_name: str = "Young",
    stats: list[tuple[str, str]] | None = None,
) -> dict:
    if stats is None:
        stats = [("PTS", "28"), ("AST", "11"), ("REB", "4"), ("FGM", "10"), ("FGA", "22")]
    return {
        "player": {
            "id": player_id,
            "team_id": team_id,
            "first_name": first_name,
            "last_name": last_name,
        },
        "stats": [
            {
                "stat": {"abbreviation": abbr},
                "value": val,
            }
            for abbr, val in stats
        ],
    }


class TestParsePlayerStatEntry:
    def test_basic_parse(self, feed):
        entry = _make_player_stat_entry()
        pbs = feed._parse_player_stat_entry(entry, "ev-1")

        assert pbs is not None
        assert pbs.player_id == "16627"
        assert pbs.first_name == "Trae"
        assert pbs.last_name == "Young"
        assert pbs.team_abbr == "ATL"
        assert pbs.pts == 28
        assert pbs.ast == 11
        assert pbs.reb == 4
        assert pbs.fgm == 10
        assert pbs.fga == 22

    def test_unknown_team_id_uses_str(self, feed):
        entry = _make_player_stat_entry(team_id=99999)
        pbs = feed._parse_player_stat_entry(entry, "ev-1")
        assert pbs is not None
        assert pbs.team_abbr == "99999"

    def test_known_team_id_resolves(self, feed):
        entry = _make_player_stat_entry(team_id=30)
        pbs = feed._parse_player_stat_entry(entry, "ev-1")
        assert pbs is not None
        assert pbs.team_abbr == "GSW"

    def test_empty_stats_list(self, feed):
        entry = _make_player_stat_entry(stats=[])
        pbs = feed._parse_player_stat_entry(entry, "ev-1")
        assert pbs is not None
        assert pbs.pts == 0
        assert pbs.ast == 0

    def test_non_numeric_stat_value_ignored(self, feed):
        entry = _make_player_stat_entry(stats=[("PTS", "N/A")])
        pbs = feed._parse_player_stat_entry(entry, "ev-1")
        assert pbs is not None
        assert pbs.pts == 0

    def test_turnover_aliases(self, feed):
        for abbr in ("TO", "TOV"):
            entry = _make_player_stat_entry(stats=[(abbr, "5")])
            pbs = feed._parse_player_stat_entry(entry, "ev-1")
            assert pbs is not None
            assert pbs.turnover == 5

    def test_three_pointers(self, feed):
        entry = _make_player_stat_entry(stats=[("3PM", "4"), ("3PA", "9")])
        pbs = feed._parse_player_stat_entry(entry, "ev-1")
        assert pbs is not None
        assert pbs.fg3m == 4
        assert pbs.fg3a == 9


# ---------------------------------------------------------------------------
# Rate limit tracking
# ---------------------------------------------------------------------------

class TestRateLimitTracking:
    def test_update_rate_headers(self, feed):
        mock_resp = MagicMock()
        mock_resp.headers = {
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Limit": "300",
        }
        feed._update_rate_headers(mock_resp)
        assert feed._rate_remaining == 42
        assert feed._rate_limit == 300

    def test_missing_headers_no_crash(self, feed):
        mock_resp = MagicMock()
        mock_resp.headers = {}
        feed._update_rate_headers(mock_resp)
        assert feed._rate_remaining is None
        assert feed._rate_limit is None

    def test_invalid_header_values_no_crash(self, feed):
        mock_resp = MagicMock()
        mock_resp.headers = {
            "X-RateLimit-Remaining": "not-a-number",
        }
        feed._update_rate_headers(mock_resp)
        assert feed._rate_remaining is None


# ---------------------------------------------------------------------------
# Feed construction
# ---------------------------------------------------------------------------

class TestFeedConstruction:
    def test_api_key_from_settings(self, settings, mock_bus):
        f = TheRundownFeed(settings, mock_bus)
        assert f._api_key == "test-tr-key"

    def test_poll_interval_minimum_enforced(self, mock_bus):
        s = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            THERUNDOWN_API_KEY="key",
            SPORTS_POLL_INTERVAL=1.0,
            OPENAI_API_KEY="",
            TELEGRAM_CHAT_ID="",
        )
        f = TheRundownFeed(s, mock_bus)
        assert f._stats_poll_interval >= TheRundownFeed.STATS_POLL_INTERVAL_MIN

    def test_stop_sets_running_false(self, feed):
        assert feed._running is True
        feed.stop()
        assert feed._running is False


# ---------------------------------------------------------------------------
# Live status set coverage
# ---------------------------------------------------------------------------

class TestLiveStatuses:
    @pytest.mark.parametrize("status", list(_LIVE_STATUSES))
    def test_live_status_produces_game_state(self, feed, status):
        payload = _make_ws_event(status=status)
        gs = feed._parse_ws_event(payload)
        assert gs is not None

    @pytest.mark.parametrize("status", [
        "STATUS_SCHEDULED", "STATUS_FINAL", "STATUS_POSTPONED",
        "STATUS_CANCELED", "STATUS_DELAYED",
    ])
    def test_non_live_status_returns_none(self, feed, status):
        payload = _make_ws_event(status=status)
        gs = feed._parse_ws_event(payload)
        assert gs is None


# ---------------------------------------------------------------------------
# Stat abbreviation map coverage
# ---------------------------------------------------------------------------

class TestStatAbbrMap:
    def test_all_core_stats_mapped(self):
        core = {"PTS", "AST", "REB", "STL", "BLK", "PF", "FGM", "FGA", "FTM", "FTA", "3PM", "3PA"}
        for abbr in core:
            assert abbr in _NBA_STAT_ABBR_MAP, f"{abbr} not in stat map"

    def test_turnover_has_two_aliases(self):
        assert _NBA_STAT_ABBR_MAP["TO"] == "turnover"
        assert _NBA_STAT_ABBR_MAP["TOV"] == "turnover"

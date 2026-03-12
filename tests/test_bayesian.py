"""Tests for the Bayesian projection model and data sources."""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from core.schemas import GameState, PlayerBoxScore
from sports.nba.bayesian import (
    over_probability,
    prior_weight,
    project_game_total,
    project_player_stat,
    project_spread,
    project_team_total,
    PRIOR_HALF_LIFE_MINUTES,
)
from sports.nba.season_averages import PlayerPrior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game(
    *,
    home_score: int = 55,
    away_score: int = 50,
    quarter: int = 3,
    clock: str = "6:00",
    player_stats: list[PlayerBoxScore] | None = None,
) -> GameState:
    return GameState(
        game_id="g1",
        home_team="Hawks",
        away_team="Mavs",
        home_abbr="ATL",
        away_abbr="DAL",
        home_score=home_score,
        away_score=away_score,
        quarter=quarter,
        clock=clock,
        timestamp=datetime.utcnow(),
        player_stats=player_stats or [],
    )


def _player(*, pts: int = 10, minutes: float = 12.0, fga: int = 8) -> PlayerBoxScore:
    return PlayerBoxScore(
        player_id="p1", first_name="Keyonte", last_name="George",
        team_abbr="DAL", minutes=minutes, pts=pts, fgm=4, fga=fga,
        fg3m=2, fg3a=4, ftm=0, fta=0, reb=3, ast=2, stl=1, blk=0,
        turnover=1, pf=1, plus_minus=0,
    )


def _prior(*, pts_per_min: float = 0.5, avg_minutes: float = 32.0) -> PlayerPrior:
    """Season avg: ~16 pts/game (0.5 pts/min * 32 min)."""
    return PlayerPrior(
        player_id="p1", first_name="Keyonte", last_name="George",
        games_played=50, avg_minutes=avg_minutes,
        pts_per_min=pts_per_min,
        reb_per_min=0.15, ast_per_min=0.10, stl_per_min=0.03,
        blk_per_min=0.01, fg3m_per_min=0.08, fga_per_min=0.35,
    )


# ===================================================================
# prior_weight
# ===================================================================

class TestPriorWeight:

    def test_at_zero_minutes(self):
        """Before MIN_LIVE_MINUTES, weight is 1.0 (full prior)."""
        assert prior_weight(0.0) == 1.0
        assert prior_weight(1.0) == 1.0

    def test_at_half_life(self):
        w = prior_weight(PRIOR_HALF_LIFE_MINUTES)
        assert abs(w - 0.5) < 0.01

    def test_at_double_half_life(self):
        w = prior_weight(2 * PRIOR_HALF_LIFE_MINUTES)
        assert abs(w - 0.25) < 0.01

    def test_at_full_game(self):
        w = prior_weight(48.0)
        assert w < 0.15  # mostly live data

    def test_monotonically_decreasing(self):
        weights = [prior_weight(m) for m in range(3, 48)]
        for i in range(1, len(weights)):
            assert weights[i] <= weights[i - 1]


# ===================================================================
# project_player_stat
# ===================================================================

class TestProjectPlayerStat:

    def test_without_prior_uses_live_pace(self):
        """No prior → pure linear extrapolation (backward compat)."""
        player = _player(pts=10, minutes=12.0)
        game = _game(quarter=2, clock="6:00")

        projected = project_player_stat(player, game, "pts", player_prior=None)
        assert projected is not None
        # 10 pts in 12 min → ~0.83 pts/min, ~36 min remaining → ~30 more
        assert projected > 25

    def test_with_prior_anchors_projection(self):
        """Season avg of 16 pts should pull down a hot-start projection."""
        player = _player(pts=10, minutes=12.0)
        game = _game(quarter=2, clock="6:00")
        prior = _prior(pts_per_min=0.5)  # season avg ~16 pts

        proj_no_prior = project_player_stat(player, game, "pts", player_prior=None)
        proj_with_prior = project_player_stat(player, game, "pts", player_prior=prior)

        assert proj_no_prior is not None
        assert proj_with_prior is not None
        # With prior, projection should be LOWER (anchored to 16 pts avg)
        assert proj_with_prior < proj_no_prior

    def test_prior_dominates_early(self):
        """With only 3 min played, prior should dominate."""
        player = _player(pts=5, minutes=3.0)
        game = _game(quarter=1, clock="9:00")
        prior = _prior(pts_per_min=0.5)  # ~16 pts season

        projected = project_player_stat(player, game, "pts", player_prior=prior)
        assert projected is not None
        # Live pace says 80 pts (5/3*48), but prior says 16
        # At 3 min, prior weight ≈ 1.0, so should be close to 16
        assert projected < 25

    def test_live_dominates_late(self):
        """With 36 min played, live data should dominate."""
        player = _player(pts=25, minutes=36.0)
        game = _game(quarter=4, clock="6:00")
        prior = _prior(pts_per_min=0.5)  # ~16 pts season

        projected = project_player_stat(player, game, "pts", player_prior=prior)
        assert projected is not None
        # Live says ~28 pts, prior says 16. At 36 min, live dominates.
        assert projected > 22

    def test_zero_minutes_returns_none(self):
        player = _player(pts=0, minutes=0.0)
        game = _game(quarter=1, clock="12:00")
        assert project_player_stat(player, game, "pts") is None

    def test_rebounds_projection(self):
        player = _player()
        player = PlayerBoxScore(
            **{**player.__dict__, "reb": 5, "minutes": 15.0}
        )
        game = _game(quarter=2, clock="3:00")
        prior = _prior()

        projected = project_player_stat(player, game, "reb", player_prior=prior)
        assert projected is not None
        assert projected > 0


# ===================================================================
# project_game_total
# ===================================================================

class TestProjectGameTotal:

    def test_without_sharp_line(self):
        """No sharp line → uses league avg 225 as prior."""
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        projected = project_game_total(game)
        assert projected is not None
        # 105 total at ~30 min → live pace ~168, prior ~225
        # Blended should be somewhere between
        assert 168 < projected < 225

    def test_with_sharp_line(self):
        """Sharp line provides better prior."""
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        proj_default = project_game_total(game, sharp_line=None)
        proj_sharp = project_game_total(game, sharp_line=210.0)
        assert proj_default is not None
        assert proj_sharp is not None
        # Sharp line of 210 is lower than default 225, so projection should differ
        assert proj_sharp != proj_default

    def test_early_game_prior_dominates(self):
        """Early in game, prior should dominate."""
        game = _game(home_score=10, away_score=8, quarter=1, clock="9:00")
        projected = project_game_total(game, sharp_line=220.0)
        assert projected is not None
        # 18 total at 3 min → live pace 288, sharp prior 220
        # Prior should pull it way down
        assert projected < 260

    def test_late_game_live_dominates(self):
        game = _game(home_score=95, away_score=90, quarter=4, clock="6:00")
        projected = project_game_total(game, sharp_line=220.0)
        assert projected is not None
        # 185 at 42 min → live pace ~211, prior 220
        # Should be close to live
        assert 200 < projected < 225


# ===================================================================
# project_team_total & project_spread
# ===================================================================

class TestProjectTeamTotal:

    def test_basic(self):
        game = _game(home_score=60, away_score=50, quarter=3, clock="6:00")
        projected = project_team_total(60, game)
        assert projected is not None
        assert projected > 0

    def test_sharp_prior(self):
        game = _game(home_score=60, away_score=50, quarter=3, clock="6:00")
        proj_default = project_team_total(60, game)
        proj_sharp = project_team_total(60, game, sharp_line=105.0)
        assert proj_default is not None
        assert proj_sharp is not None


class TestProjectSpread:

    def test_positive_margin(self):
        game = _game(home_score=70, away_score=55, quarter=3, clock="6:00")
        projected = project_spread(15, game)
        assert projected is not None
        assert projected > 0

    def test_sharp_prior(self):
        game = _game(home_score=70, away_score=55, quarter=3, clock="6:00")
        proj = project_spread(15, game, sharp_line=6.0)
        assert proj is not None


# ===================================================================
# over_probability
# ===================================================================

class TestOverProbability:

    def test_projected_above_line(self):
        prob = over_probability(230.0, 220.0, 15.0, 30.0)
        assert prob > 0.5

    def test_projected_below_line(self):
        prob = over_probability(210.0, 220.0, 15.0, 30.0)
        assert prob < 0.5

    def test_at_line(self):
        prob = over_probability(220.0, 220.0, 15.0, 30.0)
        assert abs(prob - 0.5) < 0.05

    def test_certainty_late_game(self):
        """Late game with big gap → near certainty."""
        prob = over_probability(240.0, 220.0, 15.0, 46.0)
        assert prob > 0.95


# ===================================================================
# TheRundown watcher
# ===================================================================

class TestTheRundownWatcher:

    def test_american_to_prob_favorite(self):
        from watchers.therundown_feed import TheRundownWatcher
        prob = TheRundownWatcher._american_to_prob(-110)
        assert abs(prob - 0.524) < 0.01

    def test_american_to_prob_underdog(self):
        from watchers.therundown_feed import TheRundownWatcher
        prob = TheRundownWatcher._american_to_prob(150)
        assert abs(prob - 0.40) < 0.01

    def test_american_to_prob_even(self):
        from watchers.therundown_feed import TheRundownWatcher
        prob = TheRundownWatcher._american_to_prob(100)
        assert abs(prob - 0.50) < 0.01

    def test_disabled_without_key(self):
        """Watcher should exit gracefully with no API key."""
        from unittest.mock import MagicMock
        from core.schemas import AppSettings
        settings = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            DATABASE_URL="postgresql://argus:argus@localhost:5432/argus",
            OPENAI_API_KEY="test",
            THERUNDOWN_API_KEY="",
        )
        from watchers.therundown_feed import TheRundownWatcher
        watcher = TheRundownWatcher(settings, MagicMock())
        assert watcher._api_key == ""


# ===================================================================
# SeasonAverageCache
# ===================================================================

class TestSeasonAverageCache:

    def test_parse_entry_valid(self):
        from sports.nba.season_averages import SeasonAverageCache
        entry = {
            "player_id": 123,
            "games_played": 50,
            "min": "32.5",
            "pts": 18.5,
            "reb": 5.0,
            "ast": 3.0,
            "stl": 1.0,
            "blk": 0.5,
            "fg3m": 2.0,
            "fga": 15.0,
        }
        prior = SeasonAverageCache._parse_entry(entry)
        assert prior is not None
        assert prior.player_id == "123"
        assert prior.games_played == 50
        assert abs(prior.pts_per_min - 18.5 / 32.5) < 0.01

    def test_parse_entry_too_few_games(self):
        from sports.nba.season_averages import SeasonAverageCache
        entry = {
            "player_id": 123,
            "games_played": 3,
            "min": "10.0",
            "pts": 5.0, "reb": 1.0, "ast": 0.5,
            "stl": 0.0, "blk": 0.0, "fg3m": 0.0, "fga": 3.0,
        }
        assert SeasonAverageCache._parse_entry(entry) is None

    def test_parse_entry_low_minutes(self):
        from sports.nba.season_averages import SeasonAverageCache
        entry = {
            "player_id": 123,
            "games_played": 50,
            "min": "3.0",
            "pts": 2.0, "reb": 0.5, "ast": 0.2,
            "stl": 0.0, "blk": 0.0, "fg3m": 0.0, "fga": 1.0,
        }
        assert SeasonAverageCache._parse_entry(entry) is None

    def test_get_returns_none_when_empty(self):
        from sports.nba.season_averages import SeasonAverageCache
        from core.schemas import AppSettings
        settings = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            DATABASE_URL="postgresql://argus:argus@localhost:5432/argus",
            OPENAI_API_KEY="test",
        )
        cache = SeasonAverageCache(settings)
        assert cache.get("nonexistent") is None


# ===================================================================
# Integration: Bayesian mean reversion
# ===================================================================

class TestBayesianMeanReversion:
    """Verify mean reversion uses Bayesian model when available."""

    def test_keyonte_george_25pts_dampened(self):
        """The exact bug scenario: 10 pts in 12 min, line 25+.
        With Bayesian prior (~16 pts season avg), confidence should be
        much lower than the old model's 94%."""
        from sports.nba.strategies.mean_reversion import MeanReversionStrategy

        prior = _prior(pts_per_min=0.5)  # ~16 pts/game

        class FakeSeasonCache:
            def get(self, player_id):
                return prior if player_id == "p1" else None

        s = MeanReversionStrategy(
            min_divergence_cents=5,
            season_avg_cache=FakeSeasonCache(),
        )

        player = _player(pts=10, minutes=12.0, fga=8)
        teammate = PlayerBoxScore(
            player_id="t1", first_name="Team", last_name="Mate",
            team_abbr="DAL", minutes=12.0, pts=8, fgm=3, fga=10,
            fg3m=1, fg3a=3, ftm=0, fta=0, reb=2, ast=1, stl=0,
            blk=0, turnover=0, pf=1, plus_minus=0,
        )
        game = _game(quarter=2, clock="6:00", player_stats=[player, teammate])
        from core.schemas import MarketState
        market = MarketState(
            ticker="KXNBAPTS-26MAR11NYKUTA-DALKGEORGE3-25",
            yes_bid=30, yes_ask=33,
            no_bid=67, no_ask=70,
            volume=100, timestamp=datetime.utcnow(),
        )

        prob = s.model_probability(game, market)
        assert prob is not None
        # With Bayesian prior + high-line dampening, should be well under 80%
        assert prob < 0.80, f"Bayesian prob {prob:.2f} still too high for 25+ pts"

    def test_game_total_with_sharp_line(self):
        """Sharp book line should improve game total projection."""
        from sports.nba.strategies.mean_reversion import MeanReversionStrategy

        class FakeSharpBooks:
            def get_total_line(self, game_id):
                return 220.0
            def get_spread_line(self, game_id):
                return None

        s = MeanReversionStrategy(
            min_divergence_cents=5,
            sharp_book_watcher=FakeSharpBooks(),
        )

        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        from core.schemas import MarketState
        market = MarketState(
            ticker="KXNBATOTAL-26MAR11NYKUTA-200",
            yes_bid=50, yes_ask=52,
            no_bid=48, no_ask=50,
            volume=100, timestamp=datetime.utcnow(),
        )

        prob = s.model_probability(game, market)
        assert prob is not None
        # With sharp line of 220 and pace projecting ~168 at 30 min,
        # Bayesian blend should give a probability between pure pace and prior
        assert 0.0 < prob < 1.0

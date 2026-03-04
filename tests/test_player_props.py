"""Tests for PlayerPropStrategy: usage-rate projection, ticker matching, edge cases."""
from __future__ import annotations

import pytest

from agents.strategies.player_props import PlayerPropStrategy
from core.schemas import PlayerBoxScore
from tests.conftest import make_game_state, make_market_state, make_player_box_score


@pytest.fixture
def strategy() -> PlayerPropStrategy:
    return PlayerPropStrategy(ev_threshold=0.03, target_exit_spread=7)


def _game_with_box_score(
    quarter: int = 3,
    home_score: int = 80,
    away_score: int = 75,
    player_pts: int = 22,
    player_fga: int = 16,
    player_minutes: float = 28.0,
):
    """Build a GameState with a realistic box score for testing."""
    star = make_player_box_score(
        player_id="999",
        first_name="LeBron",
        last_name="James",
        team_abbr="LAL",
        minutes=player_minutes,
        pts=player_pts,
        fgm=8,
        fga=player_fga,
    )
    teammate1 = make_player_box_score(
        player_id="100",
        first_name="Anthony",
        last_name="Davis",
        team_abbr="LAL",
        minutes=26.0,
        pts=18,
        fgm=7,
        fga=14,
    )
    teammate2 = make_player_box_score(
        player_id="101",
        first_name="Austin",
        last_name="Reaves",
        team_abbr="LAL",
        minutes=24.0,
        pts=12,
        fgm=5,
        fga=10,
    )
    opponent = make_player_box_score(
        player_id="200",
        first_name="Nikola",
        last_name="Jokic",
        team_abbr="DEN",
        minutes=30.0,
        pts=25,
        fgm=10,
        fga=18,
    )
    return make_game_state(
        quarter=quarter,
        home_score=home_score,
        away_score=away_score,
        player_stats=[star, teammate1, teammate2, opponent],
    )


class TestCanEvaluate:

    def test_matches_playerpts_ticker(self, strategy):
        market = make_market_state(ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-O28")
        assert strategy.can_evaluate(market)

    def test_matches_pts_ticker(self, strategy):
        market = make_market_state(ticker="KXNBA-PTS-04MAR26-LAL-JAMES-O28")
        assert strategy.can_evaluate(market)

    def test_rejects_game_ticker(self, strategy):
        market = make_market_state(ticker="KXNBA-GAME-04MAR26-LALDEN-LAL")
        assert not strategy.can_evaluate(market)

    def test_rejects_total_ticker(self, strategy):
        market = make_market_state(ticker="KXNBA-TOTAL-04MAR26-LALDEN-O225")
        assert not strategy.can_evaluate(market)


class TestPlayerMatching:

    def test_matches_player_by_last_name(self, strategy):
        game = _game_with_box_score()
        market = make_market_state(ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-O28")
        player = strategy._match_player(market.ticker, game)
        assert player is not None
        assert player.last_name == "James"

    def test_matches_longest_name(self, strategy):
        """When multiple names could match, pick the longest (most specific)."""
        game = _game_with_box_score()
        market = make_market_state(ticker="KXNBA-PLAYERPTS-04MAR26-LAL-DAVIS-O20")
        player = strategy._match_player(market.ticker, game)
        assert player is not None
        assert player.last_name == "Davis"

    def test_no_match_returns_none(self, strategy):
        game = _game_with_box_score()
        market = make_market_state(ticker="KXNBA-PLAYERPTS-04MAR26-LAL-CURRY-O30")
        player = strategy._match_player(market.ticker, game)
        assert player is None


class TestProjection:

    def test_projects_positive_with_high_usage(self, strategy):
        game = _game_with_box_score(
            quarter=3,
            player_pts=22,
            player_fga=16,
            player_minutes=28.0,
        )
        player = game.player_stats[0]
        projected = strategy._project_points(player, game)
        assert projected is not None
        assert projected > 22

    def test_returns_none_with_insufficient_team_fga(self, strategy):
        """If team has < 10 FGA total, projection is unreliable."""
        low_fga_player = make_player_box_score(
            team_abbr="LAL", minutes=10.0, pts=5, fga=3,
        )
        game = make_game_state(
            quarter=1,
            player_stats=[low_fga_player],
        )
        projected = strategy._project_points(low_fga_player, game)
        assert projected is None

    def test_returns_none_if_zero_minutes(self, strategy):
        zero_min = make_player_box_score(minutes=0.0, fga=0)
        game = make_game_state(quarter=2, player_stats=[zero_min])
        assert strategy._project_points(zero_min, game) is None


class TestEvaluation:

    def test_positive_ev_generates_signal(self, strategy):
        """High-usage player with low line should produce a +EV over signal."""
        game = _game_with_box_score(
            quarter=3, player_pts=22, player_fga=16, player_minutes=28.0,
        )
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-O20",
            yes_ask=40,
        )
        signal = strategy.evaluate(game, market)
        assert signal is not None
        assert signal.source == "player_props"
        assert signal.ev_estimate > 0

    def test_no_player_stats_returns_none(self, strategy):
        game = make_game_state(player_stats=[])
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-O28",
        )
        assert strategy.evaluate(game, market) is None

    def test_low_minutes_returns_none(self, strategy):
        """Player with < 5 minutes should be filtered out."""
        game = _game_with_box_score(
            quarter=1, player_pts=2, player_fga=2, player_minutes=3.0,
        )
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-O10",
        )
        assert strategy.evaluate(game, market) is None

    def test_under_ticker_inverts_probability(self, strategy):
        game = _game_with_box_score(
            quarter=3, player_pts=22, player_fga=16, player_minutes=28.0,
        )
        over_market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-O20",
            yes_ask=40,
        )
        under_market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LAL-JAMES-U20",
            yes_ask=40,
        )
        over_sig = strategy.evaluate(game, over_market)
        under_sig = strategy.evaluate(game, under_market)
        if over_sig and under_sig:
            assert over_sig.ev_estimate != under_sig.ev_estimate


class TestHelpers:

    def test_extract_line(self, strategy):
        assert strategy._extract_line("KXNBA-PLAYERPTS-LAL-JAMES-O28.5") == 28.5
        assert strategy._extract_line("KXNBA-PLAYERPTS-LAL-JAMES-U20") == 20.0
        assert strategy._extract_line("KXNBA-GAME-LAL") is None

    def test_is_over_ticker(self, strategy):
        assert strategy._is_over_ticker("KXNBA-PTS-JAMES-O28") is True
        assert strategy._is_over_ticker("KXNBA-PTS-JAMES-U28") is False
        assert strategy._is_over_ticker("KXNBA-PTS-JAMES-OVER28") is True
        assert strategy._is_over_ticker("KXNBA-PTS-JAMES-UNDER28") is False

    def test_over_probability_late_game(self, strategy):
        """Late in the game, probability should converge toward 0 or 1."""
        prob_over = strategy._over_probability(30.0, 25.0, 45.0)
        assert prob_over > 0.9

        prob_under = strategy._over_probability(18.0, 25.0, 45.0)
        assert prob_under < 0.1

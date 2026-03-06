"""Tests for strategy-level features: dynamic EV threshold and flash crash."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agents.strategies.base import BaseStrategy
from agents.strategies.flash_crash import FlashCrashStrategy
from agents.strategies.moneyline import MoneylineStrategy
from agents.strategies.player_props import PlayerPropStrategy
from agents.strategies.totals import TotalsStrategy
from core.schemas import GameState, MarketState
from tests.conftest import make_game_state, make_market_state, make_player_box_score


# ===========================================================================
# Phase 1: Piecewise quarter-based EV threshold
# ===========================================================================

Q_MULTS = (2.5, 1.75, 1.25, 0.75)


class TestTimeAdjustedThreshold:

    def test_q1_start_uses_q1_multiplier(self):
        game = make_game_state(quarter=1)
        game = game.model_copy(update={"clock": "12:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        assert threshold == pytest.approx(0.10, abs=0.001)

    def test_q1_midpoint_interpolates_toward_q2(self):
        game = make_game_state(quarter=1)
        game = game.model_copy(update={"clock": "6:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        expected = 0.04 * (2.5 + 0.5 * (1.75 - 2.5))
        assert threshold == pytest.approx(expected, abs=0.001)

    def test_q2_start_uses_q2_multiplier(self):
        game = make_game_state(quarter=2)
        game = game.model_copy(update={"clock": "12:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        assert threshold == pytest.approx(0.07, abs=0.001)

    def test_q3_start_uses_q3_multiplier(self):
        game = make_game_state(quarter=3)
        game = game.model_copy(update={"clock": "12:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        assert threshold == pytest.approx(0.05, abs=0.001)

    def test_q4_start_uses_q4_multiplier(self):
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "12:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        assert threshold == pytest.approx(0.03, abs=0.001)

    def test_q4_end_clamps_to_q4(self):
        game = make_game_state(quarter=4)
        game = game.model_copy(update={"clock": "0:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        assert threshold == pytest.approx(0.03, abs=0.001)

    def test_overtime_clamps_to_q4(self):
        game = make_game_state(quarter=5)
        game = game.model_copy(update={"clock": "5:00"})
        threshold = BaseStrategy.time_adjusted_ev_threshold(0.04, game, Q_MULTS)
        assert threshold == pytest.approx(0.03, abs=0.001)


class TestStrategyQuarterMultipliers:

    def test_moneyline_stores_multipliers(self):
        s = MoneylineStrategy(ev_threshold=0.04, quarter_multipliers=Q_MULTS)
        assert s._quarter_multipliers == Q_MULTS

    def test_totals_stores_multipliers(self):
        s = TotalsStrategy(ev_threshold=0.04, quarter_multipliers=Q_MULTS)
        assert s._quarter_multipliers == Q_MULTS

    def test_player_props_stores_multipliers(self):
        s = PlayerPropStrategy(ev_threshold=0.04, quarter_multipliers=Q_MULTS)
        assert s._quarter_multipliers == Q_MULTS

    def test_moneyline_q1_rejects_marginal(self):
        """Q1 with small diff: model prob ~0.57 at 48c ask → EV ~0.09,
        but Q1 threshold is 0.10 so it's rejected."""
        s = MoneylineStrategy(ev_threshold=0.04, quarter_multipliers=Q_MULTS)
        game = make_game_state(quarter=1, home_score=26, away_score=28)
        game = game.model_copy(update={"clock": "10:00"})
        market = make_market_state(
            ticker="KXNBAGAME-04MAR26-DENLAL-DEN", yes_bid=47, yes_ask=48,
        )
        assert s.evaluate(game, market) is None

    def test_player_props_high_threshold_rejects(self):
        """With base_threshold=0.50 (absurdly high), Q1 threshold > 1.0 — always rejects."""
        player = make_player_box_score(minutes=8.0, pts=3, fga=10)
        teammate = make_player_box_score(
            player_id="2", first_name="Anthony", last_name="Davis",
            team_abbr="LAL", minutes=8.0, pts=3, fga=8,
        )
        s = PlayerPropStrategy(ev_threshold=0.50, quarter_multipliers=Q_MULTS)
        game = make_game_state(quarter=1, home_score=6, away_score=6, player_stats=[player, teammate])
        game = game.model_copy(update={"clock": "6:00"})
        market = make_market_state(ticker="KXNBA-PLAYERPTS-04MAR26-LJAMES-O25", yes_bid=10, yes_ask=12)
        assert s.evaluate(game, market) is None


# ===========================================================================
# Phase 2: Flash crash strategy (updated with score_delta_limit)
# ===========================================================================

def _make_flash_market(
    ticker: str = "KXNBAGAME-04MAR26-DENLAL-DEN",
    yes_bid: int = 50,
    yes_ask: int = 52,
) -> MarketState:
    return make_market_state(ticker=ticker, yes_bid=yes_bid, yes_ask=yes_ask)


class TestFlashCrashCanEvaluate:

    def test_matches_nba_tickers(self):
        s = FlashCrashStrategy()
        assert s.can_evaluate(_make_flash_market(ticker="KXNBAGAME-LAL"))
        assert not s.can_evaluate(_make_flash_market(ticker="NFL-SPREAD-KC"))


class TestFlashCrashNoDrop:

    def test_no_drop_returns_none(self):
        s = FlashCrashStrategy(window_seconds=10.0, drop_threshold_cents=20)
        game = make_game_state()

        with patch("time.monotonic", return_value=100.0):
            assert s.evaluate(game, _make_flash_market(yes_bid=50, yes_ask=52)) is None

        with patch("time.monotonic", return_value=105.0):
            assert s.evaluate(game, _make_flash_market(yes_bid=48, yes_ask=50)) is None


class TestFlashCrashTriggered:

    def test_large_drop_no_score_run_fires(self):
        s = FlashCrashStrategy(
            window_seconds=10.0, drop_threshold_cents=20, exit_spread=6,
            score_delta_limit=3,
        )
        game = make_game_state(home_score=80, away_score=85)

        with patch("time.monotonic", return_value=100.0):
            s.evaluate(game, _make_flash_market(yes_bid=60, yes_ask=62))
        with patch("time.monotonic", return_value=102.0):
            s.evaluate(game, _make_flash_market(yes_bid=55, yes_ask=57))
        with patch("time.monotonic", return_value=108.0):
            result = s.evaluate(game, _make_flash_market(yes_bid=35, yes_ask=37))

        assert result is not None
        assert result.source == "flash_crash"
        assert result.entry_price == 37
        assert result.exit_price == 43  # +6c bounce target

    def test_small_score_run_within_limit_fires(self):
        """Opponent scored 2 pts (under 3-pt limit) — still behavioral."""
        s = FlashCrashStrategy(
            window_seconds=10.0, drop_threshold_cents=20, exit_spread=6,
            score_delta_limit=3,
        )
        game_before = make_game_state(home_score=80, away_score=85)

        with patch("time.monotonic", return_value=100.0):
            s.evaluate(game_before, _make_flash_market(yes_bid=60, yes_ask=62))

        game_after = make_game_state(home_score=80, away_score=87)
        with patch("time.monotonic", return_value=108.0):
            result = s.evaluate(game_after, _make_flash_market(yes_bid=35, yes_ask=37))

        assert result is not None

    def test_large_score_run_above_limit_vetoed(self):
        """Opponent scored 4 pts (above 3-pt limit) — mathematically justified drop."""
        s = FlashCrashStrategy(
            window_seconds=10.0, drop_threshold_cents=20, exit_spread=6,
            score_delta_limit=3,
        )
        game_before = make_game_state(home_score=80, away_score=85)

        with patch("time.monotonic", return_value=100.0):
            s.evaluate(game_before, _make_flash_market(yes_bid=60, yes_ask=62))

        game_after = make_game_state(home_score=80, away_score=89)
        with patch("time.monotonic", return_value=108.0):
            result = s.evaluate(game_after, _make_flash_market(yes_bid=35, yes_ask=37))

        assert result is None

    def test_sub_min_price_ignored(self):
        s = FlashCrashStrategy(
            window_seconds=10.0, drop_threshold_cents=5, min_price_cents=15,
        )
        game = make_game_state()

        with patch("time.monotonic", return_value=100.0):
            s.evaluate(game, _make_flash_market(yes_bid=20, yes_ask=22))
        with patch("time.monotonic", return_value=108.0):
            assert s.evaluate(game, _make_flash_market(yes_bid=10, yes_ask=12)) is None

    def test_insufficient_history_returns_none(self):
        s = FlashCrashStrategy(window_seconds=10.0, drop_threshold_cents=5)
        game = make_game_state()

        with patch("time.monotonic", return_value=100.0):
            assert s.evaluate(game, _make_flash_market(yes_bid=30, yes_ask=32)) is None


class TestFlashCrashWindowExpiry:

    def test_old_prices_outside_window_ignored(self):
        s = FlashCrashStrategy(
            window_seconds=10.0, drop_threshold_cents=20, exit_spread=6,
        )
        game = make_game_state(home_score=80, away_score=85)

        with patch("time.monotonic", return_value=100.0):
            s.evaluate(game, _make_flash_market(yes_bid=60, yes_ask=62))
        with patch("time.monotonic", return_value=120.0):
            s.evaluate(game, _make_flash_market(yes_bid=40, yes_ask=42))
        with patch("time.monotonic", return_value=125.0):
            assert s.evaluate(game, _make_flash_market(yes_bid=35, yes_ask=37)) is None

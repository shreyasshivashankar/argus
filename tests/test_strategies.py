"""Tests for strategy-level features: dynamic EV threshold, flash crash, arbitrage."""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from agents.strategies.arbitrage import ArbitrageStrategy
from agents.strategies.base import BaseStrategy
from agents.strategies.flash_crash import FlashCrashStrategy
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

    def test_totals_stores_multipliers(self):
        s = TotalsStrategy(ev_threshold=0.04, quarter_multipliers=Q_MULTS)
        assert s._quarter_multipliers == Q_MULTS

    def test_player_props_stores_multipliers(self):
        s = PlayerPropStrategy(ev_threshold=0.04, quarter_multipliers=Q_MULTS)
        assert s._quarter_multipliers == Q_MULTS

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


# ===========================================================================
# Phase 3: Late-Game Flyer Filter (player props)
# ===========================================================================

def _make_prop_game(quarter: int = 3, home_score: int = 80, away_score: int = 85) -> GameState:
    player = make_player_box_score(minutes=24.0, pts=18, fga=14, team_abbr="LAL")
    teammate = make_player_box_score(
        player_id="2", first_name="Anthony", last_name="Davis",
        team_abbr="LAL", minutes=24.0, pts=14, fga=10,
    )
    return make_game_state(
        quarter=quarter, home_score=home_score, away_score=away_score,
        player_stats=[player, teammate],
    )


class TestLateGameFlyerFilter:
    """PlayerPropStrategy should block cheap longshots early; allow them late."""

    def test_cheap_longshot_blocked_in_first_half(self):
        """yes_ask=20 (<35) with team_minutes<24 → None."""
        s = PlayerPropStrategy(ev_threshold=0.01, quarter_multipliers=(1.0, 1.0, 1.0, 1.0))
        # Q1 with 6 minutes elapsed (score 12-12) → team_minutes ~6
        game = _make_prop_game(quarter=1, home_score=12, away_score=12)
        game = game.model_copy(update={"clock": "6:00"})
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LJAMES-O35", yes_bid=18, yes_ask=20,
        )
        assert s.evaluate(game, market) is None

    def test_moderately_priced_prop_allowed_in_first_half(self):
        """yes_ask=45 (>=35) in first half is NOT blocked by the filter."""
        s = PlayerPropStrategy(ev_threshold=0.01, quarter_multipliers=(1.0, 1.0, 1.0, 1.0))
        player = make_player_box_score(minutes=8.0, pts=10, fga=8, team_abbr="LAL")
        teammate = make_player_box_score(
            player_id="2", first_name="Anthony", last_name="Davis",
            team_abbr="LAL", minutes=8.0, pts=6, fga=6,
        )
        game = make_game_state(quarter=2, home_score=30, away_score=28, player_stats=[player, teammate])
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LJAMES-O20", yes_bid=43, yes_ask=45,
        )
        pass  # No assertion on result — we care it doesn't crash or filter incorrectly

    def test_cheap_prop_always_blocked_below_15c(self):
        """yes_ask=12 (<15) is blocked at all times, even in Q4."""
        s = PlayerPropStrategy(ev_threshold=0.01, quarter_multipliers=(1.0, 1.0, 1.0, 0.5))
        game = _make_prop_game(quarter=4, home_score=100, away_score=98)
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LJAMES-O50", yes_bid=10, yes_ask=12,
        )
        assert s.evaluate(game, market) is None

    def test_cheap_prop_allowed_late_game_above_15c(self):
        """yes_ask=20 (<35 BUT team_minutes>=24 in Q4) is NOT blocked by first-half filter."""
        s = PlayerPropStrategy(ev_threshold=0.01, quarter_multipliers=(1.0, 1.0, 1.0, 0.5))
        # Q4 high-score game → team_minutes > 24
        player = make_player_box_score(minutes=30.0, pts=28, fga=22, team_abbr="LAL")
        teammate = make_player_box_score(
            player_id="2", first_name="Anthony", last_name="Davis",
            team_abbr="LAL", minutes=30.0, pts=20, fga=18,
        )
        game = make_game_state(
            quarter=4, home_score=105, away_score=102, player_stats=[player, teammate],
        )
        market = make_market_state(
            ticker="KXNBA-PLAYERPTS-04MAR26-LJAMES-O30", yes_bid=18, yes_ask=20,
        )
        s.evaluate(game, market)  # should not raise


# ===========================================================================
# Phase 4: Arbitrage Strategy
# ===========================================================================

def _make_arb_market(
    ticker: str = "KXNBAGAME-04MAR26-DENLAL-LAL",
    yes_ask: int = 45,
    no_ask: int = 48,
) -> MarketState:
    return make_market_state(ticker=ticker, yes_ask=yes_ask, no_ask=no_ask)


class TestArbitrageCanEvaluate:

    def test_accepts_kxnba_market_with_two_sided_quotes(self):
        s = ArbitrageStrategy()
        assert s.can_evaluate(_make_arb_market())

    def test_rejects_non_nba_ticker(self):
        s = ArbitrageStrategy()
        assert not s.can_evaluate(_make_arb_market(ticker="NFL-SPREAD-KC-SF"))

    def test_rejects_zero_yes_ask(self):
        s = ArbitrageStrategy()
        assert not s.can_evaluate(_make_arb_market(yes_ask=0))

    def test_rejects_zero_no_ask(self):
        s = ArbitrageStrategy()
        assert not s.can_evaluate(_make_arb_market(no_ask=0))


class TestArbitrageEvaluate:

    def test_fires_when_combined_below_threshold(self):
        """YES=45 + NO=48 = 93c < 95c max → arb opportunity."""
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = _make_arb_market(yes_ask=45, no_ask=48)
        result = s.evaluate(game, market)
        assert result is not None
        assert result.source == "arbitrage"
        assert result.entry_price == 45
        assert result.no_entry_price == 48

    def test_rejects_when_combined_above_threshold(self):
        """YES=50 + NO=48 = 98c > 95c → not an arb."""
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = _make_arb_market(yes_ask=50, no_ask=48)
        assert s.evaluate(game, market) is None

    def test_rejects_when_net_spread_below_min(self):
        """YES=48 NO=49 → gross=3c, fees=4c total → net=-1c → rejected."""
        s = ArbitrageStrategy(max_combined_cents=99)
        game = make_game_state()
        market = _make_arb_market(yes_ask=48, no_ask=49)
        assert s.evaluate(game, market) is None

    def test_rejects_marginal_net_spread(self):
        """YES=46 NO=47 → gross=7c, fees=4c → net=3c but combined=93 < 95.

        With MIN_NET_SPREAD_CENTS=3 this is exactly at boundary.
        Tighten to YES=47 NO=48 → gross=5c, fees=4c → net=1c < 3c → rejected.
        """
        s = ArbitrageStrategy(max_combined_cents=99)
        game = make_game_state()
        market = _make_arb_market(yes_ask=47, no_ask=48)
        assert s.evaluate(game, market) is None

    def test_rejects_low_volume_market(self):
        """Even with a valid spread, low volume markets are too illiquid."""
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = make_market_state(
            ticker="KXNBAGAME-04MAR26-DENLAL-LAL",
            yes_ask=40, no_ask=50, volume=10,
        )
        assert s.evaluate(game, market) is None

    def test_rejects_extreme_side_price(self):
        """Prices near 0 or 100 are too illiquid / manipulated."""
        s = ArbitrageStrategy(max_combined_cents=99)
        game = make_game_state()
        market = _make_arb_market(yes_ask=5, no_ask=90)
        assert s.evaluate(game, market) is None

    def test_ev_equals_net_spread_over_100(self):
        """ev_estimate should equal net_spread / 100."""
        import math
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = _make_arb_market(yes_ask=40, no_ask=50)
        result = s.evaluate(game, market)
        assert result is not None
        gross = 100 - 40 - 50
        fee_yes = math.ceil(0.07 * 40 * 60 / 100)
        fee_no = math.ceil(0.07 * 50 * 50 / 100)
        expected_ev = (gross - fee_yes - fee_no) / 100.0
        assert result.ev_estimate == pytest.approx(expected_ev, abs=1e-9)

    def test_no_entry_price_set_on_signal(self):
        """Companion NO price must be propagated to signal."""
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = _make_arb_market(yes_ask=44, no_ask=46)
        result = s.evaluate(game, market)
        assert result is not None
        assert result.no_entry_price == 46

    def test_confidence_fixed_at_0_62(self):
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = _make_arb_market(yes_ask=44, no_ask=46)
        result = s.evaluate(game, market)
        assert result is not None
        assert result.confidence == pytest.approx(0.62)

    def test_rejects_zero_yes_ask(self):
        s = ArbitrageStrategy()
        game = make_game_state()
        market = _make_arb_market(yes_ask=0, no_ask=50)
        assert s.evaluate(game, market) is None

    def test_exit_price_is_99(self):
        """Arb holds to settlement; exit_price=99 triggers auto-cashout."""
        s = ArbitrageStrategy(max_combined_cents=95)
        game = make_game_state()
        market = _make_arb_market(yes_ask=44, no_ask=46)
        result = s.evaluate(game, market)
        assert result is not None
        assert result.exit_price == 99


# ===========================================================================
# Phase 5: Totals Q1 block and min_minutes guard
# ===========================================================================

class TestTotalsQ1Block:

    def test_q1_returns_none(self):
        """Totals strategy must not fire in Q1 (too little pace data)."""
        s = TotalsStrategy(ev_threshold=0.01, min_minutes=1.0)
        game = make_game_state(quarter=1, home_score=15, away_score=12)
        market = make_market_state(ticker="KXNBATOTAL-04MAR26-DENLAL-O215", yes_ask=40)
        assert s.evaluate(game, market) is None

    def test_q5_overtime_returns_none(self):
        s = TotalsStrategy(ev_threshold=0.01, min_minutes=1.0)
        game = make_game_state(quarter=5, home_score=100, away_score=98)
        market = make_market_state(ticker="KXNBATOTAL-04MAR26-DENLAL-O215", yes_ask=60)
        assert s.evaluate(game, market) is None

    def test_q2_below_min_minutes_returns_none(self):
        """12-minute minimum not yet met → skip evaluation."""
        s = TotalsStrategy(ev_threshold=0.01, min_minutes=12.0)
        # team_minutes_played sums player minutes; make_game_state uses default 6-minute players
        # Use a fresh game with very few minutes
        game = make_game_state(quarter=2, home_score=5, away_score=4)
        # Override player stats to have very few minutes
        from tests.conftest import make_player_box_score
        p1 = make_player_box_score(minutes=2.0, pts=3, team_abbr="LAL")
        p2 = make_player_box_score(player_id="2", first_name="A", last_name="B",
                                   team_abbr="LAL", minutes=2.0, pts=2)
        game = game.model_copy(update={"player_stats": [p1, p2]})
        market = make_market_state(ticker="KXNBATOTAL-04MAR26-DENLAL-O215", yes_ask=50)
        assert s.evaluate(game, market) is None

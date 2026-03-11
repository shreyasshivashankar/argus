"""Comprehensive tests for the mean reversion strategy.

Covers: player props (pts, reb, ast, stl, blk, 3pt), game totals,
team totals, spreads, price history tracking, divergence detection,
edge cases, and bailout interface.
"""
from __future__ import annotations

import time
from datetime import datetime

import pytest

from core.schemas import GameState, MarketState, PlayerBoxScore
from sports.nba.strategies.mean_reversion import MeanReversionStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _game(
    *,
    home_score: int = 55,
    away_score: int = 50,
    quarter: int = 3,
    clock: str = "6:00",
    home_abbr: str = "ATL",
    away_abbr: str = "DAL",
    player_stats: list[PlayerBoxScore] | None = None,
    game_id: str = "game-001",
) -> GameState:
    return GameState(
        game_id=game_id,
        home_team="Hawks",
        away_team="Mavericks",
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        home_score=home_score,
        away_score=away_score,
        quarter=quarter,
        clock=clock,
        timestamp=datetime.utcnow(),
        player_stats=player_stats or [],
    )


def _market(
    *,
    ticker: str = "KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
    yes_bid: int = 45,
    yes_ask: int = 48,
) -> MarketState:
    return MarketState(
        ticker=ticker,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=100 - yes_ask,
        no_ask=100 - yes_bid,
        volume=200,
        timestamp=datetime.utcnow(),
    )


def _pj_washington(
    *,
    pts: int = 4,
    minutes: float = 12.0,
    fga: int = 6,
    reb: int = 3,
    ast: int = 2,
    stl: int = 1,
    blk: int = 0,
    fg3m: int = 1,
) -> PlayerBoxScore:
    return PlayerBoxScore(
        player_id="pw25",
        first_name="P.J.",
        last_name="Washington",
        team_abbr="DAL",
        minutes=minutes,
        pts=pts,
        fgm=3,
        fga=fga,
        fg3m=fg3m,
        fg3a=3,
        ftm=0,
        fta=0,
        reb=reb,
        ast=ast,
        stl=stl,
        blk=blk,
        turnover=1,
        pf=2,
        plus_minus=-3,
    )


def _teammate(*, fga: int = 12) -> PlayerBoxScore:
    """A generic teammate to provide team FGA context."""
    return PlayerBoxScore(
        player_id="tm01",
        first_name="Kyrie",
        last_name="Irving",
        team_abbr="DAL",
        minutes=15.0,
        pts=12,
        fgm=5,
        fga=fga,
        fg3m=2,
        fg3a=4,
        ftm=0,
        fta=0,
        reb=2,
        ast=4,
        stl=0,
        blk=0,
        turnover=0,
        pf=1,
        plus_minus=2,
    )


def _strategy(**kwargs) -> MeanReversionStrategy:
    defaults = dict(
        min_divergence_cents=8,
        exit_spread=5,
        min_minutes=6.0,
    )
    defaults.update(kwargs)
    return MeanReversionStrategy(**defaults)


def _seed_price_history(
    strat: MeanReversionStrategy,
    ticker: str,
    prices: list[int],
    interval: float = 1.0,
) -> None:
    """Seed the strategy's price history with a series of bids."""
    now = time.monotonic()
    start = now - len(prices) * interval
    for i, price in enumerate(prices):
        strat._record_price(ticker, start + i * interval, price)


# ===================================================================
# CAN_EVALUATE
# ===================================================================

class TestCanEvaluate:
    def test_accepts_nba_ticker(self):
        s = _strategy()
        m = _market(ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10")
        assert s.can_evaluate(m) is True

    def test_rejects_non_nba(self):
        s = _strategy()
        m = _market(ticker="KXMLB-NYYLAL-O8")
        assert s.can_evaluate(m) is False

    def test_accepts_totals(self):
        s = _strategy()
        m = _market(ticker="KXNBATOTAL-26MAR10DALATL-227")
        assert s.can_evaluate(m) is True

    def test_accepts_spread(self):
        s = _strategy()
        m = _market(ticker="KXNBASPREAD-26MAR10DALATL-ATL6")
        assert s.can_evaluate(m) is True

    def test_accepts_team_total(self):
        s = _strategy()
        m = _market(ticker="KXNBATEAMTOTAL-26MAR10DALATL-ATL126")
        assert s.can_evaluate(m) is True


# ===================================================================
# PLAYER PROPS — POINTS
# ===================================================================

class TestPlayerPropsPoints:
    """Test mean reversion on player-points markets."""

    def test_fires_when_price_below_fair_value(self):
        """PJ Washington has 4 pts in 12 min → pace projects ~16 pts.
        Market for 10+ is at 45c ask, but model says ~70%+ → divergence."""
        s = _strategy(min_divergence_cents=8)
        pj = _pj_washington(pts=4, minutes=12.0, fga=6)
        game = _game(
            quarter=2, clock="6:00",
            player_stats=[pj, _teammate()],
        )
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=43,
            yes_ask=45,
        )
        _seed_price_history(s, market.ticker, [65, 63, 60, 55, 50, 45])

        signal = s.evaluate(game, market)
        assert signal is not None
        assert signal.source == "mean_reversion"
        assert signal.entry_price == 45
        assert signal.ev_estimate > 0
        assert signal.confidence > 0.5

    def test_no_signal_when_price_matches_fair_value(self):
        """If market price is close to fair value, no divergence."""
        s = _strategy(min_divergence_cents=8)
        # Player with 2 pts in 12 min, line 10 → projecting ~8 pts
        # ~50% prob → fair value ~50c, market at 48c → no divergence
        pj = _pj_washington(pts=2, minutes=12.0, fga=4)
        game = _game(
            quarter=2, clock="6:00",
            player_stats=[pj, _teammate()],
        )
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=48,
            yes_ask=50,
        )
        _seed_price_history(s, market.ticker, [52, 51, 50, 49, 48])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_no_signal_too_early(self):
        """Strategy requires minimum minutes of game time."""
        s = _strategy(min_minutes=6.0)
        pj = _pj_washington(pts=2, minutes=3.0, fga=2)
        game = _game(
            quarter=1, clock="9:00",
            player_stats=[pj, _teammate()],
        )
        market = _market(yes_bid=30, yes_ask=32)
        _seed_price_history(s, market.ticker, [55, 50, 45, 35, 32])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_no_signal_without_player_stats(self):
        s = _strategy()
        game = _game(quarter=3, clock="6:00", player_stats=[])
        market = _market(yes_bid=30, yes_ask=32)
        _seed_price_history(s, market.ticker, [55, 50, 45, 35, 32])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_no_signal_player_not_in_game(self):
        """Ticker references a player not in the box score."""
        s = _strategy()
        game = _game(
            quarter=3, clock="6:00",
            player_stats=[_teammate()],  # No PJ Washington
        )
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=30, yes_ask=32,
        )
        _seed_price_history(s, market.ticker, [55, 50, 45, 35, 32])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_exit_price_splits_divergence(self):
        """Exit target should be between entry and fair value."""
        s = _strategy(min_divergence_cents=8, exit_spread=5)
        pj = _pj_washington(pts=8, minutes=15.0, fga=10)
        game = _game(
            quarter=3, clock="6:00",
            player_stats=[pj, _teammate()],
        )
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=38,
            yes_ask=40,
        )
        _seed_price_history(s, market.ticker, [65, 60, 55, 50, 45, 40])

        signal = s.evaluate(game, market)
        assert signal is not None
        assert signal.exit_price > signal.entry_price
        assert signal.exit_price <= 99


# ===================================================================
# PLAYER PROPS — OTHER STATS (reb, ast, stl, blk, 3pt)
# ===================================================================

class TestPlayerPropsOtherStats:

    def test_rebounds_market(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(reb=5, minutes=12.0)
        game = _game(quarter=2, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBAREB-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=35,
            yes_ask=38,
        )
        _seed_price_history(s, market.ticker, [60, 55, 50, 45, 38])

        signal = s.evaluate(game, market)
        # With 5 reb in 12 min, projecting ~20 reb for line 10 → high prob
        assert signal is not None
        assert signal.source == "mean_reversion"

    def test_assists_market(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(ast=4, minutes=15.0)
        game = _game(quarter=2, clock="3:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBAAST-26MAR10DALATL-DALPWASHINGTON25-5",
            yes_bid=40,
            yes_ask=42,
        )
        _seed_price_history(s, market.ticker, [65, 60, 55, 48, 42])

        signal = s.evaluate(game, market)
        assert signal is not None

    def test_steals_market(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(stl=2, minutes=15.0)
        game = _game(quarter=3, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBASTL-26MAR10DALATL-DALPWASHINGTON25-2",
            yes_bid=50,
            yes_ask=52,
        )
        _seed_price_history(s, market.ticker, [75, 70, 65, 58, 52])

        signal = s.evaluate(game, market)
        assert signal is not None

    def test_blocks_market(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(blk=2, minutes=15.0)
        game = _game(quarter=3, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBABLK-26MAR10DALATL-DALPWASHINGTON25-2",
            yes_bid=50,
            yes_ask=52,
        )
        _seed_price_history(s, market.ticker, [75, 70, 65, 58, 52])

        signal = s.evaluate(game, market)
        assert signal is not None

    def test_three_pointers_market(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(fg3m=3, minutes=15.0)
        game = _game(quarter=3, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBA3PT-26MAR10DALATL-DALPWASHINGTON25-3",
            yes_bid=45,
            yes_ask=48,
        )
        _seed_price_history(s, market.ticker, [70, 65, 60, 55, 48])

        signal = s.evaluate(game, market)
        assert signal is not None


# ===================================================================
# GAME TOTALS
# ===================================================================

class TestGameTotals:

    def test_over_total_underpriced(self):
        """Score pace projects well over the line, but market is low."""
        s = _strategy(min_divergence_cents=8)
        # 130 total in Q3 at 6:00 left (~30 min played) → pace 4.33/min → projected 208
        # Line 190 → model says ~85%+ over → fair value ~85c, market at 58c → divergence
        game = _game(home_score=68, away_score=62, quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-190",
            yes_bid=55,
            yes_ask=58,
        )
        _seed_price_history(s, market.ticker, [78, 75, 70, 65, 58])

        signal = s.evaluate(game, market)
        assert signal is not None
        assert signal.source == "mean_reversion"

    def test_total_no_divergence(self):
        """Market price matches fair value — no signal."""
        s = _strategy(min_divergence_cents=8)
        # 105 at ~30 min → pace 168 → line 200 → well under → ~20%
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-240",
            yes_bid=18,
            yes_ask=20,
        )
        _seed_price_history(s, market.ticker, [22, 21, 20, 19, 20])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_under_total_ticker(self):
        """Under market: fair value inverted."""
        s = _strategy(min_divergence_cents=8)
        # pace projects ~168 total → under 200 should be high prob
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-U240",
            yes_bid=50,
            yes_ask=52,
        )
        _seed_price_history(s, market.ticker, [78, 75, 70, 60, 52])

        signal = s.evaluate(game, market)
        assert signal is not None


# ===================================================================
# TEAM TOTALS
# ===================================================================

class TestTeamTotals:

    def test_home_team_total(self):
        s = _strategy(min_divergence_cents=8)
        # ATL (home) at 55 pts in ~30 min → pace ~88 → line 126 is high
        # But let's make score higher to create divergence
        game = _game(home_score=75, away_score=50, quarter=3, clock="6:00",
                      home_abbr="ATL", away_abbr="DAL")
        market = _market(
            ticker="KXNBATEAMTOTAL-26MAR10DALATL-ATL110",
            yes_bid=55,
            yes_ask=58,
        )
        _seed_price_history(s, market.ticker, [80, 75, 70, 65, 58])

        signal = s.evaluate(game, market)
        assert signal is not None

    def test_away_team_total(self):
        s = _strategy(min_divergence_cents=8)
        # DAL has 75 in ~30 min → pace 2.5/min → projected 120 → line 100, ~90%+
        game = _game(home_score=50, away_score=75, quarter=3, clock="6:00",
                      home_abbr="ATL", away_abbr="DAL")
        market = _market(
            ticker="KXNBATEAMTOTAL-26MAR10DALATL-DAL100",
            yes_bid=55,
            yes_ask=58,
        )
        _seed_price_history(s, market.ticker, [80, 75, 70, 65, 58])

        signal = s.evaluate(game, market)
        assert signal is not None

    def test_unknown_team_abbr(self):
        """No signal when team can't be identified in ticker."""
        s = _strategy(min_divergence_cents=5)
        game = _game(home_score=75, away_score=50, quarter=3, clock="6:00",
                      home_abbr="ATL", away_abbr="DAL")
        market = _market(
            ticker="KXNBATEAMTOTAL-26MAR10DALATL-BOS110",
            yes_bid=30, yes_ask=32,
        )
        _seed_price_history(s, market.ticker, [60, 55, 50, 40, 32])

        signal = s.evaluate(game, market)
        assert signal is None


# ===================================================================
# SPREADS
# ===================================================================

class TestSpreads:

    def test_spread_underpriced(self):
        """ATL leading by 15 in Q3 → ATL-6 should be high prob."""
        s = _strategy(min_divergence_cents=8)
        game = _game(home_score=70, away_score=55, quarter=3, clock="6:00",
                      home_abbr="ATL", away_abbr="DAL")
        market = _market(
            ticker="KXNBASPREAD-26MAR10DALATL-ATL6",
            yes_bid=55,
            yes_ask=58,
        )
        _seed_price_history(s, market.ticker, [82, 78, 72, 65, 58])

        signal = s.evaluate(game, market)
        assert signal is not None

    def test_spread_no_divergence(self):
        """Close game, spread is fairly priced."""
        s = _strategy(min_divergence_cents=8)
        game = _game(home_score=55, away_score=53, quarter=3, clock="6:00",
                      home_abbr="ATL", away_abbr="DAL")
        market = _market(
            ticker="KXNBASPREAD-26MAR10DALATL-ATL6",
            yes_bid=48,
            yes_ask=50,
        )
        _seed_price_history(s, market.ticker, [52, 51, 50, 49, 50])

        signal = s.evaluate(game, market)
        assert signal is None


# ===================================================================
# PRICE HISTORY & DIVERGENCE DETECTION
# ===================================================================

class TestPriceHistory:

    def test_no_signal_without_price_history(self):
        """First tick — no history to compare against."""
        s = _strategy()
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-200",
            yes_bid=45,
            yes_ask=48,
        )
        # No seeded history — first evaluate call
        signal = s.evaluate(game, market)
        assert signal is None

    def test_no_signal_when_price_stable(self):
        """Price hasn't dropped — recent high is near current bid."""
        s = _strategy(min_divergence_cents=8)
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-200",
            yes_bid=70,
            yes_ask=72,
        )
        # Stable prices
        _seed_price_history(s, market.ticker, [72, 71, 70, 71, 70])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_record_price_evicts_old_entries(self):
        """Old entries beyond 2x window should be evicted."""
        s = _strategy()
        ticker = "KXNBATEST"
        now = time.monotonic()

        # Add old entry
        s._record_price(ticker, now - 300, 50)
        # Add recent entry
        s._record_price(ticker, now, 45)

        buf = s._price_history[ticker]
        # Old entry should be evicted
        assert len(buf) == 1

    def test_recent_high_returns_max_in_window(self):
        s = _strategy()
        ticker = "KXNBATEST"
        _seed_price_history(s, ticker, [50, 60, 55, 45, 40], interval=1.0)

        now = time.monotonic()
        high = s._recent_high(ticker, now)
        assert high == 60

    def test_recent_high_none_with_insufficient_data(self):
        s = _strategy()
        ticker = "KXNBATEST"
        # Only 1 entry — need at least 3
        now = time.monotonic()
        s._record_price(ticker, now, 50)

        assert s._recent_high(ticker, now) is None


# ===================================================================
# EDGE CASES
# ===================================================================

class TestEdgeCases:

    def test_zero_ask_price(self):
        s = _strategy()
        game = _game(quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-200",
            yes_bid=0, yes_ask=0,
        )
        assert s.evaluate(game, market) is None

    def test_zero_bid_price(self):
        s = _strategy()
        game = _game(quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-200",
            yes_bid=0, yes_ask=5,
        )
        _seed_price_history(s, market.ticker, [30, 25, 20, 10, 5])
        assert s.evaluate(game, market) is None

    def test_no_line_in_ticker(self):
        """Ticker without a parseable numeric line."""
        s = _strategy()
        game = _game(quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBAGAME-26MAR10DALATL-DAL",
            yes_bid=20, yes_ask=22,
        )
        _seed_price_history(s, market.ticker, [50, 45, 40, 30, 22])
        assert s.evaluate(game, market) is None

    def test_exit_price_capped_at_99(self):
        """Exit price should never exceed 99c."""
        s = _strategy(exit_spread=5)
        pj = _pj_washington(pts=12, minutes=15.0, fga=10)
        game = _game(
            quarter=4, clock="2:00",
            player_stats=[pj, _teammate()],
        )
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=85,
            yes_ask=88,
        )
        _seed_price_history(s, market.ticker, [97, 95, 92, 90, 88])

        signal = s.evaluate(game, market)
        if signal is not None:
            assert signal.exit_price <= 99

    def test_player_with_zero_minutes(self):
        """Player hasn't played yet — should not fire."""
        s = _strategy()
        pj = _pj_washington(pts=0, minutes=0.0, fga=0)
        game = _game(quarter=2, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(yes_bid=30, yes_ask=32)
        _seed_price_history(s, market.ticker, [55, 50, 45, 35, 32])

        signal = s.evaluate(game, market)
        assert signal is None

    def test_overtime_game(self):
        """Strategy should still work in OT (quarter > 4)."""
        s = _strategy(min_divergence_cents=8)
        game = _game(
            home_score=110, away_score=108,
            quarter=5, clock="3:00",
        )
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-220",
            yes_bid=60,
            yes_ask=62,
        )
        _seed_price_history(s, market.ticker, [85, 80, 75, 68, 62])

        # Should handle gracefully (may or may not fire depending on projection)
        signal = s.evaluate(game, market)
        # Just verify no crash — OT projections are tricky


# ===================================================================
# BAILOUT INTERFACE
# ===================================================================

class TestModelProbability:

    def test_returns_probability_for_totals(self):
        s = _strategy()
        # 105 total at 30 min → pace 168 → line 170 is close → ~45% prob
        game = _game(home_score=55, away_score=50, quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBATOTAL-26MAR10DALATL-170",
            yes_bid=60, yes_ask=62,
        )
        prob = s.model_probability(game, market)
        assert prob is not None
        assert 0.0 < prob < 1.0

    def test_returns_probability_for_player_props(self):
        s = _strategy()
        pj = _pj_washington(pts=8, minutes=15.0, fga=8)
        game = _game(quarter=3, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=60, yes_ask=62,
        )
        prob = s.model_probability(game, market)
        assert prob is not None
        assert 0.0 < prob < 1.0

    def test_returns_none_for_unparseable_ticker(self):
        s = _strategy()
        game = _game(quarter=3, clock="6:00")
        market = _market(
            ticker="KXNBAGAME-26MAR10DALATL-DAL",
            yes_bid=60, yes_ask=62,
        )
        prob = s.model_probability(game, market)
        assert prob is None

    def test_probability_capped_at_099(self):
        """Model probability should never return 1.0 exactly."""
        s = _strategy()
        pj = _pj_washington(pts=15, minutes=10.0, fga=12)
        game = _game(
            quarter=4, clock="1:00",
            player_stats=[pj, _teammate()],
        )
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=95, yes_ask=97,
        )
        prob = s.model_probability(game, market)
        assert prob is not None
        assert prob <= 0.99


# ===================================================================
# INTERNAL HELPERS
# ===================================================================

class TestHelpers:

    def test_extract_line_numeric_suffix(self):
        s = _strategy()
        assert s._extract_line("KXNBAPTS-DAL-PLAYER-10") == 10.0
        assert s._extract_line("KXNBATOTAL-DALATL-227") == 227.0
        assert s._extract_line("KXNBASPREAD-ATL6") == 6.0
        assert s._extract_line("KXNBATEAMTOTAL-ATL126") == 126.0

    def test_extract_line_no_number(self):
        s = _strategy()
        assert s._extract_line("KXNBAGAME-DAL") is None

    def test_is_over_ticker(self):
        s = _strategy()
        assert s._is_over_ticker("KXNBA-O225") is True
        assert s._is_over_ticker("KXNBA-OVER225") is True
        assert s._is_over_ticker("KXNBA-U225") is False
        assert s._is_over_ticker("KXNBA-UNDER225") is False
        # Default to over
        assert s._is_over_ticker("KXNBA-225") is True

    def test_detect_stat_type(self):
        s = _strategy()
        assert s._detect_stat_type("KXNBAPTS-PLAYER-10") == "pts"
        assert s._detect_stat_type("KXNBAREB-PLAYER-10") == "reb"
        assert s._detect_stat_type("KXNBAAST-PLAYER-5") == "ast"
        assert s._detect_stat_type("KXNBASTL-PLAYER-2") == "stl"
        assert s._detect_stat_type("KXNBABLK-PLAYER-2") == "blk"
        assert s._detect_stat_type("KXNBA3PT-PLAYER-3") == "fg3m"
        assert s._detect_stat_type("KXNBAGAME-DAL") is None

    def test_match_player_by_initial_and_last(self):
        s = _strategy()
        pj = _pj_washington()
        game = _game(player_stats=[pj, _teammate()])
        matched = s._match_player(
            "KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10", game,
        )
        assert matched is not None
        assert matched.player_id == "pw25"

    def test_match_player_not_found(self):
        s = _strategy()
        game = _game(player_stats=[_teammate()])
        matched = s._match_player(
            "KXNBAPTS-DALPWASHINGTON25-10", game,
        )
        assert matched is None


# ===================================================================
# SIGNAL PROPERTIES
# ===================================================================

class TestSignalProperties:

    def test_signal_has_correct_source(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(pts=8, minutes=12.0, fga=8)
        game = _game(quarter=3, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=40, yes_ask=42,
        )
        _seed_price_history(s, market.ticker, [65, 60, 55, 48, 42])

        signal = s.evaluate(game, market)
        assert signal is not None
        assert signal.source == "mean_reversion"
        assert signal.game_id == "game-001"
        assert signal.ticker == market.ticker

    def test_signal_ev_is_positive(self):
        s = _strategy(min_divergence_cents=5)
        pj = _pj_washington(pts=8, minutes=12.0, fga=8)
        game = _game(quarter=3, clock="6:00", player_stats=[pj, _teammate()])
        market = _market(
            ticker="KXNBAPTS-26MAR10DALATL-DALPWASHINGTON25-10",
            yes_bid=40, yes_ask=42,
        )
        _seed_price_history(s, market.ticker, [65, 60, 55, 48, 42])

        signal = s.evaluate(game, market)
        assert signal is not None
        assert signal.ev_estimate > 0.01


# ===================================================================
# INTEGRATION: config.py registration
# ===================================================================

class TestConfigRegistration:

    def test_mean_reversion_in_all_strategies(self):
        from sports.nba.config import ALL_STRATEGIES, STRATEGY_MEAN_REVERSION
        assert STRATEGY_MEAN_REVERSION == "mean_reversion"
        assert "mean_reversion" in ALL_STRATEGIES

    def test_build_strategies_includes_mean_reversion(self):
        from core.schemas import AppSettings
        from sports.nba.config import build_strategies

        settings = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            DATABASE_URL="postgresql://argus:argus@localhost:5432/argus",
            OPENAI_API_KEY="test",
            NBA_ENABLED_STRATEGIES="mean_reversion",
        )
        strats = build_strategies(settings)
        assert len(strats) == 1
        assert strats[0].name == "mean_reversion"

    def test_build_strategies_all_enabled(self):
        from core.schemas import AppSettings
        from sports.nba.config import build_strategies

        settings = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            DATABASE_URL="postgresql://argus:argus@localhost:5432/argus",
            OPENAI_API_KEY="test",
            NBA_ENABLED_STRATEGIES="totals,player_props,arbitrage,flash_crash,mean_reversion",
        )
        strats = build_strategies(settings)
        names = {s.name for s in strats}
        assert "mean_reversion" in names
        assert len(strats) == 5

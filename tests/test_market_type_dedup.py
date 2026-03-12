"""Tests for per-market-type deduplication in the quant agent.

Verifies that the bot cannot hold correlated positions on different lines
of the same market type for the same game (e.g. Over-226 AND Over-244).
"""
from __future__ import annotations

from sports.nba.quant import NBAQuantAgent


# ===================================================================
# _market_type_key classification
# ===================================================================

class TestMarketTypeKey:

    def test_game_total(self):
        assert NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-226") == "TOTAL"
        assert NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-244") == "TOTAL"

    def test_different_lines_same_type(self):
        """Different total lines should produce the same key."""
        k1 = NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-226")
        k2 = NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-244")
        assert k1 == k2

    def test_team_total(self):
        k = NBAQuantAgent._market_type_key("KXNBATEAMTOTAL-26MAR11NYKUTA-NYK126")
        assert k == "TEAMTOTAL"

    def test_spread(self):
        k = NBAQuantAgent._market_type_key("KXNBASPREAD-26MAR11NYKUTA-NYK6")
        assert k == "SPREAD"

    def test_first_half_winner(self):
        k = NBAQuantAgent._market_type_key("KXNBA1HWINNER-26MAR11NYKUTA-NYK")
        assert k == "1HWINNER"

    def test_first_half_total(self):
        k = NBAQuantAgent._market_type_key("KXNBA1HTOTAL-26MAR11NYKUTA-125")
        assert k == "1HTOTAL"

    def test_player_props_includes_player(self):
        """Different players on same stat should be independent."""
        k1 = NBAQuantAgent._market_type_key("KXNBAPTS-26MAR11NYKUTA-NYKKTOWNS32-15")
        k2 = NBAQuantAgent._market_type_key("KXNBAPTS-26MAR11NYKUTA-NYKJBRUNSON11-20")
        assert k1 != k2

    def test_player_props_same_player_different_lines(self):
        """Same player, different lines should be same key."""
        k1 = NBAQuantAgent._market_type_key("KXNBAPTS-26MAR11NYKUTA-NYKKTOWNS32-15")
        k2 = NBAQuantAgent._market_type_key("KXNBAPTS-26MAR11NYKUTA-NYKKTOWNS32-20")
        assert k1 == k2

    def test_player_rebounds(self):
        k = NBAQuantAgent._market_type_key("KXNBAREB-26MAR11NYKUTA-NYKKTOWNS32-10")
        assert k.startswith("REB:")

    def test_player_assists(self):
        k = NBAQuantAgent._market_type_key("KXNBAAST-26MAR11NYKUTA-NYKJBRUNSON11-5")
        assert k.startswith("AST:")

    def test_player_steals(self):
        k = NBAQuantAgent._market_type_key("KXNBASTL-26MAR11NYKUTA-NYKJBRUNSON11-2")
        assert k.startswith("STL:")

    def test_player_blocks(self):
        k = NBAQuantAgent._market_type_key("KXNBABLK-26MAR11NYKUTA-NYKKTOWNS32-2")
        assert k.startswith("BLK:")

    def test_player_three_pointers(self):
        k = NBAQuantAgent._market_type_key("KXNBA3PT-26MAR11NYKUTA-NYKKTOWNS32-3")
        assert k.startswith("3PT:")

    def test_totals_independent_from_team_totals(self):
        """Game totals and team totals are different market types."""
        k1 = NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-226")
        k2 = NBAQuantAgent._market_type_key("KXNBATEAMTOTAL-26MAR11NYKUTA-NYK126")
        assert k1 != k2

    def test_totals_independent_from_spread(self):
        k1 = NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-226")
        k2 = NBAQuantAgent._market_type_key("KXNBASPREAD-26MAR11NYKUTA-NYK6")
        assert k1 != k2

    def test_pts_independent_from_reb_same_player(self):
        """Points and rebounds for same player are independent."""
        k1 = NBAQuantAgent._market_type_key("KXNBAPTS-26MAR11NYKUTA-NYKKTOWNS32-15")
        k2 = NBAQuantAgent._market_type_key("KXNBAREB-26MAR11NYKUTA-NYKKTOWNS32-10")
        assert k1 != k2


# ===================================================================
# _is_market_type_taken
# ===================================================================

class TestIsMarketTypeTaken:

    def _make_agent(self):
        """Create a minimal agent for testing (no bus/client needed)."""
        from unittest.mock import MagicMock
        from core.schemas import AppSettings
        settings = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            DATABASE_URL="postgresql://argus:argus@localhost:5432/argus",
            OPENAI_API_KEY="test",
        )
        bus = MagicMock()
        client = MagicMock()
        agent = NBAQuantAgent.__new__(NBAQuantAgent)
        agent.settings = settings
        agent._pending_market_types = {}
        agent._game_to_tickers = {}
        agent._portfolio = None
        return agent

    def test_not_taken_when_empty(self):
        agent = self._make_agent()
        assert agent._is_market_type_taken("game-1", "KXNBATOTAL-26MAR11NYKUTA-226") is False

    def test_taken_after_pending(self):
        agent = self._make_agent()
        agent._pending_market_types["game-1"] = {"TOTAL"}
        assert agent._is_market_type_taken("game-1", "KXNBATOTAL-26MAR11NYKUTA-244") is True

    def test_different_market_type_not_taken(self):
        agent = self._make_agent()
        agent._pending_market_types["game-1"] = {"TOTAL"}
        assert agent._is_market_type_taken("game-1", "KXNBASPREAD-26MAR11NYKUTA-NYK6") is False

    def test_different_game_not_taken(self):
        agent = self._make_agent()
        agent._pending_market_types["game-1"] = {"TOTAL"}
        assert agent._is_market_type_taken("game-2", "KXNBATOTAL-26MAR11CLEORL-240") is False

    def test_taken_from_portfolio_position(self):
        from core.schemas import PortfolioState, PortfolioPosition, Side
        agent = self._make_agent()
        agent._game_to_tickers["game-1"] = [
            "KXNBATOTAL-26MAR11NYKUTA-226",
            "KXNBATOTAL-26MAR11NYKUTA-244",
        ]
        agent._portfolio = PortfolioState(
            bankroll=100.0,
            positions=[
                PortfolioPosition(
                    client_order_id="exit-1",
                    ticker="KXNBATOTAL-26MAR11NYKUTA-226",
                    side=Side.YES,
                    remaining_count=10,
                    entry_vwap=79.0,
                    target_exit_price=86,
                    kalshi_order_id="k-1",
                ),
            ],
        )
        # Same market type (TOTAL), different line
        assert agent._is_market_type_taken("game-1", "KXNBATOTAL-26MAR11NYKUTA-244") is True

    def test_not_taken_if_position_closed(self):
        from core.schemas import PortfolioState, PortfolioPosition, Side
        agent = self._make_agent()
        agent._game_to_tickers["game-1"] = ["KXNBATOTAL-26MAR11NYKUTA-226"]
        agent._portfolio = PortfolioState(
            bankroll=100.0,
            positions=[
                PortfolioPosition(
                    client_order_id="exit-1",
                    ticker="KXNBATOTAL-26MAR11NYKUTA-226",
                    side=Side.YES,
                    remaining_count=0,  # fully closed
                    entry_vwap=79.0,
                    target_exit_price=86,
                    kalshi_order_id="k-1",
                ),
            ],
        )
        assert agent._is_market_type_taken("game-1", "KXNBATOTAL-26MAR11NYKUTA-244") is False

    def test_player_prop_same_player_taken(self):
        agent = self._make_agent()
        agent._pending_market_types["game-1"] = {"PTS:NYKKTOWNS32"}
        # Same player, different line
        assert agent._is_market_type_taken("game-1", "KXNBAPTS-26MAR11NYKUTA-NYKKTOWNS32-20") is True

    def test_player_prop_different_player_not_taken(self):
        agent = self._make_agent()
        agent._pending_market_types["game-1"] = {"PTS:NYKKTOWNS32"}
        # Different player
        assert agent._is_market_type_taken("game-1", "KXNBAPTS-26MAR11NYKUTA-NYKJBRUNSON11-20") is False


# ===================================================================
# Scenario: the NYK-UTA bug
# ===================================================================

class TestNYKUTAScenario:
    """Reproduce the exact bug: 7 correlated total bets on one game."""

    def test_all_total_lines_produce_same_key(self):
        tickers = [
            "KXNBATOTAL-26MAR11NYKUTA-223",
            "KXNBATOTAL-26MAR11NYKUTA-226",
            "KXNBATOTAL-26MAR11NYKUTA-229",
            "KXNBATOTAL-26MAR11NYKUTA-235",
            "KXNBATOTAL-26MAR11NYKUTA-238",
            "KXNBATOTAL-26MAR11NYKUTA-241",
            "KXNBATOTAL-26MAR11NYKUTA-244",
        ]
        keys = {NBAQuantAgent._market_type_key(t) for t in tickers}
        assert len(keys) == 1
        assert keys == {"TOTAL"}

    def test_first_half_total_independent(self):
        """1H total should be independent from game total."""
        k1 = NBAQuantAgent._market_type_key("KXNBATOTAL-26MAR11NYKUTA-226")
        k2 = NBAQuantAgent._market_type_key("KXNBA1HTOTAL-26MAR11NYKUTA-125")
        assert k1 != k2

    def test_allowed_positions_after_fix(self):
        """With the fix, only 1 TOTAL + 1 different type should be allowed."""
        from unittest.mock import MagicMock
        from core.schemas import AppSettings
        settings = AppSettings(
            KALSHI_API_KEY_ID="test",
            KALSHI_PRIVATE_KEY_PATH="/dev/null",
            KALSHI_ENV="demo",
            REDIS_URL="redis://localhost:6379",
            DATABASE_URL="postgresql://argus:argus@localhost:5432/argus",
            OPENAI_API_KEY="test",
        )
        agent = NBAQuantAgent.__new__(NBAQuantAgent)
        agent.settings = settings
        agent._pending_market_types = {}
        agent._game_to_tickers = {}
        agent._portfolio = None

        game_id = "18447767"

        # First TOTAL bet allowed
        assert not agent._is_market_type_taken(game_id, "KXNBATOTAL-26MAR11NYKUTA-226")

        # Record it
        agent._pending_market_types[game_id] = {"TOTAL"}

        # Second TOTAL bet blocked (different line, same type)
        assert agent._is_market_type_taken(game_id, "KXNBATOTAL-26MAR11NYKUTA-244")
        assert agent._is_market_type_taken(game_id, "KXNBATOTAL-26MAR11NYKUTA-235")

        # But a player prop is still allowed
        assert not agent._is_market_type_taken(game_id, "KXNBAPTS-26MAR11NYKUTA-NYKKTOWNS32-15")

        # And a spread is still allowed
        assert not agent._is_market_type_taken(game_id, "KXNBASPREAD-26MAR11NYKUTA-NYK6")

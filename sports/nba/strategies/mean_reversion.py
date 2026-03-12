"""Mean reversion strategy — buy underpriced contracts, sell on recovery.

Detects when a Kalshi contract's market price diverges significantly below
the model's fair value derived from live game data.  Works across all market
types where a fair value can be computed from game state:

  - **Player props** (pts, reb, ast, stl, blk): pace-projected stats
  - **Game totals / team totals**: pace-projected score
  - **Spreads**: projected margin from current pace

The core insight: Kalshi retail participants panic-sell when a stat line
looks cold early or a score run shifts momentum.  The model's projection
changes slowly (it is pace-based), so a rapid price drop without a
corresponding pace change is a buying opportunity.  We cash out when the
market reprices toward fair value — *not* at binary expiry.

All fair-value computations reuse the same projection math as the totals
and player-props strategies to keep the model consistent.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque

from sports.nba.strategies.base import BaseStrategy
from core.schemas import (
    Action,
    GameState,
    MarketState,
    PlayerBoxScore,
    Side,
    Signal,
    SignalStatus,
)
from core.utils import MINUTES_PER_GAME, team_minutes_played

# Bayesian model (optional — falls back to linear if not available)
try:
    from sports.nba.bayesian import (
        over_probability as bayesian_over_prob,
        project_game_total,
        project_player_stat,
        project_spread,
        project_team_total,
        GAME_TOTAL_STD_DEV,
        TEAM_TOTAL_STD_DEV,
        PLAYER_POINTS_STD_DEV as _B_PTS_STD,
        PLAYER_GENERIC_STD_DEV as _B_GEN_STD,
        SPREAD_STD_DEV,
    )
    _HAS_BAYESIAN = True
except ImportError:
    _HAS_BAYESIAN = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Player projection
_PLAYER_POINTS_STD_DEV = 6.0
_PLAYER_GENERIC_STD_DEV = 3.0  # rebounds, assists, steals, blocks
_STARTER_MINUTES_PER_GAME = 36.0
_BASELINE_USAGE_RATE = 0.20
_USAGE_BOOST_SCALE = 0.5
_USAGE_BOOST_MIN = 0.7
_USAGE_BOOST_MAX = 1.5
_MIN_PLAYER_MINUTES = 5.0
_MIN_TEAM_FGA = 10

# High-line dampening: lines above this threshold get confidence reduced.
# Scoring 25+ pts is much more volatile than 10+; the model's pace
# projection can't reliably distinguish a hot streak from sustainable pace.
_HIGH_LINE_THRESHOLD: dict[str, float] = {
    "pts": 20.0,
    "reb": 10.0,
    "ast": 8.0,
    "fg3m": 3.0,
    "stl": 2.0,
    "blk": 2.0,
}

# Player props require this multiple of min_divergence_cents to fire.
# Totals aggregate across all players (variance cancels out), but
# individual player stats are far noisier.
_PROP_DIVERGENCE_MULTIPLIER = 1.5

# Game totals
_GAME_TOTAL_STD_DEV = 15.0
_TEAM_TOTAL_STD_DEV = 9.0
_SPREAD_STD_DEV = 12.0

# Sigmoid steepness for normal CDF approximation
_SIGMOID_STEEPNESS = 1.7

# Price history for divergence detection
_PRICE_WINDOW_SECONDS = 120.0  # 2-minute lookback for recent high
_MIN_PRICE_HISTORY = 3  # need at least 3 ticks to judge

_LINE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*$")

# Stat-type keywords in Kalshi tickers → box-score attribute
# Order matters: longer/more-specific keywords first to avoid substring collisions
# (e.g. "KXNBASTL" contains "AST" so "STL" must be checked before "AST")
_STAT_TYPES: list[tuple[str, str]] = [
    ("PLAYERPTS", "pts"),
    ("3PT", "fg3m"),
    ("STL", "stl"),
    ("BLK", "blk"),
    ("PTS", "pts"),
    ("REB", "reb"),
    ("AST", "ast"),
]


class MeanReversionStrategy(BaseStrategy):
    """Buy contracts whose price has diverged below model fair value."""

    name = "mean_reversion"

    def __init__(
        self,
        min_divergence_cents: int = 8,
        exit_spread: int = 5,
        min_minutes: float = 6.0,
        quarter_multipliers: tuple[float, float, float, float] = (2.5, 1.75, 1.25, 0.75),
        season_avg_cache: object | None = None,
        sharp_book_watcher: object | None = None,
    ) -> None:
        self._min_divergence = min_divergence_cents
        self._exit_spread = exit_spread
        self._min_minutes = min_minutes
        self._quarter_multipliers = quarter_multipliers

        # Optional Bayesian data sources (injected by config.py)
        self._season_cache = season_avg_cache  # SeasonAverageCache
        self._sharp_books = sharp_book_watcher  # SharpOddsFeed

        # Rolling price history per ticker: (monotonic_ts, yes_bid)
        self._price_history: dict[str, deque[tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=500)
        )

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def can_evaluate(self, market: MarketState) -> bool:
        return market.ticker.upper().startswith("KXNBA")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        now = time.monotonic()
        ticker = market.ticker
        bid = market.yes_bid

        # Record price tick
        self._record_price(ticker, now, bid)

        if market.yes_ask <= 0 or bid <= 0:
            return None

        minutes_played = team_minutes_played(game)
        if minutes_played < self._min_minutes:
            return None

        # Compute model fair value
        fair_value = self._fair_value_cents(game, market)
        if fair_value is None:
            return None

        entry_price = market.yes_ask

        # Divergence: how far below fair value is the market?
        # Player props require a larger divergence than team-level markets
        # because individual stats are far more volatile.
        upper = market.ticker.upper()
        is_player_prop = self._detect_stat_type(upper) is not None
        min_div = self._min_divergence
        if is_player_prop:
            min_div = int(min_div * _PROP_DIVERGENCE_MULTIPLIER)

        divergence = fair_value - entry_price
        if divergence < min_div:
            return None

        # Confirm the drop is recent (price was higher in our window)
        recent_high = self._recent_high(ticker, now)
        if recent_high is not None and recent_high - bid < self._min_divergence // 2:
            return None

        # Probability is the model's fair value expressed as probability
        model_prob = min(fair_value / 100.0, 0.99)
        ev = model_prob - (entry_price / 100.0)

        if ev < 0.01:
            return None

        # Exit target: split the difference between entry and fair value
        # This gives us a realistic exit that doesn't require full reversion
        exit_price = min(entry_price + max(self._exit_spread, divergence // 2), 99)

        confidence = min(model_prob, 1.0)

        return Signal(
            ticker=ticker,
            action=Action.BUY,
            side=Side.YES,
            status=SignalStatus.VALIDATED,
            confidence=confidence,
            source=self.name,
            ev_estimate=ev,
            entry_price=entry_price,
            exit_price=exit_price,
            game_id=game.game_id,
        )

    # ------------------------------------------------------------------
    # Bailout interface
    # ------------------------------------------------------------------

    def model_probability(self, game: GameState, market: MarketState) -> float | None:
        fv = self._fair_value_cents(game, market)
        if fv is None:
            return None
        return min(fv / 100.0, 0.99)

    # ------------------------------------------------------------------
    # Fair value computation (dispatches by market type)
    # ------------------------------------------------------------------

    def _fair_value_cents(self, game: GameState, market: MarketState) -> int | None:
        """Return the model's fair-value price in cents, or None."""
        upper = market.ticker.upper()
        line = self._extract_line(market.ticker)
        if line is None:
            return None

        minutes_played = team_minutes_played(game)
        is_over = self._is_over_ticker(market.ticker)

        # --- Game totals ---
        if "TEAMTOTAL" in upper:
            return self._team_total_fair_value(game, market, line, minutes_played, is_over)
        if "TOTAL" in upper:
            return self._game_total_fair_value(game, line, minutes_played, is_over)

        # --- Spreads ---
        if "SPREAD" in upper:
            return self._spread_fair_value(game, market, line, minutes_played)

        # --- Player props (pts, reb, ast, stl, blk, 3pt) ---
        stat_attr = self._detect_stat_type(upper)
        if stat_attr is not None:
            return self._player_prop_fair_value(
                game, market, line, minutes_played, is_over, stat_attr,
            )

        return None

    # ------------------------------------------------------------------
    # Game totals
    # ------------------------------------------------------------------

    def _game_total_fair_value(
        self, game: GameState, line: float, minutes_played: float, is_over: bool,
    ) -> int | None:
        current_total = game.home_score + game.away_score
        if minutes_played <= 0:
            return None

        # Bayesian path: use sharp book line as prior
        if _HAS_BAYESIAN:
            sharp_line = None
            if self._sharp_books is not None:
                sharp_line = self._sharp_books.get_total_line(game.game_id)
            projected = project_game_total(game, sharp_line=sharp_line)
            if projected is not None:
                prob = bayesian_over_prob(projected, line, GAME_TOTAL_STD_DEV, minutes_played)
                if not is_over:
                    prob = 1.0 - prob
                return int(prob * 100)

        # Fallback: linear pace
        pace = current_total / minutes_played
        projected = pace * MINUTES_PER_GAME
        prob = self._normal_cdf_prob(projected, line, _GAME_TOTAL_STD_DEV, minutes_played)
        if not is_over:
            prob = 1.0 - prob
        return int(prob * 100)

    # ------------------------------------------------------------------
    # Team totals
    # ------------------------------------------------------------------

    def _team_total_fair_value(
        self, game: GameState, market: MarketState, line: float,
        minutes_played: float, is_over: bool,
    ) -> int | None:
        # The team abbrev is in the last segment: "...-ATL126" or "...-DAL110"
        tail = market.ticker.rsplit("-", 1)[-1].upper()
        if game.home_abbr and tail.startswith(game.home_abbr.upper()):
            score = game.home_score
        elif game.away_abbr and tail.startswith(game.away_abbr.upper()):
            score = game.away_score
        else:
            return None

        if minutes_played <= 0:
            return None

        if _HAS_BAYESIAN:
            projected = project_team_total(score, game)
            if projected is not None:
                prob = bayesian_over_prob(projected, line, TEAM_TOTAL_STD_DEV, minutes_played)
                if not is_over:
                    prob = 1.0 - prob
                return int(prob * 100)

        pace = score / minutes_played
        projected = pace * MINUTES_PER_GAME
        prob = self._normal_cdf_prob(projected, line, _TEAM_TOTAL_STD_DEV, minutes_played)
        if not is_over:
            prob = 1.0 - prob
        return int(prob * 100)

    # ------------------------------------------------------------------
    # Spreads
    # ------------------------------------------------------------------

    def _spread_fair_value(
        self, game: GameState, market: MarketState, line: float,
        minutes_played: float,
    ) -> int | None:
        # Team abbrev in last segment: "...-ATL6"
        tail = market.ticker.rsplit("-", 1)[-1].upper()
        if game.home_abbr and tail.startswith(game.home_abbr.upper()):
            current_margin = game.home_score - game.away_score
        elif game.away_abbr and tail.startswith(game.away_abbr.upper()):
            current_margin = game.away_score - game.home_score
        else:
            return None

        if minutes_played <= 0:
            return None

        if _HAS_BAYESIAN:
            sharp_line = None
            if self._sharp_books is not None:
                sharp_line = self._sharp_books.get_spread_line(game.game_id)
            projected = project_spread(current_margin, game, sharp_line=sharp_line)
            if projected is not None:
                prob = bayesian_over_prob(projected, line, SPREAD_STD_DEV, minutes_played)
                return int(prob * 100)

        margin_per_min = current_margin / minutes_played
        projected_margin = margin_per_min * MINUTES_PER_GAME
        prob = self._normal_cdf_prob(projected_margin, line, _SPREAD_STD_DEV, minutes_played)
        return int(prob * 100)

    # ------------------------------------------------------------------
    # Player props (generic across stat types)
    # ------------------------------------------------------------------

    def _player_prop_fair_value(
        self, game: GameState, market: MarketState, line: float,
        minutes_played: float, is_over: bool, stat_attr: str,
    ) -> int | None:
        if not game.player_stats:
            return None

        player = self._match_player(market.ticker, game)
        if player is None or player.minutes < _MIN_PLAYER_MINUTES:
            return None

        current_stat = getattr(player, stat_attr, None)
        if current_stat is None:
            return None

        # Try Bayesian projection with season average prior
        if _HAS_BAYESIAN:
            player_prior = None
            if self._season_cache is not None:
                player_prior = self._season_cache.get(player.player_id)
            projected = project_player_stat(player, game, stat_attr, player_prior)
            if projected is not None:
                std_dev = _B_PTS_STD if stat_attr == "pts" else _B_GEN_STD
                prob = bayesian_over_prob(projected, line, std_dev, minutes_played)
                if not is_over:
                    prob = 1.0 - prob
                # High-line dampening still applies
                high_thresh = _HIGH_LINE_THRESHOLD.get(stat_attr)
                if high_thresh is not None and line > high_thresh:
                    prob *= high_thresh / line
                return int(prob * 100)

        # Fallback: linear projection
        projected = self._project_stat(player, game, stat_attr)
        if projected is None:
            return None

        std_dev = _PLAYER_POINTS_STD_DEV if stat_attr == "pts" else _PLAYER_GENERIC_STD_DEV
        prob = self._normal_cdf_prob(projected, line, std_dev, minutes_played)
        if not is_over:
            prob = 1.0 - prob

        # Dampen confidence for high lines
        high_thresh = _HIGH_LINE_THRESHOLD.get(stat_attr)
        if high_thresh is not None and line > high_thresh:
            prob *= high_thresh / line

        return int(prob * 100)

    def _project_stat(
        self, player: PlayerBoxScore, game: GameState, stat_attr: str,
    ) -> float | None:
        """Project final stat value using minutes-based pace."""
        current_val = getattr(player, stat_attr, 0)
        if player.minutes <= 0:
            return None

        rate_per_min = current_val / player.minutes

        team_minutes = team_minutes_played(game)
        fraction_played = min(team_minutes / MINUTES_PER_GAME, 1.0) if team_minutes > 0 else 0.0
        fraction_remaining = max(1.0 - fraction_played, 0.0)
        remaining_player_minutes = _STARTER_MINUTES_PER_GAME * fraction_remaining

        # Usage boost (only for points — FGA-based)
        boost = 1.0
        if stat_attr == "pts":
            team_fga = sum(
                p.fga for p in game.player_stats if p.team_abbr == player.team_abbr
            )
            if team_fga >= _MIN_TEAM_FGA:
                usage_rate = player.fga / team_fga
                boost = 1.0 + (usage_rate - _BASELINE_USAGE_RATE) * _USAGE_BOOST_SCALE
                boost = max(min(boost, _USAGE_BOOST_MAX), _USAGE_BOOST_MIN)

        projected = current_val + (rate_per_min * remaining_player_minutes * boost)
        return projected

    # ------------------------------------------------------------------
    # Probability model
    # ------------------------------------------------------------------

    @staticmethod
    def _normal_cdf_prob(
        projected: float, line: float, base_std_dev: float, minutes_played: float,
    ) -> float:
        """P(stat > line) using sigmoid approximation to normal CDF."""
        fraction_remaining = max(1.0 - minutes_played / MINUTES_PER_GAME, 0.01)
        std_dev = base_std_dev * fraction_remaining ** 0.5

        if std_dev < 0.01:
            return 1.0 if projected > line else 0.0

        z = (projected - line) / std_dev
        return float(1.0 / (1.0 + 2.718281828 ** (-_SIGMOID_STEEPNESS * z)))

    # ------------------------------------------------------------------
    # Price tracking
    # ------------------------------------------------------------------

    def _record_price(self, ticker: str, now: float, bid: int) -> None:
        buf = self._price_history[ticker]
        buf.append((now, bid))
        # Evict old entries
        cutoff = now - _PRICE_WINDOW_SECONDS * 2
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _recent_high(self, ticker: str, now: float) -> int | None:
        """Return the highest bid in the lookback window, or None."""
        buf = self._price_history[ticker]
        cutoff = now - _PRICE_WINDOW_SECONDS
        prices = [price for ts, price in buf if ts >= cutoff]
        if len(prices) < _MIN_PRICE_HISTORY:
            return None
        return max(prices)

    # ------------------------------------------------------------------
    # Ticker parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_line(ticker: str) -> float | None:
        """Pull the numeric line from the end of a ticker."""
        # Tickers end with the line: ...-10, ...-225, ...-ATL6
        parts = ticker.rsplit("-", 1)
        if len(parts) < 2:
            return None
        tail = parts[-1]
        m = re.search(r"(\d+(?:\.\d+)?)", tail)
        return float(m.group(1)) if m else None

    @staticmethod
    def _is_over_ticker(ticker: str) -> bool:
        upper = ticker.upper()
        if "-U" in upper or "UNDER" in upper:
            return False
        return True

    @staticmethod
    def _detect_stat_type(upper_ticker: str) -> str | None:
        """Detect which stat type a ticker refers to.

        Uses the ticker prefix (e.g. KXNBAPTS, KXNBAREB) to avoid
        substring collisions like "AST" matching inside "KXNBASTL".
        """
        # Extract the market-type segment: "KXNBAPTS-..." → "PTS"
        # Ticker format: KXNBA{TYPE}-{date}{matchup}-{details}
        prefix = upper_ticker.split("-", 1)[0]  # "KXNBAPTS"
        nba_suffix = prefix.replace("KXNBA", "", 1) if prefix.startswith("KXNBA") else ""

        for keyword, attr in _STAT_TYPES:
            if nba_suffix == keyword:
                return attr
        # Fallback: check if keyword appears anywhere (for non-standard tickers)
        for keyword, attr in _STAT_TYPES:
            if keyword in upper_ticker:
                return attr
        return None

    @staticmethod
    def _match_player(ticker: str, game: GameState) -> PlayerBoxScore | None:
        """Find the player referenced in the ticker from the box score."""
        upper = ticker.upper()

        candidates: list[PlayerBoxScore] = []
        best_len = 0

        for p in game.player_stats:
            last = p.last_name.upper().replace(" ", "")
            if len(last) < 3:
                continue
            if last not in upper:
                continue

            initial_last = p.first_name[0].upper() + last if p.first_name else last
            if initial_last in upper:
                return p

            if len(last) > best_len:
                candidates = [p]
                best_len = len(last)
            elif len(last) == best_len:
                candidates.append(p)

        if len(candidates) == 1:
            return candidates[0]
        return None

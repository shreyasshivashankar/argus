"""Player prop (points) strategy using live usage-rate projection.

Projects a player's final point total from their current scoring pace
and team-level shot distribution, then compares it to the Kalshi prop
line embedded in the ticker.

Targets tickers containing a player identifier and a numeric points
line (e.g. ``KXNBA-PLAYERPTS-04MAR26-LEBRON-O28``).

The core insight: if a player is taking a disproportionate share of
their team's FGA in a live game, they are likely to end up with more
points than the pre-game line anticipated — and vice versa.
"""
from __future__ import annotations

import re

from agents.strategies.base import BaseStrategy
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

_MIN_PLAYER_MINUTES = 5.0
_MIN_TEAM_FGA = 10

# Baseline std dev for individual player point totals (points).
# Player scoring is more volatile than game totals; ~6pt over a full game
# for a typical starter, shrinking by sqrt(fraction_remaining).
_PLAYER_POINTS_STD_DEV = 6.0

# Sigmoid steepness for normal CDF approximation.
# 1.7 gives <1% error vs true Phi(z) over [-3, 3].
_SIGMOID_STEEPNESS = 1.7

# Expected starter minutes per game (used for remaining-minutes projection).
_STARTER_MINUTES_PER_GAME = 36.0

# Baseline usage rate; deviations scale the usage boost factor.
_BASELINE_USAGE_RATE = 0.20
_USAGE_BOOST_SCALE = 0.5
_USAGE_BOOST_MIN = 0.7
_USAGE_BOOST_MAX = 1.5

_LINE_RE = re.compile(r"[OU](\d+(?:\.\d+)?)", re.IGNORECASE)

_PLAYER_NAME_NORMALIZATIONS: dict[str, str] = {}


class PlayerPropStrategy(BaseStrategy):
    name = "player_props"

    def __init__(
        self,
        ev_threshold: float = 0.03,
        target_exit_spread: int = 7,
        quarter_multipliers: tuple[float, float, float, float] = (2.5, 1.75, 1.25, 0.75),
    ) -> None:
        self._ev_threshold = ev_threshold
        self._target_exit_spread = target_exit_spread
        self._quarter_multipliers = quarter_multipliers

    def can_evaluate(self, market: MarketState) -> bool:
        """Match tickers that look like player-points props.

        Kalshi player-points tickers typically contain ``PLAYERPTS`` or
        ``PTS`` with a player name segment. We avoid matching team-level
        totals by excluding ``TOTAL`` and ``GAME``.
        """
        upper = market.ticker.upper()
        if "GAME" in upper or "TOTAL" in upper:
            return False
        return "PLAYERPTS" in upper or "PTS" in upper

    def evaluate(self, game: GameState, market: MarketState) -> Signal | None:
        if market.yes_ask <= 0:
            return None

        if not game.player_stats:
            return None

        line = self._extract_line(market.ticker)
        if line is None:
            return None

        player = self._match_player(market.ticker, game)
        if player is None:
            return None

        if player.minutes < _MIN_PLAYER_MINUTES:
            return None

        team_minutes = team_minutes_played(game)

        # Late-Game Flyer Filter: ban cheap longshots in the first half (< 24 min elapsed).
        # Prevents gambling on e.g. "Durant 35+ pts" at 10c with 24 minutes of variance left.
        if team_minutes < 24.0 and market.yes_ask < 35:
            return None

        # Ban extremely cheap longshots at all times — Kelly sizing misbehaves below 15c.
        if market.yes_ask < 15:
            return None

        projected_pts = self._project_points(player, game)
        if projected_pts is None:
            return None

        is_over = self._is_over_ticker(market.ticker)
        minutes_played = team_minutes
        if is_over:
            model_prob = self._over_probability(projected_pts, line, minutes_played)
        else:
            model_prob = 1.0 - self._over_probability(projected_pts, line, minutes_played)

        entry_price_cents = market.yes_ask
        ev = model_prob * 1.0 - (entry_price_cents / 100.0)

        threshold = self.time_adjusted_ev_threshold(
            self._ev_threshold, game, self._quarter_multipliers,
        )
        if ev < threshold:
            return None

        exit_price = min(entry_price_cents + self._target_exit_spread, 99)

        return Signal(
            ticker=market.ticker,
            action=Action.BUY,
            side=Side.YES,
            status=SignalStatus.VALIDATED,
            confidence=min(model_prob, 1.0),
            source=self.name,
            ev_estimate=ev,
            entry_price=entry_price_cents,
            exit_price=exit_price,
            game_id=game.game_id,
        )

    # ------------------------------------------------------------------
    # Player matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_player(
        ticker: str, game: GameState
    ) -> PlayerBoxScore | None:
        """Find the player referenced in the ticker from the box score.

        Kalshi tickers embed a player name — typically first initial +
        last name (e.g. ``JWILLIAMS``) or just the last name. We first
        try initial+last for an exact disambiguated match, then fall
        back to longest last-name substring. If multiple players share
        a last name and the ticker lacks an initial prefix, we return
        None to avoid misattribution (the "Williams problem").
        """
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

    # ------------------------------------------------------------------
    # Projection model
    # ------------------------------------------------------------------

    @staticmethod
    def _project_points(player: PlayerBoxScore, game: GameState) -> float | None:
        """Project final points using usage-weighted pace.

        1. Calculate player's FGA share of their team's total FGA.
        2. Compute the player's current scoring rate (pts / min).
        3. Scale by expected remaining minutes (proportional to team
           minutes remaining) weighted by the player's usage share.

        This naturally captures hot/cold shooting, foul trouble
        (fewer minutes → fewer projected points), and tactical
        load shifts during games.
        """
        team_fga = 0
        for p in game.player_stats:
            if p.team_abbr == player.team_abbr:
                team_fga += p.fga

        if team_fga < _MIN_TEAM_FGA:
            return None

        usage_rate = player.fga / team_fga if team_fga > 0 else 0.0

        if player.minutes <= 0:
            return None

        pts_per_minute = player.pts / player.minutes

        team_minutes = team_minutes_played(game)
        fraction_played = min(team_minutes / MINUTES_PER_GAME, 1.0) if team_minutes > 0 else 0.0
        fraction_remaining = max(1.0 - fraction_played, 0.0)

        remaining_player_minutes = _STARTER_MINUTES_PER_GAME * fraction_remaining

        usage_boost = 1.0 + (usage_rate - _BASELINE_USAGE_RATE) * _USAGE_BOOST_SCALE
        usage_boost = max(min(usage_boost, _USAGE_BOOST_MAX), _USAGE_BOOST_MIN)

        projected = player.pts + (pts_per_minute * remaining_player_minutes * usage_boost)
        return projected

    # ------------------------------------------------------------------
    # Probability model
    # ------------------------------------------------------------------

    @staticmethod
    def _over_probability(
        projected: float,
        line: float,
        minutes_played: float,
    ) -> float:
        """P(player finishes over the line) using normal approximation.

        Player-level variance is higher than game totals, so we use a
        wider standard deviation that shrinks as the game progresses.
        """
        fraction_remaining = max(1.0 - minutes_played / MINUTES_PER_GAME, 0.01)
        std_dev = _PLAYER_POINTS_STD_DEV * fraction_remaining ** 0.5

        if std_dev < 0.01:
            return 1.0 if projected > line else 0.0

        z = (projected - line) / std_dev
        return float(1.0 / (1.0 + 2.718281828 ** (-_SIGMOID_STEEPNESS * z)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_line(ticker: str) -> float | None:
        m = _LINE_RE.search(ticker)
        return float(m.group(1)) if m else None

    @staticmethod
    def _is_over_ticker(ticker: str) -> bool:
        upper = ticker.upper()
        if "-O" in upper or "OVER" in upper:
            return True
        if "-U" in upper or "UNDER" in upper:
            return False
        return True


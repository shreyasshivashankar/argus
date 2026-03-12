"""Bayesian projection model for NBA stats.

Replaces naive linear pace projection with a prior-weighted Bayesian
update.  Early in the game the prior (season averages) dominates;
as live data accumulates, the posterior shifts toward observed pace.

The key insight: a player with a season average of 16 pts/game who
scores 10 pts in 12 minutes is NOT on pace for 40 pts.  The Bayesian
model correctly blends the prior (16 pts) with the live pace (~40 pts)
based on sample size, producing a more realistic projection (~22 pts).

For game totals, the prior comes from the pre-game line (if available
from sharp books) or a league-average default.
"""
from __future__ import annotations

import math

from core.schemas import GameState, PlayerBoxScore
from core.utils import MINUTES_PER_GAME, team_minutes_played
from sports.nba.season_averages import PlayerPrior

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many live minutes of data before the posterior fully trusts live pace.
# At PRIOR_HALF_LIFE minutes, prior and live are weighted 50/50.
PRIOR_HALF_LIFE_MINUTES = 16.0

# Default prior for game totals when no sharp book line is available.
# ~225 is the 2024-25 NBA average total.
DEFAULT_GAME_TOTAL_PRIOR = 225.0

# Default per-team prior (half of game total)
DEFAULT_TEAM_TOTAL_PRIOR = 112.5

# Starter minutes per game assumption
_STARTER_MINUTES = 36.0

# Minimum live minutes before we use any live data at all
_MIN_LIVE_MINUTES = 2.0

# Std dev constants (same as mean_reversion but can diverge as model matures)
GAME_TOTAL_STD_DEV = 15.0
TEAM_TOTAL_STD_DEV = 9.0
PLAYER_POINTS_STD_DEV = 6.0
PLAYER_GENERIC_STD_DEV = 3.0
SPREAD_STD_DEV = 12.0

# Sigmoid steepness for normal CDF approximation
_SIGMOID_K = 1.7


# ---------------------------------------------------------------------------
# Core: Bayesian blending weight
# ---------------------------------------------------------------------------

def prior_weight(live_minutes: float) -> float:
    """Return the weight given to the prior (0 to 1).

    Uses an exponential decay based on live minutes observed:
        w_prior = 2^(-live_minutes / half_life)

    At 0 min: w_prior = 1.0 (100% prior)
    At 16 min: w_prior = 0.5 (50/50)
    At 32 min: w_prior = 0.25 (mostly live)
    At 48 min: w_prior = 0.125 (almost entirely live)
    """
    if live_minutes < _MIN_LIVE_MINUTES:
        return 1.0
    return math.pow(2.0, -live_minutes / PRIOR_HALF_LIFE_MINUTES)


# ---------------------------------------------------------------------------
# Player stat projection
# ---------------------------------------------------------------------------

def project_player_stat(
    player: PlayerBoxScore,
    game: GameState,
    stat_attr: str,
    player_prior: PlayerPrior | None = None,
) -> float | None:
    """Project a player's final stat using Bayesian blending.

    If a season-average prior is available, blends it with live pace.
    Falls back to pure live pace if no prior (same as old model).
    """
    current_val = getattr(player, stat_attr, 0)
    if player.minutes <= 0:
        return None

    team_minutes = team_minutes_played(game)
    fraction_played = min(team_minutes / MINUTES_PER_GAME, 1.0)
    fraction_remaining = max(1.0 - fraction_played, 0.0)
    remaining_player_minutes = _STARTER_MINUTES * fraction_remaining

    # Live pace projection
    live_rate = current_val / player.minutes
    live_projected = current_val + live_rate * remaining_player_minutes

    # If no prior, return live projection (backward compatible)
    if player_prior is None:
        return live_projected

    # Prior projection: season rate × expected total minutes
    prior_rate_map = {
        "pts": player_prior.pts_per_min,
        "reb": player_prior.reb_per_min,
        "ast": player_prior.ast_per_min,
        "stl": player_prior.stl_per_min,
        "blk": player_prior.blk_per_min,
        "fg3m": player_prior.fg3m_per_min,
    }
    prior_rate = prior_rate_map.get(stat_attr)
    if prior_rate is None:
        return live_projected

    # Prior: expected total from season averages
    expected_total_minutes = player_prior.avg_minutes
    prior_projected = prior_rate * expected_total_minutes

    # Bayesian blend
    w_prior = prior_weight(player.minutes)
    w_live = 1.0 - w_prior

    projected = w_prior * prior_projected + w_live * live_projected
    return projected


# ---------------------------------------------------------------------------
# Game total projection
# ---------------------------------------------------------------------------

def project_game_total(
    game: GameState,
    sharp_line: float | None = None,
) -> float | None:
    """Project final combined score using Bayesian blending.

    Prior: sharp book line (if available) or league average.
    Live: current pace extrapolation.
    """
    minutes_played = team_minutes_played(game)
    if minutes_played <= 0:
        return None

    current_total = game.home_score + game.away_score
    live_projected = (current_total / minutes_played) * MINUTES_PER_GAME

    prior = sharp_line if sharp_line is not None else DEFAULT_GAME_TOTAL_PRIOR

    w_prior = prior_weight(minutes_played)
    w_live = 1.0 - w_prior

    return w_prior * prior + w_live * live_projected


# ---------------------------------------------------------------------------
# Team total projection
# ---------------------------------------------------------------------------

def project_team_total(
    team_score: int,
    game: GameState,
    sharp_line: float | None = None,
) -> float | None:
    """Project final score for one team."""
    minutes_played = team_minutes_played(game)
    if minutes_played <= 0:
        return None

    live_projected = (team_score / minutes_played) * MINUTES_PER_GAME
    prior = sharp_line if sharp_line is not None else DEFAULT_TEAM_TOTAL_PRIOR

    w_prior = prior_weight(minutes_played)
    w_live = 1.0 - w_prior

    return w_prior * prior + w_live * live_projected


# ---------------------------------------------------------------------------
# Spread projection
# ---------------------------------------------------------------------------

def project_spread(
    margin: int,
    game: GameState,
    sharp_line: float | None = None,
) -> float | None:
    """Project final margin for spread bets."""
    minutes_played = team_minutes_played(game)
    if minutes_played <= 0:
        return None

    live_projected = (margin / minutes_played) * MINUTES_PER_GAME
    prior = sharp_line if sharp_line is not None else 0.0  # no prior = even game

    w_prior = prior_weight(minutes_played)
    w_live = 1.0 - w_prior

    return w_prior * prior + w_live * live_projected


# ---------------------------------------------------------------------------
# Probability (shared normal CDF approximation)
# ---------------------------------------------------------------------------

def over_probability(
    projected: float,
    line: float,
    base_std_dev: float,
    minutes_played: float,
) -> float:
    """P(stat > line) using sigmoid approximation to normal CDF."""
    fraction_remaining = max(1.0 - minutes_played / MINUTES_PER_GAME, 0.01)
    std_dev = base_std_dev * math.sqrt(fraction_remaining)

    if std_dev < 0.01:
        return 1.0 if projected > line else 0.0

    z = (projected - line) / std_dev
    return 1.0 / (1.0 + math.exp(-_SIGMOID_K * z))

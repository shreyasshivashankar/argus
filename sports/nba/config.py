"""NBA strategy configuration and toggles.

Reads ``NBA_ENABLED_STRATEGIES`` from AppSettings to decide which
strategies the quant agent should instantiate. Strategies not in the
list are silently skipped.
"""
from __future__ import annotations

from core.schemas import AppSettings
from sports.nba.strategies.base import BaseStrategy


# Canonical strategy names (must match each strategy's ``name`` attribute).
STRATEGY_ARBITRAGE = "arbitrage"
STRATEGY_TOTALS = "totals"
STRATEGY_PLAYER_PROPS = "player_props"
STRATEGY_FLASH_CRASH = "flash_crash"
STRATEGY_MEAN_REVERSION = "mean_reversion"

ALL_STRATEGIES = [
    STRATEGY_ARBITRAGE,
    STRATEGY_TOTALS,
    STRATEGY_PLAYER_PROPS,
    STRATEGY_FLASH_CRASH,
    STRATEGY_MEAN_REVERSION,
]


def build_strategies(settings: AppSettings) -> list[BaseStrategy]:
    """Instantiate only the strategies enabled in settings."""
    from sports.nba.strategies.arbitrage import ArbitrageStrategy
    from sports.nba.strategies.flash_crash import FlashCrashStrategy
    from sports.nba.strategies.mean_reversion import MeanReversionStrategy
    from sports.nba.strategies.player_props import PlayerPropStrategy
    from sports.nba.strategies.totals import TotalsStrategy

    enabled = {s.strip().lower() for s in settings.NBA_ENABLED_STRATEGIES.split(",")}

    ev_base = settings.BASE_EV_THRESHOLD / 100.0
    q_mults = (
        settings.EV_Q1_MULTIPLIER,
        settings.EV_Q2_MULTIPLIER,
        settings.EV_Q3_MULTIPLIER,
        settings.EV_Q4_MULTIPLIER,
    )

    registry: dict[str, BaseStrategy] = {
        STRATEGY_ARBITRAGE: ArbitrageStrategy(
            max_combined_cents=settings.ARB_MAX_COMBINED_CENTS,
        ),
        STRATEGY_TOTALS: TotalsStrategy(
            ev_threshold=ev_base,
            target_exit_spread=settings.TARGET_EXIT_SPREAD,
            quarter_multipliers=q_mults,
            min_minutes=settings.TOTALS_MIN_MINUTES,
        ),
        STRATEGY_PLAYER_PROPS: PlayerPropStrategy(
            ev_threshold=ev_base,
            target_exit_spread=settings.TARGET_EXIT_SPREAD,
            quarter_multipliers=q_mults,
        ),
        STRATEGY_FLASH_CRASH: FlashCrashStrategy(
            window_seconds=settings.FLASH_CRASH_WINDOW_SECONDS,
            drop_threshold_cents=settings.FLASH_CRASH_DROP_CENTS,
            exit_spread=settings.FLASH_CRASH_EXIT_SPREAD,
            min_price_cents=settings.FLASH_CRASH_MIN_PRICE,
            score_delta_limit=settings.FLASH_CRASH_SCORE_DELTA_LIMIT,
        ),
        STRATEGY_MEAN_REVERSION: MeanReversionStrategy(
            min_divergence_cents=settings.MEAN_REVERSION_MIN_DIVERGENCE_CENTS,
            exit_spread=settings.MEAN_REVERSION_EXIT_SPREAD,
            min_minutes=settings.MEAN_REVERSION_MIN_MINUTES,
            quarter_multipliers=q_mults,
        ),
    }

    strategies = [s for name, s in registry.items() if name in enabled]
    return strategies

"""Standalone NBA runner — run the NBA bot independently.

Usage:
    python -m sports.nba.runner --env demo --paper
    python -m sports.nba.runner --env prod
    python -m sports.nba.runner --env prod --track
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal as _signal
import sys

from loguru import logger

from agents.executor import OrderExecutor
from agents.narrative import NarrativeAgent
from agents.paper_executor import PaperExecutor
from agents.track_agent import TrackAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings
from sports.nba.feed import BallDontLieFeed
from sports.nba.quant import NBAQuantAgent
from sports.nba.season_averages import SeasonAverageCache
from watchers.kalshi_feed import KalshiFeedWatcher
from watchers.sharp_odds_feed import SharpOddsFeed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argus NBA Trading Bot")
    parser.add_argument(
        "--env",
        choices=["demo", "prod"],
        default=None,
        help="Kalshi environment: 'demo' or 'prod'. Overrides KALSHI_ENV in .env",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading mode (simulated matching, no real orders)",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable trade tracker (persists trades to Postgres)",
    )
    return parser.parse_args()


async def main(
    paper_mode: bool = False,
    env_override: str | None = None,
    track_mode: bool = False,
) -> None:
    if env_override:
        os.environ["KALSHI_ENV"] = env_override

    settings = AppSettings()  # type: ignore[call-arg]

    mode_label = "PAPER" if paper_mode else "LIVE"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/nba.log", rotation="50 MB", retention="2 days", level="DEBUG", enqueue=True)
    logger.info("NBA bot starting — env={}, mode={}", settings.KALSHI_ENV, mode_label)

    # --- Core infrastructure ---
    bus = SignalBus(settings.REDIS_URL)
    client = KalshiAsyncClient(settings)

    # --- Watchers ---
    sports_feed = BallDontLieFeed(settings, bus)
    logger.info("Using BallDontLieFeed (GOAT, ~500 req/min)")
    kalshi_feed = KalshiFeedWatcher(client, bus)

    # --- Bayesian data sources ---
    season_cache = SeasonAverageCache(settings)
    sharp_odds = SharpOddsFeed(settings, bus)
    if settings.SHARPAPI_KEY:
        logger.info("SharpAPI Pinnacle odds feed enabled (poll {}s)", settings.SHARP_ODDS_POLL_INTERVAL)
    else:
        logger.info("SharpAPI disabled (no API key) — using default Bayesian priors")

    # --- Agents ---
    nba_quant = NBAQuantAgent(
        settings, bus, client,
        season_avg_cache=season_cache,
        sharp_book_watcher=sharp_odds,
    )
    narrative = NarrativeAgent(settings, bus, client)

    if paper_mode:
        executor = PaperExecutor(settings, bus, client)
        logger.info("Paper trading mode: simulated matching engine active")
    else:
        executor = OrderExecutor(settings, bus, client)
        kalshi_feed.on_fill(executor.on_fill)
        kalshi_feed.on_order_update(executor.on_order_update)

    tracker: TrackAgent | None = None
    if track_mode:
        tracker = TrackAgent(settings, bus, client, paper_mode=paper_mode)
        logger.info("Trade tracker enabled (is_paper={})", paper_mode)

    # --- Graceful shutdown ---
    shutdown_event = asyncio.Event()

    def _shutdown(sig: int) -> None:
        logger.warning("Received signal {}, shutting down…", sig)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    # --- Launch all concurrently ---
    tasks = [
        asyncio.create_task(sports_feed.run(), name="sports_feed"),
        asyncio.create_task(kalshi_feed.run(), name="kalshi_feed"),
        asyncio.create_task(sharp_odds.run(), name="sharp_odds"),
        asyncio.create_task(nba_quant.start(), name="nba_quant"),
        asyncio.create_task(narrative.start(), name="narrative"),
        asyncio.create_task(executor.start(), name="executor"),
    ]
    if tracker:
        tasks.append(asyncio.create_task(tracker.start(), name="track"))

    await shutdown_event.wait()

    # --- Teardown ---
    logger.info("Cancelling all tasks…")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if not paper_mode and isinstance(executor, OrderExecutor):
        try:
            exits_left = await executor._cancel_resting_entries()
            if exits_left:
                logger.info(
                    "{} exit order(s) left resting on Kalshi — they can still fill",
                    exits_left,
                )
        except Exception:
            logger.exception("Error cancelling resting orders on shutdown")

    sports_feed.stop()
    kalshi_feed.stop()
    sharp_odds.stop()
    nba_quant.stop()
    narrative.stop()
    executor.stop()
    if tracker:
        tracker.stop()

    await season_cache.close()
    await client.close()
    await bus.close()

    logger.info("NBA bot shut down cleanly")


if __name__ == "__main__":
    args = parse_args()
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    asyncio.run(
        main(paper_mode=args.paper, env_override=args.env, track_mode=args.track)
    )

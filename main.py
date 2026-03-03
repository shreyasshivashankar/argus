"""Argus — Production Kalshi Trading Bot

Entrypoint that wires all components together and runs them concurrently.

Usage:
    python main.py              # Production mode (real Kalshi orders)
    python main.py --paper      # Paper trading mode (simulated matching engine)
"""
from __future__ import annotations

import argparse
import asyncio
import signal as _signal
import sys

from loguru import logger

from agents.executor import OrderExecutor
from agents.narrative import NarrativeAgent
from agents.nba_quant import NBAQuantAgent
from agents.paper_executor import PaperExecutor
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings
from watchers.kalshi_feed import KalshiFeedWatcher
from watchers.sports_feed import APISportsFeed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Argus Kalshi Trading Bot")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (simulated matching engine, no real orders)",
    )
    return parser.parse_args()


async def main(paper_mode: bool = False) -> None:
    mode_label = "PAPER" if paper_mode else "LIVE"

    # --- Configuration ---
    settings = AppSettings()  # type: ignore[call-arg]

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/argus.log", rotation="50 MB", retention="7 days", level="DEBUG", enqueue=True)
    logger.info("Argus starting — env={}, mode={}", settings.KALSHI_ENV, mode_label)

    # --- Core infrastructure ---
    bus = SignalBus(settings.REDIS_URL)
    client = KalshiAsyncClient(settings)

    # --- Watchers ---
    sports_feed = APISportsFeed(settings, bus)
    kalshi_feed = KalshiFeedWatcher(client, bus)

    # --- Agents ---
    nba_quant = NBAQuantAgent(settings, bus, client)
    narrative = NarrativeAgent(settings, bus, client)

    if paper_mode:
        executor = PaperExecutor(settings, bus, client)
        logger.info("Paper trading mode: simulated matching engine active")
    else:
        executor = OrderExecutor(settings, bus, client)
        kalshi_feed.on_fill(executor.on_fill)
        kalshi_feed.on_order_update(executor.on_order_update)

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
        asyncio.create_task(nba_quant.start(), name="nba_quant"),
        asyncio.create_task(narrative.start(), name="narrative"),
        asyncio.create_task(executor.start(), name="executor"),
    ]

    # Wait for shutdown signal
    await shutdown_event.wait()

    # --- Teardown ---
    logger.info("Cancelling all tasks…")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if not paper_mode and isinstance(executor, OrderExecutor):
        try:
            await executor._cancel_all_resting()
        except Exception:
            logger.exception("Error cancelling resting orders on shutdown")

    sports_feed.stop()
    kalshi_feed.stop()
    nba_quant.stop()
    narrative.stop()
    executor.stop()

    await client.close()
    await bus.close()

    logger.info("Argus shut down cleanly")


if __name__ == "__main__":
    args = parse_args()
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main(paper_mode=args.paper))

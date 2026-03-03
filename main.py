"""Argus — Production Kalshi Trading Bot

Entrypoint that wires all components together and runs them concurrently.
"""
from __future__ import annotations

import asyncio
import signal as _signal
import sys

from loguru import logger

from agents.executor import OrderExecutor
from agents.narrative import NarrativeAgent
from agents.nba_quant import NBAQuantAgent
from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings
from watchers.kalshi_feed import KalshiFeedWatcher
from watchers.sports_feed import APISportsFeed


async def main() -> None:
    # --- Configuration ---
    settings = AppSettings()  # type: ignore[call-arg]

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/argus.log", rotation="50 MB", retention="7 days", level="DEBUG", enqueue=True)
    logger.info("Argus starting — env={}", settings.KALSHI_ENV)

    # --- Core infrastructure ---
    bus = SignalBus(settings.REDIS_URL)
    client = KalshiAsyncClient(settings)

    # --- Watchers ---
    sports_feed = APISportsFeed(settings, bus)
    kalshi_feed = KalshiFeedWatcher(client, bus)

    # --- Agents ---
    nba_quant = NBAQuantAgent(settings, bus, client)
    narrative = NarrativeAgent(settings, bus, client)
    executor = OrderExecutor(settings, bus, client)

    # Wire Kalshi WS fill/order callbacks to the executor
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

    # Cancel resting orders before exit
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
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    asyncio.run(main())

from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from datetime import datetime

from loguru import logger

from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings


class BaseAgent(ABC):
    """Abstract base class for all Argus agents.

    Provides:
    - Structured logging (loguru) with per-agent file sinks
    - Heartbeat publishing to signal:heartbeat
    - uvloop installation on start
    - Concurrent run of heartbeat + agent-specific run()

    Subclass and implement ``async def run(self)`` with the agent's main loop.
    """

    def __init__(
        self,
        name: str,
        settings: AppSettings,
        bus: SignalBus,
        client: KalshiAsyncClient,
    ) -> None:
        self.name = name
        self.settings = settings
        self.bus = bus
        self.client = client
        self._running = True

        self._setup_logging()

    def _setup_logging(self) -> None:
        logger.add(
            f"logs/{self.name}.log",
            rotation="50 MB",
            retention="2 days",
            level="DEBUG",
            filter=lambda record: record["extra"].get("agent") == self.name,
            enqueue=True,
        )
        self.log = logger.bind(agent=self.name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self) -> None:
        """Agent-specific main loop. Must be implemented by subclasses."""
        ...

    def _heartbeat_payload(self) -> dict:
        """Override in subclasses to add extra fields (e.g. api_ok)."""
        return {"agent": self.name, "ts": datetime.utcnow().isoformat()}

    async def heartbeat(self) -> None:
        while self._running:
            await self.bus.publish("signal:heartbeat", self._heartbeat_payload())
            await asyncio.sleep(self.settings.HEARTBEAT_INTERVAL)

    async def start(self) -> None:
        """Install uvloop (if available) and run heartbeat + run() concurrently."""
        self._install_uvloop()
        self.log.info("Starting agent: {}", self.name)
        try:
            await asyncio.gather(self.heartbeat(), self.run())
        except asyncio.CancelledError:
            self.log.info("Agent {} cancelled", self.name)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # uvloop
    # ------------------------------------------------------------------

    @staticmethod
    def _install_uvloop() -> None:
        if "uvloop" in sys.modules or sys.platform == "win32":
            return
        try:
            import uvloop
            uvloop.install()
        except ImportError:
            logger.debug("uvloop not available, using default event loop")

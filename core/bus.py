from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Optional

import redis.asyncio as aioredis
from loguru import logger
from pydantic import BaseModel

from core.schemas import ContextStatus


class SignalBus:
    """Async Redis wrapper providing Pub/Sub channels and a key-value context cache.

    Pub/Sub channels (event-driven):
        game:state, market:state, signal:validated, signal:executed, signal:heartbeat

    Context cache (synchronous reads):
        game:context:{game_id} -> "SAFE" | "VETO:{reason}"  (with TTL)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._pubsub: Optional[aioredis.client.PubSub] = None

    async def close(self) -> None:
        if self._pubsub:
            await self._pubsub.close()
        await self._redis.aclose()

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish(self, channel: str, message: BaseModel | dict) -> None:
        payload = message.model_dump_json() if isinstance(message, BaseModel) else json.dumps(message)
        await self._redis.publish(channel, payload)

    async def subscribe(
        self,
        channels: list[str],
        callback: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Subscribe to one or more channels and dispatch parsed messages to *callback*.

        The callback signature is ``async def handler(channel: str, data: dict)``.
        This method runs forever and should be launched as an asyncio task.
        """
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*channels)
        logger.info("Subscribed to channels: {}", channels)

        background_tasks: set[asyncio.Task] = set()

        try:
            async for raw_message in pubsub.listen():
                if raw_message["type"] != "message":
                    continue
                channel = raw_message["channel"]
                try:
                    data = json.loads(raw_message["data"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Malformed message on {}: {}", channel, raw_message["data"])
                    continue
                task = asyncio.create_task(callback(channel, data))
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
        finally:
            await pubsub.unsubscribe(*channels)
            await pubsub.close()

    # ------------------------------------------------------------------
    # Context Cache (game:context:{game_id})
    # ------------------------------------------------------------------

    async def set_context(
        self, game_id: str, status: ContextStatus, reason: str = "", ttl: int = 300
    ) -> None:
        key = f"game:context:{game_id}"
        value = status.value if status == ContextStatus.SAFE else f"VETO:{reason}"
        await self._redis.set(key, value, ex=ttl)
        logger.debug("Context set: {} = {} (TTL={}s)", key, value, ttl)

    async def get_context(self, game_id: str) -> tuple[ContextStatus, str]:
        """Read context for a game. Returns (status, reason).

        INVARIANT (fail-close): if the key is missing or expired, returns
        (VETO, "context missing – fail-close"). The system never trades blind.
        """
        key = f"game:context:{game_id}"
        raw: Optional[str] = await self._redis.get(key)

        if raw is None:
            return ContextStatus.VETO, "context missing – fail-close"

        if raw == ContextStatus.SAFE:
            return ContextStatus.SAFE, ""

        if raw.startswith("VETO:"):
            return ContextStatus.VETO, raw[5:]

        return ContextStatus.VETO, f"unexpected cache value: {raw}"

    # ------------------------------------------------------------------
    # Raw Redis access (for direct key ops)
    # ------------------------------------------------------------------

    @property
    def redis(self) -> aioredis.Redis:
        return self._redis

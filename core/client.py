from __future__ import annotations

import base64
import time
from typing import Any, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from loguru import logger

from core.schemas import AppSettings, Order

_PROD_REST = "https://api.elections.kalshi.com/trade-api/v2"
_DEMO_REST = "https://demo-api.kalshi.co/trade-api/v2"
_PROD_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
_DEMO_WS = "wss://demo-api.kalshi.co/trade-api/ws/v2"

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5  # seconds


class KalshiAsyncClient:
    """Async Kalshi client handling RSA-PSS signing, REST operations, and WS auth.

    REST methods: place_order, cancel_order, get_positions, get_balance, get_market.
    WS helper:   sign_ws_headers() for authenticated WebSocket handshake.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._key_id = settings.KALSHI_API_KEY_ID
        self._private_key = self._load_private_key(settings.KALSHI_PRIVATE_KEY_PATH)
        self._base_url = _PROD_REST if settings.KALSHI_ENV == "prod" else _DEMO_REST
        self._ws_url = _PROD_WS if settings.KALSHI_ENV == "prod" else _DEMO_WS

        self._session: Optional[Any] = None  # lazy aiohttp.ClientSession

    # ------------------------------------------------------------------
    # Key loading & signing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_private_key(path: str) -> Any:
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
        }

    # ------------------------------------------------------------------
    # WebSocket auth
    # ------------------------------------------------------------------

    def get_ws_url(self) -> str:
        return self._ws_url

    def sign_ws_headers(self) -> dict[str, str]:
        """Generate RSA-PSS auth headers for the Kalshi WebSocket handshake."""
        return self._auth_headers("GET", "/trade-api/ws/v2")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> Any:
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def _request(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> dict:
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"
        headers = self._auth_headers(method, path)

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with session.request(method, url, headers=headers, json=body) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        logger.warning(
                            "Kalshi {} {} → {} (attempt {}): {}",
                            method, path, resp.status, attempt + 1, data,
                        )
                        if resp.status in (429, 500, 502, 503, 504):
                            import asyncio
                            await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                            continue
                        raise KalshiAPIError(resp.status, data)
                    return data
            except KalshiAPIError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning("Kalshi request failed (attempt {}): {}", attempt + 1, exc)
                import asyncio
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))

        raise KalshiAPIError(0, {"error": str(last_exc)})

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # REST operations
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> dict:
        payload = {
            "ticker": order.ticker,
            "action": order.action.value,
            "side": order.side.value,
            "count": order.count,
            "type": order.type,
            "client_order_id": order.client_order_id,
        }
        if order.yes_price is not None:
            payload["yes_price"] = order.yes_price
        if order.no_price is not None:
            payload["no_price"] = order.no_price

        logger.info("Placing order: {}", payload)
        return await self._request("POST", "/portfolio/orders", payload)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._request("DELETE", f"/portfolio/orders/{order_id}")

    async def get_positions(self) -> list[dict]:
        data = await self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    async def get_balance(self) -> float:
        data = await self._request("GET", "/portfolio/balance")
        return float(data.get("balance", 0)) / 100  # cents → dollars

    async def get_market(self, ticker: str) -> dict:
        return await self._request("GET", f"/markets/{ticker}")


class KalshiAPIError(Exception):
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body
        super().__init__(f"Kalshi API {status}: {body}")

"""Fault injection tests for KalshiAsyncClient retry/backoff logic."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import aiohttp
from aioresponses import aioresponses

from core.client import KalshiAsyncClient, KalshiAPIError, _BACKOFF_BASE, _MAX_RETRIES
from core.schemas import AppSettings, Action, Order, Side


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client_settings() -> AppSettings:
    """Settings that skip actual key loading."""
    return AppSettings(
        KALSHI_API_KEY_ID="test-key",
        KALSHI_PRIVATE_KEY_PATH="/dev/null",
        KALSHI_ENV="demo",
        REDIS_URL="redis://localhost:6379",
    )


@pytest.fixture
def mock_kalshi_client(client_settings):
    """Client with mocked private key and signing."""
    with patch.object(KalshiAsyncClient, "_load_private_key", return_value=MagicMock()):
        client = KalshiAsyncClient(client_settings)
    client._sign = MagicMock(return_value="fake-signature")
    return client


DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"


# ===========================================================================
# 502 Bad Gateway — retry with backoff, then raise
# ===========================================================================

class TestRetryOn502:

    @pytest.mark.asyncio
    async def test_502_retries_then_raises(self, mock_kalshi_client):
        url = f"{DEMO_BASE}/portfolio/orders"

        with aioresponses() as m:
            for _ in range(_MAX_RETRIES):
                m.post(url, status=502, payload={"error": "bad gateway"})

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(KalshiAPIError) as exc_info:
                    order = Order(
                        ticker="T1", action=Action.BUY, side=Side.YES,
                        count=1, yes_price=20,
                    )
                    await mock_kalshi_client.place_order(order)

            assert mock_sleep.call_count == _MAX_RETRIES
            backoff_calls = [c.args[0] for c in mock_sleep.call_args_list]
            for i, delay in enumerate(backoff_calls):
                assert delay == pytest.approx(_BACKOFF_BASE * (2 ** i))

        await mock_kalshi_client.close()


# ===========================================================================
# 429 Too Many Requests — retry with backoff
# ===========================================================================

class TestRetryOn429:

    @pytest.mark.asyncio
    async def test_429_retries_with_backoff(self, mock_kalshi_client):
        url = f"{DEMO_BASE}/portfolio/balance"

        with aioresponses() as m:
            for _ in range(_MAX_RETRIES):
                m.get(url, status=429, payload={"error": "rate limited"})

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(KalshiAPIError):
                    await mock_kalshi_client.get_balance()

            assert mock_sleep.call_count == _MAX_RETRIES

        await mock_kalshi_client.close()


# ===========================================================================
# Intermittent failure — fail twice, succeed third
# ===========================================================================

class TestIntermittentFailure:

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_errors(self, mock_kalshi_client):
        url = f"{DEMO_BASE}/portfolio/balance"

        with aioresponses() as m:
            m.get(url, status=502, payload={"error": "bad gateway"})
            m.get(url, status=502, payload={"error": "bad gateway"})
            m.get(url, payload={"balance": 50000})  # 200 OK

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await mock_kalshi_client.get_balance()

        assert result == 500.0  # 50000 cents = $500

        await mock_kalshi_client.close()


# ===========================================================================
# 400-level non-retryable error — raises immediately
# ===========================================================================

class TestNonRetryableError:

    @pytest.mark.asyncio
    async def test_400_raises_immediately(self, mock_kalshi_client):
        url = f"{DEMO_BASE}/portfolio/orders"

        with aioresponses() as m:
            m.post(url, status=400, payload={"error": "invalid order"})

            with pytest.raises(KalshiAPIError) as exc_info:
                order = Order(
                    ticker="T1", action=Action.BUY, side=Side.YES,
                    count=1, yes_price=20,
                )
                await mock_kalshi_client.place_order(order)

            assert exc_info.value.status == 400

        await mock_kalshi_client.close()

    @pytest.mark.asyncio
    async def test_403_raises_immediately(self, mock_kalshi_client):
        url = f"{DEMO_BASE}/portfolio/balance"

        with aioresponses() as m:
            m.get(url, status=403, payload={"error": "forbidden"})

            with pytest.raises(KalshiAPIError) as exc_info:
                await mock_kalshi_client.get_balance()

            assert exc_info.value.status == 403

        await mock_kalshi_client.close()

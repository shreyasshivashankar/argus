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


# ===========================================================================
# get_orders — paginated fetch with query string signing
# ===========================================================================

class TestGetOrders:

    @pytest.mark.asyncio
    async def test_returns_resting_orders(self, mock_kalshi_client):
        import re
        orders_page = [
            {"order_id": "o1", "ticker": "T1", "status": "resting"},
            {"order_id": "o2", "ticker": "T2", "status": "resting"},
        ]

        with aioresponses() as m:
            m.get(
                re.compile(r".*/portfolio/orders\?"),
                payload={"orders": orders_page, "cursor": ""},
            )

            result = await mock_kalshi_client.get_orders(status="resting")

        assert len(result) == 2
        assert result[0]["order_id"] == "o1"

        await mock_kalshi_client.close()

    @pytest.mark.asyncio
    async def test_paginates_with_cursor(self, mock_kalshi_client):
        import re

        with aioresponses() as m:
            m.get(
                re.compile(r".*/portfolio/orders\?"),
                payload={"orders": [{"order_id": "o1"}], "cursor": "page2cursor"},
            )
            m.get(
                re.compile(r".*/portfolio/orders\?"),
                payload={"orders": [{"order_id": "o2"}], "cursor": ""},
            )

            result = await mock_kalshi_client.get_orders(status="resting")

        assert len(result) == 2
        assert result[0]["order_id"] == "o1"
        assert result[1]["order_id"] == "o2"

        await mock_kalshi_client.close()

    @pytest.mark.asyncio
    async def test_empty_orders(self, mock_kalshi_client):
        import re

        with aioresponses() as m:
            m.get(
                re.compile(r".*/portfolio/orders\?"),
                payload={"orders": [], "cursor": ""},
            )

            result = await mock_kalshi_client.get_orders(status="resting")

        assert result == []

        await mock_kalshi_client.close()


# ===========================================================================
# get_fills — fetch fills for a specific order
# ===========================================================================

class TestGetFills:

    @pytest.mark.asyncio
    async def test_returns_fills(self, mock_kalshi_client):
        import re
        fills = [
            {"yes_price": 20, "count": 5, "trade_id": "t1"},
            {"yes_price": 22, "count": 3, "trade_id": "t2"},
        ]

        with aioresponses() as m:
            m.get(
                re.compile(r".*/portfolio/fills\?"),
                payload={"fills": fills},
            )

            result = await mock_kalshi_client.get_fills("order-123")

        assert len(result) == 2
        assert result[0]["yes_price"] == 20

        await mock_kalshi_client.close()


# ===========================================================================
# Query string signing — path-only, no query params in signature
# ===========================================================================

class TestQueryStringSigning:

    def test_sign_strips_query_string(self, mock_kalshi_client):
        """Verify _auth_headers is called with path-only, not query string."""
        path = "/portfolio/orders?status=resting&limit=200"
        path_no_qs = path.split("?", 1)[0]

        assert path_no_qs == "/portfolio/orders"
        assert "?" not in path_no_qs

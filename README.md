# Argus

Production-grade Kalshi trading bot for NBA markets. WebSocket-first data ingestion, out-of-band LLM context monitoring, fail-close quant trigger, and a fill-aware executor with Kelly sizing.

## Prerequisites

- Python 3.11+
- Redis server
- Kalshi API key pair (RSA-PSS)
- LLM API key (Anthropic, Google Gemini, or OpenAI)
- Docker (optional, for containerized runs)

## Setup

### 1. Install dependencies

```bash
cd ~/argus-hybrid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate your Kalshi API key

**For demo (fake money):** go to [demo.kalshi.com](https://demo.kalshi.com) -> Settings -> API Keys -> Create API Key

**For prod (real money):** go to [kalshi.com](https://kalshi.com) -> Settings -> API Keys -> Create API Key

This gives you an **API Key ID** and a **private key `.pem` file** download. Place the key on your server:

```bash
# From your local machine
scp kalshi_private_key.pem shrey@kalshi-bot-instance-1:~/.ssh/kalshi_private_key.pem

# Lock down permissions
chmod 600 ~/.ssh/kalshi_private_key.pem
```

### 3. Configure `.env`

Edit `.env` with your real keys. This file is gitignored and never leaves the server.

```bash
nano .env
```

**Required fields:**

```
KALSHI_API_KEY_ID=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=/home/shrey/.ssh/kalshi_private_key.pem
```

**LLM provider** -- all three are preconfigured. Set `LLM_PROVIDER` and the matching API key:

```
LLM_PROVIDER=anthropic          # "openai", "anthropic", or "gemini"
ANTHROPIC_API_KEY=sk-ant-...    # for Claude
GEMINI_API_KEY=AIza...          # for Gemini
LLM_API_KEY=sk-...              # for OpenAI
```

### 4. Start Redis

```bash
redis-server --daemonize yes
```

## Running

The bot takes two flags: `--env` (demo or prod) and `--paper` (simulated or live orders).

```bash
# Paper trading on demo -- safest, start here
python main.py --env demo --paper

# Live orders on demo -- real Kalshi demo orders, fake money
python main.py --env demo

# Paper trading on prod -- real market data, simulated fills
python main.py --env prod --paper

# Live orders on prod -- REAL MONEY
python main.py --env prod
```

**Recommended progression:**

1. `--env demo --paper` -- verify logs, context checks, +EV detection
2. `--env demo` -- test real order placement against demo exchange
3. `--env prod --paper` -- validate against real market data
4. `--env prod` -- go live with real capital

### Running with Docker

No local Python or Redis setup needed.

```bash
# Run tests
docker build --target test -t argus-test . && docker run --rm argus-test

# Run in production mode (includes Redis)
docker-compose up -d argus

# Run in paper trading mode
docker-compose --profile paper up -d argus-paper

# View logs
docker-compose logs -f argus

# Stop
docker-compose down
```

## Running Tests

```bash
# Local
pytest tests/ -v

# Docker (no dependencies needed)
docker build --target test -t argus-test . && docker run --rm argus-test
```

## Project Structure

```
argus-hybrid/
├── main.py                    # Entrypoint (--env, --paper flags)
├── requirements.txt           # Production + test dependencies
├── pyproject.toml             # Pytest config
├── Dockerfile                 # Multi-stage: base → test → production
├── docker-compose.yml         # Redis + bot services
├── .env                       # API keys and config (gitignored)
│
├── core/                      # Infrastructure layer
│   ├── schemas.py             # Pydantic models (Signal, Order, GameState, etc.)
│   ├── bus.py                 # Redis Pub/Sub + context cache (fail-close)
│   ├── client.py              # Kalshi async REST client + RSA-PSS WS auth
│   └── base_agent.py          # BaseAgent ABC (logging, heartbeat, uvloop)
│
├── watchers/                  # Data ingestion layer
│   ├── sports_feed.py         # Sports API WebSocket + Balldontlie (research)
│   └── kalshi_feed.py         # Kalshi WS (orderbook, ticker, fill, user_orders)
│
├── agents/                    # Trading logic layer
│   ├── nba_quant.py           # NBA quant trigger (+EV detection)
│   ├── narrative.py           # Out-of-band LLM context monitor (OpenAI/Claude/Gemini)
│   ├── executor.py            # Fill-aware execution state machine
│   └── paper_executor.py      # Simulated matching engine for paper trading
│
├── tests/                     # Test suite
│   ├── conftest.py            # Shared fixtures and mock environment
│   ├── test_kalshi_feed.py    # Order book delta tests
│   ├── test_executor.py       # Kelly, kill switch, fills, VWAP, GC tests
│   ├── test_nba_quant.py      # Context cache + fail-close tests
│   └── test_client.py         # Fault injection (502/429 retry backoff)
│
├── docs/
│   └── ARCHITECTURE.md        # System architecture design doc
│
├── data/                      # Historical backtest data
└── logs/                      # Runtime logs (gitignored)
```

## Adding New Sports

1. **Create a quant agent** -- subclass `BaseAgent` in `agents/nfl_quant.py` with sport-specific probability models
2. **Add a sports feed** -- subclass `SportsFeed` in `watchers/sports_feed.py` if a different data provider is needed
3. **Register in `main.py`** -- instantiate and add to the `asyncio.gather` launch

The core infrastructure (Redis bus, Kalshi client, executor) is sport-agnostic.

## Key Invariants

- **Fail-close**: missing Redis context = VETO. Never trades blind.
- **No REST in hot path**: bankroll is background-cached. No HTTP calls during execution.
- **No LLM in hot path**: context is pre-computed out-of-band.
- **Limit orders only**: no market order codepath exists.
- **Fill-before-sell**: exit orders only dispatch after Kalshi fill confirmation.
- **Incremental exits**: partial fills are hedged immediately, not after full fill.
- **VWAP P&L**: kill switch uses actual execution prices, not limit prices.

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

The bot takes three flags: `--env` (demo or prod), `--paper` (simulated or live orders), and `--track` (enable SQLite trade persistence).

```bash
# Paper trading on demo -- safest, start here
python main.py --env demo --paper

# Paper trading + trade tracking
python main.py --env demo --paper --track

# Live orders on demo -- real Kalshi demo orders, fake money
python main.py --env demo --track

# Paper trading on prod -- real market data, simulated fills
python main.py --env prod --paper --track

# Live orders on prod -- REAL MONEY
python main.py --env prod --track
```

**Recommended progression:**

1. `--env demo --paper` -- verify logs, context checks, +EV detection
2. `--env demo` -- test real order placement against demo exchange
3. `--env prod --paper` -- validate against real market data
4. `--env prod` -- go live with real capital

### Running with Docker

No local Python or Redis setup needed. Both `argus` and `argus-paper` services have `--track` enabled by default, and `data/` is mounted so `trades.db` persists on the host.

```bash
# Run tests
docker build --target test -t argus-test . && docker run --rm argus-test

# Run in production mode (includes Redis, tracking enabled)
docker-compose up -d argus

# Run in paper trading mode (tracking enabled, is_paper=True)
docker-compose --profile paper up -d argus-paper

# Open the live terminal monitor (requires interactive TTY)
docker-compose run --rm argus-monitor

# View logs
docker-compose logs -f argus

# Stop
docker-compose down
```

## Monitoring

### Terminal Dashboard

A read-only Rich TUI that subscribes to the Redis bus and displays live P&L, win rate, agent health, and active game context. Runs in a separate terminal -- does not interfere with the trading process.

```bash
# Local
python -m scripts.monitor

# Docker (must be interactive, not detached)
docker-compose run --rm argus-monitor
```

The dashboard shows four panels: a stats header (P&L, win rate, kill switch limit), an event feed of validated signals and executions, an agent health table (last heartbeat), and an active games panel with SAFE/VETO context status.

### Trade Tracker (SQLite)

Enable with `--track` to persist every signal and execution to `data/trades.db`. The tracker tags every row with `is_paper` based on the `--paper` flag to prevent data pollution between paper and live modes.

```bash
# Query live-only trades
sqlite3 data/trades.db "SELECT * FROM trades WHERE is_paper = 0;"

# Query paper-only trades
sqlite3 data/trades.db "SELECT * FROM trades WHERE is_paper = 1;"

# Win rate for live trades
sqlite3 data/trades.db "SELECT
  COUNT(*) FILTER (WHERE pnl_dollars > 0) AS wins,
  COUNT(*) FILTER (WHERE pnl_dollars < 0) AS losses,
  SUM(pnl_dollars) AS total_pnl
FROM trades WHERE status = 'EXECUTED' AND is_paper = 0;"
```

See `docs/MONITORING.md` for full architecture details.

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
├── main.py                    # Entrypoint (--env, --paper, --track flags)
├── requirements.txt           # Production + test dependencies
├── pyproject.toml             # Pytest config
├── Dockerfile                 # Multi-stage: base → test → production
├── docker-compose.yml         # Redis + bot + monitor services
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
│   ├── paper_executor.py      # Simulated matching engine for paper trading
│   └── track_agent.py         # SQLite trade tracker (WAL mode, paper/live isolation)
│
├── scripts/                   # Standalone utilities
│   └── monitor.py             # Rich terminal dashboard (read-only Redis observer)
│
├── tests/                     # Test suite
│   ├── conftest.py            # Shared fixtures and mock environment
│   ├── test_kalshi_feed.py    # Order book delta tests
│   ├── test_executor.py       # Kelly, kill switch, fills, VWAP, GC tests
│   ├── test_nba_quant.py      # Context cache + fail-close tests
│   └── test_client.py         # Fault injection (502/429 retry backoff)
│
├── docs/
│   ├── ARCHITECTURE.md        # System architecture design doc
│   └── MONITORING.md          # Monitoring & trade tracking design doc
│
├── data/                      # Historical backtest data + trades.db
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
- **Reallocation hurdle**: new trade's total projected EV must strictly exceed foregone profit + taker fees before liquidating a resting exit.
- **Order-level targeting**: reallocation uses `target_order_id`, not ticker matching, to handle partial-fill batches correctly.

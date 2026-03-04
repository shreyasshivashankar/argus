# Argus

Trading bot that places bets on NBA markets on Kalshi. Ingests live game scores and order book data over WebSockets, runs an LLM in the background to flag injuries/momentum shifts, and only pulls the trigger when the math checks out. Sizes positions with Kelly, tracks P&L, and kills everything if losses hit a threshold.

## Quick Start (Docker)

The fastest way to run. No Python, pip, or Redis install needed -- Docker handles everything.

### 1. Get your Kalshi API key

**Demo (fake money):** [demo.kalshi.co](https://demo.kalshi.co) -> Account Settings -> Profile Settings -> Create New API Key

**Prod (real money):** [kalshi.com](https://kalshi.com) -> Account Settings -> Profile Settings -> Create New API Key

You'll get an API Key ID and a `.key` private key file. Save the private key immediately -- you can't retrieve it later. Put it somewhere safe on your server:

```bash
scp kalshi_private_key.key shrey@kalshi-bot-instance-1:~/.ssh/kalshi_private_key.key
chmod 600 ~/.ssh/kalshi_private_key.key
```

### 2. Configure `.env`

This file is gitignored. Fill in your keys:

```bash
nano .env
```

Required:

```
KALSHI_API_KEY_ID=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=/home/shrey/.ssh/kalshi_private_key.key
```

LLM -- pick one, set the provider and its key:

```
LLM_PROVIDER=anthropic          # "openai", "anthropic", or "gemini"
ANTHROPIC_API_KEY=sk-ant-...    # Claude
GEMINI_API_KEY=AIza...          # Gemini
LLM_API_KEY=sk-...              # OpenAI
```

### 3. Run

```bash
# Paper mode -- simulated fills, trade tracking on, no money at risk
docker-compose --profile paper up -d argus-paper

# Open the live terminal monitor (optional, separate terminal)
docker-compose run --rm argus-monitor

# Check logs
docker-compose logs -f argus-paper

# After the games, check results
sqlite3 data/trades.db "SELECT * FROM trades WHERE is_paper = 1;"

# Stop everything
docker-compose down
```

When you're ready to go live:

```bash
# Live orders on prod -- real money
docker-compose up -d argus
```

Trade tracking and SQLite persistence are on by default. The `data/` folder is mounted to the host so `trades.db` survives container restarts.

## Running Locally (without Docker)

If you prefer running without Docker, you'll need Python 3.11+, Redis, and the dependencies installed manually.

### Prerequisites

- Python 3.11+
- Redis
- Kalshi API key pair (RSA-PSS)
- An LLM API key -- Anthropic, Gemini, or OpenAI all work

### Setup

```bash
cd ~/argus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
redis-server --daemonize yes
```

Configure `.env` the same way as above (steps 1-2).

### Run

Three flags: `--env` (demo/prod), `--paper` (simulated fills), `--track` (log trades to SQLite).

```bash
# Paper trading on demo -- start here
python main.py --env demo --paper --track

# Live orders on demo -- real orders, fake money
python main.py --env demo --track

# Paper on prod -- real market data, simulated fills
python main.py --env prod --paper --track

# Live on prod -- real money
python main.py --env prod --track
```

Work your way up:

1. `--env demo --paper` -- make sure logs look right, context checks fire, EV detection works
2. `--env demo` -- actually place orders on the demo exchange
3. `--env prod --paper` -- run against real market data without risking anything
4. `--env prod` -- go live

## Monitoring

### Terminal Dashboard

A live terminal UI that tails the Redis event bus. Shows P&L, win rate, agent heartbeats, and game context. Read-only -- doesn't touch the trading loop.

```bash
# Local
python -m scripts.monitor

# Docker
docker-compose run --rm argus-monitor
```

### Trade Tracker

Pass `--track` to log every signal and fill to `data/trades.db`. Paper and live trades are tagged separately so they don't mix.

```bash
# Live trades
sqlite3 data/trades.db "SELECT * FROM trades WHERE is_paper = 0;"

# Paper trades
sqlite3 data/trades.db "SELECT * FROM trades WHERE is_paper = 1;"

# Win rate
sqlite3 data/trades.db "SELECT
  COUNT(*) FILTER (WHERE pnl_dollars > 0) AS wins,
  COUNT(*) FILTER (WHERE pnl_dollars < 0) AS losses,
  SUM(pnl_dollars) AS total_pnl
FROM trades WHERE status = 'EXECUTED' AND is_paper = 0;"
```

More details in `docs/MONITORING.md`.

## Tests

```bash
pytest tests/ -v

# Or via Docker
docker build --target test -t argus-test . && docker run --rm argus-test
```

## Project Structure

```
argus/
├── main.py                    # Entrypoint (--env, --paper, --track)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env                       # Your API keys (gitignored)
│
├── core/
│   ├── schemas.py             # Pydantic models and settings
│   ├── bus.py                 # Redis pub/sub + context cache
│   ├── client.py              # Kalshi REST client + WS auth
│   └── base_agent.py          # Base class for all agents
│
├── watchers/
│   ├── sports_feed.py         # Live game scores via WebSocket
│   └── kalshi_feed.py         # Kalshi order book + fills via WebSocket
│
├── agents/
│   ├── nba_quant.py           # EV detection and signal generation
│   ├── narrative.py           # LLM context monitor (injury reports, momentum)
│   ├── executor.py            # Order lifecycle, Kelly sizing, kill switch
│   ├── paper_executor.py      # Simulated matching engine for paper mode
│   └── track_agent.py         # SQLite trade logger
│
├── scripts/
│   └── monitor.py             # Terminal dashboard
│
├── tests/
│   ├── conftest.py
│   ├── test_kalshi_feed.py
│   ├── test_executor.py
│   ├── test_nba_quant.py
│   └── test_client.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── MONITORING.md
│
├── data/                      # Backtest data + trades.db
└── logs/
```

## Adding New Sports

1. Write a new quant agent -- subclass `BaseAgent` (e.g. `agents/nfl_quant.py`)
2. Add a sports feed if the data source is different
3. Wire it up in `main.py`

The core stuff (Redis bus, Kalshi client, executor) doesn't care about the sport.

## Key Invariants

- Missing context = VETO. The bot never trades blind.
- No REST calls in the hot path. Bankroll is background-cached.
- LLM runs out-of-band. Context is pre-computed, not inline.
- Limit orders only. No market order codepath.
- Exit orders wait for fill confirmation before dispatching.
- Partial fills get hedged immediately.
- Kill switch uses VWAP, not limit prices.
- Reallocation only happens when the new trade's projected EV strictly beats the foregone profit plus fees.
- Reallocation targets orders by ID, not by ticker, so partial-fill batches don't get mixed up.

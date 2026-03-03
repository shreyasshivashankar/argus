# Argus

Trading bot that places bets on NBA markets on Kalshi. Ingests live game scores and order book data over WebSockets, runs an LLM in the background to flag injuries/momentum shifts, and only pulls the trigger when the math checks out. Sizes positions with Kelly, tracks P&L, and kills everything if losses hit a threshold.

## Prerequisites

- Python 3.11+
- Redis
- Kalshi API key pair (RSA-PSS)
- An LLM API key -- Anthropic, Gemini, or OpenAI all work
- Docker (optional)

## Setup

### 1. Install dependencies

```bash
cd ~/argus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get your Kalshi API key

**Demo (fake money):** [demo.kalshi.com](https://demo.kalshi.com) -> Settings -> API Keys -> Create

**Prod (real money):** [kalshi.com](https://kalshi.com) -> Settings -> API Keys -> Create

You'll get an API Key ID and a `.pem` private key file. Put the key somewhere safe on your server:

```bash
scp kalshi_private_key.pem shrey@kalshi-bot-instance-1:~/.ssh/kalshi_private_key.pem
chmod 600 ~/.ssh/kalshi_private_key.pem
```

### 3. Configure `.env`

This file is gitignored. Fill in your keys:

```bash
nano .env
```

Required:

```
KALSHI_API_KEY_ID=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=/home/shrey/.ssh/kalshi_private_key.pem
```

LLM -- pick one, set the provider and its key:

```
LLM_PROVIDER=anthropic          # "openai", "anthropic", or "gemini"
ANTHROPIC_API_KEY=sk-ant-...    # Claude
GEMINI_API_KEY=AIza...          # Gemini
LLM_API_KEY=sk-...              # OpenAI
```

### 4. Start Redis

```bash
redis-server --daemonize yes
```

## Running

Three flags: `--env` (demo/prod), `--paper` (simulated fills), `--track` (log trades to SQLite).

```bash
# Paper trading on demo -- start here
python main.py --env demo --paper

# Same thing, but log trades to SQLite
python main.py --env demo --paper --track

# Live orders on demo -- places real orders, but it's fake money
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

### Docker

No local Python or Redis needed. `--track` is on by default, `data/` is mounted so the SQLite DB persists.

```bash
# Tests
docker build --target test -t argus-test . && docker run --rm argus-test

# Live
docker-compose up -d argus

# Paper
docker-compose --profile paper up -d argus-paper

# Terminal monitor (needs TTY)
docker-compose run --rm argus-monitor

# Logs
docker-compose logs -f argus

# Stop
docker-compose down
```

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

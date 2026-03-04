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
OPENAI_API_KEY=sk-...           # OpenAI
```

Sports data -- pick a provider:

```
SPORTS_PROVIDER=sportradar       # "sportradar" (production) or "balldontlie" (paper)
```

**Sportradar (production)** -- Push Statistics streaming feed. Sub-second latency, full player box scores. Get an API key at [developer.sportradar.com](https://developer.sportradar.com/):

```
SPORTRADAR_API_KEY=your-key-here
SPORTRADAR_ACCESS_LEVEL=production
SPORTRADAR_SEASON=2025           # starting year of the NBA season
```

**Balldontlie (paper/research)** -- REST polling. Cheaper but higher latency. Get a key at [app.balldontlie.io](https://app.balldontlie.io/):

```
BALLDONTLIE_API_KEY=your-key-here
BALLDONTLIE_TIER=all-star        # "free" (scores only), "all-star" ($10/mo, +player stats), "goat" ($40/mo)
```

Player props require at least ALL-STAR tier (or Sportradar) for live player stats.

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
│   ├── sports_feed.py         # Game scores + player stats (Sportradar streaming or Balldontlie REST)
│   └── kalshi_feed.py         # Kalshi order book + fills via WebSocket
│
├── agents/
│   ├── nba_quant.py           # OmniQuant agent — strategy ranking, context, cooldown
│   ├── strategies/
│   │   ├── base.py            # BaseStrategy interface
│   │   ├── moneyline.py       # Logistic reversal model (game-winner markets)
│   │   ├── totals.py          # Pace projection model (over/under markets)
│   │   └── player_props.py    # Usage-rate projection (player points markets)
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
│   ├── test_player_props.py
│   └── test_client.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── MONITORING.md
│
├── data/                      # Backtest data + trades.db
└── logs/
```

## Adding New Markets

To trade a new market type (e.g. spreads), add a strategy file to `agents/strategies/`. Implement `can_evaluate` (does this ticker belong to me?) and `evaluate` (is there an edge?). The OmniQuant agent picks it up automatically.

Current strategies: **moneyline** (game winner), **totals** (over/under), **player_props** (individual player points).

To add a new sport entirely, write a new quant agent (subclass `BaseAgent`), add a sports feed if the data source is different, and wire it in `main.py`. The executor, Redis bus, and Kalshi client don't care what sport the signal came from.

## How It Stays Safe

The bot is designed to fail conservatively. If the LLM goes down, context keys expire and trading stops. If bankroll data is stale by a few dollars, that's fine -- missing a trade by 200ms is worse. Limit orders only, no market orders anywhere. Exit orders only fire after Kalshi confirms the entry filled. Partial fills get their own exit immediately instead of waiting for the full order.

The kill switch tracks P&L using actual execution prices (VWAP), not limit prices, so it stays accurate across hundreds of trades. Reallocation math is unit-correct and targets specific orders by ID, not by ticker, so partial-fill batches don't get mixed up.

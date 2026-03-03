# Argus

Production-grade Kalshi trading bot for NBA markets. WebSocket-first data ingestion, out-of-band LLM context monitoring, fail-close quant trigger, and a fill-aware executor with Kelly sizing.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design, signal flow diagrams, and module map.

## Quick Start

```bash
# 1. Clone and install
cd ~/argus-hybrid
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env .env.local   # edit .env with your keys
# Place your Kalshi RSA private key at the path in KALSHI_PRIVATE_KEY_PATH

# 3. Start Redis
redis-server --daemonize yes

# 4. Run
python main.py
```

## Project Structure

```
argus-hybrid/
├── main.py                    # Entrypoint — wires and launches all components
├── requirements.txt           # Production dependencies
├── .env                       # Configuration (gitignored)
│
├── core/                      # Infrastructure layer
│   ├── schemas.py             # Pydantic models (Signal, Order, GameState, etc.)
│   ├── bus.py                 # Redis Pub/Sub + context cache
│   ├── client.py              # Kalshi async REST client + WS auth
│   └── base_agent.py          # BaseAgent ABC (logging, heartbeat, uvloop)
│
├── watchers/                  # Data ingestion layer
│   ├── sports_feed.py         # Sports API WebSocket + Balldontlie (research)
│   └── kalshi_feed.py         # Kalshi WS (orderbook, ticker, fill, user_orders)
│
├── agents/                    # Trading logic layer
│   ├── nba_quant.py           # NBA quant trigger (+EV detection)
│   ├── narrative.py           # Out-of-band LLM context monitor
│   └── executor.py            # Fill-aware execution state machine
│
├── docs/
│   └── ARCHITECTURE.md        # System architecture design doc
│
├── data/                      # Historical backtest data
└── logs/                      # Runtime logs (gitignored)
```

## Adding New Sports

The system is designed for extensibility. To add a new sport (e.g., NFL):

1. **Create a new quant agent** — subclass `BaseAgent` in `agents/nfl_quant.py` with sport-specific probability models
2. **Add a sports feed** — subclass `SportsFeed` in `watchers/sports_feed.py` if a different data provider is needed
3. **Register in `main.py`** — instantiate and add to the `asyncio.gather` launch

The core infrastructure (Redis bus, Kalshi client, executor) is sport-agnostic and shared across all agents.

## Key Invariants

- **Fail-close**: missing Redis context = VETO. Never trades blind.
- **No REST in hot path**: bankroll is background-cached. No HTTP calls during execution.
- **No LLM in hot path**: context is pre-computed out-of-band.
- **Limit orders only**: no market order codepath exists.
- **Fill-before-sell**: exit orders only dispatch after Kalshi fill confirmation.

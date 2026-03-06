"""Argus Real-Time Terminal Dashboard.

Read-only observer that subscribes to the Redis bus and renders a live
Rich TUI showing P&L, win rate, agent health, game context, and the
trade event feed.  Never publishes or mutates trading state.

Usage:
    python -m scripts.monitor          # local
    docker-compose run --rm argus-monitor  # Docker (requires TTY)
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from core.bus import SignalBus
from core.client import KalshiAsyncClient
from core.schemas import AppSettings

console = Console()

_CHANNELS = [
    "signal:validated",
    "signal:executed",
    "signal:reallocate",
    "signal:heartbeat",
    "game:state",
    "market:state",
    "portfolio:state",
]

MAX_EVENTS = 20
MAX_GAMES = 10


class ArgusMonitor:
    def __init__(self) -> None:
        self.settings = AppSettings()  # type: ignore[call-arg]
        self.bus = SignalBus(self.settings.REDIS_URL)
        self.client = KalshiAsyncClient(self.settings)

        self.trades: list[dict[str, Any]] = []
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.signals_seen: int = 0

        self.kalshi_balance: float | None = None
        self.open_positions: int = 0

        self.heartbeats: dict[str, str] = {}
        self.games: dict[str, dict[str, Any]] = {}
        self.context_cache: dict[str, str] = {}

        self.feed_ts: dict[str, datetime] = {}

        self._running = True

    # ------------------------------------------------------------------
    # Redis callback
    # ------------------------------------------------------------------

    async def _on_message(self, channel: str, data: dict[str, Any]) -> None:
        now = datetime.utcnow().strftime("%H:%M:%S")

        if channel == "signal:executed":
            entry = int(data.get("entry_price", 0))
            exit_ = int(data.get("exit_price", 0))
            pnl = (exit_ - entry) / 100.0
            self.total_pnl += pnl
            if exit_ > entry:
                self.wins += 1
            elif exit_ < entry:
                self.losses += 1
            self.trades.insert(0, {
                "time": now,
                "type": "EXECUTED",
                "ticker": data.get("ticker", "?"),
                "details": (
                    f"Entry: {entry}c  "
                    f"Exit: {exit_}c"
                ),
                "pnl": pnl,
            })

        elif channel == "portfolio:state":
            self.kalshi_balance = float(data.get("bankroll", 0))
            self.open_positions = len(data.get("positions", []))
            return

        elif channel == "signal:validated":
            self.signals_seen += 1
            ev = data.get("ev_estimate", 0)
            conf = data.get("confidence", 0)
            source = data.get("source", "?")
            self.trades.insert(0, {
                "time": now,
                "type": "+EV SIGNAL",
                "ticker": data.get("ticker", "?"),
                "details": f"[{source}] EV: +${ev:.4f}  Conf: {conf:.2f}",
                "pnl": 0.0,
            })

        elif channel == "signal:reallocate":
            target = data.get("target_order_id", "?")
            bid = data.get("entry_price", 0)
            self.trades.insert(0, {
                "time": now,
                "type": "REALLOCATE",
                "ticker": data.get("ticker", "?"),
                "details": f"Liq @{bid}c  target={target}",
                "pnl": 0.0,
            })

        elif channel == "signal:heartbeat":
            agent = data.get("agent", "?")
            self.heartbeats[agent] = now
            ts_now = datetime.utcnow()
            self.feed_ts[agent] = ts_now
            api_ok = data.get("api_ok", True)
            if agent == "sports_feed" and api_ok is not False:
                self.feed_ts["sports"] = ts_now
            elif agent == "narrative" and api_ok is not False:
                self.feed_ts["narrative"] = ts_now

        elif channel == "game:state":
            gid = data.get("game_id", "")
            if gid:
                self.games[gid] = data
                ctx_status, _ = await self.bus.get_context(gid)
                self.context_cache[gid] = ctx_status.value
            if gid and data.get("home_team") and data.get("away_team"):
                self.feed_ts["sports"] = datetime.utcnow()

        elif channel == "market:state":
            ticker = data.get("ticker", "")
            yes_bid = data.get("yes_bid")
            yes_ask = data.get("yes_ask")
            if ticker and isinstance(yes_bid, (int, float)) and isinstance(yes_ask, (int, float)):
                self.feed_ts["kalshi"] = datetime.utcnow()

        self.trades = self.trades[:MAX_EVENTS]

    # ------------------------------------------------------------------
    # Dashboard layout
    # ------------------------------------------------------------------

    def _build_dashboard(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=3),
            Layout(name="footer", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="events"),
            Layout(name="agents", size=42),
        )

        # --- Header: stats bar ---
        total = self.wins + self.losses
        wr = (self.wins / total * 100) if total else 0.0
        pnl_c = "green" if self.total_pnl >= 0 else "red"
        bal_str = (
            f"${self.kalshi_balance:.2f}"
            if self.kalshi_balance is not None
            else "..."
        )
        stats = (
            f"[bold]Balance:[/bold] [cyan]{bal_str}[/cyan]  |  "
            f"[bold]Session P&L:[/bold] [{pnl_c}]${self.total_pnl:.2f}[/{pnl_c}]  |  "
            f"[bold]Win Rate:[/bold] {wr:.1f}% ({self.wins}W-{self.losses}L)  |  "
            f"[bold]Open:[/bold] {self.open_positions}  |  "
            f"[bold]Signals:[/bold] {self.signals_seen}  |  "
            f"[bold]Kill:[/bold] ${self.settings.DAILY_STOP_LOSS_USD:.2f}  |  "
            f"[bold]Env:[/bold] {self.settings.KALSHI_ENV}"
        )
        layout["header"].update(
            Panel(stats, title="[bold blue]Argus Monitor[/bold blue]")
        )

        # --- Events table ---
        ev_table = Table(expand=True, show_edge=False, pad_edge=False)
        ev_table.add_column("Time", style="dim", width=9)
        ev_table.add_column("Event", width=12)
        ev_table.add_column("Market", style="cyan", width=20)
        ev_table.add_column("Details", style="magenta")
        ev_table.add_column("P&L", justify="right", width=10)

        for t in self.trades:
            pnl_str = ""
            if t["type"] == "EXECUTED":
                c = "green" if t["pnl"] > 0 else "red"
                pnl_str = f"[{c}]${t['pnl']:.2f}[/{c}]"
            ev_c = "green" if t["type"] == "EXECUTED" else "yellow"
            ev_table.add_row(
                t["time"],
                f"[{ev_c}]{t['type']}[/{ev_c}]",
                t["ticker"],
                t["details"],
                pnl_str,
            )

        layout["events"].update(Panel(ev_table, title="Event Feed"))

        # --- Agent & feed health ---
        ag_table = Table(expand=True, show_edge=False)
        ag_table.add_column("Component", width=16)
        ag_table.add_column("Status", width=10)
        ag_table.add_column("Last Update", width=12)

        utcnow = datetime.utcnow()

        feed_labels = [
            ("sports", "Sports Feed"),
            ("kalshi", "Kalshi WS"),
            ("narrative", "LLM / Narr."),
        ]
        all_healthy = True
        for key, label in feed_labels:
            ts = self.feed_ts.get(key)
            if ts is None:
                ag_table.add_row(f"[dim]{label}[/dim]", "[red]Unhealthy[/red]", "[dim]waiting[/dim]")
                all_healthy = False
            else:
                age = (utcnow - ts).total_seconds()
                ts_str = ts.strftime("%H:%M:%S")
                if age < 60:
                    ag_table.add_row(f"[green]{label}[/green]", "[green]Healthy[/green]", ts_str)
                elif age < 300:
                    ag_table.add_row(f"[yellow]{label}[/yellow]", "[yellow]Unhealthy[/yellow]", ts_str)
                    all_healthy = False
                else:
                    ag_table.add_row(f"[red]{label}[/red]", "[red]Unhealthy[/red]", ts_str)
                    all_healthy = False

        overall = "[green]Healthy[/green]" if all_healthy else "[red]Unhealthy[/red]"
        ag_table.add_row("", "", "")
        ag_table.add_row("[bold]APIs[/bold]", overall, "[dim]valid responses[/dim]" if all_healthy else "[dim]stale/missing[/dim]")
        ag_table.add_row("", "", "")

        for agent in ("nba_quant", "executor", "paper_executor", "track"):
            ts = self.heartbeats.get(agent, "---")
            color = "green" if ts != "---" else "dim"
            status = "[green]OK[/green]" if ts != "---" else "[dim]---[/dim]"
            ag_table.add_row(f"[{color}]{agent}[/{color}]", status, ts)

        layout["agents"].update(Panel(ag_table, title="System Health"))

        # --- Footer: active games + context ---
        gm_table = Table(expand=True, show_edge=False)
        gm_table.add_column("Game", width=20)
        gm_table.add_column("Score", width=12)
        gm_table.add_column("Q", width=4)
        gm_table.add_column("Clock", width=8)
        gm_table.add_column("Context", width=10)

        for gid, g in list(self.games.items())[:MAX_GAMES]:
            ctx = self.context_cache.get(gid, "?")
            ctx_c = "green" if ctx == "SAFE" else "red"
            gm_table.add_row(
                f"{g.get('away_team', '?')} @ {g.get('home_team', '?')}",
                f"{g.get('away_score', 0)}-{g.get('home_score', 0)}",
                str(g.get("quarter", "?")),
                str(g.get("clock", "?")),
                f"[{ctx_c}]{ctx}[/{ctx_c}]",
            )

        layout["footer"].update(Panel(gm_table, title="Active Games"))

        return layout

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _poll_balance(self) -> None:
        """Periodically fetch live balance from Kalshi REST API."""
        while self._running:
            try:
                self.kalshi_balance = await self.client.get_balance()
                positions = await self.client.get_positions()
                self.open_positions = len(
                    [p for p in positions if p.get("total_traded", 0) > 0]
                )
            except Exception:
                pass
            await asyncio.sleep(30)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop)

        sub_task = asyncio.create_task(
            self.bus.subscribe(_CHANNELS, self._on_message)
        )
        balance_task = asyncio.create_task(self._poll_balance())

        try:
            with Live(
                self._build_dashboard(),
                refresh_per_second=2,
                console=console,
                screen=False,
                transient=True,
            ) as live:
                while self._running:
                    await asyncio.sleep(0.5)
                    live.update(self._build_dashboard())
        finally:
            sub_task.cancel()
            balance_task.cancel()
            await self.client.close()
            await self.bus.close()

    def _stop(self) -> None:
        self._running = False


def main() -> None:
    monitor = ArgusMonitor()
    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()

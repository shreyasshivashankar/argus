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
from core.schemas import AppSettings

console = Console()

_CHANNELS = [
    "signal:validated",
    "signal:executed",
    "signal:heartbeat",
    "game:state",
    "market:state",
]

MAX_EVENTS = 20
MAX_GAMES = 10


class ArgusMonitor:
    def __init__(self) -> None:
        self.settings = AppSettings()  # type: ignore[call-arg]
        self.bus = SignalBus(self.settings.REDIS_URL)

        self.trades: list[dict[str, Any]] = []
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.signals_seen: int = 0

        self.heartbeats: dict[str, str] = {}
        self.games: dict[str, dict[str, Any]] = {}
        self.context_cache: dict[str, str] = {}

        self._running = True

    # ------------------------------------------------------------------
    # Redis callback
    # ------------------------------------------------------------------

    async def _on_message(self, channel: str, data: dict[str, Any]) -> None:
        now = datetime.utcnow().strftime("%H:%M:%S")

        if channel == "signal:executed":
            pnl = float(data.get("ev_estimate", 0.0))
            self.total_pnl += pnl
            if pnl > 0:
                self.wins += 1
            elif pnl < 0:
                self.losses += 1
            self.trades.insert(0, {
                "time": now,
                "type": "EXECUTED",
                "ticker": data.get("ticker", "?"),
                "details": (
                    f"Entry: {data.get('entry_price', 0)}c  "
                    f"Exit: {data.get('exit_price', 0)}c"
                ),
                "pnl": pnl,
            })

        elif channel == "signal:validated":
            self.signals_seen += 1
            ev = data.get("ev_estimate", 0)
            conf = data.get("confidence", 0)
            self.trades.insert(0, {
                "time": now,
                "type": "+EV SIGNAL",
                "ticker": data.get("ticker", "?"),
                "details": f"EV: +${ev:.4f}  Conf: {conf:.2f}",
                "pnl": 0.0,
            })

        elif channel == "signal:heartbeat":
            agent = data.get("agent", "?")
            self.heartbeats[agent] = now

        elif channel == "game:state":
            gid = data.get("game_id", "")
            if gid:
                self.games[gid] = data
                ctx_status, _ = await self.bus.get_context(gid)
                self.context_cache[gid] = ctx_status.value

        self.trades = self.trades[:MAX_EVENTS]

    # ------------------------------------------------------------------
    # Dashboard layout
    # ------------------------------------------------------------------

    def _build_dashboard(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5 + min(len(self.games), MAX_GAMES)),
        )
        layout["body"].split_row(
            Layout(name="events"),
            Layout(name="agents", size=34),
        )

        # --- Header: stats bar ---
        total = self.wins + self.losses
        wr = (self.wins / total * 100) if total else 0.0
        pnl_c = "green" if self.total_pnl >= 0 else "red"
        stats = (
            f"[bold]P&L:[/bold] [{pnl_c}]${self.total_pnl:.2f}[/{pnl_c}]  |  "
            f"[bold]Win Rate:[/bold] {wr:.1f}% ({self.wins}W-{self.losses}L)  |  "
            f"[bold]Signals:[/bold] {self.signals_seen}  |  "
            f"[bold]Kill Switch:[/bold] ${self.settings.DAILY_STOP_LOSS_USD:.2f}  |  "
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

        # --- Agent health ---
        ag_table = Table(expand=True, show_edge=False)
        ag_table.add_column("Agent", width=16)
        ag_table.add_column("Last Seen", width=12)

        for agent in ("nba_quant", "narrative", "executor", "paper_executor"):
            ts = self.heartbeats.get(agent, "---")
            color = "green" if ts != "---" else "dim"
            ag_table.add_row(f"[{color}]{agent}[/{color}]", ts)

        layout["agents"].update(Panel(ag_table, title="Agent Health"))

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

    async def run(self) -> None:
        console.clear()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop)

        sub_task = asyncio.create_task(
            self.bus.subscribe(_CHANNELS, self._on_message)
        )

        try:
            with Live(
                self._build_dashboard(),
                refresh_per_second=2,
                screen=True,
                console=console,
            ) as live:
                while self._running:
                    await asyncio.sleep(0.5)
                    live.update(self._build_dashboard())
        finally:
            sub_task.cancel()
            await self.bus.close()

    def _stop(self) -> None:
        self._running = False


def main() -> None:
    monitor = ArgusMonitor()
    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()

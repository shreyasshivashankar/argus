"""Argus Trade Report — lifetime P&L summary from Postgres.

Queries the trades table and prints a comprehensive breakdown:
overall stats, per-strategy, per-game, top tickers, and daily P&L.

Usage:
    ./report.sh                               # all live trades
    ./report.sh --paper                       # paper trades only
    ./report.sh --days 7                      # last 7 days only
    python -m scripts.report                  # local (needs DATABASE_URL)
    docker-compose run --rm --entrypoint "" argus python -m scripts.report
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta, timezone


def _connect() -> psycopg2.extensions.connection:
    dsn = os.environ.get(
        "DATABASE_URL", "postgresql://argus:argus@localhost:5432/argus"
    )
    try:
        return psycopg2.connect(dsn)
    except psycopg2.OperationalError as e:
        print(f"Cannot connect to Postgres: {e}")
        print(f"DSN: {dsn}")
        print("Make sure Postgres is running (docker-compose up -d postgres).")
        sys.exit(1)


def report(*, paper: bool = False, days: int | None = None) -> None:
    db = _connect()
    cur = db.cursor()

    is_paper = paper
    mode_label = "PAPER" if paper else "LIVE"

    date_clause = ""
    params: list = [is_paper]
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        date_clause = " AND created_at >= %s"
        params.append(cutoff)

    w = "=" * 70
    d = "-" * 70

    print(f"\n{w}")
    print(f"  ARGUS TRADE REPORT ({mode_label})")
    print(w)

    # ── Overall ──
    cur.execute(f"""
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE status = 'EXECUTED'),
            COALESCE(SUM(pnl_dollars) FILTER (WHERE status = 'EXECUTED'), 0),
            COUNT(*) FILTER (WHERE status = 'EXECUTED' AND pnl_dollars > 0),
            COUNT(*) FILTER (WHERE status = 'EXECUTED' AND pnl_dollars < 0),
            COUNT(*) FILTER (WHERE status = 'EXECUTED' AND pnl_dollars = 0),
            MIN(created_at),
            MAX(created_at)
        FROM trades
        WHERE is_paper = %s{date_clause}
    """, params)
    total, executed, pnl, wins, losses, be, first, last = cur.fetchone()
    executed = executed or 0
    wins = wins or 0
    losses = losses or 0
    be = be or 0
    pnl = pnl or 0.0

    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    fill_rate = (executed / total * 100) if total > 0 else 0
    avg = pnl / executed if executed else 0

    fmt_ts = lambda t: str(t)[:19] if t else "N/A"
    period = f"{fmt_ts(first)} → {fmt_ts(last)}"
    if days:
        period += f"  (last {days}d)"

    print(f"  Period:        {period}")
    print(f"  Signals:       {total}  ({executed} filled, "
          f"{total - executed} unfilled, {fill_rate:.0f}% fill rate)")
    print(f"  Total P&L:     ${pnl:.2f}")
    print(f"  Win Rate:      {wr:.1f}%  ({wins}W / {losses}L / {be}BE)")
    print(f"  Avg P&L/trade: ${avg:.4f}")
    print()

    # ── By strategy ──
    print(d)
    print("  BY STRATEGY")
    print(d)
    cur.execute(f"""
        SELECT
            COALESCE(source, 'unknown'),
            COUNT(*),
            ROUND(SUM(pnl_dollars)::numeric, 2),
            COUNT(*) FILTER (WHERE pnl_dollars > 0),
            COUNT(*) FILTER (WHERE pnl_dollars < 0),
            ROUND(AVG(pnl_dollars)::numeric, 4)
        FROM trades
        WHERE is_paper = %s AND status = 'EXECUTED'{date_clause}
        GROUP BY source
        ORDER BY SUM(pnl_dollars) DESC
    """, params)
    rows = cur.fetchall()
    if rows:
        for src, cnt, spnl, sw, sl, savg in rows:
            sw = sw or 0
            sl = sl or 0
            swr = (sw / (sw + sl) * 100) if (sw + sl) > 0 else 0
            print(f"  {src:20s}  trades={cnt:4d}  P&L=${spnl:>8}  "
                  f"avg=${savg:>7}/t  WR={swr:.0f}% ({sw}W-{sl}L)")
    else:
        print("  (no executed trades)")
    print()

    # ── By game ──
    print(d)
    print("  BY GAME")
    print(d)
    cur.execute(f"""
        SELECT
            game_id,
            COUNT(*),
            ROUND(SUM(pnl_dollars)::numeric, 2),
            COUNT(*) FILTER (WHERE pnl_dollars > 0),
            COUNT(*) FILTER (WHERE pnl_dollars < 0)
        FROM trades
        WHERE is_paper = %s AND status = 'EXECUTED'{date_clause}
        GROUP BY game_id
        ORDER BY SUM(pnl_dollars) DESC
        LIMIT 10
    """, params)
    rows = cur.fetchall()
    if rows:
        for gid, cnt, gpnl, gw, gl in rows:
            gw = gw or 0
            gl = gl or 0
            short_id = (gid or "?")[:8]
            print(f"  {short_id}…  trades={cnt:3d}  P&L=${gpnl:>7}  ({gw}W-{gl}L)")
    else:
        print("  (no executed trades)")
    print()

    # ── Top tickers ──
    print(d)
    print("  TOP 10 TICKERS BY P&L")
    print(d)
    cur.execute(f"""
        SELECT
            ticker,
            COUNT(*),
            ROUND(SUM(pnl_dollars)::numeric, 2),
            ROUND(AVG(entry_price)::numeric, 0),
            ROUND(AVG(exit_price)::numeric, 0)
        FROM trades
        WHERE is_paper = %s AND status = 'EXECUTED'{date_clause}
        GROUP BY ticker
        ORDER BY SUM(pnl_dollars) DESC
        LIMIT 10
    """, params)
    rows = cur.fetchall()
    if rows:
        for tkr, cnt, tpnl, aentry, aexit in rows:
            print(f"  {tkr:50s}  x{cnt:3d}  "
                  f"P&L=${tpnl:>7}  avg={aentry}→{aexit}c")
    else:
        print("  (no executed trades)")
    print()

    # ── Hourly distribution ──
    print(d)
    print("  HOURLY DISTRIBUTION (UTC)")
    print(d)
    cur.execute(f"""
        SELECT
            EXTRACT(HOUR FROM created_at)::int AS hr,
            COUNT(*),
            ROUND(SUM(pnl_dollars)::numeric, 2)
        FROM trades
        WHERE is_paper = %s AND status = 'EXECUTED'{date_clause}
        GROUP BY hr
        ORDER BY hr
    """, params)
    rows = cur.fetchall()
    if rows:
        for hr, cnt, hpnl in rows:
            bar = "█" * max(1, int(cnt / 2))
            print(f"  {hr:02d}:00  {bar:20s}  trades={cnt:3d}  P&L=${hpnl:>7}")
    else:
        print("  (no executed trades)")
    print()

    # ── Daily P&L ──
    print(d)
    print("  DAILY P&L")
    print(d)
    cur.execute(f"""
        SELECT
            DATE(created_at),
            COUNT(*) FILTER (WHERE status = 'EXECUTED'),
            COUNT(*),
            ROUND(COALESCE(SUM(pnl_dollars) FILTER (WHERE status = 'EXECUTED'), 0)::numeric, 2),
            COUNT(*) FILTER (WHERE status = 'EXECUTED' AND pnl_dollars > 0),
            COUNT(*) FILTER (WHERE status = 'EXECUTED' AND pnl_dollars < 0)
        FROM trades
        WHERE is_paper = %s{date_clause}
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at)
    """, params)
    rows = cur.fetchall()
    cumulative = 0.0
    if rows:
        for day, executed_cnt, total_cnt, dpnl, dw, dl in rows:
            dw = dw or 0
            dl = dl or 0
            dpnl = float(dpnl or 0)
            cumulative += dpnl
            fr = (executed_cnt / total_cnt * 100) if total_cnt > 0 else 0
            print(f"  {day}  signals={total_cnt:3d}  filled={executed_cnt:3d} "
                  f"({fr:.0f}%)  P&L=${dpnl:>7}  "
                  f"cumul=${cumulative:>8.2f}  ({dw}W-{dl}L)")
    else:
        print("  (no trades)")

    print(f"\n{w}\n")
    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Argus Trade Report")
    parser.add_argument("--paper", action="store_true", help="Show paper trades")
    parser.add_argument("--days", type=int, default=None, help="Limit to last N days")
    args = parser.parse_args()
    report(paper=args.paper, days=args.days)


if __name__ == "__main__":
    main()

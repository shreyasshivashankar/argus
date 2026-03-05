#!/usr/bin/env bash
# Quick P&L report — queries Postgres via the argus Docker container.
# Usage:
#   ./report.sh              # live trades, all time
#   ./report.sh --paper      # paper trades only
#   ./report.sh --days 7     # last 7 days
exec docker-compose run --rm --entrypoint "" \
  -e DATABASE_URL=postgresql://argus:argus@postgres:5432/argus \
  argus python -m scripts.report "$@"

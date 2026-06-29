#!/usr/bin/env bash
# Daily corpus auto-update — pulls the newest Gooaye episodes, transcribes + indexes
# them locally (free), and trims the window to the most recent episodes. Designed to
# be run unattended by launchd/cron. Idempotent: days with no new episode are no-ops.
#
#   bash backend/scripts/daily_update.sh            # run an update now
#   bash backend/scripts/daily_update.sh --status   # just print the range
#
# All output is appended to data/update.log with timestamps.
set -euo pipefail

BACKEND="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$BACKEND/.." && pwd)"
LOG="$ROOT/data/update.log"
mkdir -p "$ROOT/data"

cd "$BACKEND"
# shellcheck disable=SC1091
source .venv/bin/activate

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') corpus update ====="
  python scripts/update_corpus.py "$@"
  echo "[done] $(date '+%Y-%m-%d %H:%M:%S')"
  echo
} >>"$LOG" 2>&1

# Print the tail so an interactive run shows what happened.
tail -n 6 "$LOG"

#!/usr/bin/env bash
# ============================================================
# PlaudConnector launcher — activates venv and runs the daemon
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtualenv
VENV="$PROJECT_ROOT/venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
else
    echo "[PlaudConnector] ⚠️  No venv found at $VENV — using system Python" >&2
fi

# Ensure log directory exists
LOG_DIR="$HOME/Library/Logs/ClinicalWhisper"
mkdir -p "$LOG_DIR"

exec python3 "$PROJECT_ROOT/plaud_ingester.py" "$@"

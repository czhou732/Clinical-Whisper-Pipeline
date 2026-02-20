#!/usr/bin/env bash
# ============================================================
# ClinicalWhisper — Full Pipeline Auto-Start
# Launches everything you need in one command:
#   1. Whisper file watcher (main.py)
#   2. Plaud Note connector (plaud_ingester.py)
#
# Usage:
#   bash scripts/start_pipeline.sh          # normal mode
#   bash scripts/start_pipeline.sh --stop   # stop all
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_ROOT/.pids"
LOG_DIR="$HOME/Library/Logs/ClinicalWhisper"

mkdir -p "$PID_DIR" "$LOG_DIR"

# Activate virtualenv
VENV="$PROJECT_ROOT/venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
else
    echo "⚠️  No venv at $VENV — using system Python" >&2
fi

# ------------------------------------------------------------------
stop_all() {
    echo "🛑 Stopping ClinicalWhisper pipeline..."
    for pidfile in "$PID_DIR"/*.pid; do
        [ -f "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        name=$(basename "$pidfile" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && echo "   Stopped $name (PID $pid)"
        fi
        rm -f "$pidfile"
    done
    echo "Done."
    exit 0
}

if [ "${1:-}" = "--stop" ]; then
    stop_all
fi

# ------------------------------------------------------------------
echo "🚀 ClinicalWhisper Full Pipeline"
echo "   Project: $PROJECT_ROOT"
echo ""

# Ensure output directory exists
mkdir -p "/Users/peterzhou/Downloads/Obsidian Vault/USC/ClinicalWhisper_Audio"
mkdir -p "$PROJECT_ROOT/Input"

# 1. Start Whisper file watcher (main.py)
echo "👁️  Starting Whisper watcher (main.py)..."
cd "$PROJECT_ROOT"
python3 main.py >> "$LOG_DIR/whisper_watcher.log" 2>&1 &
WATCHER_PID=$!
echo "$WATCHER_PID" > "$PID_DIR/whisper_watcher.pid"
echo "   PID: $WATCHER_PID  →  $LOG_DIR/whisper_watcher.log"

# 2. Start Plaud connector
echo "🎙️  Starting Plaud connector (plaud_ingester.py)..."
python3 plaud_ingester.py >> "$LOG_DIR/plaud_connector.log" 2>&1 &
PLAUD_PID=$!
echo "$PLAUD_PID" > "$PID_DIR/plaud_connector.pid"
echo "   PID: $PLAUD_PID  →  $LOG_DIR/plaud_connector.log"

echo ""
echo "✅ Pipeline running!"
echo ""
echo "   Plaud audio from USB or ~/Downloads/PlaudExports"
echo "     → auto-copied to Input/"
echo "     → Whisper transcribes locally"
echo "     → Output to: Obsidian Vault/USC/ClinicalWhisper_Audio/"
echo ""
echo "   Logs:  tail -f $LOG_DIR/whisper_watcher.log"
echo "          tail -f $LOG_DIR/plaud_connector.log"
echo "   Stop:  bash $SCRIPT_DIR/start_pipeline.sh --stop"
echo ""

# Wait for all background processes
trap stop_all INT TERM
wait

#!/usr/bin/env bash
# ============================================================
# Install PlaudConnector as a macOS launchd service
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.clinicalwhisper.plaud-connector"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
LOG_DIR="$HOME/Library/Logs/ClinicalWhisper"

echo "🔧 PlaudConnector Installer"
echo "   Project: $PROJECT_ROOT"
echo ""

# 1. Create required directories
echo "📁 Creating directories..."
mkdir -p "$PROJECT_ROOT/PlaudArchive"
mkdir -p "$HOME/Downloads/PlaudExports"
mkdir -p "$LOG_DIR"

# 2. Make launcher executable
chmod +x "$SCRIPT_DIR/plaud_connector.sh"

# 3. Install requests if needed
echo "📦 Checking Python dependencies..."
VENV="$PROJECT_ROOT/venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi
pip install requests 2>/dev/null || pip3 install requests 2>/dev/null || echo "   (requests may already be installed)"

# 4. Unload existing plist if present
if launchctl list | grep -q "$PLIST_NAME" 2>/dev/null; then
    echo "🔄 Unloading existing service..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# 5. Copy plist to LaunchAgents
echo "📋 Installing launchd plist..."
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"

# 6. Load the service
echo "🚀 Starting service..."
launchctl load "$PLIST_DST"

echo ""
echo "✅ PlaudConnector installed and running!"
echo ""
echo "   Watching:"
echo "     • /Volumes/PLAUD*  (USB)"
echo "     • ~/Downloads/PlaudExports  (app exports)"
echo ""
echo "   Logs:    $LOG_DIR/plaud_connector.log"
echo "   Archive: $PROJECT_ROOT/PlaudArchive/"
echo ""
echo "   Commands:"
echo "     Stop:    launchctl unload ~/Library/LaunchAgents/$PLIST_NAME.plist"
echo "     Start:   launchctl load ~/Library/LaunchAgents/$PLIST_NAME.plist"
echo "     Logs:    tail -f $LOG_DIR/plaud_connector.log"
echo "     Dry-run: $SCRIPT_DIR/plaud_connector.sh --dry-run"

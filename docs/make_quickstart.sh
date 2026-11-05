#!/usr/bin/env bash
# Render docs/quickstart.html to "docs/Quick Start.pdf" (shipped in the DMG).
set -euo pipefail
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="Quick Start.pdf" "file://$PWD/quickstart.html" 2>/dev/null
mdls -name kMDItemNumberOfPages "Quick Start.pdf" 2>/dev/null || true
